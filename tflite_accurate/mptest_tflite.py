# Cmd to run: python .\mptest_tflite.py --model .\mediapipe_hand-hand_landmark_detector.tflite --lr-mode thumb --roi lower --center-crop --no-thumb-only --thumb-dx-ratio 0.35 --thumb-extend-ratio 0.25
# The above cmd parameters can be made fixed on default run

import argparse
import signal
import time
from pathlib import Path

import cv2
import imagezmq
import numpy as np
import zmq


def _load_interpreter(model_path: str):
    # Prefer tflite-runtime on Raspberry Pi; fall back to TensorFlow if installed.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except Exception:  # pragma: no cover
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    return Interpreter(model_path=model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageZMQ receiver + TFLite hand landmarks + gesture logic (mptest.py style). "
        "This version does NOT use the mediapipe Python package."
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).with_name("mediapipe_hand-hand_landmark_detector.tflite")),
        help="Path to the hand landmark .tflite model.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show receiver window (disable for headless).",
    )
    parser.add_argument(
        "--roi",
        choices=["full", "lower", "upper", "left", "right", "center"],
        default="lower",
        help="Restrict inference to a region-of-interest before resizing (reduces face false positives).",
    )
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center-crop to square before resizing.",
    )
    parser.add_argument(
        "--input-norm",
        choices=["0_1", "minus1_1"],
        default="0_1",
        help="Input normalization: 0_1 = (x/255), minus1_1 = (x/127.5 - 1).",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Minimum model score to accept landmarks.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Require N consecutive frames above score threshold.",
    )
    parser.add_argument(
        "--active-duration",
        type=float,
        default=60.0,
        help="Seconds to stay active after WAKE UP gesture.",
    )
    # Scale-invariant gesture tuning (more robust across distance-to-camera)
    parser.add_argument(
        "--fist-ratio-thresh",
        type=float,
        default=1.15,
        help="STOP if (max fingertip distance / hand_scale) is below this.",
    )
    parser.add_argument(
        "--point-index-ratio",
        type=float,
        default=1.85,
        help="Pointing requires (index_tip distance / hand_scale) above this.",
    )
    parser.add_argument(
        "--point-others-max-ratio",
        type=float,
        default=1.45,
        help="Pointing requires (max other fingertip distance / hand_scale) below this.",
    )
    parser.add_argument(
        "--point-margin-ratio",
        type=float,
        default=0.25,
        help="Pointing requires index_dist_ratio to exceed other_max_ratio by this margin.",
    )
    parser.add_argument(
        "--lr-dx-ratio",
        type=float,
        default=0.85,
        help="LEFT/RIGHT requires abs(dx)/hand_scale above this.",
    )
    parser.add_argument(
        "--lr-dominance",
        type=float,
        default=1.15,
        help="LEFT/RIGHT requires abs(dx) > abs(dy) * this factor.",
    )
    parser.add_argument(
        "--lr-mode",
        choices=["index", "thumb"],
        default="thumb",
        help="How to trigger LEFT/RIGHT: index = index-finger pointing, thumb = thumb-direction pointing.",
    )
    parser.add_argument(
        "--thumb-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In thumb mode, require only the thumb to be extended (other fingers down).",
    )
    parser.add_argument(
        "--mirror-x",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert left/right (useful if your camera feed is mirrored).",
    )
    parser.add_argument(
        "--thumb-extend-ratio",
        type=float,
        default=0.25,
        help="In thumb mode, require thumb length (|tip-ip|/hand_scale) above this.",
    )
    parser.add_argument(
        "--thumb-dx-ratio",
        type=float,
        default=0.35,
        help="In thumb mode, require abs(thumb_dx)/hand_scale above this to emit LEFT/RIGHT.",
    )
    parser.add_argument(
        "--thumb-dominance",
        type=float,
        default=1.10,
        help="In thumb mode, require abs(thumb_dx) > abs(thumb_dy) * this factor.",
    )
    parser.add_argument(
        "--pip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay model-input preview (256x256) on the main frame.",
    )
    return parser.parse_args()


def _get_output(interpreter, output_details: list[dict], name: str) -> np.ndarray | None:
    for d in output_details:
        if str(d.get("name", "")).strip().lower() == name.strip().lower():
            return interpreter.get_tensor(d["index"])
    return None


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


def hand_scale(landmarks: np.ndarray) -> float:
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    palm_width = _dist2(index_mcp, pinky_mcp)
    palm_len = _dist2(wrist, middle_mcp)
    return max(palm_width, palm_len, 1e-6)


def fingertip_dists_to_wrist(landmarks: np.ndarray) -> dict[int, float]:
    wrist = landmarks[0]
    tips = [4, 8, 12, 16, 20]
    return {t: _dist2(landmarks[t], wrist) for t in tips}


def finger_states(landmarks: np.ndarray, *, mirror_thumb: bool) -> list[int]:
    tips = [4, 8, 12, 16, 20]
    states: list[int] = []

    thumb_tip_x = float(landmarks[tips[0], 0])
    thumb_ip_x = float(landmarks[tips[0] - 1, 0])
    thumb_open = thumb_tip_x < thumb_ip_x
    if mirror_thumb:
        thumb_open = not thumb_open
    states.append(1 if thumb_open else 0)

    for i in range(1, 5):
        tip_y = float(landmarks[tips[i], 1])
        pip_y = float(landmarks[tips[i] - 2, 1])
        states.append(1 if tip_y < pip_y else 0)

    return states


def thumb_vector(landmarks: np.ndarray) -> tuple[float, float]:
    """
    Vector from thumb IP joint (3) to thumb tip (4) in normalized image coords.
    Using IP->tip is usually more stable than wrist->tip for direction.
    """
    dx = float(landmarks[4, 0] - landmarks[3, 0])
    dy = float(landmarks[4, 1] - landmarks[3, 1])
    return dx, dy


def thumb_len_ratio(landmarks: np.ndarray, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return _dist2(landmarks[4], landmarks[3]) / scale


def _apply_roi(frame_rgb: np.ndarray, roi: str, center_crop: bool) -> np.ndarray:
    h0, w0 = frame_rgb.shape[:2]
    if roi == "lower":
        frame_rgb = frame_rgb[h0 // 2 :, :]
    elif roi == "upper":
        frame_rgb = frame_rgb[: h0 // 2, :]
    elif roi == "left":
        frame_rgb = frame_rgb[:, : w0 // 2]
    elif roi == "right":
        frame_rgb = frame_rgb[:, w0 // 2 :]
    elif roi == "center":
        y0 = h0 // 4
        y1 = (h0 * 3) // 4
        x0 = w0 // 4
        x1 = (w0 * 3) // 4
        frame_rgb = frame_rgb[y0:y1, x0:x1]

    if center_crop:
        h1, w1 = frame_rgb.shape[:2]
        side = min(h1, w1)
        y0 = (h1 - side) // 2
        x0 = (w1 - side) // 2
        frame_rgb = frame_rgb[y0 : y0 + side, x0 : x0 + side]

    return frame_rgb


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists() and not model_path.is_absolute():
        alt = Path(__file__).with_name(args.model)
        if alt.exists():
            model_path = alt
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    interpreter = _load_interpreter(str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in0 = input_details[0]
    in_shape = [int(x) for x in in0["shape"]]
    if len(in_shape) != 4:
        raise RuntimeError(f"Unexpected input shape: {in_shape} (expected [1,H,W,C])")
    _, in_h, in_w, in_c = in_shape
    if in_c != 3:
        raise RuntimeError(f"Unexpected input channels: {in_c} (expected 3)")
    if in0["dtype"] != np.float32:
        raise RuntimeError(
            f"Expected float32 model input; got {in0['dtype']}. "
            "If your model is quantized, add quantization support."
        )

    # ImageZMQ receiver
    imageHub = imagezmq.ImageHub()
    imageHub.zmq_socket.setsockopt(zmq.LINGER, 0)
    imageHub.zmq_socket.setsockopt(zmq.RCVTIMEO, 500)

    # Wake state control (mptest.py semantics)
    active = False
    active_time = 0.0
    ACTIVE_DURATION = float(args.active_duration)

    stable_count = 0
    stop = False

    def _handle_sigint(_sig: int, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    # Use monotonic timestamps for any future extensions
    perf0 = time.perf_counter()

    print("Receiver started (TFLite)... (Ctrl+C to stop)")
    try:
        while not stop:
            try:
                name, frame_bgr = imageHub.recv_image()
                imageHub.send_reply(b"OK")
            except zmq.error.Again:
                continue

            # Prepare model input
            frame_rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            roi_rgb = _apply_roi(frame_rgb_full, args.roi, args.center_crop)
            resized_rgb = cv2.resize(roi_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)

            if args.input_norm == "0_1":
                inp_f32 = resized_rgb.astype(np.float32) / 255.0
            else:
                inp_f32 = resized_rgb.astype(np.float32) / 127.5 - 1.0

            interpreter.set_tensor(in0["index"], inp_f32[None, ...])
            interpreter.invoke()

            score_t = _get_output(interpreter, output_details, "scores")
            lr_t = _get_output(interpreter, output_details, "lr")
            lms_t = _get_output(interpreter, output_details, "landmarks")

            score = float(score_t.reshape(-1)[0]) if score_t is not None else 0.0
            lr = float(lr_t.reshape(-1)[0]) if lr_t is not None else 0.0
            mirror_thumb = lr >= 0.5

            if score >= args.score_thresh and lms_t is not None:
                stable_count += 1
            else:
                stable_count = 0

            gesture_text = ""

            if stable_count >= max(1, args.stable_frames) and lms_t is not None:
                landmarks = lms_t.reshape(21, 3).astype(np.float32)
                fingers = finger_states(landmarks, mirror_thumb=mirror_thumb)
                tip_dists = fingertip_dists_to_wrist(landmarks)
                max_tip_dist = max(tip_dists.values()) if tip_dists else 0.0
                s = hand_scale(landmarks)
                max_tip_ratio = max_tip_dist / s if s else 0.0

                now = time.time()

                # open palm (wake)
                if fingers == [1, 1, 1, 1, 1]:
                    active = True
                    active_time = now
                    gesture_text = "WAKE UP"

                # check active mode
                if active and now - active_time < ACTIVE_DURATION:
                    if max_tip_dist > 0 and max_tip_ratio < args.fist_ratio_thresh:
                        gesture_text = "STOP"
                    elif fingers == [0, 1, 0, 0, 0]:
                        gesture_text = "FORWARD"
                    elif fingers == [0, 1, 1, 0, 0]:
                        gesture_text = "BACKWARD"
                    else:
                        if args.lr_mode == "thumb":
                            thumb_dx, thumb_dy = thumb_vector(landmarks)
                            dx_eff = -thumb_dx if args.mirror_x else thumb_dx
                            thumb_dx_ratio = abs(dx_eff) / s
                            thumb_is_horizontal = abs(dx_eff) > abs(thumb_dy) * args.thumb_dominance
                            thumb_is_long = thumb_len_ratio(landmarks, s) >= args.thumb_extend_ratio
                            others_down = sum(fingers[1:]) == 0
                            thumb_ok = (
                                thumb_is_horizontal
                                and thumb_dx_ratio >= args.thumb_dx_ratio
                                and thumb_is_long
                                and (others_down if args.thumb_only else True)
                            )
                            if thumb_ok:
                                gesture_text = "LEFT" if dx_eff < 0 else "RIGHT"
                        else:
                            index_tip = landmarks[8]
                            wrist = landmarks[0]

                            index_dist = tip_dists.get(8, 0.0)
                            other_max = max(
                                tip_dists.get(12, 0.0),
                                tip_dists.get(16, 0.0),
                                tip_dists.get(20, 0.0),
                                tip_dists.get(4, 0.0),
                            )
                            index_ratio = index_dist / s
                            other_ratio = other_max / s

                            looks_like_point = (
                                index_ratio >= args.point_index_ratio
                                and other_ratio <= args.point_others_max_ratio
                                and (index_ratio - other_ratio) >= args.point_margin_ratio
                            )

                            if looks_like_point:
                                dx = float(index_tip[0] - wrist[0])
                                dy = float(index_tip[1] - wrist[1])
                                dx_ratio = abs(dx) / s
                                if abs(dx) > abs(dy) * args.lr_dominance and dx_ratio >= args.lr_dx_ratio:
                                    gesture_text = "LEFT" if dx < 0 else "RIGHT"
                else:
                    active = False

            if not active:
                cv2.putText(
                    frame_bgr,
                    "INACTIVE - show palm",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            if gesture_text:
                print(gesture_text)
                cv2.putText(
                    frame_bgr,
                    gesture_text,
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            if args.display:
                if args.pip:
                    pip = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                    if lms_t is not None:
                        lms = lms_t.reshape(21, 3).astype(np.float32)
                        color = (0, 255, 0) if score >= args.score_thresh else (0, 165, 255)
                        for x_n, y_n, _z in lms:
                            x_px = int(np.clip(float(x_n), 0.0, 1.0) * (in_w - 1))
                            y_px = int(np.clip(float(y_n), 0.0, 1.0) * (in_h - 1))
                            cv2.circle(pip, (x_px, y_px), 2, color, -1)

                    cv2.putText(
                        pip,
                        f"score={score:.2f} stable={stable_count}/{max(1,args.stable_frames)} lr={lr:.2f} active={int(active)}",
                        (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        pip,
                        f"lr_mode={args.lr_mode}",
                        (5, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    if args.lr_mode == "thumb" and lms_t is not None:
                        lms = lms_t.reshape(21, 3).astype(np.float32)
                        s_dbg = hand_scale(lms)
                        dx_dbg, dy_dbg = thumb_vector(lms)
                        dx_eff_dbg = -dx_dbg if args.mirror_x else dx_dbg
                        cv2.putText(
                            pip,
                            f"thumb dx={dx_eff_dbg:+.3f} dy={dy_dbg:+.3f} lenr={thumb_len_ratio(lms, s_dbg):.2f}",
                            (5, 54),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    pip_h, pip_w = pip.shape[:2]
                    scale = 0.5
                    pip_small = cv2.resize(
                        pip,
                        (int(pip_w * scale), int(pip_h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                    ph, pw = pip_small.shape[:2]
                    # top-right overlay with a 5px margin
                    y1 = 5
                    x1 = max(0, frame_bgr.shape[1] - pw - 5)
                    frame_bgr[y1 : y1 + ph, x1 : x1 + pw] = pip_small

                cv2.imshow("Receiver Frame (TFLite)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            _ = perf0  # reserved for future timing/debug
    except KeyboardInterrupt:
        stop = True
    finally:
        imageHub.close()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
