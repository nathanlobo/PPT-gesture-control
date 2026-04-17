import argparse
import time
from pathlib import Path

import numpy as np


def _load_interpreter(model_path: str):
    # Prefer tflite-runtime on Raspberry Pi; fall back to TensorFlow if installed.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except Exception:  # pragma: no cover
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    return Interpreter(model_path=model_path)


def _open_capture(source: str):
    import cv2

    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap
    return cv2.VideoCapture(source)


def _get_output_by_name(
    output_tensors: list[np.ndarray], output_details: list[dict], name: str
) -> np.ndarray | None:
    for tensor, details in zip(output_tensors, output_details):
        if str(details.get("name", "")).strip().lower() == name.strip().lower():
            return tensor
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TFLite hand landmark -> gesture logic (mptest.py style). "
        "Note: landmark-only; best results need a tight hand crop."
    )
    parser.add_argument(
        "--model",
        default="mediapipe_hand-hand_landmark_detector.tflite",
        help="Path to hand landmark detector .tflite.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or video file path.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show preview window (requires GUI).",
    )
    parser.add_argument(
        "--roi",
        choices=["full", "lower", "upper", "left", "right", "center"],
        default="full",
        help="Restrict inference to a region-of-interest before resizing.",
    )
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Center-crop to square before resizing.",
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
        "--input-norm",
        choices=["0_1", "minus1_1"],
        default="0_1",
        help="Input normalization: 0_1 = (x/255), minus1_1 = (x/127.5 - 1).",
    )
    parser.add_argument(
        "--draw-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw landmark dots even when score is below threshold (helps debugging).",
    )
    parser.add_argument(
        "--active-duration",
        type=float,
        default=60.0,
        help="Seconds to stay active after WAKE UP gesture.",
    )
    parser.add_argument(
        "--fist-dist-thresh",
        type=float,
        default=0.22,
        help="If max fingertip distance-to-wrist is below this, treat as fist (STOP).",
    )
    parser.add_argument(
        "--point-dist-thresh",
        type=float,
        default=0.28,
        help="Minimum wrist->index_tip distance to consider LEFT/RIGHT pointing.",
    )
    parser.add_argument(
        "--point-margin",
        type=float,
        default=0.04,
        help="Index tip must exceed other fingertip distances by this margin for LEFT/RIGHT.",
    )
    parser.add_argument(
        "--lr-dx-thresh",
        type=float,
        default=0.18,
        help="Minimum horizontal dx (index_tip - wrist) to emit LEFT/RIGHT.",
    )
    return parser.parse_args()


def finger_states(landmarks: np.ndarray, *, mirror_thumb: bool) -> list[int]:
    """
    landmarks: (21,3) float32 with x/y typically normalized 0..1.
    mirror_thumb: if True, reverse the thumb x-direction test.
    """
    tips = [4, 8, 12, 16, 20]
    states: list[int] = []

    # Thumb: compare tip with IP joint in x.
    thumb_tip_x = float(landmarks[tips[0], 0])
    thumb_ip_x = float(landmarks[tips[0] - 1, 0])
    thumb_open = thumb_tip_x < thumb_ip_x
    if mirror_thumb:
        thumb_open = not thumb_open
    states.append(1 if thumb_open else 0)

    # Other fingers: tip higher (smaller y) than PIP joint
    for i in range(1, 5):
        tip_y = float(landmarks[tips[i], 1])
        pip_y = float(landmarks[tips[i] - 2, 1])
        states.append(1 if tip_y < pip_y else 0)

    return states


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


def fingertip_dists_to_wrist(landmarks: np.ndarray) -> dict[int, float]:
    wrist = landmarks[0]
    tips = [4, 8, 12, 16, 20]
    return {t: _dist2(landmarks[t], wrist) for t in tips}


def main() -> int:
    import cv2

    args = parse_args()
    model_path = Path(args.model)
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

    cap = _open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    # Wake state control (same semantics as mptest.py)
    active = False
    active_time = 0.0
    ACTIVE_DURATION = float(args.active_duration)

    stable_count = 0
    last_printed = ""

    t0 = time.perf_counter()
    frames = 0
    fps = 0.0

    print("TFLite gesture receiver started... (Ctrl+C to stop)")
    try:
        while True:
            ok, frame_bgr_full = cap.read()
            if not ok:
                break

            frames += 1
            if frames % 30 == 0:
                dt = time.perf_counter() - t0
                if dt > 0:
                    fps = frames / dt

            frame_rgb_full = cv2.cvtColor(frame_bgr_full, cv2.COLOR_BGR2RGB)

            # ROI restriction to avoid face/scene confusion (still landmark-only).
            frame_rgb = frame_rgb_full
            h0, w0 = frame_rgb.shape[:2]
            if args.roi == "lower":
                frame_rgb = frame_rgb[h0 // 2 :, :]
            elif args.roi == "upper":
                frame_rgb = frame_rgb[: h0 // 2, :]
            elif args.roi == "left":
                frame_rgb = frame_rgb[:, : w0 // 2]
            elif args.roi == "right":
                frame_rgb = frame_rgb[:, w0 // 2 :]
            elif args.roi == "center":
                y0 = h0 // 4
                y1 = (h0 * 3) // 4
                x0 = w0 // 4
                x1 = (w0 * 3) // 4
                frame_rgb = frame_rgb[y0:y1, x0:x1]

            if args.center_crop:
                h1, w1 = frame_rgb.shape[:2]
                side = min(h1, w1)
                y0 = (h1 - side) // 2
                x0 = (w1 - side) // 2
                frame_rgb = frame_rgb[y0 : y0 + side, x0 : x0 + side]

            resized_rgb = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
            if args.input_norm == "0_1":
                inp_f32 = resized_rgb.astype(np.float32) / 255.0
            else:
                inp_f32 = resized_rgb.astype(np.float32) / 127.5 - 1.0
            inp = inp_f32[None, ...]

            interpreter.set_tensor(in0["index"], inp)
            interpreter.invoke()

            outs = [interpreter.get_tensor(d["index"]) for d in output_details]
            score_t = _get_output_by_name(outs, output_details, "scores")
            lr_t = _get_output_by_name(outs, output_details, "lr")
            lms_t = _get_output_by_name(outs, output_details, "landmarks")

            score = float(score_t.reshape(-1)[0]) if score_t is not None else 0.0
            lr = float(lr_t.reshape(-1)[0]) if lr_t is not None else 0.0

            # Heuristic: lr is commonly a left/right indicator. We don't know the exact convention,
            # but using it to mirror thumb logic improves robustness across hands/cameras.
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

                now = time.time()

                # open palm (wake)
                if fingers == [1, 1, 1, 1, 1]:
                    active = True
                    active_time = now
                    gesture_text = "WAKE UP"

                # check active mode
                if active and now - active_time < ACTIVE_DURATION:
                    # Robust fist check (prevents STOP confusing with LEFT/RIGHT due to noisy finger states)
                    if max_tip_dist > 0 and max_tip_dist < args.fist_dist_thresh:
                        gesture_text = "STOP"
                    elif fingers == [0, 1, 0, 0, 0]:
                        gesture_text = "FORWARD"
                    elif fingers == [0, 1, 1, 0, 0]:
                        gesture_text = "BACKWARD"
                    else:
                        # Horizontal pointing: require index tip to be clearly the most extended fingertip.
                        index_tip = landmarks[8]
                        wrist = landmarks[0]
                        index_dist = tip_dists.get(8, 0.0)
                        middle_dist = tip_dists.get(12, 0.0)
                        ring_dist = tip_dists.get(16, 0.0)
                        pinky_dist = tip_dists.get(20, 0.0)

                        looks_like_point = (
                            index_dist >= args.point_dist_thresh
                            and index_dist >= middle_dist + args.point_margin
                            and index_dist >= ring_dist + args.point_margin
                            and index_dist >= pinky_dist + args.point_margin
                        )

                        if looks_like_point:
                            dx = float(index_tip[0] - wrist[0])
                            dy = float(index_tip[1] - wrist[1])
                            if abs(dx) > abs(dy):
                                if dx < -args.lr_dx_thresh:
                                    gesture_text = "LEFT"
                                elif dx > args.lr_dx_thresh:
                                    gesture_text = "RIGHT"
                else:
                    active = False

            if gesture_text and gesture_text != last_printed:
                print(gesture_text)
                last_printed = gesture_text

            if args.display:
                # Show what the model actually sees (ROI + resize), so landmarks line up.
                preview = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                if lms_t is not None and (args.draw_all or score >= args.score_thresh):
                    landmarks = lms_t.reshape(21, 3).astype(np.float32)
                    color = (0, 255, 0) if score >= args.score_thresh else (0, 165, 255)
                    for x_n, y_n, _z in landmarks:
                        x_px = int(np.clip(float(x_n), 0.0, 1.0) * (in_w - 1))
                        y_px = int(np.clip(float(y_n), 0.0, 1.0) * (in_h - 1))
                        cv2.circle(preview, (x_px, y_px), 3, color, -1)

                cv2.putText(
                    preview,
                    f"fps={fps:.1f} score={score:.3f} stable={stable_count}/{max(1,args.stable_frames)} lr={lr:.3f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    f"active={int(active)} gesture={gesture_text}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    f"roi={args.roi} center_crop={int(args.center_crop)} norm={args.input_norm}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("TFLite Gesture Receiver", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
