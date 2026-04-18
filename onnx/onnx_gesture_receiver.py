import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ONNX hand landmark -> gesture logic (mptest.py style). "
        "Note: landmark-only; best results need a tight hand crop (ideally detector+ROI stage)."
    )
    parser.add_argument(
        "--model",
        default=str(Path("job_jpr49qk9g_optimized_onnx") / "model.onnx"),
        help="Path to model.onnx (expects external data next to it, e.g. model.data).",
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
        "--input-norm",
        choices=["0_1", "minus1_1"],
        default="0_1",
        help="Input normalization: 0_1 = (x/255), minus1_1 = (x/127.5 - 1).",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Minimum model score to accept landmarks (if score output exists).",
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
    parser.add_argument(
        "--draw-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw landmark dots even when score is below threshold (helps debugging).",
    )
    return parser.parse_args()


def _open_capture(source: str):
    import cv2

    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap
    return cv2.VideoCapture(source)


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


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


def _extract_outputs(outputs: dict[str, np.ndarray]) -> tuple[float | None, float | None, np.ndarray | None]:
    score = None
    lr = None
    landmarks = None

    for k, v in outputs.items():
        kl = k.lower()
        arr = np.asarray(v)
        if kl == "scores" or "score" in kl:
            score = float(arr.reshape(-1)[0])
        elif kl == "lr" or "handed" in kl:
            lr = float(arr.reshape(-1)[0])
        elif kl == "landmarks" or "landmark" in kl:
            if arr.size == 63:
                landmarks = arr.reshape(21, 3).astype(np.float32)
            elif arr.ndim == 3 and arr.shape[-2:] == (21, 3):
                landmarks = arr.reshape(21, 3).astype(np.float32)
            elif arr.ndim == 2 and arr.shape[-1] == 63:
                landmarks = arr.reshape(-1, 63)[0].reshape(21, 3).astype(np.float32)

    return score, lr, landmarks


def main() -> int:
    import cv2

    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is not installed.\n"
            "Install it with Python 3.11/3.12 (Windows) or system Python (Pi):\n"
            "  python -m pip install onnxruntime numpy\n"
        ) from exc

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_inputs = sess.get_inputs()
    if len(ort_inputs) < 1:
        raise RuntimeError("Model has no inputs?")

    inp0 = ort_inputs[0]
    input_name = inp0.name
    shape0 = [d if isinstance(d, int) else -1 for d in inp0.shape]
    if len(shape0) != 4:
        raise RuntimeError(f"Unexpected input rank: shape={shape0}")

    # Determine layout.
    nhwc = shape0[-1] == 3
    if nhwc:
        in_h = int(shape0[1]) if shape0[1] > 0 else 256
        in_w = int(shape0[2]) if shape0[2] > 0 else 256
    else:
        in_h = int(shape0[2]) if shape0[2] > 0 else 256
        in_w = int(shape0[3]) if shape0[3] > 0 else 256

    cap = _open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    active = False
    active_time = 0.0
    ACTIVE_DURATION = float(args.active_duration)

    stable_count = 0
    last_printed = ""

    frames = 0
    t0 = time.perf_counter()
    fps = 0.0

    print("ONNX gesture receiver started... (Ctrl+C to stop)")
    print(f"Input: name={input_name} shape={inp0.shape} layout={'NHWC' if nhwc else 'NCHW'}")

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
            frame_rgb = frame_rgb_full

            # ROI restriction.
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
                inp = resized_rgb.astype(np.float32) / 255.0
            else:
                inp = resized_rgb.astype(np.float32) / 127.5 - 1.0

            if nhwc:
                model_inp = inp[None, ...]
            else:
                model_inp = np.transpose(inp, (2, 0, 1))[None, ...]

            out_vals = sess.run(None, {input_name: model_inp})
            out_names = [o.name for o in sess.get_outputs()]
            outputs = {n: np.asarray(v) for n, v in zip(out_names, out_vals)}

            score, lr, landmarks = _extract_outputs(outputs)
            if score is None:
                score = 1.0
            if lr is None:
                lr = 0.0

            mirror_thumb = lr >= 0.5

            if score >= args.score_thresh and landmarks is not None:
                stable_count += 1
            else:
                stable_count = 0

            gesture_text = ""

            if stable_count >= max(1, args.stable_frames) and landmarks is not None:
                fingers = finger_states(landmarks, mirror_thumb=mirror_thumb)
                tip_dists = fingertip_dists_to_wrist(landmarks)
                max_tip_dist = max(tip_dists.values()) if tip_dists else 0.0

                now = time.time()

                if fingers == [1, 1, 1, 1, 1]:
                    active = True
                    active_time = now
                    gesture_text = "WAKE UP"

                if active and now - active_time < ACTIVE_DURATION:
                    if max_tip_dist > 0 and max_tip_dist < args.fist_dist_thresh:
                        gesture_text = "STOP"
                    elif fingers == [0, 1, 0, 0, 0]:
                        gesture_text = "FORWARD"
                    elif fingers == [0, 1, 1, 0, 0]:
                        gesture_text = "BACKWARD"
                    else:
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
                preview = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
                if landmarks is not None and (args.draw_all or score >= args.score_thresh):
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
                cv2.imshow("ONNX Gesture Receiver", preview)
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

