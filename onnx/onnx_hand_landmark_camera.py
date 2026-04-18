import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an ONNX hand landmark model on a camera feed (landmark-only test; no detector stage)."
    )
    parser.add_argument(
        "--model",
        default=str(Path("job_jpr49qk9g_optimized_onnx") / "model.onnx"),
        help="Path to model.onnx (expects any external data files next to it).",
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
        help="Show a preview window (requires GUI).",
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
        help="Center-crop to a square before resizing.",
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
        help="Only draw landmarks when score >= threshold (if score output exists).",
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


def _extract_by_name(outputs: dict[str, np.ndarray]) -> tuple[float | None, float | None, np.ndarray | None]:
    score = None
    lr = None
    landmarks = None
    for k, v in outputs.items():
        kl = k.lower()
        if kl == "scores" or "score" in kl:
            score = float(np.asarray(v).reshape(-1)[0])
        elif kl == "lr" or "handed" in kl:
            lr = float(np.asarray(v).reshape(-1)[0])
        elif kl == "landmarks" or "landmark" in kl:
            arr = np.asarray(v).astype(np.float32)
            if arr.size == 63:
                landmarks = arr.reshape(21, 3)
            elif arr.ndim == 3 and arr.shape[-2:] == (21, 3):
                landmarks = arr.reshape(21, 3)
            elif arr.ndim == 2 and arr.shape[-1] == 63:
                landmarks = arr.reshape(-1, 63)[0].reshape(21, 3)
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
            "onnxruntime is not installed. Install it first:\n"
            "  python -m pip install onnxruntime\n"
            "If pip says 'No matching distribution', use Python 3.11/3.12 instead of 3.13."
        ) from exc

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_inputs = sess.get_inputs()
    if len(ort_inputs) < 1:
        raise RuntimeError("Model has no inputs?")

    inp0 = ort_inputs[0]
    name0 = inp0.name
    shape0 = [d if isinstance(d, int) else -1 for d in inp0.shape]
    # Expect something like [1,256,256,3] (NHWC) or [1,3,256,256] (NCHW).
    if len(shape0) != 4:
        raise RuntimeError(f"Unexpected input rank: shape={shape0}")

    cap = _open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    t0 = time.perf_counter()
    frames = 0
    fps = 0.0

    print(f"ONNX input: name={name0} shape={inp0.shape} type={inp0.type}")
    print(
        "Note: this is a landmark-only test. Without a detector/ROI stage, false positives are expected."
    )

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

            # Resolve H/W from the ONNX input shape.
            # If unknown, default to 256.
            if shape0[-1] == 3:
                # NHWC
                in_h = int(shape0[1]) if shape0[1] > 0 else 256
                in_w = int(shape0[2]) if shape0[2] > 0 else 256
                resized_rgb = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
                if args.input_norm == "0_1":
                    inp = resized_rgb.astype(np.float32) / 255.0
                else:
                    inp = resized_rgb.astype(np.float32) / 127.5 - 1.0
                inp = inp[None, ...]
            else:
                # NCHW
                in_h = int(shape0[2]) if shape0[2] > 0 else 256
                in_w = int(shape0[3]) if shape0[3] > 0 else 256
                resized_rgb = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
                if args.input_norm == "0_1":
                    inp = resized_rgb.astype(np.float32) / 255.0
                else:
                    inp = resized_rgb.astype(np.float32) / 127.5 - 1.0
                inp = np.transpose(inp, (2, 0, 1))[None, ...]  # HWC -> CHW

            out_vals = sess.run(None, {name0: inp})
            out_names = [o.name for o in sess.get_outputs()]
            outputs = {n: np.asarray(v) for n, v in zip(out_names, out_vals)}
            score, lr, landmarks = _extract_by_name(outputs)

            preview = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
            if landmarks is not None and (score is None or score >= args.score_thresh):
                for x_n, y_n, _z in landmarks:
                    x_px = int(np.clip(float(x_n), 0.0, 1.0) * (in_w - 1))
                    y_px = int(np.clip(float(y_n), 0.0, 1.0) * (in_h - 1))
                    cv2.circle(preview, (x_px, y_px), 3, (0, 255, 0), -1)

            overlay = f"fps={fps:.1f}"
            if score is not None:
                overlay += f" score={score:.3f}"
            if lr is not None:
                overlay += f" lr={lr:.3f}"
            overlay += f" roi={args.roi} norm={args.input_norm}"
            cv2.putText(
                preview,
                overlay,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if args.display:
                cv2.imshow("ONNX Hand Landmark (landmark-only test)", preview)
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

