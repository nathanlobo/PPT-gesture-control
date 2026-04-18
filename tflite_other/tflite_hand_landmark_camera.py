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


def _dequantize(arr: np.ndarray, details: dict) -> np.ndarray:
    q = details.get("quantization_parameters") or {}
    scales = q.get("scales")
    zero_points = q.get("zero_points")
    if scales is None or zero_points is None or len(scales) == 0:
        return arr.astype(np.float32)
    # Most TFLite models use per-tensor quantization for these outputs.
    scale = float(scales[0])
    zero_point = float(zero_points[0])
    return (arr.astype(np.float32) - zero_point) * scale


def _quantize_input(arr_f32: np.ndarray, input_details: dict) -> np.ndarray:
    dtype = input_details["dtype"]
    if dtype == np.float32:
        return arr_f32.astype(np.float32)

    q = input_details.get("quantization_parameters") or {}
    scales = q.get("scales")
    zero_points = q.get("zero_points")
    if scales is None or zero_points is None or len(scales) == 0:
        return arr_f32.astype(dtype)

    scale = float(scales[0])
    zero_point = float(zero_points[0])
    if scale <= 0:
        return arr_f32.astype(dtype)

    quant = np.round(arr_f32 / scale + zero_point)
    info = np.iinfo(dtype)
    quant = np.clip(quant, info.min, info.max)
    return quant.astype(dtype)


def _extract_landmarks(output_tensors: list[np.ndarray], output_details: list[dict]) -> np.ndarray | None:
    """
    Best-effort heuristic to locate the 21x3 landmarks tensor among model outputs.

    Returns float32 array shaped (21, 3) in model output space (often normalized x/y).
    """
    candidates: list[np.ndarray] = []
    for tensor, details in zip(output_tensors, output_details):
        out = _dequantize(tensor, details)
        shape = list(out.shape)
        size = int(np.prod(shape)) if len(shape) else 0
        if size == 63:
            candidates.append(out.reshape(21, 3))
        elif shape[-1:] == [63]:
            candidates.append(out.reshape(-1, 63)[0].reshape(21, 3))
        elif len(shape) == 3 and shape[-2:] == [21, 3]:
            candidates.append(out.reshape(21, 3))

    if candidates:
        return candidates[0].astype(np.float32)
    return None


def _get_output_by_name(
    output_tensors: list[np.ndarray], output_details: list[dict], name: str
) -> tuple[np.ndarray, dict] | None:
    for tensor, details in zip(output_tensors, output_details):
        if str(details.get("name", "")).strip().lower() == name.strip().lower():
            return tensor, details
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hand landmark TFLite model on a camera feed (landmark-only test; no palm detector)."
    )
    parser.add_argument(
        "--model",
        default="mediapipe_hand-hand_landmark_detector.tflite",
        help="Path to the .tflite landmark model file.",
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
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Center-crop to a square before resizing (helps if your hand is near the center).",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Only draw landmarks when model score is >= this threshold.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Require this many consecutive frames above score threshold before drawing (reduces false positives).",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="If set, periodically save annotated frames to this directory.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
        help="Save every N frames when --save-dir is set.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until Ctrl+C).",
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


def main() -> int:
    import cv2

    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    interpreter = _load_interpreter(str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        print(f"Warning: expected 1 input, got {len(input_details)}; using input[0].")

    in0 = input_details[0]
    in_shape = [int(x) for x in in0["shape"]]
    if len(in_shape) != 4:
        raise RuntimeError(f"Unexpected input shape: {in_shape} (expected [1,H,W,C])")

    _, in_h, in_w, in_c = in_shape
    if in_c != 3:
        raise RuntimeError(f"Unexpected input channels: {in_c} (expected 3)")

    print(f"Input: shape={in_shape} dtype={in0['dtype']}")
    print(
        "Note: this script does NOT run the palm/hand detector. "
        "Results may be poor unless the frame is already a tight hand crop."
    )

    cap = _open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    frame_i = 0
    t0 = time.perf_counter()
    last_fps = 0.0
    stable_count = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Prepare input: BGR->RGB, resize to model input.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if args.center_crop:
                h0, w0 = frame_rgb.shape[:2]
                side = min(h0, w0)
                y0 = (h0 - side) // 2
                x0 = (w0 - side) // 2
                cropped = frame_rgb[y0 : y0 + side, x0 : x0 + side]
                resized = cv2.resize(cropped, (in_w, in_h), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)

            # Normalize to 0..1 for float models; quantize if needed.
            if in0["dtype"] == np.float32:
                inp_f32 = resized.astype(np.float32) / 255.0
                inp = inp_f32[None, ...]
            else:
                inp_f32 = resized.astype(np.float32)
                inp = _quantize_input(inp_f32, in0)[None, ...]

            interpreter.set_tensor(in0["index"], inp)
            interpreter.invoke()

            outs = [interpreter.get_tensor(d["index"]) for d in output_details]
            landmarks = _extract_landmarks(outs, output_details)
            score = None
            lr = None

            score_out = _get_output_by_name(outs, output_details, "scores")
            if score_out is not None:
                score_tensor, score_details = score_out
                score = float(_dequantize(score_tensor, score_details).reshape(-1)[0])

            lr_out = _get_output_by_name(outs, output_details, "lr")
            if lr_out is not None:
                lr_tensor, lr_details = lr_out
                lr = float(_dequantize(lr_tensor, lr_details).reshape(-1)[0])

            annotated = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_AREA)
            above = score is None or score >= args.score_thresh
            if above:
                stable_count += 1
            else:
                stable_count = 0

            should_draw = landmarks is not None and above and stable_count >= max(1, args.stable_frames)
            if should_draw:
                # Heuristic: assume x,y are normalized 0..1 relative to input image.
                for x_n, y_n, _z in landmarks:
                    x_px = int(np.clip(x_n, 0.0, 1.0) * (in_w - 1))
                    y_px = int(np.clip(y_n, 0.0, 1.0) * (in_h - 1))
                    cv2.circle(annotated, (x_px, y_px), 3, (0, 255, 0), -1)

            overlay = f"fps={last_fps:.1f}"
            if score is not None:
                overlay += f" score={score:.3f}"
            if lr is not None:
                overlay += f" lr={lr:.3f}"
            overlay += f" stable={stable_count}/{max(1, args.stable_frames)}"
            if args.center_crop:
                overlay += " center_crop=on"
            cv2.putText(
                annotated,
                overlay,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            frame_i += 1
            if frame_i % 30 == 0:
                dt = time.perf_counter() - t0
                if dt > 0:
                    last_fps = frame_i / dt
                    print(f"frames={frame_i} fps={last_fps:.1f} landmarks={'yes' if landmarks is not None else 'no'}")

            if args.save_dir and args.save_every > 0 and (frame_i % args.save_every == 0):
                out_path = Path(args.save_dir) / f"frame_{frame_i:06d}.jpg"
                cv2.imwrite(str(out_path), annotated)

            if args.display:
                cv2.imshow("TFLite Hand Landmark (landmark-only test)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_i >= args.max_frames:
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
