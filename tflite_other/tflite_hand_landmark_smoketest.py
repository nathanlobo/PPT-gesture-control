import argparse
from pathlib import Path

import numpy as np


def _load_interpreter(model_path: str):
    # Prefer tflite-runtime on Raspberry Pi; fall back to TensorFlow if installed.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except Exception:  # pragma: no cover
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    return Interpreter(model_path=model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for a hand landmark TFLite model (loads, prints IO, runs dummy inference)."
    )
    parser.add_argument(
        "--model",
        default="mediapipe_hand-hand_landmark_detector.tflite",
        help="Path to the .tflite model file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    interpreter = _load_interpreter(str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("== Inputs ==")
    for i, d in enumerate(input_details):
        print(f"[{i}] name={d.get('name')} shape={d.get('shape')} dtype={d.get('dtype')}")
        q = d.get("quantization_parameters") or {}
        if q.get("scales") is not None:
            print(f"    quant scales={q.get('scales')} zero_points={q.get('zero_points')}")

    print("== Outputs ==")
    for i, d in enumerate(output_details):
        print(f"[{i}] name={d.get('name')} shape={d.get('shape')} dtype={d.get('dtype')}")
        q = d.get("quantization_parameters") or {}
        if q.get("scales") is not None:
            print(f"    quant scales={q.get('scales')} zero_points={q.get('zero_points')}")

    # Dummy inference: zeros with correct input shape/dtype.
    d0 = input_details[0]
    in_shape = tuple(int(x) for x in d0["shape"])
    in_dtype = d0["dtype"]
    dummy = np.zeros(in_shape, dtype=in_dtype)
    interpreter.set_tensor(d0["index"], dummy)
    interpreter.invoke()

    print("== Output samples (first 10 values each) ==")
    for i, d in enumerate(output_details):
        out = interpreter.get_tensor(d["index"])
        flat = out.reshape(-1)
        sample = flat[:10]
        print(f"[{i}] {d.get('name')} sample={sample}")

    print("OK: model loaded and invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

