import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for an ONNX hand landmark model (prints IO, runs dummy inference)."
    )
    parser.add_argument(
        "--model",
        default=str(Path("job_jpr49qk9g_optimized_onnx") / "model.onnx"),
        help="Path to model.onnx (expects any external data files next to it).",
    )
    return parser.parse_args()


def _dtype_from_ort(t) -> np.dtype:
    # ort types look like 'tensor(float)' / 'tensor(float16)' / 'tensor(int64)' etc.
    s = str(t).lower()
    if "float16" in s:
        return np.float16
    if "float" in s:
        return np.float32
    if "uint8" in s:
        return np.uint8
    if "int8" in s:
        return np.int8
    if "int64" in s:
        return np.int64
    if "int32" in s:
        return np.int32
    return np.float32


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is not installed in this Python environment.\n"
            "On Windows with Python 3.13, pip often cannot install onnxruntime.\n"
            "Use Python 3.11/3.12 (or on Raspberry Pi, the system Python) and install with:\n"
            "  python -m pip install onnxruntime numpy\n"
        ) from exc

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    print("== Inputs ==")
    for i, inp in enumerate(inputs):
        print(f"[{i}] name={inp.name} shape={inp.shape} type={inp.type}")

    print("== Outputs ==")
    for i, out in enumerate(outputs):
        print(f"[{i}] name={out.name} shape={out.shape} type={out.type}")

    if not inputs:
        raise RuntimeError("Model has no inputs?")

    # Dummy inference: zeros for the first input, using its declared shape where possible.
    inp0 = inputs[0]
    shape = []
    for dim in inp0.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            # Unknown dimension; use a sensible default.
            shape.append(1)
    dtype = _dtype_from_ort(inp0.type)
    dummy = np.zeros(shape, dtype=dtype)

    out_vals = sess.run(None, {inp0.name: dummy})

    print("== Output samples (first 10 values each) ==")
    for i, (out_meta, val) in enumerate(zip(outputs, out_vals)):
        arr = np.asarray(val)
        flat = arr.reshape(-1)
        print(f"[{i}] {out_meta.name} sample={flat[:10]}")

    print("OK: model loaded and invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

