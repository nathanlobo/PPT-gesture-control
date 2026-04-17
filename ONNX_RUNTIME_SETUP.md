# ONNX Runtime Testing

This repo includes ONNX tests for the model in `job_jpr49qk9g_optimized_onnx/model.onnx`.

## Files

- `onnx_hand_landmark_smoketest.py` prints model inputs/outputs and runs a dummy inference.
- `onnx_hand_landmark_camera.py` runs the model on a camera feed and draws landmark dots (landmark-only; no detector stage).

## Install

### Raspberry Pi (Bullseye 64-bit)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install onnxruntime numpy
sudo apt-get update && sudo apt-get install -y python3-opencv
```

### Windows

If `pip install onnxruntime` fails with “No matching distribution”, your Python is too new.
Install Python 3.11 or 3.12, then:

```powershell
py -3.12 -m pip install onnxruntime numpy opencv-python
```

## Run

Smoke test:

```powershell
python .\onnx_hand_landmark_smoketest.py --model .\job_jpr49qk9g_optimized_onnx\model.onnx
```

Camera test (desktop GUI):

```powershell
python .\onnx_hand_landmark_camera.py --model .\job_jpr49qk9g_optimized_onnx\model.onnx --source 0 --display --roi lower --center-crop
```

