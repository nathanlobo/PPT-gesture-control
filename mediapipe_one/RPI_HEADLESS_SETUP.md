# Raspberry Pi (Headless) Setup Notes

Your current `mptest.py` uses the **MediaPipe Tasks** API (`vision.HandLandmarker`). On Raspberry Pi 4 (Bullseye 64-bit),
installing MediaPipe via `pip` often fails because official prebuilt wheels are typically not available for Pi-class Linux
architectures. In that case, you must either **build MediaPipe from source** (hard) or **offload gesture detection to a PC**
and let the Pi do only hardware control (recommended).

This repo now supports the recommended split:

- PC/laptop: runs `mptest.py` (gesture detection) and **publishes** commands.
- Raspberry Pi: runs `pi_command_receiver.py` (headless) and triggers hardware operations.

## Architecture

1. Pi captures camera frames and sends them using ImageZMQ (`mpsender.py`).
2. PC receives frames + detects gestures (`mptest.py`).
3. PC publishes the latest gesture over ZeroMQ PUB.
4. Pi subscribes and runs your GPIO/serial code on each command.

This keeps hardware control on the Pi without requiring MediaPipe to be installed on the Pi.

## Run Steps (Typical)

### 1) On the PC (receiver + gesture detection)

Start the ImageZMQ receiver and publish gestures:

```powershell
python .\mptest.py --publish tcp://*:6000
```

If you don't want a window on the PC:

```powershell
python .\mptest.py --no-display --publish tcp://*:6000
```

### 2) On the Raspberry Pi (hardware controller)

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install pyzmq
```

Run the subscriber (replace `<PC_IP>`):

```bash
python3 pi_command_receiver.py --connect tcp://<PC_IP>:6000
```

Edit `pi_command_receiver.py` and put your GPIO/serial logic inside `handle_command()`.

### 3) On the Raspberry Pi (camera sender)

Install dependencies:

```bash
python3 -m pip install imagezmq pyzmq
```

OpenCV options:

- If you're headless, you typically don't need `cv2.imshow`, so using `python3-opencv` from apt is often easiest:
  - `sudo apt-get update && sudo apt-get install -y python3-opencv`

Run the sender (replace `<PC_IP>`):

```bash
python3 mpsender.py --connect tcp://<PC_IP>:5555 --no-display
```

## Ctrl+C reliability

Both sides are configured with ZeroMQ timeouts + `LINGER=0`, so Ctrl+C should exit quickly even if the other side
disconnects or stops replying.

