import cv2
import imagezmq
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
from pathlib import Path
import signal
import zmq
import argparse
import json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageZMQ receiver + hand gesture detection.")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the receiver window (disable for headless).",
    )
    parser.add_argument(
        "--publish",
        default="",
        help="If set, publish gesture commands via ZMQ PUB bind, e.g. tcp://*:6000",
    )
    parser.add_argument(
        "--topic",
        default="gesture",
        help="ZMQ topic for published gestures.",
    )
    return parser.parse_args()


args = parse_args()

# image receiver
imageHub = imagezmq.ImageHub()
imageHub.zmq_socket.setsockopt(zmq.LINGER, 0)
imageHub.zmq_socket.setsockopt(zmq.RCVTIMEO, 500)

pub_sock = None
if args.publish:
    ctx = zmq.Context.instance()
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.LINGER, 0)
    pub_sock.bind(args.publish)

# mediapipe setup (Tasks API)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing model file: {MODEL_PATH}. Expected 'hand_landmarker.task' next to mptest.py."
    )

base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# wake state control
active = False
active_time = 0
ACTIVE_DURATION = 60  # seconds robot stays active after wake gesture

# finger detection helper
def finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]
    states = []

    # thumb
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        states.append(1)
    else:
        states.append(0)

    # other fingers
    for i in range(1,5):
        if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
            states.append(1)
        else:
            states.append(0)

    return states  # [thumb, index, middle, ring, pinky]

print("Receiver started...")
t0 = time.perf_counter()
stop = False
last_published = ""


def _handle_sigint(_sig: int, _frame) -> None:
    global stop
    stop = True


signal.signal(signal.SIGINT, _handle_sigint)

try:
    while not stop:

        try:
            name, frame = imageHub.recv_image()
            imageHub.send_reply(b"OK")
        except zmq.error.Again:
            continue

        # convert for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - t0) * 1000)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture_text = ""

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )

                fingers = finger_states(hand_landmarks)

                thumb, index, middle, ring, pinky = fingers

                now = time.time()

                # open palm (wake)
                if fingers == [1, 1, 1, 1, 1]:
                    active = True
                    active_time = now
                    gesture_text = "WAKE UP"

                # check active mode
                if active and now - active_time < ACTIVE_DURATION:

                    # fist
                    if fingers == [0, 0, 0, 0, 0]:
                        gesture_text = "STOP "

                    # index up
                    elif fingers == [0, 1, 0, 0, 0]:
                        gesture_text = "FORWARD "

                    # two fingers
                    elif fingers == [0, 1, 1, 0, 0]:
                        gesture_text = "BACKWARD"

                    else:
                        # detect horizontal pointing using landmark positions
                        index_tip = hand_landmarks[8]
                        wrist = hand_landmarks[0]

                        dx = index_tip.x - wrist.x
                        dy = index_tip.y - wrist.y

                        if abs(dx) > abs(dy):

                            if dx < -0.15:
                                gesture_text = "LEFT "

                            elif dx > 0.15:
                                gesture_text = "RIGHT "

                else:
                    active = False

        if not active:
            cv2.putText(
                frame,
                "INACTIVE - show palm ",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if gesture_text:
            print(gesture_text)
            cv2.putText(
                frame,
                gesture_text,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Publish command changes for a headless Pi hardware controller.
        if pub_sock is not None:
            cmd = (gesture_text or ("INACTIVE" if not active else "")).strip().upper()
            if cmd and cmd != last_published:
                payload = {
                    "cmd": cmd,
                    "sender": str(name),
                    "ts_ms": int(time.time() * 1000),
                    "active": bool(active),
                }
                pub_sock.send_multipart(
                    [args.topic.encode("utf-8"), json.dumps(payload).encode("utf-8")]
                )
                last_published = cmd

        if args.display:
            cv2.imshow("Receiver Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
except KeyboardInterrupt:
    stop = True
finally:
    hand_landmarker.close()
    imageHub.close()
    if pub_sock is not None:
        pub_sock.close()
    if args.display:
        cv2.destroyAllWindows()
