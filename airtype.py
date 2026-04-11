import cv2
import mediapipe as mp
import pyautogui
import time
import math
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def open_camera(camera_indexes=range(6), width=1280, height=720):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0

    for camera_index in camera_indexes:
        cap = cv2.VideoCapture(camera_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        for _ in range(5):
            success, frame = cap.read()
            if success and frame is not None:
                print(f"Using camera index {camera_index}")
                return cap

        cap.release()

    raise RuntimeError(
        "No working camera found. Check that your webcam is connected, not being "
        "used by another app, and that Windows camera privacy permissions allow Python."
    )


def draw_hand_landmarks(img, landmarks, connections):
    h, w, _ = img.shape

    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(img, start_point, end_point, (255, 255, 255), 2)

    for landmark in landmarks:
        cv2.circle(img, (int(landmark.x * w), int(landmark.y * h)), 4, (0, 255, 0), cv2.FILLED)

# Mediapipe setup
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing {MODEL_PATH.name}. Download the MediaPipe HandLandmarker model "
        "and place it beside airtype.py: "
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

mp_hands = mp.tasks.vision.HandLandmarker
mp_hands_connections = mp.tasks.vision.HandLandmarksConnections
hands = mp_hands.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)

# Camera setup
cap = open_camera()

# Keyboard layout
keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M","SPACE","BACK"]
]

# Button class
class Button:
    def __init__(self, pos, text, size=[80, 80]):
        self.pos = pos
        self.size = size
        self.text = text

# Create buttons
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50, 100*i+50], key))

# Draw keyboard
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), cv2.FILLED)
        cv2.putText(img, button.text, (x+10, y+55),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    return img

finalText = ""
last_click_time = 0

while True:
    success, img = cap.read()

    if not success:
        raise RuntimeError("Camera stopped returning frames.")

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    frame_timestamp_ms = int(time.time() * 1000)
    results = hands.detect_for_video(mp_image, frame_timestamp_ms)

    img = drawAll(img, buttonList)

    if results.hand_landmarks:
        for handLms in results.hand_landmarks:
            lmList = []

            for lm in handLms:
                h, w, c = img.shape
                lmList.append((int(lm.x*w), int(lm.y*h)))

            # Index finger (8) & thumb (4)
            x1, y1 = lmList[8]
            x2, y2 = lmList[4]

            cv2.circle(img, (x1, y1), 10, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0,255,0), cv2.FILLED)

            # Distance
            distance = math.hypot(x2-x1, y2-y1)

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                # Hover
                if x < x1 < x+w and y < y1 < y+h:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), cv2.FILLED)
                    cv2.putText(img, button.text, (x+10, y+55),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

                    # Click (pinch)
                    if distance < 45:
                        current_time = time.time()

                        if current_time - last_click_time > 0.6:
                            key = button.text

                            if key == "SPACE":
                                finalText += " "
                                pyautogui.write(" ")

                            elif key == "BACK":
                                finalText = finalText[:-1]
                                pyautogui.press("backspace")

                            else:
                                finalText += key
                                pyautogui.write(key.lower())

                            last_click_time = current_time

                            # Click effect
                            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), cv2.FILLED)

            draw_hand_landmarks(img, handLms, mp_hands_connections.HAND_CONNECTIONS)

    # Text box
    cv2.rectangle(img, (50, 400), (1200, 500), (0,0,0), cv2.FILLED)
    cv2.putText(img, finalText, (60, 470),
                cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

    cv2.imshow("AirType Keyboard", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
