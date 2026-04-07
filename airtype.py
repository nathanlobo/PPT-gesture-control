import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Camera setup
cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)

# Mediapipe setup (UPDATED SAFE INIT)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

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
        print("Camera not working")
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    img = drawAll(img, buttonList)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []

            for lm in handLms.landmark:
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

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Text box
    cv2.rectangle(img, (50, 400), (1200, 500), (0,0,0), cv2.FILLED)
    cv2.putText(img, finalText, (60, 470),
                cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

    cv2.imshow("AirType Keyboard", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()