import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller
from time import time, sleep

# Initialize MediaPipe and Keyboard
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
keyboard = Controller()

# Keyboard layout
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["<", " "]  # Backspace, Space
]

finalText = ""

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

# Draw keys
def draw_all(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Create buttons
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# For long press tracking
hoveredKey = None
hoverStartTime = None

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from camera.")
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    img = draw_all(img, buttonList)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if lmList:
                x1, y1 = lmList[8]   # Index finger tip

                keyFound = False

                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    if x < x1 < x + w and y < y1 < y + h:
                        keyFound = True
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                        if hoveredKey != button.text:
                            hoveredKey = button.text
                            hoverStartTime = time()
                        else:
                            elapsed = time() - hoverStartTime
                            if elapsed >= 2:  # Long press
                                print(f"Pressed: {button.text}")
                                if button.text == "<":
                                    finalText = finalText[:-1]
                                    keyboard.press('\b')
                                    keyboard.release('\b')
                                elif button.text == " ":
                                    finalText += " "
                                    keyboard.press(" ")
                                    keyboard.release(" ")
                                else:
                                    finalText += button.text
                                    keyboard.press(button.text)
                                    keyboard.release(button.text)
                                hoverStartTime = None  # Reset after input
                                hoveredKey = None
                        break

                if not keyFound:
                    hoveredKey = None
                    hoverStartTime = None

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Display typed text
    cv2.rectangle(img, (50, 450), (1180, 550), (255, 0, 0), cv2.FILLED)
    (text_width, _), _ = cv2.getTextSize(finalText, cv2.FONT_HERSHEY_PLAIN, 5, 5)
    x_centered = 640 - text_width // 2
    cv2.putText(img, finalText, (x_centered, 530),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
