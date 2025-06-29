import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import time
import pyautogui
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Webcam setup
cap = cv2.VideoCapture(0)

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# State tracking
pTime = 0
last_action_time = 0

def get_finger_states(lmList):
    fingers = []
    tipIds = [4, 8, 12, 16, 20]
    fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
    for id in range(1, 5):
        fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    gesture = "None"

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
            lmList = []
            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                fingerStates = get_finger_states(lmList)
                fingerCount = sum(fingerStates)
                current_time = time.time()

                # Tab switching
                if current_time - last_action_time > 1:
                    if fingerCount == 5 and label == "Right":
                        pyautogui.hotkey("ctrl", "tab")
                        gesture = "Next Tab"
                        last_action_time = current_time
                    elif fingerCount == 2 and label == "Left":
                        pyautogui.hotkey("ctrl", "shift", "tab")
                        gesture = "Prev Tab"
                        last_action_time = current_time

                # Volume control (Right hand)
                if label == "Right":
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    length = hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [30, 200], [volMin, volMax])
                    volBar = np.interp(length, [30, 200], [400, 150])
                    volPer = np.interp(length, [30, 200], [0, 100])
                    volume.SetMasterVolumeLevel(vol, None)
                    gesture = "Volume"

                    # Draw volume bar
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                # Brightness control (Left hand)
                elif label == "Left":
                    x3, y3 = lmList[4][1], lmList[4][2]
                    x4, y4 = lmList[12][1], lmList[12][2]
                    length_b = hypot(x4 - x3, y4 - y3)
                    brightness = np.interp(length_b, [30, 200], [0, 100])
                    sbc.set_brightness(int(brightness))
                    gesture = "Brightness"
                    cv2.putText(img, f'Brightness: {int(brightness)} %', (10, 470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display FPS & gesture
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 0), 2)
    cv2.putText(img, f'Gesture: {gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
