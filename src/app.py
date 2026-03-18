import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import pyttsx3
import time

model = joblib.load("models/gesture_model.pkl")

model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

cap = cv2.VideoCapture(0)
cv2.namedWindow("GestureSpeak", cv2.WINDOW_NORMAL)

current_letter = ""
word = ""
prediction_buffer = []
BUFFER_SIZE = 15
last_spoken_time = time.time()
last_letter = ""

print("GestureSpeak is running!")
print("Controls: SPACE = speak word | BACKSPACE = delete letter | ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            row = []
            for lm in hand:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                row += [lm.x, lm.y, lm.z]
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            prediction = model.predict([row])[0]
            prediction_buffer.append(prediction)

            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)

            if len(prediction_buffer) == BUFFER_SIZE:
                most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                if prediction_buffer.count(most_common) >= 12:
                    current_letter = most_common

                    now = time.time()
                    if current_letter != last_letter and now - last_spoken_time > 1.5:
                        word += current_letter
                        last_letter = current_letter
                        last_spoken_time = now

    # UI
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h-120), (w, h), (0, 0, 0), -1)

    cv2.putText(frame, f"Letter: {current_letter}", (10, h-80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, f"Word: {word}", (10, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE=speak | BKSP=delete | ESC=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("GestureSpeak", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE - speak word
        if word:
            print(f"Speaking: {word}")
            engine.say(word)
            engine.runAndWait()
            word = ""
            last_letter = ""
    elif key == 8:  # BACKSPACE - delete last letter
        word = word[:-1]
        last_letter = ""

cap.release()
cv2.destroyAllWindows()