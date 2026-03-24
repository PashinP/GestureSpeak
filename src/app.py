import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
from gtts import gTTS
import os
import tempfile
import time

# Load model
model = joblib.load("models/gesture_model.pkl")

# Load MediaPipe
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Language options
LANGUAGES = {
    'E': ('en', 'English'),
    'H': ('hi', 'Hindi'),
    'T': ('ta', 'Tamil'),
    'G': ('te', 'Telugu')
}
current_lang = 'E'

cap = cv2.VideoCapture(0)
cv2.namedWindow("GestureSpeak", cv2.WINDOW_NORMAL)

current_letter = ""
word = ""
prediction_buffer = []
BUFFER_SIZE = 15
last_spoken_time = time.time()
last_letter = ""

print("GestureSpeak is running!")
print("SPACE = speak word | BACKSPACE = delete | L = change language | ESC = quit")

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
    cv2.rectangle(frame, (0, h-130), (w, h), (0, 0, 0), -1)

    lang_name = LANGUAGES[current_lang][1]
    cv2.putText(frame, f"Lang: {lang_name} (press L to change)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 1)
    cv2.putText(frame, f"Letter: {current_letter}", (10, h-85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, f"Word: {word}", (10, h-45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE=speak | BKSP=delete | ESC=quit", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("GestureSpeak", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE - speak
        if word:
            lang_code, lang_name = LANGUAGES[current_lang]
            print(f"Speaking in {lang_name}: {word}")
            tts = gTTS(text=word, lang=lang_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tmp_path = f.name
            tts.save(tmp_path)
            os.system(f"afplay {tmp_path}")
            os.remove(tmp_path)
            word = ""
            last_letter = ""
    elif key == 8:  # BACKSPACE
        word = word[:-1]
        last_letter = ""
    elif key == ord('l'):  # Change language
        langs = list(LANGUAGES.keys())
        idx = langs.index(current_lang)
        current_lang = langs[(idx + 1) % len(langs)]
        print(f"Language: {LANGUAGES[current_lang][1]}")

cap.release()
cv2.destroyAllWindows()