import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import time

model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

CSV_FILE = "data/gestures.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

cap = cv2.VideoCapture(0)
cv2.namedWindow("GestureSpeak — Collect Data", cv2.WINDOW_NORMAL)

current_label = None
count = 0
collecting = False
countdown_active = False
countdown_start = 0
last_capture_time = 0
SAMPLES_PER_LABEL = 300
CAPTURE_DELAY = 0.1
COUNTDOWN_SEC = 3
word_map = {
            65: "HELLO",
            66: "YES",
            67: "NO",
            68: "HELP",
            69: "PLEASE",
            70: "THANKYOU",
            71: "SORRY",
            72: "GOOD",
            73: "BAD",
            74: "STOP"
        }

print("Controls:")
print("Letters: a-z | Numbers: 0-9 | Words: A-J | Quit: ESC")
print("Word Map:", " | ".join([f"{chr(k)}={v}" for k, v in word_map.items()]))
print("Press 'ESC' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    landmarks_row = None

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            row = []
            for lm in hand:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                row += [lm.x, lm.y, lm.z]
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            landmarks_row = row
    if countdown_active:
        elapsed = time.time() - countdown_start
        remaining = COUNTDOWN_SEC - int(elapsed)
        if remaining > 0:
            cv2.putText(frame, f"GET READY: {remaining}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        else:
            countdown_active = False
            collecting = True
            last_capture_time = time.time()
            print(f"Recording {SAMPLES_PER_LABEL} samples for '{current_label}'...")
    if collecting and landmarks_row and count < SAMPLES_PER_LABEL:
        now = time.time()
        if now - last_capture_time >= CAPTURE_DELAY:
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks_row + [current_label])
            count += 1
            last_capture_time = now

        if count >= SAMPLES_PER_LABEL:
            print(f"Done! {SAMPLES_PER_LABEL} samples saved for '{current_label}'")
            collecting = False

    status = f"Label: {current_label} | Saved: {count}/{SAMPLES_PER_LABEL}"
    color = (0, 255, 0) if collecting else (0, 255, 255)
    cv2.putText(frame, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if collecting:
        cv2.putText(frame, "RECORDING...", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("GestureSpeak — Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to quit
        break
    elif not collecting and not countdown_active:
        if 97 <= key <= 122: # alphabets
            current_label = chr(key).upper()
            count = 0
            countdown_active = True
            countdown_start = time.time()
            print(f"Collecting 300 samples for: {current_label}' — starting in {COUNTDOWN_SEC}s...")
        elif 48 <= key <= 57:  # 0-9 numbers
            current_label = chr(key)
            count = 0
            countdown_active = True
            countdown_start = time.time()
            print(f"Prepare sign for: '{current_label}' — starting in {COUNTDOWN_SEC}s...")
        elif 65 <= key <= 74:  # words
            current_label = word_map[key]
            count = 0
            countdown_active = True
            countdown_start = time.time()
            print(f"Prepare sign for: '{current_label}' — starting in {COUNTDOWN_SEC}s...")

cap.release()
cv2.destroyAllWindows()