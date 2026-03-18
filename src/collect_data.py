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
SAMPLES_PER_LABEL = 50
collecting = False

print("Controls:")
print("Press any letter (a-z) to start AUTO collecting 50 samples")
print("Press 'q' to quit")

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

    if collecting and landmarks_row and count < SAMPLES_PER_LABEL:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(landmarks_row + [current_label])
        count += 1
        time.sleep(0.05)

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
    elif 97 <= key <= 122:
        current_label = chr(key).upper()
        count = 0
        collecting = True
        print(f"Collecting 50 samples for: {current_label} — hold the sign steady!")

cap.release()
cv2.destroyAllWindows()