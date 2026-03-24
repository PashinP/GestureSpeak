import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = "data/archive-5/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "data/kaggle_landmarks.csv"
MAX_PER_CLASS = 1000  # <-- Change this to get more/less samples per class

header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

all_rows = []
total = 0
skipped = 0

labels = sorted(os.listdir(DATASET_PATH))

for label in labels:
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    img_files = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
    print(f"Processing {label}... ({len(img_files)} images found)")
    count = 0

    for idx, img_file in enumerate(img_files):
        if count >= MAX_PER_CLASS:
            break

        img_path = os.path.join(label_path, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                row = []
                for lm in hand:
                    row += [lm.x, lm.y, lm.z]
                row.append(label)
                all_rows.append(row)
                count += 1
                total += 1
        else:
            skipped += 1

        # Progress every 100 images
        if (idx + 1) % 100 == 0:
            print(f"  ... processed {idx + 1}/{len(img_files)} images, {count} landmarks so far")

    print(f"  {label}: {count} landmarks extracted")

# Write all at once (much faster than opening file per row)
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"\nDone! Total: {total} | Skipped: {skipped}")
print(f"Saved to {OUTPUT_CSV}")