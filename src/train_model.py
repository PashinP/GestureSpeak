import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load both datasets
df1 = pd.read_csv("data/gestures.csv")
df2 = pd.read_csv("data/kaggle_landmarks.csv")

# Merge
df = pd.concat([df1, df2], ignore_index=True)

# Remove classes with too few samples (e.g., 'nothing' has only 1)
class_counts = df['label'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['label'].isin(valid_classes)]

print(f"Total samples: {len(df)}")
print(df['label'].value_counts().sort_index())

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining model...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gesture_model.pkl")
print("\nModel saved!")