<div align="center">

# 🤟 GestureSpeak

### *Real-time Sign Language Recognition with Multilingual Voice Output*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **GestureSpeak** is a real-time computer vision system that recognizes sign language hand gestures through a webcam and converts them into spoken words — with multilingual output support for English, Hindi, Tamil, and Telugu. Built to bridge the communication gap for the deaf and hard-of-hearing community.

<br/>

---

</div>

## 📌 Overview

No special hardware. No gloves. Just a camera and Python.

GestureSpeak watches your hand through a webcam, identifies the sign language gesture using an AI model, displays the recognized text on screen, and speaks it out loud in your chosen language — all in real time.

---

## 🧠 How It Works

```
📷  Webcam Feed
      ↓
🖐️  MediaPipe — detects hand & extracts 21 landmark points (x, y, z)
      ↓
🔢  Feature Extraction — 42 coordinate values per frame
      ↓
🤖  ML Classifier — Random Forest / SVM trained on gesture data
      ↓
📝  Text Output — recognized letter or word displayed on screen
      ↓
🔊  Speech Output — spoken aloud via gTTS in the selected language
```

---

## 🌐 Multilingual Support

| Language | Script   | Speech Output |
|----------|----------|:-------------:|
| English  | Latin    | ✅ |
| Hindi    | देवनागरी | ✅ |
| Tamil    | தமிழ்    | ✅ |
| Telugu   | తెలుగు   | ✅ |

---

## 🗂️ Project Structure

```
GestureSpeak/
│
├── 📁 data/                   # Collected gesture datasets (CSV)
│   └── gestures.csv
│
├── 📁 models/                 # Trained and saved ML models
│   └── gesture_model.pkl
│
├── 📁 src/                    # Core source code
│   ├── collect_data.py        # Webcam tool to collect gesture training data
│   ├── train_model.py         # Trains and evaluates the ML classifier
│   └── app.py                 # Main app — real-time prediction + speech output
│
├── 📁 docs/                   # Report, diagrams, presentation assets
│
├── 📁 notebooks/              # Experiments and learning scripts
│
├── requirements.txt           # All project dependencies
├── README.md                  # You are here
└── LICENSE
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or above
- A working webcam
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/GestureSpeak.git
cd GestureSpeak

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python src/app.py
```

> Full usage instructions will be updated as development progresses.

---

## 🛠️ Tech Stack

| Layer            | Technology       | Purpose                                        |
|------------------|------------------|------------------------------------------------|
| Language         | Python 3.10+     | Core development                               |
| Computer Vision  | OpenCV           | Camera feed and frame processing               |
| Hand Detection   | MediaPipe        | Real-time 21-point hand landmark extraction    |
| Machine Learning | scikit-learn     | Gesture classification (Random Forest / SVM)   |
| Data Handling    | NumPy, Pandas    | Feature extraction and dataset management      |
| Speech Output    | gTTS + pyttsx3   | Multilingual text-to-speech                    |
| Visualization    | Matplotlib       | Model evaluation and confusion matrix plots    |

---

## 🗓️ Development Roadmap

| Week | Milestone                                    | Status         |
|------|----------------------------------------------|----------------|
| 1    | Repository setup, environment configuration  | 🟡 In Progress |
| 2    | OpenCV + MediaPipe hand detection working    | ⬜ Pending      |
| 3    | Data collection tool built, dataset collected| ⬜ Pending      |
| 4    | ML model trained and evaluated               | ⬜ Pending      |
| 5    | First working end-to-end prototype           | ⬜ Pending      |
| 6    | Full integration with multilingual output    | ⬜ Pending      |
| 7    | Testing, bug fixes, accuracy improvements    | ⬜ Pending      |
| 8    | Documentation, report, and presentation      | ⬜ Pending      |

---

## 👥 Team

| Name           | Role                               | GitHub         |
|----------------|------------------------------------|----------------|
| [Your Name]    | Vision Pipeline & Documentation    | [@username]()  |
| [Member 2]     | ML Model & Data Engineering        | [@username]()  |
| [Member 3]     | Application & Multilingual Output  | [@username]()  |

---

## 🎓 Academic Context

> **Institution:** [Your College Name]  
> **Department:** Computer Science & Engineering (AI & Data Science)  
> **Year:** First Year — Group Project  
> **Problem Statement:** Accessibility Technology — Sign Language to Speech Conversion  

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built for accessibility — GestureSpeak, 2025*

</div>
