# 🚗 Driver Drowsiness Detection System Using CNN

A real-time computer vision application that detects driver drowsiness using a Convolutional Neural Network (CNN). The system continuously monitors the driver's eyes through a webcam and triggers an alert when signs of fatigue are detected, helping reduce the risk of accidents caused by drowsy driving.

---

## 📌 Project Overview

Driver fatigue is one of the leading causes of road accidents worldwide. This project uses Deep Learning and Computer Vision techniques to monitor a driver's eye state in real time.

The application captures video from a webcam, detects the driver's face and eyes, classifies eye states as **Open** or **Closed** using a trained CNN model, and generates an alert when prolonged eye closure indicates drowsiness.

---

## ✨ Features

* Real-time webcam monitoring
* Face detection using OpenCV Haar Cascades
* Eye detection and tracking
* CNN-based eye state classification
* Drowsiness score calculation
* Audio alert system
* Live status display
* Video recording support
* User-friendly interface

---

## 🛠️ Technologies Used

| Technology | Purpose                    |
| ---------- | -------------------------- |
| Python     | Programming Language       |
| OpenCV     | Computer Vision Operations |
| TensorFlow | Deep Learning Framework    |
| Keras      | CNN Model Development      |
| NumPy      | Numerical Computations     |
| Pygame     | Audio Alert System         |

---

## 📂 Project Structure

```text
DROWSY-DRIVER-DETECTION-SYSTEM-USING-CNN/
│
├── dataset/
│   ├── train/
│   │   ├── Open/
│   │   └── Closed/
│   │
│   └── test/
│       ├── Open/
│       └── Closed/
│
├── drowsydetect.py
├── drowsy_model_final.h5
├── alert.wav
├── drowsiness_output.avi
├── README.md
│
└── screenshots/
```

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Madhavi-05/DROWSY-DRIVER-DETECTION-SYSTEM-USING-CNN.git
```

### Step 2: Navigate to Project Directory

```bash
cd DROWSY-DRIVER-DETECTION-SYSTEM-USING-CNN
```

### Step 3: Install Dependencies

```bash
pip install opencv-python tensorflow keras numpy pygame
```

---

## ▶️ Running the Application

Execute the following command:

```bash
python drowsydetect.py
```

The webcam will start automatically and begin monitoring the driver's eye state.

Press **Q** to exit the application.

---

## 🧠 How It Works

### 1. Video Capture

The webcam continuously captures video frames.

### 2. Face Detection

The system detects the driver's face using Haar Cascade classifiers.

### 3. Eye Detection

The eye regions are extracted from the detected face.

### 4. CNN Prediction

The trained CNN model classifies each eye as:

* Open
* Closed

### 5. Drowsiness Detection

When both eyes remain closed for a specified duration:

* Drowsiness score increases
* Warning message is displayed
* Alarm sound is triggered

### 6. Alert Generation

The driver receives an immediate visual and audio warning.

---

## 📊 Model Information

The CNN model was trained on a dataset containing images of:

* Open Eyes
* Closed Eyes

The trained model learns visual patterns associated with eye states and performs real-time classification during monitoring.

---

## 📸 Screenshots

Add screenshots here after uploading them to your repository.

### Detection Screen

```text
screenshots/detection.png
```

### Drowsiness Alert

```text
screenshots/alert.png
```

---

## 🚀 Future Enhancements

* Yawning Detection
* Head Pose Estimation
* Night-Time Monitoring
* Mobile Application Integration
* Driver Analytics Dashboard
* Cloud-Based Monitoring System
* Emergency Notification Feature

---

## 🎯 Applications

* Smart Vehicle Safety Systems
* Driver Monitoring Systems
* Fleet Management Solutions
* Transportation Safety Applications
* Research in Computer Vision and Deep Learning

---

## 👩‍💻 Author

**Madhavi jayalaxmi Naripireddy**

B.Tech Graduate | Aspiring Data Analyst & Software Developer

GitHub: https://github.com/Madhavi-05


