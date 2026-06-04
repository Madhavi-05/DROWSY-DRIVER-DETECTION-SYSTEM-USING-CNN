import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from pygame import mixer
import time
from datetime import datetime

# Load model
model = load_model("drowsy_model_final.h5")
labels = ['Closed', 'Open']

# Initialize sound alert
mixer.init()
mixer.music.load("alert.wav")

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
left_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow("Drowsiness Detection", cv2.WINDOW_NORMAL)
cv2.moveWindow("Drowsiness Detection", 300, 150)

# Video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('drowsiness_output.avi', fourcc, 20.0, (frame_width, frame_height))

# Trackers
score = 0
start_time = None

def predict_eye(eye_img):
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
    eye_img = eye_img.astype("float32") / 255.0
    eye_img = np.expand_dims(eye_img, axis=0)
    pred = model.predict(eye_img, verbose=0)
    return labels[np.argmax(pred)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    left_status = right_status = 'Open'

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (90, 90, 90), 2)
        break

    left_eye = left_cascade.detectMultiScale(gray)
    for (x, y, w, h) in left_eye:
        roi = gray[y:y+h, x:x+w]
        left_status = predict_eye(roi)
        break

    right_eye = right_cascade.detectMultiScale(gray)
    for (x, y, w, h) in right_eye:
        roi = gray[y:y+h, x:x+w]
        right_status = predict_eye(roi)
        break

    eye_status = f"L: {left_status} | R: {right_status}"

    # Score logic
    if left_status == 'Closed' and right_status == 'Closed':
        score += 1
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= 3:
            if not mixer.music.get_busy():
                mixer.music.play()
            cv2.putText(frame, "DROWSINESS ALERT!", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    else:
        start_time = None
        mixer.music.stop()
        score = max(score - 1, 0)

    # Timestamp (top-right)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (frame.shape[1] - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Eye Status (bottom-left in blue)
    cv2.putText(frame, f"Eye Status: {eye_status}",
                (20, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

    # Score (bottom-left below status, in green)
    cv2.putText(frame, f"Drowsiness Score: {score}",
                (20, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

    # Show and save video
    cv2.imshow("Drowsiness Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
