import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import time


model = YOLO('yolov8n-oiv7.pt')  # will auto-download if not found

# Initialize pyttsx3 for Windows
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start video capture
cap = cv2.VideoCapture(0)

prev_detections = {}
frame_count = 0

def estimate_distance(bbox_width, frame_width):
    known_width = 20.0  # cm (adjust for your test object)
    focal_length = 500  # estimate for standard webcams
    return (known_width * focal_length) / bbox_width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = {}

    frame_h, frame_w = frame.shape[:2]

    for r in results.boxes:
        cls_id = int(r.cls.item())
        conf = float(r.conf.item())
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        w = x2 - x1

        distance = estimate_distance(w, frame_w)
        key = f"{label}"

        if key in detections:
            detections[key].append(distance)
        else:
            detections[key] = [distance]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {int(distance)}cm', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_count += 1
    if frame_count % 20 == 0:
        for label, distances in detections.items():
            count = len(distances)
            avg_distance = sum(distances) / count

            prev_avg = np.mean(prev_detections.get(label, [avg_distance]))
            direction = "closer" if avg_distance < prev_avg - 5 else (
                "farther" if avg_distance > prev_avg + 5 else "at the same distance"
            )

            message = f"{count} {label}, around {int(avg_distance)} centimeters, getting {direction}"
            print(message)
            speak(message)
            prev_detections[label] = distances

    cv2.imshow("Smart Glasses Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
