from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import pyttsx3

# Load YOLOv8 OIV7 model
model = YOLO("yolov8n-oiv7.pt")

# Camera setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

# Object hierarchy for filtering
HIERARCHY = {
    "person": ["man", "woman", "boy", "girl"],
    "clothing": ["shirt", "pants", "coat", "dress", "jacket"],
    "face": ["glasses", "eyes", "nose", "mouth"]
}

# Shared variables
latest_message = ""
spoken_message = ""
message_lock = threading.Lock()

# Estimate object distance
def estimate_distance(bbox_width, frame_width):
    known_width = 20.0  # cm
    focal_length = 500
    return (known_width * focal_length) / bbox_width

# Filter child detections if parent is found
def filter_labels(detections):
    labels_to_remove = set()
    all_labels = set(detections.keys())
    for parent, children in HIERARCHY.items():
        if parent in all_labels:
            for child in children:
                if child in all_labels:
                    labels_to_remove.add(child)
    for label in labels_to_remove:
        del detections[label]
    return detections

# Detection thread
def yolo_detection():
    global latest_message
    prev_detections = {}
    frame_count = 0

    while True:
        frame = picam2.capture_array()
        results = model(frame)
        detections = {}

        frame_h, frame_w = frame.shape[:2]

        for r in results[0].boxes:
            cls_id = int(r.cls.item())
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            w = x2 - x1

            distance = estimate_distance(w, frame_w)

            if label in detections:
                detections[label].append(distance)
            else:
                detections[label] = [distance]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {int(distance)}cm', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections = filter_labels(detections)

        frame_count += 1
        if frame_count % 20 == 0:
            new_message = ""
            for label, distances in detections.items():
                count = len(distances)
                avg_distance = sum(distances) / count
                prev_avg = np.mean(prev_detections.get(label, [avg_distance]))
                direction = " getting closer" if avg_distance < prev_avg - 5 else (
                    "getting farther" if avg_distance > prev_avg + 5 else " ")

                part = f"{count} {label}(s), around {int(avg_distance)} centimeters, {direction}"
                new_message += part + ". "
                prev_detections[label] = distances

            with message_lock:
                latest_message = new_message.strip()

        cv2.imshow("YOLOv8-OIV7 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

# Voice thread
def text_to_speech():
    global spoken_message
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)
    engine.setProperty('volume', 0.5)
    engine.setProperty('voice', 23)
    while True:
        with message_lock:
            msg = latest_message

        if msg and msg != spoken_message:
            print(f"[TTS] {msg}")
            engine.say(msg)
            engine.runAndWait()
            spoken_message = msg
        else:
            time.sleep(0.5)

# Start both threads
if __name__ == "__main__":
    t1 = threading.Thread(target=yolo_detection)
    t2 = threading.Thread(target=text_to_speech)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
