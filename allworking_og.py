from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import pyttsx3
from queue import Queue

# Load YOLOv8 OIV7 model
model = YOLO("yolov8n-oiv7.pt")

# Shared queue for TTS messages
message_queue = Queue()

# Initialize Picamera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

# Distance estimation helper
def estimate_distance(bbox_width, frame_width):
    known_width = 20.0  # cm
    focal_length = 500
    return (known_width * focal_length) / bbox_width

# YOLO detection thread
def yolo_detection():
    prev_detections = {}
    frame_count = 0
    while True:
        frame = picam2.capture_array()
        results = model(frame)
        detections = {}

        frame_h, frame_w = frame.shape[:2]

        for r in results[0].boxes:
            cls_id = int(r.cls.item())
            conf = float(r.conf.item())
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            w = x2 - x1

            distance = estimate_distance(w, frame_w)
            key = f"{label}"

            print(f"{label} detected at approximately {int(distance)} cm")

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
                direction = " getting closer" if avg_distance < prev_avg - 5 else (
                    "getting farther" if avg_distance > prev_avg + 5 else " "
                )

                message = f"A {label}, at {int(avg_distance)} centimeters, {direction}"
                print(message)
                message_queue.put(message)

                prev_detections[label] = distances

        cv2.imshow("YOLOv8-OIV7 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

# TTS thread
def text_to_speech():
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)            # Slow speech a bit
    engine.setProperty('voice', 'English (Great Britain)')  # Use the f3 female voice
    engine.setProperty('volume', 0.5)          # Set volume (0.0 to 1.0)

    while True:
        if not message_queue.empty():
            msg = message_queue.get()
            print(f"[TTS] Speaking: {msg}")
            engine.say(msg)
            engine.runAndWait()
        else:
            time.sleep(0.5)

# Start threads
if __name__ == "__main__":
    t1 = threading.Thread(target=yolo_detection)
    t2 = threading.Thread(target=text_to_speech)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
