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
    prev_detections = {}  # {label: {'last_distance': float, 'last_seen_frame': int, 'last_spoken_frame': int}}
    frame_count = 0
    ema_alpha = 0.3  # Adjust EMA smoothing factor (you might need to experiment with this)
    speech_debounce_frames = 10 # Reduce debounce for more responsiveness

    while True:
        frame = picam2.capture_array()
        results = model(frame)
        current_detections = {}

        frame_h, frame_w = frame.shape[:2]

        for r in results[0].boxes:
            cls_id = int(r.cls.item())
            conf = float(r.conf.item())
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            w = x2 - x1

            distance = estimate_distance(w, frame_w)
            key = f"{label}"
            current_detections[key] = {'distance': distance, 'bbox': (x1, y1, x2, y2)}

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {int(distance)}cm', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for label, current_data in current_detections.items():
            distance = current_data['distance']
            prev = prev_detections.get(label)
            speak = False
            direction = ""

            if prev:
                prev_dist = prev['last_distance']
                prev_spoken_frame = prev.get('last_spoken_frame', -speech_debounce_frames - 1) # Initialize if not present

                # Apply EMA smoothing
                smoothed_dist = ema_alpha * distance + (1 - ema_alpha) * prev_dist

                if abs(smoothed_dist - prev_dist) > 5:
                    direction = "getting closer" if smoothed_dist < prev_dist else "getting farther"
                    if (frame_count - prev_spoken_frame) > speech_debounce_frames:
                        speak = True
                distance = smoothed_dist
            else:
                # New object
                speak = True

            if speak:
                message = f"A {label}, at {int(distance)} centimeters, {direction}"
                print(message)
                try:
                    message_queue.put(message, block=False) # Non-blocking put
                except queue.Full:
                    print("[Queue Full] Dropping message.")

                prev_detections[label] = {
                    'last_distance': distance,
                    'last_seen_frame': frame_count,
                    'last_spoken_frame': frame_count
                }
            elif prev:
                # Update distance and last seen frame even if not speaking
                prev_detections[label]['last_distance'] = distance
                prev_detections[label]['last_seen_frame'] = frame_count

        # Clean up stale objects not seen for 30 frames
        unseen_labels = [label for label, data in prev_detections.items()
                         if frame_count - data['last_seen_frame'] > 30]
        for label in unseen_labels:
            del prev_detections[label]

        frame_count += 1

        cv2.imshow("YOLOv8-OIV7 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

# TTS thread
def text_to_speech():
    engine = pyttsx3.init()
    engine.setProperty('rate', 120)              
    engine.setProperty('volume', 0.5)  
    engine.setProperty('voice', 23)         
    
    # Optional: set voice to female British (if available)
    voices = engine.getProperty('voices')
    for voice in voices:
        if "english" in voice.name.lower() and "gb" in voice.id.lower():
            engine.setProperty('voice', 23)
            break

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
