from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 OIV7 model
model = YOLO("yolov8n-oiv7.pt")

# Setup picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Set resolution
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

prev_detections = {}
frame_count = 0

def estimate_distance(bbox_width, frame_width):
    known_width = 20.0  # cm (adjust for your test object)
    focal_length = 500  # estimated focal length for standard webcams
    return (known_width * focal_length) / bbox_width

while True:
    frame = picam2.capture_array()  # Capture the frame from the camera

    # Run YOLO detection
    results = model(frame)
    detections = {}

    frame_h, frame_w = frame.shape[:2]  # Get frame dimensions

    # Loop over detected objects
    for r in results[0].boxes:
        cls_id = int(r.cls.item())  # Class ID of the detected object
        conf = float(r.conf.item())  # Confidence score
        label = model.names[cls_id]  # Get the label of the detected object
        x1, y1, x2, y2 = map(int, r.xyxy[0])  # Get bounding box coordinates
        w = x2 - x1  # Bounding box width

        # Estimate the distance based on the bounding box width
        distance = estimate_distance(w, frame_w)
        key = f"{label}"

        # Print distance info to terminal
        print(f"{label} detected at approximately {int(distance)} cm")

        # Store distances for each detected object label
        if key in detections:
            detections[key].append(distance)
        else:
            detections[key] = [distance]

        # Draw bounding boxes and labels on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {int(distance)}cm', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_count += 1
    if frame_count % 20 == 0:  # Every 20 frames, process the detections
        for label, distances in detections.items():
            count = len(distances)  # Number of detections for this label
            avg_distance = sum(distances) / count  # Average distance

            prev_avg = np.mean(prev_detections.get(label, [avg_distance]))  # Previous average distance
            direction = "closer" if avg_distance < prev_avg - 5 else (
                "farther" if avg_distance > prev_avg + 5 else "at the same distance"
            )

            # Print detection information in the console
            message = f"{count} {label}, around {int(avg_distance)} centimeters, getting {direction}"
            print(message)

            # Update the previous detections with the new ones
            prev_detections[label] = distances

    # Display the annotated frame
    cv2.imshow("YOLOv8-OIV7 Detection", frame)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
