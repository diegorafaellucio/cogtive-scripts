import cv2
import random
import numpy as np
import os
import sys
from collections import defaultdict

# Try to import YOLO, handle potential version issues
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics==8.0.20")
    from ultralytics import YOLO

# Load YOLO model (use the specified model)
model = YOLO("/home/diego/projects/COGTIVE/aivision-core/data/models/betterbeef/weight.pt")

# Open video file
video_path = "/home/diego/Videos/Cogtive/betterbeef2.mp4"
cap = cv2.VideoCapture(video_path)

# Validate video input
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get actual FPS

# Read first frame to determine frame size
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Reset video capture to start from the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define codec and create VideoWriter using MP4V (saves as MP4)
output_path = "pupee_output_tracked_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4V codec for MP4 compatibility
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Ensure VideoWriter initialized correctly
if not out.isOpened():
    print("Error: VideoWriter failed to initialize.")
    cap.release()
    exit()

# Generate unique colors for each class
random.seed(42)
class_names = model.names
colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(class_names))}

track_history = defaultdict(lambda: [])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the full frame for processing
    annotated_frame = frame.copy()

    results = model.track(annotated_frame, persist=True, tracker="bytetrack.yaml", conf=0.5, iou=0.3)

    if results[0].boxes:
        try:
            detections = sorted(
                zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls, results[0].boxes.id),
                key=lambda x: x[0][2],
                reverse=True
            )

            for box, conf, cls, track_id in detections:
                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                confidence = float(conf)
                class_id = int(cls)
                label = class_names[class_id]
                color = colors[class_id]
                track_id = int(track_id)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {confidence:.2f} (ID: {track_id})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        except Exception as e:
            print(f"Tracking error: {e}")

    cv2.imshow("YOLOv8 Object Tracking", annotated_frame)

    # Debug: Print frame number
    print(f"Writing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking video saved as {output_path}")
