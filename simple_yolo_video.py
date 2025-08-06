#!/usr/bin/env python3.11
"""
Simple YOLO Video Processing Script

A simplified version of the video processing script to help diagnose issues.
"""

import cv2
import sys
from ultralytics import YOLO
import time

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_yolo_video.py <model_path> <video_path>")
        return
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully")
        print(f"Model task: {model.task}")
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open the video
    print(f"Opening video from {video_path}...")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video opened successfully: {width}x{height}, {fps} FPS, {total_frames} frames")
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    # Create output video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    print("Starting to process frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 10th frame for speed during testing
        if frame_count % 10 == 0:
            # Run detection with low confidence threshold
            results = model(frame, conf=0.05, verbose=False)
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Get detection info
            boxes = results[0].boxes
            if len(boxes) > 0:
                print(f"Frame {frame_count}: {len(boxes)} detections")
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    print(f"  Detection {i+1}: {cls_name} ({conf:.2f})")
            else:
                print(f"Frame {frame_count}: No detections")
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Display progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) - {fps_processing:.1f} FPS")
            
            # Show frame (press 'q' to quit)
            cv2.imshow('YOLO Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # For non-processed frames, just write the original frame
            out.write(frame)
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
