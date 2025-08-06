#!/usr/bin/env python3.11
"""
Basic YOLO Video Processor

A simple and reliable script to process video with YOLO and save the results.
Processes up to 1 minute of video and then stops.
"""

import os
import sys
import cv2
from ultralytics import YOLO
import time

def main():
    # Set model path directly
    model_path = "/home/diego/2TB/yolo/Trains/v8/object_detection/SHARP_KNIFE_1.0/GERAL_1.0/trains/nano/416/runs/detect/first_model_v2/weights/best.pt"
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python basic_yolo_processor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Set output path in current directory
    output_path = os.path.join(os.getcwd(), "output_video.mp4")
    print(f"Output will be saved to: {output_path}")
    
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        print(f"Model loaded successfully")
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Open video
    try:
        print(f"Opening video from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file")
            sys.exit(1)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Calculate frames for 1 minute of video
        frames_for_one_minute = int(fps * 60)
        print(f"Will process {frames_for_one_minute} frames (1 minute of video)")
    except Exception as e:
        print(f"Error opening video: {e}")
        sys.exit(1)
    
    # Create video writer
    try:
        print(f"Creating video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not create output video writer")
            sys.exit(1)
    except Exception as e:
        print(f"Error creating video writer: {e}")
        sys.exit(1)
    
    # Process video
    try:
        print("Starting video processing...")
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        # Process every 5th frame to speed up processing
        process_every_n_frames = 5
        
        # Set detection confidence threshold
        conf_threshold = 0.20
        print(f"Using detection confidence threshold: {conf_threshold}")
        
        while frame_count < frames_for_one_minute:  # Stop after 1 minute of video
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            # Process frame
            if frame_count % process_every_n_frames == 0:
                print(f"Processing frame {frame_count}/{frames_for_one_minute} ({frame_count/frames_for_one_minute*100:.1f}%)")
                
                # Run YOLO detection
                results = model(frame, conf=conf_threshold, verbose=False)
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
                # Count detections
                boxes = results[0].boxes
                num_detections = len(boxes)
                detection_count += num_detections
                
                # Print detection info
                if num_detections > 0:
                    print(f"  Found {num_detections} detections in frame {frame_count}")
                    for i, box in enumerate(boxes[:3]):  # Show only first 3 detections
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_id]
                        print(f"    - {cls_name} ({conf:.2f})")
                
                # Write processed frame
                out.write(annotated_frame)
            else:
                # Write original frame for frames we skip
                out.write(frame)
            
            # Update counter
            frame_count += 1
            
            # Show progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                remaining = (frames_for_one_minute - frame_count) / fps_processing
                print(f"Progress: {frame_count}/{frames_for_one_minute} frames, "
                      f"{fps_processing:.1f} FPS, "
                      f"ETA: {remaining/60:.1f} minutes")
        
        # Print summary
        elapsed = time.time() - start_time
        print("\nProcessing complete:")
        print(f"- Processed {frame_count} frames in {elapsed:.1f} seconds")
        print(f"- Processing speed: {frame_count/elapsed:.1f} FPS")
        print(f"- Total detections: {detection_count}")
        print(f"- Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("Cleaning up resources...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
