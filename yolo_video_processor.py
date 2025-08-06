#!/usr/bin/env python3.11
"""
YOLO Video Processor

This script processes a video using a YOLOv8 model, detects objects,
and outputs a new video with bounding boxes drawn around the detected objects.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import argparse
import traceback

def process_video(model_path, video_path, output_path=None, conf_threshold=0.1, 
                  iou_threshold=0.45, show_preview=False, skip_frames=0):
    """
    Process a video using a YOLOv8 model and output a new video with bounding boxes.
    
    Args:
        model_path (str): Path to the YOLOv8 model (.pt file)
        video_path (str): Path to the input video
        output_path (str, optional): Path for the output video. If None, will be derived from input path.
        conf_threshold (float, optional): Confidence threshold for detections. Default is 0.1.
        iou_threshold (float, optional): IOU threshold for NMS. Default is 0.45.
        show_preview (bool, optional): Whether to show a preview window. Default is False.
        skip_frames (int, optional): Process only every Nth frame. Default is 0 (process all frames).
    
    Returns:
        str: Path to the output video
    """
    try:
        # Load the YOLOv8 model
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        
        # Print model information
        print(f"Model information:")
        print(f"- Task: {model.task}")
        print(f"- Names: {model.names}")
        
        # Open the video
        print(f"Opening video from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Create output path if not provided
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_processed.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video writer")
        
        # Process the video
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        # Define colors for different classes
        np.random.seed(42)  # for reproducibility
        colors = {i: tuple(map(int, np.random.randint(0, 255, size=3))) for i in range(100)}
        
        print("Processing video frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                out.write(frame)  # Write original frame
                frame_count += 1
                continue
            
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # Create a copy of the frame for drawing
            annotated_frame = frame.copy()
            
            # Process detections
            if len(results) > 0:
                boxes = results[0].boxes
                frame_detections = len(boxes)
                detection_count += frame_detections
                
                if frame_count % 100 == 0 or frame_detections > 0:
                    print(f"Frame {frame_count}: {frame_detections} detections")
                
                # Draw bounding boxes and labels
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    # Get color for this class
                    color = colors.get(cls_id, (0, 255, 0))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    text = f"{cls_name} {conf:.2f}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                if frame_count % 100 == 0:
                    print(f"Frame {frame_count}: No detections")
            
            # Show preview if requested
            if show_preview:
                cv2.imshow("YOLO Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write the frame to the output video
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                frames_per_second = frame_count / elapsed_time
                remaining_frames = total_frames - frame_count
                estimated_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0
                
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({frame_count/total_frames*100:.1f}%) - "
                      f"ETA: {estimated_time/60:.1f} minutes")
        
        # Release resources
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\nVideo processing complete:")
        print(f"- Processed {frame_count} frames in {elapsed_time:.1f} seconds")
        print(f"- Average speed: {frame_count/elapsed_time:.1f} FPS")
        print(f"- Total detections: {detection_count}")
        print(f"- Average detections per frame: {detection_count/frame_count:.2f}")
        print(f"- Output saved to: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error in process_video: {e}")
        print(traceback.format_exc())
        raise

def main():
    """Main function to parse arguments and run the video processing."""
    parser = argparse.ArgumentParser(description="Process video with YOLOv8 model")
    parser.add_argument("model_path", help="Path to the YOLOv8 model (.pt file)")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("-o", "--output", help="Path for the output video")
    parser.add_argument("-c", "--conf", type=float, default=0.1, 
                        help="Confidence threshold for detections (default: 0.1)")
    parser.add_argument("-i", "--iou", type=float, default=0.45,
                        help="IOU threshold for NMS (default: 0.45)")
    parser.add_argument("-p", "--preview", action="store_true",
                        help="Show preview window while processing")
    parser.add_argument("-s", "--skip", type=int, default=0,
                        help="Process only every Nth frame (default: 0, process all frames)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        process_video(
            args.model_path, 
            args.video_path, 
            args.output, 
            args.conf, 
            args.iou, 
            args.preview,
            args.skip
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
