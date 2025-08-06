#!/usr/bin/env python3
"""
Video Processing Script with YOLOv8 for People Detection

This script processes a video file using YOLOv8 to detect people,
draws bounding boxes around them, and generates an output video.
It also displays the processing in real-time.
Enhanced to better detect small people in the video.
"""

import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process video with YOLOv8 for people detection')
    parser.add_argument('--video', type=str, default='/home/diego/Downloads/Canaa_norte_segmento_000.mp4',
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_people_detection.mp4',
                        help='Path to output video file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--small-conf', type=float, default=0.15,
                        help='Lower confidence threshold for small person detections')
    parser.add_argument('--model', type=str, default='yolov8l.pt',
                        help='YOLOv8 model to use (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show video while processing')
    parser.add_argument('--enhance-small', action='store_true', default=True,
                        help='Enable enhancements for small person detection')
    parser.add_argument('--small-size-threshold', type=float, default=0.02,
                        help='Threshold for considering a detection as small (as fraction of image area)')
    return parser.parse_args()

def enhance_image_for_small_objects(image):
    """
    Enhance image to make small objects more detectable.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
    
    return enhanced_image

def multi_scale_detection(model, frame, conf, classes=0):
    """
    Perform detection at multiple scales to better capture small objects.
    Returns combined results.
    """
    # Original scale detection
    results_original = model(frame, conf=conf, verbose=False, classes=classes)
    
    # Create a higher resolution version for small object detection
    h, w = frame.shape[:2]
    frame_upscaled = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    results_upscaled = model(frame_upscaled, conf=conf*0.8, verbose=False, classes=classes)  # Slightly lower threshold for upscaled
    
    # Instead of trying to combine the results objects (which is complex),
    # let's extract the detections from both and create a new annotated frame
    detections = []
    
    # Extract detections from original results
    if len(results_original) > 0 and len(results_original[0].boxes) > 0:
        for i in range(len(results_original[0].boxes)):
            box = results_original[0].boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf_val,
                'source': 'original'
            })
    
    # Extract detections from upscaled results and scale them back
    if len(results_upscaled) > 0 and len(results_upscaled[0].boxes) > 0:
        for i in range(len(results_upscaled[0].boxes)):
            box = results_upscaled[0].boxes[i]
            # Scale coordinates back to original size
            xyxy = box.xyxy[0].cpu().numpy() / 2
            x1, y1, x2, y2 = map(int, xyxy)
            conf_val = float(box.conf[0].cpu().numpy())
            
            # Check if this detection overlaps significantly with any existing detection
            is_duplicate = False
            for det in detections:
                iou = calculate_iou([x1, y1, x2, y2], det['bbox'])
                if iou > 0.5:  # If IoU > 0.5, consider it a duplicate
                    # Keep the one with higher confidence
                    if conf_val > det['conf']:
                        det['bbox'] = [x1, y1, x2, y2]
                        det['conf'] = conf_val
                        det['source'] = 'upscaled'
                    is_duplicate = True
                    break
            
            # If not a duplicate, add as a new detection
            if not is_duplicate:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf_val,
                    'source': 'upscaled'
                })
    
    # Return the original results for consistency, but we'll use the detections list later
    return results_original, detections

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    # Determine coordinates of intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # Calculate area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate area of union
    area_union = area_box1 + area_box2 - area_inter
    
    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0
    
    return iou

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load the YOLOv8 model (using a larger model for better detection)
    print(f"Loading YOLOv8 model: {args.model}...")
    model = YOLO(args.model)
    
    # Open the video file
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer
    output_path = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    # Calculate area thresholds for small objects
    image_area = width * height
    small_area_threshold = image_area * args.small_size_threshold
    
    print("Starting to process frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Enhance image if enabled
        if args.enhance_small:
            enhanced_frame = enhance_image_for_small_objects(frame)
        else:
            enhanced_frame = frame
        
        # Run YOLOv8 inference with multi-scale detection if enabled
        if args.enhance_small:
            _, detections = multi_scale_detection(model, enhanced_frame, args.conf, classes=0)
        else:
            results = model(enhanced_frame, conf=args.conf, verbose=False, classes=0)  # class 0 is person in COCO dataset
            
            # Convert results to our detection format for consistent processing
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    box = results[0].boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf_val = float(box.conf[0].cpu().numpy())
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf_val,
                        'source': 'original'
                    })
        
        # Create a copy of the frame to draw on
        annotated_frame = frame.copy()
        
        # Process detection results
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            
            # Calculate box area
            box_area = (x2 - x1) * (y2 - y1)
            
            # Determine if this is a small person
            is_small = box_area < small_area_threshold
            
            # Apply different confidence threshold for small people
            if is_small and conf < args.conf and conf >= args.small_conf:
                # This is a small person with lower confidence
                box_color = (0, 165, 255)  # Orange for small people
                label_prefix = "Small Person"
            else:
                box_color = (0, 255, 0)  # Green for normal detections
                label_prefix = "Person"
            
            # Skip if confidence is too low (based on size)
            if is_small and conf < args.small_conf:
                continue
            if not is_small and conf < args.conf:
                continue
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Add label with confidence
            label = f"{label_prefix}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Display progress
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - {fps_processing:.1f} FPS")
        
        # Show frame if requested
        if args.show:
            # Resize for display if the frame is too large
            display_frame = annotated_frame
            if width > 1280 or height > 720:
                display_frame = cv2.resize(annotated_frame, (1280, 720))
            
            cv2.imshow('People Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
