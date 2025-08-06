import cv2
import numpy as np
import random
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Try to import YOLO, handle potential version issues
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics==8.0.20")
    from ultralytics import YOLO

# Define the Region of Interest (ROI)
ROI_COORDS = [(570, 190), (1030, 415)]  # [(x1, y1), (x2, y2)]

class KalmanTracker:
    """
    Kalman Filter based tracker for object tracking
    """
    def __init__(self, bbox, class_id, track_id):
        self.track_id = track_id
        self.class_id = class_id
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        
        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Set process noise and measurement noise
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Initialize state with bounding box [x, y, w, h, vx, vy, vw, vh]
        x, y, w, h = self._bbox_to_xyxy(bbox)
        self.kf.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        
        # History of positions for visualization
        self.history = []
        
    def _bbox_to_xyxy(self, bbox):
        """Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        return x1, y1, x2 - x1, y2 - y1
        
    def _xyxy_to_bbox(self, x, y, w, h):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return [x, y, x + w, y + h]
    
    def predict(self):
        """Predict next state using Kalman filter"""
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        # Kalman prediction
        prediction = self.kf.predict()
        
        # Get predicted bbox
        x, y, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
        predicted_bbox = self._xyxy_to_bbox(float(x), float(y), float(w), float(h))
        
        return predicted_bbox
    
    def update(self, bbox):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.hit_streak += 1
        
        # Convert bbox to measurement format
        x, y, w, h = self._bbox_to_xyxy(bbox)
        measurement = np.array([[x], [y], [w], [h]], np.float32)
        
        # Update Kalman filter with new measurement
        self.kf.correct(measurement)
        
        # Update position history for visualization
        state = self.kf.statePost
        x, y = int(state[0]), int(state[1])
        self.history.append((x, y))
        # No longer limiting history length to maintain full tracking path
    
    def get_state(self):
        """Get current state of the tracker"""
        state = self.kf.statePost
        x, y, w, h = float(state[0][0]), float(state[1][0]), float(state[2][0]), float(state[3][0])
        return self._xyxy_to_bbox(x, y, w, h)


class MultiObjectTracker:
    """
    Multi-object tracker using Kalman Filter
    """
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # Maximum frames to keep a track alive without matching
        self.min_hits = min_hits  # Minimum hits to start tracking
        self.iou_threshold = iou_threshold  # IOU threshold for matching
        
        self.trackers = []  # List of active trackers
        self.frame_count = 0
        
        # Counters for each class
        self.roi_trackers = set()  # Set to track objects currently in ROI
        self.roi_tracker_counts = {}  # Dictionary to track how many frames each object has been in ROI
        self.cumulative_roi_trackers = set()  # Set to track objects that have been in ROI for minimum frames
        self.class_counters = defaultdict(int)
        
        # Minimum frames an object must stay in ROI to be counted
        self.min_roi_frames = 10
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IOU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IOU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
        
    def _calculate_roi_intersection(self, bbox):
        """Calculate intersection between a bounding box and the ROI
        Returns intersection score (0-1) and whether the bbox intersects with ROI"""
        # ROI in [x1, y1, x2, y2] format
        roi = [ROI_COORDS[0][0], ROI_COORDS[0][1], ROI_COORDS[1][0], ROI_COORDS[1][1]]
        
        # Get coordinates of intersection
        x_left = max(bbox[0], roi[0])
        y_top = max(bbox[1], roi[1])
        x_right = min(bbox[2], roi[2])
        y_bottom = min(bbox[3], roi[3])
        
        # Calculate area of intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0, False
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Calculate intersection score (percentage of bbox in ROI)
        intersection_score = intersection_area / bbox_area if bbox_area > 0 else 0
        
        # Check if there is any intersection
        is_intersecting = intersection_area > 0
        
        return intersection_score, is_intersecting
    
    def update(self, detections):
        """
        Update trackers with new detections
        
        Args:
            detections: List of tuples (bbox, confidence, class_id)
        
        Returns:
            List of tuples (bbox, track_id, class_id)
        """
        self.frame_count += 1
        
        # Get predictions from existing trackers
        predicted_bboxes = []
        for tracker in self.trackers:
            predicted_bbox = tracker.predict()
            predicted_bboxes.append(predicted_bbox)
        
        # Match detections with existing trackers
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))
        
        if len(predicted_bboxes) > 0 and len(detections) > 0:
            # Calculate IOU matrix
            iou_matrix = np.zeros((len(detections), len(predicted_bboxes)), dtype=np.float32)
            for d, detection in enumerate(detections):
                for t, prediction in enumerate(predicted_bboxes):
                    iou_matrix[d, t] = self._calculate_iou(detection[0], prediction)
            
            # Use Hungarian algorithm for optimal assignment
            # For simplicity, we'll use a greedy approach here
            while True:
                # Find max IOU
                if np.max(iou_matrix) < self.iou_threshold:
                    break
                
                # Get indices of max IOU
                d, t = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                # Add to matched indices
                matched_indices.append((d, t))
                
                # Remove from unmatched
                if d in unmatched_detections:
                    unmatched_detections.remove(d)
                if t in unmatched_trackers:
                    unmatched_trackers.remove(t)
                
                # Set row and column to -1 to avoid reusing
                iou_matrix[d, :] = -1
                iou_matrix[:, t] = -1
        
        # Update matched trackers
        for d, t in matched_indices:
            self.trackers[t].update(detections[d][0])
        
        # Create new trackers for unmatched detections
        for i in unmatched_detections:
            bbox, conf, class_id = detections[i]
            
            # Increment counter for this class
            self.class_counters[class_id] += 1
            
            # Create new tracker with class-specific ID
            new_tracker = KalmanTracker(bbox, class_id, self.class_counters[class_id])
            self.trackers.append(new_tracker)
        
        # Remove dead trackers
        self.trackers = [t for i, t in enumerate(self.trackers) 
                         if i not in unmatched_trackers or 
                         t.time_since_update <= self.max_age]
        
        # Return active tracks with ROI information
        result = []
        self.roi_trackers = set()  # Reset ROI trackers for this frame
        
        for tracker in self.trackers:
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = tracker.get_state()
                
                # Calculate ROI intersection
                intersection_score, is_intersecting = self._calculate_roi_intersection(bbox)
                
                # If intersecting with ROI, add to roi_trackers set and update frame count
                if is_intersecting:
                    self.roi_trackers.add(tracker.track_id)
                    
                    # Increment frame count for this tracker in ROI
                    if tracker.track_id in self.roi_tracker_counts:
                        self.roi_tracker_counts[tracker.track_id] += 1
                    else:
                        self.roi_tracker_counts[tracker.track_id] = 1
                    
                    # Add to cumulative set if it has been in ROI for minimum frames
                    if self.roi_tracker_counts[tracker.track_id] >= self.min_roi_frames:
                        self.cumulative_roi_trackers.add(tracker.track_id)
                else:
                    # Reset counter if object leaves ROI
                    if tracker.track_id in self.roi_tracker_counts:
                        self.roi_tracker_counts[tracker.track_id] = 0
                
                result.append((bbox, tracker.track_id, tracker.class_id, tracker.history, intersection_score))
        
        return result


def main():
    # Load YOLO model
    model_path = "/home/diego/projects/COGTIVE/aivision-core/data/models/betterbeef/weight.pt"
    try:
        model = YOLO(model_path)
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
        
    # Define ROI color
    ROI_COLOR = (0, 0, 255)  # Red for ROI
    
    # Open video file
    video_path = "/home/diego/Videos/Cogtive/betterbeef2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Validate video input
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit(1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Read first frame to validate
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        exit(1)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Set up output video with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"kalman_tracked_5min_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: VideoWriter failed to initialize.")
        cap.release()
        exit(1)
    
    # Initialize tracker with increased max_age for longer persistence
    tracker = MultiObjectTracker(max_age=120, min_hits=3, iou_threshold=0.3)
    
    # Use purple and green colors for detected objects
    class_names = model.names
    # Purple and green colors for different classes
    colors = {}
    for i in range(len(class_names)):
        if i % 2 == 0:  # Even class IDs get purple
            colors[i] = (255, 0, 255)  # Purple
        else:  # Odd class IDs get green
            colors[i] = (0, 255, 0)  # Green
    
    # Class-specific counters for visualization
    class_track_counts = defaultdict(int)
    
    # Counter for objects in ROI
    roi_object_counts = defaultdict(int)
    
    # Set to track all unique objects that have entered the ROI
    all_roi_objects = set()
    
    frame_count = 0
    
    # Calculate how many frames to process for 5 minutes
    total_frames_for_5min = int(fps * 60 * 5)
    print(f"Processing {total_frames_for_5min} frames for a 5-minute video at {fps} FPS")
    
    # Track start time for progress reporting
    start_time = time.time()
    
    # Calculate frames to show progress (every 5%)
    progress_interval = max(1, total_frames_for_5min // 20)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached before 5 minutes.")
            break
        
        frame_count += 1
        
        # Stop after processing enough frames for 5 minutes
        if frame_count > total_frames_for_5min:
            print(f"Reached 5 minutes of video ({total_frames_for_5min} frames).")
            break
        
        # Show progress every 5%
        if frame_count % progress_interval == 0:
            progress = (frame_count / total_frames_for_5min) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (frame_count / total_frames_for_5min)
            remaining_time = estimated_total - elapsed_time
            print(f"Progress: {progress:.1f}% - Frame {frame_count}/{total_frames_for_5min} - Est. remaining: {remaining_time:.1f}s")
        
        annotated_frame = frame.copy()
        
        # Run YOLO detection with 20% confidence threshold
        results = model(frame, conf=0.2, iou=0.3)
        
        # Extract all detections for display
        all_detections = []
        # Extract MEAT detections for tracking
        meat_detections = []
        
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                # Add to all detections for display
                all_detections.append((box, conf, cls))
                
                # Only track MEAT class
                if class_names[cls] == 'MEAT':
                    meat_detections.append((box, conf, cls))
        
        # Update tracker with MEAT detections only
        tracks = tracker.update(meat_detections)
        
        # Draw all detections first (including PERSON)
        for bbox, conf, class_id in all_detections:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Get class-specific color
            color = colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw class name and confidence
            label = f"{class_names[class_id]} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracking results (only for MEAT)
        for bbox, track_id, class_id, history, intersection_score in tracks:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Get class-specific color
            color = colors[class_id]
            
            # Update class-specific counter if this is a new track
            if track_id > class_track_counts[class_id]:
                class_track_counts[class_id] = track_id
            
            # Check if object is in ROI
            is_in_roi = track_id in tracker.roi_trackers
            
            # Draw track ID on top of the already drawn bounding box
            track_label = f"#{track_id}"
            if is_in_roi:
                # Add intersection score for objects in ROI
                track_label += f" {intersection_score:.2f}"
                # Use a different color (red) for objects in ROI
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ROI_COLOR, 3)
            
            cv2.putText(annotated_frame, track_label, (x1, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw tracking history (trail) in white color
            if len(history) > 1:
                for i in range(1, len(history)):
                    # Draw line between consecutive points with white color
                    cv2.line(annotated_frame, history[i-1], history[i], (255, 255, 255), 2)
        
        # Draw class counters in top-left corner
        y_offset = 30
        for cls_id, count in class_track_counts.items():
            text = f"{class_names[cls_id]}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cls_id], 2)
            y_offset += 30
            
        # Draw ROI on the frame
        cv2.rectangle(annotated_frame, ROI_COORDS[0], ROI_COORDS[1], ROI_COLOR, 2)
        
        # Draw ROI counters at the top of the image
        # Current objects in ROI
        roi_count = len(tracker.roi_trackers)
        roi_text = f"Objects in ROI: {roi_count}"
        cv2.putText(annotated_frame, roi_text, (frame_width // 2 - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ROI_COLOR, 2)
        
        # Cumulative unique objects that have entered ROI for at least min_roi_frames
        cumulative_roi_count = len(tracker.cumulative_roi_trackers)
        cumulative_text = f"Objects Passed (10+ frames): {cumulative_roi_count}"
        cv2.putText(annotated_frame, cumulative_text, (frame_width // 2 - 100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ROI_COLOR, 2)
        
        # Display frame number and time information
        elapsed_video_time = frame_count / fps
        minutes = int(elapsed_video_time // 60)
        seconds = int(elapsed_video_time % 60)
        time_text = f"Video Time: {minutes:02d}:{seconds:02d}"
        
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames_for_5min}", (10, frame_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, time_text, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Show frame (optional, can be commented out for faster processing)
        cv2.imshow("Kalman Filter Tracking (5min)", annotated_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break
    
    # Calculate actual processing time
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds.")
    print(f"Processed {frame_count} frames out of {total_frames_for_5min} planned.")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"5-minute tracking video saved as {output_path}")


if __name__ == "__main__":
    main()
