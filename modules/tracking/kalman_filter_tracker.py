import cv2
import numpy as np
import random
import os
import sys
import json
import pickle
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import YOLO, handle potential version issues
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics==8.0.20")
    from ultralytics import YOLO

# Define the Region of Interest (ROI)
ROI_COORDS = [(570, 190), (1030, 415)]  # [(x1, y1), (x2, y2)]

# Define workstation ROIs - will be populated dynamically
WORKSTATION_ROIS = {}

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
        self.last_position = bbox  # Store last known position
        
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
        
        # Initialize error covariance
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        
        # History of positions for visualization
        self.history = []
        
        # Additional metadata for JSON tracking
        self.workstation = None
        self.region = None
        
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
        
        # If prediction is unreasonable (e.g., negative width/height), use last position
        if w <= 0 or h <= 0:
            return self.last_position
        
        return predicted_bbox
    
    def update(self, bbox):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.hit_streak += 1
        self.last_position = bbox  # Update last known position
        
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
        
        # Ensure width and height are positive
        w = max(5, w)  # Minimum width of 5 pixels
        h = max(5, h)  # Minimum height of 5 pixels
        
        return self._xyxy_to_bbox(x, y, w, h)


class MultiObjectTracker:
    """
    Multi-object tracker using Kalman Filter
    """
    def __init__(self, max_age=None, min_hits=None, iou_threshold=None, state_path=None):
        # Load parameters from environment variables if available, otherwise use defaults
        self.max_age = int(os.environ.get('KALMAN_MAX_AGE', 10)) if max_age is None else max_age
        self.min_hits = int(os.environ.get('KALMAN_MIN_HITS', 3)) if min_hits is None else min_hits
        self.iou_threshold = float(os.environ.get('KALMAN_IOU_THRESHOLD', 0.3)) if iou_threshold is None else iou_threshold
        
        self.trackers = []  # List of active trackers
        self.frame_count = 0
        
        # Counters for each class
        self.roi_trackers = set()  # Set to track objects currently in ROI
        self.roi_tracker_counts = {}  # Dictionary to track how many frames each object has been in ROI
        self.cumulative_roi_trackers = set()  # Set to track objects that have been in ROI for minimum frames
        self.class_counters = defaultdict(int)
        
        # Minimum frames an object must stay in ROI to be counted
        self.min_roi_frames = int(os.environ.get('KALMAN_MIN_ROI_FRAMES', 10))
        
        # Store track history for persistent tracking
        self.track_history = {}  # Dictionary to store track history by class and location
        
        # Track history retention (in frames)
        self.history_retention = int(os.environ.get('KALMAN_HISTORY_RETENTION', 300))
        
        # Grid size for location binning (smaller grid for more precise location tracking)
        self.grid_size = 15  # Reduced to 15 for even more precise location binning
        
        # Store appearance features for better matching
        self.appearance_features = {}  # Dictionary to store appearance features by track_id
        
        # Global object registry to maintain persistent IDs
        self.global_objects = {}  # Dictionary to store all objects ever seen
        
        # Spatial grid to quickly find nearby objects
        self.spatial_grid = {}  # Dictionary to store objects by grid location
        
        # Path to save/load tracker state
        self.state_path = state_path if state_path else "/home/diego/projects/COGTIVE/aivision-core/data/tracking/tracker_state.pkl"
        
        # Load state if it exists
        self.load_state()
        
        # Auto-save interval (in frames)
        self.save_interval = 30  # Save state every 30 frames
        self.last_save_time = time.time()
        self.save_timeout = 5  # Minimum seconds between saves to prevent excessive disk I/O
        
        # Workstation and region tracking
        self.workstation_regions = {}  # Dictionary to store regions by workstation
        
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
        
    def _calculate_roi_intersection(self, bbox, workstation=None, region=None):
        """Calculate intersection between a bounding box and the ROI
        Returns intersection score (0-1) and whether the bbox intersects with ROI"""
        # If workstation and region are provided, use those for ROI calculation
        if workstation and region and workstation in WORKSTATION_ROIS and region in WORKSTATION_ROIS[workstation]:
            roi = WORKSTATION_ROIS[workstation][region]
        else:
            # Fall back to default ROI
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
            detections: List of tuples (bbox, confidence, class_id, track_id, workstation, region)
                track_id, workstation, and region are optional
        
        Returns:
            List of tuples (bbox, track_id, class_id, history, intersection_score)
        """
        self.frame_count += 1
        
        # Predict new locations of existing trackers
        for tracker in self.trackers:
            tracker.predict()
        
        # Get predicted states of all trackers
        predicted_states = [tracker.get_state() for tracker in self.trackers]
        
        # Update spatial grid with current tracker positions
        self._update_spatial_grid()
        
        # Extract detections without tracking metadata for matching
        detections_for_matching = [(d[0], d[1], d[2]) for d in detections]
        
        # Match detections with existing trackers
        matched_indices, unmatched_detections, unmatched_trackers = self._match_detections_to_trackers(
            detections_for_matching, predicted_states
        )
        
        # Update matched trackers with new detections
        for i, j in matched_indices:
            bbox, _, class_id = detections_for_matching[j]
            # Get full detection data including optional fields
            full_detection = detections[j]
            
            # Check if a tracking ID was provided in the detection
            if len(full_detection) > 3 and full_detection[3] is not None:
                # If the tracking ID doesn't match, handle as a new detection
                if full_detection[3] != self.trackers[i].track_id:
                    unmatched_detections.append(j)
                    unmatched_trackers.append(i)
                    continue
            
            self.trackers[i].update(bbox)
            
            # Update workstation and region if provided
            if len(full_detection) > 4 and full_detection[4] is not None:
                self.trackers[i].workstation = full_detection[4]
            if len(full_detection) > 5 and full_detection[5] is not None:
                self.trackers[i].region = full_detection[5]
            
            # Store appearance features for this track
            self._update_appearance_features(self.trackers[i].track_id, bbox, class_id)
            
            # Update global object registry
            track_id = self.trackers[i].track_id
            self._update_global_object(track_id, bbox, class_id)
        
        # Process unmatched detections - try to associate with existing global objects or use provided tracking ID
        for i in unmatched_detections:
            bbox, conf, class_id = detections_for_matching[i]
            # Get full detection data including optional fields
            full_detection = detections[i]
            
            # Check if a tracking ID was provided in the detection
            provided_track_id = None
            if len(full_detection) > 3 and full_detection[3] is not None:
                provided_track_id = full_detection[3]
            
            # If tracking ID was provided, use it; otherwise find best match or create new
            if provided_track_id is not None:
                track_id = provided_track_id
                
                # Check if this ID already exists in our trackers
                existing_tracker = None
                for t in self.trackers:
                    if t.track_id == track_id:
                        existing_tracker = t
                        break
                
                if existing_tracker is not None:
                    # Update existing tracker
                    existing_tracker.update(bbox)
                else:
                    # Create new tracker with provided ID
                    new_tracker = KalmanTracker(bbox, class_id, track_id)
                    self.trackers.append(new_tracker)
                    
                    # Register this object globally if not already registered
                    if track_id not in self.global_objects:
                        self._register_global_object(track_id, bbox, class_id)
                    else:
                        self._update_global_object(track_id, bbox, class_id)
            else:
                # No tracking ID provided, try to find a match with any inactive but recent tracker
                track_id = self._find_best_global_match(bbox, class_id)
                
                if track_id is None:
                    # No match found in global registry, create new tracker with new ID
                    self.class_counters[class_id] += 1
                    track_id = self.class_counters[class_id]
                    new_tracker = KalmanTracker(bbox, class_id, track_id)
                    self.trackers.append(new_tracker)
                    
                    # Register this new object globally
                    self._register_global_object(track_id, bbox, class_id)
                else:
                    # Match found in global registry, create new tracker with existing ID
                    new_tracker = KalmanTracker(bbox, class_id, track_id)
                    self.trackers.append(new_tracker)
                    
                    # Update the global object
                    self._update_global_object(track_id, bbox, class_id)
            
            # Set workstation and region if provided
            tracker_to_update = None
            for t in self.trackers:
                if t.track_id == track_id:
                    tracker_to_update = t
                    break
            
            if tracker_to_update is not None:
                if len(full_detection) > 4 and full_detection[4] is not None:
                    tracker_to_update.workstation = full_detection[4]
                if len(full_detection) > 5 and full_detection[5] is not None:
                    tracker_to_update.region = full_detection[5]
            
            # Update appearance features
            self._update_appearance_features(track_id, bbox, class_id)
            
            # Create a key for this detection based on class and location
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Use smaller grid size for more precise location binning
            location_key = f"{class_id}_{int(center_x/self.grid_size)}_{int(center_y/self.grid_size)}"
            
            # Update track history
            self.track_history[location_key] = {
                'track_id': track_id,
                'last_seen': self.frame_count,
                'bbox': bbox,
                'confidence': conf
            }
        
        # Remove dead trackers but keep them in global registry
        self.trackers = [t for i, t in enumerate(self.trackers) 
                         if i not in unmatched_trackers or 
                         t.time_since_update <= self.max_age]
        
        # Update track history for all active trackers
        for tracker in self.trackers:
            bbox = tracker.get_state()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Use smaller grid size for more precise location binning
            location_key = f"{tracker.class_id}_{int(center_x/self.grid_size)}_{int(center_y/self.grid_size)}"
            
            self.track_history[location_key] = {
                'track_id': tracker.track_id,
                'last_seen': self.frame_count,
                'bbox': bbox,
                'confidence': 1.0  # Assume high confidence for tracked objects
            }
            
            # Update global object
            self._update_global_object(tracker.track_id, bbox, tracker.class_id)
        
        # Clean up old track history entries
        keys_to_remove = []
        for key, value in self.track_history.items():
            if self.frame_count - value['last_seen'] > self.history_retention:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.track_history[key]
        
        # Auto-save state periodically
        if self.frame_count % self.save_interval == 0 and time.time() - self.last_save_time > self.save_timeout:
            self.save_state()
            self.last_save_time = time.time()
        
        # Return active tracks with ROI information
        result = []
        self.roi_trackers = set()  # Reset ROI trackers for this frame
        
        for tracker in self.trackers:
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = tracker.get_state()
                
                # Calculate ROI intersection using workstation and region if available
                intersection_score, is_intersecting = self._calculate_roi_intersection(
                    bbox, tracker.workstation, tracker.region
                )
                
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
    
    def _update_spatial_grid(self):
        """Update spatial grid with current tracker positions"""
        # Clear the spatial grid
        self.spatial_grid = {}
        
        # Add all active trackers to the spatial grid
        for tracker in self.trackers:
            bbox = tracker.get_state()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            grid_x = int(center_x / self.grid_size)
            grid_y = int(center_y / self.grid_size)
            
            grid_key = f"{grid_x}_{grid_y}"
            
            if grid_key not in self.spatial_grid:
                self.spatial_grid[grid_key] = []
                
            self.spatial_grid[grid_key].append({
                'track_id': tracker.track_id,
                'class_id': tracker.class_id,
                'bbox': bbox,
                'center': (center_x, center_y)
            })
    
    def _update_appearance_features(self, track_id, bbox, class_id):
        """
        Update appearance features for a tracked object
        
        Args:
            track_id: ID of the tracked object
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class ID of the object
        """
        # Simple feature is just the size and aspect ratio of the bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 1.0
        size = width * height
        
        # Store features
        self.appearance_features[track_id] = {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'size': size,
            'class_id': class_id
        }
    
    def _register_global_object(self, track_id, bbox, class_id):
        """Register a new object in the global registry"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        self.global_objects[track_id] = {
            'class_id': class_id,
            'last_bbox': bbox,
            'last_seen': self.frame_count,
            'first_seen': self.frame_count,
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'center': (center_x, center_y),
            'positions': [(center_x, center_y)],
            'active': True
        }
    
    def _update_global_object(self, track_id, bbox, class_id):
        """Update an existing object in the global registry"""
        if track_id in self.global_objects:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Update object properties
            self.global_objects[track_id]['last_bbox'] = bbox
            self.global_objects[track_id]['last_seen'] = self.frame_count
            self.global_objects[track_id]['width'] = width
            self.global_objects[track_id]['height'] = height
            self.global_objects[track_id]['aspect_ratio'] = width / height if height > 0 else 0
            self.global_objects[track_id]['center'] = (center_x, center_y)
            self.global_objects[track_id]['positions'].append((center_x, center_y))
            self.global_objects[track_id]['active'] = True
            
            # Limit the number of positions stored to prevent memory issues
            if len(self.global_objects[track_id]['positions']) > 100:
                self.global_objects[track_id]['positions'] = self.global_objects[track_id]['positions'][-100:]
        else:
            # Object not found, register it
            self._register_global_object(track_id, bbox, class_id)
    
    def _find_best_global_match(self, bbox, class_id):
        """
        Find the best match for this detection in the global object registry
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class ID
            
        Returns:
            track_id if match found, None otherwise
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 0
        
        best_match = None
        best_score = 0.1  # Minimum threshold for a match
        
        # Check nearby grid cells for potential matches
        for dx in range(-2, 3):  # Expanded to 5x5 grid
            for dy in range(-2, 3):
                grid_x = int(center_x / self.grid_size) + dx
                grid_y = int(center_y / self.grid_size) + dy
                
                grid_key = f"{grid_x}_{grid_y}"
                
                if grid_key in self.spatial_grid:
                    for obj in self.spatial_grid[grid_key]:
                        if obj['class_id'] == class_id:
                            # Calculate Euclidean distance between centers
                            obj_center_x, obj_center_y = obj['center']
                            distance = np.sqrt((center_x - obj_center_x)**2 + (center_y - obj_center_y)**2)
                            
                            # Calculate size similarity
                            obj_width = obj['bbox'][2] - obj['bbox'][0]
                            obj_height = obj['bbox'][3] - obj['bbox'][1]
                            size_diff = abs(width * height - obj_width * obj_height) / (width * height)
                            
                            # Combined score
                            score = (1.0 / (1.0 + distance/100)) * 0.5 + (1.0 / (1.0 + size_diff)) * 0.3 + (1.0 / (1.0 + abs(aspect_ratio - obj['aspect_ratio']))) * 0.2
                            
                            if score > best_score:
                                best_score = score
                                best_match = obj['track_id']
        
        # If no match found in spatial grid, check all global objects
        if best_match is None:
            for track_id, obj in self.global_objects.items():
                if obj['class_id'] == class_id and self.frame_count - obj['last_seen'] <= self.history_retention:
                    obj_center_x, obj_center_y = obj['center']
                    distance = np.sqrt((center_x - obj_center_x)**2 + (center_y - obj_center_y)**2)
                    
                    # If object is close enough, consider it a match
                    if distance < 100:  # Increased distance threshold
                        size_diff = abs(width * height - obj['width'] * obj['height']) / (width * height)
                        aspect_diff = abs(aspect_ratio - obj['aspect_ratio'])
                        
                        # Combined score
                        score = (1.0 / (1.0 + distance/100)) * 0.6 + (1.0 / (1.0 + size_diff)) * 0.3 + (1.0 / (1.0 + aspect_diff)) * 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_match = track_id
        
        return best_match
    
    def _match_detections_to_trackers(self, detections, predicted_states):
        """
        Match detections with existing trackers
        
        Args:
            detections: List of tuples (bbox, confidence, class_id)
            predicted_states: List of predicted states of all trackers
        
        Returns:
            matched_indices: List of tuples (tracker_index, detection_index)
            unmatched_detections: List of indices of unmatched detections
            unmatched_trackers: List of indices of unmatched trackers
        """
        # Calculate distance matrix instead of IOU matrix
        distance_matrix = np.zeros((len(detections), len(predicted_states)), dtype=np.float32)
        for d, detection in enumerate(detections):
            det_bbox = detection[0]
            det_center_x = (det_bbox[0] + det_bbox[2]) / 2
            det_center_y = (det_bbox[1] + det_bbox[3]) / 2
            
            for t, prediction in enumerate(predicted_states):
                pred_center_x = (prediction[0] + prediction[2]) / 2
                pred_center_y = (prediction[1] + prediction[3]) / 2
                
                # Calculate Euclidean distance between centers
                distance = np.sqrt((det_center_x - pred_center_x)**2 + (det_center_y - pred_center_y)**2)
                distance_matrix[d, t] = distance
        
        # Use Hungarian algorithm for optimal assignment
        # For simplicity, we'll use a greedy approach here
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(predicted_states)))
        
        # Define a maximum distance threshold (replacing IoU threshold)
        max_distance_threshold = 100  # Adjust this value based on your specific needs
        
        while True:
            # Find min distance
            if len(unmatched_detections) == 0 or len(unmatched_trackers) == 0:
                break
                
            # Create a mask for valid entries
            valid_entries = np.ones(distance_matrix.shape, dtype=bool)
            for d in range(len(detections)):
                if d not in unmatched_detections:
                    valid_entries[d, :] = False
            for t in range(len(predicted_states)):
                if t not in unmatched_trackers:
                    valid_entries[:, t] = False
            
            # Apply mask
            masked_distance_matrix = np.where(valid_entries, distance_matrix, np.inf)
            
            # If all remaining distances are too large, break
            if np.min(masked_distance_matrix) > max_distance_threshold:
                break
            
            # Get indices of min distance
            d, t = np.unravel_index(np.argmin(masked_distance_matrix), masked_distance_matrix.shape)
            
            # Add to matched indices
            matched_indices.append((t, d))
            
            # Remove from unmatched
            if d in unmatched_detections:
                unmatched_detections.remove(d)
            if t in unmatched_trackers:
                unmatched_trackers.remove(t)
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    def save_state(self):
        """Save tracker state to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            
            # Prepare state dictionary
            state = {
                'frame_count': self.frame_count,
                'class_counters': dict(self.class_counters),
                'global_objects': self.global_objects,
                'track_history': self.track_history,
                'appearance_features': self.appearance_features,
                'roi_tracker_counts': self.roi_tracker_counts,
                'cumulative_roi_trackers': list(self.cumulative_roi_trackers)
            }
            
            # Save state to disk
            with open(self.state_path, 'wb') as f:
                pickle.dump(state, f)
                
            print(f"Tracker state saved to {self.state_path}")
        except Exception as e:
            print(f"Error saving tracker state: {e}")
    
    def load_state(self):
        """Load tracker state from disk"""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'rb') as f:
                    state = pickle.load(f)
                
                # Restore state
                self.frame_count = state.get('frame_count', 0)
                self.class_counters = defaultdict(int, state.get('class_counters', {}))
                self.global_objects = state.get('global_objects', {})
                self.track_history = state.get('track_history', {})
                self.appearance_features = state.get('appearance_features', {})
                self.roi_tracker_counts = state.get('roi_tracker_counts', {})
                self.cumulative_roi_trackers = set(state.get('cumulative_roi_trackers', []))
                
                print(f"Tracker state loaded from {self.state_path}")
                print(f"Loaded {len(self.global_objects)} global objects")
                
                # Recreate active trackers from global objects
                self._recreate_trackers_from_global_objects()
            else:
                print(f"No tracker state found at {self.state_path}")
        except Exception as e:
            print(f"Error loading tracker state: {e}")
    
    def _recreate_trackers_from_global_objects(self):
        """Recreate active trackers from global objects"""
        # Clear existing trackers
        self.trackers = []
        
        # Find recently active global objects
        recent_frame_threshold = self.frame_count - 10  # Consider objects seen in the last 10 frames
        
        for track_id, obj in self.global_objects.items():
            if obj['last_seen'] >= recent_frame_threshold:
                # Create new tracker for this object
                bbox = obj['last_bbox']
                class_id = obj['class_id']
                
                new_tracker = KalmanTracker(bbox, class_id, track_id)
                self.trackers.append(new_tracker)
    
    def set_workstation_roi(self, workstation, region, roi_coords):
        """
        Set ROI coordinates for a specific workstation and region
        
        Args:
            workstation: Workstation name
            region: Region name
            roi_coords: ROI coordinates in format [x1, y1, x2, y2]
        """
        if workstation not in WORKSTATION_ROIS:
            WORKSTATION_ROIS[workstation] = {}
        
        WORKSTATION_ROIS[workstation][region] = roi_coords
        
        # Update workstation regions mapping
        if workstation not in self.workstation_regions:
            self.workstation_regions[workstation] = []
        
        if region not in self.workstation_regions[workstation]:
            self.workstation_regions[workstation].append(region)


def process_json_data(json_data, tracker=None, state_path=None):
    """
    Process JSON tracking data and update the tracker
    
    Args:
        json_data: JSON data in the format provided in the example
        tracker: MultiObjectTracker instance (optional)
        state_path: Path to save/load tracker state (optional)
        
    Returns:
        Updated JSON data with tracking information and the tracker instance
    """
    # Initialize tracker if not provided
    if tracker is None:
        tracker = MultiObjectTracker(
            max_age=120, 
            min_hits=3, 
            iou_threshold=0.3, 
            state_path=state_path if state_path else "/home/diego/projects/COGTIVE/aivision-core/data/tracking/json_tracker_state.pkl"
        )
    
    # Define class mapping
    class_mapping = {
        'MEAT': 0,
        'PERSON': 1
    }
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    
    # Update workstation ROIs
    if 'workstations' in json_data:
        for workstation in json_data['workstations']:
            workstation_name = workstation['workstationName']
            if 'regionsOfInterest' in workstation:
                for region in workstation['regionsOfInterest']:
                    region_name = region['regionName']
                    # Set default ROI if not already set
                    if workstation_name not in WORKSTATION_ROIS or region_name not in WORKSTATION_ROIS[workstation_name]:
                        # Use default ROI coordinates for now
                        tracker.set_workstation_roi(workstation_name, region_name, 
                                                  [ROI_COORDS[0][0], ROI_COORDS[0][1], ROI_COORDS[1][0], ROI_COORDS[1][1]])
    
    # Extract detections from JSON
    detections = []
    if 'objectsInFrame' in json_data:
        for obj in json_data['objectsInFrame']:
            # Convert class label to class ID
            class_label = obj['classLabel']
            class_id = class_mapping.get(class_label, 0)  # Default to 0 if class not found
            
            # Convert bounding box format
            bbox_dict = obj['boundingBox']
            x = bbox_dict['x']
            y = bbox_dict['y']
            width = bbox_dict['width']
            height = bbox_dict['height']
            bbox = [x, y, x + width, y + height]  # Convert to [x1, y1, x2, y2] format
            
            # Get confidence
            confidence = obj['confidence']
            
            # Get tracking ID if available
            tracking_id = int(obj['trackingId']) if 'trackingId' in obj else None
            
            # Add workstation and region information
            workstation_name = obj.get('workstationName', None)
            region_name = obj.get('regionName', None)
            
            # Add to detections list
            detections.append((bbox, confidence, class_id, tracking_id, workstation_name, region_name))
    
    # Update tracker with detections
    tracks = tracker.update(detections)
    
    # Update JSON data with tracking results
    updated_objects = []
    for obj in json_data.get('objectsInFrame', []):
        # Find corresponding track
        for bbox, track_id, class_id, history, intersection_score in tracks:
            # Check if this is the same object based on tracking ID
            if 'trackingId' in obj and int(obj['trackingId']) == track_id:
                # Update bounding box
                x1, y1, x2, y2 = [int(v) for v in bbox]
                obj['boundingBox'] = {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                
                # Add tracking information
                obj['trackingInfo'] = {
                    'intersectionScore': float(intersection_score),
                    'isInROI': track_id in tracker.roi_trackers,
                    'timeInROI': tracker.roi_tracker_counts.get(track_id, 0)
                }
                break
        
        updated_objects.append(obj)
    
    # Update JSON data
    json_data['objectsInFrame'] = updated_objects
    
    # Update detections section with cumulative tracking information
    updated_detections = []
    for detection in json_data.get('detections', []):
        class_label = detection['classLabel']
        tracking_ids = detection.get('trackingIds', [])
        
        # Check if any of the tracking IDs are in the ROI
        in_roi = any(int(tid) in tracker.roi_trackers for tid in tracking_ids)
        
        # Check if any of the tracking IDs have been in the ROI for minimum frames
        in_cumulative = any(int(tid) in tracker.cumulative_roi_trackers for tid in tracking_ids)
        
        # Update workstation information
        for workstation in detection.get('workstations', []):
            workstation_name = workstation['workstationName']
            for roi in workstation.get('regionsOfInterests', []):
                roi_name = roi['regionName']
                
                # Update counters based on tracking information
                if in_roi:
                    roi['instantCounter'] = sum(1 for tid in tracking_ids if int(tid) in tracker.roi_trackers)
                else:
                    roi['instantCounter'] = 0
                    
                if in_cumulative:
                    roi['counter'] = sum(1 for tid in tracking_ids if int(tid) in tracker.cumulative_roi_trackers)
        
        updated_detections.append(detection)
    
    json_data['detections'] = updated_detections
    
    return json_data, tracker


def main():
    # Check if JSON file path is provided as command line argument
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        json_file_path = sys.argv[1]
        
        # Load JSON data
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Process JSON data
        processed_data, _ = process_json_data(json_data)
        
        # Save processed data
        output_path = json_file_path.replace('.json', '_processed.json')
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Processed JSON data saved to {output_path}")
        return
    
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
    
    # Set up output video
    output_path = "kalman_tracked_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: VideoWriter failed to initialize.")
        cap.release()
        exit(1)
    
    # Initialize tracker with increased max_age for longer persistence
    # Use a specific state path for this video
    state_path = "/home/diego/projects/COGTIVE/aivision-core/data/tracking/betterbeef_tracker_state.pkl"
    tracker = MultiObjectTracker(max_age=120, min_hits=3, iou_threshold=0.3, state_path=state_path)
    
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
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
            
            # Draw bounding box with different style based on tracking status
            if tracker.trackers[tracks.index((bbox, track_id, class_id, history, intersection_score))].time_since_update == 0:
                # Object is currently being tracked - solid line
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            else:
                # Object was tracked but not matched in current frame - dashed line
                for i in range(x1, x2, 10):
                    cv2.line(annotated_frame, (i, y1), (min(i+5, x2), y1), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (i, y2), (min(i+5, x2), y2), (0, 0, 255), 2)
                for i in range(y1, y2, 10):
                    cv2.line(annotated_frame, (x1, i), (x1, min(i+5, y2)), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (x2, i), (x2, min(i+5, y2)), (0, 0, 255), 2)
            
            # Draw track ID with colored background for better visibility
            track_label = f"#{track_id}"
            
            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for track ID
            bg_color = (50, 50, 200) if is_in_roi else color  # Blue background for ROI objects
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_height - 8), 
                (x1 + text_width + 8, y1), 
                bg_color, 
                -1  # Filled rectangle
            )
            
            # Draw track ID text in white for better contrast
            cv2.putText(
                annotated_frame, 
                track_label, 
                (x1 + 4, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255),  # White text
                2
            )
            
            # Draw intersection score for objects in ROI
            if is_in_roi:
                score_label = f"{intersection_score:.2f}"
                cv2.putText(
                    annotated_frame, 
                    score_label, 
                    (x1, y1 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 255),  # Yellow text for score
                    2
                )
                
                # Highlight objects in ROI with a thicker border
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ROI_COLOR, 3)
            
            # Draw tracking history (path)
            if len(history) > 1:
                # Convert history points to numpy array
                points = np.array(history, dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                
                # Draw polylines for tracking path
                cv2.polylines(
                    annotated_frame, 
                    [points], 
                    False, 
                    color, 
                    2
                )
        
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
        
        # Display frame number
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show and write frame
        cv2.imshow("Kalman Filter Tracking", annotated_frame)
        out.write(annotated_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Tracking video saved as {output_path}")


if __name__ == "__main__":
    main()
