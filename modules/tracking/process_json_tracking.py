import json
import os
import sys
import time
from collections import defaultdict
import numpy as np
import cv2
import pickle

# Add the parent directory to the path to import the kalman_filter_tracker module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.kalman_filter_tracker import KalmanTracker, MultiObjectTracker

class JSONTracker:
    """
    Process JSON tracking data using Kalman Filter tracker
    """
    def __init__(self, state_path=None):
        # Initialize the tracker
        self.tracker = MultiObjectTracker(
            max_age=120, 
            min_hits=3, 
            iou_threshold=0.3, 
            state_path=state_path if state_path else "/home/diego/projects/COGTIVE/aivision-core/data/tracking/json_tracker_state.pkl"
        )
        
        # Store workstation and region information
        self.workstations = {}
        self.class_mapping = {
            'MEAT': 0,
            'PERSON': 1
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        
    def _convert_bbox_format(self, bbox_dict):
        """
        Convert bounding box from JSON format (x, y, width, height) to tracker format (x1, y1, x2, y2)
        """
        x = bbox_dict['x']
        y = bbox_dict['y']
        width = bbox_dict['width']
        height = bbox_dict['height']
        
        return [x, y, x + width, y + height]
    
    def _convert_bbox_to_json(self, bbox):
        """
        Convert bounding box from tracker format (x1, y1, x2, y2) to JSON format (x, y, width, height)
        """
        x1, y1, x2, y2 = bbox
        return {
            'x': int(x1),
            'y': int(y1),
            'width': int(x2 - x1),
            'height': int(y2 - y1)
        }
    
    def process_json_data(self, json_data):
        """
        Process JSON tracking data and update the tracker
        
        Args:
            json_data: JSON data in the format provided in the example
            
        Returns:
            Updated JSON data with tracking information
        """
        # Extract client_id and frame_id from top level if available
        client_id = json_data.get('client_id', None)
        frame_id = json_data.get('frame_id', None)
        
        # Check if we have the new format with tracker_response
        tracker_response = json_data.get('tracker_response', None)
        
        # Update workstation information from tracker_response if available
        if tracker_response and 'detections' in tracker_response:
            for detection in tracker_response['detections']:
                if 'workstations' in detection:
                    for workstation in detection['workstations']:
                        workstation_name = workstation['workstationName']
                        regions = []
                        if 'regionsOfInterests' in workstation:
                            for region in workstation['regionsOfInterests']:
                                regions.append(region['regionName'])
                        self.workstations[workstation_name] = regions
        
        # Extract detections from JSON
        detections = []
        objects_by_tracking_id = {}
        
        # Process objectsInFrame
        if 'objectsInFrame' in json_data:
            for obj in json_data['objectsInFrame']:
                # Convert class label to class ID
                class_label = obj['classLabel']
                class_id = self.class_mapping.get(class_label, 0)  # Default to 0 if class not found
                
                # Convert bounding box format
                bbox = self._convert_bbox_format(obj['boundingBox'])
                
                # Get confidence
                confidence = obj['confidence']
                
                # Get tracking ID if available, otherwise use None
                tracking_id = int(obj['trackingId']) if 'trackingId' in obj else None
                
                # Add workstation and region information
                workstation_name = obj.get('workstationName', None)
                region_name = obj.get('regionName', None)
                
                # Add to detections list
                detections.append((bbox, confidence, class_id, tracking_id, workstation_name, region_name))
                
                # Store object by tracking ID for later use
                if tracking_id is not None:
                    if tracking_id not in objects_by_tracking_id:
                        objects_by_tracking_id[tracking_id] = []
                    objects_by_tracking_id[tracking_id].append(obj)
        
        # Update tracker with detections
        tracks = self.tracker.update(detections)
        
        # Update JSON data with tracking results
        updated_objects = []
        for obj in json_data.get('objectsInFrame', []):
            # Find corresponding track
            for bbox, track_id, class_id, history, intersection_score in tracks:
                # Check if this is the same object based on tracking ID
                if 'trackingId' in obj and int(obj['trackingId']) == track_id:
                    # Update bounding box with estimated position from Kalman filter
                    obj['trackingEstimatedBoundingBox'] = self._convert_bbox_to_json(bbox)
                    
                    # Add tracking information
                    obj['trackingInfo'] = {
                        'intersectionScore': float(intersection_score),
                        'isInROI': track_id in self.tracker.roi_trackers,
                        'timeInROI': self.tracker.roi_tracker_counts.get(track_id, 0)
                    }
                    
                    # Add tracking age if available in the tracker's global objects
                    if track_id in self.tracker.global_objects:
                        obj['trackingAge'] = self.tracker.global_objects[track_id].get('age', 0)
                    else:
                        obj['trackingAge'] = 0
                    break
            
            updated_objects.append(obj)
        
        # Update JSON data
        json_data['objectsInFrame'] = updated_objects
        
        # Update tracker_response if it exists
        if tracker_response:
            # Update detections section with cumulative tracking information
            updated_detections = []
            for detection in tracker_response.get('detections', []):
                class_label = detection['classLabel']
                
                # Get tracking IDs for this class from objectsInFrame
                tracking_ids = []
                for obj in json_data.get('objectsInFrame', []):
                    if obj.get('classLabel') == class_label and 'trackingId' in obj:
                        tracking_id = int(obj['trackingId'])
                        if tracking_id not in tracking_ids:
                            tracking_ids.append(tracking_id)
                
                # Add tracking IDs to detection
                detection['trackingIds'] = [str(tid) for tid in tracking_ids]
                
                # Check if any of the tracking IDs are in the ROI
                in_roi = any(tid in self.tracker.roi_trackers for tid in tracking_ids)
                
                # Check if any of the tracking IDs have been in the ROI for minimum frames
                in_cumulative = any(tid in self.tracker.cumulative_roi_trackers for tid in tracking_ids)
                
                # Update workstation information
                for workstation in detection.get('workstations', []):
                    workstation_name = workstation['workstationName']
                    for roi in workstation.get('regionsOfInterests', []):
                        roi_name = roi['regionName']
                        
                        # Update modes based on tracking information
                        modes = []
                        for mode in roi.get('modes', []):
                            if mode.get('key') == 'instantaneous':
                                if in_roi:
                                    mode['Value'] = sum(1 for tid in tracking_ids if tid in self.tracker.roi_trackers)
                                else:
                                    mode['Value'] = 0
                            elif mode.get('key') == 'accumulated':
                                if in_cumulative:
                                    mode['Value'] = sum(1 for tid in tracking_ids if tid in self.tracker.cumulative_roi_trackers)
                
                updated_detections.append(detection)
            
            tracker_response['detections'] = updated_detections
            json_data['tracker_response'] = tracker_response
        
        return json_data
    
    def save_state(self):
        """Save tracker state"""
        self.tracker.save_state()
    
    def load_state(self):
        """Load tracker state"""
        self.tracker.load_state()


def process_json_file(json_file_path, output_file_path=None, state_path=None):
    """
    Process a JSON file containing tracking data
    
    Args:
        json_file_path: Path to the JSON file
        output_file_path: Path to save the processed JSON file (optional)
        state_path: Path to save/load tracker state (optional)
        
    Returns:
        Processed JSON data
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    # Initialize tracker
    tracker = JSONTracker(state_path=state_path)
    
    # Process JSON data
    processed_data = tracker.process_json_data(json_data)
    
    # Save processed data if output file path is provided
    if output_file_path:
        with open(output_file_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
    
    # Save tracker state
    tracker.save_state()
    
    return processed_data


def process_json_string(json_string, state_path=None):
    """
    Process a JSON string containing tracking data
    
    Args:
        json_string: JSON string
        state_path: Path to save/load tracker state (optional)
        
    Returns:
        Processed JSON data as a string
    """
    # Parse JSON string
    json_data = json.loads(json_string)
    
    # Initialize tracker
    tracker = JSONTracker(state_path=state_path)
    
    # Process JSON data
    processed_data = tracker.process_json_data(json_data)
    
    # Save tracker state
    tracker.save_state()
    
    # Return processed data as JSON string
    return json.dumps(processed_data, indent=2)


if __name__ == "__main__":
    # Check if JSON file path is provided as command line argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        output_file_path = sys.argv[1].replace('.json', '_processed.json')
        process_json_file(json_file_path, output_file_path)
        print(f"Processed JSON data saved to {output_file_path}")
    else:
        print("Please provide a JSON file path as a command line argument.")
        sys.exit(1)
