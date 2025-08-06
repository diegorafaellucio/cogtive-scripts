#!/usr/bin/env python3
import json
import os
import sys
from modules.tracking.process_json_tracking import JSONTracker, process_json_string

def main():
    """
    Test the JSON tracking functionality with the new JSON format
    """
    # Check if JSON file path is provided as command line argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # Use the sample JSON data from temp.json
        json_file_path = "temp.json"
        
        if not os.path.exists(json_file_path):
            print(f"Error: {json_file_path} does not exist. Please provide a valid JSON file.")
            sys.exit(1)
    
    print(f"Processing JSON data from {json_file_path}...")
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    # Initialize the tracker
    tracker = JSONTracker()
    
    # Process JSON data
    processed_data = tracker.process_json_data(json_data)
    
    # Save processed data
    output_path = json_file_path.replace('.json', '_processed.json')
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed JSON data saved to {output_path}")
    
    # Print tracking statistics
    print("\nTracking Statistics:")
    print(f"Total objects tracked: {len(tracker.tracker.global_objects)}")
    print(f"Objects in ROI: {len(tracker.tracker.roi_trackers)}")
    print(f"Objects that have passed through ROI: {len(tracker.tracker.cumulative_roi_trackers)}")
    
    # Print details for each object in ROI
    if tracker.tracker.roi_trackers:
        print("\nObjects currently in ROI:")
        for track_id in tracker.tracker.roi_trackers:
            if track_id in tracker.tracker.global_objects:
                obj = tracker.tracker.global_objects[track_id]
                class_id = obj.get('class_id')
                class_name = tracker.reverse_class_mapping.get(class_id, "UNKNOWN")
                time_in_roi = tracker.tracker.roi_tracker_counts.get(track_id, 0)
                print(f"  ID: {track_id}, Class: {class_name}, Time in ROI: {time_in_roi} frames")
    
    # Print details for unique objects by class
    unique_objects_by_class = {}
    for obj in processed_data.get('objectsInFrame', []):
        class_label = obj.get('classLabel')
        tracking_id = obj.get('trackingId')
        
        if class_label and tracking_id:
            if class_label not in unique_objects_by_class:
                unique_objects_by_class[class_label] = set()
            unique_objects_by_class[class_label].add(tracking_id)
    
    print("\nUnique objects by class:")
    for class_label, tracking_ids in unique_objects_by_class.items():
        print(f"  {class_label}: {len(tracking_ids)} unique objects")
    
    # Save tracker state
    tracker.save_state()
    print("\nTracker state saved.")

if __name__ == "__main__":
    main()
