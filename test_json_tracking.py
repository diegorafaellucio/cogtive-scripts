#!/usr/bin/env python3
import json
import sys
import os
from modules.tracking.process_json_tracking import JSONTracker

def main():
    """
    Test the JSON tracking functionality with a sample JSON file
    """
    # Check if JSON file path is provided as command line argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # Use the sample JSON data from the project
        json_file_path = "sample_tracking_data.json"
        
        # If the sample file doesn't exist, create it with the provided JSON data
        if not os.path.exists(json_file_path):
            sample_data = {
                'status': 'success',
                'clientId': 'betterbeef',
                'frameId': '696e01cb-6600-4673-ba9a-91a5fbfba385',
                'frameTimestamp': '2025-05-15T05:34:59.907Z',
                'processTimestamp': '2025-05-15T08:36:39.941907Z',
                'workstations': [
                    {
                        'workstationName': 'WORKSTATION2',
                        'regionsOfInterest': [
                            {
                                'regionName': 'AREA1'
                            }
                        ]
                    }
                ],
                'objectsInFrame': [
                    {
                        'classLabel': 'MEAT',
                        'confidence': 0.9254270792007446,
                        'workstationName': 'WORKSTATION2',
                        'regionName': 'AREA1',
                        'boundingBox': {
                            'x': 940,
                            'y': 536,
                            'width': 243,
                            'height': 175
                        },
                        'modelName': 'BETTERBEEF',
                        'trackingId': '78'
                    },
                    # ... more objects ...
                ],
                'detections': [
                    {
                        'classLabel': 'MEAT',
                        'startTimestamp': '2025-05-15T05:34:59.907Z',
                        'endTimestamp': '2025-05-15T05:34:59.907Z',
                        'counterSince': '2025-05-15T05:34:59.907Z',
                        'trackingIds': [
                            '78'
                        ],
                        'workstations': [
                            {
                                'workstationName': 'WORKSTATION2',
                                'regionsOfInterests': [
                                    {
                                        'regionName': 'AREA1',
                                        'counter': 0,
                                        'instantCounter': 1
                                    }
                                ]
                            }
                        ]
                    },
                    # ... more detections ...
                ]
            }
            
            # Save sample data to file
            with open(json_file_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            print(f"Created sample JSON file: {json_file_path}")
        else:
            print(f"Sample file {json_file_path} already exists, using it for testing.")
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    print(f"Processing JSON data with {len(json_data.get('objectsInFrame', []))} objects...")
    
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
                class_id = obj['class_id']
                class_name = tracker.reverse_class_mapping.get(class_id, "UNKNOWN")
                time_in_roi = tracker.tracker.roi_tracker_counts.get(track_id, 0)
                print(f"  ID: {track_id}, Class: {class_name}, Time in ROI: {time_in_roi} frames")
    
    # Save tracker state
    tracker.save_state()
    print("\nTracker state saved.")

if __name__ == "__main__":
    main()
