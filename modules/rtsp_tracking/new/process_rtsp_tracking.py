#!/usr/bin/env python3
"""
RTSP Stream Processing Script for Tracking

This script processes frames from an RTSP stream and sends them to the tracking endpoint.
"""

import os
import json
import requests
import uuid
import hashlib
from datetime import datetime, timedelta
import time
import argparse
import cv2
import numpy as np
import imutils
from draw_utils import draw_template_data, draw_tracking_data, draw_region_counts, draw_tracking_trails


def process_rtsp_for_tracking(rtsp_url, tracking_url, template_path, delay=0.1, show_frames=True, save_video=False,
                              output_video_path=None):
    """
    Process frames from an RTSP stream and send them to the tracking endpoint.

    Args:
        rtsp_url (str): URL of the RTSP stream
        tracking_url (str): URL of the tracking endpoint
        template_path (str): Path to custom template JSON file
        delay (float, optional): Delay between requests in seconds
        show_frames (bool, optional): Whether to display frames with tracking information
        save_video (bool, optional): Whether to save processed frames as video
        output_video_path (str, optional): Path for output video file
    """
    # Initialize video capture from RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 FPS if unable to determine

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Connected to RTSP stream: {rtsp_url}")
    print(f"Stream FPS: {fps}")
    print(f"Resolution: {width}x{height}")

    # Initialize video writer if saving video
    video_writer = None
    if save_video:
        if output_video_path is None:
            # Generate default output path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             f"tracking_output_{timestamp}.mp4")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_video_path}")
            save_video = False
        else:
            print(f"Video output will be saved to: {output_video_path}")

    # Load template from file (required)
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template = json.load(f)
    else:
        print(f"Error: Template file not found or not specified at {template_path}")
        return

    # Process frames
    start_time = datetime.now()
    frame_count = 0
    results = []
    
    # Initialize tracking history for drawing trails
    tracking_history = {}  # Dictionary: tracking_id -> list of (center_x, center_y) positions
    max_trail_length = 20  # Maximum number of positions to keep in history

    try:
        while True:
            # Read frame from stream
            ret, frame = cap.read()

            # frame = imutils.resize(frame, width=640)
            if not ret:
                print("End of video stream reached or failed to read frame.")
                # For video files, simply break when we reach the end
                # For RTSP streams, you could add reconnection logic here if needed
                break

            # Generate frame ID and timestamp
            # Ensure start_time is a datetime object before adding timedelta
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            frame_time = start_time + timedelta(seconds=frame_count / fps)
            frame_timestamp = frame_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            # Create UUID based on timestamp and frame_count for unique frame_id per iteration
            unique_string = f"{frame_timestamp}_{frame_count}_{time.time()}"
            frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

            process_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            frame_count += 1

            # Update template with new frame information
            template["frame_id"] = frame_id
            template["frameId"] = frame_id  # Also update frameId field
            template["frameTimestamp"] = frame_timestamp

            if "tracker_response" in template:
                template["tracker_response"]["frameId"] = frame_id
                template["tracker_response"]["frameTimestamp"] = frame_timestamp
                template["tracker_response"]["processTimestamp"] = process_timestamp

            # Prepare the request
            template_json = json.dumps(template)

            # Save frame to temporary file
            temp_image_path = "/tmp/rtsp_frame.jpg"
            cv2.imwrite(temp_image_path, frame)

            try:
                with open(temp_image_path, 'rb') as img_file:
                    files = {
                        'image': ('rtsp_frame.jpg', img_file, 'image/jpeg'),
                        'processing_template': ('template.json', template_json, 'application/json')
                    }

                    # Send the request
                    response = requests.post(tracking_url, files=files)

                    # Process the response
                    if response.status_code == 200:
                        result = response.json()
                        results.append(result)
                        
                        # Update tracking history with current object positions
                        tracked_objects = result.get('trackedObjects', [])
                        
                        # Get current tracking IDs for cleanup
                        current_tracking_ids = set()
                        
                        for obj in tracked_objects:
                            tracking_id = obj.get('tracking_id', 'N/A')
                            box = obj.get('box', [])
                            
                            if tracking_id != 'N/A' and len(box) == 4:
                                current_tracking_ids.add(tracking_id)
                                
                                # Check if box format is [x1, y1, x2, y2] or [x, y, w, h]
                                x1, y1, x2, y2 = box
                                
                                # If the values seem to be width/height format (x2, y2 are small),
                                # treat as [x, y, w, h]
                                if x2 < x1 or y2 < y1 or (x2 < 100 and y2 < 100):
                                    # Format is [x, y, w, h]
                                    x, y, w, h = box
                                    center_x = x + w // 2
                                    center_y = y + h // 2
                                else:
                                    # Format is [x1, y1, x2, y2]
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                
                                # Validate coordinates are reasonable
                                if center_x < 0 or center_y < 0 or center_x > width or center_y > height:
                                    continue  # Skip invalid coordinates
                                
                                # Initialize tracking history for new objects
                                if tracking_id not in tracking_history:
                                    tracking_history[tracking_id] = []
                                
                                # Add current position to history
                                tracking_history[tracking_id].append((center_x, center_y))
                                
                                # Limit history length to prevent memory issues
                                if len(tracking_history[tracking_id]) > max_trail_length:
                                    tracking_history[tracking_id] = tracking_history[tracking_id][-max_trail_length:]

                        # Cleanup inactive tracking history
                        # Remove tracking histories for objects no longer detected
                        inactive_ids = set(tracking_history.keys()) - current_tracking_ids
                        for inactive_id in inactive_ids:
                            del tracking_history[inactive_id]

                        # Save the JSON response to a file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        json_filename = f"tracking_response_{timestamp}.json"
                        json_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_responses",
                                                     json_filename)

                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)

                        # Write JSON to file
                        with open(json_filepath, 'w') as json_file:
                            json.dump(result, json_file, indent=4)

                        # Process frame for display and/or video output
                        if show_frames or save_video:
                            # Use the draw_utils to draw template data on the frame
                            display_frame = frame.copy()
                            
                            # Validate frame before processing
                            if display_frame is None or display_frame.size == 0:
                                print(f"Warning: Invalid frame at frame {frame_count}, skipping display")
                                continue

                            # Also draw tracking data if available
                            display_frame = draw_tracking_data(display_frame, result)
                            
                            # Validate after each drawing operation
                            if display_frame is None or display_frame.size == 0:
                                print(f"Warning: Frame corrupted after draw_tracking_data at frame {frame_count}")
                                display_frame = frame.copy()

                            # Draw region counts
                            display_frame = draw_region_counts(display_frame, template, result)
                            
                            # Validate after region counts
                            if display_frame is None or display_frame.size == 0:
                                print(f"Warning: Frame corrupted after draw_region_counts at frame {frame_count}")
                                display_frame = frame.copy()

                            display_frame = draw_template_data(display_frame, template)
                            
                            # Validate after template data
                            if display_frame is None or display_frame.size == 0:
                                print(f"Warning: Frame corrupted after draw_template_data at frame {frame_count}")
                                display_frame = frame.copy()

                            # Get list of active tracking IDs from current frame
                            active_tracking_ids = [obj.get('tracking_id', 'N/A') for obj in tracked_objects 
                                                if obj.get('tracking_id', 'N/A') != 'N/A']
                            
                            # Draw tracking trails
                            display_frame = draw_tracking_trails(display_frame, tracking_history, active_tracking_ids, max_trail_length)
                            
                            # Final validation before display/save
                            if display_frame is None or display_frame.size == 0:
                                print(f"Warning: Frame corrupted after draw_tracking_trails at frame {frame_count}")
                                display_frame = frame.copy()

                            # Save frame to video if enabled
                            if save_video and video_writer is not None:
                                video_writer.write(display_frame)

                            # Display the frame if enabled - with final validation
                            if show_frames:
                                # Double-check frame validity before imshow
                                if display_frame is not None and display_frame.size > 0 and len(display_frame.shape) >= 2:
                                    if display_frame.shape[0] > 0 and display_frame.shape[1] > 0:
                                        cv2.imshow('RTSP Tracking', display_frame)
                                    else:
                                        print(f"Warning: Invalid frame dimensions {display_frame.shape} at frame {frame_count}")
                                else:
                                    print(f"Warning: Invalid frame data at frame {frame_count}")

                                # Check for exit key
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    else:
                        print(f"Error processing frame {frame_count}: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")

            # Add delay between requests
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\nProcessing stopped by user")

    finally:
        # Release resources
        cap.release()
        if show_frames:
            cv2.destroyAllWindows()

        # Release video writer
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_video_path}")

        # Save final results
        if results:
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../examples/rtsp_tracking_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nProcessing complete. Results saved to {output_file}")
            print(f"Total frames processed: {len(results)}")

            # Print summary of tracked objects
            last_result = results[-1]
            tracker_response = last_result.get('tracker_response', {})
            detections = tracker_response.get('detections', [])

            print("\nTracking Summary:")
            for detection in detections:
                print(
                    f"- {detection['classLabel']}: Tracked from {detection.get('startTimestamp')} to {detection.get('endTimestamp')}")
                for workstation in detection.get('workstations', []):
                    for region in workstation.get('regionsOfInterests', []):
                        print(f"  - {workstation['workstationName']}/{region['regionName']}:")
                        for mode in region.get('modes', []):
                            mode_key = mode.get('key', 'unknown')
                            mode_value = mode.get('Value', 0)
                            print(f"    - {mode_key}: {mode_value}")


def main():
    parser = argparse.ArgumentParser(description="Process RTSP stream for tracking")
    # parser.add_argument("--rtsp", default="rtsp://34.194.31.98:8554/live/liveStream_UQEM3803255DY_0_0", help="RTSP stream URL")
    parser.add_argument("--rtsp", default="/home/diego/2TB/videos/cogtive/betterbeef/gravacao_2025-06-24_14-21-51.mp4",
                        help="RTSP stream URL")
    parser.add_argument("--url", default="http://127.0.0.1:8000/tracker/track",
                        help="URL of the tracking endpoint")
    parser.add_argument("--template", help="Path to template JSON file",
                        default="/home/diego/Projects/COGTIVE/aivision-core/data/json/new_sample_request_better_beef_workstation_from_rodrigo_updated.json")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between requests in seconds")
    parser.add_argument("--no-display", action="store_true", default=False,
                        help="Disable frame display")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save processed frames as video output")
    parser.add_argument("--output-video", type=str, default=None,
                        help="Path for output video file (default: auto-generated with timestamp)")

    args = parser.parse_args()

    process_rtsp_for_tracking(args.rtsp, args.url, args.template, args.delay,
                              not args.no_display, args.save_video, args.output_video)


if __name__ == "__main__":
    main()