#!/usr/bin/env python3
"""
RTSP Stream Viewer

A simple OpenCV-based RTSP stream viewer that displays video from an RTSP source
with frame rate and resolution information.
"""

import cv2
import time
import sys
import os

# RTSP stream link - using the specified stream
# rtsp_url = "rtsp://34.194.31.98:8554/live/liveStream_UQEM3803255DY_0_0"
rtsp_url = "rtsp://34.194.31.98:8554/live/stream_1"

def try_connect_with_protocol(protocol):
    """Try to connect to the RTSP stream using the specified transport protocol"""
    print(f"Trying to connect with {protocol.upper()} transport protocol...")
    
    # Set transport protocol
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{protocol}"
    
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    
    # Set additional properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Buffer size
    
    # Check if the stream opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream with {protocol.upper()}.")
        return None
    
    # Try to read a frame to confirm connection
    ret, _ = cap.read()
    if not ret:
        print(f"Error: Could not read frame with {protocol.upper()}.")
        cap.release()
        return None
    
    # If we got here, connection is successful
    print(f"Successfully connected using {protocol.upper()} transport protocol.")
    return cap

def main():
    print(f"Attempting to connect to {rtsp_url}...")
    
    # Try different transport protocols in order of preference
    protocols = ["tcp", "udp", "http"]
    cap = None
    
    for protocol in protocols:
        cap = try_connect_with_protocol(protocol)
        if cap is not None:
            break
    
    if cap is None:
        print(f"Failed to connect to {rtsp_url} after trying all transport protocols.")
        return

    print(f"Successfully connected to {rtsp_url}... Press 'q' to exit.")

    # Create a window
    cv2.namedWindow("RTSP Stream Viewer", cv2.WINDOW_NORMAL)

    # Display FPS information
    fps_start_time = time.time()
    fps = 0
    frame_count = 0
    connection_lost_time = None
    timeout = 10  # seconds to try reconnecting before giving up

    try:
        while True:
            # Read a frame from the stream
            ret, frame = cap.read()

            if not ret:
                if connection_lost_time is None:
                    connection_lost_time = time.time()
                    print("Warning: Connection lost. Attempting to recover...")
                
                # Check if we've been trying to recover for too long
                if time.time() - connection_lost_time > timeout:
                    print(f"Error: Could not recover connection after {timeout} seconds.")
                    break
                
                # Wait a bit before retrying
                time.sleep(0.5)
                continue  # Skip this frame and retry
            
            # Reset connection lost timer if we successfully got a frame
            connection_lost_time = None
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame resolution
            height, width = frame.shape[:2]
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("RTSP Stream Viewer", frame)
            
            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStream viewing interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Stream closed.")

if __name__ == "__main__":
    main()
