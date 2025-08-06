#!/usr/bin/env python3
"""
RTSP Stream Reader

This script reads frames from an RTSP stream and displays them.
It can also save the stream to a video file if requested.
"""

import argparse
import cv2
import time
import os
from datetime import datetime

def read_rtsp_stream(rtsp_url, output_path=None, show_frames=True, delay=0.01, transport_protocol="tcp"):
    """
    Read frames from an RTSP stream and optionally save to a video file.
    
    Args:
        rtsp_url (str): URL of the RTSP stream
        output_path (str, optional): Path to save the video file
        show_frames (bool, optional): Whether to display frames
        delay (float, optional): Delay between reading frames in seconds
        transport_protocol (str, optional): RTSP transport protocol ('tcp', 'udp', 'http', 'https')
    """
    # Set RTSP transport protocol
    if transport_protocol == "tcp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    elif transport_protocol == "udp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    elif transport_protocol == "http":
        # Try HTTP tunneling
        if not rtsp_url.startswith("http"):
            http_url = rtsp_url.replace("rtsp://", "http://")
            cap = cv2.VideoCapture(http_url)
        else:
            cap = cv2.VideoCapture(rtsp_url)
    elif transport_protocol == "https":
        # Try HTTPS tunneling
        if not rtsp_url.startswith("https"):
            https_url = rtsp_url.replace("rtsp://", "https://")
            cap = cv2.VideoCapture(https_url)
        else:
            cap = cv2.VideoCapture(rtsp_url)
    else:
        # Default approach
        cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Error: Could not open stream at {rtsp_url} with transport protocol {transport_protocol}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 FPS if unable to determine
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Connected to stream: {rtsp_url}")
    print(f"Using transport protocol: {transport_protocol}")
    print(f"Stream properties - FPS: {fps}, Resolution: {width}x{height}")
    
    # Initialize video writer if output path is provided
    video_writer = None
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving video to: {output_path}")
    
    # Process frames
    start_time = datetime.now()
    frame_count = 0
    
    try:
        while True:
            # Read frame from stream
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from stream. Reconnecting...")
                # Try to reconnect
                cap.release()
                time.sleep(2)
                
                # Reconnect with the same transport protocol
                if transport_protocol == "tcp":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                elif transport_protocol == "udp":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(rtsp_url)
                    
                if not cap.isOpened():
                    print(f"Error: Could not reconnect to stream at {rtsp_url}")
                    break
                continue
            
            frame_count += 1
            
            # Add frame information as text
            elapsed_time = (datetime.now() - start_time).total_seconds()
            actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            frame_info = f"Frame: {frame_count} | FPS: {actual_fps:.2f}"
            cv2.putText(frame, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame to video file if enabled
            if video_writer:
                video_writer.write(frame)
            
            # Display the frame if show_frames is enabled
            if show_frames:
                cv2.imshow('Stream Viewer', frame)
                
                # Break the loop if 'q' is pressed
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("\nStream reading stopped by user")
                    break
            
            # Add delay between frames
            if delay > 0:
                time.sleep(delay)
            
            # Print status every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames | Current FPS: {actual_fps:.2f}")
    
    except KeyboardInterrupt:
        print("\nStream reading stopped by user")
    
    finally:
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        if show_frames:
            cv2.destroyAllWindows()
        
        # Print summary
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\nStream reading complete.")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count / elapsed_time if elapsed_time > 0 else 0:.2f}")
        if output_path and os.path.exists(output_path):
            print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Read RTSP stream")
    parser.add_argument("--rtsp", default="rtsp://34.194.31.98:8554/live/liveStream_UQEM3803255DY_0_0", 
                        help="RTSP stream URL")
    parser.add_argument("--output", default=None, 
                        help="Path to save the video file (optional)")
    parser.add_argument("--no-display", action="store_true", 
                        help="Disable frame display")
    parser.add_argument("--delay", type=float, default=0.01, 
                        help="Delay between reading frames in seconds")
    parser.add_argument("--transport", default="tcp", choices=["tcp", "udp", "http", "https"],
                        help="RTSP transport protocol (tcp, udp, http, https)")
    
    args = parser.parse_args()
    
    read_rtsp_stream(args.rtsp, args.output, not args.no_display, args.delay, args.transport)

if __name__ == "__main__":
    main()
