#!/usr/bin/env python3
"""
Script to save 30 minutes of frames from multiple RTSP streams.
Each stream's frames are saved in a separate directory with sequential numbering.
Uses the native FPS from each camera.
"""

import cv2
import os
import time
import threading
from datetime import datetime
import argparse
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('rtsp_saver')

class RTSPFrameSaver:
    def __init__(self, rtsp_url, output_dir=None, duration_minutes=30):
        """
        Initialize the RTSP frame saver.
        
        Args:
            rtsp_url: RTSP stream URL
            output_dir: Directory to save frames (if None, will use stream name)
            duration_minutes: How long to capture frames (in minutes)
        """
        self.rtsp_url = rtsp_url
        self.duration_minutes = duration_minutes
        
        # Parse the stream name from the URL
        parsed_url = urlparse(rtsp_url)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) > 1:
            self.stream_name = path_parts[-1]
        else:
            self.stream_name = f"stream_{int(time.time())}"
            
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.stream_name)
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized saver for {self.stream_name}")
        logger.info(f"Saving frames to {self.output_dir}")
        logger.info(f"Will capture for {self.duration_minutes} minutes at native camera FPS")

    def save_frames(self):
        """Capture and save frames from the RTSP stream using the camera's native FPS."""
        cap = cv2.VideoCapture(self.rtsp_url)
        
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
            return False
        
        # Get the camera's FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning(f"Could not determine FPS for {self.stream_name}, using default of 30 FPS")
            fps = 30
        
        logger.info(f"Successfully connected to stream: {self.stream_name} with {fps} FPS")
        
        # Calculate total frames to capture
        total_seconds = self.duration_minutes * 60
        frame_count = 0
        start_time = time.time()
        end_time = start_time + total_seconds
        
        while time.time() < end_time:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from {self.stream_name}, retrying...")
                # Try to reconnect
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    logger.error(f"Failed to reconnect to {self.stream_name}, exiting")
                    break
                continue
            
            # Save the frame with the specified naming pattern
            frame_filename = os.path.join(self.output_dir, f"{frame_count:09d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                logger.info(f"{self.stream_name}: Saved {frame_count} frames, " 
                            f"elapsed: {elapsed:.1f}s, remaining: {end_time - current_time:.1f}s")
        
        cap.release()
        logger.info(f"Completed saving {frame_count} frames for {self.stream_name}")
        return True


def process_stream(rtsp_url, base_output_dir, duration_minutes):
    """Process a single RTSP stream in a separate thread."""
    try:
        # Parse the stream name from the URL to use as directory name
        parsed_url = urlparse(rtsp_url)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) > 1:
            stream_name = path_parts[-1]
        else:
            stream_name = f"stream_{int(time.time())}"
            
        output_dir = os.path.join(base_output_dir, stream_name)
        
        saver = RTSPFrameSaver(
            rtsp_url=rtsp_url,
            output_dir=output_dir,
            duration_minutes=duration_minutes
        )
        saver.save_frames()
    except Exception as e:
        logger.error(f"Error processing stream {rtsp_url}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Save frames from RTSP streams')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration to capture in minutes (default: 30)')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base output directory')
    parser.add_argument('--streams-file', type=str,
                        help='File containing RTSP stream URLs (one per line)')
    parser.add_argument('--streams', type=str, nargs='+',
                        help='RTSP stream URLs')
    
    args = parser.parse_args()
    
    # Collect stream URLs
    rtsp_urls = []
    
    # From command line arguments
    if args.streams:
        rtsp_urls.extend(args.streams)
    
    # From file
    if args.streams_file and os.path.exists(args.streams_file):
        with open(args.streams_file, 'r') as f:
            file_urls = [line.strip() for line in f if line.strip() and line.strip().startswith('rtsp://')]
            rtsp_urls.extend(file_urls)
    
    # Default streams if none provided
    if not rtsp_urls:
        rtsp_urls = [
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM2300245A8_0_0",
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM230178461_0_0",
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM2300940D3_0_0",
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM23003994O_0_0",
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM2300001VN_0_0",
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM2301662AO_0_0",
            "rtsp://34.194.31.98:8554/live/liveStream_1LEM23000159Q_0_0"
        ]
    
    logger.info(f"Starting to process {len(rtsp_urls)} RTSP streams")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Duration: {args.duration} minutes")
    logger.info(f"Using native camera FPS for each stream")
    
    # Create threads for each stream
    threads = []
    for url in rtsp_urls:
        thread = threading.Thread(
            target=process_stream,
            args=(url, args.output_dir, args.duration)
        )
        thread.daemon = True
        threads.append(thread)
        thread.start()
        logger.info(f"Started thread for {url}")
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    logger.info("All streams processed successfully")


if __name__ == "__main__":
    main()
