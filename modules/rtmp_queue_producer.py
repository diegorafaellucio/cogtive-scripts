#!/usr/bin/env python3
"""
Video Publisher to MediaMTX

This script reads frames from a local video file and publishes them to a MediaMTX server
using RTMP protocol with PyAV (Python bindings for FFmpeg).
"""

import cv2
import time
import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import threading
import queue
import av
from fractions import Fraction

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VideoPublisher")

class RTMPPublisher:
    """Class to handle RTMP publishing using PyAV"""
    
    def __init__(self, rtmp_url, width, height, fps=30):
        """
        Initialize the RTMP publisher
        
        Args:
            rtmp_url (str): RTMP URL to publish to
            width (int): Video width
            height (int): Video height
            fps (float): Frames per second
        """
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=100)  # Queue for frames
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the RTMP publishing thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("RTMP publisher already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._publish_thread)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started RTMP publisher to {self.rtmp_url}")
    
    def stop(self):
        """Stop the RTMP publishing thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("Stopped RTMP publisher")
    
    def add_frame(self, frame):
        """
        Add a frame to the publishing queue
        
        Args:
            frame (numpy.ndarray): BGR image frame
        
        Returns:
            bool: True if frame was added, False if queue is full
        """
        try:
            self.frame_queue.put(frame, block=False)
            return True
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
            return False
    
    def _publish_thread(self):
        """Thread function for publishing frames"""
        try:
            # Create output container for RTMP
            output = av.open(self.rtmp_url, mode='w', format='flv')
            
            # Convert fps to Fraction for PyAV
            fps_fraction = Fraction(int(self.fps * 1000), 1000)
            
            # Add video stream
            stream = output.add_stream('h264', rate=fps_fraction)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = 'yuv420p'
            
            # Set codec options for low latency
            stream.codec_context.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile': 'baseline',
                'crf': '23'
            }
            
            frame_count = 0
            
            while self.running:
                try:
                    # Get frame from queue with timeout
                    bgr_frame = self.frame_queue.get(timeout=1.0)
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    
                    # Create PyAV frame
                    frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
                    
                    # Encode and send the frame
                    for packet in stream.encode(frame):
                        output.mux(packet)
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.info(f"Published {frame_count} frames")
                    
                except queue.Empty:
                    # No frames in queue
                    pass
            
            # Flush the encoder
            for packet in stream.encode(None):
                output.mux(packet)
            
            # Close the output
            output.close()
            
        except Exception as e:
            logger.error(f"Error in RTMP publishing thread: {str(e)}")
            self.running = False

def get_video_properties(video_path):
    """Get video properties using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }

def publish_video_with_pyav(video_path, rtmp_url, loop=True):
    """
    Publish video to RTMP server using PyAV
    
    Args:
        video_path (str): Path to the video file
        rtmp_url (str): RTMP URL to publish to
        loop (bool): Whether to loop the video (defaults to True)
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Get video properties
    props = get_video_properties(video_path)
    if not props:
        return False
    
    width = props['width']
    height = props['height']
    fps = props['fps']
    
    # Initialize RTMP publisher
    try:
        publisher = RTMPPublisher(rtmp_url, width, height, fps)
        publisher.start()
    except Exception as e:
        logger.error(f"Failed to initialize RTMP publisher: {str(e)}")
        return False
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file: {video_path}")
        publisher.stop()
        return False
    
    logger.info(f"Started publishing video to {rtmp_url}")
    logger.info(f"Press Ctrl+C to stop publishing")
    
    try:
        frame_time = 1.0 / fps
        last_time = time.time()
        
        # Continue running until manually stopped
        while True:
            # Read frame
            ret, frame = cap.read()
            
            # If end of video, reset to beginning
            if not ret:
                logger.info("End of video reached, looping back to beginning")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Add frame to publisher queue
            publisher.add_frame(frame)
            
            # Maintain frame rate
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_time = time.time()
    
    except KeyboardInterrupt:
        logger.info("Publishing stopped by user")
        return True
    
    except Exception as e:
        logger.error(f"Error during publishing: {str(e)}")
        return False
    
    finally:
        # Clean up
        publisher.stop()
        cap.release()
        logger.info("Video publisher closed")

def main():
    """Main function to parse arguments and start publishing"""
    parser = argparse.ArgumentParser(description="Video Publisher to MediaMTX")
    parser.add_argument("--video", 
                        default="/home/diego/Videos/Cogtive/betterbeef2.mp4",
                        help="Path to the video file")
    parser.add_argument("--rtmp", 
                        default="rtmp://34.194.31.98:1935/live/simulacao_better_beef",
                        help="RTMP URL to publish to")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Loop the video when it reaches the end (default: True)")
    
    args = parser.parse_args()
    
    # Verify video file exists
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    # Get video properties
    props = get_video_properties(args.video)
    if props:
        logger.info(f"Video: {Path(args.video).name}")
        logger.info(f"Resolution: {props['width']}x{props['height']}")
        logger.info(f"FPS: {props['fps']}")
        logger.info(f"Total frames: {props['total_frames']}")
    
    logger.info(f"Publishing to: {args.rtmp}")
    logger.info(f"Loop: {args.loop}")
    
    # Start publishing
    publish_video_with_pyav(args.video, args.rtmp, args.loop)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
