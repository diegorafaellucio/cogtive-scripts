#!/usr/bin/env python3
"""
Video Frame Extractor

This script extracts all frames from a video file and saves them as individual images.
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, file_format='jpg', quality=95):
    """
    Extract all frames from a video file and save them as individual images.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frames
        file_format (str): Format to save the images (jpg, png)
        quality (int): Image quality for jpg format (0-100)
    
    Returns:
        int: Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total Frames: {total_frames}")
    print(f"  - Duration: {total_frames/fps:.2f} seconds")
    
    # Set image format parameters
    if file_format.lower() == 'jpg' or file_format.lower() == 'jpeg':
        extension = '.jpg'
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif file_format.lower() == 'png':
        extension = '.png'
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, 10 - quality // 10))]
    else:
        print(f"Unsupported format: {file_format}. Using jpg instead.")
        extension = '.jpg'
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    
    # Extract frames with progress bar
    frame_count = 0
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Save the frame
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}{extension}")
            cv2.imwrite(frame_filename, frame, encode_params)
            
            frame_count += 1
            pbar.update(1)
    
    # Release the video capture object
    video.release()
    
    print(f"\nExtracted {frame_count} frames to {output_dir}")
    return frame_count

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("-o", "--output", default="./frames", help="Output directory for frames")
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "jpeg", "png"], 
                        help="Image format (jpg, png)")
    parser.add_argument("-q", "--quality", type=int, default=95, 
                        help="Image quality (0-100, higher is better)")
    
    args = parser.parse_args()
    
    # Get absolute paths
    video_path = os.path.abspath(args.video_path)
    output_dir = os.path.abspath(args.output)
    
    # Extract frames
    extract_frames(video_path, output_dir, args.format, args.quality)

if __name__ == "__main__":
    main()
