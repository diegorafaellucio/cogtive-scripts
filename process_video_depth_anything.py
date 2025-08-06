#!/usr/bin/env python3
"""
Video Processing Script with Depth-Anything-V2-Small

This script processes a video file using Depth-Anything-V2-Small model,
generates depth maps for each frame, and creates an output video.
"""

import cv2
import torch
import numpy as np
import argparse
import os
import time
import matplotlib
from pathlib import Path

# Import Depth Anything V2 model
from depth_anything_v2.dpt import DepthAnythingV2

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process video with Depth-Anything-V2-Small')
    parser.add_argument('--video', type=str, default='/home/diego/Downloads/Canaa_norte_prensa_segmento_000.mp4',
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_depth_anything.mp4',
                        help='Path to output video file')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input size for model inference')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Encoder model size (vits=Small, vitb=Base, vitl=Large, vitg=Giant)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show video while processing')
    parser.add_argument('--side-by-side', action='store_true', default=True,
                        help='Show original and depth side by side')
    parser.add_argument('--grayscale', action='store_true', default=False,
                        help='Use grayscale depth map instead of colored')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load the Depth-Anything-V2 model
    print(f"Loading Depth-Anything-V2-{args.encoder.upper()} model...")
    model = DepthAnythingV2(**model_configs[args.encoder])
    
    # Check if checkpoints directory exists
    checkpoint_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint from the Depth-Anything-V2 repository")
        print("and place it in the 'checkpoints' directory.")
        return
    
    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    # Open the video file
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer
    output_path = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Determine output dimensions
    if args.side_by_side:
        output_width = width * 2 + 50  # Add 50px margin between frames
        output_height = height
    else:
        output_width = width
        output_height = height
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Set up colormap for depth visualization
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    print("Starting to process frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run depth estimation
        depth = model.infer_image(frame, args.input_size)
        
        # Normalize depth for visualization
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Apply colormap or keep grayscale
        if args.grayscale:
            depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Create output frame
        if args.side_by_side:
            # Create a white margin between the original and depth frames
            margin = np.ones((height, 50, 3), dtype=np.uint8) * 255
            # Concatenate original frame, margin, and depth visualization
            output_frame = cv2.hconcat([frame, margin, depth_vis])
        else:
            output_frame = depth_vis
        
        # Write frame to output video
        out.write(output_frame)
        
        # Display progress
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - {fps_processing:.1f} FPS")
        
        # Show frame if requested
        if args.show:
            # Resize for display if the frame is too large
            display_frame = output_frame
            if output_width > 1280 or output_height > 720:
                scale = min(1280 / output_width, 720 / output_height)
                new_width = int(output_width * scale)
                new_height = int(output_height * scale)
                display_frame = cv2.resize(output_frame, (new_width, new_height))
            
            cv2.imshow('Depth Anything V2 Processing', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Average processing speed: {frame_count / (time.time() - start_time):.2f} FPS")

if __name__ == "__main__":
    main()
