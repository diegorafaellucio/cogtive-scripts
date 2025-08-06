#!/bin/bash
# Script to start saving frames from RTSP streams

# Change to the script directory
cd "$(dirname "$0")"

# Check if OpenCV is installed
if ! python3.11 -c "import cv2" &> /dev/null; then
    echo "OpenCV is not installed. Installing..."
    pip3 install opencv-python
fi

# Run the Python script with default parameters
python3.11 save_rtsp_frames.py --duration 30 --output-dir "$(pwd)"

echo "Frame saving complete!"
