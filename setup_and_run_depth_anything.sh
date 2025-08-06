#!/bin/bash
# Script to set up Depth-Anything-V2 and process a video

# Set up variables
VIDEO_PATH="/home/diego/Downloads/Canaa_norte_prensa_segmento_000.mp4"
OUTPUT_PATH="output_depth_anything.mp4"
MODEL_SIZE="vits"  # Small model

# Create a directory for Depth-Anything-V2
mkdir -p depth_anything_repo
cd depth_anything_repo

# Clone the repository if it doesn't exist
if [ ! -d "Depth-Anything-V2" ]; then
    echo "Cloning Depth-Anything-V2 repository..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    cd Depth-Anything-V2
    pip3.11 install -r requirements.txt
else
    cd Depth-Anything-V2
fi

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download the model if it doesn't exist
if [ ! -f "checkpoints/depth_anything_v2_${MODEL_SIZE}.pth" ]; then
    echo "Downloading Depth-Anything-V2-${MODEL_SIZE} model..."
    
    # Download links from the repository
    if [ "$MODEL_SIZE" = "vits" ]; then
        # Small model
        wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -O checkpoints/depth_anything_v2_vits.pth
    elif [ "$MODEL_SIZE" = "vitb" ]; then
        # Base model
        wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -O checkpoints/depth_anything_v2_vitb.pth
    elif [ "$MODEL_SIZE" = "vitl" ]; then
        # Large model
        wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -O checkpoints/depth_anything_v2_vitl.pth
    elif [ "$MODEL_SIZE" = "vitg" ]; then
        # Giant model
        wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth -O checkpoints/depth_anything_v2_vitg.pth
    fi
fi

# Copy our processing script to the repository directory
cp ../../process_video_depth_anything.py .

# Run the video processing script
echo "Processing video with Depth-Anything-V2-${MODEL_SIZE}..."
python3.11 process_video_depth_anything.py --video "$VIDEO_PATH" --output "$OUTPUT_PATH" --encoder "$MODEL_SIZE" --side-by-side

# Copy the output back to the main directory
cp "$OUTPUT_PATH" ../../

echo "Processing complete. Output saved to $OUTPUT_PATH"
