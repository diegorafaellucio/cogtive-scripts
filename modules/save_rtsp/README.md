# RTSP Frame Saver

This tool saves frames from multiple RTSP streams for a specified duration. Each stream's frames are saved in a separate directory with sequential numbering.

## Features

- Captures frames from multiple RTSP streams simultaneously
- Saves frames with sequential numbering (000000000.jpg format)
- Uses native camera FPS for each stream
- Automatic reconnection if a stream disconnects
- Multithreaded processing for efficient capture

## Default RTSP Streams

The script is pre-configured with the following RTSP streams:

```
rtsp://34.194.31.98:8554/live/liveStream_1LEM2300245A8_0_0
rtsp://34.194.31.98:8554/live/liveStream_1LEM230178461_0_0
rtsp://34.194.31.98:8554/live/liveStream_1LEM2300940D3_0_0
rtsp://34.194.31.98:8554/live/liveStream_1LEM23003994O_0_0
rtsp://34.194.31.98:8554/live/liveStream_1LEM2300001VN_0_0
rtsp://34.194.31.98:8554/live/liveStream_1LEM2301662AO_0_0
rtsp://34.194.31.98:8554/live/liveStream_1LEM23000159Q_0_0
```

## Usage

### Quick Start

Run the included shell script to start saving frames with default settings:

```bash
./start_saving.sh
```

This will:
1. Check if OpenCV is installed and install it if needed
2. Run the script with default settings (30 minutes duration, using native camera FPS)

### Manual Execution

You can also run the Python script directly with custom parameters:

```bash
python3.11 save_rtsp_frames.py --duration 30 --output-dir "/path/to/output"
```

### Command Line Arguments

- `--duration`: Duration to capture in minutes (default: 30)
- `--output-dir`: Base output directory (default: script directory)
- `--streams`: RTSP stream URLs (space-separated)
- `--streams-file`: File containing RTSP stream URLs (one per line)

## Output

Frames are saved in directories named after each stream, with sequential numbering:

```
/modules/save_rtsp/liveStream_1LEM2300245A8_0_0/000000000.jpg
/modules/save_rtsp/liveStream_1LEM2300245A8_0_0/000000001.jpg
...
```

## Requirements

- Python 3.11
- OpenCV (opencv-python)
