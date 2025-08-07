# Background Subtraction Module

A comprehensive, clean-architecture background subtraction toolkit for video processing using OpenCV algorithms.

## 🏗️ Architecture Overview

The module follows **clean architecture principles** with clear separation of concerns:

```
background_subtraction/
├── config.py              # Configuration management
├── background_processor.py # Core processing logic
├── visualizer.py          # Visualization and rendering
├── main.py                # Command-line interface
├── run_background_subtraction.py  # Simple runner for specific video
├── requirements.txt       # Dependencies
├── __init__.py           # Module initialization
└── README.md             # Documentation
```

### Components

1. **Configuration Layer** (`config.py`)
   - Centralized configuration management
   - Type-safe dataclasses for all parameters
   - Algorithm-specific settings

2. **Processing Layer** (`background_processor.py`)
   - `BackgroundProcessor`: Core algorithm implementation
   - `VideoProcessor`: High-level video orchestration
   - Clean separation of preprocessing, processing, and post-processing

3. **Visualization Layer** (`visualizer.py`)
   - `BackgroundVisualizationRenderer`: Display and rendering logic
   - `StatisticsReporter`: Results analysis and reporting
   - Multi-view displays and overlays

4. **Interface Layer** (`main.py`)
   - Command-line interface with comprehensive options
   - Argument parsing and validation
   - Orchestrates all components

## 🚀 Quick Start

### Simple Usage (for specific video)

```bash
cd /home/diego/Projects/COGTIVE/scripts/modules/backgraound_subtraction
python run_background_subtraction.py
```

This will process the video: `/home/diego/2TB/videos/cogtive/betterbeef/gravacao_2025-06-24_14-21-51.mp4`

### Advanced Usage (command-line interface)

```bash
python main.py /path/to/video.mp4 --algorithm MOG2 --output output.mp4
```

## 📋 Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **MOG2** | Gaussian Mixture Model | General purpose, adaptive backgrounds |
| **KNN** | K-Nearest Neighbors | Complex backgrounds, multiple objects |
| **GMG** | Gaussian Mixture-based | Static cameras, requires opencv-contrib |

## ⚙️ Configuration Options

### Background Subtractor Parameters
- `--history`: Background history frames (default: 500)
- `--var-threshold`: Variance threshold for MOG2 (default: 16.0)
- `--learning-rate`: Learning rate (-1 for automatic)

### Processing Parameters
- `--min-area`: Minimum contour area (default: 100)
- `--max-area`: Maximum contour area (default: 50000)
- `--resize`: Resize factor for processing (default: 1.0)

### Visualization Options
- `--no-display`: Disable real-time display
- `--save-frames`: Save individual frames
- `--window-size WIDTH HEIGHT`: Display window size

## 🎮 Interactive Controls

During processing:
- **'q' or ESC**: Quit processing
- **'p'**: Pause/unpause
- **'s'**: Save current frame manually

## 📊 Output Features

### Real-time Display
- Multi-view layout (original, foreground mask, background model)
- Live statistics overlay (FPS, frame count, object count)
- Color-coded contour detection

### Statistics Reporting
- Processing performance metrics
- Object detection counts
- Algorithm configuration summary
- Export to file option

## 🔧 Technical Features

### Clean Architecture Benefits
- **Modularity**: Each component has single responsibility
- **Testability**: Components can be tested independently
- **Extensibility**: Easy to add new algorithms or features
- **Maintainability**: Clear code organization and documentation

### Performance Optimizations
- Configurable image resizing for real-time processing
- Morphological operations for noise reduction
- Contour filtering by area
- Efficient memory management

### Error Handling
- Input validation and file existence checks
- Graceful error recovery
- Comprehensive logging system
- Resource cleanup with context managers

## 📖 Code Examples

### Basic Processing
```python
from config import BackgroundSubtractionConfig
from background_processor import VideoProcessor
from visualizer import BackgroundVisualizationRenderer

# Create configuration
config = BackgroundSubtractionConfig(
    input_video="input.mp4",
    algorithm="MOG2"
)

# Process video
with VideoProcessor(config) as processor:
    processor.setup_video_capture()
    renderer = BackgroundVisualizationRenderer(config)
    
    while True:
        ret, frame = processor.cap.read()
        if not ret:
            break
            
        results = processor.processor.process_frame(frame)
        if not renderer.display_frame(results):
            break
```

### Custom Configuration
```python
config = BackgroundSubtractionConfig("video.mp4", "KNN")

# Customize parameters
config.bg_subtractor.history = 300
config.processing.min_contour_area = 500
config.visualization.window_size = (1200, 800)
```

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GMG algorithm support (optional)
pip install opencv-contrib-python
```

## 📋 Requirements

- Python >= 3.7
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- pathlib2 >= 2.3.0

## 🎯 Use Cases

- **Security Surveillance**: Motion detection in security cameras
- **Traffic Monitoring**: Vehicle counting and tracking
- **Sports Analysis**: Player and ball tracking
- **Quality Control**: Object detection in manufacturing
- **Research**: Computer vision algorithm development

## 🚦 Performance Guidelines

### For Real-time Processing
- Use `--resize 0.5` for faster processing
- Set `--min-area 200` to filter small noise
- Use MOG2 algorithm for best speed/accuracy balance

### For High Accuracy
- Use full resolution (`--resize 1.0`)
- Lower `--var-threshold` for sensitive detection
- Use KNN algorithm for complex backgrounds

## 🔍 Troubleshooting

### Common Issues
1. **Video not found**: Check file path and permissions
2. **Slow processing**: Try reducing resize factor
3. **Too many false positives**: Increase variance threshold
4. **Missing objects**: Decrease variance threshold or minimum area

### Debug Options
- Use `--log-level DEBUG` for detailed logging
- Enable `--save-frames` to inspect individual frames
- Use `--stats-output stats.txt` to analyze performance

## 🤝 Architecture Principles Applied

1. **Single Responsibility**: Each class has one clear purpose
2. **Open/Closed**: Easy to extend with new algorithms
3. **Dependency Inversion**: Components depend on abstractions
4. **Interface Segregation**: Clean, focused interfaces
5. **Separation of Concerns**: UI, logic, and data are separate

This module demonstrates professional software development practices suitable for production environments while maintaining simplicity for research and experimentation.
