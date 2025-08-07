"""
Background Subtraction Module

A comprehensive background subtraction toolkit for video processing.
Implements clean architecture with modular components for easy extension.

Main Components:
- BackgroundSubtractionConfig: Configuration management
- VideoProcessor: High-level video processing orchestrator  
- BackgroundProcessor: Core background subtraction algorithms
- BackgroundVisualizationRenderer: Visualization and display
- StatisticsReporter: Results reporting and analysis

Supported Algorithms:
- MOG2: Gaussian Mixture Model
- KNN: K-Nearest Neighbors
- GMG: Gaussian Mixture-based (requires opencv-contrib-python)

Example Usage:
    from config import BackgroundSubtractionConfig
    from background_processor import VideoProcessor
    from visualizer import BackgroundVisualizationRenderer
    
    config = BackgroundSubtractionConfig(
        input_video="input.mp4",
        algorithm="MOG2",
        output_video="output.mp4"
    )
    
    with VideoProcessor(config) as processor:
        # Setup and process video
        pass

Author: Cascade AI
Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Cascade AI"
__all__ = [
    "BackgroundSubtractionConfig",
    "VideoProcessor", 
    "BackgroundProcessor",
    "BackgroundVisualizationRenderer",
    "StatisticsReporter"
]

# Import main classes for easy access
from .config import BackgroundSubtractionConfig
from .background_processor import VideoProcessor, BackgroundProcessor  
from .visualizer import BackgroundVisualizationRenderer, StatisticsReporter
