"""
Configuration module for background subtraction.
Centralizes all configuration parameters for clean architecture.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import cv2


@dataclass
class BackgroundSubtractorConfig:
    """Configuration for background subtractor algorithms."""
    
    # MOG2 Parameters
    history: int = 500
    var_threshold: float = 16.0
    detect_shadows: bool = True
    
    # KNN Parameters
    knn_history: int = 500
    knn_dist2_threshold: float = 400.0
    knn_detect_shadows: bool = True
    
    # GMG Parameters (if available)
    gmg_init_frames: int = 120
    gmg_decision_threshold: float = 0.8


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    input_path: str
    output_path: Optional[str] = None
    background_image_path: Optional[str] = None  # Static background image
    output_format: str = 'mp4v'
    resize_factor: float = 1.0
    fps_limit: Optional[int] = None
    start_frame: int = 0
    end_frame: Optional[int] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    
    # Morphological operations
    use_morphology: bool = True
    kernel_size: Tuple[int, int] = (5, 5)
    opening_iterations: int = 2
    closing_iterations: int = 2
    
    # Noise reduction
    blur_kernel_size: int = 5
    use_gaussian_blur: bool = True
    
    # Contour filtering
    min_contour_area: int = 100
    max_contour_area: int = 50000
    
    # Learning rate
    learning_rate: float = -1  # -1 for automatic


@dataclass
class VisualizationConfig:
    """Configuration for visualization and output."""
    
    show_original: bool = True
    show_foreground: bool = True
    show_background: bool = True
    show_contours: bool = True
    save_frames: bool = False
    frames_output_dir: str = "frames_output"
    
    # Display parameters
    window_size: Tuple[int, int] = (800, 600)
    fps_display: bool = True
    frame_counter_display: bool = True


class BackgroundSubtractionConfig:
    """Main configuration class combining all sub-configurations."""
    
    def __init__(
        self,
        input_video: str,
        algorithm: str = "MOG2",
        output_video: Optional[str] = None
    ):
        self.algorithm = algorithm.upper()
        
        # Sub-configurations
        self.bg_subtractor = BackgroundSubtractorConfig()
        self.video = VideoConfig(input_path=input_video, output_path=output_video)
        self.processing = ProcessingConfig()
        self.visualization = VisualizationConfig()
        
        # Validate algorithm
        if self.algorithm not in ["MOG2", "KNN", "GMG"]:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use MOG2, KNN, or GMG")
    
    def get_background_subtractor(self):
        """Create and return the configured background subtractor."""
        if self.algorithm == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self.bg_subtractor.history,
                varThreshold=self.bg_subtractor.var_threshold,
                detectShadows=self.bg_subtractor.detect_shadows
            )
        elif self.algorithm == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                history=self.bg_subtractor.knn_history,
                dist2Threshold=self.bg_subtractor.knn_dist2_threshold,
                detectShadows=self.bg_subtractor.knn_detect_shadows
            )
        elif self.algorithm == "GMG":
            # GMG is not available in all OpenCV versions
            try:
                bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
                    initializationFrames=self.bg_subtractor.gmg_init_frames,
                    decisionThreshold=self.bg_subtractor.gmg_decision_threshold
                )
                return bg_subtractor
            except AttributeError:
                raise ImportError("GMG algorithm requires opencv-contrib-python")
        
        raise ValueError(f"Unknown algorithm: {self.algorithm}")
