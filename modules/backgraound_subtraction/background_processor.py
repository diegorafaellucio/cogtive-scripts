"""
Core background subtraction processor module.
Implements clean architecture with separation of concerns.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, List, Dict, Any
import logging
from pathlib import Path

try:
    from .config import BackgroundSubtractionConfig
except ImportError:
    from config import BackgroundSubtractionConfig


class BackgroundProcessor:
    """Core processor for background subtraction operations."""
    
    def __init__(self, config: BackgroundSubtractionConfig):
        self.config = config
        self.bg_subtractor = config.get_background_subtractor()
        self.logger = logging.getLogger(__name__)
        
        # Load static background image if provided
        self.static_background = None
        if config.video.background_image_path:
            self.static_background = self._load_background_image()
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'processing_time': 0.0,
            'avg_fps': 0.0,
            'contours_detected': 0
        }
        
        # Create morphological kernel
        if config.processing.use_morphology:
            self.morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                config.processing.kernel_size
            )
        else:
            self.morph_kernel = None
    
    def _load_background_image(self) -> Optional[np.ndarray]:
        """Load static background image."""
        try:
            bg_image = cv2.imread(self.config.video.background_image_path)
            if bg_image is None:
                self.logger.error(f"Failed to load background image: {self.config.video.background_image_path}")
                return None
            
            self.logger.info(f"✅ Loaded static background: {self.config.video.background_image_path}")
            return bg_image
        except Exception as e:
            self.logger.error(f"Error loading background image: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before background subtraction."""
        processed = frame.copy()
        
        # H.264-specific preprocessing to reduce compression artifacts
        # Apply bilateral filter to reduce compression noise while preserving edges
        processed = cv2.bilateralFilter(processed, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Resize if needed
        if self.config.video.resize_factor != 1.0:
            h, w = processed.shape[:2]
            new_size = (
                int(w * self.config.video.resize_factor),
                int(h * self.config.video.resize_factor)
            )
            processed = cv2.resize(processed, new_size)
        
        # Apply stronger Gaussian blur for H.264 artifacts (after resize to be more efficient)
        if self.config.processing.use_gaussian_blur:
            # Use larger kernel for H.264 noise suppression
            kernel_size = max(7, self.config.processing.blur_kernel_size)
            processed = cv2.GaussianBlur(
                processed,
                (kernel_size,) * 2,
                0
            )
        
        return processed
    
    def apply_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """Apply background subtraction to the frame."""
        if self.static_background is not None:
            # Use static background subtraction
            return self._apply_static_background_subtraction(frame)
        else:
            # Use adaptive background subtraction
            foreground_mask = self.bg_subtractor.apply(
                frame, 
                learningRate=self.config.processing.learning_rate
            )
            return foreground_mask
    
    def _apply_static_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhanced background subtraction using static background image."""
        # Resize static background to match frame size if needed
        h, w = frame.shape[:2]
        bg_h, bg_w = self.static_background.shape[:2]
        
        if (bg_h, bg_w) != (h, w):
            background = cv2.resize(self.static_background, (w, h))
        else:
            background = self.static_background
        
        # Simple absolute difference in BGR space
        diff = cv2.absdiff(frame, background)
        
        # Convert to grayscale
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        
        # Balanced threshold for H.264: detect objects while reducing compression noise
        threshold_value = 35  # Balanced compromise: reduced from 70 to detect more objects
        _, foreground_mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Better morphology to clean up noise while preserving real objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel to preserve objects
        # Remove small noise but preserve real objects
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Fill gaps in objects
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        return foreground_mask
    
    def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Enhanced post-process the foreground mask for better detection."""
        processed_mask = mask.copy()
        
        if self.config.processing.use_morphology and self.morph_kernel is not None:
            # First, apply opening to remove small noise
            processed_mask = cv2.morphologyEx(
                processed_mask, 
                cv2.MORPH_OPEN, 
                self.morph_kernel,
                iterations=1  # Reduced iterations to preserve details
            )
            
            # Then, apply closing to fill small holes
            processed_mask = cv2.morphologyEx(
                processed_mask,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=2  # More iterations for better gap filling
            )
            
            # Additional dilation to make sure we capture full objects
            processed_mask = cv2.dilate(processed_mask, self.morph_kernel, iterations=1)
        
        # Apply median filter to reduce noise while preserving edges
        processed_mask = cv2.medianBlur(processed_mask, 5)
        
        return processed_mask
    
    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find and filter contours in the foreground mask."""
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if (self.config.processing.min_contour_area <= area <= 
                self.config.processing.max_contour_area):
                filtered_contours.append(contour)
        
        self.stats['contours_detected'] += len(filtered_contours)
        return filtered_contours
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and return all results."""
        start_time = time.time()
        
        # Preprocess
        preprocessed = self.preprocess_frame(frame)
        
        # Apply background subtraction
        foreground_mask = self.apply_background_subtraction(preprocessed)
        
        # Post-process mask
        processed_mask = self.post_process_mask(foreground_mask)
        
        # Find contours
        contours = self.find_contours(processed_mask)
        
        # Get background model (if available)
        try:
            if self.static_background is not None:
                # Use static background for display
                h, w = preprocessed.shape[:2]
                bg_h, bg_w = self.static_background.shape[:2]
                if (bg_h, bg_w) != (h, w):
                    background_image = cv2.resize(self.static_background, (w, h))
                else:
                    background_image = self.static_background
            else:
                # Use adaptive background from algorithm
                background_image = self.bg_subtractor.getBackgroundImage()
        except:
            background_image = None
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['frames_processed'] += 1
        self.stats['processing_time'] += processing_time
        self.stats['avg_fps'] = self.stats['frames_processed'] / self.stats['processing_time']
        
        return {
            'original': frame,
            'preprocessed': preprocessed,
            'foreground_mask': foreground_mask,
            'processed_mask': processed_mask,
            'contours': contours,
            'background': background_image,
            'processing_time': processing_time,
            'frame_number': self.stats['frames_processed']
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'frames_processed': 0,
            'processing_time': 0.0,
            'avg_fps': 0.0,
            'contours_detected': 0
        }


class VideoProcessor:
    """High-level video processing orchestrator."""
    
    def __init__(self, config: BackgroundSubtractionConfig):
        self.config = config
        self.processor = BackgroundProcessor(config)
        self.logger = logging.getLogger(__name__)
        
        # Video capture and writer
        self.cap = None
        self.writer = None
        
    def setup_video_capture(self) -> bool:
        """Setup video capture."""
        self.cap = cv2.VideoCapture(self.config.video.input_path)
        
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open video: {self.config.video.input_path}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video info: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Set start frame if specified
        if self.config.video.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.video.start_frame)
        
        return True
    
    def setup_video_writer(self) -> bool:
        """Setup video writer if output path is specified."""
        if not self.config.video.output_path:
            return True
        
        # Adjust dimensions if resize factor is used
        output_width = int(self.width * self.config.video.resize_factor)
        output_height = int(self.height * self.config.video.resize_factor)
        
        # Use original FPS or limited FPS
        output_fps = self.config.video.fps_limit or self.fps
        
        fourcc = cv2.VideoWriter_fourcc(*self.config.video.output_format)
        self.writer = cv2.VideoWriter(
            self.config.video.output_path,
            fourcc,
            output_fps,
            (output_width, output_height)
        )
        
        if not self.writer.isOpened():
            self.logger.error(f"Failed to open video writer: {self.config.video.output_path}")
            return False
        
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
