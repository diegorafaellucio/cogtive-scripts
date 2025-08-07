"""
Visualization module for background subtraction results.
Handles all display and output rendering functionality.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

try:
    from .config import BackgroundSubtractionConfig
except ImportError:
    from config import BackgroundSubtractionConfig


class BackgroundVisualizationRenderer:
    """Handles visualization and rendering of background subtraction results."""
    
    def __init__(self, config: BackgroundSubtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Color definitions
        self.colors = {
            'contour': (0, 255, 0),      # Green
            'bounding_box': (255, 0, 0),  # Blue
            'text': (255, 255, 255),     # White
            'text_bg': (0, 0, 0),        # Black
            'fps_text': (0, 255, 255),   # Yellow
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        
        # Setup output directory for frames if needed
        if config.visualization.save_frames:
            self.frames_dir = Path(config.visualization.frames_output_dir)
            self.frames_dir.mkdir(exist_ok=True)
            self.logger.info(f"Frames will be saved to: {self.frames_dir}")
    
    def draw_contours_on_frame(self, frame: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
        """Draw contours and bounding boxes on frame."""
        result = frame.copy()
        
        for i, contour in enumerate(contours):
            # Draw contour
            cv2.drawContours(result, [contour], -1, self.colors['contour'], 2)
            
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), self.colors['bounding_box'], 2)
            
            # Add contour area text
            area = cv2.contourArea(contour)
            text = f"A:{int(area)}"
            cv2.putText(result, text, (x, y - 10), self.font, 0.5, self.colors['text'], 1)
    
        return result
    
    def add_info_overlay(self, frame: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Add information overlay to frame."""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Prepare info text
        texts = []
        
        if self.config.visualization.frame_counter_display:
            texts.append(f"Frame: {info.get('frame_number', 0)}")
        
        if self.config.visualization.fps_display:
            processing_time = info.get('processing_time', 0)
            fps = 1.0 / processing_time if processing_time > 0 else 0
            texts.append(f"FPS: {fps:.1f}")
        
        contours = info.get('contours', [])
        texts.append(f"Objects: {len(contours)}")
        
        # Draw background for text
        if texts:
            text_height = 30
            bg_height = len(texts) * text_height + 10
            cv2.rectangle(result, (10, 10), (250, bg_height), self.colors['text_bg'], -1)
            
            # Draw texts
            for i, text in enumerate(texts):
                y_pos = 35 + i * text_height
                cv2.putText(result, text, (15, y_pos), self.font, 
                           self.font_scale, self.colors['fps_text'], self.thickness)
        
        return result
    
    def create_multi_view_display(self, results: Dict[str, Any]) -> np.ndarray:
        """Create multi-view display with original, mask, and background."""
        original = results['original']
        foreground_mask = results['processed_mask']
        background = results.get('background')
        contours = results['contours']
        
        # Resize images to consistent size
        display_height = 400
        h, w = original.shape[:2]
        aspect_ratio = w / h
        display_width = int(display_height * aspect_ratio)
        
        # Resize original
        original_resized = cv2.resize(original, (display_width, display_height))
        
        # Create colored foreground mask
        foreground_colored = cv2.applyColorMap(foreground_mask, cv2.COLORMAP_HOT)
        foreground_resized = cv2.resize(foreground_colored, (display_width, display_height))
        
        # Draw contours on original
        original_with_contours = self.draw_contours_on_frame(original_resized, 
                                                           self._resize_contours(contours, original.shape[:2], 
                                                                                (display_width, display_height)))
        
        # Add info overlay
        original_with_info = self.add_info_overlay(original_with_contours, results)
        
        views_to_show = []
        labels = []
        
        if self.config.visualization.show_original:
            views_to_show.append(original_with_info)
            labels.append("Original + Detections")
        
        if self.config.visualization.show_foreground:
            views_to_show.append(foreground_resized)
            labels.append("Foreground Mask")
        
        if self.config.visualization.show_background and background is not None:
            background_resized = cv2.resize(background, (display_width, display_height))
            views_to_show.append(background_resized)
            labels.append("Background Model")
        
        if not views_to_show:
            return original_with_info
        
        # Create labels for each view
        labeled_views = []
        for view, label in zip(views_to_show, labels):
            labeled_view = view.copy()
            cv2.putText(labeled_view, label, (10, 25), self.font, 0.8, 
                       self.colors['fps_text'], 2)
            labeled_views.append(labeled_view)
        
        # Arrange views
        if len(labeled_views) == 1:
            return labeled_views[0]
        elif len(labeled_views) == 2:
            return np.hstack(labeled_views)
        elif len(labeled_views) == 3:
            top_row = np.hstack(labeled_views[:2])
            bottom_view = labeled_views[2]
            # Resize bottom view to match top row width
            bottom_resized = cv2.resize(bottom_view, (top_row.shape[1], display_height))
            return np.vstack([top_row, bottom_resized])
        else:
            # Arrange in 2x2 grid
            top_row = np.hstack(labeled_views[:2])
            bottom_row = np.hstack(labeled_views[2:4])
            return np.vstack([top_row, bottom_row])
    
    def _resize_contours(self, contours: List[np.ndarray], 
                        original_shape: Tuple[int, int], 
                        target_shape: Tuple[int, int]) -> List[np.ndarray]:
        """Resize contours to match target shape."""
        if not contours:
            return contours
        
        orig_h, orig_w = original_shape
        target_w, target_h = target_shape
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        resized_contours = []
        for contour in contours:
            resized_contour = contour.copy().astype(np.float32)
            resized_contour[:, 0, 0] *= scale_x
            resized_contour[:, 0, 1] *= scale_y
            resized_contours.append(resized_contour.astype(np.int32))
        
        return resized_contours
    
    def save_frame(self, frame: np.ndarray, frame_number: int, suffix: str = ""):
        """Save frame to disk if configured."""
        if not self.config.visualization.save_frames:
            return
        
        filename = f"frame_{frame_number:06d}{suffix}.png"
        filepath = self.frames_dir / filename
        cv2.imwrite(str(filepath), frame)
    
    def display_frame(self, results: Dict[str, Any]) -> bool:
        """Display frame and handle user input. Returns False if user wants to quit."""
        display_frame = self.create_multi_view_display(results)
        
        # Resize display if needed
        target_width, target_height = self.config.visualization.window_size
        if display_frame.shape[1] > target_width or display_frame.shape[0] > target_height:
            # Calculate scale to fit within target size
            scale_w = target_width / display_frame.shape[1]
            scale_h = target_height / display_frame.shape[0]
            scale = min(scale_w, scale_h)
            
            new_width = int(display_frame.shape[1] * scale)
            new_height = int(display_frame.shape[0] * scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
        
        cv2.imshow('Background Subtraction', display_frame)
        
        # Save frame if configured
        if self.config.visualization.save_frames:
            self.save_frame(display_frame, results['frame_number'])
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('s'):  # Save current frame
            filename = f"manual_save_frame_{results['frame_number']}.png"
            cv2.imwrite(filename, display_frame)
            self.logger.info(f"Frame saved as {filename}")
        elif key == ord('p'):  # Pause/unpause
            self.logger.info("Paused. Press any key to continue...")
            cv2.waitKey(0)
        
        return True


class StatisticsReporter:
    """Handles statistics reporting and summary generation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def print_summary(self, stats: Dict[str, Any], config: BackgroundSubtractionConfig):
        """Print processing summary."""
        print("\n" + "="*60)
        print("BACKGROUND SUBTRACTION PROCESSING SUMMARY")
        print("="*60)
        print(f"Input Video: {config.video.input_path}")
        print(f"Algorithm: {config.algorithm}")
        print(f"Total Frames Processed: {stats['frames_processed']}")
        print(f"Total Processing Time: {stats['processing_time']:.2f}s")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Total Objects Detected: {stats['contours_detected']}")
        
        if config.video.output_path:
            print(f"Output Video: {config.video.output_path}")
        
        if config.visualization.save_frames:
            print(f"Frames Saved: {config.visualization.frames_output_dir}")
        
        print("="*60)
    
    def save_statistics(self, stats: Dict[str, Any], config: BackgroundSubtractionConfig, 
                       output_path: str):
        """Save statistics to file."""
        try:
            with open(output_path, 'w') as f:
                f.write("Background Subtraction Processing Statistics\n")
                f.write("="*50 + "\n")
                f.write(f"Input Video: {config.video.input_path}\n")
                f.write(f"Algorithm: {config.algorithm}\n")
                f.write(f"Total Frames: {stats['frames_processed']}\n")
                f.write(f"Processing Time: {stats['processing_time']:.2f}s\n")
                f.write(f"Average FPS: {stats['avg_fps']:.2f}\n")
                f.write(f"Objects Detected: {stats['contours_detected']}\n")
                
                # Configuration details
                f.write("\nConfiguration Details:\n")
                f.write("-"*30 + "\n")
                f.write(f"Background Subtractor History: {config.bg_subtractor.history}\n")
                f.write(f"Variance Threshold: {config.bg_subtractor.var_threshold}\n")
                f.write(f"Detect Shadows: {config.bg_subtractor.detect_shadows}\n")
                f.write(f"Min Contour Area: {config.processing.min_contour_area}\n")
                f.write(f"Max Contour Area: {config.processing.max_contour_area}\n")
                f.write(f"Learning Rate: {config.processing.learning_rate}\n")
                
            self.logger.info(f"Statistics saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
