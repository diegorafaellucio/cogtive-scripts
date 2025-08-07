#!/usr/bin/env python3.11
"""
Simple runner script for background subtraction on the specified video.
Configured for the specific input video provided by the user.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import BackgroundSubtractionConfig
from background_processor import VideoProcessor
from visualizer import BackgroundVisualizationRenderer, StatisticsReporter


def main():
    """Run background subtraction on the specified video."""
    
    # Input video path as specified by user
    input_video = "/home/diego/2TB/videos/cogtive/betterbeef/gravacao_2025-06-24_14-21-51.mp4"
    background_image = "/home/diego/Downloads/frame_1.jpg"
    
    # Verify input exists
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video not found: {input_video}")
        print("Please verify the video path exists.")
        return 1
    
    # Verify background image exists
    if not os.path.exists(background_image):
        print(f"❌ Error: Background image not found: {background_image}")
        print("Please verify the background image path exists.")
        return 1
    
    print("🎬 COGTIVE Background Subtraction Processing")
    print("=" * 50)
    print(f"📹 Input: {input_video}")
    print(f"🖼️  Background: {background_image}")
    print(f"🔧 Algorithm: Static Background Subtraction")
    print("=" * 50)
    
    try:
        # Create configuration
        config = BackgroundSubtractionConfig(
            input_video=input_video,
            algorithm="MOG2",  # Still use MOG2 as fallback algorithm
            output_video=None  # No output video, just display
        )
        
        # Set background image path
        config.video.background_image_path = background_image
        
        # Optimize parameters for balanced motion detection
        config.bg_subtractor.var_threshold = 25.0  # Balanced threshold to reduce noise
        config.bg_subtractor.detect_shadows = True
        
        # Balanced processing settings - reduce false positives
        config.processing.min_contour_area = 150   # Higher minimum to filter small noise
        config.processing.max_contour_area = 15000 # Reasonable maximum for objects
        config.processing.use_morphology = True
        config.processing.kernel_size = (3, 3)     # Standard kernel size
        config.processing.opening_iterations = 2   # More noise removal
        config.processing.closing_iterations = 2   # Better gap filling
        config.processing.use_gaussian_blur = True # Enable blur to reduce noise
        config.processing.blur_kernel_size = 5     # Stronger blur for noise reduction
        
        # Visualization settings
        config.visualization.show_original = True
        config.visualization.show_foreground = True
        config.visualization.show_background = True
        config.visualization.window_size = (1200, 800)
        config.visualization.fps_display = True
        config.visualization.frame_counter_display = True
        
        # Video processing settings
        config.video.resize_factor = 0.7  # Resize for better performance
        
        print("🚀 Starting background subtraction processing...")
        print("Controls:")
        print("  • Press 'q' or ESC to quit")
        print("  • Press 'p' to pause/unpause")
        print("  • Press 's' to save current frame")
        print("-" * 50)
        
        # Process video
        with VideoProcessor(config) as processor:
            if not processor.setup_video_capture():
                print("❌ Failed to setup video capture")
                return 1
            
            renderer = BackgroundVisualizationRenderer(config)
            stats_reporter = StatisticsReporter()
            
            frame_count = 0
            
            while True:
                ret, frame = processor.cap.read()
                
                if not ret:
                    print("✅ Reached end of video")
                    break
                
                frame_count += 1
                
                # Process frame
                results = processor.processor.process_frame(frame)
                
                # Display results
                if not renderer.display_frame(results):
                    print("🛑 Processing stopped by user")
                    break
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    stats = processor.processor.get_statistics()
                    print(f"📊 Processed {frame_count} frames | "
                          f"Avg FPS: {stats['avg_fps']:.2f} | "
                          f"Objects detected: {stats['contours_detected']}")
        
        # Print final statistics
        final_stats = processor.processor.get_statistics()
        stats_reporter.print_summary(final_stats, config)
        
        print("🎉 Background subtraction completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
