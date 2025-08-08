#!/usr/bin/env python3
"""
Drawing utilities for RTSP tracking visualization.

This module provides functions to draw tracking information on frames.
"""

import cv2
import numpy as np

def get_optimal_font_scale(frame, text, target_width_ratio=0.05):
    """
    Calculate optimal font scale based on image dimensions.
    
    Args:
        frame: The input frame
        text: Text to be displayed
        target_width_ratio: Desired text width as a ratio of frame width
        
    Returns:
        Optimal font scale for the given image dimensions
    """
    height, width = frame.shape[:2]
    target_width = width * target_width_ratio
    
    # Start with a small font scale
    font_scale = 0.1
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

    # Increase font scale until target width is reached
    if text_size[0] > 0:  # Avoid division by zero
        font_scale *= target_width / text_size[0]

    # Limit maximum and minimum font scale
    return max(0.3, min(2.0, font_scale))

def draw_tracking_data(frame, result):
    """
    Draw tracking data on the frame.

    Args:
        frame: The frame to draw on
        result: The tracking result data containing tracking information

    Returns:
        The frame with tracking information drawn on it
    """
    if not result:
        return frame

    # Make a copy of the frame to avoid modifying the original
    display_frame = frame.copy()

    # Draw regions of interest
    draw_regions_of_interest(display_frame, result)

    # Draw tracked objects
    draw_tracked_objects(display_frame, result.get('trackedObjects', []))

    # Add frame and client information
    # draw_frame_info(display_frame, result)

    return display_frame

def draw_regions_of_interest(frame, result):
    """Draw regions of interest on the frame."""
    if not result or 'tracker_response' not in result:
        return

    tracker_response = result.get('tracker_response', {})
    detections = tracker_response.get('detections', [])

    # Draw regions of interest
    for detection in detections:
        for workstation in detection.get('workstations', []):
            for roi in workstation.get('regionsOfInterests', []):
                # Check if region has polygon points
                if 'polygon' in roi:
                    points = np.array(roi['polygon'], dtype=np.int32)
                    if len(points) > 0:
                        # Calculate center of region
                        region_center_x = np.mean(points[:, 0, 0]) if points.ndim == 3 else np.mean(points[:, 0])
                        region_center_y = np.mean(points[:, 0, 1]) if points.ndim == 3 else np.mean(points[:, 1])

                        # Calculate font scale based on frame dimensions
                        region_text = f"{workstation['workstationName']}/{roi['regionName']}"
                        font_scale = get_optimal_font_scale(frame, region_text)

                        # Use same font scale and thickness as detection totals
                        font_scale = 0.5  # Same size as class labels in totals
                        font_scale = max(0.4, font_scale)  # Ensure minimum readable size
                        text_thickness = 1  # Ultra-thin font like totals - no bold appearance

                        # Add region name
                        # cv2.putText(frame, region_text,
                        #             (int(region_center_x), int(region_center_y) - int(30 * font_scale * 2)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                        #             text_thickness)

                        # Print all modes inside the region
                        # y_offset = 0
                        # for det in detections:
                        #     for ws in det.get('workstations', []):
                        #         if ws.get('workstationName') == workstation.get('workstationName'):
                        #             for region in ws.get('regionsOfInterests', []):
                        #                 if region.get('regionName') == roi.get('regionName'):
                        #                     for mode in region.get('modes', []):
                        #                         mode_key = mode.get('key', 'unknown')
                        #                         mode_value = mode.get('Value', 0)
                        #                         count_text = f"{det['classLabel']}: {mode_key}={mode_value}"
                        #
                        #                         # Calculate font scale for count text
                        #                         count_font_scale = get_optimal_font_scale(frame, count_text, 0.04)
                        #
                        #                         # Display count inside the region
                        #                         cv2.putText(frame, count_text,
                        #                                     (int(region_center_x), int(region_center_y) + y_offset),
                        #                                     cv2.FONT_HERSHEY_SIMPLEX, count_font_scale, (255, 255, 255),
                        #                                     max(1, int(count_font_scale * 2)))
                        #                         y_offset += int(20 * count_font_scale * 2)

def draw_tracked_objects(frame, tracked_objects):
    """Draw tracked objects on the frame."""
    for obj in tracked_objects:
        # Get bounding box
        box = obj.get('box', [])
        if len(box) == 4:
            x, y, w, h = box

            # Determine color based on class
            class_label = obj.get('label', '')

            # Use different colors for different classes
            if class_label.upper() == 'PERSON':
                color = (0, 0, 255)  # Red for persons
            else:
                color = (0, 255, 0)  # Green for other objects

            # Draw bounding box
            box_thickness = max(1, int(min(frame.shape[0], frame.shape[1]) / 500))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

            # Draw polygon if available
            polygon = obj.get('polygon', [])
            if polygon and len(polygon) >= 3:  # Need at least 3 points for a polygon
                try:
                    # Convert polygon points to numpy array
                    points = np.array(polygon, dtype=np.int32)
                    
                    # Draw polygon outline in purple
                    purple_color = (128, 0, 128)  # Purple color in BGR
                    cv2.polylines(frame, [points], True, purple_color, box_thickness)
                    
                    # Optionally draw filled polygon with transparency
                    # Create a copy for transparency effect
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points], purple_color)
                    # Blend with original frame (20% transparency)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    
                except Exception as e:
                    print(f"Error drawing polygon for object {obj.get('tracking_id', 'N/A')}: {e}")

            # Add object information
            confidence = obj.get('confidence', 0)
            tracking_id = obj.get('tracking_id', 'N/A')
            tracking_age = obj.get('age', 0)
            model_name = obj.get('model_name', 'N/A')

            # Format text with more detailed information
            text_lines = [
                f"{class_label} (Conf: {confidence:.2f})",
                f"ID: {tracking_id} | Age: {tracking_age}",
                f"Model: {model_name}"
            ]

            # Calculate font scale based on frame dimensions - reduced for smaller labels
            font_scale = get_optimal_font_scale(frame, text_lines[0], 0.05)  # Reduced from 0.08 to 0.05
            font_scale = max(0.4, font_scale)  # Reduced minimum from 0.6 to 0.4
            
            # Calculate text thickness for better visibility
            text_thickness = max(1, int(font_scale * 1.2))  # Reduced from 2 to 1.2 for thinner font
            
            # Calculate line height with increased spacing
            line_height = int(40 * font_scale)  # Increased from 30 to 40 for more spacing between lines
            
            # Start position for text - ensure it's above the bounding box
            text_start_y = max(line_height * len(text_lines), y - 10)

            # Draw text with background for better visibility
            for i, line in enumerate(text_lines):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                text_y = text_start_y - (i * line_height)
                
                # Draw black background rectangle for text
                bg_padding = 5
                cv2.rectangle(frame, 
                            (x - bg_padding, text_y - text_size[1] - bg_padding), 
                            (x + text_size[0] + bg_padding, text_y + bg_padding), 
                            (0, 0, 0), -1)
                
                # Draw white text on black background for maximum contrast
                cv2.putText(frame, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, (255, 255, 255), text_thickness)

def draw_frame_info(frame, result):
    """Draw frame and client information on the frame."""
    frame_id = result.get('frame_id', 'N/A')
    client_id = result.get('client_id', 'N/A')

    # Add timestamp if available
    timestamp = "N/A"
    if 'tracker_response' in result:
        timestamp = result['tracker_response'].get('frameTimestamp', 'N/A')

    # Draw frame information at the top of the frame
    info_text = f"Client: {client_id} | Frame: {frame_id} | Time: {timestamp}"

    # Calculate font scale based on frame width - reduced for smaller labels
    font_scale = get_optimal_font_scale(frame, info_text, 0.10)  # Reduced from 0.15 to 0.10
    font_scale = max(0.5, font_scale)  # Reduced minimum from 0.7 to 0.5
    text_thickness = max(1, int(font_scale * 1.2))  # Reduced from 2 to 1.2 for thinner font
    
    # Calculate text size and position
    text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
    text_x = 10
    text_y = int(40 * font_scale)
    
    # Draw black background rectangle for text
    bg_padding = 8
    cv2.rectangle(frame, 
                (text_x - bg_padding, text_y - text_size[1] - bg_padding), 
                (text_x + text_size[0] + bg_padding, text_y + bg_padding), 
                (0, 0, 0), -1)
    
    # Draw white text on black background for maximum contrast
    cv2.putText(frame, info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
              font_scale, (255, 255, 255), text_thickness)

def draw_template_data(frame, template):
    """
    Draw template data on the frame (workstations and regions of interest).

    Args:
        frame: The frame to draw on
        template: The template data containing workstation and ROI information

    Returns:
        The frame with template information drawn on it
    """
    if not template or 'workstations' not in template:
        return frame

    # Make a copy of the frame to avoid modifying the original
    display_frame = frame.copy()

    # Draw workstations and regions of interest from template
    for workstation in template.get('workstations', []):
        workstation_name = workstation.get('workstationName', 'Unknown')

        for roi in workstation.get('regionsOfInterest', []):
            region_name = roi.get('regionName', 'Unknown')

            # Draw polygon if coordinates are available
            if 'polygonCoordinates' in roi:
                points = np.array(roi['polygonCoordinates'], dtype=np.int32)
                if len(points) > 0:
                    # Calculate line thickness proportional to image size
                    line_thickness = max(1, int(min(frame.shape[0], frame.shape[1]) / 500))

                    # Draw polygon outline
                    cv2.polylines(display_frame, [points], True, (0, 255, 255), line_thickness)

                    # Fill polygon with semi-transparent color
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [points], (0, 255, 255))
                    cv2.addWeighted(display_frame, 0.9, overlay, 0.1, 0, display_frame)

                    # Calculate position closer to the top of the bounding box
                    min_y = int(np.min(points[:, 1]))  # Top of the region
                    region_center_x = int(np.mean(points[:, 0]))

                    # Add region and workstation name
                    label_text = f"{workstation_name}/{region_name}"

                    # Use same font scale and thickness as detection totals
                    font_scale = 0.5  # Same size as class labels in totals
                    font_scale = max(0.4, font_scale)  # Ensure minimum readable size
                    text_thickness = 1  # Ultra-thin font like totals - no bold appearance

                    # Position text with more spacing from the top of the region
                    text_y = min_y + int(50 * font_scale)  # Increased from 30 to 50 for more spacing from top

                    # Add background rectangle for text
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                    cv2.rectangle(display_frame,
                                (region_center_x - text_size[0]//2 - 5, text_y - int(25 * font_scale)),
                                (region_center_x + text_size[0]//2 + 5, text_y - int(5 * font_scale)),
                                (0, 0, 0), -1)

                    # Add text
                    cv2.putText(display_frame, label_text,
                              (region_center_x - text_size[0]//2, text_y - int(10 * font_scale)),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), text_thickness)

                    # Add detection rules information
                    # y_offset = int(20 * font_scale)
                    # for detection_rule in roi.get('detectionRules', []):
                    #     for model in detection_rule.get('models', []):
                    #         model_name = model.get('modelName', 'Unknown')
                    #         for class_label in model.get('classLabels', []):
                    #             label = class_label.get('label', 'Unknown')
                    #             modes = class_label.get('modes', [])

                    #             # Create text for class and modes
                    #             modes_text = ', '.join(modes) if modes else 'No modes'
                    #             rule_text = f"{label}: {modes_text}"

                    #             # Calculate font scale for rule text
                    #             rule_font_scale = get_optimal_font_scale(frame, rule_text, 0.05)  # Reduced from 0.08 to 0.05
                    #             rule_font_scale = max(0.3, rule_font_scale)  # Reduced minimum from 0.5 to 0.3
                    #             rule_text_thickness = max(1, int(rule_font_scale * 1.2))  # Reduced from 2 to 1.2 for thinner font
                                
                    #             # Add background for rule text
                    #             rule_text_size = cv2.getTextSize(rule_text, cv2.FONT_HERSHEY_SIMPLEX, rule_font_scale, rule_text_thickness)[0]
                    #             cv2.rectangle(display_frame,
                    #                         (region_center_x - rule_text_size[0]//2 - 3, text_y + y_offset - int(15 * rule_font_scale)),
                    #                         (region_center_x + rule_text_size[0]//2 + 3, text_y + y_offset + int(5 * rule_font_scale)),
                    #                         (0, 0, 0), -1)

                    #             # Add rule text
                    #             cv2.putText(display_frame, rule_text,
                    #                       (region_center_x - rule_text_size[0]//2, text_y + y_offset),
                    #                       cv2.FONT_HERSHEY_SIMPLEX, rule_font_scale, (255, 255, 255), rule_text_thickness)
                    #             y_offset += int(20 * rule_font_scale)

    return display_frame

def draw_region_counts(frame, template, result):
    """
    Draw region counting information by crossing template and response data.

    Args:
        frame: The frame to draw on
        template: The template data containing workstation and ROI information
        result: The tracking result data containing detection counts

    Returns:
        The frame with counting information drawn on it
    """
    if not result or 'tracker_response' not in result:
        return frame

    # Make a copy of the frame to avoid modifying the original
    display_frame = frame.copy()

    tracker_response = result.get('tracker_response', {})
    detections = tracker_response.get('detections', [])

    # Draw counting information for each region from template
    for workstation in template.get('workstations', []):
        workstation_name = workstation.get('workstationName', 'Unknown')

        for roi in workstation.get('regionsOfInterest', []):
            region_name = roi.get('regionName', 'Unknown')

            # Find corresponding polygon coordinates for positioning
            if 'polygonCoordinates' in roi:
                points = np.array(roi['polygonCoordinates'], dtype=np.int32)
                if len(points) > 0:
                    # Calculate region boundaries
                    min_x = int(np.min(points[:, 0]))
                    max_x = int(np.max(points[:, 0]))
                    min_y = int(np.min(points[:, 1]))
                    max_y = int(np.max(points[:, 1]))
                    region_center_x = int(np.mean(points[:, 0]))
                    
                    # Position totals above the bottom line of the workstation
                    # Calculate how many lines we need for vertical stacking
                    num_classes = len(detections)
                    line_height = 25
                    total_height = num_classes * line_height
                    totals_start_y = max_y - total_height - 10  # 10px margin from bottom
                    
                    # Group all detection data for this region first
                    region_data = {}
                    for detection in detections:
                        class_label = detection.get('classLabel', 'Unknown')
                        for ws in detection.get('workstations', []):
                            if ws.get('workstationName') == workstation_name:
                                for region in ws.get('regionsOfInterests', []):
                                    if region.get('regionName') == region_name:
                                        if class_label not in region_data:
                                            region_data[class_label] = []
                                        for mode in region.get('modes', []):
                                            region_data[class_label].append(mode)
                    
                    # Draw classes vertically stacked
                    current_y = totals_start_y
                    
                    for class_label, modes in region_data.items():
                        # Build complete text for this class
                        class_counts = []
                        for mode in modes:
                            mode_key = mode.get('key', 'unknown')
                            mode_value = mode.get('Value', 0)
                            
                            # Abbreviate common mode names
                            if mode_key.lower() in ['counting', 'instantaneous']:
                                mode_key = 'ins'
                            elif mode_key.lower() in ['detection', 'accumulated']:
                                mode_key = 'accu'
                            
                            class_counts.append(f"{mode_key}:{mode_value}")
                        
                        if class_counts:
                            full_text = f"{class_label}: {' | '.join(class_counts)}"
                            
                            font_scale = 0.45
                            text_thickness = 1
                            
                            # Calculate text size for centering
                            text_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                            text_x = region_center_x - (text_size[0] // 2)  # Center horizontally
                            
                            # Determine colors based on content
                            if "PERSON" in full_text:
                                text_color = (100, 150, 255)  # Light blue for PERSON
                                bg_color = (20, 30, 50)       # Dark blue background
                            elif "MEAT" in full_text:
                                text_color = (150, 255, 150)  # Light green for MEAT
                                bg_color = (20, 50, 20)       # Dark green background
                            else:
                                text_color = (200, 200, 200)  # Light gray for others
                                bg_color = (40, 40, 40)       # Dark gray background
                            
                            # Draw background rectangle
                            padding = 6
                            cv2.rectangle(display_frame,
                                        (text_x - padding, current_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, current_y + padding),
                                        bg_color, -1)
                            
                            # Draw text
                            cv2.putText(display_frame, full_text,
                                      (text_x, current_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
                            
                            # Move to next line
                            current_y += line_height
    
    return display_frame

def draw_tracking_trails(frame, tracking_history, active_tracking_ids=None, max_trail_length=20):
    """
    Draw tracking trails for all objects based on their position history.
    
    Args:
        frame: The frame to draw on
        tracking_history: Dictionary with tracking_id as key and list of (center_x, center_y) as values
        active_tracking_ids: List of currently active tracking IDs (if None, all trails use their default colors)
        max_trail_length: Maximum number of points in the trail
    """
    if frame is None or tracking_history is None:
        return frame
    
    # **EMERGENCY FALLBACK: Disable trails temporarily for debugging**
    # Uncomment the next line to completely disable trail drawing for testing
    # return frame
    
    # Get frame dimensions for boundary checking
    frame_height, frame_width = frame.shape[:2]
    
    # **SIMPLIFIED APPROACH: Only draw for active objects**
    if active_tracking_ids is None:
        return frame  # Don't draw any trails if no active IDs
    
    for tracking_id in active_tracking_ids:
        if tracking_id not in tracking_history:
            continue
            
        positions = tracking_history[tracking_id]
        
        # Need at least 2 points to draw a line
        if len(positions) < 2:
            continue
        
        # **SIMPLE COLOR SCHEME**
        try:
            color_index = int(tracking_id) % 3 if tracking_id.isdigit() else hash(tracking_id) % 3
        except:
            color_index = 0
        
        # Use only 3 distinct colors to avoid confusion
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
        ]
        color = colors[color_index]
        
        # **STRICT COORDINATE VALIDATION**
        valid_positions = []
        for pos in positions[-min(len(positions), 10)]:  # Only use last 10 positions
            try:
                x, y = int(pos[0]), int(pos[1])
                # Very strict bounds checking
                if 10 <= x <= frame_width-10 and 10 <= y <= frame_height-10:
                    valid_positions.append((x, y))
            except (ValueError, TypeError, IndexError):
                continue
        
        if len(valid_positions) < 2:
            continue
        
        # **SIMPLE LINE DRAWING - NO COMPLEX EFFECTS**
        try:
            # Draw only the last few segments to avoid clutter
            for i in range(1, min(len(valid_positions), 5)):  # Max 4 line segments
                pt1 = valid_positions[i-1]
                pt2 = valid_positions[i]
                
                # Calculate simple distance check
                dist = ((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)**0.5
                
                # Skip if distance is too large (teleportation) or too small
                if dist > 100 or dist < 2:
                    continue
                
                # Draw simple line with fixed thickness
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw only the current position (most recent)
            if valid_positions:
                current_pos = valid_positions[-1]
                cv2.circle(frame, current_pos, 5, color, 2)
                cv2.circle(frame, current_pos, 3, (255, 255, 255), -1)
                
        except Exception as e:
            # If anything goes wrong, skip this trail entirely
            print(f"Warning: Skipping trail for ID {tracking_id}: {e}")
            continue
    
    return frame
