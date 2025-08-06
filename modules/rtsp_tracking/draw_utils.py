#!/usr/bin/env python3
"""
Drawing utilities for RTSP tracking visualization.

This module provides functions to draw tracking information on frames.
"""

import cv2
import numpy as np


def get_optimal_font_scale(frame, text, target_width_ratio=0.08):
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
    return max(0.6, min(3.0, font_scale))


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
    # draw_regions_of_interest(display_frame, result)

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
                        # Draw polygon
                        cv2.polylines(frame, [points], True, (255, 0, 0), 2)

                        # Calculate center of region
                        region_center_x = np.mean(points[:, 0, 0]) if points.ndim == 3 else np.mean(points[:, 0])
                        region_center_y = np.mean(points[:, 0, 1]) if points.ndim == 3 else np.mean(points[:, 1])

                        # Calculate font scale based on frame dimensions
                        region_text = f"{workstation['workstationName']}/{roi['regionName']}"
                        font_scale = get_optimal_font_scale(frame, region_text)

                        # Add region name
                        cv2.putText(frame, region_text,
                                    (int(region_center_x), int(region_center_y) - int(30 * font_scale * 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                                    max(1, int(font_scale * 2)))

                        # Print all modes inside the region
                        y_offset = 0
                        for det in detections:
                            for ws in det.get('workstations', []):
                                if ws.get('workstationName') == workstation.get('workstationName'):
                                    for region in ws.get('regionsOfInterests', []):
                                        if region.get('regionName') == roi.get('regionName'):
                                            for mode in region.get('modes', []):
                                                mode_key = mode.get('key', 'unknown')
                                                mode_value = mode.get('Value', 0)
                                                count_text = f"{det['classLabel']}: {mode_key}={mode_value}"

                                                # Calculate font scale for count text
                                                count_font_scale = get_optimal_font_scale(frame, count_text, 0.04)

                                                # Display count inside the region
                                                cv2.putText(frame, count_text,
                                                            (int(region_center_x), int(region_center_y) + y_offset),
                                                            cv2.FONT_HERSHEY_SIMPLEX, count_font_scale, (255, 255, 255),
                                                            max(1, int(count_font_scale * 2)))
                                                y_offset += int(30 * count_font_scale * 2)


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

            # Draw estimated bounding box in purple if available
            # estimated_box = obj.get('estimated_box', {})
            # if estimated_box and all(k in estimated_box for k in ['x', 'y', 'width', 'height']):
            #     est_x = int(estimated_box['x'])
            #     est_y = int(estimated_box['y'])
            #     est_w = int(estimated_box['width'])
            #     est_h = int(estimated_box['height'])
            #
            #     # Draw estimated bounding box in purple
            #     purple_color = (128, 0, 128)  # Purple color in BGR
            #     cv2.rectangle(frame, (est_x, est_y), (est_x + est_w, est_y + est_h), purple_color, box_thickness)

            # Add object information
            tracking_id = obj.get('tracking_id', 'N/A')
            tracking_age = obj.get('age', 0)

            # Format text with only ID and age on a single line
            text_lines = [
                f"ID: {tracking_id}, Age: {tracking_age}"
            ]

            # Calculate font scale based on frame dimensions and object size
            font_scale = get_optimal_font_scale(frame, text_lines[0], 0.05)  # Increased ratio for better readability
            text_y = y - int(10 * font_scale)
            text_thickness = max(1, int(font_scale * 2))

            # Draw text with background for better visibility
            for line in text_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                # cv2.rectangle(frame, (x, text_y - int(20 * font_scale)), (x + text_size[0], text_y), (0, 0, 0), -1)
                cv2.putText(frame, line, (x, text_y - int(5 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, text_thickness)
                text_y -= int(30 * font_scale)


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

    # Calculate font scale based on frame width
    font_scale = get_optimal_font_scale(frame, info_text, 0.25)
    text_thickness = max(1, int(font_scale * 2))

    cv2.putText(frame, info_text, (10, int(30 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
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

                    # Calculate font scale based on frame dimensions
                    font_scale = get_optimal_font_scale(frame, label_text, 0.05)
                    text_thickness = max(1, int(font_scale * 2))

                    # Position text closer to the top of the region
                    text_y = min_y + int(30 * font_scale)  # Adjusted from the top edge

                    # Add background rectangle for text
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                    # cv2.rectangle(display_frame,
                    #               (region_center_x - text_size[0] // 2 - 5, text_y - int(25 * font_scale)),
                    #               (region_center_x + text_size[0] // 2 + 5, text_y - int(5 * font_scale)),
                    #               (0, 0, 0), -1)

                    # Add text
                    cv2.putText(display_frame, label_text,
                                (region_center_x - text_size[0] // 2, text_y - int(10 * font_scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), text_thickness)

                    # Add detection rules information
                    y_offset = int(30 * font_scale)
                    # for detection_rule in roi.get('detectionRules', []):
                    #     for model in detection_rule.get('models', []):
                    #         model_name = model.get('modelName', 'Unknown')
                    #         for class_label in model.get('classLabels', []):
                    #             label = class_label.get('label', 'Unknown')
                    #             modes = class_label.get('modes', [])
                    #
                    #             # Create text for class and modes
                    #             modes_text = ', '.join(modes) if modes else 'No modes'
                    #             rule_text = f"{label}: {modes_text}"
                    #
                    #             # Calculate font scale for rule text
                    #             rule_font_scale = get_optimal_font_scale(frame, rule_text, 0.04)
                    #             rule_text_thickness = max(1, int(rule_font_scale))
                    #
                    #             # Add background for rule text
                    #             rule_text_size = cv2.getTextSize(rule_text, cv2.FONT_HERSHEY_SIMPLEX, rule_font_scale,
                    #                                              rule_text_thickness)[0]
                    #             cv2.rectangle(display_frame,
                    #                           (region_center_x - rule_text_size[0] // 2 - 3,
                    #                            text_y + y_offset - int(15 * rule_font_scale)),
                    #                           (region_center_x + rule_text_size[0] // 2 + 3,
                    #                            text_y + y_offset + int(5 * rule_font_scale)),
                    #                           (0, 0, 0), -1)
                    #
                    #             # Add rule text
                    #             cv2.putText(display_frame, rule_text,
                    #                         (region_center_x - rule_text_size[0] // 2, text_y + y_offset),
                    #                         cv2.FONT_HERSHEY_SIMPLEX, rule_font_scale, (255, 255, 255),
                    #                         rule_text_thickness)
                    #             y_offset += int(30 * rule_font_scale)

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
                    # Calculate center and bottom of region for text placement
                    region_center_x = int(np.mean(points[:, 0]))
                    region_center_y = int(np.mean(points[:, 1]))
                    region_bottom_y = int(np.max(points[:, 1]))
                    region_top_y = int(np.min(points[:, 1]))
                    
                    # Position text lower in the region (closer to bottom)
                    # Calculate position that is 90% from top to bottom (instead of 80%)
                    region_height = region_bottom_y - region_top_y
                    text_y_position = region_top_y + int(region_height * 0.9)
                    
                    # Prepare compact display format
                    class_counts = {}
                    
                    # Collect all counts by class
                    for detection in detections:
                        class_label = detection.get('classLabel', 'Unknown')
                        
                        # Find workstation data in detection
                        for ws in detection.get('workstations', []):
                            if ws.get('workstationName') == workstation_name:
                                for region in ws.get('regionsOfInterests', []):
                                    if region.get('regionName') == region_name:
                                        # Collect mode counts for this class
                                        if class_label not in class_counts:
                                            class_counts[class_label] = {}
                                            
                                        for mode in region.get('modes', []):
                                            mode_key = mode.get('key', 'unknown')
                                            mode_value = mode.get('Value', 0)
                                            class_counts[class_label][mode_key] = mode_value
                    
                    # Display counts in compact format
                    if class_counts:
                        # Start position for text (inside the region)
                        y_position = text_y_position
                        
                        # Collect all class information first
                        class_info = []
                        total_width = 0
                        padding = 20  # Horizontal padding between classes
                        
                        for class_label, modes in class_counts.items():
                            instantaneous_value = modes.get('instantaneous', 0)
                            accumulated_value = modes.get('accumulated', 0)
                            
                            # Calculate font scale (smaller for compactness)
                            font_scale = get_optimal_font_scale(frame, class_label, 0.03)
                            text_thickness = max(1, int(font_scale))
                            
                            # Get text sizes
                            label_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      font_scale, text_thickness)[0]
                            inst_text = f"inst: {instantaneous_value}"
                            inst_size = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      font_scale, text_thickness)[0]
                            accum_text = f"accum: {accumulated_value}"
                            accum_size = cv2.getTextSize(accum_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      font_scale, text_thickness)[0]
                            
                            # Find width for this class block
                            class_width = max(label_size[0], inst_size[0], accum_size[0])
                            
                            # Store class information
                            class_info.append({
                                'label': class_label,
                                'label_size': label_size,
                                'inst_text': inst_text,
                                'inst_size': inst_size,
                                'inst_value': instantaneous_value,
                                'accum_text': accum_text,
                                'accum_size': accum_size,
                                'accum_value': accumulated_value,
                                'width': class_width,
                                'font_scale': font_scale,
                                'text_thickness': text_thickness
                            })
                            
                            total_width += class_width + padding
                        
                        # Calculate starting x position to center all classes
                        start_x = region_center_x - (total_width - padding) // 2
                        current_x = start_x
                        
                        # Calculate background rectangle dimensions
                        rect_height = int(20 * font_scale * 3)  # Height for 3 lines
                        
                        # Draw background for all classes
                        if class_info:
                            cv2.rectangle(display_frame,
                                        (start_x - 5, y_position - int(25 * font_scale)),  # Increased top padding from 15 to 25
                                        (start_x + total_width - padding + 5, y_position + rect_height),
                                        (0, 0, 0), -1)
                        
                        # Draw all classes side by side
                        for info in class_info:
                            # Draw label
                            label_x = current_x + (info['width'] - info['label_size'][0]) // 2
                            cv2.putText(display_frame, info['label'],
                                      (label_x, y_position),
                                      cv2.FONT_HERSHEY_SIMPLEX, info['font_scale'], (255, 255, 0), 
                                      info['text_thickness'])
                            
                            # Draw instantaneous count
                            inst_y = y_position + int(20 * info['font_scale'])
                            inst_x = current_x + (info['width'] - info['inst_size'][0]) // 2
                            inst_color = (0, 255, 0) if info['inst_value'] > 0 else (255, 255, 255)
                            cv2.putText(display_frame, info['inst_text'],
                                      (inst_x, inst_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, info['font_scale'], inst_color, 
                                      info['text_thickness'])
                            
                            # Draw accumulated count
                            accum_y = inst_y + int(20 * info['font_scale'])
                            accum_x = current_x + (info['width'] - info['accum_size'][0]) // 2
                            accum_color = (0, 255, 0) if info['accum_value'] > 0 else (255, 255, 255)
                            cv2.putText(display_frame, info['accum_text'],
                                      (accum_x, accum_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, info['font_scale'], accum_color, 
                                      info['text_thickness'])
                            
                            # Move to next class position
                            current_x += info['width'] + padding

    return display_frame
