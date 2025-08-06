#!/usr/bin/env python3
"""
Script to read YOLO models and display the classes they were trained on.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


def find_yolo_models(directory: str) -> List[str]:
    """Find all YOLO model files (.pt) in the given directory."""
    model_files = []
    for file in os.listdir(directory):
        if file.endswith('.pt') and 'yolo' in file.lower():
            model_files.append(os.path.join(directory, file))
    return model_files


def show_model_classes(model_path: str) -> None:
    """Load a YOLO model and display its classes."""
    print(f"\nAnalyzing model: {os.path.basename(model_path)}")
    print("-" * 50)
    
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Get the class names
        class_names = model.names
        
        # Display the classes
        print(f"Total classes: {len(class_names)}")
        print("\nClass ID -> Class Name:")
        for idx, name in class_names.items():
            print(f"{idx}: {name}")
            
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Show classes for YOLO models')
    parser.add_argument('--model', type=str, help='Path to specific YOLO model file', default='/home/diego/360.pt')
    parser.add_argument('--dir', type=str, default=os.getcwd(), 
                        help='Directory to search for YOLO models (default: current directory)')
    
    args = parser.parse_args()
    
    if args.model:
        if os.path.exists(args.model):
            show_model_classes(args.model)
        else:
            print(f"Error: Model file not found: {args.model}")
    else:
        models = find_yolo_models(args.dir)
        if models:
            print(f"Found {len(models)} YOLO model(s):")
            for model in models:
                show_model_classes(model)
        else:
            print(f"No YOLO model files found in {args.dir}")
            print("Please specify a model file with --model or a directory with --dir")


if __name__ == "__main__":
    main()
