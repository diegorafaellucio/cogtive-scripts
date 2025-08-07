import os
import gc
from pathlib import Path
from ultralytics import YOLO
import torch

def process_images_in_batches(model, images_folder, batch_size=8, save_txt=True, save_conf=True):
    """
    Process images in smaller batches to avoid memory overflow
    """
    images_folder = Path(images_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = [f for f in images_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        try:
            # Process current batch
            batch_paths = [str(f) for f in batch_files]
            results = model.predict(batch_paths, save_txt=save_txt, save_conf=save_conf, verbose=False)
            
            # Clear results from memory
            del results
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            continue

def main():
    # Load model
    model_path = "/home/diego/2TB/yolo/Trains/runs/ecotrace_bruise_bm_9.0+pgo_1.0_416_small_sgd_normalizado/weights/best.pt"
    model = YOLO(model_path)
    
    # Set model to use less memory
    model.model.eval()  # Set to evaluation mode
    
    images_folder = "/home/diego/bts/IMAGES"
    
    # Process images in smaller batches (adjust batch_size based on your available memory)
    process_images_in_batches(
        model=model,
        images_folder=images_folder,
        batch_size=4,  # Reduce this number if still running out of memory
        save_txt=True,
        save_conf=True
    )
    
    print("Processing completed!")

if __name__ == "__main__":
    main()