import os
import shutil
import random

# Define source and destination directories
source_dir = "/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/12-CARCASS_CLASSIFICATION/ECOTRACE/GERAL/2.0/WITHOUT_CARCASS/IMAGES"
destination_dir = "/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/12-CARCASS_CLASSIFICATION/ECOTRACE/GERAL/2.0/WITHOUT_CARCASS/SELECTED_IMAGES"

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all image files in the source directory
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Shuffle the list of images
random.shuffle(all_images)

# Move exactly 20,000 images
moved_count = 0
for image in all_images:
    if moved_count >= 20000:
        break
    src_path = os.path.join(source_dir, image)
    dest_path = os.path.join(destination_dir, image)
    shutil.move(src_path, dest_path)
    moved_count += 1

print(f"Moved {moved_count} images to {destination_dir}")
