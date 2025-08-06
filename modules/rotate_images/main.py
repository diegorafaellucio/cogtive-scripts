import os
import cv2

# Define paths
image_dir = "/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/12-CARCASS_CLASSIFICATION/ECOTRACE/GERAL/2.0/images"
output_dir = os.path.join(image_dir, "rotated")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Function to rotate images counterclockwise by 90 degrees
def rotate_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path}, unable to read.")
        return
    rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(output_path, rotated)

# Process all images
for img_file in image_files:
    src_path = os.path.join(image_dir, img_file)
    dest_path = os.path.join(output_dir, img_file)
    rotate_image(src_path, dest_path)
    print(f"Rotated: {img_file} -> {dest_path}")
