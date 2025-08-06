import cv2
import os
import tqdm

# Input and output directories
input_dir = "/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/2.0/IMAGES"

# Create output directory if it doesn't exist

# Process all images in the directory
for filename in tqdm.tqdm(os.listdir(input_dir)):
    file_path = os.path.join(input_dir, filename)

    # Read image
    image = cv2.imread(file_path)
    if image is None:
        continue  # Skip non-image files

    # Get dimensions
    height, width, _ = image.shape

    # Crop 260 pixels from left and right
    image = image[:,80:width - 100]

    # Save the cropped image
    output_path = os.path.join(input_dir, filename)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    cv2.imwrite(output_path, image)

print("Cropping completed!")
