import os
import json
import cv2
import tqdm

# 🚀 CONFIGURATION
image_dir = "/home/diego/bts/IMAGES"
yolo_output_dir = "runs/detect/predict/labels"
output_json_prefix = "preannotations_batch"
batch_size = 20000

# Class mapping (YOLO class ID → Label Studio label)
CLASS_MAPPING = {
    0: "FALHA",
    1: "LEVE",
    2: "MODERADA",
    3: "GRAVE",
    4: "GRAVE_ABSCESSO"
}

def save_batch(annotations, batch_number):
    """Save a batch of annotations to a JSON file"""
    output_filename = f"{output_json_prefix}_{batch_number:03d}.json"
    with open(output_filename, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"✅ Batch {batch_number} saved to {output_filename} ({len(annotations)} annotations)")

# 📝 Store Label Studio formatted annotations
label_studio_annotations = []
batch_number = 1
processed_count = 0

# Get all txt files and sort them for consistent batching
txt_files = [f for f in os.listdir(yolo_output_dir) if f.endswith(".txt")]
txt_files.sort()  # Sort for consistent ordering

print(f"🔍 Found {len(txt_files)} annotation files")
print(f"📦 Creating batches of {batch_size} images each")

for txt_file in tqdm.tqdm(txt_files, desc="Processing annotations"):
    # Get corresponding image file path
    image_filename = txt_file.replace(".txt", ".jpg")
    image_path = f"/data/local-files?d=suporte_cliente_datasets/BOVINOS/LABEL_STUDIO/BTS_ABCESSO/{image_filename}"
    local_image_path = os.path.join(image_dir, image_filename)

    # Load image to get dimensions
    if not os.path.exists(local_image_path):
        print(f"❌ Image not found: {local_image_path}")
        continue

    img = cv2.imread(local_image_path)
    if img is None:
        print(f"❌ Could not load image: {local_image_path}")
        continue
        
    original_height, original_width = img.shape[:2]  # Get image dimensions

    annotations = []
    with open(os.path.join(yolo_output_dir, txt_file), "r") as f:
        for line in f.readlines():
            values = line.strip().split()
            if len(values) < 6:
                continue  # Skip invalid lines

            class_id, x_center, y_center, width, height, conf = map(float, values)
            class_id = int(class_id)  # Convert to integer
            class_label = CLASS_MAPPING.get(class_id, "unknown")  # Get class name

            # Convert YOLO normalized values to absolute positions
            annotations.append({
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": (x_center - width / 2) * 100,
                    "y": (y_center - height / 2) * 100,
                    "width": width * 100,
                    "height": height * 100,
                    "rectanglelabels": [class_label]
                }
            })

    # Build full annotation structure
    annotation_data = {
        "data": {"image": image_path},
        "predictions": [{
            "result": annotations
        }]
    }

    label_studio_annotations.append(annotation_data)
    processed_count += 1

    # Check if we've reached the batch size
    if len(label_studio_annotations) >= batch_size:
        save_batch(label_studio_annotations, batch_number)
        label_studio_annotations = []  # Reset for next batch
        batch_number += 1

# Save any remaining annotations in the last batch
if label_studio_annotations:
    save_batch(label_studio_annotations, batch_number)

print(f"🎉 Processing completed!")
print(f"📊 Total images processed: {processed_count}")
print(f"📦 Total batches created: {batch_number}")
print(f"📁 Batch files: {output_json_prefix}_001.json to {output_json_prefix}_{batch_number:03d}.json")
