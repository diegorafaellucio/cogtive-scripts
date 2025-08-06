import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image

# Define paths
base_dir = "/home/diego/Nextcloud/8TB/PROJECTS/PycharmProjects/COGTIVE/scripts/modules/export_train_yolo/databases/marfrig"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
output_dir = os.path.join(base_dir, "voc_labels")

# Class labels mapping
class_labels = {
    "0": "porta_aberta",
    "1": "porta_fechada",
}

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of YOLO label files
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]


# Function to convert YOLO format to Pascal VOC

def yolo_to_voc(image_path, label_path, output_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    image = Image.open(image_path)
    width, height = image.size

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

        xmin = int((x_center - bbox_width / 2) * width)
        ymin = int((y_center - bbox_height / 2) * height)
        xmax = int((x_center + bbox_width / 2) * width)
        ymax = int((y_center + bbox_height / 2) * height)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_labels.get(class_id, "unknown")
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    # Format XML output
    tree = ET.ElementTree(annotation)
    xml_str = ET.tostring(annotation, encoding='utf-8')
    pretty_xml = parseString(xml_str).toprettyxml(indent="  ")

    with open(output_path, "w") as xml_file:
        xml_file.write(pretty_xml)


# Convert all labels
for label_file in label_files:
    image_file = label_file.replace(".txt", ".jpg")  # Adjust if images have a different extension
    image_path = os.path.join(images_dir, image_file)
    label_path = os.path.join(labels_dir, label_file)
    output_path = os.path.join(output_dir, label_file.replace(".txt", ".xml"))

    if os.path.exists(image_path):
        yolo_to_voc(image_path, label_path, output_path)
        print(f"Converted: {label_file} -> {output_path}")
    else:
        print(f"Skipping {label_file}, image not found!")