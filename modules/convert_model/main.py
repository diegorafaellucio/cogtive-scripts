from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/diego/2TB/yolo/Trains/v8/object_detection/MARFRIG_1.0/GERAL_1.0/trains/nano/416/runs/detect/train_no_augmentatio_no_erase_and_no_crop_fraction_no_scale_no_translate_4_classes/weights/best.pt")

# Export the model to ONNX format
model.export(format="onnx")