import os
import json
from PIL import Image

# Paths to the datasets
base_paths = ["../dataset/train/", "../dataset/test/"]
base_label_paths = ["../dataset/train/labels", "../dataset/test/labels"]



for base_path, labels_path in zip(base_paths, base_label_paths):
    annotation_path = os.path.join(base_path, 'annotations.json')

    # Load annotations.json
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Create labels directory if not exists
    os.makedirs(labels_path, exist_ok=True)

    for key, data in annotations.items():
        filename = data['filename']
        image_path = os.path.join(base_path, "images", filename)

        # Open image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        detections = data['detections']

        # Skip if no detections
        if not detections:
            continue

        label_lines = []

        for det in detections:
            label = det['label'] - 1  # YOLO labels typically start from 0
            x_min, y_min, x_max, y_max = det['bbox']

            # Calculate YOLO format (normalized center x, center y, width, height)
            x_center = (x_min + x_max) / 2.0 / width
            y_center = (y_min + y_max) / 2.0 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            label_lines.append(f"{label} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Save YOLO label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(labels_path, label_filename), 'w') as label_file:
            label_file.write('\n'.join(label_lines))

print("Conversion completed successfully.")