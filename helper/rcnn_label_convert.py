import os
import json
from cProfile import label
from pathlib import Path
from PIL import Image



def convert_yolo_to_json(img_dir, label_dir, output_json_path):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    annotation_dict = {}
    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    for img_path in img_paths:
        filename = img_path.name
        label_path = label_dir / f"{img_path.stem}.txt"

        with Image.open(img_path) as img:
            width, height = img.size

        detections = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, cx, cy, w, h = map(float, parts)
                    x1 = int((cx - w / 2) * width)
                    y1 = int((cy - h / 2) * height)
                    x2 = int((cx + w / 2) * width)
                    y2 = int((cy + h / 2) * height)

                    detections.append({
                        "label": int(class_id),
                        "bbox": [x1,y1,x2, y2]
                    })
        annotation_dict[filename] = {
            "filename": filename,
            "detections": detections
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(annotation_dict, f, indent=2, ensure_ascii=False)

    print(f"Annotations saved to {output_json_path}")

if __name__ == '__main__':
    convert_yolo_to_json(
        img_dir="../dataset/train/images",
        label_dir="../dataset/train/labels",
        output_json_path="../dataset/train/annotations.json"
    )