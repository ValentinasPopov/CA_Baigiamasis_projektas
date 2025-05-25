from pathlib import Path
import cv2
import os
import json

from sympy import false

from helper.load_detection_labels import DetectLabels
from scripts import  image_dataset_splitter



class Label:
    def __init__(self, path: str):
        self.path = path
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_rects = []
        self.img = None
        self.annotations = {}
        self.selected_anomaly = False
        self.selecting = False

        self.defect_labels = DetectLabels("config/detect_labels.yaml")
        self.label2idx = {label: i for i, label in enumerate(self.defect_labels)}
        self.defect_label_keys = {ord(str(i)): label for i, label in enumerate(self.defect_labels, start=1)}


    def draw_rectangle(self, event, x, y, flags, param):

        if self.selecting:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            temp = self.img.copy()
            cv2.rectangle(temp, (self.ix, self.iy), (x, y), (0, 255, 255), 2)
            cv2.imshow("Image", temp)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.selecting = True

            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            bbox = [x1, y1, x2, y2]
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (23, 255, 255), 2)

            # Show label choices
            for i, label in enumerate(self.defect_labels, start=1):
                cv2.putText(self.img, f"{i}: {label}", (10, 60 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Image", self.img)

            while True:
                key = cv2.waitKey(0)
                if key in self.defect_label_keys:
                    label = self.defect_label_keys[key]
                    self.current_rects.append({
                        "bbox": bbox,
                        "label": label
                    })
                    self.selected_anomaly = True
                    self.selecting = False
                    break

    def save_labels_to_json(self, output_path):
        annotations_path = output_path / "annotations.json"

        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)

    def annotate_images(self, input_img_paths, output_path, labels):
        labels_key_dict = {ord(str(i)): label for i, label in enumerate(labels, start=1)}

        # Create output folders
        for label in labels:
            (output_path / label).mkdir(parents=True, exist_ok=True)

        for idx, img_path in enumerate(input_img_paths):
            self.img = cv2.imread(str(img_path))
            self.current_rects = []
            self.selected_anomaly = False

            display_img = self.img.copy()
            cv2.putText(display_img, f"{img_path.name} | left: {len(input_img_paths) - idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, label in enumerate(labels, start=1):
                cv2.putText(display_img, f"{i}: {label}", (10, 60 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Image", self.draw_rectangle)
            cv2.imshow("Image", display_img)

            while True:
                key = cv2.waitKey(0)

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    splitter = image_dataset_splitter.DatasetSplitter(self.path)
                    splitter.split_to_train_test_images(self.annotations)
                    return

                if not self.selected_anomaly and key in labels_key_dict:
                    label = labels_key_dict[key]
                    self.annotations[img_path.name] = {
                        "filename": img_path.name,
                        "detections": []
                    }
                    output_img_path = output_path / label / img_path.name
                    img_path.rename(output_img_path)
                    break

                if self.selected_anomaly:
                    label = "anomaly"
                    self.annotations[img_path.name] = {
                        "filename": img_path.name,
                        "detections": [
                            {
                                "label": self.label2idx[obj["label"]]+1,
                                "bbox": obj["bbox"]
                            } for obj in self.current_rects
                        ]
                    }
                    output_img_path = output_path / label / img_path.name
                    img_path.rename(output_img_path)
                    break

            cv2.destroyAllWindows()

        # Final split + annotation save after all labeling is done
        splitter = image_dataset_splitter.DatasetSplitter(self.path)
        splitter.split_to_train_test_images(self.annotations)

        return True

    def run(self):
        input_path = Path(os.path.join(self.path, "raw"))
        if not input_path.exists() or not input_path.is_dir():
            print(f"Directory not found: {input_path}")
            return

        input_img_paths = sorted([
            p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if not input_img_paths:
            print("No photos found in RAW folder.")
            return

        output_path = Path(self.path)
        output_path.mkdir(parents=True, exist_ok=True)



        return self.annotate_images(
            input_img_paths=input_img_paths,
            output_path=output_path,
            labels=["good", "anomaly"]
        )

