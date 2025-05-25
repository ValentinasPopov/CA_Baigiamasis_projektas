from os import mkdir
from pathlib import Path
import json
import os
import PIL
from PIL import Image

class DatasetSplitter:

    def __init__(self, path):
        self.path = Path(path)  # Ensure path is a Path object
        self.train_path = self.path / "train"
        self.test_path = self.path / "test"

    #Paskirstoma train(80%) ir test(20%) folderius paveikslus
    def dataset_size(self, dataset):
        train_size = int(0.8 * len(dataset))
        test_size = int(0.2 * len(dataset))
        return train_size, test_size
    #
    def ensure_folder(self, label, base_path):
        folder = base_path / label
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        return folder
    #
    def move_files(self, files, dest_dir):
        for file in files:
            if file.is_file():
                target_path = dest_dir / file.name
                file.rename(target_path)

    def move_annotations(self, files, dest_dir, label, annotations):
        moved_annotations = []
        for file in files:
            if file.is_file():
                if file.name in annotations:
                    moved_annotations.append(file.name)
                else:
                    print(f"[Warning] No annotation for: {file.name}")
        return moved_annotations
    #
    def save_labels_to_json(self, annotations, output_path, image_names):
        annotations_path = output_path / "annotations.json"

        if annotations_path.exists():
            with open(annotations_path, "r", encoding="utf-8") as f:
                existing_annotations = json.load(f)
        else:
            existing_annotations = {}

        # Filter new annotations
        filtered_annotations = {
            name: annotations[name] for name in image_names if name in annotations
        }

        existing_annotations.update(filtered_annotations)

        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(existing_annotations, f, indent=2, ensure_ascii=False)

    def save_labels_to_txt(
        self,
        annotations,
        output_path,
        image_names
    ):

        # 1) Locate the JSON
        annotations_path = output_path / "annotations.json"
        if not annotations_path.exists():
            print(f"[Warning] {annotations_path} not found, skipping TXT export.")
            return

        # 2) Load it
        with annotations_path.open("r", encoding="utf-8") as f:
            all_annotations = json.load(f)

        # 3) Prepare labels/ folder
        labels_dir = output_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 4) Iterate images
        for img_name in image_names:
            entry = all_annotations.get(img_name)
            if not entry:
                print(f"[Warning] No annotation for {img_name}")
                continue

            dets = entry.get("detections", [])
            if not dets:
                continue

            # 5) Get image size
            img_file = output_path / "images" / entry["filename"]
            with Image.open(img_file) as img:
                img_w, img_h = img.size

            # 6) Build YOLO lines
            lines = []
            for det in dets:
                lbl  = det["label"]
                x0,y0,x1,y1 = det["bbox"]
                xc = (x0 + x1) / 2.0 / img_w
                yc = (y0 + y1) / 2.0 / img_h
                bw = (x1 - x0) / img_w
                bh = (y1 - y0) / img_h
                lines.append(f"{lbl} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # 7) Write .txt named by stem
            stem = Path(entry["filename"]).stem
            txt_path = labels_dir / f"{stem}.txt"
            txt_path.write_text("\n".join(lines), encoding="utf-8")

        print("Convert from JSON to YOLO TXT labels completed.")


    def split_to_train_test_images(
        self,
        annotations
    ):
        train_names = []
        test_names  = []

        for category in ["good", "anomaly"]:
            src = self.path / category
            if not src.exists():
                continue

            imgs = [p for p in src.iterdir() if p.is_file()]
            n_train, _ = self.dataset_size(imgs)
            train_imgs, test_imgs = imgs[:n_train], imgs[n_train:]

            print(f"\nCategory '{category}': {len(imgs)} images → "
                  f"{len(train_imgs)} train, {len(test_imgs)} test")

            # ensure image dirs
            train_img_dir = self.train_path / "images"
            test_img_dir  = self.test_path  / "images"
            train_img_dir.mkdir(parents=True, exist_ok=True)
            test_img_dir.mkdir(parents=True, exist_ok=True)

            # record and move
            train_names += self.move_annotations(train_imgs, self.train_path, category, annotations)
            test_names  += self.move_annotations(test_imgs,  self.test_path,  category, annotations)
            self.move_files(train_imgs, train_img_dir)
            self.move_files(test_imgs,  test_img_dir)

        # 8) Save JSON & TXT in each split
        for out_path, names in (
            (self.train_path, train_names),
            (self.test_path,  test_names)
        ):
            # write the filtered annotations.json
            self.save_labels_to_json(annotations, out_path, names)
            # then write the YOLO .txt files
            self.save_labels_to_txt(      annotations, out_path, names)

        print("Dataset split complete: JSON and TXT labels written.")
