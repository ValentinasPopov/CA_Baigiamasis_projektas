

from pathlib import Path
import json

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

    def split_to_train_test_images(self, annotations):
        train_names = []
        test_names = []

        for category in ['good', 'anomaly']:
            source_dir = self.path / category
            if not source_dir.exists():
                continue

            images = list(source_dir.glob("*"))
            images = [img for img in images if img.is_file()]

            train_size, _ = self.dataset_size(images)
            train_imgs = images[:train_size]
            test_imgs = images[train_size:]

            print(f"\nSplitting '{category}': {len(images)} total")
            print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

            train_dir = self.ensure_folder("", self.path / "train")
            test_dir = self.ensure_folder("", self.path / "test")

            train_names += self.move_annotations(train_imgs, self.train_path, category, annotations)
            test_names += self.move_annotations(test_imgs, self.test_path, category, annotations)

            self.move_files(train_imgs, train_dir)
            self.move_files(test_imgs, test_dir)

            print(f"Moved images for category: {category}")

        # Save JSON annotations
        self.save_labels_to_json(annotations, self.train_path, train_names)
        self.save_labels_to_json(annotations, self.test_path, test_names)