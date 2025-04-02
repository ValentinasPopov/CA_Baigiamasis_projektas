

from pathlib import Path

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
    #
    def split_to_train_test_images(self):
        for category in ['good', 'anomaly']:
            source_dir = self.path / category

            if not source_dir.exists():
                continue
            images = list(source_dir.glob("*"))

            train_size, test_size = self.dataset_size(images)
            train_imgs = images[:train_size]
            test_imgs = images[test_size:]

            print(f"\nSplitting '{category}': {len(images)} total")
            print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

            train_dir = self.ensure_folder(category, self.path / "train")
            test_dir = self.ensure_folder(category, self.path / "test")

            self.move_files(train_imgs, train_dir)
            self.move_files(test_imgs, test_dir)

            print(f"Moved images for category: {category}")