import yaml

class DetectLabels:
    def __init__(self, classes_yaml_path: str):
        with open(classes_yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        try:
            # pull the list out of the YAML
            self.defect_labels = cfg['defect_classes']
        except (TypeError, KeyError):
            raise ValueError("`detect_classes.yaml` must contain a top-level 'defect_labels'")

    def __iter__(self):
        # make DetectLabels iterable
        return iter(self.defect_labels)

    def __len__(self):
        # allow len(detect_labels)
        return len(self.defect_labels)

    def __getitem__(self, idx):
        # allow indexing: detect_labels[3]
        return self.defect_labels[idx]