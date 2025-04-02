import shutil
import random
from pathlib import Path
from typing import List
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import defaultdict


class ImageDataLoader:

    def __init__(self, path):
        self.path = Path(path)  # Ensure path is a Path object
        self.train_path = self.path / "train"
        self.test_path = self.path / "test"

    #
    def get_train_test_loaders(self, batch_size=16):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.ImageFolder(self.train_path, transform=transform)
        test_dataset = datasets.ImageFolder(self.test_path, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, test_loader

    #
    def get_cv_train_test_loaders(self, batch_size=16, n_folds=2):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = datasets.ImageFolder(self.train_path, transform=transform)
        targets = np.array([sample[1] for sample in dataset.samples])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=False)

            folds.append((train_loader, val_loader))
            print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")

        return folds