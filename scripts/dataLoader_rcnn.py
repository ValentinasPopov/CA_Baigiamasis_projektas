import glob
import os
import random

import torch
import torchvision
from PIL import Image, ImageFilter
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import json

import torchvision.transforms.functional as TF
import numpy as np

from helper.load_detection_labels import DetectLabels


def load_images_and_anns(im_dir, annotation_json_file, label2idx):
    im_infos = []

    print(f"Loading annotation file: {annotation_json_file}")
    with open(annotation_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img_name, content in data.items():
        im_info = {
            'img_id': img_name.split('.')[0],
            'filename': os.path.join(im_dir, img_name),
            'width': content.get('width', 0),
            'height': content.get('height', 0),
            'detections': []
        }

        detections = content.get('detections', [])
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            if x2 > x1 and y2 > y1:
                label = detection['label']
                if isinstance(label, str):
                    label = label2idx.get(label, 0)
                im_info['detections'].append({
                    'label': label,
                    'bbox': [x1, y1, x2, y2]
                })

        # Even if no detections, still include this image
        im_infos.append(im_info)

    print(f"Total images{len(im_infos)}")
    return im_infos
class RCNNDataset(Dataset):
    def __init__(self, split, im_dir, annotation_json_path):

        self.split = split
        self.im_dir = im_dir
        self.annotation_json_path = annotation_json_path
        labels_obj = DetectLabels("config/detect_labels.yaml")
        classes = sorted(labels_obj)
        classes = ['background'] + list(labels_obj)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, annotation_json_path, self.label2idx)
        print(classes)
    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        info = self.images_info[index]
        img = Image.open(info['filename']).convert("RGB")

        # Pull out the raw lists
        bboxes_list = [d['bbox'] for d in info['detections']]
        labels_list = [d['label'] for d in info['detections']]

        # Make sure boxes is always a [N,4] tensor, even if N==0
        if len(bboxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(bboxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)

        # Only augment in train
        if self.split == 'train':
            w, h = img.size

            # 1) Random horizontal flip

            if random.random() < 0.5 and boxes.size(0) > 0:
                img = TF.hflip(img)
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2


            # 2) Random 90° rotation

            if random.random() < 0.5 and boxes.size(0) > 0:
                img = img.rotate(90, expand=True)
                new_w, new_h = h, w
                x1, y1, x2, y2 = boxes.unbind(1)
                nx1 = y1
                ny1 = new_h - x2
                nx2 = y2
                ny2 = new_h - x1
                boxes = torch.stack([nx1, ny1, nx2, ny2], dim=1)
                w, h = new_w, new_h

            # 3) Brightness & contrast jitter
            if random.random() < 0.5:
                b = random.uniform(0.8, 1.2)
                c = random.uniform(0.8, 1.2)
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_contrast(img, c)

            # 4) Gaussian blur

            if random.random() < 0.2:
                radius = random.uniform(0.0, 1.5)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))


            # 5) Additive Gaussian noise

            if random.random() < 0.2:
                arr = np.array(img).astype(np.float32)
                noise = np.random.normal(0, 10, arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)


        # Final conversion to tensor
        img_tensor = TF.to_tensor(img)

        target = {
            'bboxes': boxes,
            'labels': labels
        }
        return img_tensor, target, info['filename']
