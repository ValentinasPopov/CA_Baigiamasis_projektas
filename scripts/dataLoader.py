import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import json


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
class VOCDataset(Dataset):
    def __init__(self, split, im_dir, annotation_json_path):

        self.split = split
        self.im_dir = im_dir
        self.annotation_json_path = annotation_json_path
        classes = [
            'knok', 'scratchs'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, annotation_json_path, self.label2idx)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2 - x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return im_tensor, targets, im_info['filename']