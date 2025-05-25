import glob
import os
import random
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms.functional as TF

from torch.utils.data.dataset import Dataset

from helper.load_detection_labels import DetectLabels

def load_images_and_anns(im_dir, annotation_json_file, label2idx):
    im_infos = []

    print(f"Įkeliamas anotacijos failas: {annotation_json_file}")
    with open(annotation_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img_name, content in data.items():

        # Sukurti informaciją dictionary kiekvienai paveikslui
        im_info = {
            'img_id': img_name.split('.')[0],
            'filename': os.path.join(im_dir, img_name),
            'width': content.get('width', 0),
            'height': content.get('height', 0),
            'detections': []
        }

        # Ištraukti raw aptikimų sąrašą (arba tuščias, jei jo nėra)
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

        # Jei nėra aptikimo atveų, vis tiek įtraukiamas vaizdas
        im_infos.append(im_info)

    print(f"Viso paveikslų: {len(im_infos)}")
    return im_infos
class RCNNDataset(Dataset):
    def __init__(self, split, im_dir, annotation_json_path):

        self.split = split
        self.im_dir = im_dir
        self.annotation_json_path = annotation_json_path

        # Užkraunamas klasių pavadinimus iš YAML
        labels_obj = DetectLabels("config/detect_labels.yaml")
        classes = sorted(labels_obj)
        classes = ['background'] + list(labels_obj)

        #label2idx mapping
        self.label2idx = {}
        for idx in range(len(classes)):
            class_name = classes[idx]
            self.label2idx[class_name] = idx

        #idx2label mapping
        self.idx2label = {}
        for idx in range(len(classes)):
            class_name = classes[idx]
            self.idx2label[idx] = class_name

        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir,
                                                annotation_json_path,
                                                self.label2idx)
        print(classes)
    def __len__(self):
        # Grąžina Dataset paveikslų skaičių
        return len(self.images_info)

    def __getitem__(self, index):
        info = self.images_info[index]
        img = Image.open(info['filename']).convert("RGB")

        # Ištraukiama bounding box ir koordinačių labels sąrašus
        bboxes_list = [d['bbox'] for d in info['detections']]
        labels_list = [d['label'] for d in info['detections']]

        # Jei detekcijų nėra, paruošiam tuščius tensor’us formatu [0,4] ir [0]
        if len(bboxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(bboxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)

        # Only augment in train
        if self.split == 'train':
            w, h = img.size

            # Augmentacijos

            # Atsitiktinis horizontalus apvertimas
            if random.random() < 0.5 and boxes.size(0) > 0:
                img = TF.hflip(img)
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2

            # Atsitiktinis 90° pasukimas
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

            # Ryškumo ir kontrasto jitter
            if random.random() < 0.5:
                b = random.uniform(0.8, 1.2)
                c = random.uniform(0.8, 1.2)
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_contrast(img, c)

        # Galutinis konvertavimas į tensor
        img_tensor = TF.to_tensor(img)

        target = {
            'bboxes': boxes,
            'labels': labels
        }
        return img_tensor, target, info['filename']
