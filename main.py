

1# main.py
from sympy import false

from scripts import labeling
from scripts import  image_dataset_splitter

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.image_dataset_splitter import DatasetSplitter

from scripts.inference_rcnn import infer
from scripts.train_rcnn import train
from models.Yolo import run_inference_yolo, train_custom_yolo

def main():
    path = "dataset"

    while True:
        # Step 1: Label images
        print(
              "Label photos ⎯ 1\n"
              "Training YOLO ⎯ 2\n"
              "Training RCNN ⎯ 3\n"
              "Run inference YOLO ⎯ 4\n"
              "Run inference RCNN ⎯ 5\n")
        user_value = input("Enter a value: ")

        if user_value == "1":
            print("Starting labeling tool...")
            label_tool = labeling.Label(path)
            label = label_tool.run()
            if not label:
                user_value = "0"
        elif user_value == "2":
            train_custom_yolo()
        elif user_value == "3":
            train_rcnn()
        elif user_value == "4":
            run_inference_yolo()
        elif user_value == "5":
            infer_rcnn()
        else:
            break

if __name__ == '__main__':
    main()