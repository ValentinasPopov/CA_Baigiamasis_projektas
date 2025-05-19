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

def main():
    path = "dataset"

    while True:
        # Step 1: Label images
        print("Label photos: 1 \nTraining: 2 \nShowing: 3 \nQuit: 4 \n", )
        user_value = input("Enter a value: ")

        if user_value == "1":
            print("Starting labeling tool...")
            label_tool = labeling.Label(path)
            label = label_tool.run()
            if not label:
                user_value = "0"
        elif user_value == "2":
            splitter = DatasetSplitter(path)
            splitter.split_to_train_test_images()
        elif user_value == "3":

            print(os.path.exists("dataset/train/images/annotations.json"))
        else:
            break

if __name__ == '__main__':
    main()