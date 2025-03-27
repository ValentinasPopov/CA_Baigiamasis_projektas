# main.py

from scripts import labeling
from scripts import dataLoader

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



def main():
    path = "images"

    # Step 1: Label images
    label_tool = labeling.Label(path)
    label_tool.run()

    # Step 2: Split data into train/test folders
    loader = dataLoader.ImageDataLoader(path)
    loader.split_to_train_test_images()


if __name__ == '__main__':
    main()