# main.py
from sympy import false

from scripts import labeling
from scripts import dataLoader
from scripts import  image_dataset_splitter

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



def main():
    path = "dataset"

    while True:
        # Step 1: Label images
        print("Label photos: 1 \nSplitting photos: 2 \nTraining: 3 \nQuit: 4 \n", )
        user_value = input("Enter a value: ")

        if user_value == "1":
            print("Starting labeling tool...")
            label_tool = labeling.Label(path)
            label = label_tool.run()
            if not label:
                user_value.split()

        elif user_value == "2":
            print("not working")
        elif user_value == "3":
            print("not working")
        else:
            break

if __name__ == '__main__':
    main()