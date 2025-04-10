# main.py
from sympy import false

from scripts import labeling
from scripts import  image_dataset_splitter

from scripts.dataLoader import get_train_test_loaders, get_cv_train_test_loaders
from scripts.model import CustomVGG
from scripts.helper import train, evaluate, predict_localize
from scripts.constants import NEG_CLASS

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



def main():
    path = "dataset"
    subset_name = "wood"

    # Parameters
    batch_size = 10
    target_train_accuracy = 0.98
    lr = 0.0001
    epochs = 10
    class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    heatmap_thres = 0.953
    n_cv_folds = 5

    # Data
    train_loader, test_loader = get_train_test_loaders(
        path=path, batch_size=batch_size, test_size=0.2, random_state=42,
    )

    # Model training
    model = CustomVGG()

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


            class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weight)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            model = train(
                train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy
            )

            model_path = f"weights/{subset_name}_model.h5"
            torch.save(model, model_path)

            #Evalution
            evaluate(model, test_loader, device)

            #Cross Validation

            cv_folds = get_cv_train_test_loaders(
                path=path,
                batch_size=batch_size,
                n_folds=n_cv_folds,
            )

            class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weight)

            for i, (train_loader, test_loader) in enumerate(cv_folds):
                print(f"Fold {i + 1}/{n_cv_folds}")
                model = CustomVGG(input_size)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                model = train(train_loader, model, optimizer, criterion, epochs, device)
                evaluate(model, test_loader, device)

        elif user_value == "3":
            predict_localize(model, test_loader, device, thres=heatmap_thres, n_samples=6, show_heatmap=True)
        else:
            break

if __name__ == '__main__':
    main()