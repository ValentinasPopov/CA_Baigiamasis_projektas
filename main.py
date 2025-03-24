# This is a sample Python script.

from scripts import labeling
from scripts import dataLoader

def main():
    path = "images"

    #
    label_tool = labeling.Label(path)
    label_tool.run()

    #
    loader = dataLoader.DataLoader(path)
    loader.split_to_train_test_images()

if __name__ == '__main__':
    main()


