# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Available datasets: https://pytorch.org/vision/stable/datasets.html

# from __future__ import print_function, division
import os
import torch
# import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
import glob
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Testing section
DATASET_PATH = "datasets/clarencezhao/"
filelist = glob.glob(f"{DATASET_PATH}train/decimal/*.*")
numbers_frame = np.array([np.array(Image.open(fname)) for fname in filelist])   # https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array

# print(f"Image name: {numbers_frame[0]}")
# print(f"numbers shape: {numbers_frame.shape}")
# print(f"First 4 numbers: {numbers_frame[:4]}")
# print(list(numbers_frame[0]))

print(f"{DATASET_PATH}train/decimal/{numbers_frame[0]}")

exit()

# Working section
class CustomImageDataset(Dataset):
    def __init__(self, truths_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = np.array(os.listdir(truths_dir))  # The truths_dir is usually: "datasets/clarencezhao/train/" # https://docs.python.org/3/library/os.html#os.listdir
        self.img_dir = img_dir  # The img_dir is usually: "datasets/clarencezhao/train/INSERT_SYMBOL_HERE/"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
