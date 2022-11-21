import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from tensorflow import keras

image_x = 28
image_y = 28
hidden_sizes = [128, 64] # 28 was 128
output_size = 10
model = nn.Sequential(nn.Linear(image_x*image_y, hidden_sizes[0]), nn.ReLU(), nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], output_size), nn.LogSoftmax(dim=1))
checkpoint = torch.load(Path("trained_models/trial_model_1.pth"))
model.load_state_dict(checkpoint)

file_path = "image_01.jpeg"

test_image = tf.keras.preprocessing.image.load_img(file_path)

image = cv2.imread(file_path)
# print(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

chars = []
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
    if (w >= 0 and w <= 150) and (h >= 15 and h <= 120):
        # extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        print(tH, tW)

        # if the width is greater than the height, resize along the
        # width dimension
        if tW > image_x:
            thresh = imutils.resize(thresh, width=image_x)
        # otherwise, resize along the height
        elif tH > image_y:
            thresh = imutils.resize(thresh, height=image_y)

        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 28x28
        (tH, tW) = thresh.shape
        dX = int(max(0, image_x - tW) / 2.0)
        dY = int(max(0, image_y - tH) / 2.0)

        # pad the image and force 28x28 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded = cv2.resize(padded, (image_x, image_y))

        # prepare the padded image for classification via our
        # handwriting OCR model
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)

        # update our list of characters that will be OCR'd
        chars.append((padded, (x, y, w, h)))

boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
print(chars.shape)
chars = torch.from_numpy(chars)

# OCR the characters using our handwriting recognition model
for char in chars:
    preds = model(char)

    prediction = int(torch.max(preds.data, 1)[1].numpy())
    print(prediction)
