# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# With help of https://github.com/paulbaumgarten/pytorch-mnist-demo

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2
import imutils
import numpy as np
# To solve equations
from sympy import symbols, solve    # https://problemsolvingwithpython.com/10-Symbolic-Math/10.06-Solving-Equations/


# Neural network configurations
input_layer = 100 * 100
layer_1 = 128
layer_2 = 64
output_layer = 21
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, output_layer)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Read image from which text needs to be extracted
img_path = "images/equation_test.jpg"
img = cv2.imread(img_path)
# Copy image for drawing rectangles
img2 = img.copy()

img[img > 100] = 255  # type: ignore    # https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value/19666680#19666680

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area of the rectangle to be detected.
# A smaller value like (10, 10) will detect each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Get all the x-positions of the bounding boxes
all_x = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    all_x.append(x)

contours = np.array(contours, dtype=object)

# Bubble sort the array
sorted = False
while sorted is False:
    for pos in range(len(all_x) - 1):
        if all_x[pos] > all_x[pos + 1]:
            # Use all_x as a basis to sort the contours array
            temp_x = all_x[pos]
            all_x[pos] = all_x[pos + 1]
            all_x[pos + 1] = temp_x
            temp_contour = contours[pos]
            contours[pos] = contours[pos + 1]
            contours[pos + 1] = temp_contour
            sorted = True
    for i in range(len(all_x)-1):
        if all_x[i] > all_x[i + 1]:
            sorted = False

# Get the areas of all the detected numbers
areas = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    areas.append(w*h)

# Looping through the identified contours
# Filtering them and a rectangular part is cropped and returned
area_threshold = 600
border_size = 25
chars = []
i = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    if areas[i] > area_threshold:
        # Cropping the text block for giving input to OCR
        cropped = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

        (tH, tW) = cropped.shape

        thresh = cv2.copyMakeBorder(  # https://docs.opencv.org/4.x/dc/da3/tutorial_copyMakeBorder.html
                    src=cropped,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )

        # if the width is greater than the height, resize along the width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=100, height=100)
        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=100)

        # re-grab the image dimensions (now that its been resized) and then determine how much we need to pad the width and height such that our image will be 28x28
        (tH, tW) = thresh.shape # type: ignore
        dX = int(max(0, 100 - tW) / 2)
        dY = int(max(0, 100 - tH) / 2)

        # pad the image and force 28x28 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded = cv2.resize(padded, (100, 100))

        chars.append(padded)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("", img)
        cv2.imshow("", rect)
        cv2.waitKey(0)

    i += 1

model = NeuralNetwork()
model_path = "trained_models/100_epoch_model_2_#3.pth"
model.load_state_dict(torch.load(model_path))
classes = ["(", ")", ".", "/", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", "*", "2", "x", "y", "z", "0"]

model.eval()
equation = []

for img in chars:
    img = Image.fromarray(img)  # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/43234001#43234001
    img = transforms.ToTensor()(img)
    with torch.no_grad():
        pred = model(img)
        # print(" --> ", pred[0])
        best = pred[0].argmax(0)
        equation.append(classes[best])
        print(f"Prediction is {classes[best]}")

equation = "".join(equation)
print(equation)
# print(solve(equation))
# answer = eval(equation) # https://towardsdatascience.com/python-eval-built-in-function-601f87db191
# print(answer)