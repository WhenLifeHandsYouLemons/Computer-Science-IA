# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# With help of https://github.com/paulbaumgarten/pytorch-mnist-demo

"""
# From https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/10
pil_img = Image.open(img)
print(pil_img.size)

pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
print(pil_to_tensor.shape)

tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
import cv2
from imutils.contours import sort_contours
import imutils


# 6-layer 784-50-100-500-1000-10-10
input_layer = 28 * 28
hidden_layer_1 = 50
hidden_layer_2 = 100
hidden_layer_3 = 500
hidden_layer_4 = 10
output_layer = 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, hidden_layer_3),
            nn.ReLU(),
            nn.Linear(hidden_layer_3, hidden_layer_4),
            nn.ReLU(),
            nn.Linear(hidden_layer_4, output_layer)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# The input images have to be white on black
PATH = "images/test_image_#1.png"
border_size = 50
src = cv2.imread(PATH)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

chars = []

for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # filter out bounding boxes, ensuring they are neither too small nor too large
    # if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):  # The section under this was in this if statement
    # extract the character and threshold it to make the character appear as *white* (foreground) on a *black* background, then grab the width and height of the thresholded image
    roi = gray[y:y + h, x:x + w]
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (tH, tW) = thresh.shape

    thresh = cv2.copyMakeBorder(  # https://docs.opencv.org/4.x/dc/da3/tutorial_copyMakeBorder.html
                 src=thresh,
                 top=border_size,
                 bottom=border_size,
                 left=border_size,
                 right=border_size,
                 borderType=cv2.BORDER_CONSTANT,
                 value=[0, 0, 0]
              )

    # if the width is greater than the height, resize along the width dimension
    if tW > tH:
        thresh = imutils.resize(thresh, width=28)
    # otherwise, resize along the height
    else:
        thresh = imutils.resize(thresh, height=28)

    # re-grab the image dimensions (now that its been resized) and then determine how much we need to pad the width and height such that our image will be 28x28
    (tH, tW) = thresh.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)

    # pad the image and force 28x28 dimensions
    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.resize(padded, (28, 28))

    # update our list of characters that will be OCR'd
    chars.append(padded)

# for char in chars:
#     cv2.imshow("char", char)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

model = NeuralNetwork()
model_path = "trained_models/100_epoch_model_4.pth" # 100 epoch & 4 layers has the best accuracy of 94.7%
model.load_state_dict(torch.load(model_path))
print(f"Using: {model_path}")
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model.eval()

for img in chars:
    img = Image.fromarray(img)  # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/43234001#43234001
    tester = transforms.ToTensor()(img)
    with torch.no_grad():
        pred = model(tester)
        # print(" --> ", pred[0])
        best = pred[0].argmax(0)
        print(f"Prediction is {classes[best]}")

# I imagine the next step is to use your own datasets. This looks useful
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html