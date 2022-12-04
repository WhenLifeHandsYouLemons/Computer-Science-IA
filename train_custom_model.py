# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Available datasets: https://pytorch.org/vision/stable/datasets.html

# from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torchvision.io import read_image
import glob
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Setting base path
DATASET_PATH = "datasets/clarencezhao_sagyamthapa"

# Convert PIL image to Tensor format
transform = transforms.Compose([    # https://discuss.pytorch.org/t/image-file-reading-typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-class-pil-image-image/9909
    transforms.Grayscale(num_output_channels=1),    # https://www.programcreek.com/python/example/117700/torchvision.transforms.Grayscale
    transforms.ToTensor()
])

# Initialise test and train data
# https://www.isical.ac.in/~crohme/CROHME_data.html
training_data = datasets.ImageFolder(   # https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    root=f"{DATASET_PATH}/train",
    transform=transform
)

test_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

# Show 3 images from train data
# for i in range(4):
#     plt.imshow(training_data[i][0][0])
#     plt.waitforbuttonpress(0)
# plt.close()

# Load test and train data
batch_size = 63 # 10*2*2*3*3*7 possible nice batch numbers
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Check whether the arrays are the right size
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


input_layer = 100 * 100
hidden_layer_1 = 50
hidden_layer_2 = 100
hidden_layer_3 = 500
hidden_layer_4 = 20
output_layer = 21

layer_1 = 128
layer_2 = 64

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(input_layer, hidden_layer_1),
            # nn.ReLU(),
            # nn.Linear(hidden_layer_1, hidden_layer_2),
            # nn.ReLU(),
            # nn.Linear(hidden_layer_2, hidden_layer_3),
            # nn.ReLU(),
            # nn.Linear(hidden_layer_3, hidden_layer_4),
            # nn.ReLU(),
            # nn.Linear(hidden_layer_4, output_layer)

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


model = NeuralNetwork().to(device)
# print(model)

losses = []
corrects = [0]
times = []
i = 0

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            losses.append(loss)
            times.append(i)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return 1


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    corrects.append(correct)  # type: ignore
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if correct >= corrects[-2]:
        PATH = "trained_models/100_epoch_model_2_#3.pth"
        torch.save(model.state_dict(), PATH)
        print(f"Saved PyTorch Model State to {PATH}")



epochs = 500
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    i += train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

    while len(corrects) != len(losses):
        corrects.append(corrects[-1])

print("Done!")


# https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
plt.plot(times, losses, label = "Loss")
plt.plot(times, corrects, label = "Accuracy")
plt.xlabel("Time")
plt.ylabel("Percentage")
plt.title("Model Graph")
plt.xlim(left=-0.001)
plt.legend()
plt.show()
plt.waitforbuttonpress(0)
plt.close()
