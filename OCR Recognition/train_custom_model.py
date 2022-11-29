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


# Clarence Zhao Dataset
DATASET_PATH = "datasets/clarencezhao_sagyamthapa"


# class CustomDataset(Dataset):
#     def __init__(self, img_dir, train, transform=None, target_transform=None):
#         # All labels
#         self.img_labels = np.array(["decimals", "division", "eight", "equal", "five", "four", "minus", "nine", "one", "plus", "seven", "six", "three", "times", "two", "zero"]) # https://docs.python.org/3/library/os.html#os.listdir
#         self.labels = []

#         # If training model
#         if train is True:
#             self.train_maximums = [0, 513, 1057, 1486, 2040, 2471, 2902, 3451, 3881, 4313, 4858, 5288, 5717, 6146, 6701, 7131, 7557]
#             self.filelist = glob.glob(f"{DATASET_PATH}/train/*.*")
#             self.rgb_numbers_frame = np.array([np.array(Image.open(fname)) for fname in self.filelist])   # https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array

#             self.numbers_frame = []
#             for image in self.rgb_numbers_frame:
#                 temp_image_array = []
#                 for row in image:
#                     temp_row_array = []
#                     for pixel in row:
#                         average = sum(pixel) / len(pixel)
#                         temp_row_array.append(average)
#                     temp_image_array.append(np.array(temp_row_array))
#                 self.numbers_frame.append(np.array(temp_image_array))
#             self.numbers_frame = np.array(self.numbers_frame)

#             i = 1
#             while i != len(self.train_maximums):
#                 for j in range(self.train_maximums[i-1], self.train_maximums[i]):
#                     self.labels.append(self.img_labels[i-1])
#                 i += 1
#             self.labels = np.array(self.labels)
#         # If testing model
#         else:
#             self.test_maximums = [0, 76, 154, 208, 288, 342, 396, 476, 530, 585, 663, 717, 770, 824, 904, 958, 1010]
#             self.filelist = glob.glob(f"{DATASET_PATH}/test/*.*")
#             self.rgb_numbers_frame = np.array([np.array(Image.open(fname)) for fname in self.filelist])   # https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array

#             self.numbers_frame = []
#             for image in self.rgb_numbers_frame:
#                 temp_image_array = []
#                 for row in image:
#                     temp_row_array = []
#                     for pixel in row:
#                         average = sum(pixel) / len(pixel)
#                         temp_row_array.append(average)
#                     temp_image_array.append(np.array(temp_row_array))
#                 self.numbers_frame.append(np.array(temp_image_array))
#             self.numbers_frame = np.array(self.numbers_frame)

#             i = 0
#             while i != len(self.test_maximums):
#                 for j in range(self.test_maximums[i-1], self.test_maximums[i]):
#                     self.labels.append(self.img_labels[i-1])
#                 i += 1
#             self.labels = np.array(self.labels)

#         # Image directory
#         self.img_dir = img_dir  # The img_dir is "datasets/clarencezhao/train/" or "datasets/clarencezhao/eval/"

#         # Any image transforms
#         self.transform = transform
#         self.target_transform = target_transform

#     # Number of classes
#     def __len__(self):
#         return len(self.img_labels)

#     # Get the image and label of a specific index
#     def __getitem__(self, idx):
#         image = self.numbers_frame[idx]
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)

#         return image, label


# Convert PIL image to Tensor format
transform = transforms.Compose([    # https://discuss.pytorch.org/t/image-file-reading-typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-class-pil-image-image/9909
    transforms.Grayscale(num_output_channels=1),    # https://www.programcreek.com/python/example/117700/torchvision.transforms.Grayscale
    transforms.ToTensor()
])

# Initialise test and train data
training_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

test_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

# Show 3 images from train data
for i in range(4):
    plt.imshow(training_data[i][0][0])
    plt.waitforbuttonpress(0)
plt.close()

# Load test and train data
batch_size = 21 # 10*2*2*3*3*7 possible batch numbers
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
hidden_layer_4 = 19
output_layer = 19

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


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
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

PATH = "trained_models/100_epoch_model_2_#2.pth"
torch.save(model.state_dict(), PATH)
print(f"Saved PyTorch Model State to {PATH}")
