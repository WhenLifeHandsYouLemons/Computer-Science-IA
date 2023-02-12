# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Available datasets: https://pytorch.org/vision/stable/datasets.html

import torch
from skimage import transform
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn
from neural_network import NeuralNetwork

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Setting base path
DATASET_PATH = "datasets/combined"

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
for i in range(4):
    plt.imshow(training_data[i][0][0])
    plt.waitforbuttonpress(0)
plt.close()

# Load test and train data
batch_size = 50 # Possible nice batch numbers: 10*10*5*131
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

# Create model
model = NeuralNetwork().to(device)
print(model)

losses = []
corrects = [0.0]
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
    corrects.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if correct >= max(corrects):
        PATH = "trained_models/3_layer_model.pth"
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
