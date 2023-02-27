"""
Adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Datasets used:
1. https://www.isical.ac.in/~crohme/CROHME_data.html
    a. Used https://github.com/ThomasLech/CROHME_extractor to extract CROHME data into image files
2. https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
3. https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork


"""
Loading dataset and sorting it into training testing
"""
DATASET_PATH = "datasets"

# Predefined transforms to convert PIL image to Tensor format and have one colour channel
transform = transforms.Compose([    # https://discuss.pytorch.org/t/image-file-reading-typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-class-pil-image-image/9909
    transforms.Grayscale(num_output_channels=1),    # https://www.programcreek.com/python/example/117700/torchvision.transforms.Grayscale
    transforms.ToTensor()
])

# Initialise test and train data
training_data = torchvision.datasets.ImageFolder(   # https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    root=f"{DATASET_PATH}/train",
    transform=transform
)

test_data = torchvision.datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

# Smaller batch size, faster to train
batch_size = 4
# Create dataloaders for training and testing
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

classes = ("(", ")", ".", "/", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", "*", "2", "x", "y", "0")   # https://stackoverflow.com/a/6478018

print("Loaded datasets")


"""
Dataset Checking
"""
# Get random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
npimg = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()


"""
Model Initialisation
"""
net = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Created model")


"""
Train Model
"""
# Number of epochs to train the model for
epochs = 2  # Was 2-3
for epoch in range(epochs):
    running_loss = 0.0

    # Go through the dataset in each of the mini-batches
    for i, data in enumerate(train_loader, 0):
        # Get the inputs
        # Data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print(f"epoch {epoch+1} finished")

print("Finished training model")


"""
Save Model
"""
PATH = "trained_models/2c-3l-2p-cnn_model-#3.pth"
torch.save(net.state_dict(), PATH)

print(f"Saved model to '{PATH}'")
