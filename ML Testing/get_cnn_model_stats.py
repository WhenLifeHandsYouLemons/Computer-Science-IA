import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork

DATASET_PATH = "datasets/combined"
MODEL_PATH = "trained_models/2c-3l-2p-cnn_model-#2.pth"

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
Test Model On Single Image
"""
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Show images
npimg = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Load model into memory
net = NeuralNetwork()
net.load_state_dict(torch.load(MODEL_PATH))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))


"""
Test Model On All Images
"""
correct = 0
total = 0
# Don't need to get the gradient as we're not training the model
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        # The prediction with the highest confidence is chosen
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# Gradients don't need to be calculated
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# Print accuracy for each class
total_percent = 0
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    total_percent += accuracy
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

total_percent /= len(classes)
print(f"Total accuracy is: {total_percent} %")
