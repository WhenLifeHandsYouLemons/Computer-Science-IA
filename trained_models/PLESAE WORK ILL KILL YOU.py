import torch
from torchvision.transforms import transforms
from PIL import Image
# from cnn_main import CNNet
from pathlib import Path
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

# model = CNNet(5)
input_size = 28 * 28
hidden_sizes = [128, 64]
output_size = 10
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], output_size), nn.LogSoftmax(dim=1))
checkpoint = torch.load(Path("trained_models/trial_model_1.pth"))
model.load_state_dict(checkpoint)
# model.eval()

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open(Path("image_01.jpeg"))

input = trans(image)

input = input.view(1, 3, 32, 32)

output = model(input)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

# if (prediction == 0):
#     print('daisy')
# if (prediction == 1):
#     print('dandelion')
# if (prediction == 2):
#     print('rose')
# if (prediction == 3):
#     print('sunflower')
# if (prediction == 4):
#     print('tulip')
