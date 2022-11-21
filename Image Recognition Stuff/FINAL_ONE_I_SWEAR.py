from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

# Download datasets
data_path = 'data/'
mnist = datasets.MNIST(data_path, train=True, download=True)
mnist_val = datasets.MNIST(data_path, train=False, download=True)
mnist = datasets.MNIST(data_path, train=True, download=False, transform=transforms.ToTensor())

imgs = torch.stack([img_t for img_t, _ in mnist], dim=3)
print('get mean')
print(imgs.view(1, -1).mean(dim=1))
print('get standard deviation')
print(imgs.view(1, -1).std(dim=1))

mnist = datasets.MNIST(data_path, train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
mnist_val = datasets.MNIST(data_path, train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))

# print("Shape of training data",mnist.shape)

input_size = 28 * 28
hidden_sizes = [128, 64]
output_size = 10
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], output_size), nn.LogSoftmax(dim=1))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()
n_epochs = 10
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        batch_size = imgs.shape[0]
        output = model(imgs.view(batch_size, -1))
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=64, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy:", (correct / total) * 100)

PATH = "trained_models/trial_model_3.pth"
torch.save(model.state_dict(), PATH)
