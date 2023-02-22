# For machine learning
import torch
from torchvision import transforms
from PIL import Image
from neural_network import NeuralNetwork

# For augmenting input image
from image_processing import augment_image, get_contours, sort_chars, get_chars
print("Imported modules")

# Variables to change
IMG_PATH = "static/images/examples/example1.jpg"
model_path = "trained_models/2c-3l-2p-cnn_model-#2.pth"
classes = ("(", ")", ".", "/", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", "*", "2", "x", "y", "0")   # https://stackoverflow.com/a/6478018
print("Loaded image and classes")

augmented_img = augment_image(IMG_PATH)
contours = get_contours(augmented_img)
print("Got contours")

sorted_contours = sort_chars(contours)
print("Sorted contours")

chars = get_chars(IMG_PATH, sorted_contours, 600, 25, True)
print("Got characters")

# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
print("Loaded model")

# Use model
model.eval()
equation = []

for char in chars:
    char = Image.fromarray(char)  # https://stackoverflow.com/a/43234001
    char = transforms.ToTensor()(char)
    char = char[None, :]    # https://sparrow.dev/adding-a-dimension-to-a-tensor-in-pytorch/
    with torch.no_grad():
        pred = model(char)
        # print(" --> ", pred[0])
        best = pred[0].argmax(0)
        equation.append(classes[best])
        print(f"Prediction is {classes[best]}")

print("".join(equation))