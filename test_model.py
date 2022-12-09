# For machine learning
import torch
from torchvision import transforms
from PIL import Image
from neural_network import NeuralNetwork

# For augmenting input image
from augment_image import get_bounding_boxes, get_chars, sort_chars
print("Imported modules")

# Variables to change
IMG_PATH = "images/handwritten_test.jpg"
model_path = "trained_models/2_layer_90.5%_model.pth"
classes = ["(", ")", ".", "/", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", "*", "2", "x", "y", "z", "0"]   # https://stackoverflow.com/questions/6477823/display-special-characters-when-using-print-statement/6478018#6478018
print("Loaded image")

# Get characters
contours = get_bounding_boxes(IMG_PATH)
print("Got contours")

sorted_contours = sort_chars(contours)
print("Sorted contours")

chars = get_chars(IMG_PATH, sorted_contours, True)
print("Got characters")

# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
print("Loaded model")

# Use model
model.eval()
equation = []

for char in chars:
    char = Image.fromarray(char)  # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/43234001#43234001
    char = transforms.ToTensor()(char)
    with torch.no_grad():
        pred = model(char)
        # print(" --> ", pred[0])
        best = pred[0].argmax(0)
        equation.append(classes[best])
        print(f"Prediction is {classes[best]}")
