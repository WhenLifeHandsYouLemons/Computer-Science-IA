# Built using: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# With help of: https://github.com/paulbaumgarten/pytorch-mnist-demo
import os

# For machine learning
import torch
from torchvision import transforms
from PIL import Image
from neural_network import NeuralNetwork

# For augmenting input image
from augment_image import get_bounding_boxes, get_chars, sort_chars

dir = os.path.dirname(__file__)

def predict(rendered = True, IMG_PATH = None):
    if IMG_PATH == None:
        IMG_PATH = os.path.join(dir, "static", "images", "uploads", "image.jpg")

    contours = get_bounding_boxes(IMG_PATH)

    sorted_contours = sort_chars(contours)

    chars = get_chars(IMG_PATH, sorted_contours)

    model = NeuralNetwork()
    model_path = os.path.join(dir, "trained_models", "3_layer_98.8%_model.pth")
    model.load_state_dict(torch.load(model_path))

    if rendered == True:
        classes = ["(", ")", ".", r"\div", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", r"\times", "2", "x", "y", "z", "0"]   # https://stackoverflow.com/questions/6477823/display-special-characters-when-using-print-statement/6478018#6478018
    else:
        classes = ["(", ")", ".", "/", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", "*", "2", "x", "y", "z", "0"]

    model.eval()
    equation = []

    for char in chars:
        char = Image.fromarray(char)  # https://stackoverflow.com/a/43234001
        char = transforms.ToTensor()(char)
        with torch.no_grad():
            pred = model(char)
            best = pred[0].argmax(0)
            equation.append(classes[best])
            # print("\n\n --> ", pred[0])
            # print(f"Prediction is {classes[best]}\n\n")

    if rendered == True:
        equation = " ".join(equation)
        equation = f"$$ {equation} $$"    # https://docs.mathjax.org/en/latest/basic/mathematics.html#:~:text=The%20default%20math%20delimiters%20are,%5C)%20for%20in%2Dline%20mathematics.
    else:
        equation = "".join(equation)

    return equation
