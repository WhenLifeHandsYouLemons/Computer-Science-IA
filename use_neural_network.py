# Built off of: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# With help of: https://github.com/paulbaumgarten/pytorch-mnist-demo

# For machine learning
import torch
from torchvision import transforms
from PIL import Image
from neural_network import NeuralNetwork

# For augmenting input image
from augment_image import get_bounding_boxes, get_chars, sort_chars
# print("Imported modules")

def predict(IMG_PATH = None):
    if IMG_PATH == None:
        IMG_PATH = "uploads/image.jpg"    # THIS PATH FOR FINAL
    # print("Loaded image")

    contours = get_bounding_boxes(IMG_PATH)
    # print("Got contours")

    sorted_contours = sort_chars(contours)
    # print("Sorted contours")

    chars = get_chars(IMG_PATH, sorted_contours)
    # print("Got characters")

    model = NeuralNetwork()
    model_path = "trained_models/100_epoch_model_2_#3.pth"
    model.load_state_dict(torch.load(model_path))
    # print("Loaded model")

    classes = ["(", ")", ".", r"\div", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", r"\times", "2", "x", "y", "z", "0"]   # https://stackoverflow.com/questions/6477823/display-special-characters-when-using-print-statement/6478018#6478018

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
            # print(f"Prediction is {classes[best]}")

    equation = " ".join(equation)
    equation = f"$${equation}$$"    # https://docs.mathjax.org/en/latest/basic/mathematics.html#:~:text=The%20default%20math%20delimiters%20are,%5C)%20for%20in%2Dline%20mathematics.
    # print(f"Prediction: {equation}")
    # print(solve(equation))
    # answer = eval(equation) # https://towardsdatascience.com/python-eval-built-in-function-601f87db191
    # print(answer)
    return equation
