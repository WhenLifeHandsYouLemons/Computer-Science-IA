"""
Built using: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
With help of: https://github.com/paulbaumgarten/pytorch-mnist-demo
"""
import os

#* For predicting
import torch
from torchvision import transforms
from PIL import Image
from neural_network import NeuralNetwork

#* For solver
import sympy as sp  # https://problemsolvingwithpython.com/10-Symbolic-Math/10.06-Solving-Equations/

dir = os.path.dirname(__file__)

def predict(CHARS, MODEL_PATH, DEBUG = False):
    #* Load model into memory
    model = NeuralNetwork()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    #* Initialise character classes
    classes = ("(", ")", ".", "/", "8", "=", "5", "4", "-", "9", "1", "+", "7", "6", "3", "*", "2", "x", "y", "0")
    equation = []

    #* Put each character image through the mdoel
    for char in CHARS:
        char = Image.fromarray(char)  # https://stackoverflow.com/a/43234001
        char = transforms.ToTensor()(char)
        char = char[None, :]    # https://sparrow.dev/adding-a-dimension-to-a-tensor-in-pytorch/
        with torch.no_grad():
            pred = model(char)
            best = pred[0].argmax(0)
            equation.append(classes[best])
            if DEBUG is True:
                print("\n\n --> ", pred[0])
                print(f"Prediction is {classes[best]}\n\n")

    equation = "".join(equation)

    return equation

def combine_equation(equation):
    s_equation_array = list(equation)

    r_equation = []
    for char in s_equation_array:
        if char == "/":
            r_equation.append(r"\div")  # https://stackoverflow.com/questions/6477823/display-special-characters-when-using-print-statement/6478018#6478018
        elif char == "*":
            r_equation.append(r"\times")
        else:
            r_equation.append(char)

    # Add a multiplication sign before and after variables
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    numbers = "0123456789"
    char = 0
    while char < len(s_equation_array):
        # If current character is a letter
        if s_equation_array[char] in alphabet:
            # If it's not the first character
            if char != 0:
                # If previous character is a letter or number
                if s_equation_array[char-1] in alphabet or str(s_equation_array[char-1]) in numbers:
                    s_equation_array.insert(char, "*")
                    char += 1
            # If it's not the last character
            if char != len(s_equation_array)-1:
                # If next character is a letter of number
                if s_equation_array[char+1] in alphabet or str(s_equation_array[char+1]) in numbers:
                    s_equation_array.insert(char+1, "*")
                    char += 1
            char += 2
        else:
            char += 1

    r_equation = " ".join(r_equation)
    r_equation = f"$$ {r_equation} $$"    # https://docs.mathjax.org/en/latest/basic/mathematics.html#:~:text=The%20default%20math%20delimiters%20are,%5C)%20for%20in%2Dline%20mathematics.

    s_equation = "".join(s_equation_array)

    return r_equation, s_equation

def render_math(math):
    sp.init_printing()  # https://stackoverflow.com/a/50447890
    r_math = sp.latex(sp.sympify(f'{"".join(math)}'))   # https://stackoverflow.com/a/4308411
    return r_math

def solver(s_equation, DEBUG = False):
    try:
        var = ""
        for char in s_equation:
            if char in "abcdefghijklmnopqrstuvwxyz":
                var = char
        if "=" in s_equation:
            sympy_eq = sp.sympify("Eq(" + s_equation.replace("=", ",") + ")") # https://stackoverflow.com/a/50047781
        else:
            sympy_eq = sp.sympify("Eq(x," + s_equation + ")")
        answer = sp.solve(sympy_eq)

        if len(answer) == 1:
            answer = render_math(str(answer[0])).lower()
            return f"$$ {var} = {answer} $$"

        answer_list = []
        for i in answer:
            answer_list.append(render_math(str(i)).lower())
            answer_list.append(", ")
        answer_list.pop()

        answer = "".join(answer_list)
        return f"$$ {var} = {answer} $$"
    except Exception as e:
        if DEBUG is True:
            print(e)
        return "The input doesn't seem to be correct, try typing out the equation"
