from image_processing import get_contours, sort_chars, get_chars
from predict_solve import predict, combine_equation, solver
print("Imported modules")

# Path pointers
IMG_PATH = "static/images/examples/example1.jpg"
MODEL_NAME = "trained_models/2c-3l-2p-cnn_model-#1.pth"

# Get characters
contours = get_contours(IMG_PATH)
print("Got contours")

sorted_contours = sort_chars(contours)
print("Sorted contours")

chars = get_chars(IMG_PATH, sorted_contours, 600, 25, True) # Debug is True
print("Got characters")

# Predict characters in input image
equation_array = predict(chars, MODEL_NAME)
r_equation, s_equation = combine_equation(equation_array)
print("Using an image:")
print(r_equation + solver(s_equation))

# Manually typing an equation
r_equation, s_equation = combine_equation("12x^3-28x^2-3x+8=0") # Can be complex
r_equation, s_equation = combine_equation("8+3")    # And also very simple

print("Manually typing the equation:")
print(r_equation + solver(s_equation))  # Debug is off
