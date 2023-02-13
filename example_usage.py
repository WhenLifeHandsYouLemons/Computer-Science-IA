from image_processing import augment_image, get_contours, sort_chars, get_chars
from predict_solve import predict, combine_equation, solver
print("Imported modules")

# Variables to change
IMG_PATH = "static/images/examples/example1.jpg"
MODEL_NAME = "trained_models/3_layer_model.pth"
print("Loaded image")


# Get characters
augmented_img = augment_image(IMG_PATH)
print("Augmented image")

contours = get_contours(augmented_img)
print("Got contours")

sorted_contours = sort_chars(contours)
print("Sorted contours")

chars = get_chars(IMG_PATH, sorted_contours, 600, 25, True) # Debug is True
print("Got characters")


equation_array = predict(chars, MODEL_NAME)
r_equation, s_equation = combine_equation(equation_array)

# Manually typing an equation
r_equation, s_equation = combine_equation("12x^3-28x^2-3x+8=0")
r_equation, s_equation = combine_equation("9+3")

print(r_equation + solver(s_equation))  # Debug is off
