# Computer Science IA Documentation

This project has been created for the May 2023 IB Computer Science IA Examination.

Hosted at: [mathconverter.pythonanywhere.com](https://mathconverter.pythonanywhere.com/)

---

## `image_processing.py` Functions

The functions in this file assume you have `cv2` and `numpy` installed.

### `get_contours()`

The `get_contours` function takes in an image path as a string and outputs an array of contours that can be used with the `cv2.boundingRect()` function to get bounding boxes around key features in an image.

The contrast can be changed by adjusting the value inside the brackets on `line 10`. The lower the value, the higher the contrast, however it can cut into the image causing it to not be detected properly.

Example:

```python
contours = get_contours("path/to/image/here")
```

### `sort_chars()`

The `sort_chars` function takes in an array of contours as input and outputs the same array of contours that is sorted in reading order (left-to-right).

This function uses bubble sort to sort the array as there would be very few items to sort.

Example:

```python
contours = get_contours("path/to/image/here")
sorted_contours = sort_chars(contours)
```

### `get_chars()`

The `get_chars` function takes in an image path as a string, an array of contours, an integer defining the area threshold, an integer defining the border size, and boolean of whether debugging tools are needed.

The contours can be taken from the [`get_contours`](#sort_chars) function.

The area threshold determines the minimum allowed area for the bounding box that can be accepted and used. This value can be any integer more than `0`.

The border size parameter defines the thickness of the added border in pixels for better OCR. This value can be any non-negative integer.

The debug mode displays the image that has been inputted and draws a bounding box around all allowed areas. This value is optional and is `False` by default.

Example:

```python
image_path = "path/to/image/here"
contours = get_contours(image_path)
characters = get_chars(image_path, contours, 750, 30, False)
```

---

## `predict_solve.py` Functions

The functions in this file assume `os`, `torch`, `torchvision`, `PIL`, and `sympy` are already installed.

This also requires another file called [`neural_network.py`](https://github.com/WhenLifeHandsYouLemons/Computer-Science-IA/blob/b5d019866da69b253c282128e2f118d974313bb6/neural_network.py) with a class called `NeuralNetwork` to be in the same directory as this one.

### `predict()`

The `predict` function takes in an array of characters, a string referencing the model path, and a boolean stating whether debug tools are needed. It outputs a string of all the predictions combined into one.

The array of characters can be taken from the [`get_chars`](#get_chars) function.

The model path is a string pointing to the model that would be used. This is depedent on the [`neural_network.py`](https://github.com/WhenLifeHandsYouLemons/Computer-Science-IA/blob/b5d019866da69b253c282128e2f118d974313bb6/neural_network.py) file.

The debug parameter displays each of the predictions that the network outputs. This is an optional setting and is set to `False` by default.

Example:

```python
image_path = "path/to/image/here"
model_path = "path/to/model/here"
contours = get_contours(image_path)
characters = get_chars(image_path, contours, 750, 30, False)
predicted_equation = predict(characters, model_path, False)
```

### `combine_equation()`

The `combine_equation` function takes an equation as a string and outputs a string to be used for rendering and another for solving.

The equation can be inputted manually or can also be taken from the [`predict`](#predict) function.

The rendering equation is a string that is formatted for LaTeX rendering (can use MathJax to render LaTeX in webpages).

The solving equation is a string that is formatted for use in the [`solver`](#solver) function.

Example:

```python
render_equation, solve_equation = combine_equation("2x+6=12")
```

### `solver()`

The `solver` equation takes in an equation as a string and a boolean for if debug tools are needed, and outputs an answer as a string.

The inputted equation can be taken from the solving equation of the [`combine_equation`](#combine_equation) function.

The debug tools are optional and are `False` by default.

The output is a string which has been LaTeX formatted for displaying on a webpage through the use of MathJax.

Examples:

```python
render_equation, solve_equation = combine_equation("2x+6=12")
rendered_answer = solver(solve_equation)
```

### `render_math()`

The `render_math` function is an internally used function to convert the answer into a LaTeX formatted version which is then returned.

It takes an string as an input and formats it to follow the LaTeX rules using the SymPy library.

Example:

```python
rendered_answer = render_math("3/4 + 27")
```

---

Usage of all the functions listed above can be seen in the [`example_usage.py`](https://github.com/WhenLifeHandsYouLemons/Computer-Science-IA/blob/b5d019866da69b253c282128e2f118d974313bb6/example_usage.py) file.
