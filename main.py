import os
import flask
from image_processing import augment_image, get_contours, sort_chars, get_chars
from predict_solve import predict, combine_equation, solver

app = flask.Flask(__name__)

dir = os.path.dirname(__file__) # https://stackoverflow.com/a/918178
UPLOADS_DIR = os.path.join(dir, "static", "images", "uploads")
EXAMPLE_FOLDER = os.path.join(dir, "static", "images", "examples")
IMG_PATH = os.path.join(UPLOADS_DIR, "image.jpg")
MODEL_PATH = os.path.join(dir, "trained_models", "2c-3l-2p-3e-cnn_model-#2.pth")
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOADS_DIR   # https://stackoverflow.com/a/46794505
app.config["EXAMPLE_FOLDER"] = EXAMPLE_FOLDER

@app.route("/")
def home():
    try:
        os.remove(os.path.join(UPLOADS_DIR, "image.jpg"))
    finally:
        return flask.render_template("home.html")

@app.route("/uploaded", methods=["POST"])
def upload_success():
    if flask.request.method == "POST":
        f = flask.request.files["uploaded_file"]

        # Save the image with the filename "image.jpg" for ease of access
        f.save(IMG_PATH)

        # Try and run the image through the model
        try:
            # Augment image & get contours
            augmented_img = augment_image(IMG_PATH)
            contours = get_contours(augmented_img)
            sorted_contours = sort_chars(contours)

            # Using the contours, get each character
            chars = get_chars(IMG_PATH, sorted_contours, 500, 23)
            equation_array = predict(chars, MODEL_PATH)

            # Combine the equation into two different styles
            render_equation, solve_equation = combine_equation(equation_array)

            # Use the prediction to get an answer
            answer = solver(solve_equation)

            # Can run model and return a string to pass to the HTML file
            return flask.render_template("upload_success.html", name=f.filename, math_input=render_equation, answer=answer)
        # If it doesn't work for any reason, go to the help page
        except Exception as e:
            print(e)
            os.remove(os.path.join(UPLOADS_DIR, "image.jpg"))
            return flask.render_template("help.html")

# Add approute for editing math page
@app.route("/typed", methods=["POST"])
def type_success():
    if flask.request.method == "POST":
        text = flask.request.form['typed_math']   # https://stackoverflow.com/a/12278642

        render_equation, solve_equation = combine_equation(text)
        answer = solver(solve_equation)

        return flask.render_template("typed_success.html", math_input=render_equation, answer=answer)

# The help page
@app.route("/help")
def help_page():
    return flask.render_template("help.html")

# If the user enter the wrong URL
@app.errorhandler(404)
def not_found(e):
    return flask.render_template("error_404.html")


# Only for running locally
app.run(host="localhost", port=8080, debug=True)
