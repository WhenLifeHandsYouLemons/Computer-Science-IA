import os
import flask
from use_neural_network import predict
from solve_equations import solver

app = flask.Flask(__name__)

dir = os.path.dirname(__file__) # https://stackoverflow.com/a/918178
UPLOADS_DIR = os.path.join(dir, "static", "images", "uploads")
EXAMPLE_FOLDER = os.path.join(dir, "static", "images", "examples")
# os.makedirs(UPLOADS_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOADS_DIR   # https://stackoverflow.com/a/46794505
app.config["EXAMPLE_FOLDER"] = EXAMPLE_FOLDER

@app.route("/")
def home():
    try:
        os.remove(os.path.join(UPLOADS_DIR, "image.jpg"))
    finally:
        return flask.render_template("home.html")

@app.route("/upload", methods=["POST"])
def success():
    if flask.request.method == "POST":
        f = flask.request.files["uploaded_file"]

        # Save the image with the filename "image.jpg" for ease of access
        f.save(os.path.join(UPLOADS_DIR, "image.jpg"))

        # Try and run the image through the model
        try:
            # Run the prediction function to get a prediction
            render_prediction = predict(True)
            solve_prediction = predict(False)

            # Use the prediction in the solver to get an answer
            answer = solver(solve_prediction)

            # Can run model and return a string to pass to the HTML file
            return flask.render_template("upload_success.html", name=f.filename, math_input=render_prediction, answer=answer)
        # If it doesn't work for any reason, go to the help page
        except Exception as e:
            print(e)
            return flask.render_template("help.html")

# Add approute for editing math page

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
