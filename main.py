import os
from flask import *
from use_neural_network import predict

app = Flask(__name__)

uploads_dir = os.path.join("", "uploads")
os.makedirs(uploads_dir, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])  # type: ignore
def success():
    if request.method == "POST":
        f = request.files["uploaded_file"]

        # Try and run the image through the model
        try:
            f.save(os.path.join(uploads_dir, "image.jpg"))

            prediction = predict()
            answer = 0

            # Can run model and return a string to pass into this return line as name
            return render_template("upload_success.html", name=f.filename, math_input=prediction, answer=answer)
        # If it doesn't work for any reason, go to the help page
        except:
            return render_template("help.html")

# Add approute for editing math page

@app.route("/help")
def help():
    return render_template("help.html")

@app.errorhandler(404)
def not_found(e):
    return render_template("error_404.html")


# Only for running locally
# app.run(host="localhost", port=8080, debug=True)
