import os
from flask import *
import proccess_image
# import werkzeug

app = Flask(__name__)

uploads_dir = os.path.join("", "uploads")
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['uploaded_file']
        f.save(os.path.join(uploads_dir, "image.jpg"))
        # Only if you have werkzeug.secure_filename() working
        # f.save(os.path.join(uploads_dir, werkzeug.secure_filename(f.filename)))
        return render_template("upload_success.html", name=f.filename)

@app.errorhandler(404)
def not_found(e):
    return render_template("error_404.html")


# Only for running locally
app.run(host='localhost', port=8080, debug=True)
