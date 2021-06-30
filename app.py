from flask import (
    Flask,
    render_template,
    url_for,
    redirect,
    request,
    flash,
    session,
    send_file,
)
import uuid
from flask_session import Session
from werkzeug.utils import secure_filename
import os
from module.mask import Mask

ROOT_DIR = "module"
RCNN = Mask(
    PathToROOT=os.path.join(ROOT_DIR, "detector/models"),
    classMap=os.path.join(ROOT_DIR, "detector/taco_config/map_10.csv"),
    modelName="mask_rcnn_taco_0100",
    pathToDataset=os.path.join(ROOT_DIR, "data"),
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = b"XY\x8cI\xb3\x82\xeb\xc1\xe4\xa4t\xa3\x8b`\x9b\xd2"
Session(app)

# no cache from chorme and safari (dev pruporse)
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.cache_control.max_age = 0
    return response


ALLOWED_EXTENSIONS = [".jpg", ".png"]


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No File")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pathFileName = os.path.join(
                app.config["UPLOAD_FOLDER"], uuid.uuid4().hex + filename
            )
            session["image"] = pathFileName
            file.save(pathFileName)
            return render_template("index.html", preview="True")

    return render_template("index.html")


@app.route("/mask", methods=["POST", "GET"])
def dectect():
    pathoutput = os.path.join("static", "result", os.path.basename(session["image"]))
    RCNN.detectIMG(session["image"], pathoutput)
    return render_template("mask.html", file=pathoutput)


if __name__ == "__main__":
    app.run(debug=True)
