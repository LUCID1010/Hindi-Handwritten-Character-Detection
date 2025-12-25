from flask import Flask, render_template, request
import cv2, os
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
from utils.segment import segment_characters
from utils.predict import predict_character

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

model = load_model("model/hindi_cnn_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = cv2.imread(path)
        processed = preprocess_image(img)
        chars = segment_characters(processed)

        for ch in chars:
            text += predict_character(ch, model)

    return render_template("index.html", text=text)

if __name__ == "__main__":
    app.run(debug=True)

