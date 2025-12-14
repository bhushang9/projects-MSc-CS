from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# -------------------- Flask Setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Load Models --------------------
cnn_model = tf.keras.models.load_model("models/cnn_tomato_model.h5")
resnet_model = tf.keras.models.load_model("models/resnet50_tomato.keras")
mobilenet_model = tf.keras.models.load_model("models/V2Net_tomato.keras")
knn_model = joblib.load("models/knn_tomato.pkl")

# -------------------- Class Names --------------------
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

# -------------------- Image Preprocessing --------------------
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------- Feature Extraction for KNN --------------------
feature_extractor = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_features(image_path):
    img = preprocess_image(image_path)
    features = feature_extractor.predict(img)
    return features

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    model_used = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        model_choice = request.form["model"]

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            if model_choice == "cnn":
                preds = cnn_model.predict(preprocess_image(filepath))
                model_used = "CNN"

            elif model_choice == "resnet":
                preds = resnet_model.predict(preprocess_image(filepath))
                model_used = "ResNet50"

            elif model_choice == "mobilenet":
                preds = mobilenet_model.predict(preprocess_image(filepath))
                model_used = "MobileNetV2"

            elif model_choice == "knn":
                features = extract_features(filepath)
                preds = knn_model.predict_proba(features)
                model_used = "KNN"

            class_index = np.argmax(preds)
            prediction = class_names[class_index]
            confidence = round(float(np.max(preds)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        model_used=model_used,
        filename=filename
    )

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
