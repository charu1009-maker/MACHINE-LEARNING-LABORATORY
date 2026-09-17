from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = Flask(__name__)

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "pothole_model.keras"
IMAGE_SIZE = (224, 224)

CLASSES = ["high", "low", "medium"]

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# LOAD MODEL
# =========================

print("Loading pothole severity model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template("index.html")


# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    # Save uploaded image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Load image
    image = tf.keras.utils.load_img(
        filepath,
        target_size=IMAGE_SIZE
    )

    # Convert image to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    probabilities = prediction[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASSES[predicted_index]

    confidence = probabilities[predicted_index] * 100

    # Individual probabilities
    high_probability = probabilities[0] * 100
    low_probability = probabilities[1] * 100
    medium_probability = probabilities[2] * 100

    return render_template(
        "index.html",
        prediction=predicted_class.upper(),
        confidence=f"{confidence:.2f}",
        high=f"{high_probability:.2f}",
        low=f"{low_probability:.2f}",
        medium=f"{medium_probability:.2f}",
        image_path=filepath
    )


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":
    app.run(debug=True)