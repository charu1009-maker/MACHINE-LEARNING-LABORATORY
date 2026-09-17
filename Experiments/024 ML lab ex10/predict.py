import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "pothole_model.keras"
IMAGE_SIZE = (224, 224)

CLASSES = ["high", "low", "medium"]

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================
# GET IMAGE PATH
# =========================

image_path = input("\nEnter pothole image path: ")

# =========================
# LOAD IMAGE
# =========================

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# MobileNetV2 preprocessing
image_array = preprocess_input(image_array)

# =========================
# PREDICTION
# =========================

prediction = model.predict(
    image_array,
    verbose=0
)

probabilities = prediction[0]

predicted_index = np.argmax(probabilities)

predicted_class = CLASSES[predicted_index]

confidence = probabilities[predicted_index] * 100

# =========================
# DISPLAY RESULTS
# =========================

print("\n================================")
print("POTHOLE SEVERITY RESULT")
print("================================")

print(
    f"HIGH       : {probabilities[0] * 100:.2f}%"
)

print(
    f"LOW        : {probabilities[1] * 100:.2f}%"
)

print(
    f"MEDIUM     : {probabilities[2] * 100:.2f}%"
)

print("--------------------------------")

print(
    f"Final Severity : {predicted_class.upper()}"
)

print(
    f"Confidence     : {confidence:.2f}%"
)

print("================================")