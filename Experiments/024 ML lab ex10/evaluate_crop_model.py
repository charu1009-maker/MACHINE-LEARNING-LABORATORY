import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path("pothole_crops_split")
MODEL_PATH = "pothole_crop_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading crop model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!\n")

# ==========================================
# LOAD TEST DATASET
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:")
print(class_names)

# ==========================================
# PREDICTION
# ==========================================

print("\nEvaluating test dataset...\n")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# ==========================================
# ACCURACY
# ==========================================

accuracy = np.mean(
    np.array(y_true) == np.array(y_pred)
)

print("================================")
print("TEST RESULTS")
print("================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print("--------------------------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=2
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("Confusion Matrix:")
print("--------------------------------")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

print("\n================================")
print("Evaluation completed!")
print("================================")