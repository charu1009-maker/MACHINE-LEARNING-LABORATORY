import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_split")

CLASSES = ["high", "low", "medium"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

# =========================
# CREATE OUTPUT FOLDERS
# =========================

for split in ["train", "validation", "test"]:

    for class_name in CLASSES:

        folder = OUTPUT_DIR / split / class_name

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

# =========================
# SPLIT DATASET
# =========================

print("Creating stratified dataset split...\n")

for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name

    images = [
        file for file in source_folder.iterdir()
        if file.is_file()
    ]

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    # =========================
    # COPY FILES
    # =========================

    for image in train_images:

        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    for image in val_images:

        shutil.copy2(
            image,
            OUTPUT_DIR / "validation" / class_name / image.name
        )

    for image in test_images:

        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )

    # =========================
    # PRINT COUNTS
    # =========================

    print(
        f"{class_name.upper():8} "
        f"Total: {total:3} | "
        f"Train: {len(train_images):3} | "
        f"Validation: {len(val_images):3} | "
        f"Test: {len(test_images):3}"
    )

print("\n================================")
print("Dataset split completed!")
print("================================")