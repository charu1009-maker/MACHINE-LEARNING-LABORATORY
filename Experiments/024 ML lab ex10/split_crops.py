import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("pothole_crops")
OUTPUT_DIR = Path("pothole_crops_split")

CLASSES = ["high", "low", "medium"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)


# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)


print("Creating leakage-safe crop dataset split...\n")


# ==========================================
# COLLECT ALL CROPS
# ==========================================

all_crops = []

for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name

    for image in source_folder.iterdir():

        if image.is_file():

            all_crops.append({
                "file": image,
                "class": class_name
            })


# ==========================================
# GROUP CROPS BY ORIGINAL IMAGE
# ==========================================

groups = {}

for item in all_crops:

    filename = item["file"].stem

    # Example:
    # img-500_1 -> img-500
    # img-500_2 -> img-500

    original_name = filename.rsplit("_", 1)[0]

    if original_name not in groups:
        groups[original_name] = []

    groups[original_name].append(item)


print("Total crops :", len(all_crops))
print("Original images/groups :", len(groups))
print()


# ==========================================
# SHUFFLE GROUPS
# ==========================================

group_names = list(groups.keys())

random.shuffle(group_names)

total_groups = len(group_names)

train_group_count = int(
    total_groups * TRAIN_RATIO
)

val_group_count = int(
    total_groups * VAL_RATIO
)

train_groups = group_names[
    :train_group_count
]

val_groups = group_names[
    train_group_count:
    train_group_count + val_group_count
]

test_groups = group_names[
    train_group_count + val_group_count:
]


# Convert to sets for fast lookup

train_groups = set(train_groups)
val_groups = set(val_groups)
test_groups = set(test_groups)


# ==========================================
# COPY FILES
# ==========================================

counts = {
    "train": {
        "high": 0,
        "low": 0,
        "medium": 0
    },
    "validation": {
        "high": 0,
        "low": 0,
        "medium": 0
    },
    "test": {
        "high": 0,
        "low": 0,
        "medium": 0
    }
}


for group_name, crops in groups.items():

    if group_name in train_groups:
        split = "train"

    elif group_name in val_groups:
        split = "validation"

    else:
        split = "test"


    for item in crops:

        source_file = item["file"]
        class_name = item["class"]

        destination = (
            OUTPUT_DIR
            / split
            / class_name
            / source_file.name
        )

        shutil.copy2(
            source_file,
            destination
        )

        counts[split][class_name] += 1


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("=========================================")
print("GROUPED DATASET SPLIT")
print("=========================================")

for class_name in CLASSES:

    total = (
        counts["train"][class_name]
        + counts["validation"][class_name]
        + counts["test"][class_name]
    )

    print(
        f"{class_name.upper():8} "
        f"Total: {total:4} | "
        f"Train: {counts['train'][class_name]:4} | "
        f"Validation: {counts['validation'][class_name]:4} | "
        f"Test: {counts['test'][class_name]:4}"
    )


print("\n=========================================")
print("Crop dataset split completed!")
print("=========================================")

print("\nImportant:")
print("Crops from the same original image")
print("are kept in the same dataset split.")
print("Therefore, train/test leakage is avoided.")