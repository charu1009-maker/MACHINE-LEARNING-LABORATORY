import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("images")
ANNOTATION_DIR = Path("annotations")
OUTPUT_DIR = Path("pothole_crops")

LABEL_MAP = {
    "minor_pothole": "low",
    "medium_pothole": "medium",
    "major_pothole": "high"
}

# Create output folders
for class_name in ["low", "medium", "high"]:
    (OUTPUT_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

counts = {
    "low": 0,
    "medium": 0,
    "high": 0,
    "missing": 0
}

print("Creating pothole crops...\n")

for xml_file in ANNOTATION_DIR.glob("*.xml"):

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image filename
        filename_tag = root.find("filename")

        if filename_tag is None:
            print("Filename not found:", xml_file.name)
            counts["missing"] += 1
            continue

        image_name = filename_tag.text.strip()

        # Find image
        image_path = IMAGE_DIR / image_name

        # Handle UUID-prefixed filenames
        if not image_path.exists():

            base_name = Path(image_name).stem

            if "-" in base_name:
                parts = base_name.split("-")

                if len(parts) >= 3:

                    simple_name = (
                        parts[-2] + "-" + parts[-1]
                    )

                    for ext in [".jpg", ".jpeg", ".png"]:

                        possible = (
                            IMAGE_DIR /
                            (simple_name + ext)
                        )

                        if possible.exists():
                            image_path = possible
                            break

        # Image still not found
        if not image_path.exists():
            print("Image not found:", image_name)
            counts["missing"] += 1
            continue

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Count potholes in this image
        pothole_number = 0

        # Process every pothole
        for obj in root.findall("object"):

            name_tag = obj.find("name")

            if name_tag is None:
                continue

            original_label = (
                name_tag.text.strip().lower()
            )

            if original_label not in LABEL_MAP:
                continue

            severity = LABEL_MAP[original_label]

            # Get bounding box
            bbox = obj.find("bndbox")

            if bbox is None:
                continue

            xmin = int(
                float(bbox.find("xmin").text)
            )

            ymin = int(
                float(bbox.find("ymin").text)
            )

            xmax = int(
                float(bbox.find("xmax").text)
            )

            ymax = int(
                float(bbox.find("ymax").text)
            )

            # Keep coordinates inside image
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)

            # Check valid bounding box
            if xmax <= xmin or ymax <= ymin:
                continue

            # Crop pothole
            crop = image.crop(
                (xmin, ymin, xmax, ymax)
            )

            pothole_number += 1

            # Output filename
            output_name = (
                f"{image_path.stem}_"
                f"{pothole_number}.jpg"
            )

            output_path = (
                OUTPUT_DIR /
                severity /
                output_name
            )

            # Save crop
            crop.save(
                output_path,
                quality=95
            )

            counts[severity] += 1

    except Exception as e:

        print(
            "Error processing:",
            xml_file.name
        )

        print("Reason:", e)

        counts["missing"] += 1


print("\n================================")
print("Pothole crop creation completed!")
print("================================")

print("LOW       :", counts["low"])
print("MEDIUM    :", counts["medium"])
print("HIGH      :", counts["high"])
print("MISSING   :", counts["missing"])

print("================================")