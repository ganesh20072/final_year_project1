from pathlib import Path
import random

import cv2
import numpy as np

ROOT = Path("object_dataset_fixed/train")
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

OUTPUT_DIR = Path("robot_outputs/annotation_inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bowl",
    1: "Cup",
    2: "Fork",
    3: "Knife",
    4: "Plate",
    5: "Spoon",
}

random.seed(42)

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

random.shuffle(image_paths)

selected = []

# Prefer images containing multiple annotated objects.
for image_path in image_paths:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    lines = [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    if len(lines) >= 2:
        selected.append((image_path, label_path))

    if len(selected) == 12:
        break

if not selected:
    raise RuntimeError("No images with multiple annotations were found.")

tiles = []

for image_path, label_path in selected:
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if not parts:
            continue

        class_id = int(parts[0])
        coords = np.array(
            [float(value) for value in parts[1:]],
            dtype=np.float32,
        )

        if len(coords) < 6 or len(coords) % 2:
            continue

        points = coords.reshape(-1, 2)

        points[:, 0] *= width
        points[:, 1] *= height

        polygon = np.round(points).astype(np.int32)

        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=max(1, width // 300),
        )

        center = np.mean(polygon, axis=0).astype(int)

        name = CLASS_NAMES.get(class_id, f"Unknown {class_id}")

        cv2.putText(
            image,
            name,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, width / 1000),
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Resize each image for easier visual inspection.
    scale = min(320 / width, 240 / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = cv2.resize(image, (new_width, new_height))

    canvas = np.full((270, 340, 3), 255, dtype=np.uint8)

    x = (340 - new_width) // 2
    y = 22 + (240 - new_height) // 2

    canvas[y:y + new_height, x:x + new_width] = image

    filename = image_path.name[:38]

    cv2.putText(
        canvas,
        filename,
        (5, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    tiles.append(canvas)

if not tiles:
    raise RuntimeError("No annotation images could be visualized.")

columns = 3
rows = (len(tiles) + columns - 1) // columns

sheet = np.full(
    (rows * 270, columns * 340, 3),
    255,
    dtype=np.uint8,
)

for index, tile in enumerate(tiles):
    row = index // columns
    column = index % columns

    sheet[
        row * 270:(row + 1) * 270,
        column * 340:(column + 1) * 340,
    ] = tile

output_path = OUTPUT_DIR / "training_annotations.jpg"

if not cv2.imwrite(str(output_path), sheet):
    raise RuntimeError("Could not save annotation contact sheet.")

print("Images inspected:", len(tiles))
print("Saved:", output_path)
