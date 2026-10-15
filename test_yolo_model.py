from ultralytics import YOLO
from pathlib import Path

# Load your existing model
model = YOLO("yolo26n.pt")

print("\nYOLO26n MODEL TEST")
print("-" * 40)

# Display relevant classes
target_classes = [
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "dining table",
    "banana",
    "apple",
    "sandwich",
    "pizza",
]

available_classes = set(model.names.values())

for name in target_classes:
    status = "AVAILABLE" if name in available_classes else "NOT AVAILABLE"
    print(f"{name:15} : {status}")

# Check for test images
image_dir = Path("test_images")

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

images = [
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
] if image_dir.exists() else []

print("\nTest images found:", len(images))

if images:
    for image_path in images:
        print(f"\nTesting: {image_path}")

        results = model.predict(
            source=str(image_path),
            conf=0.25,
            save=True,
            verbose=False
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            print(
                f"Detected: {class_name:15} "
                f"Confidence: {confidence:.2f}"
            )

    print("\nAnnotated images saved by Ultralytics.")
else:
    print(
        "No test images yet. Add some table images "
        "to the test_images folder and run again."
    )