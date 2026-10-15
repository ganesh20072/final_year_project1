from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = (
    "runs/segment/runs/table_robot/"
    "kitchenware_segmentation_v2/weights/best.pt"
)

CONFIDENCE_THRESHOLD = 0.40

model = YOLO(MODEL_PATH)


def detect_tableware(image_path, show=True):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return None

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]
    detected_objects = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detected_objects.append({
                "class": class_name,
                "confidence": round(confidence, 3)
            })

    print(f"\nImage: {Path(image_path).name}")

    if detected_objects:
        print("Tableware detected:")

        for obj in detected_objects:
            print(
                f"- {obj['class']}: "
                f"{obj['confidence'] * 100:.1f}% confidence"
            )
    else:
        print("No known tableware detected.")

    if show:
        annotated_image = result.plot()
        cv2.imshow("YOLO Tableware Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_objects


if __name__ == "__main__":
    image_path = input(
        "Enter the image path (example: test_images/pic1.jpg): "
    ).strip()

    detections = detect_tableware(image_path)

    if detections is not None:
        if detections:
            print("\nSAFETY STATUS: STOP — tableware detected.")
        else:
            print(
                "\nSAFETY STATUS: NO KNOWN TABLEWARE DETECTED. "
                "This does not yet mean cleaning is safe."
            )