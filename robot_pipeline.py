from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from contamination_detection import analyze_contamination


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "runs/segment/runs/table_robot/"
    "kitchenware_segmentation_v4_grouped/weights/best.pt"
)

OUTPUT_DIR = Path("robot_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum confidence for a detection to be reported.
# Lower-confidence detections are still treated cautiously.
CONFIDENCE_THRESHOLD = 0.25

# Confidence below this value is reported as uncertain.
UNCERTAIN_CONFIDENCE = 0.60

# Minimum valid dimensions for a selected tabletop.
MIN_TABLE_WIDTH = 50
MIN_TABLE_HEIGHT = 50


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained tableware segmentation model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nERROR: YOLO model not found:\n{MODEL_PATH}\n"
            "Check the model path before running the pipeline."
        )

    model = YOLO(str(MODEL_PATH))

    print("YOLO tableware model loaded successfully.")

    return model


# ============================================================
# SELECT TABLETOP CORNERS
# ============================================================

def select_table(image):
    """
    Manually select the four tabletop corners.

    Click in this order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left

    Press ESC to cancel.
    """

    points = []
    display = image.copy()

    window_name = "Select Table - Click 4 Corners"

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(points) >= 4:
            return

        points.append((x, y))

        cv2.circle(
            display,
            (x, y),
            6,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            display,
            str(len(points)),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        if len(points) > 1:
            cv2.line(
                display,
                points[-2],
                points[-1],
                (0, 255, 0),
                2,
            )

        if len(points) == 4:
            cv2.line(
                display,
                points[3],
                points[0],
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    cv2.imshow(window_name, display)

    print("\nSelect the four tabletop corners:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("Press ESC to cancel.")

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                points.clear()
                break

            if len(points) == 4:
                break

    finally:
        cv2.destroyWindow(window_name)

    if len(points) != 4:
        print("Table selection cancelled.")
        return None

    return np.array(points, dtype=np.float32)


# ============================================================
# VALIDATE TABLE CORNERS
# ============================================================

def validate_table_corners(points):
    """Reject invalid or degenerate tabletop selections."""

    if points is None or points.shape != (4, 2):
        return False

    if not np.all(np.isfinite(points)):
        return False

    contour = points.reshape((-1, 1, 2))

    area = abs(cv2.contourArea(contour))

    if area < MIN_TABLE_WIDTH * MIN_TABLE_HEIGHT:
        return False

    return True


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def crop_table(image, points):
    """
    Transform the selected tabletop into a rectangular image.

    The returned image is perspective-corrected.
    """

    top_left, top_right, bottom_right, bottom_left = points

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    if width < MIN_TABLE_WIDTH or height < MIN_TABLE_HEIGHT:
        return None

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination,
    )

    corrected_table = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
    )

    return corrected_table


# ============================================================
# CREATE TABLETOP MASK
# ============================================================

def create_table_mask(table_image):
    """
    Create a mask for the perspective-corrected tabletop.

    White = selected tabletop region.
    Black = outside the selected tabletop.

    IMPORTANT:
    Because the perspective-corrected image is rectangular,
    this mask covers the entire corrected image.

    It does not remove plates, cups, cutlery, shadows,
    or other objects lying on the tabletop.
    """

    height, width = table_image.shape[:2]

    if height == 0 or width == 0:
        return None

    table_mask = np.full(
        (height, width),
        255,
        dtype=np.uint8,
    )

    return table_mask


# ============================================================
# TABLEWARE DETECTION
# ============================================================

def detect_tableware(model, table_image):
    """
    Detect known tableware using the trained YOLO model.

    Returns:
        annotated_image
        detections
        segmentation_mask
    """

    results = model.predict(
        source=table_image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    if not results:
        raise RuntimeError(
            "YOLO did not return a prediction result."
        )

    result = results[0]

    annotated_image = result.plot()

    detections = []

    height, width = table_image.shape[:2]

    # Union of detected segmentation masks.
    # This can later help exclude detected tableware from
    # contamination analysis.
    object_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    boxes = result.boxes
    masks = result.masks

    if boxes is not None:
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = model.names.get(
                class_id,
                f"Unknown class {class_id}",
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                }
            )

            # Use segmentation masks when the model supplies them.
            if masks is not None and index < len(masks.data):
                mask = masks.data[index].cpu().numpy()

                resized_mask = cv2.resize(
                    mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                object_mask[resized_mask > 0.5] = 255

    return annotated_image, detections, object_mask


# ============================================================
# SAVE IMAGE SAFELY
# ============================================================

def save_image(filename, image):
    """Save an output image and report failures."""

    if image is None:
        print(f"WARNING: Could not save {filename}: image is empty.")
        return False

    output_path = OUTPUT_DIR / filename

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        print(f"WARNING: Failed to save {output_path}")
        return False

    return True


# ============================================================
# MAIN ROBOT PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("AI TABLE-CLEANING ROBOT")
    print("Table Understanding + Object Detection")
    print("+ Experimental Contamination Analysis")
    print("=" * 60)

    # --------------------------------------------------------
    # Load the model
    # --------------------------------------------------------

    try:
        model = load_model()

    except Exception as error:
        print(f"\nSTOP: {error}")
        print("The robot must not start cleaning.")
        return

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_path = input(
        "\nEnter image path "
        "(example: test_images/tb5.jpeg): "
    ).strip()

    if not image_path:
        print("STOP: No image path provided.")
        return

    image = cv2.imread(image_path)

    if image is None:
        print(f"STOP: Could not load image: {image_path}")
        print("Check the path and image format.")
        return

    print(f"\nImage loaded: {image_path}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")

    # --------------------------------------------------------
    # Select table boundaries
    # --------------------------------------------------------

    points = select_table(image)

    if points is None:
        print("\nSAFETY STATUS: STOP")
        print("The tabletop boundary was not selected.")
        return

    if not validate_table_corners(points):
        print("\nSAFETY STATUS: STOP")
        print("The selected tabletop corners are invalid.")
        return

    # --------------------------------------------------------
    # Correct perspective
    # --------------------------------------------------------

    table = crop_table(
        image,
        points,
    )

    if table is None:
        print("\nSAFETY STATUS: STOP")
        print("Could not create a valid tabletop image.")
        return

    save_image(
        "table_crop.jpg",
        table,
    )

    print("\nTable perspective correction completed.")
    print(f"Corrected dimensions: {table.shape[1]} x {table.shape[0]}")

    # --------------------------------------------------------
    # Create tabletop mask
    # --------------------------------------------------------

    table_mask = create_table_mask(table)

    if table_mask is None:
        print("\nSAFETY STATUS: STOP")
        print("Could not create a tabletop mask.")
        return

    save_image(
        "table_mask.jpg",
        table_mask,
    )

    # --------------------------------------------------------
    # Detect tableware
    # --------------------------------------------------------

    try:
        annotated, detections, object_mask = detect_tableware(
            model,
            table,
        )

    except Exception as error:
        print(f"\nERROR during object detection: {error}")
        print("\nSAFETY STATUS: STOP")
        print("Object detection failed. Cleaning is prohibited.")
        return

    save_image(
        "tableware_detections.jpg",
        annotated,
    )

    save_image(
        "detected_objects_mask.jpg",
        object_mask,
    )

    # --------------------------------------------------------
    # Exclude detected object pixels from contamination analysis
    # --------------------------------------------------------

    # Remove the pixels covered by detected segmentation masks.
    # This is only an approximation: missed objects, incomplete
    # masks, and segmentation errors may still remain.

    exposed_table_mask = cv2.bitwise_and(
        table_mask,
        cv2.bitwise_not(object_mask),
    )

    save_image(
        "exposed_table_mask.jpg",
        exposed_table_mask,
    )

    # --------------------------------------------------------
    # Analyze possible contamination
    # --------------------------------------------------------

    try:
        contamination = analyze_contamination(
            table,
            table_mask=exposed_table_mask,
        )

    except Exception as error:
        print(f"\nERROR during contamination analysis: {error}")
        print("\nSAFETY STATUS: STOP")
        print("Contamination analysis failed.")
        return

    percentages = contamination.get(
        "percentages",
        {},
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTAL COLOUR-BASED CONTAMINATION ESTIMATES")
    print("=" * 60)

    print(
        f"Oil-like pixels: "
        f"{percentages.get('oil', 'N/A')}%"
    )

    print(
        f"Water-like pixels: "
        f"{percentages.get('water', 'N/A')}%"
    )

    print(
        f"Food-like pixels: "
        f"{percentages.get('food', 'N/A')}%"
    )

    print(
        "\nWARNING: These values are colour-based estimates, "
        "not verified measurements of actual contamination."
    )

    # --------------------------------------------------------
    # Save contamination results
    # --------------------------------------------------------

    save_image(
        "contamination_overlay.jpg",
        contamination.get("overlay"),
    )

    save_image(
        "contamination_heatmap.jpg",
        contamination.get("heatmap"),
    )

    save_image(
        "oil_mask.jpg",
        contamination.get("oil_mask"),
    )

    save_image(
        "water_mask.jpg",
        contamination.get("water_mask"),
    )

    save_image(
        "food_mask.jpg",
        contamination.get("food_mask"),
    )

    # --------------------------------------------------------
    # Report detected objects
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TABLEWARE DETECTION RESULTS")
    print("=" * 60)

    if detections:
        for item in detections:
            confidence = item["confidence"] * 100
            class_name = item["class"]

            if item["confidence"] < UNCERTAIN_CONFIDENCE:
                print(
                    f"- Possible {class_name}: "
                    f"{confidence:.1f}% confidence "
                    "(uncertain)"
                )

            else:
                print(
                    f"- {class_name}: "
                    f"{confidence:.1f}% confidence"
                )

    else:
        print("No known tableware detected.")

    # --------------------------------------------------------
    # Conservative safety decision
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if detections:
        print("SAFETY STATUS: STOP")
        print(
            "Tableware or possible tableware detected. "
            "Remove objects and inspect the tabletop."
        )

    else:
        print("SAFETY STATUS: NOT CLEARED")
        print(
            "No known tableware was detected, but this does "
            "not prove that the tabletop is safe."
        )

        print(
            "Additional checks for unknown objects, "
            "contamination, boundary reliability, and "
            "hardware safety are required."
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    try:
        cv2.imshow(
            "Corrected Table",
            table,
        )

        cv2.imshow(
            "Tableware Detection",
            annotated,
        )

        cv2.imshow(
            "Table Mask",
            table_mask,
        )

        cv2.imshow(
            "Detected Objects Mask",
            object_mask,
        )

        cv2.imshow(
            "Exposed Table Mask",
            exposed_table_mask,
        )

        cv2.imshow(
            "Contamination Overlay",
            contamination["overlay"],
        )

        cv2.imshow(
            "Contamination Heatmap",
            contamination["heatmap"],
        )

        cv2.waitKey(0)

    except cv2.error as error:
        print(f"\nWARNING: Could not display an image window: {error}")

    finally:
        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")
    print("\nPipeline execution completed.")
    print("Cleaning remains disabled pending safety verification.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()