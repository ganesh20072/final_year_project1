import cv2
import numpy as np
from pathlib import Path

IMAGE_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("table_outputs")

OUTPUT_FOLDER.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def detect_table(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    display = image.copy()
    points = []

    print(f"\nImage: {image_path.name}")
    print("Click the FOUR corners of the tabletop.")
    print("Click in this order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("Press R to reset or ESC to cancel.")

    def mouse_callback(event, x, y, flags, param):
        nonlocal display, points

        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)

            cv2.putText(
                display,
                str(len(points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if len(points) > 1:
                cv2.line(
                    display,
                    tuple(points[-2]),
                    tuple(points[-1]),
                    (0, 255, 0),
                    2
                )

            if len(points) == 4:
                cv2.line(
                    display,
                    tuple(points[-1]),
                    tuple(points[0]),
                    (0, 255, 0),
                    2
                )

    window = "Select Table Corners"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return

        if key in (ord("r"), ord("R")):
            points.clear()
            display = image.copy()

        if len(points) == 4:
            break

    cv2.destroyWindow(window)

    pts = np.array(points, dtype=np.float32)

    # Estimate the dimensions of the rectified tabletop
    top_width = np.linalg.norm(pts[1] - pts[0])
    bottom_width = np.linalg.norm(pts[2] - pts[3])
    left_height = np.linalg.norm(pts[3] - pts[0])
    right_height = np.linalg.norm(pts[2] - pts[1])

    output_width = max(1, int(max(top_width, bottom_width)))
    output_height = max(1, int(max(left_height, right_height)))

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(pts, destination)

    top_view = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height)
    )

    # Draw the selected tabletop boundary
    boundary = image.copy()

    cv2.polylines(
        boundary,
        [pts.astype(np.int32)],
        True,
        (0, 255, 0),
        4
    )

    cv2.imwrite(
        str(OUTPUT_FOLDER / f"{image_path.stem}_boundary.jpg"),
        boundary
    )

    cv2.imwrite(
        str(OUTPUT_FOLDER / f"{image_path.stem}_topview.jpg"),
        top_view
    )

    print("Table boundary selected successfully.")
    print(f"Approximate image width: {output_width} pixels")
    print(f"Approximate image height: {output_height} pixels")
    print("Saved results in table_outputs/")

    cv2.imshow("Selected Table Boundary", boundary)
    cv2.imshow("Perspective-Corrected Table", top_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if not IMAGE_FOLDER.exists():
        print("ERROR: test_images folder not found.")
        return

    images = sorted(
        p for p in IMAGE_FOLDER.iterdir()
        if p.suffix.lower() in EXTENSIONS
    )

    if not images:
        print("No images found in test_images/")
        return

    for image_path in images:
        detect_table(image_path)


if __name__ == "__main__":
    main()