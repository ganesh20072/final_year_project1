import cv2
import numpy as np


def analyze_contamination(
    image,
    table_mask=None
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # -----------------------------
    # WATER-LIKE REGION
    # -----------------------------

    water_mask = (
        (s > 15) &
        (s < 150) &
        (v > 100)
    ).astype(np.uint8) * 255

    # -----------------------------
    # OIL-LIKE REGION
    # -----------------------------

    oil_mask = (
        (s > 20) &
        (s < 180) &
        (v > 40) &
        (v < 210)
    ).astype(np.uint8) * 255

    # -----------------------------
    # FOOD-LIKE REGION
    # -----------------------------

    food_mask = (
        (h > 5) &
        (h < 35) &
        (s > 70) &
        (v > 60)
    ).astype(np.uint8) * 255

    # -----------------------------
    # TABLE REGION ONLY
    # -----------------------------

    if table_mask is not None:

        oil_mask = cv2.bitwise_and(
            oil_mask,
            table_mask
        )

        water_mask = cv2.bitwise_and(
            water_mask,
            table_mask
        )

        food_mask = cv2.bitwise_and(
            food_mask,
            table_mask
        )

        table_pixels = np.sum(
            table_mask > 0
        )

    else:

        table_pixels = (
            image.shape[0] *
            image.shape[1]
        )

    # -----------------------------
    # CLEAN MASKS
    # -----------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    oil_mask = cv2.morphologyEx(
        oil_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    oil_mask = cv2.morphologyEx(
        oil_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    water_mask = cv2.morphologyEx(
        water_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    food_mask = cv2.morphologyEx(
        food_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # -----------------------------
    # PERCENTAGE
    # -----------------------------

    def percentage(mask):

        pixels = np.sum(mask > 0)

        return round(
            100 * pixels /
            max(table_pixels, 1),
            2
        )

    contamination = {

        "oil": percentage(oil_mask),

        "water": percentage(water_mask),

        "food": percentage(food_mask)
    }

    # -----------------------------
    # HEATMAP
    # -----------------------------

    combined = cv2.addWeighted(
        oil_mask,
        0.5,
        water_mask,
        0.3,
        0
    )

    combined = cv2.addWeighted(
        combined,
        1.0,
        food_mask,
        0.3,
        0
    )

    heatmap = cv2.applyColorMap(
        combined,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap,
        0.35,
        0
    )

    return {

        "oil_mask": oil_mask,

        "water_mask": water_mask,

        "food_mask": food_mask,

        "heatmap": heatmap,

        "overlay": overlay,

        "percentages": contamination
    }

