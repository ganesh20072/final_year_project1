"""
Cleaning decision engine for the AI table-cleaning robot.

This module evaluates perception results and returns a conservative
status. It does not control motors, pumps, or other hardware.
"""


def evaluate_cleaning_readiness(
    camera_ok=False,
    table_boundary_ok=False,
    detection_ok=False,
    detections=None,
    contamination_ok=False,
    contamination=None,
    manual_inspection_required=True,
):
    """
    Evaluate whether the table can proceed to a future cleaning stage.

    Important:
    This function never authorizes physical cleaning.
    A CLEAR result means only that the supplied checks passed;
    independent hardware and safety interlocks are still required.
    """

    detections = detections or []
    contamination = contamination or {}

    reasons = []

    if not camera_ok:
        reasons.append("Camera image unavailable or invalid.")

    if not table_boundary_ok:
        reasons.append("Table boundary is missing or unreliable.")

    if not detection_ok:
        reasons.append("Tableware detection failed or is unavailable.")

    if detections:
        detected_classes = sorted(
            {
                str(item.get("class", "Unknown"))
                for item in detections
            }
        )
        reasons.append(
            "Tableware detected: " + ", ".join(detected_classes) + "."
        )

    if not contamination_ok:
        reasons.append(
            "Contamination analysis is unavailable or unreliable."
        )

    # Colour-based estimates are experimental, so they cannot
    # independently establish that a table is clean.
    if contamination:
        reasons.append(
            "Colour-based contamination estimates require validation."
        )

    if manual_inspection_required:
        reasons.append(
            "Manual inspection is required before cleaning can be considered."
        )

    # Physical cleaning remains disabled in this development stage.
    return {
        "status": "STOP" if detections else "NOT CLEARED",
        "cleaning_authorized": False,
        "reasons": reasons,
        "contamination_estimates": contamination,
    }


if __name__ == "__main__":
    # Basic development test: tableware is present.
    result = evaluate_cleaning_readiness(
        camera_ok=True,
        table_boundary_ok=True,
        detection_ok=True,
        detections=[
            {"class": "Cup", "confidence": 0.95},
            {"class": "Plate", "confidence": 0.91},
        ],
        contamination_ok=True,
        contamination={
            "oil": 12.5,
            "water": 8.0,
            "food": 5.5,
        },
        manual_inspection_required=True,
    )

    print("CLEANING DECISION TEST")
    print("-" * 40)
    print("Status:", result["status"])
    print("Cleaning authorized:", result["cleaning_authorized"])

    for reason in result["reasons"]:
        print("-", reason)
