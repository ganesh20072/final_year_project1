import cv2
import streamlit as st
import numpy as np
from PIL import Image

from table_understanding import detect_table
from object_detection import (
    detect_objects,
    safety_decision
)
from contamination_detection import (
    analyze_contamination
)


st.set_page_config(
    page_title="AI Table Analysis",
    layout="wide"
)


st.title(
    "🤖 AI Restaurant Table Analysis"
)

st.write(
    "Table Understanding → "
    "Object Detection → "
    "Contamination Analysis"
)


uploaded_file = st.file_uploader(
    "Upload a table image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    pil_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(
        pil_image
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # =====================================
    # INPUT
    # =====================================

    st.subheader(
        "1. Input Image"
    )

    st.image(
        pil_image,
        use_container_width=True
    )

    # =====================================
    # TABLE UNDERSTANDING
    # =====================================

    st.subheader(
        "2. Table Understanding"
    )

    table_result = detect_table(
        image_bgr
    )

    table_image = (
        table_result["image"]
    )

    table_mask = (
        table_result["mask"]
    )

    st.image(
        cv2.cvtColor(
            table_image,
            cv2.COLOR_BGR2RGB
        ),
        caption="Detected Table Boundary",
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Table Shape",
            table_result["shape"]
        )

    with col2:

        st.metric(
            "Pixel Width",
            table_result["pixel_width"]
            if table_result["pixel_width"]
            else "N/A"
        )

    with col3:

        st.metric(
            "Pixel Height",
            table_result["pixel_height"]
            if table_result["pixel_height"]
            else "N/A"
        )

    st.info(
        "Physical dimensions require "
        "camera calibration/reference information. "
        "Current prototype displays pixel dimensions."
    )

    # =====================================
    # OBJECT DETECTION
    # =====================================

    st.subheader(
        "3. Object Detection"
    )

    object_image, detections = (
        detect_objects(image_bgr)
    )

    decision = safety_decision(
        detections
    )

    st.image(
        cv2.cvtColor(
            object_image,
            cv2.COLOR_BGR2RGB
        ),
        caption="Detected Objects",
        use_container_width=True
    )

    if detections:

        st.write(
            "### Detected Objects"
        )

        for item in detections:

            st.write(
                f"**{item['class']}** — "
                f"{item['confidence']:.2f}"
            )

    else:

        st.write(
            "No objects detected."
        )

    # =====================================
    # SAFETY STATUS
    # =====================================

    if decision["ready"]:

        st.success(
            "✅ TABLE READY FOR "
            "CONTAMINATION ANALYSIS"
        )

    else:

        st.error(
            "❌ TABLE NOT READY"
        )

        st.warning(
            decision["reason"]
        )

    # =====================================
    # CONTAMINATION
    # =====================================

    st.subheader(
        "4. Contamination Analysis"
    )

    contamination = (
        analyze_contamination(
            image_bgr,
            table_mask
        )
    )

    st.image(
        cv2.cvtColor(
            contamination["overlay"],
            cv2.COLOR_BGR2RGB
        ),
        caption="Contamination Visualization",
        use_container_width=True
    )

    p = contamination[
        "percentages"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Oil-like Region",
            f"{p['oil']}%"
        )

    with c2:

        st.metric(
            "Water-like Region",
            f"{p['water']}%"
        )

    with c3:

        st.metric(
            "Food-like Region",
            f"{p['food']}%"
        )

    # =====================================
    # FINAL RESULT
    # =====================================

    st.subheader(
        "5. Final Analysis"
    )

    if not decision["ready"]:

        st.error(
            "❌ TABLE NOT READY — "
            "REMOVE OBJECTS AND RESCAN"
        )

    else:

        st.success(
            "✅ TABLE CLEAR OF "
            "DETECTED BLOCKING OBJECTS"
        )

        st.info(
            "Contamination analysis completed."
        )
