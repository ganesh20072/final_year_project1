from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from contamination_detection import analyze_contamination


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "table_robot"
    / "kitchenware_segmentation_v4_grouped"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = BASE_DIR / "robot_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIDENCE = 0.15
DEFAULT_IMAGE_SIZE = 960

# Prototype only. These settings do not establish robot safety.


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TableBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DASHBOARD STYLE
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1120;
        }

        [data-testid="stSidebar"] {
            background-color: #111a2e;
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin-top: 4px;
            margin-bottom: 22px;
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-top: 12px;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.85rem;
        }

        div[data-testid="stMetric"] {
            background: #131e33;
            border: 1px solid #263550;
            padding: 16px;
            border-radius: 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: #a8b6ca;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        .stButton button {
            border-radius: 9px;
            min-height: 42px;
            font-weight: 600;
        }

        hr {
            border-color: #263550;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{model_path}"
        )

    return YOLO(str(model_path))


# ============================================================
# CAMERA
# ============================================================

def capture_camera_image(camera_index):
    """
    Capture one frame from a camera accessible through OpenCV.

    This is suitable for a supported USB webcam or camera
    exposed as a video device. Raspberry Pi camera access
    may require a separate camera backend.
    """

    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check camera permissions, connection, and index."
        )

    try:
        # Allow the camera a moment to initialize.
        for _ in range(3):
            camera.read()

        success, frame = camera.read()

        if not success or frame is None:
            raise RuntimeError(
                "Camera opened, but no image could be captured."
            )

        return frame

    finally:
        camera.release()


# ============================================================
# IMAGE INPUT
# ============================================================

def get_input_image(source_mode, camera_index):
    """
    Return a BGR image.

    Camera mode captures directly from the connected camera.
    Test-image mode is provided for debugging.
    """

    if source_mode == "Connected camera":

        frame = capture_camera_image(camera_index)

        return frame, "Connected camera"

    uploaded_file = st.session_state.get("uploaded_test_image")

    if uploaded_file is None:
        return None, None

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        rgb = np.array(pil_image)

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        return bgr, uploaded_file.name

    except Exception as error:
        raise RuntimeError(
            f"Could not read the selected image: {error}"
        ) from error


# ============================================================
# TABLE BOUNDARY
# ============================================================

def select_table_region(image):
    """
    Use the full captured image as the provisional table region.

    IMPORTANT:
    This is NOT automatic table-boundary detection.

    For controlled testing, the camera should be positioned
    so the tabletop fills the image and objects outside the
    table are minimized. A validated automatic boundary
    detector can replace this function later.
    """

    height, width = image.shape[:2]

    if width < 50 or height < 50:
        raise ValueError("The captured image is too small.")

    # Provisional region: entire image.
    # Do not interpret this as a verified tabletop boundary.
    table = image.copy()

    table_mask = np.full(
        (height, width),
        255,
        dtype=np.uint8,
    )

    return table, table_mask


# ============================================================
# KITCHENWARE DETECTION
# ============================================================

def detect_kitchenware(model, table, confidence, image_size):
    results = model.predict(
        source=table,
        imgsz=image_size,
        conf=confidence,
        verbose=False,
    )

    if not results:
        raise RuntimeError("YOLO returned no prediction results.")

    result = results[0]

    annotated = result.plot()

    height, width = table.shape[:2]

    object_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    detections = []

    boxes = result.boxes
    masks = result.masks

    if boxes is not None:

        for index, box in enumerate(boxes):

            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            class_name = model.names.get(
                class_id,
                f"Unknown class {class_id}",
            )

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": score,
                    "box": xyxy,
                }
            )

            if masks is not None and index < len(masks.data):

                mask = (
                    masks.data[index]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                object_mask[mask > 0.5] = 255

    return annotated, detections, object_mask


# ============================================================
# COMPLETE PERCEPTION ANALYSIS
# ============================================================

def analyze_image(model, image, confidence, image_size):
    """
    Run kitchenware detection and experimental contamination
    analysis on one image.

    No robot actuators are controlled by this function.
    """

    table, table_mask = select_table_region(image)

    annotated, detections, object_mask = detect_kitchenware(
        model,
        table,
        confidence,
        image_size,
    )

    exposed_table_mask = cv2.bitwise_and(
        table_mask,
        cv2.bitwise_not(object_mask),
    )

    contamination = analyze_contamination(
        table,
        table_mask=exposed_table_mask,
    )

    # Conservative prototype status.
    #
    # No detections must never be treated as proof that a
    # table is clear. The current system has no independently
    # validated boundary or unknown-object safety detector.

    if detections:
        status = "STOP"
        reason = (
            "Known kitchenware was detected. "
            "The table is not cleared for cleaning."
        )

    else:
        status = "NOT CLEARED"
        reason = (
            "No known kitchenware was detected. "
            "The system cannot confirm that the table is safe."
        )

    return {
        "table": table,
        "table_mask": table_mask,
        "annotated": annotated,
        "detections": detections,
        "object_mask": object_mask,
        "exposed_table_mask": exposed_table_mask,
        "contamination": contamination,
        "status": status,
        "reason": reason,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    images = {
        f"{timestamp}_table.jpg": results["table"],
        f"{timestamp}_detections.jpg": results["annotated"],
        f"{timestamp}_object_mask.jpg": results["object_mask"],
        f"{timestamp}_exposed_mask.jpg": results[
            "exposed_table_mask"
        ],
        f"{timestamp}_contamination_overlay.jpg": results[
            "contamination"
        ]["overlay"],
        f"{timestamp}_contamination_heatmap.jpg": results[
            "contamination"
        ]["heatmap"],
    }

    saved_paths = []

    for filename, image in images.items():

        path = OUTPUT_DIR / filename

        if cv2.imwrite(str(path), image):
            saved_paths.append(path)

    return saved_paths


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 TableBot AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Intelligent Table Monitoring and Cleaning Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "RESEARCH PROTOTYPE — Camera detection and contamination "
    "analysis are not validated safety systems. This dashboard "
    "does not authorize or operate the cleaning hardware."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Control Panel")

    st.caption("Configure the image source and AI inference.")

    source_mode = st.radio(
        "Image source",
        [
            "Connected camera",
            "Test image",
        ],
        index=0,
    )

    camera_index = st.selectbox(
        "Camera device index",
        options=[0, 1, 2, 3],
        index=0,
        help=(
            "Try index 0 first. If multiple cameras are connected, "
            "test another index."
        ),
        disabled=(source_mode != "Connected camera"),
    )

    st.divider()

    confidence = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.80,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help=(
            "Lower thresholds can reveal more detections, "
            "but may also increase false positives."
        ),
    )

    image_size = st.select_slider(
        "YOLO inference resolution",
        options=[640, 800, 960, 1280],
        value=DEFAULT_IMAGE_SIZE,
    )

    st.caption(
        "960 px and confidence 0.15 are experimental settings, "
        "not validated safety thresholds."
    )

    st.divider()

    st.markdown("### Model information")

    st.write("**Architecture:** YOLO segmentation")

    st.write("**Classes:** Bowl, Cup, Fork, Knife, Plate, Spoon")

    st.write("**Model version:** v4 grouped fine-tune")

    st.divider()

    if source_mode == "Test image":

        st.session_state.setdefault(
            "uploaded_test_image",
            None,
        )

        st.file_uploader(
            "Choose a test image",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploaded_test_image",
        )

        st.caption(
            "Test-image mode is for debugging only. "
            "Normal camera operation does not require uploads."
        )


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model(str(MODEL_PATH))

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.code(str(MODEL_PATH))
    st.stop()


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("AI Model", "Loaded")

col2.metric(
    "Input Mode",
    "Camera" if source_mode == "Connected camera" else "Test image",
)

col3.metric("YOLO Image Size", f"{image_size} px")

col4.metric(
    "Confidence",
    f"{confidence:.2f}",
)


# ============================================================
# CAPTURE AND ANALYZE
# ============================================================

st.divider()

st.subheader("📷 Camera and Analysis")

if source_mode == "Connected camera":

    st.info(
        "Click Capture and Analyze to capture a fresh image "
        "from the selected camera and run the AI pipeline. "
        "The captured image is processed automatically; no "
        "manual image upload is required."
    )

    capture_button = st.button(
        "📸 Capture and Analyze",
        type="primary",
        use_container_width=True,
    )

else:

    st.info(
        "Test-image mode is intended for development and debugging."
    )

    capture_button = st.button(
        "🔍 Analyze Test Image",
        type="primary",
        use_container_width=True,
    )


if capture_button:

    try:
        with st.spinner("Capturing image and running AI analysis..."):

            image, source_description = get_input_image(
                source_mode,
                camera_index,
            )

            if image is None:
                st.error(
                    "No test image selected. Choose an image "
                    "in the sidebar or switch to camera mode."
                )
                st.stop()

            results = analyze_image(
                model,
                image,
                confidence,
                image_size,
            )

            saved_paths = save_results(results)

            # Store results for rendering after the button run.
            st.session_state["latest_results"] = results
            st.session_state["latest_source"] = source_description
            st.session_state["latest_time"] = datetime.now().strftime(
                "%d %b %Y, %I:%M:%S %p"
            )
            st.session_state["latest_saved_paths"] = [
                str(path) for path in saved_paths
            ]

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.warning(
            "The image could not be fully analyzed. "
            "Do not authorize cleaning."
        )


# ============================================================
# DISPLAY LATEST RESULTS
# ============================================================

results = st.session_state.get("latest_results")

if results is None:

    st.info(
        "No analysis results yet. Capture an image from the "
        "connected camera to begin."
    )

else:

    st.divider()

    st.subheader("🛡️ Robot Status")

    if results["status"] == "STOP":
        st.error(f"STATUS: {results['status']}")
    else:
        st.warning(f"STATUS: {results['status']}")

    st.write(results["reason"])

    st.caption(
        f"Source: {st.session_state.get('latest_source', 'Unknown')} "
        f"| Last analysis: "
        f"{st.session_state.get('latest_time', 'Unknown')}"
    )

    # --------------------------------------------------------
    # RESULT SUMMARY
    # --------------------------------------------------------

    detections = results["detections"]
    percentages = results["contamination"]["percentages"]

    unique_classes = sorted(
        {item["class"] for item in detections}
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Objects Detected", len(detections))

    m2.metric("Unique Classes", len(unique_classes))

    m3.metric(
        "Oil-like Pixels",
        f"{percentages.get('oil', 0):.2f}%",
    )

    m4.metric(
        "Food-like Pixels",
        f"{percentages.get('food', 0):.2f}%",
    )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📷 Camera & Table",
            "🍽️ Kitchenware",
            "🧪 Contamination",
            "📁 Saved Results",
        ]
    )

    with tab1:

        left, right = st.columns(2)

        with left:
            st.markdown("#### Table Image")

            st.image(
                cv2.cvtColor(
                    results["table"],
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with right:
            st.markdown("#### Table Region Mask")

            st.image(
                results["table_mask"],
                clamp=True,
                use_container_width=True,
            )

        st.caption(
            "The current implementation uses the full image as "
            "the provisional table region. Automatic boundary "
            "detection is not implemented in this dashboard."
        )

    with tab2:

        st.markdown("#### Kitchenware Detection")

        st.image(
            cv2.cvtColor(
                results["annotated"],
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

        if detections:

            st.markdown("#### Detected Objects")

            rows = []

            for index, item in enumerate(detections, start=1):

                x1, y1, x2, y2 = item["box"]

                rows.append(
                    {
                        "No.": index,
                        "Class": item["class"],
                        "Confidence (%)": round(
                            item["confidence"] * 100,
                            2,
                        ),
                        "Bounding Box": (
                            f"{x1}, {y1}, {x2}, {y2}"
                        ),
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.warning(
                "No known kitchenware was detected. "
                "This does not prove that the table is empty."
            )

        if unique_classes:

            st.markdown("#### Recognized Classes")

            st.write(", ".join(unique_classes))

    with tab3:

        st.warning(
            "These visualizations use colour thresholds. "
            "They do not confirm the presence of oil, water, "
            "or food."
        )

        left, right = st.columns(2)

        with left:

            st.markdown("#### Contamination Overlay")

            st.image(
                cv2.cvtColor(
                    results["contamination"]["overlay"],
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with right:

            st.markdown("#### Contamination Heatmap")

            st.image(
                cv2.cvtColor(
                    results["contamination"]["heatmap"],
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        st.markdown("#### Experimental Pixel Estimates")

        contamination_rows = [
            {
                "Category": category.title(),
                "Flagged Pixels (%)": value,
                "Interpretation": "Unvalidated colour-based estimate",
            }
            for category, value in percentages.items()
        ]

        st.dataframe(
            contamination_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Categories may overlap. These percentages should not "
            "be interpreted as actual contamination coverage."
        )

    with tab4:

        st.markdown("#### Output Files")

        saved_paths = st.session_state.get(
            "latest_saved_paths",
            [],
        )

        if saved_paths:

            for path_string in saved_paths:

                path = Path(path_string)

                if path.exists():

                    with open(path, "rb") as file:

                        st.download_button(
                            label=f"Download {path.name}",
                            data=file.read(),
                            file_name=path.name,
                            mime="image/jpeg",
                            key=f"download_{path.name}",
                        )

        else:

            st.info("No saved result images are available.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
        TableBot AI · Final-Year Project Prototype<br>
        Camera inference · YOLO segmentation · Experimental
        contamination analysis<br><br>
        Physical cleaning remains disabled. This dashboard
        does not validate that a table is safe to clean.
    </div>
    """,
    unsafe_allow_html=True,
)