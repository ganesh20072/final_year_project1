from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from contamination_detection import analyze_contamination
from cleaning_decision import evaluate_cleaning_readiness


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "segment" / "runs" / "table_robot"
    / "kitchenware_segmentation_v4_grouped" / "weights" / "best.pt"
)

OUTPUT_DIR = BASE_DIR / "robot_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIDENCE = 0.15
DEFAULT_IMAGE_SIZE = 960

st.set_page_config(
    page_title="TableBot AI",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background-color: #0b1120;}
[data-testid="stSidebar"] {background-color: #111a2e;}
div[data-testid="stMetric"] {
    background: #131e33;
    border: 1px solid #263550;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(str(path))


# ============================================================
# IMAGE INPUT
# ============================================================

def capture_camera_image(camera_index):
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Check the camera connection and permissions."
        )

    try:
        frame = None

        for _ in range(5):
            success, candidate = camera.read()

            if success and candidate is not None:
                frame = candidate

        if frame is None:
            raise RuntimeError("Camera opened but no image was captured.")

        return frame

    finally:
        camera.release()


def get_input_image(source_mode, camera_index):
    if source_mode == "Connected camera":
        return capture_camera_image(camera_index), "Connected camera"

    uploaded = st.session_state.get("uploaded_test_image")

    if uploaded is None:
        return None, None

    uploaded.seek(0)

    rgb = np.array(Image.open(uploaded).convert("RGB"))

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return image, uploaded.name


# ============================================================
# TABLE REGION
# ============================================================

def select_table_region(image):
    """
    Temporary implementation: treats the entire image as the table.

    Automatic table-boundary detection is not implemented yet.
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid input image.")

    height, width = image.shape[:2]

    if width < 50 or height < 50:
        raise ValueError("Image is too small.")

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

def detect_kitchenware(model, image, confidence, image_size):
    predictions = model.predict(
        source=image,
        imgsz=image_size,
        conf=confidence,
        verbose=False,
    )

    if not predictions:
        raise RuntimeError("YOLO returned no prediction results.")

    result = predictions[0]
    annotated = result.plot()

    height, width = image.shape[:2]

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

            if isinstance(model.names, dict):
                class_name = model.names.get(
                    class_id,
                    f"Unknown {class_id}",
                )
            else:
                class_name = model.names[class_id]

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

            detections.append({
                "class": class_name,
                "confidence": score,
                "box": coordinates,
            })

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
# COMPLETE AI PIPELINE + DECISION ENGINE
# ============================================================

def analyze_image(model, image, confidence, image_size):

    # 1. Obtain the provisional table region.
    table, table_mask = select_table_region(image)

    # Not verified: automatic boundary detection is not implemented.
    table_boundary_verified = False

    # 2. Detect kitchenware.
    annotated, detections, object_mask = detect_kitchenware(
        model,
        table,
        confidence,
        image_size,
    )

    # 3. Exclude detected kitchenware from contamination analysis.
    exposed_table_mask = cv2.bitwise_and(
        table_mask,
        cv2.bitwise_not(object_mask),
    )

    # 4. Run experimental colour-based contamination analysis.
    contamination = analyze_contamination(
        table,
        table_mask=exposed_table_mask,
    )

    if not isinstance(contamination, dict):
        raise RuntimeError("Unexpected contamination analysis result.")

    percentages = contamination.get("percentages", {})

    estimates = {
        str(category): float(value)
        for category, value in percentages.items()
    }

    # 5. Evaluate the result using cleaning_decision.py.
    decision = evaluate_cleaning_readiness(
        camera_ok=True,
        table_boundary_ok=table_boundary_verified,
        detection_ok=True,
        detections=detections,

        # Colour thresholds are experimental, not validated.
        contamination_ok=False,
        contamination=estimates,

        # Manual inspection remains mandatory.
        manual_inspection_required=True,
    )

    return {
        "table": table,
        "table_mask": table_mask,
        "annotated": annotated,
        "detections": detections,
        "object_mask": object_mask,
        "exposed_table_mask": exposed_table_mask,
        "contamination": contamination,
        "decision": decision,
        "status": decision["status"],
        "reason": " ".join(decision.get("reasons", [])),
        "table_boundary_verified": table_boundary_verified,

        # Never authorize physical cleaning in this prototype.
        "cleaning_authorized": False,
    }


# ============================================================
# SAVE OUTPUT IMAGES
# ============================================================

def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    images = {
        "table": results["table"],
        "detections": results["annotated"],
        "object_mask": results["object_mask"],
        "exposed_mask": results["exposed_table_mask"],
        "contamination_overlay": results["contamination"]["overlay"],
        "contamination_heatmap": results["contamination"]["heatmap"],
    }

    saved = []

    for name, image in images.items():
        path = OUTPUT_DIR / f"{timestamp}_{name}.jpg"

        if cv2.imwrite(str(path), image):
            saved.append(str(path))

    return saved


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("🤖 TableBot AI")

st.markdown(
    "Intelligent Table Monitoring and Cleaning Research Prototype"
)

st.warning(
    "Research prototype only. Detection and contamination analysis "
    "are not validated safety systems. Physical cleaning remains disabled."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Control Panel")

    source_mode = st.radio(
        "Image source",
        ["Connected camera", "Test image"],
    )

    camera_index = st.selectbox(
        "Camera index",
        [0, 1, 2, 3],
        disabled=source_mode != "Connected camera",
    )

    confidence = st.slider(
        "Detection confidence",
        0.05,
        0.80,
        DEFAULT_CONFIDENCE,
        0.05,
    )

    image_size = st.select_slider(
        "YOLO inference resolution",
        [640, 800, 960, 1280],
        value=DEFAULT_IMAGE_SIZE,
    )

    st.caption(
        "Confidence and image-size settings are experimental, "
        "not validated safety thresholds."
    )

    if source_mode == "Test image":
        st.file_uploader(
            "Choose a test image",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploaded_test_image",
        )

    st.divider()

    st.write("**Model:** YOLO segmentation")
    st.write("**Classes:** Bowl, Cup, Fork, Knife, Plate, Spoon")
    st.write("**Version:** v4 grouped fine-tune")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model(MODEL_PATH)

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.code(str(MODEL_PATH))
    st.stop()


# ============================================================
# MODEL METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric("AI Model", "Loaded")
c2.metric("Input", source_mode)
c3.metric("Confidence", f"{confidence:.2f}")


# ============================================================
# CAPTURE AND ANALYZE
# ============================================================

st.divider()
st.subheader("📷 Camera and Analysis")

if source_mode == "Connected camera":
    st.info(
        "Capture a fresh image from the selected camera. "
        "Raspberry Pi Camera Module 3 may require Picamera2 when "
        "this dashboard runs directly on the Raspberry Pi."
    )

    run_analysis = st.button(
        "📸 Capture and Analyze",
        type="primary",
    )

else:
    st.info("Test-image mode is for development and debugging.")

    run_analysis = st.button(
        "🔍 Analyze Test Image",
        type="primary",
    )


if run_analysis:

    try:
        with st.spinner("Running the AI pipeline..."):

            image, source = get_input_image(
                source_mode,
                camera_index,
            )

            if image is None:
                st.error("Please select a test image first.")
                st.stop()

            results = analyze_image(
                model,
                image,
                confidence,
                image_size,
            )

            saved = save_results(results)

            st.session_state["latest_results"] = results
            st.session_state["latest_source"] = source
            st.session_state["latest_time"] = datetime.now().strftime(
                "%d %b %Y, %I:%M:%S %p"
            )
            st.session_state["latest_saved_paths"] = saved

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.warning("Do not authorize cleaning if analysis fails.")


# ============================================================
# DISPLAY RESULTS
# ============================================================

results = st.session_state.get("latest_results")

if results is None:

    st.info("Capture an image or select a test image to begin.")

else:

    st.divider()
    st.subheader("🛡️ Robot Status")

    if results["status"] == "STOP":
        st.error("STATUS: STOP")
    else:
        st.warning("STATUS: NOT CLEARED")

    st.write(results["reason"])

    st.caption(
        f"Source: {st.session_state.get('latest_source', 'Unknown')} | "
        f"Last analysis: {st.session_state.get('latest_time', 'Unknown')}"
    )

    st.subheader("Decision Engine Evaluation")

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Table Boundary Verified",
        "Yes" if results["table_boundary_verified"] else "No",
    )

    d2.metric(
        "Objects Detected",
        len(results["detections"]),
    )

    d3.metric(
        "Cleaning Authorized",
        "YES" if results["cleaning_authorized"] else "NO",
    )

    st.markdown("#### Reasons for the Decision")

    for reason in results["decision"].get("reasons", []):
        st.write(f"• {reason}")

    st.info(
        "Cleaning remains disabled. No detections does not prove "
        "that a table is empty or safe to clean."
    )

    detections = results["detections"]
    contamination = results["contamination"]
    percentages = contamination.get("percentages", {})

    classes = sorted({item["class"] for item in detections})

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Objects", len(detections))
    m2.metric("Unique Classes", len(classes))
    m3.metric("Oil-like Pixels", f"{percentages.get('oil', 0):.2f}%")
    m4.metric("Food-like Pixels", f"{percentages.get('food', 0):.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Table",
        "🍽️ Kitchenware",
        "🧪 Contamination",
        "📁 Saved Results",
    ])

    with tab1:

        left, right = st.columns(2)

        with left:
            st.subheader("Table Image")
            st.image(
                cv2.cvtColor(results["table"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with right:
            st.subheader("Provisional Table Mask")
            st.image(
                results["table_mask"],
                clamp=True,
                use_container_width=True,
            )

        st.caption(
            "The entire image is currently treated as the table. "
            "Automatic table-boundary detection is not implemented."
        )

    with tab2:

        st.subheader("Kitchenware Detection")

        st.image(
            cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        if detections:

            rows = []

            for number, item in enumerate(detections, start=1):
                x1, y1, x2, y2 = item["box"]

                rows.append({
                    "No.": number,
                    "Class": item["class"],
                    "Confidence (%)": round(item["confidence"] * 100, 2),
                    "Bounding Box": f"{x1}, {y1}, {x2}, {y2}",
                })

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "No known kitchenware detected. This does not prove "
                "that the table is empty."
            )

    with tab3:

        st.warning(
            "Colour-based visualizations are experimental and do not "
            "confirm the presence of oil, water, or food."
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Contamination Overlay")
            st.image(
                cv2.cvtColor(contamination["overlay"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with right:
            st.subheader("Contamination Heatmap")
            st.image(
                cv2.cvtColor(contamination["heatmap"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        rows = [
            {
                "Category": str(category).title(),
                "Flagged Pixels (%)": float(value),
                "Interpretation": "Unvalidated colour estimate",
            }
            for category, value in percentages.items()
        ]

        if rows:
            st.dataframe(rows, use_container_width=True)

        st.caption(
            "Categories may overlap. Percentages are not actual "
            "contamination coverage."
        )

    with tab4:

        st.subheader("Saved Output Images")

        saved_paths = st.session_state.get("latest_saved_paths", [])

        if not saved_paths:
            st.info("No saved output images.")

        for path_string in saved_paths:

            path = Path(path_string)

            if path.exists():
                with open(path, "rb") as file:
                    st.download_button(
                        f"Download {path.name}",
                        data=file.read(),
                        file_name=path.name,
                        mime="image/jpeg",
                        key=f"download_{path.name}",
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TableBot AI | Final-Year Project Prototype | YOLO segmentation "
    "| Experimental contamination analysis. Physical cleaning remains disabled."
)