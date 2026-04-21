import io
import os
import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    ort = None
    ORT_IMPORT_ERROR = exc
else:
    ORT_IMPORT_ERROR = None


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
SVM_MODEL_PATH = MODEL_DIR / "svm_model_160x160.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
ARCFACE_MODEL_PATH = MODEL_DIR / "arcface_w600k_r50.onnx"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
ARCFACE_INPUT_SIZE = (112, 112)
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_MARGIN_THRESHOLD = 0.08
DEFAULT_DETECTION_THRESHOLD = 0.70
MIN_DET_FACE_SIZE = 60
MAX_FACES = 5


# =========================
# UI setup
# =========================
st.set_page_config(
    page_title="Face Recognition Mobile App",
    page_icon="📷",
    layout="centered",
)

st.title("📷 Face Recognition")
st.caption("Camera or upload image → detect face → ArcFace embedding → SVM identity prediction")


# =========================
# Helpers
# =========================
def file_must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


@st.cache_resource(show_spinner=False)
def load_runtime():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Run: pip install onnxruntime"
        ) from ORT_IMPORT_ERROR

    svm_path = file_must_exist(SVM_MODEL_PATH, "SVM model")
    encoder_path = file_must_exist(LABEL_ENCODER_PATH, "Label encoder")
    arcface_path = file_must_exist(ARCFACE_MODEL_PATH, "ArcFace model")
    yunet_path = file_must_exist(YUNET_MODEL_PATH, "YuNet detector model")

    with open(svm_path, "rb") as f:
        svm_model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Keep ONNX runtime CPU-only for portability.
    session = ort.InferenceSession(str(arcface_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    detector = cv.FaceDetectorYN_create(
        str(yunet_path),
        "",
        (320, 320),
        DEFAULT_DETECTION_THRESHOLD,
        0.5,
        5000,
    )

    return {
        "svm_model": svm_model,
        "label_encoder": label_encoder,
        "arc_session": session,
        "arc_input_name": input_name,
        "arc_output_name": output_name,
        "detector": detector,
    }


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    image = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def detect_faces_bgr(bgr_image: np.ndarray, detector, score_threshold: float):
    image = bgr_image.copy()
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    detector.setScoreThreshold(float(score_threshold))
    _, faces = detector.detect(image)

    detections = []
    if faces is None:
        return detections

    for row in faces:
        x, y, fw, fh = row[:4]
        score = float(row[14]) if len(row) > 14 else 0.0
        if score < score_threshold:
            continue

        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(w, x + fw))
        y2 = int(min(h, y + fh))
        if x2 <= x1 or y2 <= y1:
            continue
        if min(x2 - x1, y2 - y1) < MIN_DET_FACE_SIZE:
            continue

        detections.append({
            "box": (x1, y1, x2, y2),
            "score": score,
            "area": (x2 - x1) * (y2 - y1),
        })

    detections.sort(key=lambda d: (d["score"], d["area"]), reverse=True)
    return detections[:MAX_FACES]



def face_crop_for_embedding(rgb_image: np.ndarray, box):
    x1, y1, x2, y2 = box
    crop = rgb_image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty face crop")
    crop = cv.resize(crop, ARCFACE_INPUT_SIZE, interpolation=cv.INTER_AREA)
    return crop



def get_embedding(face_rgb: np.ndarray, runtime) -> np.ndarray:
    face = face_rgb.astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))[None, ...]
    embedding = runtime["arc_session"].run(
        [runtime["arc_output_name"]],
        {runtime["arc_input_name"]: face},
    )[0]
    embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(norms, 1e-12, None)
    return embedding



def predict_identity(embedding: np.ndarray, runtime, conf_threshold: float, margin_threshold: float):
    svm_model = runtime["svm_model"]
    label_encoder = runtime["label_encoder"]

    if not hasattr(svm_model, "predict_proba"):
        pred_idx = int(svm_model.predict(embedding)[0])
        name = str(label_encoder.inverse_transform([pred_idx])[0])
        return {
            "name": name,
            "confidence": None,
            "margin": None,
            "second_best": None,
            "accepted": True,
        }

    probs = svm_model.predict_proba(embedding)[0]
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    if probs.shape[0] > 1:
        sorted_probs = np.sort(probs)
        second_prob = float(sorted_probs[-2])
    else:
        second_prob = 0.0

    margin = best_prob - second_prob
    accepted = (best_prob >= conf_threshold) and (margin >= margin_threshold)

    if accepted:
        encoded_class = int(svm_model.classes_[best_idx]) if hasattr(svm_model, "classes_") else best_idx
        name = str(label_encoder.inverse_transform([encoded_class])[0])
    else:
        name = "Unknown"

    second_best = second_prob if probs.shape[0] > 1 else None
    return {
        "name": name,
        "confidence": best_prob,
        "margin": margin,
        "second_best": second_best,
        "accepted": accepted,
    }



def draw_result(image_bgr: np.ndarray, detections, results):
    annotated = image_bgr.copy()
    for det, res in zip(detections, results):
        x1, y1, x2, y2 = det["box"]
        color = (0, 255, 0) if res["accepted"] else (0, 0, 255)
        cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        conf_txt = "N/A" if res["confidence"] is None else f"{res['confidence']:.2f}"
        label = f"{res['name']} | {conf_txt}"
        cv.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv.LINE_AA,
        )
    return annotated


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Settings")
    conf_threshold = st.slider("SVM confidence threshold", 0.0, 1.0, DEFAULT_CONF_THRESHOLD, 0.01)
    margin_threshold = st.slider("Top-1 vs Top-2 margin", 0.0, 1.0, DEFAULT_MARGIN_THRESHOLD, 0.01)
    det_threshold = st.slider("Face detection threshold", 0.1, 1.0, DEFAULT_DETECTION_THRESHOLD, 0.01)
    source_mode = st.radio("Input source", ["Camera", "Upload"], horizontal=True)

    st.markdown("---")
    st.caption("Expected files")
    st.code(
        "model/arcface_w600k_r50.onnx\n"
        "model/face_detection_yunet_2023mar.onnx\n"
        "model/svm_model_160x160.pkl\n"
        "model/label_encoder.pkl",
        language="text",
    )


# =========================
# Input
# =========================
uploaded = None
if source_mode == "Camera":
    uploaded = st.camera_input("Take a picture")
else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])


# =========================
# Main processing
# =========================
try:
    runtime = load_runtime()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if uploaded is None:
    st.info("Capture or upload an image to start recognition.")
    st.stop()

try:
    image_bgr = uploaded_file_to_bgr(uploaded)
except Exception as exc:
    st.error(f"Image read failed: {exc}")
    st.stop()

runtime["detector"].setScoreThreshold(float(det_threshold))
detections = detect_faces_bgr(image_bgr, runtime["detector"], det_threshold)

if not detections:
    st.image(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB), caption="Input image", use_container_width=True)
    st.warning("No face detected.")
    st.stop()

rgb_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
results = []
for det in detections:
    face_rgb = face_crop_for_embedding(rgb_image, det["box"])
    embedding = get_embedding(face_rgb, runtime)
    result = predict_identity(embedding, runtime, conf_threshold, margin_threshold)
    results.append(result)

annotated = draw_result(image_bgr, detections, results)

st.image(cv.cvtColor(annotated, cv.COLOR_BGR2RGB), caption="Recognition result", use_container_width=True)

st.subheader("Predictions")
for idx, (det, res) in enumerate(zip(detections, results), start=1):
    x1, y1, x2, y2 = det["box"]
    with st.container(border=True):
        st.markdown(f"**Face {idx}**")
        st.write({
            "name": res["name"],
            "accepted": res["accepted"],
            "confidence": None if res["confidence"] is None else round(res["confidence"], 4),
            "margin": None if res["margin"] is None else round(res["margin"], 4),
            "second_best": None if res["second_best"] is None else round(res["second_best"], 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "det_score": round(det["score"], 4),
        })
