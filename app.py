import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Face Recognition", page_icon="🧑", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
ARCFACE_MODEL_PATH = MODEL_DIR / "arcface_w600k_r50.onnx"
SVM_MODEL_PATH = MODEL_DIR / "svm_model_160x160.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

EMBEDDING_SIZE = (112, 112)
DETECTION_SCORE_THRESHOLD = 0.70
SVM_CONF_THRESHOLD = 0.50
MAX_IMAGE_SIDE = 1280


def assert_model_files():
    required = [
        YUNET_MODEL_PATH,
        ARCFACE_MODEL_PATH,
        SVM_MODEL_PATH,
        LABEL_ENCODER_PATH,
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model files in ./model: " + ", ".join(missing)
        )


@st.cache_resource(show_spinner=True)
def load_models():
    assert_model_files()

    detector = cv.FaceDetectorYN_create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        DETECTION_SCORE_THRESHOLD,
        0.5,
        5000,
    )

    ort_session = ort.InferenceSession(
        str(ARCFACE_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    arcface_input_name = ort_session.get_inputs()[0].name
    arcface_output_name = ort_session.get_outputs()[0].name

    with open(SVM_MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return {
        "detector": detector,
        "ort_session": ort_session,
        "arcface_input_name": arcface_input_name,
        "arcface_output_name": arcface_output_name,
        "svm_model": svm_model,
        "label_encoder": label_encoder,
    }


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def get_embedding(face_rgb: np.ndarray, models: dict) -> np.ndarray:
    face = cv.resize(face_rgb, EMBEDDING_SIZE, interpolation=cv.INTER_AREA)
    face = face.astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0).astype(np.float32)

    output = models["ort_session"].run(
        [models["arcface_output_name"]],
        {models["arcface_input_name"]: face},
    )[0]
    output = np.asarray(output, dtype=np.float32)
    return l2_normalize(output)


def predict_identity(embedding: np.ndarray, models: dict):
    svm_model = models["svm_model"]
    label_encoder = models["label_encoder"]

    probs = svm_model.predict_proba(embedding)[0]
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    if best_prob < SVM_CONF_THRESHOLD:
        return "Unknown", best_prob

    label = svm_model.classes_[best_idx]
    try:
        name = label_encoder.inverse_transform([label])[0]
    except Exception:
        name = str(label)

    return str(name), best_prob


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    arr = np.array(image)

    h, w = arr.shape[:2]
    max_side = max(h, w)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(max_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        arr = cv.resize(arr, (new_w, new_h), interpolation=cv.INTER_AREA)

    return arr


def detect_and_recognize(rgb_image: np.ndarray, models: dict):
    detector = models["detector"]

    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
    h, w = bgr_image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr_image)

    annotated = bgr_image.copy()
    detections = []

    if faces is None or len(faces) == 0:
        return cv.cvtColor(annotated, cv.COLOR_BGR2RGB), detections

    for row in faces:
        x, y, fw, fh = row[:4]
        score = float(row[14]) if len(row) > 14 else 0.0

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + fw))
        y2 = min(h, int(y + fh))

        if x2 <= x1 or y2 <= y1:
            continue

        face_rgb = rgb_image[y1:y2, x1:x2]
        if face_rgb.size == 0:
            continue

        emb = get_embedding(face_rgb, models)
        name, conf = predict_identity(emb, models)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} | {conf:.2f}"
        cv.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv.LINE_AA,
        )

        detections.append(
            {
                "name": name,
                "confidence": round(conf, 4),
                "detector_score": round(score, 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    annotated_rgb = cv.cvtColor(annotated, cv.COLOR_BGR2RGB)
    return annotated_rgb, detections


st.title("Live Face Recognition")
st.caption("Pure Streamlit version. Use camera capture or upload an image.")

with st.sidebar:
    st.subheader("Settings")
    source = st.radio("Input source", ["Camera", "Upload"], index=0)
    st.write("Model folder:")
    st.code(str(MODEL_DIR), language="text")

try:
    models = load_models()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

image = None
if source == "Camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file)
else:
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    rgb_image = prepare_image(image)

    with st.spinner("Detecting and recognizing faces..."):
        annotated_rgb, detections = detect_and_recognize(rgb_image, models)

    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb_image, caption="Input", use_container_width=True)
    with c2:
        st.image(annotated_rgb, caption="Result", use_container_width=True)

    if detections:
        st.success(f"Detected {len(detections)} face(s)")
        for idx, det in enumerate(detections, start=1):
            st.write(
                f"**Face {idx}:** {det['name']} | confidence={det['confidence']} | detector_score={det['detector_score']}"
            )
    else:
        st.warning("No face detected")
else:
    st.info("Capture or upload an image to start recognition.")
