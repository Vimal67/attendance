import argparse
import datetime
import json
import logging
import os
import pickle
import re
import tempfile
import time
import zipfile
from queue import Empty, Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from urllib.request import urlretrieve
from zoneinfo import ZoneInfo

from env_loader import load_environment

load_environment()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2 as cv
import numpy as np

from config import RTSP_URL

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except Exception as exc:
    BYTETracker = None
    BYTE_TRACK_IMPORT_ERROR = exc
else:
    BYTE_TRACK_IMPORT_ERROR = None

try:
    cv.ocl.setUseOpenCL(False)
except Exception:
    pass


# ----------------------------
# Config
# ----------------------------
def _read_float_env(name, default, *, min_value=None, max_value=None):
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return float(value)


def _read_int_env(name, default, *, min_value=None, max_value=None):
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return int(value)


FRAME_WIDTH = _read_int_env("FRAME_WIDTH", 1280, min_value=320, max_value=3840)
FRAME_HEIGHT = _read_int_env("FRAME_HEIGHT", 720, min_value=240, max_value=2160)
FRAME_QUEUE_SIZE = _read_int_env("FRAME_QUEUE_SIZE", 5, min_value=1, max_value=128)
RECOGNITION_QUEUE_SIZE = _read_int_env("RECOGNITION_QUEUE_SIZE", 64, min_value=1, max_value=1024)
MAX_DISPLAY_FPS = 120.0

DETECTION_CONF_THRESHOLD = _read_float_env("DETECTION_CONF_THRESHOLD", 0.40, min_value=0.0, max_value=1.0)
SVM_CONF_THRESHOLD = 90 #_read_float_env("SVM_CONF_THRESHOLD", 0.30, min_value=0.0, max_value=1.0)
SVM_PROB_MARGIN_THRESHOLD = _read_float_env("SVM_PROB_MARGIN_THRESHOLD", 0.25, min_value=0.0, max_value=1.0)
MIN_DET_FACE_SIZE = _read_int_env("MIN_DET_FACE_SIZE", 60, min_value=10, max_value=4096)
MAX_DET_PER_FRAME = _read_int_env("MAX_DET_PER_FRAME", 5, min_value=1, max_value=64)
MIN_FACE_BOX_ASPECT = _read_float_env("MIN_FACE_BOX_ASPECT", 0.20, min_value=0.2, max_value=5.0)
MAX_FACE_BOX_ASPECT = _read_float_env("MAX_FACE_BOX_ASPECT", 1.8, min_value=MIN_FACE_BOX_ASPECT, max_value=5.0)
STRICT_FACE_POSE_FILTER = os.getenv("STRICT_FACE_POSE_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_FRONT_SIDE_YAW_RATIO = _read_float_env("MAX_FRONT_SIDE_YAW_RATIO", 0.32, min_value=0.05, max_value=2.0)
MAX_FRONT_SIDE_ROLL_DEG = _read_float_env("MAX_FRONT_SIDE_ROLL_DEG", 18.0, min_value=1.0, max_value=90.0)
MIN_FRONT_SIDE_PITCH_RATIO = _read_float_env("MIN_FRONT_SIDE_PITCH_RATIO", 0.45, min_value=0.05, max_value=10.0)
MAX_FRONT_SIDE_PITCH_RATIO = _read_float_env("MAX_FRONT_SIDE_PITCH_RATIO", 2.20, min_value=MIN_FRONT_SIDE_PITCH_RATIO + 0.05, max_value=10.0)
RECOGNITION_CONFIRM_FRAMES = 3#_read_int_env("RECOGNITION_CONFIRM_FRAMES", 1, min_value=1, max_value=20)
RECOGNITION_MIN_LOCK_CONF = _read_float_env("RECOGNITION_MIN_LOCK_CONF", 0.30, min_value=0.0, max_value=1.0)
SHOW_UNKNOWN_OVERLAYS = os.getenv("SHOW_UNKNOWN_OVERLAYS", "1").strip().lower() not in {"0", "false", "no", "off"}
DETECT_EVERY_N = _read_int_env("DETECT_EVERY_N", 1, min_value=1, max_value=100)
DETECTION_SCALE = _read_float_env("DETECTION_SCALE", 0.75, min_value=0.35, max_value=1.0)
MAX_DET_STALE_FRAMES = _read_int_env("MAX_DET_STALE_FRAMES", DETECT_EVERY_N * 2, min_value=1, max_value=1000)
CAMERA_REOPEN_AFTER_SECONDS = _read_float_env("CAMERA_REOPEN_AFTER_SECONDS", 5.0, min_value=0.5, max_value=120.0)
RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;discardcorrupt|"
    "flags;low_delay|"
    "max_delay;500000|"
    "stimeout;5000000"
)
MAIN_HEALTHCHECK_INTERVAL_SECONDS = 1.0

BYTE_TRACK_HIGH_THRESH = _read_float_env("BYTE_TRACK_HIGH_THRESH", 0.25, min_value=0.0, max_value=1.0)
BYTE_TRACK_LOW_THRESH = _read_float_env("BYTE_TRACK_LOW_THRESH", 0.10, min_value=0.0, max_value=1.0)
BYTE_TRACK_NEW_TRACK_THRESH = _read_float_env("BYTE_TRACK_NEW_TRACK_THRESH", 0.25, min_value=0.0, max_value=1.0)
BYTE_TRACK_BUFFER = _read_int_env("BYTE_TRACK_BUFFER", 5, min_value=1, max_value=200)
BYTE_TRACK_MATCH_THRESH = _read_float_env("BYTE_TRACK_MATCH_THRESH", 0.8, min_value=0.0, max_value=1.0)
BYTE_TRACK_FUSE_SCORE = True
TRACKER_BACKEND = "bytesort"

DEFAULT_YUNET_MODEL_PATH = os.path.join("model", "face_detection_yunet_2023mar.onnx")
YUNET_MODEL_PATH = os.getenv("YUNET_MODEL_PATH", DEFAULT_YUNET_MODEL_PATH).strip() or DEFAULT_YUNET_MODEL_PATH
YUNET_SCORE_THRESHOLD = _read_float_env("YUNET_SCORE_THRESHOLD", 0.7, min_value=0.0, max_value=1.0)
YUNET_NMS_THRESHOLD = _read_float_env("YUNET_NMS_THRESHOLD", 0.5, min_value=0.0, max_value=1.0)
YUNET_TOP_K = _read_int_env("YUNET_TOP_K", 5000, min_value=1, max_value=50000)
YUNET_AUTO_DOWNLOAD = os.getenv("YUNET_AUTO_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
YUNET_MODEL_URL = os.getenv(
    "YUNET_MODEL_URL",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
).strip()

DEFAULT_ARCFACE_MODEL_PATH = os.path.join("model", "arcface_w600k_r50.onnx")
ARCFACE_MODEL_PATH = os.getenv("ARCFACE_MODEL_PATH", DEFAULT_ARCFACE_MODEL_PATH).strip() or DEFAULT_ARCFACE_MODEL_PATH
ARCFACE_AUTO_DOWNLOAD = os.getenv("ARCFACE_AUTO_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
ARCFACE_MODEL_URL = os.getenv(
    "ARCFACE_MODEL_URL",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
).strip()
ARCFACE_INPUT_SIZE = (112, 112)
INDIA_TZ = ZoneInfo("Asia/Kolkata")


# ----------------------------
# Logging / globals
# ----------------------------
def _now_local():
    return datetime.datetime.now(INDIA_TZ).replace(tzinfo=None)


def _normalize_embedding_backend(value):
    raw = str(value or "").strip().lower()
    if raw in {"insightface", "arcface", "arc", ""}:
        return "insightface"
    return "insightface"


def _normalize_embedding_device(value):
    raw = str(value or "").strip().lower()
    if raw in {"cuda", "gpu", "cuda:0"}:
        return "cuda"
    if raw in {"cpu"}:
        return "cpu"
    return "auto"


EMBEDDING_BACKEND = _normalize_embedding_backend(os.getenv("EMBEDDING_BACKEND", "insightface"))
EMBEDDING_DEVICE = "cpu"
EMBEDDING_INPUT_SIZE = ARCFACE_INPUT_SIZE

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recognition_only")

ONNXRuntime = None
ONNXRUNTIME_IMPORT_ERROR = None

model_load_lock = Lock()
embedder_lock = Lock()
model_predict_lock = Lock()

embedder = None
_svm_model = None
_label_encoder = None
_loaded_model_artifact_signature = None
_last_model_artifact_check_ts = 0.0
MODEL_META_PATH = os.getenv("MODEL_META_PATH", os.path.join("model", "model_meta.json")).strip() or os.path.join("model", "model_meta.json")
SVM_MODEL_PATH = os.getenv("SVM_MODEL_PATH", os.path.join("model", "svm_model_160x160.pkl")).strip() or os.path.join("model", "svm_model_160x160.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", os.path.join("model", "label_encoder.pkl")).strip() or os.path.join("model", "label_encoder.pkl")
MODEL_RELOAD_CHECK_INTERVAL_SECONDS = _read_float_env("MODEL_RELOAD_CHECK_INTERVAL_SECONDS", 2.0, min_value=0.2, max_value=60.0)


# ----------------------------
# SVM / ArcFace helpers
# ----------------------------
def _resolve_svm_artifact_paths():
    model_path = os.path.normpath(os.path.expanduser(str(SVM_MODEL_PATH or "").strip()))
    encoder_path = os.path.normpath(os.path.expanduser(str(LABEL_ENCODER_PATH or "").strip()))
    meta_path = os.path.normpath(os.path.expanduser(str(MODEL_META_PATH or "").strip()))

    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception as exc:
            logger.warning("Failed to parse model metadata %s: %s. Using configured artifact paths.", meta_path, exc)
        else:
            meta_model = str(meta.get("model_path") or "").strip()
            meta_encoder = str(meta.get("encoder_path") or "").strip()
            if meta_model and meta_encoder:
                model_path = os.path.normpath(os.path.expanduser(meta_model))
                encoder_path = os.path.normpath(os.path.expanduser(meta_encoder))
    return model_path, encoder_path


def _validate_svm_artifact_compatibility(loaded_model, loaded_encoder, model_path, encoder_path):
    enc_classes = np.asarray(getattr(loaded_encoder, "classes_", []), dtype=object).reshape(-1)
    if enc_classes.size <= 0:
        raise RuntimeError(f"Label encoder has no classes: {encoder_path}")

    model_classes = np.asarray(getattr(loaded_model, "classes_", [])).reshape(-1)
    if model_classes.size <= 0:
        raise RuntimeError(f"SVM model has no classes_: {model_path}")

    try:
        model_classes_int = model_classes.astype(np.int64)
    except Exception:
        if model_classes.size != enc_classes.size:
            raise RuntimeError(
                f"Model/encoder class-count mismatch: model classes={int(model_classes.size)}, encoder classes={int(enc_classes.size)}"
            )
        return

    expected = np.arange(enc_classes.size, dtype=np.int64)
    if model_classes_int.size != enc_classes.size or not np.array_equal(np.sort(model_classes_int), expected):
        raise RuntimeError(
            f"Model/encoder class mismatch: model classes={int(model_classes_int.size)}, encoder classes={int(enc_classes.size)}"
        )


def ensure_yunet_model():
    model_path = os.path.expanduser(YUNET_MODEL_PATH)
    if os.path.exists(model_path):
        return model_path
    if not YUNET_AUTO_DOWNLOAD:
        raise FileNotFoundError(f"YuNet model not found: {model_path}")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    logger.info("Downloading YuNet model to %s", model_path)
    urlretrieve(YUNET_MODEL_URL, model_path)
    return model_path


def ensure_arcface_model():
    model_path = os.path.expanduser(ARCFACE_MODEL_PATH)
    if os.path.exists(model_path):
        return model_path
    if not ARCFACE_AUTO_DOWNLOAD:
        raise FileNotFoundError(f"ArcFace model not found: {model_path}")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    logger.info("Downloading ArcFace model to %s", model_path)

    def _extract_arcface_from_zip(zip_path, output_path):
        expected_name = os.path.basename(output_path).lower()
        candidate_suffixes = [expected_name, "w600k_r50.onnx", "glintr100.onnx", "w600k_mbf.onnx"]
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            picked = None
            lower_names = {name.lower(): name for name in names}
            for suffix in candidate_suffixes:
                for low_name, orig_name in lower_names.items():
                    if low_name.endswith("/" + suffix) or low_name == suffix:
                        picked = orig_name
                        break
                if picked is not None:
                    break
            if picked is None:
                for orig_name in names:
                    if orig_name.lower().endswith(".onnx"):
                        picked = orig_name
                        break
            if picked is None:
                raise RuntimeError("No ONNX model found in downloaded ArcFace ZIP archive")
            with zf.open(picked, "r") as src, open(output_path, "wb") as dst:
                dst.write(src.read())

    if ARCFACE_MODEL_URL.lower().endswith(".zip"):
        fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        try:
            urlretrieve(ARCFACE_MODEL_URL, tmp_zip)
            _extract_arcface_from_zip(tmp_zip, model_path)
        finally:
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
    else:
        urlretrieve(ARCFACE_MODEL_URL, model_path)
    return model_path


def _opencv_cuda_available():
    try:
        return bool(hasattr(cv, "cuda") and cv.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:
        return False


def _load_onnxruntime_module():
    global ONNXRuntime, ONNXRUNTIME_IMPORT_ERROR
    if ONNXRuntime is None and ONNXRUNTIME_IMPORT_ERROR is None:
        try:
            import onnxruntime as _onnxruntime
        except Exception as exc:
            ONNXRUNTIME_IMPORT_ERROR = exc
        else:
            ONNXRuntime = _onnxruntime
    return ONNXRuntime


def _onnxruntime_cuda_prereqs():
    if os.name != "nt":
        return True, []
    required_dlls = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
    path_entries = [p.strip().strip('"') for p in str(os.environ.get("PATH", "")).split(os.pathsep) if p.strip()]
    missing = []
    for dll_name in required_dlls:
        found = False
        for entry in path_entries:
            if os.path.isfile(os.path.join(entry, dll_name)):
                found = True
                break
        if not found:
            missing.append(dll_name)
    return len(missing) == 0, missing


class _ArcFaceONNXRuntimeAdapter:
    def __init__(self, model_path, requested_device="auto"):
        ort = _load_onnxruntime_module()
        if ort is None:
            raise RuntimeError("onnxruntime is not available") from ONNXRUNTIME_IMPORT_ERROR

        self.model_path = model_path
        self.input_size = ARCFACE_INPUT_SIZE
        self.requested_device = _normalize_embedding_device(requested_device)
        self.cuda_prereq_missing = []

        available_providers = list(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if self.requested_device in {"auto", "cuda"} and "CUDAExecutionProvider" in available_providers:
            cuda_ready, missing = _onnxruntime_cuda_prereqs()
            self.cuda_prereq_missing = missing
            if cuda_ready:
                providers.insert(0, "CUDAExecutionProvider")

        session_options = ort.SessionOptions()
        try:
            session_options.log_severity_level = 3
        except Exception:
            pass
        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        active_providers = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
        self.runtime = f"onnxruntime:{active_providers[0] if active_providers else 'unknown'}"

    def embeddings(self, batch):
        arr = np.asarray(batch, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        if arr.ndim != 4:
            raise ValueError(f"Expected batch shape (N,H,W,C), got {arr.shape!r}")

        faces = []
        target_w, target_h = self.input_size
        for idx in range(arr.shape[0]):
            face = arr[idx]
            if face.shape[1] != target_w or face.shape[0] != target_h:
                face = cv.resize(face, (target_w, target_h), interpolation=cv.INTER_AREA)
            faces.append(face)

        tensor = np.stack(faces, axis=0).astype(np.float32)
        tensor = (tensor - 127.5) / 127.5
        tensor = np.transpose(tensor, (0, 3, 1, 2))
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return np.asarray(out, dtype=np.float32).reshape((arr.shape[0], -1))


class _ArcFaceOpenCVAdapter:
    def __init__(self, model_path, requested_device="auto"):
        self.model_path = model_path
        self.input_size = ARCFACE_INPUT_SIZE
        self.requested_device = _normalize_embedding_device(requested_device)
        self.net = cv.dnn.readNetFromONNX(model_path)
        self.device = "cpu"
        self.runtime = "opencv-dnn:cpu"

        if self.requested_device in {"auto", "cuda"} and _opencv_cuda_available():
            try:
                self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
                self.device = "cuda"
                self.runtime = "opencv-dnn:cuda"
            except Exception:
                self._configure_cpu_backend()
        else:
            self._configure_cpu_backend()

    def _configure_cpu_backend(self):
        try:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        except Exception:
            pass
        self.device = "cpu"
        self.runtime = "opencv-dnn:cpu"

    def embeddings(self, batch):
        arr = np.asarray(batch, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        if arr.ndim != 4:
            raise ValueError(f"Expected batch shape (N,H,W,C), got {arr.shape!r}")

        faces = []
        target_w, target_h = self.input_size
        for idx in range(arr.shape[0]):
            face = arr[idx]
            if face.shape[1] != target_w or face.shape[0] != target_h:
                face = cv.resize(face, (target_w, target_h), interpolation=cv.INTER_AREA)
            faces.append(face)

        blob = cv.dnn.blobFromImages(
            faces,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()
        return np.asarray(out, dtype=np.float32).reshape((arr.shape[0], -1))


def _create_arcface_embedder(model_path, requested_device):
    try:
        return _ArcFaceONNXRuntimeAdapter(model_path=model_path, requested_device="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ArcFace model with onnxruntime from: {model_path}. "
            f"Install onnxruntime and verify the ONNX file is valid."
        ) from exc

def _model_artifact_signature():
    model_path, encoder_path = _resolve_svm_artifact_paths()
    try:
        model_stat = os.stat(model_path)
        encoder_stat = os.stat(encoder_path)
    except OSError:
        return None
    return (
        os.path.abspath(model_path),
        int(model_stat.st_mtime_ns),
        int(model_stat.st_size),
        os.path.abspath(encoder_path),
        int(encoder_stat.st_mtime_ns),
        int(encoder_stat.st_size),
    )


def reload_shared_models(force=False):
    global embedder, _svm_model, _label_encoder
    global _loaded_model_artifact_signature, _last_model_artifact_check_ts

    with model_load_lock:
        if embedder is None:
            model_path = ensure_arcface_model()
            embedder = _create_arcface_embedder(model_path=model_path, requested_device=EMBEDDING_DEVICE)

        signature = _model_artifact_signature()
        if signature is None:
            model_path, encoder_path = _resolve_svm_artifact_paths()
            raise FileNotFoundError(
                f"SVM artifacts not found. Expected both files:\n- model: {model_path}\n- encoder: {encoder_path}"
            )

        should_reload = bool(force) or signature != _loaded_model_artifact_signature or _svm_model is None or _label_encoder is None
        if not should_reload:
            _last_model_artifact_check_ts = time.monotonic()
            return

        model_path, encoder_path = _resolve_svm_artifact_paths()
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            loaded_encoder = pickle.load(f)

        _validate_svm_artifact_compatibility(loaded_model, loaded_encoder, model_path, encoder_path)

        with model_predict_lock:
            _svm_model = loaded_model
            _label_encoder = loaded_encoder

        _loaded_model_artifact_signature = signature
        _last_model_artifact_check_ts = time.monotonic()


def initialize_shared_models():
    reload_shared_models(force=False)


def _refresh_model_artifacts_if_needed():
    global _last_model_artifact_check_ts
    now_ts = time.monotonic()
    if (now_ts - _last_model_artifact_check_ts) < MODEL_RELOAD_CHECK_INTERVAL_SECONDS:
        return
    _last_model_artifact_check_ts = now_ts
    signature = _model_artifact_signature()
    if signature is None:
        return
    if signature != _loaded_model_artifact_signature or _svm_model is None or _label_encoder is None:
        reload_shared_models(force=False)


def get_embeddings_batch(face_imgs):
    global embedder
    arr = np.asarray(face_imgs, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 4:
        raise ValueError(f"Expected face batch with shape (N,H,W,C), got {arr.shape!r}")
    with embedder_lock:
        yhat = embedder.embeddings(arr)
    embs = np.asarray(yhat, dtype=np.float32).reshape((arr.shape[0], -1))
    norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    return embs / norms


def get_embeddings(face_img):
    return np.asarray(get_embeddings_batch([face_img])[0], dtype=np.float32)


def _predict_identities_svm(embs):
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected embedding matrix with shape (N,D), got {arr.shape!r}")

    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)

    with model_predict_lock:
        local_model = _svm_model
        local_encoder = _label_encoder
    if local_model is None or local_encoder is None:
        return [("Unknown", 0.0) for _ in range(arr.shape[0])]

    try:
        with model_predict_lock:
            probs = np.asarray(local_model.predict_proba(arr), dtype=np.float32)
            model_classes = np.asarray(getattr(local_model, "classes_", []))
            encoder_classes = np.asarray(getattr(local_encoder, "classes_", []), dtype=object)
    except Exception:
        logger.exception("SVM predict_proba failed")
        return [("Unknown", 0.0) for _ in range(arr.shape[0])]

    out = []
    for i in range(probs.shape[0]):
        try:
            row = probs[i]
            best_idx = int(np.argmax(row))
            best_prob = float(row[best_idx])
            second_prob = float(np.partition(row, -2)[-2:].min()) if row.shape[0] > 1 else 0.0
            margin = best_prob - second_prob

            if best_prob < SVM_CONF_THRESHOLD or margin < SVM_PROB_MARGIN_THRESHOLD:
                out.append(("Unknown", max(0.0, best_prob)))
                continue

            model_class = model_classes[best_idx] if model_classes.size > best_idx else best_idx
            try:
                class_idx = int(model_class)
                label = str(encoder_classes[class_idx]) if 0 <= class_idx < encoder_classes.size else "Unknown"
            except Exception:
                label = str(model_class)
            out.append((label or "Unknown", max(0.0, min(1.0, best_prob))))
        except Exception:
            logger.exception("SVM post-process failed for query %d", i)
            out.append(("Unknown", 0.0))
    return out


def predict_identities(embs):
    _refresh_model_artifacts_if_needed()
    return _predict_identities_svm(embs)


def predict_identity(emb):
    return predict_identities([emb])[0]


def redact_source(source):
    if not isinstance(source, str):
        return repr(source)
    return re.sub(r"(://[^:/@]+:)[^@]+@", r"\1***@", source)


def normalize_video_source(source):
    if isinstance(source, int):
        return source
    if source is None:
        raise ValueError("Camera source is not set")
    if isinstance(source, str):
        source = source.strip()
        if not source:
            raise ValueError("Camera source is empty")
        if source.isdigit():
            return int(source)
        return source
    raise ValueError(f"Unsupported camera source type: {type(source)!r}")


def open_video_source(source):
    source = normalize_video_source(source)
    cam_index = source if isinstance(source, int) else None

    if cam_index is not None:
        candidates = [cv.CAP_DSHOW, cv.CAP_MSMF, None] if os.name == "nt" else [None]
        for backend in candidates:
            cap = cv.VideoCapture(cam_index) if backend is None else cv.VideoCapture(cam_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok = False
            for _ in range(10):
                try:
                    grabbed, frame = cap.read()
                except Exception:
                    grabbed, frame = False, None
                if grabbed and frame is not None and frame.size > 0:
                    ok = True
                    break
                time.sleep(0.02)
            if ok:
                return cap
            cap.release()
        return cv.VideoCapture(cam_index)

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_FFMPEG_OPTIONS)
    cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv.VideoCapture(source)
    return cap


# ----------------------------
# Runtime worker
# ----------------------------
class CameraWorker:
    def __init__(self, camera_id, source_cfg, display_name=None, show_window=False):
        self.camera_id = str(camera_id)
        self.display_name = (display_name or self.camera_id).strip() or self.camera_id
        self.source_cfg = source_cfg
        self.show_window = bool(show_window)

        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.recognition_queue = Queue(maxsize=RECOGNITION_QUEUE_SIZE)
        self.stop_event = Event()

        self.latest_frame_lock = Lock()
        self.track_state_lock = Lock()

        self.latest_raw_frame = None
        self.latest_annotated_frame = None

        self.track_id_to_name = {}
        self.track_id_to_conf = {}
        self.track_id_pending = {}
        self.track_id_recog_seq = {}
        self.track_id_inflight = {}
        self._last_detections = []

        self.capture_thread = None
        self.recognition_thread = None
        self.recognizer_thread = None

        self.detector = None
        self.tracker = None
        self.tracker_backend = TRACKER_BACKEND

    def _set_latest_frame(self, frame, annotated=False):
        with self.latest_frame_lock:
            if annotated:
                self.latest_annotated_frame = frame.copy()
            else:
                self.latest_raw_frame = frame.copy()

    def _initialize_detector_and_tracker(self):
        if self.detector is None:
            model_path = ensure_yunet_model()
            self.detector = cv.FaceDetectorYN_create(
                model_path,
                "",
                (FRAME_WIDTH, FRAME_HEIGHT),
                YUNET_SCORE_THRESHOLD,
                YUNET_NMS_THRESHOLD,
                YUNET_TOP_K,
            )

        if self.tracker is None:
            if BYTETracker is None:
                raise RuntimeError(f"ByteTrack is not available: {BYTE_TRACK_IMPORT_ERROR}")
            bt_args = SimpleNamespace(
                track_high_thresh=BYTE_TRACK_HIGH_THRESH,
                track_low_thresh=BYTE_TRACK_LOW_THRESH,
                new_track_thresh=BYTE_TRACK_NEW_TRACK_THRESH,
                track_buffer=BYTE_TRACK_BUFFER,
                match_thresh=BYTE_TRACK_MATCH_THRESH,
                fuse_score=BYTE_TRACK_FUSE_SCORE,
            )
            self.tracker = BYTETracker(args=bt_args, frame_rate=30)

    def _detect_faces(self, bgr_frame):
        def _face_box_ok(x1, y1, x2, y2):
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            if bw <= 0 or bh <= 0:
                return False
            if min(bw, bh) < MIN_DET_FACE_SIZE:
                return False
            aspect = bw / max(bh, 1.0)
            return MIN_FACE_BOX_ASPECT <= aspect <= MAX_FACE_BOX_ASPECT

        def _front_or_slight_side_ok(row, x1, y1, x2, y2):
            if not STRICT_FACE_POSE_FILTER:
                return True
            if len(row) < 14:
                return False
            try:
                pts = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            except Exception:
                return False
            if not np.isfinite(pts).all():
                return False

            box_w = max(1.0, float(x2 - x1))
            box_h = max(1.0, float(y2 - y1))
            margin_x = max(2.0, box_w * 0.15)
            margin_y = max(2.0, box_h * 0.15)
            if (
                np.any(pts[:, 0] < (float(x1) - margin_x))
                or np.any(pts[:, 0] > (float(x2) + margin_x))
                or np.any(pts[:, 1] < (float(y1) - margin_y))
                or np.any(pts[:, 1] > (float(y2) + margin_y))
            ):
                return False

            eyes = pts[:2]
            nose = pts[2]
            mouth = pts[3:5]
            eye_order = np.argsort(eyes[:, 0])
            left_eye = eyes[int(eye_order[0])]
            right_eye = eyes[int(eye_order[1])]
            eye_dx = float(right_eye[0] - left_eye[0])
            eye_dy = float(right_eye[1] - left_eye[1])
            inter_eye = float(np.hypot(eye_dx, eye_dy))
            if inter_eye < 6.0:
                return False
            roll_deg = abs(float(np.degrees(np.arctan2(eye_dy, max(eye_dx, 1e-6)))))
            if roll_deg > MAX_FRONT_SIDE_ROLL_DEG:
                return False
            eye_mid_x = float((left_eye[0] + right_eye[0]) * 0.5)
            yaw_ratio = abs(float((nose[0] - eye_mid_x) / max(inter_eye, 1e-6)))
            if yaw_ratio > MAX_FRONT_SIDE_YAW_RATIO:
                return False
            mouth_width = float(np.hypot(mouth[0][0] - mouth[1][0], mouth[0][1] - mouth[1][1]))
            if mouth_width < (0.25 * inter_eye):
                return False
            eye_mid_y = float((left_eye[1] + right_eye[1]) * 0.5)
            mouth_mid_y = float((mouth[0][1] + mouth[1][1]) * 0.5)
            upper = float(nose[1] - eye_mid_y)
            lower = float(mouth_mid_y - nose[1])
            if upper <= 1.0 or lower <= 1.0:
                return False
            pitch_ratio = float(upper / max(lower, 1e-6))
            return MIN_FRONT_SIDE_PITCH_RATIO <= pitch_ratio <= MAX_FRONT_SIDE_PITCH_RATIO

        h, w = bgr_frame.shape[:2]
        if h <= 0 or w <= 0:
            return []
        detections = []
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_frame)
        faces = [] if faces is None else faces

        for row in faces:
            x, y, fw, fh = row[:4]
            score = float(row[14]) if len(row) > 14 else 0.99
            if score < DETECTION_CONF_THRESHOLD:
                continue
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(w, x + fw))
            y2 = int(min(h, y + fh))
            if not _face_box_ok(x1, y1, x2, y2):
                continue
            if not _front_or_slight_side_ok(row, x1, y1, x2, y2):
                continue
            detections.append([x1, y1, x2, y2, score])

        if len(detections) > MAX_DET_PER_FRAME:
            detections.sort(key=lambda d: (float(d[4]), (d[2] - d[0]) * (d[3] - d[1])), reverse=True)
            detections = detections[:MAX_DET_PER_FRAME]
        self._last_detections = detections
        return detections

    def _run_tracker(self, detections, frame):
        if detections:
            det_np = np.array(detections, dtype=np.float32)
            xyxy = det_np[:, :4]
            conf = det_np[:, 4]
            cls = np.zeros((det_np.shape[0],), dtype=np.float32)
            xywh = np.empty((det_np.shape[0], 4), dtype=np.float32)
            xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
            xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
            xywh[:, 2] = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
            xywh[:, 3] = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
        else:
            xywh = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls = np.empty((0,), dtype=np.float32)

        bt_results = SimpleNamespace(xywh=xywh, conf=conf, cls=cls)
        tracked = self.tracker.update(bt_results, img=frame)
        if tracked is None:
            return np.empty((0, 5), dtype=np.float32)
        return tracked

    def _clear_inflight_for_job(self, track_id, seq):
        with self.track_state_lock:
            inflight_seq = self.track_id_inflight.get(track_id)
            if inflight_seq == seq:
                self.track_id_inflight.pop(track_id, None)

    def _enqueue_recognition_job(self, track_id, face_region):
        with self.track_state_lock:
            if track_id in self.track_id_to_name or track_id in self.track_id_inflight:
                return False
            seq = int(self.track_id_recog_seq.get(track_id, 0)) + 1
            self.track_id_recog_seq[track_id] = seq
            self.track_id_inflight[track_id] = seq

        job = {"track_id": int(track_id), "seq": int(seq), "face_region": np.asarray(face_region, dtype=np.float32).copy()}
        try:
            self.recognition_queue.put_nowait(job)
            return True
        except Exception:
            self._clear_inflight_for_job(track_id, seq)
            return False

    def _clear_runtime_state(self):
        with self.track_state_lock:
            self.track_id_to_name.clear()
            self.track_id_to_conf.clear()
            self.track_id_pending.clear()
            self.track_id_recog_seq.clear()
            self.track_id_inflight.clear()
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            try:
                self.tracker.reset()
            except Exception:
                logger.exception("[%s] Failed to reset tracker state", self.camera_id)
        with self.latest_frame_lock:
            self.latest_raw_frame = None
            self.latest_annotated_frame = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        while not self.recognition_queue.empty():
            try:
                self.recognition_queue.get_nowait()
                self.recognition_queue.task_done()
            except Exception:
                break

    def alive(self):
        return (
            self.capture_thread is not None
            and self.capture_thread.is_alive()
            and self.recognition_thread is not None
            and self.recognition_thread.is_alive()
            and self.recognizer_thread is not None
            and self.recognizer_thread.is_alive()
            and not self.stop_event.is_set()
        )

    def start(self):
        initialize_shared_models()
        self._initialize_detector_and_tracker()
        self._clear_runtime_state()
        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True, name=f"capture-{self.camera_id}")
        self.recognition_thread = Thread(target=self._recognition_loop, daemon=True, name=f"track-{self.camera_id}")
        self.recognizer_thread = Thread(target=self._recognizer_loop, daemon=True, name=f"recognizer-{self.camera_id}")
        self.capture_thread.start()
        self.recognition_thread.start()
        self.recognizer_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5.0)
        if self.recognition_thread is not None:
            self.recognition_thread.join(timeout=5.0)
        if self.recognizer_thread is not None:
            self.recognizer_thread.join(timeout=5.0)
        self.capture_thread = None
        self.recognition_thread = None
        self.recognizer_thread = None
        if self.show_window:
            try:
                cv.destroyWindow(self._window_name())
            except Exception:
                pass

    def _window_name(self):
        return f"Face Recognition + Tracking [{self.camera_id}]"

    def _capture_loop(self):
        logger.info("[%s] Capture thread starting on source %s", self.camera_id, redact_source(self.source_cfg))
        cap = open_video_source(self.source_cfg)
        if not cap.isOpened():
            logger.error("[%s] Unable to open camera source: %s", self.camera_id, redact_source(self.source_cfg))
            self.stop_event.set()
            return

        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        last_ok_ts = time.time()
        while not self.stop_event.is_set():
            try:
                res, frame = cap.read()
            except Exception:
                logger.exception("[%s] Camera read exception", self.camera_id)
                res, frame = False, None

            if not res:
                if time.time() - last_ok_ts >= CAMERA_REOPEN_AFTER_SECONDS:
                    logger.warning("[%s] Camera read failed, reopening source.", self.camera_id)
                    cap.release()
                    cap = open_video_source(self.source_cfg)
                    if not cap.isOpened():
                        time.sleep(1.0)
                        continue
                    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                    last_ok_ts = time.time()
                time.sleep(0.05)
                continue

            last_ok_ts = time.time()
            try:
                frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)
            except Exception:
                logger.exception("[%s] Frame resize failed", self.camera_id)
                time.sleep(0.05)
                continue

            self._set_latest_frame(frame, annotated=False)
            if self.frame_queue.full():
                continue
            self.frame_queue.put(frame)

        cap.release()
        logger.info("[%s] Capture thread stopped", self.camera_id)

    def _recognition_loop(self):
        logger.info("[%s] Tracking thread started", self.camera_id)
        prev_frame_ts = time.perf_counter()
        frame_idx = 0
        last_detections = []
        last_det_frame = -10**9

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            frame_idx += 1
            now_ts = time.perf_counter()
            frame_dt = max(now_ts - prev_frame_ts, 1e-6)
            prev_frame_ts = now_ts
            _ = min(1.0 / frame_dt, MAX_DISPLAY_FPS)

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            detections = []
            run_detection = (frame_idx % DETECT_EVERY_N == 0)
            if run_detection:
                try:
                    if DETECTION_SCALE < 1.0:
                        small_w = max(1, int(frame.shape[1] * DETECTION_SCALE))
                        small_h = max(1, int(frame.shape[0] * DETECTION_SCALE))
                        small_frame = cv.resize(frame, (small_w, small_h), interpolation=cv.INTER_AREA)
                        small_detections = self._detect_faces(small_frame)
                        scale_x = frame.shape[1] / float(small_w)
                        scale_y = frame.shape[0] / float(small_h)
                        for det in small_detections:
                            x1, y1, x2, y2, score = det
                            sx1 = int(max(0, x1 * scale_x))
                            sy1 = int(max(0, y1 * scale_y))
                            sx2 = int(min(frame.shape[1], x2 * scale_x))
                            sy2 = int(min(frame.shape[0], y2 * scale_y))
                            if sx2 > sx1 and sy2 > sy1:
                                detections.append([sx1, sy1, sx2, sy2, float(score)])
                    else:
                        detections = self._detect_faces(frame)
                except Exception:
                    detections = []
                last_detections = list(detections)
                last_det_frame = frame_idx
            else:
                detections = list(last_detections) if (frame_idx - last_det_frame) <= MAX_DET_STALE_FRAMES else []

            tracked = self._run_tracker(detections, frame)
            for d in tracked:
                if len(d) < 5:
                    continue
                x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                track_id = int(d[4])
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1]))
                y2 = max(0, min(y2, frame.shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue

                with self.track_state_lock:
                    locked_name = self.track_id_to_name.get(track_id)
                    confidence = self.track_id_to_conf.get(track_id, 0.0)

                if not locked_name:
                    try:
                        face_region = rgb_frame[y1:y2, x1:x2]
                        if face_region.size != 0:
                            face_region = cv.resize(face_region, EMBEDDING_INPUT_SIZE, interpolation=cv.INTER_AREA)
                            self._enqueue_recognition_job(track_id, face_region)
                    except Exception:
                        pass

                predicted_name = locked_name or "Unknown"
                if predicted_name != "Unknown" or SHOW_UNKNOWN_OVERLAYS:
                    box_color = (0, 255, 0) if predicted_name != "Unknown" else (0, 0, 255)
                    cv.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{predicted_name} | {confidence:.2f}"
                    cv.putText(frame, label, (x1, max(20, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            self._set_latest_frame(frame, annotated=True)
            if self.show_window:
                cv.imshow(self._window_name(), frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()
                    break

        logger.info("[%s] Tracking thread stopped", self.camera_id)

    def _recognizer_loop(self):
        logger.info("[%s] Recognizer thread started", self.camera_id)
        while not self.stop_event.is_set() or not self.recognition_queue.empty():
            try:
                job = self.recognition_queue.get(timeout=0.1)
            except Empty:
                continue

            track_id = int((job or {}).get("track_id") or 0)
            seq = int((job or {}).get("seq") or 0)
            face_region = (job or {}).get("face_region")
            try:
                candidate_name, candidate_conf = ("Unknown", 0.0)
                if face_region is not None:
                    emb = get_embeddings(face_region)
                    candidate_name, candidate_conf = predict_identity(emb)

                with self.track_state_lock:
                    inflight_seq = self.track_id_inflight.get(track_id)
                    if inflight_seq == seq:
                        self.track_id_inflight.pop(track_id, None)

                    latest_seq = int(self.track_id_recog_seq.get(track_id, 0))
                    if seq != latest_seq:
                        continue

                    if candidate_name != "Unknown" and candidate_conf >= RECOGNITION_MIN_LOCK_CONF:
                        pending = self.track_id_pending.get(track_id)
                        if pending and pending.get("name") == candidate_name:
                            pending["count"] = int(pending.get("count", 0)) + 1
                            pending["conf"] = max(float(pending.get("conf", 0.0)), float(candidate_conf))
                        else:
                            pending = {"name": str(candidate_name), "count": 1, "conf": float(candidate_conf)}
                            self.track_id_pending[track_id] = pending

                        if int(pending.get("count", 0)) >= RECOGNITION_CONFIRM_FRAMES:
                            locked_name = str(pending.get("name") or "")
                            confidence = float(pending.get("conf", candidate_conf))
                            if locked_name:
                                self.track_id_to_name[track_id] = locked_name
                                self.track_id_to_conf[track_id] = confidence
                            self.track_id_pending.pop(track_id, None)
            except Exception:
                logger.exception("[%s] Recognizer job failed (track=%s, seq=%s)", self.camera_id, track_id, seq)
                self._clear_inflight_for_job(track_id, seq)
            finally:
                try:
                    self.recognition_queue.task_done()
                except Exception:
                    pass

        logger.info("[%s] Recognizer thread stopped", self.camera_id)


# ----------------------------
# Minimal main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Recognition-only runner")
    parser.add_argument("--source", default="", help="Camera source override")
    parser.add_argument("--name", default="default", help="Display name")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window")
    args = parser.parse_args()

    source_override = args.source.strip() if isinstance(args.source, str) else args.source
    if source_override == "":
        source_override = None

    source_cfg = normalize_video_source(source_override if source_override is not None else RTSP_URL)
    worker = CameraWorker(camera_id="default", source_cfg=source_cfg, display_name=args.name or "default", show_window=not args.no_window)
    worker.start()
    logger.info("Recognition-only started for source=%s", redact_source(str(source_cfg)))

    try:
        while worker.alive() and not worker.stop_event.is_set():
            time.sleep(MAIN_HEALTHCHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
