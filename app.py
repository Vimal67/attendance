from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import cv2 as cv
import numpy as np
import onnxruntime as ort
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
ARCFACE_MODEL_PATH = MODEL_DIR / "arcface_w600k_r50.onnx"
SVM_MODEL_PATH = MODEL_DIR / "svm_model_160x160.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

EMBEDDING_SIZE = (112, 112)
DETECTION_SCORE_THRESHOLD = 0.70
SVM_CONF_THRESHOLD = 0.50

app = FastAPI(title="Live Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None
ort_session = None
arcface_input_name = None
arcface_output_name = None
svm_model = None
label_encoder = None


def _assert_model_files():
    required = [
        YUNET_MODEL_PATH,
        ARCFACE_MODEL_PATH,
        SVM_MODEL_PATH,
        LABEL_ENCODER_PATH,
    ]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model files in ./model: " + ", ".join(missing)
        )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def load_models():
    global detector, ort_session, arcface_input_name, arcface_output_name, svm_model, label_encoder

    if detector is not None:
        return

    _assert_model_files()

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


def get_embedding(face_rgb: np.ndarray) -> np.ndarray:
    face = cv.resize(face_rgb, EMBEDDING_SIZE, interpolation=cv.INTER_AREA)
    face = face.astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0).astype(np.float32)

    output = ort_session.run([arcface_output_name], {arcface_input_name: face})[0]
    output = np.asarray(output, dtype=np.float32)
    return l2_normalize(output)


def predict_identity(embedding: np.ndarray):
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


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Live Face Recognition</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111; color:#fff; margin:0; padding:20px; text-align:center; }
    .wrap { max-width:700px; margin:auto; }
    .video-box { position:relative; display:inline-block; width:100%; max-width:640px; }
    video, canvas.overlay { width:100%; border-radius:12px; border:2px solid #333; margin-top:12px; }
    canvas.overlay { position:absolute; left:0; top:0; pointer-events:none; }
    #status { margin-top:16px; font-size:18px; min-height:28px; }
    button { margin:8px; padding:12px 18px; border:none; border-radius:10px; font-size:16px; cursor:pointer; }
    .note { color:#bbb; font-size:14px; margin-top:10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Live Face Recognition</h2>
    <div class="video-box">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay" class="overlay"></canvas>
    </div>
    <canvas id="capture" style="display:none;"></canvas>
    <div id="status">Starting camera...</div>
    <div>
      <button id="startBtn">Start Detection</button>
      <button id="stopBtn">Stop Detection</button>
    </div>
    <div class="note">This is sampled live detection. One frame is sent every ~700 ms.</div>
  </div>

  <script>
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const capture = document.getElementById("capture");
    const statusDiv = document.getElementById("status");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");

    let stream = null;
    let intervalId = null;
    let isSending = false;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false
        });
        video.srcObject = stream;
        statusDiv.innerText = "Camera ready";
      } catch (err) {
        statusDiv.innerText = "Camera access denied or not available";
        console.error(err);
      }
    }

    function drawDetections(detections) {
      if (!video.videoWidth || !video.videoHeight) return;

      overlay.width = video.clientWidth;
      overlay.height = video.clientHeight;

      const scaleX = overlay.width / video.videoWidth;
      const scaleY = overlay.height / video.videoHeight;

      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      ctx.lineWidth = 2;
      ctx.font = "16px Arial";

      detections.forEach(det => {
        const x = det.box.x1 * scaleX;
        const y = det.box.y1 * scaleY;
        const w = (det.box.x2 - det.box.x1) * scaleX;
        const h = (det.box.y2 - det.box.y1) * scaleY;

        ctx.strokeStyle = det.name === "Unknown" ? "#ff3b30" : "#00ff66";
        ctx.fillStyle = ctx.strokeStyle;
        ctx.strokeRect(x, y, w, h);

        const label = `${det.name} (${det.confidence})`;
        const textWidth = ctx.measureText(label).width + 10;
        ctx.fillRect(x, Math.max(0, y - 24), textWidth, 22);
        ctx.fillStyle = "#000";
        ctx.fillText(label, x + 5, Math.max(16, y - 8));
      });
    }

    async function sendFrame() {
      if (isSending || !video.videoWidth || !video.videoHeight) return;
      isSending = true;

      capture.width = video.videoWidth;
      capture.height = video.videoHeight;

      const ctx = capture.getContext("2d");
      ctx.drawImage(video, 0, 0, capture.width, capture.height);

      capture.toBlob(async (blob) => {
        try {
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          const response = await fetch("/recognize", {
            method: "POST",
            body: formData
          });

          const data = await response.json();

          if (data.detections && data.detections.length > 0) {
            const top = data.detections[0];
            statusDiv.innerText = `Name: ${top.name} | Confidence: ${top.confidence}`;
            drawDetections(data.detections);
          } else {
            statusDiv.innerText = "No face detected";
            drawDetections([]);
          }
        } catch (err) {
          console.error(err);
          statusDiv.innerText = "Backend error";
        } finally {
          isSending = false;
        }
      }, "image/jpeg", 0.8);
    }

    startBtn.addEventListener("click", () => {
      if (!intervalId) {
        intervalId = setInterval(sendFrame, 700);
        statusDiv.innerText = "Live detection started";
      }
    });

    stopBtn.addEventListener("click", () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
        statusDiv.innerText = "Detection stopped";
      }
      drawDetections([]);
    });

    video.addEventListener("loadedmetadata", () => {
      overlay.width = video.clientWidth;
      overlay.height = video.clientHeight;
    });

    startCamera();
  </script>
</body>
</html>
"""


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv.imdecode(arr, cv.IMREAD_COLOR)

    if bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)

    if faces is None or len(faces) == 0:
        return {"detections": []}

    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    detections = []

    for row in faces:
        x, y, fw, fh = row[:4]
        score = float(row[14]) if len(row) > 14 else 0.0

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + fw))
        y2 = min(h, int(y + fh))

        if x2 <= x1 or y2 <= y1:
            continue

        face_rgb = rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            continue

        emb = get_embedding(face_rgb)
        name, conf = predict_identity(emb)

        detections.append({
            "name": name,
            "confidence": round(conf, 4),
            "detector_score": round(score, 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {"detections": detections}
