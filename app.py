# app.py — Minimal, production-style UI for Deepfake Detection (ResNet-18)
# Works with your best_model.pth and your src/model/resnet_model.py

import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

# Make your "src" importable (so we can import your exact model)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.model.resnet_model import get_resnet18  # your training architecture

# ---------------- UI basics ----------------
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
st.markdown("<h2>🛡️ Deepfake Detector</h2>", unsafe_allow_html=True)
st.caption("Upload a single image (.jpg/.png/.webp) or a single video (.mp4/.avi/.mov/.mkv).")

# ---------------- constants ----------------
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_EXTS  = {"jpg","jpeg","png","bmp","webp"}
VID_EXTS  = {"mp4","avi","mov","mkv","m4v","webm"}

# ---------------- face crop (same policy as training) ----------------
@st.cache_resource
def _face_cascade():
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(xml))

def _center_square(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    s = min(w, h)
    x0 = (w - s) // 2
    y0 = (h - s) // 2
    return pil_img.crop((x0, y0, x0 + s, y0 + s))

def crop_face_pil(pil_img: Image.Image, margin_ratio: float = 0.18, min_face: int = 40) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face, min_face))
    if len(faces) == 0:
        face = np.array(_center_square(pil_img))
    else:
        x, y, w, h = sorted(list(faces), key=lambda b: b[2]*b[3], reverse=True)[0]
        m = int(margin_ratio * max(w, h))
        x0, y0 = max(0, x-m), max(0, y-m)
        x1, y1 = min(img.shape[1], x+w+m), min(img.shape[0], y+h+m)
        face = img[y0:y1, x0:x1]
    return Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

@st.cache_resource
def get_transform():
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# ---------------- model loading ----------------
@st.cache_resource
def load_model(weights_path: str, device: torch.device):
    model = get_resnet18(num_classes=2, dropout=0.6, pretrained=False).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model", "model_state"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
    model.load_state_dict(state or ckpt, strict=False)
    model.eval().to(memory_format=torch.channels_last)
    return model

def get_device():
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- inference helpers ----------------
@torch.inference_mode()
def predict_image(model, device, pil_img: Image.Image, tta: bool = True):
    x = get_transform()(crop_face_pil(pil_img)).unsqueeze(0).to(device, memory_format=torch.channels_last)
    logits = model(x)
    if tta:
        logits = (logits + model(T.functional.hflip(x))) / 2.0
    probs = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()
    return int(probs.argmax()), probs  # 0=REAL, 1=FAKE

def sample_indices(total: int, num: int):
    if total <= 0: return []
    if num >= total: return list(range(total))
    return list(np.linspace(0, total-1, num=num, dtype=int))

@torch.inference_mode()
def predict_video(model, device, tmp_path: Path, frames: int = 48, tta: bool = True, progress=None):
    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened(): raise RuntimeError("Failed to open video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = sample_indices(total, frames)
    logits_accum, used = None, 0
    tr = get_transform()
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = crop_face_pil(pil)
        x = tr(face).unsqueeze(0).to(device, memory_format=torch.channels_last)
        l = model(x)
        if tta: l = (l + model(T.functional.hflip(x))) / 2.0
        logits_accum = l if logits_accum is None else (logits_accum + l)
        used += 1
        if progress: progress.progress((i+1)/max(1,len(idxs)))
    cap.release()
    if used == 0: raise RuntimeError("Could not read frames.")
    probs = torch.softmax(logits_accum/used, dim=1).squeeze(0).float().cpu().numpy()
    return int(probs.argmax()), probs, used

# ---------------- sidebar (light) ----------------
with st.sidebar:
    weights = st.text_input("Weights (.pth)", value="best_model.pth")
    frames = st.slider("Frames for videos", 8, 128, 48, 8)
    tta = st.checkbox("TTA (flip)", value=True)
    threshold = st.slider("FAKE threshold", 0.10, 0.90, 0.50, 0.01)

# ---------------- single uploader ----------------
uploaded = st.file_uploader("Drag & drop an image or a video", type=list(IMG_EXTS | VID_EXTS))
run = st.button("🔍 Run Prediction")

# ---------------- run ----------------
# ---------------- run ----------------
if run:
    if uploaded is None:
        st.warning("Please upload one file first.")
        st.stop()
    if not Path(weights).exists():
        st.error(f"Weights file not found: {weights}")
        st.stop()

    device = get_device()
    model = load_model(weights, device)

    name = uploaded.name
    ext = name.split(".")[-1].lower()

    # common 2-column layout for preview (left) + results (right)
    preview_col, result_col = st.columns([1.1, 1.9])

    # IMAGE
    if ext in IMG_EXTS:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        with preview_col:
            st.subheader("Preview")
            # fixed-width preview — does NOT stretch
            st.image(img, width=340)

        with result_col:
            with st.spinner("Predicting…"):
                cls, probs = predict_image(model, device, img, tta=tta)
            prob_real, prob_fake = float(probs[0]), float(probs[1])
            label = "FAKE" if prob_fake >= float(threshold) else "REAL"

            st.subheader("Result")
            st.success(f"Prediction: {label}")
            st.progress(min(1.0, prob_fake), text=f"P(fake) = {prob_fake:.4f}")
            st.caption(f"Real: {prob_real:.4f} • Fake: {prob_fake:.4f} • Threshold: {threshold:.2f}")

    # VIDEO
    elif ext in VID_EXTS:
        tmp = ROOT / "._tmp_upload_video"
        tmp.write_bytes(uploaded.read())

        with preview_col:
            st.subheader("Preview")
            # rendered inside the narrow left column -> smaller, contained
            st.video(str(tmp))

        with result_col:
            st.subheader("Result")
            prog = st.progress(0.0, text="Processing frames…")
            with st.spinner("Predicting…"):
                cls, probs, used = predict_video(model, device, tmp, frames=frames, tta=tta, progress=prog)
            tmp.unlink(missing_ok=True)

            prob_real, prob_fake = float(probs[0]), float(probs[1])
            label = "FAKE" if prob_fake >= float(threshold) else "REAL"

            st.success(f"Video prediction: {label}")
            st.caption(f"Frames used: {used} • Real: {prob_real:.4f} • Fake: {prob_fake:.4f} • Threshold: {threshold:.2f}")
            st.progress(min(1.0, prob_fake), text=f"P(fake) = {prob_fake:.4f}")

    else:
        st.error("Unsupported file type.")

