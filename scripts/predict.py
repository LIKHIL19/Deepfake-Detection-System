import argparse, os, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Make src importable
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model.resnet_model import get_resnet18

# ---------- Face crop (same policy as training) ----------
def crop_face(pil_img: Image.Image, margin_ratio: float = 0.15) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        # Center square fallback
        w, h = pil_img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return pil_img.crop((left, top, left + s, top + s))
    x, y, w, h = sorted(list(faces), key=lambda b: b[2] * b[3], reverse=True)[0]
    m = int(margin_ratio * max(w, h))
    x0 = max(0, x - m); y0 = max(0, y - m)
    x1 = min(img.shape[1], x + w + m); y1 = min(img.shape[0], y + h + m)
    return Image.fromarray(img[y0:y1, x0:x1])

# ---------- Preprocess ----------
def get_transform(size=224):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

# ---------- Image prediction with optional TTA ----------
@torch.no_grad()
def predict_image(model, device, img: Image.Image, tta: bool = False):
    xform = get_transform(224)
    face = crop_face(img)
    x = xform(face).unsqueeze(0).to(device)
    model.eval()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type=='cuda'):
        logits = model(x)
        if tta:
            x_flip = torch.flip(x, dims=[3])
            logits = (logits + model(x_flip)) / 2.0
        prob = F.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
    cls = int(np.argmax(prob))
    return cls, prob  # 0=real, 1=fake

# ---------- Video prediction (sample N frames, mean logits) ----------
def sample_indices(num_frames, num_samples):
    if num_frames <= 0:
        return []
    if num_frames <= num_samples:
        return list(range(num_frames))
    step = num_frames / float(num_samples)
    return [int(i * step) for i in range(num_samples)]

@torch.no_grad()
def predict_video(model, device, video_path: str, frames: int = 48, tta: bool = False, batch: int = 32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(total, frames)
    xform = get_transform(224)
    tensors = []

    # Read selected frames
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:  # skip if read fails
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        face = crop_face(pil)
        tensors.append(xform(face))
    cap.release()

    if len(tensors) == 0:
        raise RuntimeError("No frames gathered for prediction.")

    # Batch through the network
    model.eval()
    logits_list = []
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type=='cuda'):
        for i in range(0, len(tensors), batch):
            x = torch.stack(tensors[i:i+batch]).to(device)
            out = model(x)
            if tta:
                x_flip = torch.flip(x, dims=[3])
                out = (out + model(x_flip)) / 2.0
            logits_list.append(out.detach())
    logits = torch.cat(logits_list, dim=0)                  # [F,2]
    mean_logits = logits.mean(dim=0, keepdim=True)          # [1,2]
    prob = F.softmax(mean_logits, dim=1)[0].float().cpu().numpy()
    cls = int(np.argmax(prob))  # 0=real, 1=fake
    return cls, prob, len(tensors)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="ResNet-18 Deepfake inference (image or video).")
    ap.add_argument("--input", required=True, help="Path to image or video")
    ap.add_argument("--weights", default="best_model.pth", help="Model weights (.pth)")
    ap.add_argument("--frames", type=int, default=48, help="Frames to sample for videos")
    ap.add_argument("--tta", action="store_true", help="Enable horizontal-flip TTA")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(num_classes=2, dropout=0.6, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(memory_format=torch.channels_last)

    path = Path(args.input)
    ext = path.suffix.lower()
    image_exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    video_exts = {".mp4",".avi",".mov",".mkv",".m4v",".webm"}

    if ext in image_exts:
        img = Image.open(path).convert("RGB")
        cls, prob = predict_image(model, device, img, tta=args.tta)
        label = "FAKE" if cls==1 else "REAL"
        print(f"[IMAGE] {path.name} → {label} (P(fake)={prob[1]:.4f})")
    elif ext in video_exts:
        cls, prob, used = predict_video(model, device, str(path), frames=args.frames, tta=args.tta)
        label = "FAKE" if cls==1 else "REAL"
        print(f"[VIDEO] {path.name} → {label}  | used_frames={used}  | P(fake)={prob[1]:.4f}")
    else:
        raise ValueError(f"Unknown file type: {ext}")

if __name__ == "__main__":
    main()
