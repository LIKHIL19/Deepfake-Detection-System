# scripts/eval_test.py
# Evaluate ResNet18 on test.csv (raw frames, no pre-crop in CSV)
# Usage:
#   python scripts/eval_test.py --csv "C:\PROJECTS\Deepfake project\splits\test.csv" --weights .\best_model.pth --tta

import argparse
import sys
from pathlib import Path
import io
import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

# make project "src" importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.model.resnet_model import get_resnet18

# ---------- constants ----------
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES = ["REAL", "FAKE"]  # model output index 0=REAL, 1=FAKE

# ---------- face crop ----------
def _face_cascade():
    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(xml))

def center_square(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    s = min(w, h); x0=(w-s)//2; y0=(h-s)//2
    return pil_img.crop((x0,y0,x0+s,y0+s))

def crop_face_pil(pil_img: Image.Image, margin_ratio: float = 0.18, min_face: int = 40) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face, min_face))
    if len(faces) == 0:
        face = np.array(center_square(pil_img))
    else:
        x, y, w, h = sorted(list(faces), key=lambda b: b[2]*b[3], reverse=True)[0]
        m = int(margin_ratio * max(w, h))
        x0, y0 = max(0, x-m), max(0, y-m)
        x1, y1 = min(img.shape[1], x+w+m), min(img.shape[0], y+h+m)
        face = img[y0:y1, x0:x1]
    return Image.fromarray(face).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

TRANSFORM = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# ---------- CSV helpers ----------
PATH_COL_CANDIDATES = ["path","frame","frame_path","filepath","image","img","img_path","file","filename"]
LABEL_COL_CANDIDATES = ["label","target","y","class","is_fake","is_real"]
VID_COL_CANDIDATES = ["video_id","video","vid","clip","source","filename_no_ext"]

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in low: return low[c]
    return None

def infer_and_align_labels(df, path_col, label_col):
    """
    Ensure labels match model indices (0=REAL, 1=FAKE).
    If CSV uses opposite convention, flip.
    """
    labels = df[label_col].astype(int).to_numpy().copy()
    # Look at up to 200 rows and guess by path text
    sample = df.head(min(200, len(df)))
    votes = 0
    for _, r in sample.iterrows():
        p = str(r[path_col]).lower()
        lbl = int(r[label_col])
        # If file path contains 'fake', model's FAKE index is 1.
        # We vote +1 when lbl==1 AND 'fake' in path, or lbl==0 AND 'real' in path.
        if "fake" in p and lbl == 1: votes += 1
        if "real" in p and lbl == 0: votes += 1
    # If votes are low (< 60%), flip labels
    ratio = votes / max(1, len(sample))
    flipped = False
    if ratio < 0.60:
        labels = 1 - labels
        flipped = True
    return labels, flipped

def default_group_from_path(path_str: str):
    """
    A heuristic video id from the path:
    use the parent folder name (the one right above the filename).
    """
    p = Path(path_str)
    return p.parent.name

# ---------- Dataset ----------
class TestFramesDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, path_col: str, label_arr: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.paths = self.df[path_col].tolist()
        self.labels = label_arr.astype(np.int64)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        face = crop_face_pil(im)
        x = TRANSFORM(face)
        return x, y, p  # return path for grouping later

# ---------- metrics ----------
def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    out["precision"] = float(p); out["recall"] = float(r); out["f1"] = float(f1)
    # probs[:,1] is P(fake)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:,1]))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob[:,1]))
    except Exception:
        out["pr_auc"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return out

# ---------- eval ----------
@torch.inference_mode()
def run_eval(csv_path: str, weights_path: str, batch_size: int = 64, tta: bool = True, video_group: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    path_col = pick_column(df, PATH_COL_CANDIDATES)
    label_col = pick_column(df, LABEL_COL_CANDIDATES)
    if path_col is None or label_col is None:
        raise ValueError(f"Could not find path/label columns in CSV. Have columns: {list(df.columns)}")

    # Ensure absolute paths
    df[path_col] = df[path_col].apply(lambda p: str(Path(p).resolve()))

    labels, flipped = infer_and_align_labels(df, path_col, label_col)
    if flipped:
        print("⚠️ Detected label convention mismatch; flipped labels to match model (0=REAL,1=FAKE).")

    ds = TestFramesDataset(df, path_col, labels)
    # num_workers=0 to avoid Windows PyTorch multiprocess pitfalls with OpenCV objects
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    from src.model.resnet_model import get_resnet18
    model = get_resnet18(num_classes=2, dropout=0.6, pretrained=False).to(device).eval().to(memory_format=torch.channels_last)
    ckpt = torch.load(weights_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model", "model_state"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
    model.load_state_dict(state or ckpt, strict=False)

    all_probs = []
    all_preds = []
    all_y = []
    all_paths = []

    for x, y, p in loader:
        x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
        logits = model(x)
        if tta:
            logits = (logits + model(T.functional.hflip(x))) / 2.0
        probs = torch.softmax(logits, dim=1).float().cpu().numpy()
        pred = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.append(pred)
        all_y.append(y.numpy())
        all_paths += list(p)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    print("\n=== Frame-level metrics ===")
    frame_metrics = compute_metrics(y_true, y_pred, y_prob)
    for k,v in frame_metrics.items():
        if k!="confusion_matrix":
            print(f"{k:>18}: {v:.4f}")
    print("confusion_matrix:", frame_metrics["confusion_matrix"])

    # ----- Video-level aggregation -----
    # choose group column: existing CSV col or derive from path parent folder
    if video_group is None:
        vid_col = pick_column(df, VID_COL_CANDIDATES)
    else:
        vid_col = video_group if video_group in df.columns else None

    groups = defaultdict(list)
    for i, p in enumerate(all_paths):
        if vid_col is not None:
            g = str(df.iloc[i][vid_col])
        else:
            g = default_group_from_path(p)
        groups[g].append(i)

    vid_y_true, vid_y_pred, vid_y_prob = [], [], []
    for g, idxs in groups.items():
        probs_mean = y_prob[idxs].mean(axis=0)  # average logits->softmax or average probs; we use probs (ok for eval)
        vid_y_prob.append(probs_mean)
        vid_y_pred.append(int(np.argmax(probs_mean)))
        # video label = majority frame label (or just the first's ground truth)
        vid_y_true.append(int(np.round(y_true[idxs].mean())))  # robust if CSV has consistent labels

    vid_y_true = np.array(vid_y_true)
    vid_y_pred = np.array(vid_y_pred)
    vid_y_prob = np.vstack(vid_y_prob)

    print("\n=== Video-level metrics ===")
    video_metrics = compute_metrics(vid_y_true, vid_y_pred, vid_y_prob)
    for k,v in video_metrics.items():
        if k!="confusion_matrix":
            print(f"{k:>18}: {v:.4f}")
    print("confusion_matrix:", video_metrics["confusion_matrix"])

    # Optional: write a small report next to CSV
    out_txt = Path(csv_path).with_name("test_eval_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Frame-level:\n")
        for k,v in frame_metrics.items():
            if k!="confusion_matrix":
                f.write(f"{k}: {v:.6f}\n")
        f.write(f"confusion_matrix: {frame_metrics['confusion_matrix']}\n\n")
        f.write("Video-level:\n")
        for k,v in video_metrics.items():
            if k!="confusion_matrix":
                f.write(f"{k}: {v:.6f}\n")
        f.write(f"confusion_matrix: {video_metrics['confusion_matrix']}\n")
    print(f"\nSaved summary → {out_txt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to test.csv (raw frames)")
    ap.add_argument("--weights", default="best_model.pth", help="Model weights .pth")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", action="store_true", help="Enable simple TTA (H-flip)")
    ap.add_argument("--video_group", default=None, help="Optional CSV column to group frames into videos")
    args = ap.parse_args()
    run_eval(args.csv, args.weights, batch_size=args.batch, tta=args.tta, video_group=args.video_group)

if __name__ == "__main__":
    main()
