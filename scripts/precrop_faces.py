# scripts/precrop_faces_v2.py
import os, sys
from pathlib import Path
from glob import glob
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

def crop_face(pil_img: Image.Image) -> Image.Image:
    """Largest-face crop with small margin; center square fallback."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        # center square
        w, h = pil_img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return pil_img.crop((left, top, left + s, top + s))
    x, y, w, h = sorted(list(faces), key=lambda b: b[2] * b[3], reverse=True)[0]
    m = int(0.15 * max(w, h))
    x0 = max(0, x - m); y0 = max(0, y - m)
    x1 = min(img.shape[1], x + w + m); y1 = min(img.shape[0], y + h + m)
    return Image.fromarray(img[y0:y1, x0:x1])

def sanitize_rel_path(p: Path) -> Path:
    # Make a safe mirror path under out_root (handles Windows drive letters etc.)
    rel = p.as_posix().replace(":/", "/").replace(":", "")
    return Path(rel)

def expand_rows_from_csv(df: pd.DataFrame):
    """
    Returns a list of (frame_path:str, video_id:str, label:int) from either schema:
    - Per-frame: one of ['frame_path','path','filename'] + 'video_id','label'
    - Per-video: 'frames_dir' + 'video_id','label' (expands *.jpg recursively)
    """
    rows = []
    path_cols = [c for c in ["frame_path", "path", "filename"] if c in df.columns]
    if path_cols:
        pcol = path_cols[0]
        for _, r in df.iterrows():
            rows.append((str(r[pcol]), str(r["video_id"]), int(r["label"])))
    elif "frames_dir" in df.columns:
        for _, r in df.iterrows():
            vdir = Path(str(r["frames_dir"]))
            # Find jpgs (flat or nested)
            jpgs = sorted(glob(str(vdir / "*.jpg")))
            if not jpgs:
                jpgs = sorted(glob(str(vdir / "**" / "*.jpg"), recursive=True))
            for fp in jpgs:
                rows.append((fp, str(r["video_id"]), int(r["label"])))
    else:
        raise KeyError("CSV must include 'frames_dir' or one of ['frame_path','path','filename'] plus 'video_id','label'.")
    if not rows:
        raise RuntimeError("No frames found. Check CSV paths / frame extraction.")
    return rows

def process_csv(csv_in: str, csv_out: str, out_root: str):
    df = pd.read_csv(csv_in)
    rows = expand_rows_from_csv(df)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    new_records = []
    skipped = 0

    for src_path, vid, lab in tqdm(rows, desc=f"Cropping faces for {csv_in}"):
        try:
            src_p = Path(src_path)
            # mirror original path under out_root
            save_p = out_root / sanitize_rel_path(src_p)
            save_p.parent.mkdir(parents=True, exist_ok=True)

            if not save_p.exists():
                with Image.open(src_p).convert("RGB") as im:
                    face = crop_face(im)
                    face.save(save_p, quality=95)
            new_records.append((save_p.as_posix(), vid, lab))
        except Exception as e:
            skipped += 1

    out_df = pd.DataFrame(new_records, columns=["frame_path", "video_id", "label"])
    out_df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out} | rows: {len(out_df)} | skipped: {skipped}")

if __name__ == "__main__":
    out_root = "preprocessed/faces"
    Path(out_root).mkdir(parents=True, exist_ok=True)

    process_csv("splits/train.csv", "splits/train_faces.csv", out_root)
    process_csv("splits/val.csv",   "splits/val_faces.csv",   out_root)
    print("Done. Now use splits/train_faces.csv and splits/val_faces.csv in training.")
