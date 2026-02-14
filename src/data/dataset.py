import os
from glob import glob
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FrameDataset(Dataset):
    """
    CSV schemas supported:
      A) Per-frame: one of ['frame_path','path','filename'] + 'video_id','label'
      B) Per-video: 'frames_dir' + 'video_id','label' (expands *.jpg recursively)

    Returns: (image_tensor, label_int, video_id_str)
    """

    def __init__(self, csv_file: str, augment: bool = False, face_crop: bool = True):
        self.df = pd.read_csv(csv_file)
        self.augment = augment
        self.face_crop = face_crop

        # ---- Build sample list (detect schema) ----
        path_cols = [c for c in ["frame_path", "path", "filename"] if c in self.df.columns]
        self.samples = []

        if path_cols:
            pcol = path_cols[0]
            if "video_id" not in self.df.columns or "label" not in self.df.columns:
                raise KeyError("Per-frame CSV must include 'video_id' and 'label'.")
            for _, row in self.df.iterrows():
                self.samples.append((str(row[pcol]), int(row["label"]), str(row["video_id"])))
        elif "frames_dir" in self.df.columns:
            if "video_id" not in self.df.columns or "label" not in self.df.columns:
                raise KeyError("Per-video CSV must include 'video_id' and 'label'.")
            for _, row in self.df.iterrows():
                vdir = str(row["frames_dir"])
                jpgs = sorted(glob(os.path.join(vdir, "*.jpg")))
                if not jpgs:
                    jpgs = sorted(glob(os.path.join(vdir, "**", "*.jpg"), recursive=True))
                for fp in jpgs:
                    self.samples.append((fp, int(row["label"]), str(row["video_id"])))
        else:
            raise KeyError("CSV must include either 'frames_dir' or one of ['frame_path','path','filename'].")

        if not self.samples:
            raise RuntimeError("No frames found from CSV. Check your paths and extracted frames.")

        # ---- Transforms ----
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.base_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            self.norm
        ])

        # Stronger, video-artifact-friendly augmentation
        self.augment_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),
            self.norm
        ])

        # ---- Face detector (lazy & picklable) ----
        self._cascade_path: Optional[str] = None
        self._face_detector = None
        if self.face_crop:
            self._cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )

    # Multiprocessing safety (Windows pickling)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_face_detector"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._face_detector = None

    def _ensure_detector(self):
        if self._face_detector is None and self._cascade_path is not None:
            self._face_detector = cv2.CascadeClassifier(self._cascade_path)

    def _crop_face(self, img_pil: Image.Image) -> Image.Image:
        if not self.face_crop:
            return img_pil
        self._ensure_detector()
        if self._face_detector is None:
            return img_pil

        img = np.array(img_pil.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            # Center square fallback
            w, h = img_pil.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            return img_pil.crop((left, top, left + s, top + s))

        x, y, w, h = sorted(list(faces), key=lambda b: b[2] * b[3], reverse=True)[0]
        m = int(0.15 * max(w, h))
        x0 = max(0, x - m); y0 = max(0, y - m)
        x1 = min(img.shape[1], x + w + m); y1 = min(img.shape[0], y + h + m)
        return Image.fromarray(img[y0:y1, x0:x1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        fp, label, vid = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        img = self._crop_face(img)
        x = self.augment_transform(img) if self.augment else self.base_transform(img)
        return x, int(label), str(vid)
