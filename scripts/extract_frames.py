import argparse
from pathlib import Path
import cv2
import random
from tqdm import tqdm
import math
import os
import yaml
import time
import csv


def load_config(cfg_path="config.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_videos(data_root: Path):
    """
    Collect exactly 2500 videos:
      - 1000 real from 'original'
      - 1500 fake (300 from each fake category)
    """
    all_videos = list(data_root.rglob("*.mp4"))

    fake_cats = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    real_cats = ["original"]

    selected = []

    # --- Real videos (1000) ---
    real_videos = [v for v in all_videos if any(c.lower() in [p.lower() for p in v.parts] for c in real_cats)]
    random.seed(42)
    random.shuffle(real_videos)
    selected.extend(real_videos[:1000])

    # --- Fake videos (300 per category = 1500 total) ---
    for cat in fake_cats:
        cat_videos = [v for v in all_videos if cat.lower() in [p.lower() for p in v.parts]]
        random.shuffle(cat_videos)
        selected.extend(cat_videos[:300])

    return selected


def extract_frames(video_path: Path, out_dir: Path, category: str, jpg_quality: int = 95):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 200:
        num_to_keep = total_frames
    else:
        if category == "fake":
            num_to_keep = 150
        else:  # real
            num_to_keep = 225
    step = max(1, total_frames // num_to_keep)
    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0 and saved < num_to_keep:
            out_path = out_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-videos", type=int, default=None, help="Limit number of videos (debug)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    out_root = Path(cfg["out_root"])
    jpg_quality = int(cfg.get("jpg_quality", 95))

    videos = collect_videos(data_root)
    if args.max_videos:
        videos = videos[: args.max_videos]

    logs_dir = ensure_dir(Path("logs"))
    log_file = logs_dir / "extraction_report.csv"
    write_header = not log_file.exists()

    with log_file.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["video", "category", "frames_saved", "time_sec"])

        pbar = tqdm(videos, desc="Extracting ALL (real+fake) frames", unit="video")
        total_saved = 0
        start_time = time.time()

        for idx, v in enumerate(pbar, 1):
            category = "fake"
            if "original" in [p.lower() for p in v.parts]:
                category = "real"

            vid_id = "__".join(v.relative_to(data_root).with_suffix("").parts)
            out_dir = ensure_dir(out_root / category / vid_id)

            t0 = time.time()
            saved = extract_frames(v, out_dir, category, jpg_quality)
            t1 = time.time()
            duration = round(t1 - t0, 2)

            total_saved += saved

            writer.writerow([str(v), category, saved, duration])

            elapsed = time.time() - start_time
            avg_per_video = elapsed / idx
            remaining = len(videos) - idx
            eta = remaining * avg_per_video / 60

            pbar.set_postfix({
                "frames": saved,
                "ETA(min)": f"{eta:.1f}"
            })

        print(f"Done. Extracted {len(videos)} videos → total frames saved: {total_saved}")
        print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
