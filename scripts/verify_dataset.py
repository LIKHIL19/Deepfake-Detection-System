import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import yaml


def load_config(cfg_path: str | Path = "config.yaml") -> dict:
    cfg_path = Path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ["data_root", "out_root", "metadata_csv"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = str(Path(cfg[k]))
    return cfg


def scan_videos(root: Path) -> list[Path]:
    return list(root.rglob("*.mp4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    videos = scan_videos(data_root)
    print(f"Found {len(videos)} .mp4 files under: {data_root}\n")

    # Count categories based on folder names
    buckets = Counter()
    for v in videos:
        parts = [str(p).lower() for p in v.parts]
        for m in ["deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter", "original"]:
            if m in parts:
                buckets[m] += 1
                break
    if buckets:
        print("Approx. category counts:")
        for k, v in buckets.items():
            print(f"  {k:<20} {v}")
        print()

    # Metadata CSV check
    if cfg.get("metadata_csv"):
        meta = Path(cfg["metadata_csv"])
        if not meta.exists():
            print(f"WARNING: metadata_csv does not exist: {meta}")
        else:
            df = pd.read_csv(meta)
            col = None
            for c in ["relpath", "path", "filename", "video_path"]:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                print("Metadata CSV loaded but no path-like column found.")
            else:
                missing = []
                for rp in df[col].astype(str):
                    cand = data_root / rp
                    if not cand.exists():
                        missing.append(rp)
                print(f"Metadata rows: {len(df)}  |  Missing files: {len(missing)}")
                if missing[:10]:
                    print("First missing examples:")
                    for m in missing[:10]:
                        print("  -", m)


if __name__ == "__main__":
    main()