import argparse
from pathlib import Path
import random
import csv
import yaml
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(cfg_path="config.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["out_root"])
    splits_dir = ensure_dir("splits")

    # Collect all video frame folders
    real_dirs = [p for p in (out_root / "real").rglob("*") if p.is_dir()]
    fake_dirs = [p for p in (out_root / "fake").rglob("*") if p.is_dir()]

    # Assign labels
    all_data = [(d.name, str(d.resolve()), 0) for d in real_dirs] + \
               [(d.name, str(d.resolve()), 1) for d in fake_dirs]

    # Shuffle
    random.seed(42)
    random.shuffle(all_data)

    n = len(all_data)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_set = all_data[:n_train]
    val_set = all_data[n_train:n_train+n_val]
    test_set = all_data[n_train+n_val:]

    # Write CSVs
    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "frames_dir", "label"])
            writer.writerows(rows)

    write_csv(splits_dir / "train.csv", train_set)
    write_csv(splits_dir / "val.csv", val_set)
    write_csv(splits_dir / "test.csv", test_set)

    print(f"Total videos: {n}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(f"CSV files saved in {splits_dir}/")


if __name__ == "__main__":
    main()
