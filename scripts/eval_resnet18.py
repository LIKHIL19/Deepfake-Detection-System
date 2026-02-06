# scripts/eval_resnet18.py
# SAFE-IMPORT / SAFE-STREAMLIT:
# - This file will NOT run evaluation on import or when launched by Streamlit
#   unless you explicitly pass --run or set RUN_EVAL=1 in the environment.

import os, sys, time
from pathlib import Path
from contextlib import nullcontext

# make src/ importable (same as your train file)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FrameDataset
from src.model.resnet_model import get_resnet18

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')  # PyTorch 2.x
except Exception:
    pass


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")


def load_model(weights_path, num_classes, device, dropout=0.6):
    model = get_resnet18(num_classes=num_classes, dropout=dropout, pretrained=False).to(device)
    model.to(memory_format=torch.channels_last)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype):
    criterion = nn.CrossEntropyLoss()
    try:
        num_classes = model.fc.out_features
    except Exception:
        num_classes = 2

    total = correct = 0
    running_loss = 0.0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    correct_per_class = torch.zeros(num_classes, dtype=torch.int64)
    total_per_class   = torch.zeros(num_classes, dtype=torch.int64)

    start = time.time()
    autocast_ctx = (torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext())

    loop = tqdm(enumerate(loader), total=len(loader), desc="Testing", leave=False)
    for i, batch in loop:
        # (img, label) or (img, label, path)
        if len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(1)
        bs = labels.size(0)
        total += bs
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * bs

        for t, p in zip(labels.view(-1), preds.view(-1)):
            t = t.long(); p = p.long()
            confmat[t, p] += 1
            total_per_class[t] += 1
            if t == p:
                correct_per_class[t] += 1

        # ETA
        elapsed = time.time() - start
        done = i + 1
        avg_bt = elapsed / max(done, 1)
        remaining = (len(loader) - done) * avg_bt
        loop.set_postfix(acc=f"{(correct/max(total,1)):.4f}", eta=f"{remaining/60:.2f}m")

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    per_class_acc = [
        (correct_per_class[c].item() / total_per_class[c].item()) if total_per_class[c] > 0 else 0.0
        for c in range(num_classes)
    ]
    total_time_min = (time.time() - start) / 60.0

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "per_class_acc": [float(x) for x in per_class_acc],
        "confmat": confmat.cpu().numpy(),
        "time_min": float(total_time_min),
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate ResNet-18 on a test CSV (explicit-run only). "
                    "This script stays idle unless you pass --run or set RUN_EVAL=1."
    )
    # NOTE: None of these are required unless you actually run
    p.add_argument("--run", action="store_true", help="Actually run the evaluation")
    p.add_argument("--csv", type=str, default=None, help=r'Path to test CSV, e.g. C:\PROJECTS\Deepfake project\splits\test.csv')
    p.add_argument("--weights", type=str, default="best_model.pth", help="Model weights path")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", type=str2bool, default=True)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--face-crop", type=str2bool, default=True,
                   help="Runtime face crop (default True since your test frames are NOT pre-cropped).")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Explicit-run gating
    should_run = bool(args.run) or os.environ.get("RUN_EVAL", "0") == "1"
    if not should_run:
        print("[eval_resnet18] Idle: no evaluation triggered (omit --run for Streamlit/interactive use).")
        print("               To run from CLI: python scripts/eval_resnet18.py --run --csv <path> --weights <path>")
        return

    if not args.csv:
        raise SystemExit("Error: --csv is required when --run is provided (or set RUN_EVAL=1).")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"Using device: {device} | AMP: {use_amp} | dtype: {'bf16' if amp_dtype==torch.bfloat16 else 'fp16'}")
    print(f"Test CSV: {csv_path}\nWeights: {args.weights}\nFace crop: {args.face_crop}")

    test_dataset = FrameDataset(str(csv_path), augment=False, face_crop=args.face_crop)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    model = load_model(args.weights, num_classes=args.num_classes, device=device, dropout=args.dropout)
    results = evaluate(model, test_loader, device, use_amp, amp_dtype)

    print("\n=== Test Summary ===")
    print(f" Loss: {results['loss']:.4f}")
    print(f" Acc : {results['accuracy']:.4f}")
    print(f" Time: {results['time_min']:.2f} min")
    print(" Per-class Acc:")
    for i, a in enumerate(results["per_class_acc"]):
        print(f"  Class {i}: {a:.4f}")

    print("\n Confusion Matrix (rows=true, cols=pred):")
    with np.printoptions(suppress=True):
        print(results["confmat"])


if __name__ == "__main__":
    main()
