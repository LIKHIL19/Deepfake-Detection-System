# scripts/train_resnet18.py
import sys, os, time
from pathlib import Path
from collections import Counter

# make src/ importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.dataset import FrameDataset
from src.model.resnet_model import get_resnet18

# ---- SPEED KNOBS ----
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ------------- CONFIG -------------
TRAIN_CSV = "splits/train_faces.csv"
VAL_CSV   = "splits/val_faces.csv"

BATCH_SIZE = 8
NUM_EPOCHS_THIS_RUN = 15
BASE_LR   = 5e-5                # backbone
HEAD_LR   = 5e-4                # head
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05

CHECKPOINT_FILE = "checkpoint.pth"
BEST_MODEL_FILE = "best_model.pth"

NUM_WORKERS = 2
PIN_MEMORY = True

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

USE_MIXUP = True
MIXUP_ALPHA = 0.2

# ------------- Sampler -------------
def build_balanced_sampler(dataset: FrameDataset):
    labels = [lab for _, lab, _ in dataset.samples]
    counts = Counter(labels)
    cls_w = {c: 1.0 / counts[c] for c in counts}
    sample_w = [cls_w[lab] for lab in labels]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

# ------------- MixUp -------------
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------- BN Freeze / Unfreeze -------------
def freeze_batchnorm(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def staged_unfreeze(model: nn.Module, stage: int):
    # stage0: freeze conv1/bn1/layer1; stage1: unfreeze layer2; stage2: unfreeze all
    if stage == 0:
        for n, p in model.named_parameters():
            if n.startswith(("conv1", "bn1", "layer1")):
                p.requires_grad = False
            else:
                p.requires_grad = True
    elif stage == 1:
        for n, p in model.named_parameters():
            if n.startswith(("conv1", "bn1", "layer1")):
                p.requires_grad = False
            elif n.startswith("layer2"):
                p.requires_grad = True
    elif stage == 2:
        for p in model.parameters():
            p.requires_grad = True

# ------------- Train / Eval -------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx, total_epochs, scaler):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    start_t = time.time()
    loop = tqdm(loader, desc=f"Train E{epoch_idx}/{total_epochs}", leave=False)

    autocast_ctx = (torch.amp.autocast('cuda', dtype=AMP_DTYPE) if USE_AMP
                    else torch.autocast(device_type='cpu', enabled=False))

    for images, labels, _ in loop:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            if USE_MIXUP:
                images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_correct += (outputs.argmax(1) == (labels if not USE_MIXUP else y_a)).sum().item()  # proxy
        running_total += bs
        loop.set_postfix(loss=f"{loss.item():.4f}")

    elapsed = time.time() - start_t
    train_loss = running_loss / max(running_total, 1)
    train_acc  = running_correct / max(running_total, 1)
    return train_loss, train_acc, elapsed

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, running_total = 0.0, 0, 0

    autocast_ctx = (torch.amp.autocast('cuda', dtype=AMP_DTYPE) if USE_AMP
                    else torch.autocast(device_type='cpu', enabled=False))

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_correct += (outputs.argmax(1) == labels).sum().item()
        running_total += bs

    val_loss = running_loss / max(running_total, 1)
    val_acc  = running_correct / max(running_total, 1)
    return val_loss, val_acc

# ------------- Main -------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "| AMP:", USE_AMP, "| dtype:", "bf16" if AMP_DTYPE==torch.bfloat16 else "fp16")

    # datasets & loaders (use pre-cropped faces; runtime face_crop OFF)
    train_dataset = FrameDataset(TRAIN_CSV, augment=True,  face_crop=False)
    val_dataset   = FrameDataset(VAL_CSV,   augment=False, face_crop=False)

    train_sampler = build_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=2
    )

    # model
    model = get_resnet18(num_classes=2, dropout=0.6, pretrained=True).to(device)
    model.to(memory_format=torch.channels_last)
    freeze_batchnorm(model)          # small-batch stability
    staged_unfreeze(model, stage=0)  # freeze very early layers initially

    # param groups
    back_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc" not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc" in n]

    optimizer = optim.AdamW([
        {"params": back_params, "lr": BASE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    # label-smoothing CE (sampler already balances classes; no class weights here)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # cosine schedule (epoch-wise)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_THIS_RUN, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # ----- Resume if checkpoint exists -----
    start_epoch = 0
    best_val_acc, best_val_loss = 0.0, float("inf")
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch   = ckpt.get("epoch", 0)
            best_val_acc  = ckpt.get("best_val_acc", 0.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f" Resuming from epoch {start_epoch} | best so far: "
                  f"val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")
        except Exception as e:
            print(" Could not fully load checkpoint; starting fresh.", e)

    total_start = time.time()

    try:
        for e in range(start_epoch + 1, NUM_EPOCHS_THIS_RUN + 1):
            # staged unfreeze
            if e == 2: staged_unfreeze(model, stage=1)
            if e == 3: staged_unfreeze(model, stage=2)

            epoch_start = time.time()

            train_loss, train_acc, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, e, NUM_EPOCHS_THIS_RUN, scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            scheduler.step()  # cosine schedule per epoch

            # save checkpoint every epoch
            torch.save({
                "epoch": e,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss
            }, CHECKPOINT_FILE)

            # save best by val accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), BEST_MODEL_FILE)
                print(" Saved new BEST model (by val_acc)")

            # epoch summary + ETA
            epoch_time = time.time() - epoch_start
            remaining = (NUM_EPOCHS_THIS_RUN - e) * epoch_time
            print(f"\nEpoch {e}/{NUM_EPOCHS_THIS_RUN} — Time: {epoch_time/60:.2f} min | ETA (this run): {remaining/60:.2f} min")
            print(f"  Train  Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val    Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f"  LRs: {[g['lr'] for g in optimizer.param_groups]}")

    except KeyboardInterrupt:
        # Save a last-safe checkpoint and exit cleanly
        last_epoch = max(start_epoch, e - 1 if 'e' in locals() else 0)
        torch.save({
            "epoch": last_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss
        }, CHECKPOINT_FILE)
        print("\n Interrupted. Saved checkpoint. Re-run to resume.")
        return

    total_time = (time.time() - total_start) / 60
    print(f"\nRun complete in {total_time:.1f} min. Best so far → val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
