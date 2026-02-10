import torch
from torch.utils.data import DataLoader
from src.data.dataset import FrameDataset
import sys
from pathlib import Path

# Add project root (parent of "src") to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

train_dataset = FrameDataset("splits/train.csv", train=True)
val_dataset = FrameDataset("splits/val.csv", train=False)

# Use batch_size=1 because each video can have 150–225 frames
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Check one sample
images, label = next(iter(train_loader))
print("Images shape:", images.shape)   # [1, Nframes, 3, 224, 224]
print("Label:", label)
