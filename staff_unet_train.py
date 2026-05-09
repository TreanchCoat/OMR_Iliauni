"""
staff_unet_train.py — Train the U-Net on the CVC-MUSCIMA manifest.

Run with sensible defaults:
    python staff_unet_train.py muscima_manifest.json

Or with explicit parameters:
    python staff_unet_train.py muscima_manifest.json \
        --epochs 30 --batch-size 4 --size 512 --lr 1e-4 \
        --output models/staff_unet.pth

Tested on
---------
NVIDIA RTX 3050 (4 GB):  batch=2 at 512×512  → ~3 hours for 30 epochs
Larger GPU (8 GB+):       batch=8 at 512×512  → ~1 hour for 30 epochs
CPU-only (not recommended): batch=1 at 256×256 → many hours
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from staff_unet import UNet, DiceBCELoss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class StafflineDataset(Dataset):
    """
    Loads (image, staff-line-mask) pairs from a manifest entry list.

    The CVC-MUSCIMA `gt/` images are pre-binarised staff-only masks where
    BLACK pixels (== 0) are staff lines and WHITE (== 255) is background —
    we invert that here so positives are 1.0 and background is 0.0.
    """

    def __init__(self, entries: list, size: int = 512, augment: bool = False):
        self.entries = entries
        self.size = size
        self.augment = augment

        if augment:
            try:
                import albumentations as A
            except ImportError:
                raise ImportError('Install albumentations:  pip install albumentations')
            self.aug = A.Compose([
                A.Rotate(limit=8, p=0.7),
                A.Perspective(scale=(0.02, 0.08), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = cv2.imread(entry['image_path'])
        if image is None:
            raise RuntimeError(f'Cannot read image: {entry["image_path"]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Cannot read mask: {entry["mask_path"]}')

        # CVC-MUSCIMA convention: BLACK background, WHITE = ink (staff lines /
        # music symbols).  Real-world scanned scores use the opposite
        # convention: WHITE paper, BLACK ink.  Invert the input image so the
        # model trains on what real scans actually look like — that way no
        # inference-time inversion is needed.
        image = 255 - image

        # Resize to model input size
        image = cv2.resize(image, (self.size, self.size),
                            interpolation=cv2.INTER_AREA)
        mask  = cv2.resize(mask,  (self.size, self.size),
                            interpolation=cv2.INTER_NEAREST)

        # Mask: WHITE (255) = staff line in CVC-MUSCIMA gt files.
        # We want 1.0 for staff line, 0.0 for background.
        mask = (mask >= 128).astype(np.float32)

        if self.aug is not None:
            out = self.aug(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        # → tensors
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Intersection-over-union for binary segmentation."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection
    return (intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    manifest:    str
    output:      str   = 'models/staff_unet.pth'
    epochs:      int   = 30
    batch_size:  int   = 2
    size:        int   = 512
    lr:          float = 1e-4
    num_workers: int   = 2
    val_every:   int   = 1
    augment:     bool  = True


def train(cfg: TrainConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # Load manifest
    with open(cfg.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    print(f'Manifest: {len(manifest["train"])} train, {len(manifest["val"])} val pairs')

    train_ds = StafflineDataset(manifest['train'], size=cfg.size, augment=cfg.augment)
    val_ds   = StafflineDataset(manifest['val'],   size=cfg.size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=(device == 'cuda'))

    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        t0 = time.time()
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        train_dt   = time.time() - t0

        # ── Validate ──
        val_iou = float('nan')
        val_loss = float('nan')
        if epoch % cfg.val_every == 0:
            model.eval()
            iou_sum = 0.0
            loss_sum = 0.0
            n_v = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks  = masks.to(device, non_blocking=True)
                    logits = model(images)
                    loss_sum += criterion(logits, masks).item()
                    iou_sum  += iou_score(logits, masks)
                    n_v += 1
            val_iou  = iou_sum  / max(n_v, 1)
            val_loss = loss_sum / max(n_v, 1)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), str(out_path))
                marker = '  ★ saved'
            else:
                marker = ''
        else:
            marker = ''

        print(f'Epoch {epoch:3d}/{cfg.epochs}  '
              f'train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'val_IoU={val_iou:.4f}  '
              f'time={train_dt:.0f}s{marker}')

        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'val_iou':    val_iou,
            'time_sec':   train_dt,
        })

    history_path = out_path.with_suffix('.history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config':       vars(cfg),
            'best_val_iou': best_val_iou,
            'history':      history,
        }, f, indent=2)

    print(f'\nBest validation IoU: {best_val_iou:.4f}')
    print(f'Best model:    {out_path}')
    print(f'Training log:  {history_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', help='Path to manifest JSON from dataset_prep.py')
    parser.add_argument('--output',     default='models/staff_unet.pth')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=2)
    parser.add_argument('--size',       type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int,  default=2)
    parser.add_argument('--no-augment', action='store_true',
                          help='Disable data augmentation (faster but worse generalization)')
    args = parser.parse_args()

    cfg = TrainConfig(
        manifest    = args.manifest,
        output      = args.output,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        size        = args.size,
        lr          = args.lr,
        num_workers = args.num_workers,
        augment     = not args.no_augment,
    )
    train(cfg)


if __name__ == '__main__':
    main()
