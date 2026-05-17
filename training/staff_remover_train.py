"""
staff_remover_train.py — Train a U-Net to remove staff lines from a
binarised music page.

Idea
----
Same architecture as ``staff_unet`` (the staff-line *detection* model
that ``staff_detector_unet`` uses), but trained for a different task:

    Input   ─►  full page (binarised)
    Target  ─►  symbol-only page (staff lines erased, notes intact)

CVC-MUSCIMA conveniently ships both per page:

    image/pNN.png    ← page with staves
    gt/pNN.png       ← staff-lines-only mask          (used for detection)
    symbol/pNN.png   ← staff-lines-removed page       (used HERE)

At inference, the trained model can replace the classical
``staff_remover.remove_staff_lines`` whenever the classical version
damages thin notes — see ``predict_clean`` at the bottom of this file
for the inference helper.

Manifest
--------
The script reuses the manifest produced by ``dataset_prep.py``.  It
derives the ``symbol_path`` from each entry's ``image_path`` by
swapping the ``image/`` directory for ``symbol/``.  No new manifest
needed.

Usage
-----
::

    python staff_remover_train.py muscima_manifest.json \\
        --epochs 25 --batch-size 4 --size 512 \\
        --output models/staff_remover.pth

    # Quick CPU smoke test (don't train this for real on CPU):
    python staff_remover_train.py muscima_manifest.json \\
        --epochs 1 --batch-size 1 --size 256 --num-workers 0

Recommended hardware (rough)
-----------------------------
RTX 3050 (4 GB):  batch=2 at 512×512   → ~4-8 hours for 25-30 epochs
RTX 4090 (24 GB): batch=8 at 512×512   → ~1-2 hours for 25-30 epochs
RTX 5090 (32 GB): batch=12 at 512×512  → ~45-90 minutes for 25-30 epochs

If you only have a 3050, start with the ``ideal`` variant subset (50
writers × 20 pages = 1 000 pairs) instead of all 11 variants — it
converges in ~1 hour and is enough to verify the approach.

The earlier "two days of training" estimate I tossed out was assuming a
3050 with the full 11-variant dataset (11 000 pairs) and a fair amount
of slack for tuning runs.  On a 5090 even the full run is well under
half a day.
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

# Reuse the U-Net architecture + loss already in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from staff_unet import UNet, DiceBCELoss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _symbol_path_for(image_path: str) -> str:
    """
    Derive the path of the staff-removed page from an ``image/`` path
    by replacing the directory name.  Both files have the same stem.
    """
    p = Path(image_path)
    parts = list(p.parts)
    # Walk up looking for a parent component called 'image' and swap
    # it for 'symbol'.  This is the convention used by CVC-MUSCIMA's
    # writer-level layout (`w-NN/image/pXX.png`).
    for i, comp in enumerate(parts):
        if comp == 'image':
            parts[i] = 'symbol'
            return str(Path(*parts))
    raise ValueError(f'Cannot derive symbol path from {image_path!r} '
                     f'— no "image" component in the path.')


class RemovalDataset(Dataset):
    """
    Loads (image, symbol-only) pairs from a manifest entry list.

    Both files use CVC-MUSCIMA's convention (BLACK background, WHITE ink).
    We invert the input so the model trains on real-scan convention
    (WHITE paper, BLACK ink) and we treat the target as a binary
    "this pixel is a SYMBOL" mask (1.0 where the ``symbol/`` file has
    ink, 0.0 elsewhere).
    """

    def __init__(self, entries: list, size: int = 512,
                 augment: bool = True):
        self.entries = entries
        self.size = size
        self.augment = augment

        if augment:
            try:
                import albumentations as A
            except ImportError:
                raise ImportError(
                    'Install albumentations:  pip install albumentations')
            # Mild augmentation only.  Anything that warps the
            # geometry too much (large rotations, big perspective)
            # would break the pixel-aligned input/target relationship.
            self.aug = A.Compose([
                A.Rotate(limit=4, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path  = entry['image_path']
        symbol_path = entry.get('symbol_path') or _symbol_path_for(image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        symbol = cv2.imread(symbol_path, cv2.IMREAD_GRAYSCALE)
        if symbol is None:
            raise RuntimeError(f'Cannot read symbol: {symbol_path}')

        # CVC-MUSCIMA convention -> real-scan convention.
        image = 255 - image

        # Resize.  Symbol mask uses INTER_NEAREST so we don't smear
        # the binary boundary between ink and not-ink.
        image  = cv2.resize(image,  (self.size, self.size),
                            interpolation=cv2.INTER_AREA)
        symbol = cv2.resize(symbol, (self.size, self.size),
                            interpolation=cv2.INTER_NEAREST)

        # WHITE (>=128) in `symbol/*.png` = symbol-ink pixel.  We want
        # the target to be 1.0 there.
        mask = (symbol >= 128).astype(np.float32)

        if self.aug is not None:
            out = self.aug(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(logits: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                   threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).float().sum().item()
    total   = float(targets.numel())
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    manifest:    str
    output:      str   = 'models/staff_remover.pth'
    epochs:      int   = 25
    batch_size:  int   = 2
    size:        int   = 512
    lr:          float = 1e-4
    num_workers: int   = 2
    val_every:   int   = 1
    augment:     bool  = True
    resume:      str   = ''
    variants:    str   = ''   # comma-separated names; empty = all


def train(cfg: TrainConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    with open(cfg.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    if cfg.variants:
        wanted = {v.strip() for v in cfg.variants.split(',') if v.strip()}
        manifest['train'] = [p for p in manifest['train']
                             if p.get('variant') in wanted]
        manifest['val']   = [p for p in manifest['val']
                             if p.get('variant') in wanted]
        print(f'Restricted to variants {sorted(wanted)}: '
              f'{len(manifest["train"])} train, {len(manifest["val"])} val')

    print(f'Manifest: {len(manifest["train"])} train, '
          f'{len(manifest["val"])} val pairs')

    train_ds = RemovalDataset(manifest['train'], size=cfg.size,
                               augment=cfg.augment)
    val_ds   = RemovalDataset(manifest['val'],   size=cfg.size,
                               augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device == 'cuda'))

    model = UNet().to(device)
    if cfg.resume:
        print(f'Resuming from {cfg.resume}')
        state = torch.load(cfg.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        loss_sum = 0.0
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

            loss_sum += loss.item()
            n_batches += 1

        train_loss = loss_sum / max(n_batches, 1)
        train_dt   = time.time() - t0

        # ── Validate ──
        val_iou = float('nan')
        val_loss = float('nan')
        val_acc  = float('nan')
        if epoch % cfg.val_every == 0:
            model.eval()
            iou_sum = loss_sum = acc_sum = 0.0
            n_v = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks  = masks.to(device, non_blocking=True)
                    logits = model(images)
                    loss_sum += criterion(logits, masks).item()
                    iou_sum  += iou_score(logits, masks)
                    acc_sum  += pixel_accuracy(logits, masks)
                    n_v += 1
            val_iou  = iou_sum  / max(n_v, 1)
            val_loss = loss_sum / max(n_v, 1)
            val_acc  = acc_sum  / max(n_v, 1)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), str(out_path))
                marker = '  * saved'
            else:
                marker = ''
        else:
            marker = ''

        print(f'Epoch {epoch:3d}/{cfg.epochs}  '
              f'train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'val_IoU={val_iou:.4f}  '
              f'val_acc={val_acc:.4f}  '
              f'time={train_dt:.0f}s{marker}')

        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'val_iou':    val_iou,
            'val_acc':    val_acc,
            'time_sec':   train_dt,
        })

    history_path = out_path.with_suffix('.history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config':       vars(cfg),
            'best_val_iou': best_val_iou,
            'history':      history,
        }, f, indent=2)

    print()
    print(f'Best validation IoU: {best_val_iou:.4f}')
    print(f'Best model:    {out_path}')
    print(f'Training log:  {history_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_clean(model_path: str, image: np.ndarray,
                  size: int = 512, threshold: float = 0.5
                  ) -> np.ndarray:
    """
    Run the trained removal U-Net on a single page and return a
    cleaned BINARY image (255 where the model says "symbol pixel",
    0 elsewhere).

    Use this as a drop-in replacement for
    ``staff_remover.clean_staff_image`` when the classical removal is
    damaging thin notes.
    """
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    img_rgb = 255 - img_rgb            # match training convention
    resized = cv2.resize(img_rgb, (size, size),
                         interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = UNet().to(device).eval()
    state  = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()

    probs   = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    cleaned = ((probs > threshold) * 255).astype(np.uint8)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest',
                        help='Path to manifest JSON from dataset_prep.py')
    parser.add_argument('--output',     default='models/staff_remover.pth')
    parser.add_argument('--epochs',     type=int,   default=25)
    parser.add_argument('--batch-size', type=int,   default=2)
    parser.add_argument('--size',       type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int,  default=2)
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable mild augmentation (faster, slightly '
                             'worse generalisation).')
    parser.add_argument('--resume',     default='',
                        help='Path to a previous .pth to resume training from.')
    parser.add_argument('--variants',   default='',
                        help='Comma-separated CVC-MUSCIMA variant names to '
                             'restrict the dataset to (default: all).  '
                             'Examples: ideal | ideal,kanungo | staffline-interruption')
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
        resume      = args.resume,
        variants    = args.variants,
    )
    train(cfg)


if __name__ == '__main__':
    main()
