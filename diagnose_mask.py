"""
diagnose_mask.py — Quickly check what's actually in a CVC-MUSCIMA gt mask.
"""

import sys
import json
import cv2
import numpy as np

if len(sys.argv) < 2:
    print('Usage: python diagnose_mask.py muscima_manifest.json')
    sys.exit(1)

with open(sys.argv[1]) as f:
    manifest = json.load(f)

# Look at the first 3 train pairs
for i, entry in enumerate(manifest['train'][:3]):
    print(f'\n── Pair {i+1} ──')
    print(f'  Image: {entry["image_path"]}')
    print(f'  Mask:  {entry["mask_path"]}')

    img  = cv2.imread(entry['image_path'], cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)

    if img is None:
        print('  Failed to read image')
        continue
    if mask is None:
        print('  Failed to read mask')
        continue

    print(f'  Image shape:  {img.shape}, dtype={img.dtype}')
    print(f'  Image: min={img.min()}, max={img.max()}, mean={img.mean():.1f}')
    print(f'  Image: % black (<128)            = {100 * (img < 128).mean():.2f}%')
    print(f'  Image: % white (>=128)           = {100 * (img >= 128).mean():.2f}%')

    print(f'  Mask shape:   {mask.shape}, dtype={mask.dtype}')
    print(f'  Mask:  min={mask.min()}, max={mask.max()}, mean={mask.mean():.1f}')
    print(f'  Mask:  % black (<128)            = {100 * (mask < 128).mean():.2f}%')
    print(f'  Mask:  % white (>=128)           = {100 * (mask >= 128).mean():.2f}%')
    print(f'  Mask unique values:  {np.unique(mask)[:10]}')

    # Save side-by-side for visual inspection
    if i == 0:
        h, w = img.shape
        # Resize to same width if needed
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        side = np.hstack([img, np.full((h, 20), 128, dtype=np.uint8), mask])
        cv2.imwrite('diagnose_pair_1.png', side)
        print(f'  → saved diagnose_pair_1.png  (image | mask)')