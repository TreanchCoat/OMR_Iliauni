"""
dataset_prep.py — Find and pair CVC-MUSCIMA Staff Removal images & masks.

Run once after unzipping the CVC-MUSCIMA Staff Removal set.
Produces a JSON manifest that the training script consumes.

Expected dataset layout (after unzipping CVCMUSCIMA_SR.zip)
-----------------------------------------------------------
    CVCMUSCIMA_SR/
      CvcMuscima-Distortions/
        ideal/
          w-01/                                 ← per-writer folder
            image/
              p001.png   p002.png   …           ← full music page (input)
            gt/
              p001.png   p002.png   …           ← staff-only (this is our mask)
            symbol/
              p001.png   p002.png   …           ← staff-less version (not used)
          w-02/  …  w-50/
        staffline-interruption/                 ← distortion variant
        kanungo/                                ← another distortion variant
        … (9 distortion folders + 'ideal')

Each writer has 20 pages.  With 50 writers and 11 variants (ideal + 10
distortions) we get 50 × 20 × 11 = 11,000 image/mask pairs.

Usage
-----
    python dataset_prep.py path/to/CVCMUSCIMA_SR
    python dataset_prep.py path/to/CVCMUSCIMA_SR --variants ideal kanungo
    python dataset_prep.py path/to/CVCMUSCIMA_SR --output muscima_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def find_pairs(root: Path,
               variants: Optional[List[str]] = None) -> list:
    """
    Walk the dataset and return a list of {image_path, mask_path, writer,
    page, variant} dicts.

    Parameters
    ----------
    root      Path to the unzipped CVCMUSCIMA_SR directory.
    variants  If given, restrict to these distortion folder names.  Default
              is all variants found on disk (ideal + 10 distortions).
    """
    distortions_dir = root / 'CvcMuscima-Distortions'
    if not distortions_dir.exists():
        # Some distributions skip the wrapper directory — try root directly
        distortions_dir = root
    if not distortions_dir.exists():
        raise FileNotFoundError(f'Cannot find CvcMuscima-Distortions under {root}')

    available_variants = sorted(p.name for p in distortions_dir.iterdir()
                                  if p.is_dir())
    if variants is None:
        variants = available_variants
    else:
        missing = set(variants) - set(available_variants)
        if missing:
            raise ValueError(f'Variants not found: {missing}.  '
                              f'Available: {available_variants}')

    pairs = []
    for variant in variants:
        variant_dir = distortions_dir / variant
        for writer_dir in sorted(variant_dir.iterdir()):
            if not writer_dir.is_dir() or not writer_dir.name.startswith('w-'):
                continue
            image_dir = writer_dir / 'image'
            gt_dir    = writer_dir / 'gt'
            if not image_dir.exists() or not gt_dir.exists():
                continue
            for img_path in sorted(image_dir.glob('*.png')):
                mask_path = gt_dir / img_path.name
                if not mask_path.exists():
                    continue
                pairs.append({
                    'image_path': str(img_path.resolve()),
                    'mask_path':  str(mask_path.resolve()),
                    'writer':     writer_dir.name,
                    'page':       img_path.stem,
                    'variant':    variant,
                })
    return pairs


def split_train_val(pairs: list, val_writers: int = 5,
                    seed: int = 42) -> tuple:
    """
    Hold out N writers (5 by default) as the validation set.  Splitting by
    writer rather than by image prevents leakage — the model never sees the
    same handwriting style during training that it's evaluated on.
    """
    import random
    rng = random.Random(seed)
    all_writers = sorted({p['writer'] for p in pairs})
    rng.shuffle(all_writers)
    val_set = set(all_writers[:val_writers])
    train = [p for p in pairs if p['writer'] not in val_set]
    val   = [p for p in pairs if p['writer']     in val_set]
    return train, val, sorted(val_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root',
                          help='Path to unzipped CVCMUSCIMA_SR directory')
    parser.add_argument('--variants', nargs='+',
                          help='Distortion variants to include '
                               '(default: all found)')
    parser.add_argument('--output', default='muscima_manifest.json',
                          help='Where to write the manifest JSON')
    parser.add_argument('--val-writers', type=int, default=5,
                          help='Number of writers to hold out for validation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f'ERROR: dataset root not found: {root}')
        sys.exit(1)

    pairs = find_pairs(root, args.variants)
    if not pairs:
        print('ERROR: no image/mask pairs found.  Check dataset layout.')
        sys.exit(1)

    train, val, val_writers = split_train_val(pairs, args.val_writers, args.seed)

    manifest = {
        'dataset_root': str(root.resolve()),
        'variants_used': sorted({p['variant'] for p in pairs}),
        'val_writers':   val_writers,
        'train':         train,
        'val':           val,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Found {len(pairs)} pairs total')
    print(f'  Train: {len(train)} pairs  ({len(set(p["writer"] for p in train))} writers)')
    print(f'  Val:   {len(val)} pairs    ({len(val_writers)} writers: {val_writers})')
    print(f'  Variants: {manifest["variants_used"]}')
    print(f'\nManifest written to: {args.output}')


if __name__ == '__main__':
    main()
