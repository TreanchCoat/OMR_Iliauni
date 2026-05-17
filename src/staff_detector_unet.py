"""
staff_detector_unet.py — Staff detection via the trained U-Net.

Runs a U-Net to produce a staff line probability mask, then post-processes
the mask into the same staff dict structure that staff_rectifier consumes:
    {
        'tracks':       list of 5 polylines (each [(x, y), ...])
        'top_curve':    polyline for the top staff line
        'bottom_curve': polyline for the bottom staff line
        'left_x':       leftmost x covered
        'right_x':      rightmost x covered
        'line_spacing': median spacing between adjacent lines
    }

Pipeline integration mirrors staff_detector_yolo.py — the rectifier can
try YOLO first, then U-Net, then classical, with each fallback chosen by
score.

Setup
-----
1. Train the model:
       python dataset_prep.py path/to/CVCMUSCIMA_SR
       python staff_unet_train.py muscima_manifest.json
2. The default output path is models/staff_unet.pth.  Override via
   $UNET_MODEL_PATH if you save to a custom location.

Public API
----------
    detect_staves_unet(img, threshold=0.5) -> List[dict]
    is_unet_available() -> bool
    get_model_path() -> Optional[Path]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL = None
_MODEL_PATH: Optional[Path] = None
_DEVICE = None


def get_model_path() -> Optional[Path]:
    """Search standard locations for the trained U-Net checkpoint."""
    env_path = os.environ.get('UNET_MODEL_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    # This file lives at <project>/src/staff_detector_unet.py, so the
    # project root is two levels up.
    project_root = Path(__file__).resolve().parent.parent
    models_dir = Path(os.environ.get('MODELS_DIR', project_root / 'models'))
    candidates = [
        models_dir / 'staff_unet.pth',
        models_dir / 'staff_unet_best.pth',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def is_unet_available() -> bool:
    """True iff the U-Net checkpoint exists and torch is importable."""
    if get_model_path() is None:
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model():
    """Lazy-load and cache the U-Net."""
    global _MODEL, _MODEL_PATH, _DEVICE

    if _MODEL is not None:
        return _MODEL

    model_path = get_model_path()
    if model_path is None:
        raise FileNotFoundError(
            'Trained U-Net not found.  Train one with staff_unet_train.py\n'
            'or set UNET_MODEL_PATH to point at the .pth file.'
        )

    import torch
    from staff_unet import UNet

    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _MODEL = UNet().to(_DEVICE)
    state = torch.load(str(model_path), map_location=_DEVICE, weights_only=True)
    _MODEL.load_state_dict(state)
    _MODEL.eval()
    _MODEL_PATH = model_path
    return _MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Inference: produce a probability mask
# ─────────────────────────────────────────────────────────────────────────────

def predict_mask(img,
                  inference_size: int = 512,
                  threshold: float = 0.5,
                  tile_overlap: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the U-Net at full resolution by tiling.

    The model was trained at 512×512.  For a 2550×4200 page, simply resizing
    to 512×512 would crush staff lines (originally 1-2 px tall) into
    sub-pixel artifacts that the model can't see.  Instead we slide a 512×512
    window across the page at native resolution and stitch the per-tile
    predictions back together.

    Tiles overlap by `tile_overlap` pixels and predictions are averaged in
    the overlap regions to avoid seam artifacts.

    Returns (probability_mask, binary_mask), both at FULL IMAGE resolution.
    """
    import torch

    model = _load_model()

    if img.ndim == 2:
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h_full, w_full = img_gray.shape

    # Binarize once on the full image — matches CVC-MUSCIMA training data
    _, binarised = cv2.threshold(img_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If the page is smaller than one tile, we don't need tiling at all
    if h_full <= inference_size and w_full <= inference_size:
        # Single-tile path: pad to inference_size, predict, crop
        padded = np.full((inference_size, inference_size), 255, dtype=np.uint8)
        padded[:h_full, :w_full] = binarised
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        prob_full = probs[:h_full, :w_full]
        binary_full = (prob_full > threshold).astype(np.uint8) * 255
        return prob_full, binary_full

    # Tiled inference
    stride = inference_size - tile_overlap
    prob_accum = np.zeros((h_full, w_full), dtype=np.float32)
    weight_accum = np.zeros((h_full, w_full), dtype=np.float32)

    # Generate tile coordinates that cover the full image including edges
    ys = list(range(0, max(1, h_full - inference_size + 1), stride))
    if ys[-1] + inference_size < h_full:
        ys.append(h_full - inference_size)
    xs = list(range(0, max(1, w_full - inference_size + 1), stride))
    if xs[-1] + inference_size < w_full:
        xs.append(w_full - inference_size)

    n_tiles = len(ys) * len(xs)
    print(f'  Tiled inference: {n_tiles} tiles ({len(ys)}×{len(xs)})')

    for ty in ys:
        for tx in xs:
            # Extract tile (always exactly inference_size × inference_size)
            tile_h = min(inference_size, h_full - ty)
            tile_w = min(inference_size, w_full - tx)
            tile = np.full((inference_size, inference_size), 255, dtype=np.uint8)
            tile[:tile_h, :tile_w] = binarised[ty:ty+tile_h, tx:tx+tile_w]

            # Predict
            rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            x = rgb.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            x = torch.from_numpy(x).unsqueeze(0).to(_DEVICE)
            with torch.no_grad():
                tile_probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()

            # Accumulate the actual region (ignore padding region in tiles
            # smaller than 512×512 at the edges)
            prob_accum[ty:ty+tile_h, tx:tx+tile_w] += tile_probs[:tile_h, :tile_w]
            weight_accum[ty:ty+tile_h, tx:tx+tile_w] += 1.0

    prob_full = prob_accum / np.maximum(weight_accum, 1e-6)
    binary_full = (prob_full > threshold).astype(np.uint8) * 255

    return prob_full, binary_full


# ─────────────────────────────────────────────────────────────────────────────
# Mask → staff dicts
# ─────────────────────────────────────────────────────────────────────────────

def _extract_staff_lines_from_mask(binary_mask: np.ndarray,
                                     min_line_len_frac: float = 0.15,
                                     max_line_thickness: int = 8) -> list:
    """
    Convert a binary staff-line mask into a list of (left_x, right_x,
    y_polyline) tuples — one per detected line.

    Strategy: each connected component in the mask is one staff line.
    For each component, sample the y-coordinate at evenly-spaced x positions
    by taking the centroid of foreground pixels in each x-column.
    """
    h, w = binary_mask.shape

    # Connected components — one per staff line
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8)

    min_line_len = int(w * min_line_len_frac)
    lines = []

    for lbl in range(1, n_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]

        # Filter: needs to be long and thin
        if cw < min_line_len:
            continue
        if ch > max_line_thickness:
            # Multiple staff lines might be merged into one component if
            # the U-Net output is fat.  Try splitting by horizontal-stripe
            # row sums (handled later by line-count check on the staff).
            pass
        if area < min_line_len:
            continue

        # Sample y at ~24 evenly-spaced x positions across the component
        n_samples = max(8, min(48, cw // 30))
        xs = np.linspace(x + 1, x + cw - 2, n_samples).astype(int)
        track = []
        for xi in xs:
            col = labels[:, xi] == lbl
            ys = np.where(col)[0]
            if len(ys) > 0:
                track.append((int(xi), int(np.mean(ys))))
        if len(track) >= 4:
            lines.append({
                'left_x':  x,
                'right_x': x + cw,
                'track':   track,
                'mean_y':  float(np.mean([p[1] for p in track])),
            })

    lines.sort(key=lambda l: l['mean_y'])

    # Merge fragments at the same y — connected-components splits a single
    # staff line at every gap, so we re-merge anything within ~5 pixels of
    # the same y as one line.
    merged = []
    y_merge_tol = 5
    for line in lines:
        if merged and abs(line['mean_y'] - merged[-1]['mean_y']) <= y_merge_tol:
            # Merge into previous: concatenate tracks, expand extent
            prev = merged[-1]
            prev['track'].extend(line['track'])
            prev['track'].sort(key=lambda p: p[0])
            prev['left_x']  = min(prev['left_x'],  line['left_x'])
            prev['right_x'] = max(prev['right_x'], line['right_x'])
            ys = [p[1] for p in prev['track']]
            prev['mean_y'] = float(np.mean(ys))
        else:
            merged.append(line)

    return merged


def _group_lines_into_staves(lines: list,
                               spacing_tol: float = 0.4) -> list:
    """
    Group consecutive detected lines into 5-line staves.

    Algorithm: estimate typical inter-line spacing from all consecutive
    gaps, then walk through the sorted lines forming groups where the gap
    between successive lines is consistent with that spacing.  A group of
    5 lines with consistent spacing is a staff.
    """
    if len(lines) < 5:
        return []

    ys = np.array([l['mean_y'] for l in lines])
    gaps = np.diff(ys)
    if len(gaps) == 0:
        return []

    # Estimate staff line spacing from the small gaps (the median of
    # gaps that are ≤ 1.5× the overall median).
    overall_median = float(np.median(gaps))
    small = gaps[gaps <= overall_median * 1.5]
    target_spacing = float(np.median(small)) if len(small) else overall_median
    if target_spacing <= 0:
        return []
    max_inside_gap = target_spacing * (1 + spacing_tol)

    # Walk through lines, building groups
    staves = []
    used = set()
    i = 0
    while i + 4 < len(lines):
        if i in used:
            i += 1
            continue
        group_y = ys[i:i + 5]
        group_gaps = np.diff(group_y)
        if (group_gaps.max() <= max_inside_gap and
                group_gaps.min() >= target_spacing * (1 - spacing_tol)):
            staves.append(lines[i:i + 5])
            for k in range(5):
                used.add(i + k)
            i += 5
        else:
            i += 1
    return staves


def _staves_to_dicts(staves: list) -> list:
    """Convert grouped lines into the staff-rectifier-compatible dict shape."""
    out = []
    for group in staves:
        tracks = [g['track'] for g in group]
        all_x = [p[0] for tr in tracks for p in tr]
        ys = [g['mean_y'] for g in group]
        spacing = float(np.median(np.diff(sorted(ys))))
        out.append({
            'tracks':       tracks,
            'top_curve':    tracks[0],
            'bottom_curve': tracks[4],
            'left_x':       min(all_x),
            'right_x':      max(all_x),
            'line_spacing': spacing,
            '_unet_source': True,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect_staves_unet(img,
                        threshold: float = 0.2,
                        inference_size: int = 512,
                        verbose: bool = False) -> List[dict]:
    """
    Detect staves on a page using the trained U-Net.

    Parameters
    ----------
    img             grayscale or BGR ndarray of the full page
    threshold       probability threshold for binarising the mask
    inference_size  side length the model expects (must match training)
    verbose         if True, print intermediate counts
    """
    prob, binary = predict_mask(img, inference_size=inference_size,
                                  threshold=threshold)
    if verbose:
        print(f'  U-Net mask: {(binary > 0).sum()} foreground px '
              f'({100 * (binary > 0).mean():.2f}% of image)')

    # Light morphology to clean speckle noise + bridge gaps within lines.
    # The kernel needs to be wide enough to close gaps in a single staff line
    # (which can be hundreds of px on handwritten pages).  ~10% of image width
    # is generous but rarely merges adjacent staff lines vertically since the
    # kernel is only 1 px tall.
    if (binary > 0).any():
        kx = max(50, binary.shape[1] // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = _extract_staff_lines_from_mask(binary)
    if verbose:
        print(f'  Extracted {len(lines)} candidate lines from mask')

    grouped = _group_lines_into_staves(lines)
    if verbose:
        print(f'  Grouped into {len(grouped)} staves')

    return _staves_to_dicts(grouped)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print(f'Model path: {get_model_path()}')
    print(f'U-Net available: {is_unet_available()}')

    if not is_unet_available():
        print('\nTrain the U-Net first or set UNET_MODEL_PATH.')
        sys.exit(1)

    test_img = sys.argv[1] if len(sys.argv) > 1 else 'score.png'
    img = cv2.imread(test_img)
    if img is None:
        print(f'Cannot read {test_img}')
        sys.exit(1)

    print(f'\nDetecting staves on {test_img} …')
    staves = detect_staves_unet(img, verbose=True)
    print(f'\nFound {len(staves)} staves')
    for i, s in enumerate(staves):
        print(f'  Staff {i+1}: extent=({s["left_x"]}-{s["right_x"]}) '
              f'spacing={s["line_spacing"]:.1f}')

    # Save visualization
    vis = img.copy()
    prob, binary = predict_mask(img)
    cv2.imwrite('unet_mask.png', binary)
    cv2.imwrite('unet_prob.png', (prob * 255).astype(np.uint8))
    for i, s in enumerate(staves):
        for tr in s['tracks']:
            pts = np.array(tr, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 1)
        # Bounding box from extent
        ys_top = [p[1] for p in s['top_curve']]
        ys_bot = [p[1] for p in s['bottom_curve']]
        cv2.rectangle(vis,
                       (s['left_x'], min(ys_top) - 5),
                       (s['right_x'], max(ys_bot) + 5),
                       (0, 255, 0), 2)
    cv2.imwrite('unet_staff_test.png', vis)
    print('\nSaved unet_mask.png, unet_prob.png, unet_staff_test.png')
