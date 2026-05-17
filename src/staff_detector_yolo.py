"""
staff_detector_yolo.py — Staff detection via the OLA pretrained YOLOv8 model.

Uses the pretrained model from
    https://github.com/v-dvorak/omr-layout-analysis
which was trained on 7,013 images including MUSCIMA++ (handwritten music
notation), so it generalizes much better to handwritten input than our
classical morphology-based detector.

Pipeline integration
--------------------
This module returns staves in the same dict shape that staff_rectifier
already consumes:
    {
        'tracks':       list of 5 polylines (each [(x, y), ...])
        'top_curve':    polyline for the top staff line
        'bottom_curve': polyline for the bottom staff line
        'left_x':       leftmost x covered
        'right_x':      rightmost x covered
        'line_spacing': median spacing between adjacent lines
    }

The 5 line y-positions are recovered by running local peak-finding INSIDE
each YOLO bbox.  That's much easier than peak-finding on the whole page
because we already know where the staff is.

Setup (manual download)
-----------------------
1. Download the latest OLA model release:
     https://github.com/v-dvorak/omr-layout-analysis/releases
   Look for the .pt file — the v2.0 release weighs ~50 MB.

2. Place it at one of these locations (the loader checks them in order):
     <env var: OLA_MODEL_PATH>
     <project>/models/ola_v2.pt
     <project>/models/omr_layout_analysis.pt

3. The first call to detect_staves_yolo() will load the model and cache
   it in a module-level singleton.

Public API
----------
    detect_staves_yolo(img_gray_or_color, conf_threshold=0.4) -> List[dict]
    is_yolo_available() -> bool
    get_model_path() -> Optional[Path]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

# Class names from the OLA dataset.  We only care about 'staves' here, but
# keeping the full list helps when reading model.names mappings.
OLA_TARGET_CLASS = 'staves'

# Cached YOLO instance — first call loads, subsequent calls reuse.
_MODEL = None
_MODEL_PATH: Optional[Path] = None


def get_model_path() -> Optional[Path]:
    """
    Locate the OLA model file.  Returns None if no model is found.

    Search order:
        1. $OLA_MODEL_PATH (environment variable)
        2. <project>/models/ola_v2.pt
        3. <project>/models/omr_layout_analysis.pt
        4. <project>/models/ola.pt
    """
    env_path = os.environ.get('OLA_MODEL_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # This file lives at <project>/src/staff_detector_yolo.py, so the
    # project root is two levels up.
    project_root = Path(__file__).resolve().parent.parent
    models_dir = Path(os.environ.get('MODELS_DIR', project_root / 'models'))
    candidates = [
        models_dir / 'ola_v2.pt',
        models_dir / 'omr_layout_analysis.pt',
        models_dir / 'ola.pt',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def is_yolo_available() -> bool:
    """True iff the OLA model file exists and ultralytics is importable."""
    if get_model_path() is None:
        return False
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model():
    """Lazy-load the YOLO model on first use, cache it after."""
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return _MODEL

    model_path = get_model_path()
    if model_path is None:
        raise FileNotFoundError(
            'OLA model file not found.  Download from\n'
            '  https://github.com/v-dvorak/omr-layout-analysis/releases\n'
            'and place it at <project>/models/ola_v2.pt, or set the\n'
            'OLA_MODEL_PATH environment variable.'
        )

    from ultralytics import YOLO
    _MODEL = YOLO(str(model_path))
    _MODEL_PATH = model_path
    return _MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Per-bbox line extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_lines_from_bbox(img_gray: np.ndarray,
                              x1: int, y1: int, x2: int, y2: int,
                              n_strips: int = 12,
                              fallback_estimate: bool = True
                              ) -> Tuple[List[List[Tuple[int, int]]], float]:
    """
    Given a YOLO bbox containing a staff, find the 5 staff lines inside it.

    Strategy
    --------
    Slice the bbox into vertical strips, find horizontal peaks in each strip
    (same approach as the classical detector but locally — we already know
    the search region is exactly one staff).  Group peaks across strips into
    5 polyline tracks.

    Returns (tracks, line_spacing).  Each track is [(x, y), ...] in
    full-image coordinates.

    Falls back to evenly-spaced lines based on bbox height if peak-finding
    can't find 5 stable rows (rare, but possible on very faint input).
    """
    h_box = y2 - y1
    w_box = x2 - x1
    if h_box <= 0 or w_box <= 0:
        return [], 0.0

    # Binarize the local crop (Otsu is fine inside a known staff region —
    # contrast is already optimized for the staff).
    crop = img_gray[y1:y2, x1:x2]
    _, binary = cv2.threshold(crop, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Mild horizontal morph to clean up
    kernel_w = max(8, int(w_box * 0.05))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Slice into strips and find peak rows in each
    strip_w = max(1, w_box // n_strips)
    strip_data: List[Tuple[int, List[int]]] = []
    for i in range(n_strips):
        sx1 = i * strip_w
        sx2 = min(sx1 + strip_w, w_box)
        if sx2 <= sx1:
            continue
        strip = horiz[:, sx1:sx2]
        row_sum = strip.sum(axis=1)
        if row_sum.max() == 0:
            continue
        peaks, _ = find_peaks(row_sum,
                                height=row_sum.max() * 0.4,
                                distance=2)
        if len(peaks) == 0:
            continue
        strip_data.append((sx1 + (sx2 - sx1) // 2, sorted(peaks.tolist())))

    # If we found nothing, fall back to evenly-spaced lines
    if not strip_data:
        if not fallback_estimate:
            return [], 0.0
        return _fallback_evenly_spaced(x1, y1, x2, y2)

    # Estimate line spacing from the densest strip
    sample = sorted(strip_data, key=lambda s: -len(s[1]))[0][1]
    if len(sample) >= 2:
        gaps = np.diff(sample)
        small = gaps[gaps <= np.median(gaps) * 1.5]
        est_spacing = float(np.median(small)) if len(small) else float(np.median(gaps))
    else:
        est_spacing = h_box / 4.0

    # Walk strips left→right, extending tracks
    tracks_local: List[List[Tuple[int, int]]] = []
    y_match_tol = max(3.0, est_spacing * 0.4)

    for x_center, ys in strip_data:
        for y in ys:
            best_tr = None
            best_d  = y_match_tol
            for tr in tracks_local:
                last_x, last_y = tr[-1]
                if x_center - last_x > strip_w * 4:
                    continue
                d = abs(y - last_y)
                if d < best_d:
                    best_d  = d
                    best_tr = tr
            if best_tr is not None:
                best_tr.append((x_center, y))
            else:
                tracks_local.append([(x_center, y)])

    # Drop short tracks (noise) — keep tracks that span at least 30% of strips
    min_pts = max(3, int(n_strips * 0.30))
    tracks_local = [t for t in tracks_local if len(t) >= min_pts]
    tracks_local.sort(key=lambda t: np.mean([p[1] for p in t]))

    # If we still don't have 5 tracks, use evenly-spaced fallback
    if len(tracks_local) != 5:
        if fallback_estimate:
            return _fallback_evenly_spaced(x1, y1, x2, y2)
        return [], est_spacing

    # Convert local coords back to full-image coords
    tracks_global: List[List[Tuple[int, int]]] = []
    for tr in tracks_local:
        tracks_global.append([(p[0] + x1, p[1] + y1) for p in tr])

    return tracks_global, est_spacing


def _fallback_evenly_spaced(x1: int, y1: int, x2: int, y2: int
                             ) -> Tuple[List[List[Tuple[int, int]]], float]:
    """
    Last-resort: assume the bbox is tight around the staff and put 5 evenly
    spaced lines from top to bottom.  Each track is a 2-point polyline.
    """
    line_spacing = (y2 - y1) / 4.0
    tracks = []
    for k in range(5):
        y = int(round(y1 + k * line_spacing))
        tracks.append([(x1, y), (x2 - 1, y)])
    return tracks, line_spacing


# ─────────────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────────────

def detect_staves_yolo(img,
                        conf_threshold: float = 0.4,
                        imgsz: int = 1280,
                        verbose: bool = False) -> List[dict]:
    """
    Detect staves on a page using the OLA YOLOv8 model.

    Parameters
    ----------
    img             grayscale or BGR ndarray of the full page
    conf_threshold  YOLO confidence threshold (default 0.4)
    imgsz           inference size; 1280 works well for typical page sizes
    verbose         if True, print bbox details

    Returns
    -------
    list of staff dicts with the same shape produced by the classical
    detector — usable directly by staff_rectifier downstream code.
    """
    if img.ndim == 2:
        img_gray = img
        img_for_yolo = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_for_yolo = img

    model = _load_model()
    results = model(img_for_yolo,
                     conf=conf_threshold,
                     imgsz=imgsz,
                     verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    # Resolve target class id by name (the OLA model's class indices are
    # not guaranteed to be stable across releases, so name-based lookup
    # is safer than hardcoding).
    name_to_id = {v: k for k, v in model.names.items()}
    target_id  = name_to_id.get(OLA_TARGET_CLASS)
    if target_id is None:
        if verbose:
            print(f"WARN: model has no class named '{OLA_TARGET_CLASS}'. "
                  f"Available: {list(model.names.values())}")
        return []

    xyxy   = results.boxes.xyxy.cpu().numpy()
    confs  = results.boxes.conf.cpu().numpy()
    cls_id = results.boxes.cls.cpu().numpy().astype(int)

    # Filter to staff bboxes only
    mask = cls_id == target_id
    xyxy  = xyxy[mask]
    confs = confs[mask]

    if verbose:
        print(f'  YOLO found {len(xyxy)} staff bboxes')

    img_h, img_w = img_gray.shape
    staves: List[dict] = []
    for (x1f, y1f, x2f, y2f), conf in zip(xyxy, confs):
        x1 = max(0, int(x1f)); y1 = max(0, int(y1f))
        x2 = min(img_w, int(x2f)); y2 = min(img_h, int(y2f))

        # Reject suspiciously small detections (less than 4 px tall — even
        # the smallest realistic staff has several pixels per line gap).
        if y2 - y1 < 8 or x2 - x1 < 20:
            continue

        tracks, spacing = _extract_lines_from_bbox(img_gray, x1, y1, x2, y2)
        if len(tracks) != 5:
            # Couldn't extract 5 lines — skip this bbox rather than emit
            # an inconsistent staff dict.
            if verbose:
                print(f'    skipped bbox @ y={y1}-{y2}: '
                      f'extracted {len(tracks)} tracks')
            continue

        # All x positions across the 5 tracks
        all_x = [p[0] for tr in tracks for p in tr]

        staves.append({
            'tracks':       tracks,
            'top_curve':    tracks[0],
            'bottom_curve': tracks[4],
            'left_x':       min(all_x),
            'right_x':      max(all_x),
            'line_spacing': float(spacing),
            # Extra metadata not used by the rectifier but handy for debugging
            '_yolo_bbox':   (x1, y1, x2, y2),
            '_yolo_conf':   float(conf),
        })

    # Sort top-to-bottom (consistent with classical detector)
    staves.sort(key=lambda s: np.mean([p[1] for p in s['top_curve']]))
    return staves


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    test_img = sys.argv[1] if len(sys.argv) > 1 else \
        r'S:\mmdetection\data\my_images\img_1.png'

    print(f'Model path: {get_model_path()}')
    print(f'YOLO available: {is_yolo_available()}')

    if not is_yolo_available():
        print('\nInstall ultralytics and download the OLA model first:')
        print('  pip install ultralytics')
        print('  https://github.com/v-dvorak/omr-layout-analysis/releases')
        sys.exit(1)

    img = cv2.imread(test_img)
    if img is None:
        print(f'Cannot read {test_img}')
        sys.exit(1)

    print(f'\nDetecting on {test_img} …')
    staves = detect_staves_yolo(img, verbose=True)
    print(f'\nFound {len(staves)} staves:')
    for i, s in enumerate(staves):
        bx1, by1, bx2, by2 = s['_yolo_bbox']
        print(f'  Staff {i+1}: bbox=({bx1},{by1})-({bx2},{by2})  '
              f'conf={s["_yolo_conf"]:.2f}  '
              f'extent=({s["left_x"]}-{s["right_x"]})  '
              f'spacing={s["line_spacing"]:.1f}')

    # Save a quick visualization
    vis = img.copy()
    for i, s in enumerate(staves):
        bx1, by1, bx2, by2 = s['_yolo_bbox']
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        for tr in s['tracks']:
            pts = np.array(tr, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 1)
    cv2.imwrite('yolo_staff_test.png', vis)
    print('\nSaved yolo_staff_test.png')
