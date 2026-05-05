"""
bbox_refiner.py — Tighten YOLO bounding boxes using image content.

Why
---
YOLO bounding boxes are usually a few pixels larger than the actual symbol —
sometimes a lot more on small classes like noteheads.  Loose boxes hurt
downstream pitch calculation: the box center can drift up or down by a couple
pixels, which is enough to misread G4 as A4 on a tight staff.

How it works
------------
The pipeline already produces `pstaff.cleaned` — a binary image with the
staff lines removed.  For each detection:

  1. Take the YOLO box, expand by `padding_px` to give us a search region.
  2. Find connected components of foreground pixels inside that region.
  3. Pick the component closest to the original YOLO box center
     (filtered by a minimum-area threshold to ignore noise).
  4. Use that component's tight bbox as the refined box.

For solid noteheads the component IS the notehead.  For hollow noteheads
(half / whole) the staff-line removal step leaves the head's outline as a
single connected ring, so the same logic works — its bounding box is the
head's outer extent.

For symbols the model defaults are wrong on (slurs span hundreds of pixels;
beams are wide horizontal bars), per-class config below disables refinement
or relaxes the area threshold.

Public API
----------
    refine_detection(detection, cleaned, class_name=None) -> Detection
    refine_page(page_detections, processed_score) -> PageDetections   (in-place)
    visualize_refinement(processed_score, before, after, output_dir)

Usage
-----
    from bbox_refiner import refine_page
    refine_page(detections, processed)         # mutates detections in place
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-class refinement config
# ─────────────────────────────────────────────────────────────────────────────
#
# `enabled`      Whether to refine this class at all.
# `padding_px`   Pixels added on every side to YOLO box for the search region.
# `min_area`     Minimum component area (px²) to consider a valid match.
# `recenter`     If True, refined box is centered on the component centroid;
#                if False, the component's bounding box is used directly.
#                For noteheads we use centroid-based recentering with the
#                ORIGINAL box width/height kept — pitch only depends on cy,
#                and centroid is more robust than tight-bbox center for
#                hollow heads on a noisy background.

REFINE_CONFIG: Dict[str, dict] = {
    # ── Notehead family — the high-priority case ──
    'noteheadBlack':       {'enabled': True,  'padding_px': 4,  'min_area': 12, 'recenter': 'centroid_keep_size'},
    'noteheadHalf':        {'enabled': True,  'padding_px': 4,  'min_area': 12, 'recenter': 'centroid_keep_size'},
    'noteheadWhole':       {'enabled': True,  'padding_px': 4,  'min_area': 12, 'recenter': 'centroid_keep_size'},
    'noteheadDoubleWhole': {'enabled': True,  'padding_px': 4,  'min_area': 12, 'recenter': 'centroid_keep_size'},

    # ── Clefs ──
    'clefG':               {'enabled': True,  'padding_px': 6,  'min_area': 80, 'recenter': 'tight_bbox'},
    'clefF':               {'enabled': True,  'padding_px': 6,  'min_area': 80, 'recenter': 'tight_bbox'},
    'clefCAlto':           {'enabled': True,  'padding_px': 6,  'min_area': 80, 'recenter': 'tight_bbox'},
    'clefCTenor':          {'enabled': True,  'padding_px': 6,  'min_area': 80, 'recenter': 'tight_bbox'},

    # ── Accidentals ──
    'keyFlat':             {'enabled': True,  'padding_px': 3,  'min_area': 8,  'recenter': 'tight_bbox'},
    'keySharp':            {'enabled': True,  'padding_px': 3,  'min_area': 8,  'recenter': 'tight_bbox'},
    'keyNatural':          {'enabled': True,  'padding_px': 3,  'min_area': 8,  'recenter': 'tight_bbox'},
    'accidentalSharp':     {'enabled': True,  'padding_px': 3,  'min_area': 8,  'recenter': 'tight_bbox'},
    'accidentalFlat':      {'enabled': True,  'padding_px': 3,  'min_area': 8,  'recenter': 'tight_bbox'},
    'accidentalNatural':   {'enabled': True,  'padding_px': 3,  'min_area': 8,  'recenter': 'tight_bbox'},

    # ── Rests — keep enabled but more padding (rests vary in shape) ──
    'restWhole':           {'enabled': True,  'padding_px': 4,  'min_area': 10, 'recenter': 'tight_bbox'},
    'restHalf':            {'enabled': True,  'padding_px': 4,  'min_area': 10, 'recenter': 'tight_bbox'},
    'restQuarter':         {'enabled': True,  'padding_px': 4,  'min_area': 10, 'recenter': 'tight_bbox'},
    'restEighth':          {'enabled': True,  'padding_px': 4,  'min_area': 10, 'recenter': 'tight_bbox'},
    'rest8th':             {'enabled': True,  'padding_px': 4,  'min_area': 10, 'recenter': 'tight_bbox'},
    'rest16th':            {'enabled': True,  'padding_px': 4,  'min_area': 10, 'recenter': 'tight_bbox'},

    # ── Things we DO NOT refine ──
    # Slurs and beams have huge bboxes that span multiple symbols.
    # The connected component logic would shrink them to a single point.
    'slur':                {'enabled': False},
    'tie':                 {'enabled': False},
    'beam':                {'enabled': False},
    # Staff is a structural detection, not a symbol.
    'staff':               {'enabled': False},
    # clef8 is tiny and unreliable.
    'clef8':               {'enabled': False},
}

# Default config for unknown classes — refine, but conservatively.
_DEFAULT_CONFIG = {'enabled': True, 'padding_px': 4, 'min_area': 8,
                   'recenter': 'tight_bbox'}


# ─────────────────────────────────────────────────────────────────────────────
# Single-detection refinement
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_component(cleaned: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int,
                         orig_cx: float, orig_cy: float,
                         min_area: int) -> Optional[Tuple]:
    """
    Find the connected foreground component closest to (orig_cx, orig_cy)
    inside the rectangle [x1..x2, y1..y2].

    Returns (cx, cy, bx1, by1, bx2, by2) in cleaned-image coords, or None.
    """
    h, w = cleaned.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    region = cleaned[y1:y2, x1:x2]

    # Foreground = black pixels (== 0).  Use 8-connectivity.
    fg = (region == 0).astype(np.uint8) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=8)

    if n_labels <= 1:
        return None

    best       = None
    best_dist2 = float('inf')
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        bx = stats[label_id, cv2.CC_STAT_LEFT]
        by = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        cy_local, cx_local = centroids[label_id][1], centroids[label_id][0]

        # Convert to crop-image coords
        comp_cx = cx_local + x1
        comp_cy = cy_local + y1

        d2 = (comp_cx - orig_cx) ** 2 + (comp_cy - orig_cy) ** 2
        if d2 < best_dist2:
            best_dist2 = d2
            best = (comp_cx, comp_cy,
                    bx + x1, by + y1,
                    bx + x1 + bw, by + y1 + bh)

    return best


def refine_detection(detection,
                     cleaned: np.ndarray,
                     crop_y1: int,
                     class_name: Optional[str] = None,
                     config_override: Optional[dict] = None) -> bool:
    """
    Refine a single Detection in place.

    Returns True if the detection was modified, False otherwise.
    `class_name` defaults to detection.class_name; pass it to override.
    `config_override` lets callers pass a custom config dict for this call.
    """
    cls = class_name or detection.class_name
    cfg = config_override or REFINE_CONFIG.get(cls, _DEFAULT_CONFIG)

    if not cfg.get('enabled', False):
        return False

    pad      = cfg.get('padding_px', 4)
    min_area = cfg.get('min_area', 8)
    mode     = cfg.get('recenter', 'tight_bbox')

    sx1 = detection.x1 - pad
    sy1 = detection.y1 - pad
    sx2 = detection.x2 + pad
    sy2 = detection.y2 + pad

    found = _find_best_component(
        cleaned, sx1, sy1, sx2, sy2,
        detection.cx, detection.cy, min_area
    )
    if found is None:
        return False

    comp_cx, comp_cy, bx1, by1, bx2, by2 = found

    if mode == 'centroid_keep_size':
        # Keep original box dimensions, but recenter on component centroid.
        # This is best for noteheads — pitch only depends on cy and the
        # original size is usually about right.
        new_cx = int(round(comp_cx))
        new_cy = int(round(comp_cy))
        half_w = (detection.x2 - detection.x1) // 2
        half_h = (detection.y2 - detection.y1) // 2
        new_x1 = new_cx - half_w
        new_y1 = new_cy - half_h
        new_x2 = new_cx + half_w
        new_y2 = new_cy + half_h
    else:
        # 'tight_bbox' — use the component's actual bounding box.
        new_x1, new_y1, new_x2, new_y2 = bx1, by1, bx2, by2
        new_cx = (new_x1 + new_x2) // 2
        new_cy = (new_y1 + new_y2) // 2

    # Sanity check: refusal cases that suggest a bad refinement
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    orig_w = detection.x2 - detection.x1
    orig_h = detection.y2 - detection.y1
    if new_w <= 1 or new_h <= 1:
        return False
    # Reject if refined box is over 2× the original dimension —
    # we've probably latched onto a stem or beam.
    if new_w > orig_w * 2 or new_h > orig_h * 2:
        return False

    detection.x1 = int(new_x1)
    detection.y1 = int(new_y1)
    detection.x2 = int(new_x2)
    detection.y2 = int(new_y2)
    detection.cx = int(new_cx)
    detection.cy = int(new_cy)

    # full_* uses absolute (rectified-image) coordinates
    detection.full_cx = detection.cx
    detection.full_cy = detection.cy + crop_y1

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Whole-page refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_page(page_detections, processed_score,
                verbose: bool = True) -> int:
    """
    Refine every detection on the page in place.  Returns the number of
    detections that were modified.

    The two inputs must come from the same image — the function pairs each
    StaffDetections with its corresponding ProcessedStaff by part/staff index.
    """
    n_modified = 0
    n_total    = 0

    for part_staves, staff_dets_list in zip(processed_score.parts,
                                              page_detections.parts):
        for pstaff, sdet in zip(part_staves, staff_dets_list):
            cleaned = pstaff.cleaned
            crop_y1 = pstaff.crop_y1
            for det in sdet.detections:
                n_total += 1
                if refine_detection(det, cleaned, crop_y1):
                    n_modified += 1

    if verbose:
        print(f'Refined {n_modified} / {n_total} detections '
              f'({100 * n_modified / max(n_total, 1):.1f}%)')
    return n_modified


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_refinement(processed_score,
                         before_detections,
                         after_detections,
                         output_dir: str,
                         only_classes: Optional[List[str]] = None):
    """
    Side-by-side comparison: original YOLO boxes (red) vs refined boxes
    (green) overlaid on the cleaned staff crop.

    Saves one image per staff: <part_id>_staff<NN>_refine.png

    Pass `only_classes=['noteheadBlack', 'noteheadHalf']` etc. to focus on
    specific symbol types.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = list(zip(processed_score.parts,
                     before_detections.parts,
                     after_detections.parts))

    for part_staves, before_part, after_part in pairs:
        for pstaff, before_sd, after_sd in zip(part_staves, before_part, after_part):
            base = pstaff.crop
            if base.ndim == 2:
                vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            else:
                vis = base.copy()

            for b, a in zip(before_sd.detections, after_sd.detections):
                if only_classes and b.class_name not in only_classes:
                    continue
                # Original (red, dashed-effect via thinner stroke)
                cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2),
                              (0, 0, 255), 1)
                # Refined (green, thicker)
                cv2.rectangle(vis, (a.x1, a.y1), (a.x2, a.y2),
                              (0, 200, 0), 2)
                # Centers
                cv2.circle(vis, (b.cx, b.cy), 2, (0, 0, 255), -1)
                cv2.circle(vis, (a.cx, a.cy), 2, (0, 200, 0), -1)

            cv2.putText(vis,
                        f'{after_sd.part_id} staff{after_sd.staff_in_part+1} '
                        f'  red=YOLO   green=refined',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)

            fname = f'{after_sd.part_id}_staff{after_sd.staff_in_part+1:02d}_refine.png'
            cv2.imwrite(str(out / fname), vis)

    print(f'Refinement comparisons → {output_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# Helper for callers who want a copy instead of in-place mutation
# ─────────────────────────────────────────────────────────────────────────────

def refine_page_copy(page_detections, processed_score, verbose: bool = True):
    """
    Returns a deep copy of page_detections with refinement applied,
    leaving the original untouched.
    """
    refined = copy.deepcopy(page_detections)
    refine_page(refined, processed_score, verbose=verbose)
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append(r'S:\omr')
    from preprocessing import preprocess_image
    from symbol_detector import detect_page

    test_img   = r'S:\mmdetection\data\my_images\img_1.png'
    model_path = r'S:\omr\models\deepscores_crops_v1.pt'
    out_dir    = r'S:\omr\refine_test'

    print('Preprocessing …')
    processed = preprocess_image(test_img)

    print('Detecting …')
    detections = detect_page(processed, model_path)

    print('Saving original detections for comparison …')
    before = copy.deepcopy(detections)

    print('Refining …')
    refine_page(detections, processed)

    print('Visualizing …')
    visualize_refinement(processed, before, detections, out_dir,
                         only_classes=['noteheadBlack', 'noteheadHalf',
                                       'noteheadWhole'])
    print(f'Done → {out_dir}')
