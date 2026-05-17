"""
clef_splitter.py — split staves by detected clef positions.

When the U-Net detects a single staff on a page row that actually
contains *two* staves side-by-side (a printed score with two systems
per line, the rectifier merging them into one band, …), we end up
with one big "staff" that spans the whole row.  This module fixes
that by:

    1.  Running the symbol-detection YOLO once on the rectified page
        (filtered to clef classes only).
    2.  For each Stage-1-detected staff, counting clefs that fall
        inside its vertical band.
    3.  If a staff has more than one clef separated by a significant
        horizontal gap, splitting its metadata into one entry per
        clef — each sub-staff's ``left_x`` is just left of its clef
        and ``right_x`` is just left of the next clef (or the
        original ``right_x`` for the last sub-staff).

The output is a new ``staves_meta`` list that the rest of the
pipeline (preprocessing, symbol detection, XML build) consumes
unchanged.  All other staff fields (``line_positions``,
``line_spacing``, vertical band) are inherited from the parent.

This module is deliberately decoupled from the YOLO weights path —
``detect_clefs_in_image`` accepts a model path or a preloaded model
so the caller can reuse the model already loaded for Stage 3.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Classes treated as "main" clefs — clef8 is the small "8" beneath a
# treble clef that marks octave-down transposition; it should NOT be
# treated as a separate stave anchor.
CLEF_CLASSES = frozenset({'clefG', 'clefF', 'clefCAlto', 'clefCTenor'})

# Detection threshold and grouping tolerances.
CLEF_CONF_THRESHOLD = 0.25
MIN_CLEF_GAP_FRAC   = 0.08    # clefs closer than this fraction of the
                              # staff's width are treated as duplicates
LEFT_PADDING_FRAC   = 0.030   # how much to pull each sub-stave's left
                              # edge left of its clef (frac of img_w).
                              # Doubled from the original 0.015 so the
                              # right edge of each sub-stave finishes
                              # well clear of the next clef's bbox.


def detect_clefs_in_image(image: np.ndarray,
                          model_path: str,
                          conf_threshold: float = CLEF_CONF_THRESHOLD,
                          imgsz: int = 1280,
                          model=None,
                          ) -> List[Dict]:
    """
    Run YOLO on the rectified page and return every clef detection.

    Each result dict has::

        {'class': str,         # one of CLEF_CLASSES
         'conf':  float,
         'cx': int, 'cy': int, # centre in image coords
         'x1': int, 'y1': int, # bbox in image coords
         'x2': int, 'y2': int}

    ``model`` lets you pass a pre-loaded ultralytics YOLO instance to
    avoid the file-load cost when the caller already has it open.
    """
    if model is None:
        from ultralytics import YOLO
        model = YOLO(model_path)

    if image.ndim == 2:
        infer = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        infer = image

    res = model(infer, conf=conf_threshold, imgsz=imgsz, verbose=False)[0]
    class_names = model.names
    out: List[Dict] = []
    if res.boxes is None:
        return out
    xyxy   = res.boxes.xyxy.cpu().numpy()
    confs  = res.boxes.conf.cpu().numpy()
    cls_id = res.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_id):
        name = class_names[int(cid)]
        if name not in CLEF_CLASSES:
            continue
        out.append({
            'class': name,
            'conf':  round(float(c), 4),
            'cx':    int((x1 + x2) / 2),
            'cy':    int((y1 + y2) / 2),
            'x1':    int(x1), 'y1': int(y1),
            'x2':    int(x2), 'y2': int(y2),
        })
    return out


def _clefs_in_band(clefs: List[Dict],
                   top_y: int,
                   bot_y: int,
                   margin: int = 0) -> List[Dict]:
    """Return clefs whose centre y falls inside [top_y-margin, bot_y+margin]."""
    return [c for c in clefs if (top_y - margin) <= c['cy'] <= (bot_y + margin)]


def _deduplicate_clefs(clefs: List[Dict],
                       min_gap_px: float) -> List[Dict]:
    """
    Collapse clefs whose x-centres are within ``min_gap_px`` (two YOLO
    detections of the same real clef).  Keeps the highest-confidence
    one in each cluster.
    """
    if not clefs:
        return []
    sorted_c = sorted(clefs, key=lambda c: c['cx'])
    groups: List[List[Dict]] = [[sorted_c[0]]]
    for c in sorted_c[1:]:
        if c['cx'] - groups[-1][-1]['cx'] <= min_gap_px:
            groups[-1].append(c)
        else:
            groups.append([c])
    return [max(g, key=lambda c: c['conf']) for g in groups]


def split_staves_by_clefs(staves_meta: List[Dict],
                          clefs: List[Dict],
                          img_w: int) -> List[Dict]:
    """
    Return a new staves_meta list where any staff that contained
    multiple clefs has been split into one entry per clef.

    Sub-stave geometry:
      - vertical band, line_positions, line_spacing → inherited from
        the parent.
      - left_x  = max(0, clef_x - left_padding)
      - right_x = midpoint to next clef (in this staff), or the
        parent's right_x for the last sub-staff.

    Staves with 0 or 1 clef are returned unchanged.
    """
    if not staves_meta:
        return staves_meta
    if not clefs:
        return staves_meta

    left_pad = max(8, int(round(LEFT_PADDING_FRAC * img_w)))

    out: List[Dict] = []
    for s in staves_meta:
        top_y = int(s['top_y'])
        bot_y = int(s['bot_y'])
        # A bit of margin so a clef sitting slightly above the top
        # line still counts as belonging to this staff.
        margin = max(4, int(0.5 * (bot_y - top_y)))
        band_clefs = _clefs_in_band(clefs, top_y, bot_y, margin=margin)

        if len(band_clefs) <= 1:
            out.append(s)
            continue

        staff_w  = max(1, int(s['right_x']) - int(s['left_x']))
        min_gap  = max(8, int(MIN_CLEF_GAP_FRAC * staff_w))
        band_clefs = _deduplicate_clefs(band_clefs, min_gap)
        band_clefs.sort(key=lambda c: c['cx'])

        if len(band_clefs) <= 1:
            out.append(s)
            continue

        # Boundary rule using the actual clef bounding boxes.  Each
        # sub-stave starts a few pixels left of its clef's LEFT edge
        # (clef.x1, not clef.cx) and ends one pixel before the NEXT
        # sub-stave's left edge.  Using x1 instead of cx accounts for
        # the clef glyph's width — a previous version used the centre
        # and ended up letting the *first* stave's right edge stretch
        # into the second clef's bounding box.
        parent_left  = int(s['left_x'])
        parent_right = int(s['right_x'])

        clef_lefts: List[int] = []
        for clef in band_clefs:
            # Anchor on x1 (the detected left edge of the clef) — clamp
            # to parent bounds and apply a small left padding so the
            # clef itself is fully inside its sub-stave.
            anchor = int(clef.get('x1', clef['cx']))
            cl_left = anchor - left_pad
            cl_left = max(parent_left, min(parent_right - 10, cl_left))
            clef_lefts.append(cl_left)
        # Guarantee strictly increasing left edges (rare safety net
        # when two clefs survive de-dup but are very close).
        for i in range(1, len(clef_lefts)):
            if clef_lefts[i] <= clef_lefts[i - 1]:
                clef_lefts[i] = clef_lefts[i - 1] + 1

        for i, clef in enumerate(band_clefs):
            sub_left  = clef_lefts[i]
            # End one pixel before the next sub-stave begins so the
            # first stave never includes any of the next clef's
            # bounding box.
            sub_right = (clef_lefts[i + 1] - 1
                         if i + 1 < len(band_clefs)
                         else parent_right)
            if sub_right - sub_left < 10:
                # Degenerate slice; skip to avoid empty crops.
                continue
            new_s = dict(s)          # shallow copy preserves all fields
            new_s['left_x']  = int(sub_left)
            new_s['right_x'] = int(sub_right)
            new_s['_clef_class'] = clef['class']
            new_s['_clef_cx']    = int(clef['cx'])
            out.append(new_s)

    return out


def annotate_image_with_clefs(image: np.ndarray,
                              clefs: List[Dict],
                              out_path: str) -> None:
    """Save a debug visualisation showing every detected clef."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in clefs:
        cv2.rectangle(vis, (c['x1'], c['y1']), (c['x2'], c['y2']),
                      (0, 0, 255), 2)
        cv2.putText(vis, f"{c['class']} {c['conf']:.2f}",
                    (c['x1'], max(12, c['y1'] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)
