"""
score_analyzer.py

First stage of the OMR pipeline.
Takes an image path, detects all staff lines, groups them into parts,
crops each staff (to its actual horizontal extent + padding), and returns
structured data organized by part.

Updated logic
-------------
* Each staff crop covers only the horizontal extent of that staff (with
  configurable padding), not the full page width.  This prevents wasted
  pixels and mis-cropping when a page contains partial-width staves
  (e.g. cadenzas, two-system-per-row layouts).

* Staves are grouped into "rows" by clustering on top_y, then within each
  row sorted left-to-right by left_x.  This means side-by-side staves at
  the same y are correctly ordered: the leftmost one is staff #1 of that
  row, not interleaved with a different system.

* Part assignment happens within each row: position-in-row % num_parts.

Usage:
    from score_analyzer import analyze_score
    score = analyze_score('page1.png')
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from scipy.signal import find_peaks


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class StaffData:
    """
    All data for a single detected staff line.
    All coordinates are in FULL IMAGE pixels.
    """
    staff_idx: int          # global index across whole page (0-based)
    part_id: str            # e.g. 'P1', 'P2', 'P3'
    staff_in_part: int      # index within the part (0-based)

    # Full image coordinates
    top_y: int              # y of top staff line
    bot_y: int              # y of bottom staff line
    left_x: int             # x where staff actually starts (foreground extent)
    right_x: int            # x where staff actually ends
    line_spacing: float     # pixels between adjacent staff lines

    # 5 individual staff line y positions in full image coords
    line_positions: List[int] = field(default_factory=list)

    # Crop coordinates in full image (now reflect the actual x-extent + padding)
    crop_y1: int = 0
    crop_y2: int = 0
    crop_x1: int = 0
    crop_x2: int = 0

    # The actual cropped image
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    # Which "row" of staves this belongs to (rows = systems running across page)
    row_idx: int = 0


@dataclass
class PartData:
    """One instrument part - contains one staff per system."""
    part_id: str            # e.g. 'P1'
    part_idx: int           # 0-based index
    staves: List[StaffData] = field(default_factory=list)


@dataclass
class ScoreData:
    """Full page analysis result."""
    image_path: str
    img_h: int
    img_w: int
    num_parts: int
    parts: List[PartData] = field(default_factory=list)
    all_staves: List[StaffData] = field(default_factory=list)


# ─────────────────────────────────────────────
# Part count (manually adjustable)
# ─────────────────────────────────────────────

def get_num_parts() -> int:
    """
    Returns the number of instrument parts per system.

    ADJUST THIS VALUE for different scores:
    - 3 for 3-clarinet Georgian folk music
    - 2 for piano (treble + bass)
    - 1 for single melody line
    """
    return 3


# ─────────────────────────────────────────────
# Staff line detection
# ─────────────────────────────────────────────

def _binarize(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _horizontal_strokes(binary: np.ndarray, min_run_frac: float = 0.15) -> np.ndarray:
    """
    Extract long horizontal strokes.  min_run_frac is reduced from 0.3 to
    0.15 so that half-width staves still survive the morph operation.
    """
    w = max(20, int(binary.shape[1] * min_run_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _cluster_close_peaks(peaks: List[int], max_gap: int = 2) -> List[int]:
    if not peaks:
        return []
    peaks = sorted(peaks)
    clusters = [[peaks[0]]]
    for p in peaks[1:]:
        if p - clusters[-1][-1] <= max_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(round(sum(c) / len(c))) for c in clusters]


def _group_lines_into_staves(lines: List[int], spacing_tol: float = 0.35) -> List[List[int]]:
    if len(lines) < 3:
        return []
    lines = sorted(lines)
    gaps = [lines[i+1] - lines[i] for i in range(len(lines)-1) if lines[i+1] > lines[i]]
    if not gaps:
        return []
    median_gap = float(np.median(gaps))
    small = [g for g in gaps if g <= median_gap * 1.5]
    target_spacing = float(np.median(small)) if small else median_gap
    if target_spacing <= 0:
        return []
    tol = max(1.5, target_spacing * spacing_tol)

    staves = []
    used = set()
    for start_idx in range(len(lines)):
        if start_idx in used:
            continue
        top = lines[start_idx]
        predicted = [top + k * target_spacing for k in range(5)]
        matched_indices = []
        final_rows = []
        for pred in predicted:
            best_j, best_dist = -1, tol
            for j, ln in enumerate(lines):
                if j in used or j in matched_indices:
                    continue
                d = abs(ln - pred)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                matched_indices.append(best_j)
                final_rows.append(lines[best_j])
            else:
                final_rows.append(int(round(pred)))
        if len(matched_indices) >= 3:
            final_rows.sort()
            staves.append(final_rows)
            used.update(matched_indices)

    staves.sort(key=lambda s: s[0])
    return staves


def _staff_horizontal_extent(horizontal: np.ndarray, top_y: int, bot_y: int,
                              threshold_frac: float = 0.2) -> Tuple[int, int]:
    """
    Find leftmost and rightmost x where the staff lines actually exist.
    Used to compute crop_x1/crop_x2 so each crop matches its real width.
    """
    band = horizontal[top_y:bot_y+1, :]
    if band.size == 0:
        return 0, horizontal.shape[1] - 1
    col_sum = band.sum(axis=0)
    if col_sum.max() == 0:
        return 0, horizontal.shape[1] - 1
    threshold = col_sum.max() * threshold_frac
    active = np.where(col_sum > threshold)[0]
    if active.size == 0:
        return 0, horizontal.shape[1] - 1
    return int(active[0]), int(active[-1])


def _detect_raw_staves(image_path: str) -> tuple:
    """
    Detect all raw staff groups from image.

    The peak threshold is now expressed relative to a half-width baseline
    (instead of the global max), so partial-width staves register as peaks
    too.
    """
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape

    binary = _binarize(img_gray)
    horizontal = _horizontal_strokes(binary)

    row_sum = horizontal.sum(axis=1)
    max_val = row_sum.max()
    if max_val == 0:
        return img_color, img_gray, [], [], horizontal

    # Lower threshold (was 0.3 of max) — half-width staves were being filtered
    # out when a full-width staff dominated the max.
    peaks, _ = find_peaks(row_sum, height=max_val * 0.15, distance=2)
    clustered = _cluster_close_peaks(peaks.tolist(), max_gap=2)
    staves_lines = _group_lines_into_staves(clustered)

    line_spacings = []
    for window in staves_lines:
        gaps = [window[k+1] - window[k] for k in range(4)]
        line_spacings.append(float(sum(gaps) / 4.0))

    return img_color, img_gray, staves_lines, line_spacings, horizontal


# ─────────────────────────────────────────────
# Row clustering & ordering
# ─────────────────────────────────────────────

def _cluster_into_rows(staves_meta: List[dict],
                        row_y_tolerance: Optional[float] = None) -> List[List[int]]:
    """
    Group staves into horizontal rows by clustering on top_y.

    A "row" is a horizontal band of the page where one or more staves coexist.
    On a normal page each row is a single system; on a page with two systems
    side-by-side a row contains both of them.

    staves_meta — list of dicts each containing 'top_y', 'bot_y', 'left_x'.
    row_y_tolerance — staves whose top_y differ by less than this are
                      considered to be in the same row.  If None, computed
                      from the median staff height.

    Returns a list of rows; each row is a list of staff indices into
    staves_meta, sorted by left_x (left to right).
    """
    if not staves_meta:
        return []

    # Default tolerance: half the median staff height.  This is generous
    # enough to handle slight y-misalignment between side-by-side staves
    # but tight enough that vertically separated systems stay in separate rows.
    if row_y_tolerance is None:
        heights = [s['bot_y'] - s['top_y'] for s in staves_meta]
        row_y_tolerance = float(np.median(heights)) * 0.5

    # Sort by top_y
    sorted_idxs = sorted(range(len(staves_meta)),
                          key=lambda i: staves_meta[i]['top_y'])

    rows: List[List[int]] = []
    current_row: List[int] = []
    current_anchor_y: Optional[float] = None

    for idx in sorted_idxs:
        y = staves_meta[idx]['top_y']
        if current_anchor_y is None or abs(y - current_anchor_y) <= row_y_tolerance:
            current_row.append(idx)
            # Update anchor to running mean (more stable than just first y)
            ys = [staves_meta[i]['top_y'] for i in current_row]
            current_anchor_y = float(np.mean(ys))
        else:
            rows.append(current_row)
            current_row = [idx]
            current_anchor_y = float(y)
    if current_row:
        rows.append(current_row)

    # Sort each row left-to-right by left_x
    for row in rows:
        row.sort(key=lambda i: staves_meta[i]['left_x'])

    return rows


# ─────────────────────────────────────────────
# Cropping (now uses actual horizontal extent)
# ─────────────────────────────────────────────

def _crop_staff(img_color: np.ndarray, img_h: int, img_w: int,
                top_y: int, bot_y: int,
                left_x: int, right_x: int,
                padding_ratio: float = 0.95,
                horizontal_padding_ratio: float = 0.05) -> tuple:
    """
    Crop a staff using its actual horizontal extent (plus padding) rather
    than the full page width.

    Vertical padding (padding_ratio * staff_height) leaves room for stems
    above and below the staff.  Horizontal padding (small, default 5%) makes
    sure clefs/key signatures at the very left and final barlines at the
    very right aren't trimmed.

    Returns (crop, crop_y1, crop_y2, crop_x1, crop_x2).
    """
    staff_height = bot_y - top_y
    v_pad = int(staff_height * padding_ratio)
    h_pad = int((right_x - left_x) * horizontal_padding_ratio)
    h_pad = max(h_pad, 8)   # always at least a few pixels of horizontal margin

    crop_y1 = max(0, top_y - v_pad)
    crop_y2 = min(img_h, bot_y + v_pad)
    crop_x1 = max(0, left_x - h_pad)
    crop_x2 = min(img_w, right_x + h_pad)

    crop = img_color[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, crop_y1, crop_y2, crop_x1, crop_x2


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

def analyze_score(image_path: str,
                  padding_ratio: float = 0.95,
                  horizontal_padding_ratio: float = 0.05) -> ScoreData:
    """
    Main entry point. Analyzes a full score page.

    Args:
        image_path:               path to the score image
        padding_ratio:            vertical padding around each staff crop
        horizontal_padding_ratio: horizontal margin beyond the staff extent

    Returns ScoreData with parts and all_staves populated.

    Each StaffData has:
        .crop             — cropped image (actual width × padded height)
        .line_positions   — 5 y coords in FULL IMAGE pixels
        .crop_x1/x2       — horizontal crop boundaries (now reflect extent)
        .crop_y1/y2       — vertical crop boundaries
        .left_x/right_x   — the staff's actual extent in full image coords
        .row_idx          — which row of the page this staff belongs to
    """
    img_color, img_gray, staves_lines, line_spacings, horizontal = \
        _detect_raw_staves(image_path)

    img_h, img_w = img_gray.shape
    num_parts = get_num_parts()

    if not staves_lines:
        print(f'No staves detected in {image_path}')
        return ScoreData(image_path, img_h, img_w, num_parts)

    print(f'Detected {len(staves_lines)} staves')

    # ── Step 1: build per-staff metadata (without cropping yet) ──
    staves_meta: List[dict] = []
    for window, spacing in zip(staves_lines, line_spacings):
        top_y = window[0]
        bot_y = window[-1]
        left_x, right_x = _staff_horizontal_extent(horizontal, top_y, bot_y)
        staves_meta.append({
            'top_y':          top_y,
            'bot_y':          bot_y,
            'left_x':         left_x,
            'right_x':        right_x,
            'line_spacing':   spacing,
            'line_positions': window,
        })

    # ── Step 2: cluster into rows, then sort each row by x ──
    rows = _cluster_into_rows(staves_meta)
    print(f'Grouped into {len(rows)} rows; '
          f'staves per row: {[len(r) for r in rows]}')

    # ── Step 3: assemble in row-major, left-to-right order ──
    parts = [PartData(part_id=f'P{i+1}', part_idx=i)
             for i in range(num_parts)]
    all_staves: List[StaffData] = []

    staff_idx = 0
    for row_idx, row in enumerate(rows):
        # Within a row, position 0 → P1, 1 → P2, …, then wrap with %.
        # If a row has more staves than parts (e.g. 6 staves with 3 parts =
        # two systems side-by-side), the wrap correctly assigns P1-P3 to
        # the left system and P1-P3 again to the right system, but those
        # right-system staves are appended to the same parts as additional
        # entries — which corresponds to the next "system" of each part.
        for pos, idx in enumerate(row):
            meta = staves_meta[idx]
            part_idx = pos % num_parts
            part = parts[part_idx]

            crop, cy1, cy2, cx1, cx2 = _crop_staff(
                img_color, img_h, img_w,
                meta['top_y'], meta['bot_y'],
                meta['left_x'], meta['right_x'],
                padding_ratio,
                horizontal_padding_ratio,
            )

            staff_data = StaffData(
                staff_idx       = staff_idx,
                part_id         = f'P{part_idx + 1}',
                staff_in_part   = len(part.staves),
                top_y           = meta['top_y'],
                bot_y           = meta['bot_y'],
                left_x          = meta['left_x'],
                right_x         = meta['right_x'],
                line_spacing    = meta['line_spacing'],
                line_positions  = meta['line_positions'],
                crop_y1         = cy1,
                crop_y2         = cy2,
                crop_x1         = cx1,
                crop_x2         = cx2,
                crop            = crop,
                row_idx         = row_idx,
            )
            part.staves.append(staff_data)
            all_staves.append(staff_data)
            staff_idx += 1

    score = ScoreData(
        image_path = image_path,
        img_h      = img_h,
        img_w      = img_w,
        num_parts  = num_parts,
        parts      = parts,
        all_staves = all_staves,
    )
    return score


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("score_analyzer.py: Run via pipeline.py or import analyze_score.")
