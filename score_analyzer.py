"""
score_analyzer.py

First stage of the OMR pipeline.
Takes an image path, detects all staff lines, groups them into parts,
crops each staff, and returns structured data organized by part.

Usage:
    from score_analyzer import analyze_score
    score = analyze_score('page1.png')
    
    # score.parts is a list of Part objects
    for part in score.parts:
        print(f'Part {part.part_id}: {len(part.staves)} staves')
        for staff in part.staves:
            print(f'  Staff y={staff.top_y}-{staff.bot_y}')
            # staff.crop is the cropped image
            # staff.lines is list of 5 y positions in FULL IMAGE coords
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
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
    left_x: int             # x where staff starts
    right_x: int            # x where staff ends
    line_spacing: float     # pixels between adjacent staff lines

    # 5 individual staff line y positions in full image coords
    line_positions: List[int] = field(default_factory=list)

    # Crop coordinates in full image
    crop_y1: int = 0
    crop_y2: int = 0
    crop_x1: int = 0
    crop_x2: int = 0

    # The actual cropped image (full width, padded height)
    crop: Optional[np.ndarray] = field(default=None, repr=False)


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
    - 4 for string quartet, etc.
    
    Future improvement: auto-detect from brace/bracket symbols.
    """
    return 3


# ─────────────────────────────────────────────
# Staff line detection (from uploaded script logic)
# ─────────────────────────────────────────────

def _binarize(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def _horizontal_strokes(binary: np.ndarray, min_run_frac: float = 0.3) -> np.ndarray:
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

def _staff_horizontal_extent(horizontal: np.ndarray, top_y: int, bot_y: int) -> tuple:
    band = horizontal[top_y:bot_y+1, :]
    if band.size == 0:
        return 0, horizontal.shape[1] - 1
    col_sum = band.sum(axis=0)
    if col_sum.max() == 0:
        return 0, horizontal.shape[1] - 1
    threshold = col_sum.max() * 0.2
    active = np.where(col_sum > threshold)[0]
    if active.size == 0:
        return 0, horizontal.shape[1] - 1
    return int(active[0]), int(active[-1])

def _detect_raw_staves(image_path: str) -> tuple:
    """
    Detect all raw staff groups from image.
    Returns (img_color, img_gray, staves_lines, line_spacings, horizontal_mask)
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

    peaks, _ = find_peaks(row_sum, height=max_val * 0.3, distance=2)
    clustered = _cluster_close_peaks(peaks.tolist(), max_gap=2)
    staves_lines = _group_lines_into_staves(clustered)

    line_spacings = []
    for window in staves_lines:
        gaps = [window[k+1] - window[k] for k in range(4)]
        line_spacings.append(float(sum(gaps) / 4.0))

    return img_color, img_gray, staves_lines, line_spacings, horizontal


# ─────────────────────────────────────────────
# Grouping into parts
# ─────────────────────────────────────────────

def _group_staves_into_parts(staves_lines: List[List[int]], num_parts: int) -> List[List[int]]:
    """
    Group detected staves into parts based on num_parts.
    
    If num_parts=3 and there are 9 staves:
    - System 1: staves 0,1,2  -> part 0, part 1, part 2
    - System 2: staves 3,4,5  -> part 0, part 1, part 2
    - System 3: staves 6,7,8  -> part 0, part 1, part 2
    
    Returns list of lists, one per part, each containing
    the indices of staves belonging to that part.
    """
    part_groups = [[] for _ in range(num_parts)]
    for i, _ in enumerate(staves_lines):
        part_idx = i % num_parts
        part_groups[part_idx].append(i)
    return part_groups


# ─────────────────────────────────────────────
# Cropping
# ─────────────────────────────────────────────

def _crop_staff(img_color: np.ndarray, img_h: int, img_w: int,
                top_y: int, bot_y: int, padding_ratio: float = 0.95) -> tuple:
    """
    Crop a staff from the full image using full width.
    Returns (crop, crop_y1, crop_y2)
    """
    staff_height = bot_y - top_y
    pad = int(staff_height * padding_ratio)
    crop_y1 = max(0, top_y - pad)
    crop_y2 = min(img_h, bot_y + pad)
    crop = img_color[crop_y1:crop_y2, 0:img_w]
    return crop, crop_y1, crop_y2


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

def analyze_score(image_path: str, padding_ratio: float = 0.95) -> ScoreData:
    """
    Main entry point. Analyzes a full score page.
    
    Args:
        image_path:    path to the score image
        padding_ratio: vertical padding around each staff crop (default 0.95)
    
    Returns ScoreData containing:
        score.parts       - list of PartData, one per instrument
        score.all_staves  - flat list of all StaffData sorted top to bottom
        
    Each StaffData has:
        .crop             - cropped image (full width, padded height)
        .line_positions   - list of 5 y coords in FULL IMAGE pixels
        .crop_y1/y2       - crop boundaries in full image
        .part_id          - 'P1', 'P2', etc.
        .top_y, .bot_y    - staff boundaries in full image
        .line_spacing     - pixels between staff lines
    
    Example:
        score = analyze_score('page1.png')
        for part in score.parts:
            for staff in part.staves:
                cv2.imwrite(f'{part.part_id}_staff.png', staff.crop)
                print(staff.line_positions)  # [y1,y2,y3,y4,y5] in full image
    """
    img_color, img_gray, staves_lines, line_spacings, horizontal = \
        _detect_raw_staves(image_path)

    img_h, img_w = img_gray.shape
    num_parts = get_num_parts()

    if not staves_lines:
        print(f'No staves detected in {image_path}')
        return ScoreData(image_path, img_h, img_w, num_parts)

    print(f'Detected {len(staves_lines)} staves, {num_parts} parts per system')

    # Group stave indices into parts
    part_groups = _group_staves_into_parts(staves_lines, num_parts)

    # Build PartData objects
    parts = [PartData(part_id=f'P{i+1}', part_idx=i)
             for i in range(num_parts)]

    # Build all StaffData
    all_staves = []
    for staff_idx, (window, spacing) in enumerate(zip(staves_lines, line_spacings)):
        part_idx = staff_idx % num_parts
        part = parts[part_idx]

        top_y = window[0]
        bot_y = window[-1]
        left_x, right_x = _staff_horizontal_extent(horizontal, top_y, bot_y)

        crop, crop_y1, crop_y2 = _crop_staff(
            img_color, img_h, img_w, top_y, bot_y, padding_ratio)

        staff_data = StaffData(
            staff_idx=staff_idx,
            part_id=f'P{part_idx+1}',
            staff_in_part=len(part.staves),
            top_y=top_y,
            bot_y=bot_y,
            left_x=left_x,
            right_x=right_x,
            line_spacing=spacing,
            line_positions=window,       # full image y coords
            crop_y1=crop_y1,
            crop_y2=crop_y2,
            crop_x1=0,
            crop_x2=img_w,
            crop=crop
        )

        part.staves.append(staff_data)
        all_staves.append(staff_data)

    score = ScoreData(
        image_path=image_path,
        img_h=img_h,
        img_w=img_w,
        num_parts=num_parts,
        parts=parts,
        all_staves=all_staves
    )

    return score


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os

    test_img = r'S:\mmdetection\data\my_images\img_1.png'
    score = analyze_score(test_img)

    print(f'\nImage: {score.image_path}')
    print(f'Size: {score.img_w}x{score.img_h}')
    print(f'Parts: {score.num_parts}')
    print(f'Total staves detected: {len(score.all_staves)}')

    for part in score.parts:
        print(f'\n{part.part_id} ({len(part.staves)} staves):')
        for staff in part.staves:
            print(f'  Staff {staff.staff_in_part+1}: '
                  f'y={staff.top_y}-{staff.bot_y} '
                  f'spacing={staff.line_spacing:.1f}px '
                  f'lines={staff.line_positions}')

    # Save crops for verification
    out_dir = r'S:\omr\score_test'
    os.makedirs(out_dir, exist_ok=True)
    for staff in score.all_staves:
        path = os.path.join(out_dir,
            f'{staff.part_id}_staff{staff.staff_in_part+1:02d}.png')
        cv2.imwrite(path, staff.crop)
    print(f'\nCrops saved to {out_dir}')