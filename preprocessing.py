"""
preprocessing.py

Takes a score image, detects staves, groups into parts, crops each staff,
and removes staff lines. Returns structured data ready for symbol detection.

Called from:
    symbol_detector.py (pipeline)
    
Standalone test:
    python preprocessing.py
"""

from __future__ import annotations

import sys
import os
sys.path.append(r'S:\omr')

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from score_analyzer import analyze_score, ScoreData, StaffData
from staff_remover import clean_staff_image


@dataclass
class ProcessedStaff:
    """
    A single staff after full preprocessing.
    Contains both the original crop and the cleaned version,
    plus all coordinate metadata for XML reconstruction.
    """
    # From score_analyzer
    staff_data: StaffData

    # Processed images
    crop: np.ndarray = field(repr=False)          # original color crop
    cleaned: np.ndarray = field(repr=False)        # binarized, staff lines removed

    # Shortcuts to commonly used fields
    @property
    def part_id(self): return self.staff_data.part_id
    @property
    def staff_idx(self): return self.staff_data.staff_idx
    @property
    def staff_in_part(self): return self.staff_data.staff_in_part
    @property
    def top_y(self): return self.staff_data.top_y
    @property
    def bot_y(self): return self.staff_data.bot_y
    @property
    def left_x(self): return self.staff_data.left_x
    @property
    def right_x(self): return self.staff_data.right_x
    @property
    def line_positions(self): return self.staff_data.line_positions
    @property
    def line_spacing(self): return self.staff_data.line_spacing
    @property
    def crop_y1(self): return self.staff_data.crop_y1
    @property
    def crop_y2(self): return self.staff_data.crop_y2
    @property
    def img_w(self): return self.staff_data.crop_x2


@dataclass
class ProcessedScore:
    """Full preprocessing result for one page."""
    image_path: str
    img_h: int
    img_w: int
    num_parts: int
    # All processed staves grouped by part
    # parts[0] = list of ProcessedStaff for P1, parts[1] for P2, etc.
    parts: List[List[ProcessedStaff]] = field(default_factory=list)
    # Flat list sorted top to bottom
    all_staves: List[ProcessedStaff] = field(default_factory=list)


def preprocess_image(image_path: str, padding_ratio: float = 0.95) -> ProcessedScore:
    """
    Full preprocessing pipeline for a single score page.

    Steps:
        1. Detect staff lines and group into parts (score_analyzer)
        2. Crop each staff at full width with padding
        3. Binarize and remove staff lines (staff_remover)

    Args:
        image_path:    path to score image
        padding_ratio: vertical padding around staff (default 0.95)

    Returns ProcessedScore with:
        .parts[i]         - list of ProcessedStaff for part i
        .all_staves       - all staves sorted top to bottom
        Each ProcessedStaff has:
            .cleaned          - binary image ready for YOLO
            .crop             - original color crop
            .line_positions   - 5 y coords in full image pixels
            .crop_y1/y2       - crop offsets for coordinate restoration
            .part_id          - 'P1', 'P2', etc.
    """
    print(f'Preprocessing: {image_path}')

    # Step 1: Detect and group staves
    score = analyze_score(image_path, padding_ratio=padding_ratio)

    if not score.all_staves:
        print('No staves detected!')
        return ProcessedScore(
            image_path=image_path,
            img_h=score.img_h,
            img_w=score.img_w,
            num_parts=score.num_parts
        )

    print(f'Found {len(score.all_staves)} staves across {score.num_parts} parts')

    # Step 2 & 3: Clean each staff crop
    num_parts = score.num_parts
    parts: List[List[ProcessedStaff]] = [[] for _ in range(num_parts)]
    all_staves: List[ProcessedStaff] = []

    for staff_data in score.all_staves:
        crop = staff_data.crop

        # Binarize and remove staff lines
        cleaned = clean_staff_image(crop)

        processed = ProcessedStaff(
            staff_data=staff_data,
            crop=crop,
            cleaned=cleaned
        )

        part_idx = staff_data.part_idx if hasattr(staff_data, 'part_idx') \
            else int(staff_data.part_id[1:]) - 1
        parts[part_idx].append(processed)
        all_staves.append(processed)

    result = ProcessedScore(
        image_path=image_path,
        img_h=score.img_h,
        img_w=score.img_w,
        num_parts=num_parts,
        parts=parts,
        all_staves=all_staves
    )

    print(f'Preprocessing done. {len(all_staves)} staves ready.')
    return result


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os

    test_img = r'S:\mmdetection\data\my_images\img_1.png'
    out_dir = r'S:\omr\preprocess_test'
    os.makedirs(out_dir, exist_ok=True)

    result = preprocess_image(test_img)

    print(f'\nResults:')
    for staff in result.all_staves:
        print(f'  {staff.part_id} staff {staff.staff_in_part+1}: '
              f'y={staff.top_y}-{staff.bot_y} '
              f'spacing={staff.line_spacing:.1f}px')

        # Save both original and cleaned crops
        cv2.imwrite(
            os.path.join(out_dir, f'{staff.part_id}_staff{staff.staff_in_part+1:02d}.png'),
            staff.crop)
        cv2.imwrite(
            os.path.join(out_dir, f'{staff.part_id}_staff{staff.staff_in_part+1:02d}_clean.png'),
            staff.cleaned)

    print(f'\nSaved to {out_dir}')