"""
measure_detector.py — Detect barline x-coordinates in cleaned staff images.

Position in the pipeline
------------------------
Runs AFTER preprocessing.py (which gives us cleaned, staff-line-removed
crops) and EITHER before or in parallel with symbol_detector.  The output is
a list of x-coordinates per staff that score_to_xml can use to split each
staff into multiple measures.

How it works
------------
After staff_remover removes the horizontal staff lines, vertical strokes that
spanned the full staff height stay intact (they're preserved because pixels
above and below them are also foreground).  A barline is one of these strokes
that:

  1. Has at least one foreground pixel near the TOP staff line position
  2. Has at least one foreground pixel near the BOTTOM staff line position
  3. Has high foreground coverage between top and bottom lines
  4. Is narrow (1–8 px wide)

Stems are filtered out because they typically don't reach BOTH the top and
bottom staff lines — a stem-up from a note in the bottom space reaches the
top, but not the bottom; a stem-down from a high note reaches the bottom
but not the top.

When YOLO detections are also available, you can call
`filter_with_noteheads()` to remove any candidates that have a notehead
within a small horizontal distance — those are stems, not barlines.

Public API
----------
    detect_barlines_in_staff(cleaned, line_positions_crop, ...) -> List[int]
    detect_page_barlines(processed_score) -> PageBarlines
    filter_with_noteheads(barlines, noteheads, max_dx) -> List[int]
    visualize_barlines(processed_score, page_barlines, output_dir)

Usage
-----
    from preprocessing  import preprocess_image
    from measure_detector import detect_page_barlines

    processed = preprocess_image('page1.png')
    barlines  = detect_page_barlines(processed)

    for staff_bl in barlines.all_staves:
        print(f'{staff_bl.part_id} staff{staff_bl.staff_in_part+1}: '
              f'{len(staff_bl.barline_xs)} barlines '
              f'at x={staff_bl.barline_xs}')
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StaffBarlines:
    """Barline x-coordinates for one staff."""
    part_id:         str
    staff_in_part:   int
    barline_xs:      List[int] = field(default_factory=list)   # in rectified-image coords

    @property
    def num_barlines(self) -> int:
        return len(self.barline_xs)


@dataclass
class PageBarlines:
    """Barline information for an entire page."""
    parts:       List[List[StaffBarlines]] = field(default_factory=list)
    all_staves:  List[StaffBarlines]       = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Single-staff detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_barlines_in_staff(cleaned:             np.ndarray,
                             line_positions_crop: List[int],
                             tol_y:               int   = 4,
                             min_coverage:        float = 0.7,
                             max_width:           int   = 8,
                             min_width:           int   = 1) -> List[int]:
    """
    Find barline x-coordinates in a single cleaned staff image.

    Parameters
    ----------
    cleaned              binary image where 0 = foreground (black), 255 = bg
    line_positions_crop  5 staff line y-positions in CROP coordinates
                         (subtract crop_y1 from full-image positions)
    tol_y                tolerance window (px) for "near top/bottom line"
    min_coverage         fraction of column between top and bottom that must
                         be foreground (default 0.7)
    max_width            maximum pixel width of a single barline (filters
                         brace symbols and other thick verticals)
    min_width            minimum pixel width (filters salt-and-pepper noise)

    Returns
    -------
    list of x-coordinates (one per detected barline)
    """
    h, w = cleaned.shape[:2]
    # Make sure the indices are inside the image
    top_y = max(0, min(h - 1, line_positions_crop[0]))
    bot_y = max(0, min(h - 1, line_positions_crop[-1]))
    if bot_y <= top_y:
        return []

    # Foreground mask
    fg = (cleaned == 0)

    # Coverage between top and bottom staff lines
    band = fg[top_y:bot_y + 1, :]
    coverage = band.mean(axis=0)              # shape (w,)

    # Must have foreground near top and bottom lines
    near_top = fg[max(0, top_y - tol_y):min(h, top_y + tol_y + 1), :].any(axis=0)
    near_bot = fg[max(0, bot_y - tol_y):min(h, bot_y + tol_y + 1), :].any(axis=0)

    is_candidate = near_top & near_bot & (coverage > min_coverage)

    # Group runs of consecutive True columns into barlines
    barlines: List[int] = []
    in_run = False
    run_start = 0
    for x in range(w):
        if is_candidate[x] and not in_run:
            in_run = True
            run_start = x
        elif not is_candidate[x] and in_run:
            in_run = False
            run_end = x - 1
            run_w = run_end - run_start + 1
            if min_width <= run_w <= max_width:
                barlines.append((run_start + run_end) // 2)
    if in_run:
        run_end = w - 1
        run_w = run_end - run_start + 1
        if min_width <= run_w <= max_width:
            barlines.append((run_start + run_end) // 2)

    return barlines


def filter_with_noteheads(barline_xs:   List[int],
                          notehead_xs:  List[int],
                          max_dx:       int = 12) -> List[int]:
    """
    Remove barline candidates that lie within max_dx of any notehead — those
    are most likely stems.

    Pass this `notehead_xs` from your symbol_detector output if you want
    cleaner barlines.  Optional; the geometric detector already filters most
    stems on its own.
    """
    return [x for x in barline_xs
            if all(abs(x - nx) > max_dx for nx in notehead_xs)]


# ─────────────────────────────────────────────────────────────────────────────
# Whole-page detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_page_barlines(processed_score,
                         tol_y:        int   = 4,
                         min_coverage: float = 0.7,
                         max_width:    int   = 8) -> PageBarlines:
    """
    Run barline detection on every staff in a ProcessedScore.
    Returns a PageBarlines with the same parts/staves structure as the
    input ProcessedScore.
    """
    num_parts = processed_score.num_parts
    parts: List[List[StaffBarlines]] = [[] for _ in range(num_parts)]
    all_staves: List[StaffBarlines]  = []

    for part_idx, part_staves in enumerate(processed_score.parts):
        for pstaff in part_staves:
            # Convert full-image staff line y's into crop coordinates
            line_positions_crop = [y - pstaff.crop_y1 for y in pstaff.line_positions]

            xs = detect_barlines_in_staff(
                pstaff.cleaned,
                line_positions_crop,
                tol_y=tol_y,
                min_coverage=min_coverage,
                max_width=max_width,
            )

            sb = StaffBarlines(
                part_id       = pstaff.part_id,
                staff_in_part = pstaff.staff_in_part,
                barline_xs    = xs,
            )
            parts[part_idx].append(sb)
            all_staves.append(sb)

    return PageBarlines(parts=parts, all_staves=all_staves)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization & I/O
# ─────────────────────────────────────────────────────────────────────────────

def visualize_barlines(processed_score, page_barlines: PageBarlines,
                       output_dir: str):
    """
    Save one image per staff with detected barlines drawn as red vertical
    lines on top of the cleaned staff crop.  Useful for tuning thresholds.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for part_staves, part_bls in zip(processed_score.parts, page_barlines.parts):
        for pstaff, sb in zip(part_staves, part_bls):
            # Convert binary crop to BGR for color overlay
            if pstaff.cleaned.ndim == 2:
                vis = cv2.cvtColor(pstaff.cleaned, cv2.COLOR_GRAY2BGR)
            else:
                vis = pstaff.cleaned.copy()

            for x in sb.barline_xs:
                cv2.line(vis, (x, 0), (x, vis.shape[0] - 1), (0, 0, 255), 2)

            cv2.putText(vis, f'{sb.part_id} st{sb.staff_in_part+1}: '
                              f'{sb.num_barlines} barlines',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)

            fname = f'{sb.part_id}_staff{sb.staff_in_part+1:02d}_barlines.png'
            cv2.imwrite(str(out / fname), vis)

    print(f'Barline visualizations saved to {out}')


def save_json(page_barlines: PageBarlines, output_path: str):
    """Write barline data as a flat list of staff dicts."""
    data = [asdict(sb) for sb in page_barlines.all_staves]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'Barlines JSON → {output_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append(r'S:\omr')
    from preprocessing import preprocess_image

    test_img = r'S:\mmdetection\data\my_images\img_1.png'
    out_dir  = r'S:\omr\barline_test'

    processed = preprocess_image(test_img)
    barlines  = detect_page_barlines(processed)

    print('\nBarlines per staff:')
    for sb in barlines.all_staves:
        print(f'  {sb.part_id} staff{sb.staff_in_part+1}: '
              f'{sb.num_barlines} → {sb.barline_xs}')

    visualize_barlines(processed, barlines, out_dir)
    save_json(barlines, str(Path(out_dir) / 'barlines.json'))
