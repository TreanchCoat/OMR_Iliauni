"""
staff_rectifier.py — Detect curved staff lines and rectify them via homography.

Designed for handwritten or photographed scores where staff lines are not
perfectly straight (rotation, perspective skew, mild curvature).

Pipeline
--------
1. Detect staff lines as polylines by slicing the image into vertical strips
   and tracking peaks across strips. Handles curves that horizontal-morphology
   on the full image would miss.
2. Group every 5 consecutive lines (with consistent spacing) into a staff.
3. For each staff, fit a quadrilateral to the top-left / top-right /
   bottom-right / bottom-left corners (extrapolated to image edges).
4. Rectangular-crop each staff (with vertical padding for stems/ledger lines).
5. Apply a 4-point perspective warp on each cut to map the quad to a clean
   rectangle of fixed line-spacing.
6. Stitch all rectified staves vertically into a single output image.

Visualization
-------------
This is the debug version — every step writes intermediate files into the
output directory:
    01_detected_staves.png      original image with curves + quads overlaid
    02_cut_NN.png               each rectangular cut, with source quad drawn
    03_rectified_NN.png         each cut after homography
    03_compare_NN.png           side-by-side cut vs rectified
    04_final_stitched.png       final output for the OMR pipeline

Once this looks right, the inner functions can be imported into a pipeline
module without the visualization steps.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Tunable parameters
# ─────────────────────────────────────────────────────────────────────────────

N_STRIPS              = 24       # Vertical strips for tracing curves
KERNEL_WIDTH_RATIO    = 0.04     # Horizontal morph kernel width as % of img_w
PEAK_HEIGHT_RATIO     = 0.5      # Min peak height as fraction of strip max
PEAK_DISTANCE         = 4        # Min vertical separation between peaks (px)
SPACING_TOL           = 0.30     # Max gap deviation from median in a staff (30%)
MIN_TRACK_FRAC        = 0.25     # Min fraction of strips a track must span

TARGET_LINE_SPACING   = 18       # Output line spacing in pixels
VERTICAL_PAD          = 100      # Padding above & below staff in rectified out
STAVES_GAP            = 80       # Gap between rectified staves in stitched img

# Drawing
COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),
          (255,0,255),(255,255,0),(128,0,255),(0,128,255)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Curve detection
# ─────────────────────────────────────────────────────────────────────────────

def _binarize(img_gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def detect_staff_curves(img_gray: np.ndarray) -> List[dict]:
    """
    Detect staff lines as polylines (handles curvature/rotation).

    Returns a list of staff dicts, one per detected 5-line group:
        {
            'tracks':       list of 5 polylines, each [(x, y), ...]
            'top_curve':    polyline for the top staff line
            'bottom_curve': polyline for the bottom staff line
            'left_x':       leftmost x covered by any line in this staff
            'right_x':      rightmost x covered
            'line_spacing': median spacing between adjacent lines
        }
    """
    img_h, img_w = img_gray.shape
    binary = _binarize(img_gray)

    # Mild horizontal morph — narrow enough to follow curves
    kernel_w = max(20, int(img_w * KERNEL_WIDTH_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Slice into vertical strips, find peak rows in each
    strip_w = img_w // N_STRIPS
    strip_data: List[Tuple[int, List[int]]] = []
    for i in range(N_STRIPS):
        x1 = i * strip_w
        x2 = min(x1 + strip_w, img_w)
        if x2 <= x1:
            continue
        strip = horizontal[:, x1:x2]
        row_sum = strip.sum(axis=1)
        if row_sum.max() == 0:
            continue
        peaks, _ = find_peaks(row_sum,
                              height=row_sum.max() * PEAK_HEIGHT_RATIO,
                              distance=PEAK_DISTANCE)
        if len(peaks) == 0:
            continue
        strip_data.append(((x1 + x2) // 2, sorted(peaks.tolist())))

    if not strip_data:
        return []

    # Estimate typical staff-line spacing from the first dense strip
    sample = sorted(strip_data, key=lambda s: -len(s[1]))[0][1]
    if len(sample) >= 2:
        gaps = np.diff(sample)
        small_gaps = gaps[gaps <= np.median(gaps) * 1.5]
        est_spacing = float(np.median(small_gaps)) if len(small_gaps) else float(np.median(gaps))
    else:
        est_spacing = 18.0
    y_match_tol = max(3.0, est_spacing * 0.4)

    # Walk strips left→right, extending tracks by best y-match
    tracks: List[List[Tuple[int, int]]] = []
    for x_center, ys in strip_data:
        # Each peak greedily picks the closest still-open track
        for y in ys:
            best_track = None
            best_dist  = y_match_tol
            for tr in tracks:
                last_x, last_y = tr[-1]
                # Must be a recent track (within 2 strip widths)
                if x_center - last_x > strip_w * 2.5:
                    continue
                d = abs(y - last_y)
                if d < best_dist:
                    best_dist  = d
                    best_track = tr
            if best_track is not None:
                best_track.append((x_center, y))
            else:
                tracks.append([(x_center, y)])

    # Drop short tracks (likely noise)
    min_track_pts = max(3, int(N_STRIPS * MIN_TRACK_FRAC))
    tracks = [t for t in tracks if len(t) >= min_track_pts]

    # Sort by mean y
    tracks.sort(key=lambda t: np.mean([p[1] for p in t]))
    if len(tracks) < 5:
        return []

    # Group consecutive 5-tracks into staves where spacing is consistent
    track_y = [float(np.mean([p[1] for p in t])) for t in tracks]
    staves: List[dict] = []
    used = set()
    i = 0
    while i + 4 < len(tracks):
        if i in used:
            i += 1
            continue
        group_y = track_y[i:i+5]
        gaps = [group_y[k+1] - group_y[k] for k in range(4)]
        if min(gaps) <= 0:
            i += 1
            continue
        median_gap = float(np.median(gaps))
        if all(abs(g - median_gap) / median_gap < SPACING_TOL for g in gaps):
            staff_tracks = tracks[i:i+5]
            all_x = [p[0] for t in staff_tracks for p in t]
            staves.append({
                'tracks':       staff_tracks,
                'top_curve':    staff_tracks[0],
                'bottom_curve': staff_tracks[4],
                'left_x':       min(all_x),
                'right_x':      max(all_x),
                'line_spacing': median_gap,
            })
            for k in range(5):
                used.add(i + k)
            i += 5
        else:
            i += 1
    return staves


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quadrilateral fitting
# ─────────────────────────────────────────────────────────────────────────────

def get_staff_quad(staff: dict, img_w: int) -> Tuple[Tuple[int, int], ...]:
    """
    Get 4 corner points (TL, TR, BR, BL) for a staff, extrapolated to image
    edges using a linear fit on each curve.

    These corners define the source quad for homography — they trace the
    actual top and bottom staff lines as straight lines.
    """
    def fit_line_at(curve: List[Tuple[int, int]], x_target: int) -> int:
        if len(curve) < 2:
            return curve[0][1] if curve else 0
        xs = np.array([p[0] for p in curve], dtype=float)
        ys = np.array([p[1] for p in curve], dtype=float)
        m, b = np.polyfit(xs, ys, 1)
        return int(round(m * x_target + b))

    left, right = 0, img_w - 1
    tl = (left,  fit_line_at(staff['top_curve'],    left))
    tr = (right, fit_line_at(staff['top_curve'],    right))
    br = (right, fit_line_at(staff['bottom_curve'], right))
    bl = (left,  fit_line_at(staff['bottom_curve'], left))
    return tl, tr, br, bl


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cutting
# ─────────────────────────────────────────────────────────────────────────────

def cut_staff(img_color: np.ndarray, staff: dict,
              padding_px: Optional[int] = None) -> Tuple[np.ndarray, int, Tuple]:
    """
    Crop a rectangular band around a staff.

    Returns
    -------
    cut          rectangular crop (full width, padded height)
    y_offset     y of cut top edge in original image
    cut_corners  (TL, TR, BR, BL) of staff lines in cut-local coordinates
    """
    img_h, img_w = img_color.shape[:2]
    tl, tr, br, bl = get_staff_quad(staff, img_w)

    if padding_px is None:
        # Pad by ~1 staff height on each side (room for ledger lines, fermatas)
        padding_px = int(staff['line_spacing'] * 4)

    ys = [tl[1], tr[1], br[1], bl[1]]
    y_min = max(0, min(ys) - padding_px)
    y_max = min(img_h, max(ys) + padding_px)

    cut = img_color[y_min:y_max, :].copy()
    cut_corners = tuple((p[0], p[1] - y_min) for p in (tl, tr, br, bl))
    return cut, y_min, cut_corners


# ─────────────────────────────────────────────────────────────────────────────
# 4. Homography rectification
# ─────────────────────────────────────────────────────────────────────────────

def rectify_cut(cut: np.ndarray,
                cut_corners: Tuple,
                target_width: int,
                target_line_spacing: int = TARGET_LINE_SPACING,
                vertical_pad: int = VERTICAL_PAD) -> np.ndarray:
    """
    4-point perspective warp turning the source quad into a clean rectangle.

    Output: target_width × (target_staff_h + 2*vertical_pad).
    """
    tl, tr, br, bl = cut_corners
    src = np.float32([tl, tr, br, bl])

    target_staff_h = target_line_spacing * 4
    dst_h = target_staff_h + 2 * vertical_pad

    dst = np.float32([
        (0,                vertical_pad),
        (target_width - 1, vertical_pad),
        (target_width - 1, vertical_pad + target_staff_h),
        (0,                vertical_pad + target_staff_h),
    ])

    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(cut, H, (target_width, dst_h),
                                  borderValue=(255, 255, 255))
    return warped


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stitching
# ─────────────────────────────────────────────────────────────────────────────

def stitch_staves(rectified: List[np.ndarray],
                  gap: int = STAVES_GAP,
                  bg=(255, 255, 255)) -> Optional[np.ndarray]:
    if not rectified:
        return None
    w = rectified[0].shape[1]
    total_h = sum(r.shape[0] for r in rectified) + gap * (len(rectified) - 1)
    canvas = np.full((total_h, w, 3), bg, dtype=np.uint8)
    y = 0
    for r in rectified:
        h = r.shape[0]
        canvas[y:y+h, :] = r
        y += h + gap
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_detection(img_color: np.ndarray, staves: List[dict]) -> np.ndarray:
    vis = img_color.copy()
    for i, staff in enumerate(staves):
        color = COLORS[i % len(COLORS)]
        # Draw each detected line curve
        for tr in staff['tracks']:
            pts = np.array(tr, dtype=np.int32)
            cv2.polylines(vis, [pts], False, color, 2)
        # Draw fitted bounding quad (extrapolated to image edges)
        tl, tr, br, bl = get_staff_quad(staff, img_color.shape[1])
        quad = np.array([tl, tr, br, bl], dtype=np.int32)
        cv2.polylines(vis, [quad], True, color, 3)
        cv2.putText(vis, f'Staff {i+1}', (tl[0] + 20, tl[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis


def draw_cut_with_quad(cut: np.ndarray, cut_corners: Tuple,
                       label: str = '') -> np.ndarray:
    vis = cut.copy()
    quad = np.array(cut_corners, dtype=np.int32)
    cv2.polylines(vis, [quad], True, (0, 0, 255), 2)
    if label:
        cv2.putText(vis, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def make_comparison(cut: np.ndarray, rectified: np.ndarray,
                    cut_corners: Tuple, label: str = '') -> np.ndarray:
    """Side-by-side cut (with quad) and rectified result."""
    left = draw_cut_with_quad(cut, cut_corners, f'{label}  source')

    # Match heights
    h = max(left.shape[0], rectified.shape[0])
    def pad_to_h(img):
        if img.shape[0] == h:
            return img
        pad = np.full((h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
        return np.vstack([img, pad])
    left  = pad_to_h(left)
    right = pad_to_h(rectified)

    # 8-px white separator
    sep = np.full((h, 8, 3), 200, dtype=np.uint8)
    return np.hstack([left, sep, right])


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path: str,
                  output_dir: str,
                  target_width: Optional[int] = None,
                  target_line_spacing: int = TARGET_LINE_SPACING,
                  vertical_pad: int = VERTICAL_PAD,
                  gap: int = STAVES_GAP,
                  visualize: bool = True) -> Optional[np.ndarray]:
    """
    Run the full curve-detect → cut → rectify → stitch pipeline.

    With visualize=True, every step writes an intermediate file into
    output_dir for inspection.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape
    if target_width is None:
        target_width = img_w

    # ── Step 1: detect curves ──
    staves = detect_staff_curves(img_gray)
    print(f'[1/4] Detected {len(staves)} staves')
    if not staves:
        print('     No staves detected — aborting.')
        return None

    if visualize:
        vis1 = draw_detection(img_color, staves)
        cv2.imwrite(str(out / '01_detected_staves.png'), vis1)
        print(f'      → {out / "01_detected_staves.png"}')

    # ── Step 2: cut each staff ──
    cuts = []
    for i, staff in enumerate(staves):
        cut, y_off, cut_corners = cut_staff(img_color, staff)
        cuts.append((cut, cut_corners))
        if visualize:
            vis2 = draw_cut_with_quad(cut, cut_corners, f'Staff {i+1}')
            cv2.imwrite(str(out / f'02_cut_{i+1:02d}.png'), vis2)
    print(f'[2/4] Saved {len(cuts)} cuts')

    # ── Step 3: rectify each cut ──
    rectified = []
    for i, (cut, cut_corners) in enumerate(cuts):
        rect = rectify_cut(cut, cut_corners, target_width,
                           target_line_spacing, vertical_pad)
        rectified.append(rect)
        if visualize:
            cv2.imwrite(str(out / f'03_rectified_{i+1:02d}.png'), rect)
            cmp = make_comparison(cut, rect, cut_corners, f'Staff {i+1}')
            cv2.imwrite(str(out / f'03_compare_{i+1:02d}.png'), cmp)
    print(f'[3/4] Rectified {len(rectified)} staves')

    # ── Step 4: stitch ──
    final = stitch_staves(rectified, gap=gap)
    if final is None:
        return None
    out_path = out / '04_final_stitched.png'
    cv2.imwrite(str(out_path), final)
    print(f'[4/4] Final image: {final.shape[1]}×{final.shape[0]}  →  {out_path}')

    return final


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        # Default test paths matching the existing pipeline
        test_img = r'S:\mmdetection\data\my_images\img_1.png'
        out_dir  = r'S:\omr\rectify_test'
    else:
        test_img = sys.argv[1]
        out_dir  = sys.argv[2] if len(sys.argv) > 2 else 'rectify_output'

    print(f'Input : {test_img}')
    print(f'Output: {out_dir}\n')
    process_image(test_img, out_dir)
