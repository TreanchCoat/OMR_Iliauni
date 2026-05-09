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

def _binarize_otsu(img_gray: np.ndarray) -> np.ndarray:
    """Otsu thresholding — best for clean printed scores with even contrast."""
    _, b = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def _binarize_adaptive(img_gray: np.ndarray, block: int = 31, C: int = 10) -> np.ndarray:
    """
    Adaptive (mean) thresholding — handles uneven lighting, faded ink, and
    paper texture much better than Otsu.  Best default for handwritten and
    photographed scores.

    block — neighbourhood size (must be odd); larger = smoother local mean
    C     — constant subtracted from mean; larger = more aggressive thresholding
    """
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block, C,
    )


def _binarize_combined(img_gray: np.ndarray) -> np.ndarray:
    """
    Union of Otsu + adaptive — rescues lines that one method misses.
    Slightly noisier but the strip-tracker handles noise well via min_track_pts.
    """
    a = _binarize_otsu(img_gray)
    b = _binarize_adaptive(img_gray)
    return cv2.bitwise_or(a, b)


# Available strategies for multi-pass detection.  Each entry is a complete
# parameter set; the detector tries them in order from strict to permissive
# and keeps the result that finds the most staves.
DETECTION_STRATEGIES = [
    # 1. Default — printed scores
    {
        'name':              'printed',
        'binarizer':         _binarize_otsu,
        'n_strips':          24,
        'kernel_width_ratio': 0.04,
        'peak_height_ratio': 0.5,
        'min_track_frac':    0.25,
        'spacing_tol':       0.30,
        'y_match_factor':    0.4,
    },
    # 2. Adaptive threshold — uneven scans
    {
        'name':              'uneven_scan',
        'binarizer':         _binarize_adaptive,
        'n_strips':          24,
        'kernel_width_ratio': 0.04,
        'peak_height_ratio': 0.4,
        'min_track_frac':    0.25,
        'spacing_tol':       0.30,
        'y_match_factor':    0.4,
    },
    # 3. Handwritten — finer strips, looser tolerances
    {
        'name':              'handwritten',
        'binarizer':         _binarize_combined,
        'n_strips':          40,
        'kernel_width_ratio': 0.025,
        'peak_height_ratio': 0.30,
        'min_track_frac':    0.18,
        'spacing_tol':       0.45,
        'y_match_factor':    0.6,
    },
    # 4. Last resort — heavy curvature, faint lines, very irregular
    {
        'name':              'permissive',
        'binarizer':         _binarize_combined,
        'n_strips':          60,
        'kernel_width_ratio': 0.018,
        'peak_height_ratio': 0.22,
        'min_track_frac':    0.12,
        'spacing_tol':       0.55,
        'y_match_factor':    0.75,
    },
]


def _detect_with_params(img_gray: np.ndarray, params: dict) -> List[dict]:
    """
    Single-pass curve detection with one parameter set.  This is the same
    algorithm as before but parameterised; the multi-strategy wrapper below
    tries several configurations.
    """
    img_h, img_w = img_gray.shape

    binary = params['binarizer'](img_gray)

    n_strips           = params['n_strips']
    kernel_width_ratio = params['kernel_width_ratio']
    peak_height_ratio  = params['peak_height_ratio']
    min_track_frac     = params['min_track_frac']
    spacing_tol        = params['spacing_tol']
    y_match_factor     = params['y_match_factor']

    # Horizontal morph — narrow enough to follow curves
    kernel_w = max(20, int(img_w * kernel_width_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Slice into vertical strips, find peak rows in each
    strip_w = max(1, img_w // n_strips)
    strip_data: List[Tuple[int, List[int]]] = []
    for i in range(n_strips):
        x1 = i * strip_w
        x2 = min(x1 + strip_w, img_w)
        if x2 <= x1:
            continue
        strip = horizontal[:, x1:x2]
        row_sum = strip.sum(axis=1)
        if row_sum.max() == 0:
            continue
        peaks, _ = find_peaks(row_sum,
                              height=row_sum.max() * peak_height_ratio,
                              distance=PEAK_DISTANCE)
        if len(peaks) == 0:
            continue
        strip_data.append(((x1 + x2) // 2, sorted(peaks.tolist())))

    if not strip_data:
        return []

    # Estimate typical staff-line spacing from the densest strip
    sample = sorted(strip_data, key=lambda s: -len(s[1]))[0][1]
    if len(sample) >= 2:
        gaps = np.diff(sample)
        small_gaps = gaps[gaps <= np.median(gaps) * 1.5]
        est_spacing = float(np.median(small_gaps)) if len(small_gaps) else float(np.median(gaps))
    else:
        est_spacing = 18.0
    y_match_tol = max(3.0, est_spacing * y_match_factor)

    # Walk strips left→right, extending tracks by best y-match
    tracks: List[List[Tuple[int, int]]] = []
    for x_center, ys in strip_data:
        for y in ys:
            best_track = None
            best_dist  = y_match_tol
            for tr in tracks:
                last_x, last_y = tr[-1]
                # Bridge gaps up to 4 strip widths (helps when a strip's
                # peak detection failed in the middle of a curve).
                if x_center - last_x > strip_w * 4:
                    continue
                d = abs(y - last_y)
                if d < best_dist:
                    best_dist  = d
                    best_track = tr
            if best_track is not None:
                best_track.append((x_center, y))
            else:
                tracks.append([(x_center, y)])

    # Drop short tracks
    min_track_pts = max(3, int(n_strips * min_track_frac))
    tracks = [t for t in tracks if len(t) >= min_track_pts]

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
        if all(abs(g - median_gap) / median_gap < spacing_tol for g in gaps):
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


def _score_detection(staves: List[dict]) -> Tuple[int, float]:
    """
    Score a detection result for ranking across strategies.
    Returns (n_staves, consistency_score).  Higher is better.

    consistency_score rewards detections where all staves have similar
    line_spacing (real staves on a page should — different spacings often
    mean spurious detections).
    """
    n = len(staves)
    if n == 0:
        return (0, 0.0)
    spacings = np.array([s['line_spacing'] for s in staves])
    if n < 2:
        return (n, 1.0)
    cv_inv = 1.0 / (1.0 + np.std(spacings) / max(np.mean(spacings), 1e-3))
    return (n, cv_inv)


def detect_staff_curves(img_gray: np.ndarray,
                        verbose: bool = False,
                        force_strategy: Optional[str] = None,
                        use_unet: bool = True,
                        use_yolo: bool = True,
                        yolo_conf: float = 0.25,
                        unet_threshold: float = 0.5) -> List[dict]:
    """
    Multi-strategy staff line detection.

    Detector priority (best-first):
        1. U-Net (semantic segmentation) — most robust on handwritten input
        2. OLA YOLOv8 — bbox-based, trained on MUSCIMA++
        3. Classical multi-strategy (morphology + curve tracing)

    The first detector that's available AND produces a plausible result
    wins.  All three are gracefully skipped if their model file or
    Python dependency isn't installed.

    Parameters
    ----------
    img_gray        grayscale ndarray of the page
    verbose         if True, print per-strategy results
    force_strategy  if given, only run that one strategy.  Valid values:
                    'unet', 'yolo', 'printed', 'uneven_scan',
                    'handwritten', 'permissive'.
    use_unet        if True, try U-Net before YOLO and classical
    use_yolo        if True, try YOLO if U-Net is unavailable / weak
    yolo_conf       confidence threshold for YOLO detections
    unet_threshold  probability threshold for U-Net mask binarisation
    """
    # ── Forced strategy mode ──
    if force_strategy is not None:
        if force_strategy == 'unet':
            try:
                from staff_detector_unet import detect_staves_unet
                return detect_staves_unet(img_gray, threshold=unet_threshold,
                                            verbose=verbose)
            except (ImportError, FileNotFoundError) as e:
                raise RuntimeError(f'Cannot use U-Net strategy: {e}')
        if force_strategy == 'yolo':
            try:
                from staff_detector_yolo import detect_staves_yolo
                return detect_staves_yolo(img_gray, conf_threshold=yolo_conf,
                                            verbose=verbose)
            except (ImportError, FileNotFoundError) as e:
                raise RuntimeError(f'Cannot use YOLO strategy: {e}')
        params = next((s for s in DETECTION_STRATEGIES
                       if s['name'] == force_strategy), None)
        if params is None:
            raise ValueError(f'Unknown strategy: {force_strategy}')
        return _detect_with_params(img_gray, params)

    # ── 1. U-Net (best for handwritten) ──
    if use_unet:
        try:
            from staff_detector_unet import detect_staves_unet, is_unet_available
            if is_unet_available():
                unet_staves = detect_staves_unet(img_gray,
                                                   threshold=unet_threshold,
                                                   verbose=verbose)
                unet_score = _score_detection(unet_staves)
                if verbose:
                    print(f"      strategy {'unet':14s}: "
                          f"{len(unet_staves)} staves, "
                          f"consistency={unet_score[1]:.2f}")
                if unet_score[0] >= 1 and unet_score[1] > 0.5:
                    if verbose:
                        print(f"      → using U-Net with "
                              f"{len(unet_staves)} staves")
                    return unet_staves
                elif verbose:
                    print('        (U-Net output too weak, falling back)')
            elif verbose:
                print('      (U-Net unavailable — trying YOLO)')
        except ImportError as e:
            if verbose:
                print(f'      (U-Net import failed: {e})')

    # ── 2. YOLO (good for printed) ──
    if use_yolo:
        try:
            from staff_detector_yolo import detect_staves_yolo, is_yolo_available
            if is_yolo_available():
                yolo_staves = detect_staves_yolo(img_gray, conf_threshold=yolo_conf,
                                                   verbose=verbose)
                yolo_score = _score_detection(yolo_staves)
                if verbose:
                    print(f"      strategy {'yolo':14s}: "
                          f"{len(yolo_staves)} staves, "
                          f"consistency={yolo_score[1]:.2f}")
                if yolo_score[0] >= 1 and yolo_score[1] > 0.5:
                    if verbose:
                        print(f"      → using YOLO with "
                              f"{len(yolo_staves)} staves")
                    return yolo_staves
                elif verbose:
                    print('        (YOLO output too weak, falling back)')
            elif verbose:
                print('      (YOLO unavailable — falling back to classical)')
        except ImportError as e:
            if verbose:
                print(f'      (YOLO import failed: {e})')

    # ── 3. Classical multi-strategy fallback ──
    best_staves: List[dict] = []
    best_score = (0, 0.0)
    best_name  = None

    for params in DETECTION_STRATEGIES:
        staves = _detect_with_params(img_gray, params)
        score = _score_detection(staves)
        if verbose:
            print(f"      strategy {params['name']:14s}: "
                  f"{len(staves)} staves, consistency={score[1]:.2f}")
        if score > best_score:
            best_score  = score
            best_staves = staves
            best_name   = params['name']
        # Early exit: if a strict strategy already finds many staves with
        # high consistency, no need to try the permissive ones (they tend
        # to add false positives).
        if score[0] >= 6 and score[1] > 0.85:
            break

    if verbose and best_name:
        print(f"      → using strategy '{best_name}' with {len(best_staves)} staves")

    return best_staves


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quadrilateral fitting
# ─────────────────────────────────────────────────────────────────────────────

def get_staff_quad(staff: dict, img_w: int,
                   margin_px: int = 0,
                   clamp_to_extent: bool = True) -> Tuple[Tuple[int, int], ...]:
    """
    Get 4 corner points (TL, TR, BR, BL) for a staff.

    The corner X positions are taken from the staff's actual horizontal
    extent (`left_x` / `right_x` in the staff dict), with an optional margin.
    The Y positions are taken from a linear fit on each curve evaluated at
    those X positions.

    Setting clamp_to_extent=False reverts to the old behaviour (extrapolate
    out to image edges) — useful for visualizations only.
    """
    def fit_line_at(curve: List[Tuple[int, int]], x_target: int) -> int:
        if len(curve) < 2:
            return curve[0][1] if curve else 0
        xs = np.array([p[0] for p in curve], dtype=float)
        ys = np.array([p[1] for p in curve], dtype=float)
        m, b = np.polyfit(xs, ys, 1)
        return int(round(m * x_target + b))

    if clamp_to_extent:
        left  = max(0,         staff['left_x']  - margin_px)
        right = min(img_w - 1, staff['right_x'] + margin_px)
    else:
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
              padding_px: Optional[int] = None,
              horizontal_margin_px: int = 30
              ) -> Tuple[np.ndarray, int, int, Tuple]:
    """
    Crop a band around a staff, sized to the staff's actual extent.

    horizontal_margin_px adds room on the left/right beyond the staff edges
    so that clefs/key signatures and final barlines aren't trimmed by the
    homography that follows.

    Returns
    -------
    cut          rectangular crop, sized to the staff
    y_offset     y of cut top edge in original image
    x_offset     x of cut left edge in original image
    cut_corners  (TL, TR, BR, BL) of staff lines in cut-local coordinates
    """
    img_h, img_w = img_color.shape[:2]
    tl, tr, br, bl = get_staff_quad(staff, img_w,
                                     margin_px=horizontal_margin_px,
                                     clamp_to_extent=True)

    if padding_px is None:
        # Pad by ~1 staff height on each side (room for ledger lines, fermatas)
        padding_px = int(staff['line_spacing'] * 4)

    ys = [tl[1], tr[1], br[1], bl[1]]
    xs = [tl[0], tr[0], br[0], bl[0]]
    y_min = max(0, min(ys) - padding_px)
    y_max = min(img_h, max(ys) + padding_px)
    x_min = max(0, min(xs))
    x_max = min(img_w, max(xs) + 1)

    cut = img_color[y_min:y_max, x_min:x_max].copy()
    cut_corners = tuple((p[0] - x_min, p[1] - y_min)
                         for p in (tl, tr, br, bl))
    return cut, y_min, x_min, cut_corners


# ─────────────────────────────────────────────────────────────────────────────
# 4. Homography rectification
# ─────────────────────────────────────────────────────────────────────────────

def rectify_cut(cut: np.ndarray,
                cut_corners: Tuple,
                target_width: Optional[int] = None,
                target_line_spacing: int = TARGET_LINE_SPACING,
                vertical_pad: int = VERTICAL_PAD,
                horizontal_pad: int = 30) -> np.ndarray:
    """
    4-point perspective warp turning the source quad into a clean rectangle.

    If `target_width` is None (recommended), it's derived from the source
    quad's actual width so a half-width staff stays half-width and isn't
    stretched.  Pass an explicit value only when you need to force all
    output staves to the same width (which used to be the default behaviour
    and caused the squishing artifact on partial-width staves).

    Output: (target_width + 2*horizontal_pad) × (target_staff_h + 2*vertical_pad).
    """
    tl, tr, br, bl = cut_corners

    if target_width is None:
        # Use mean of top and bottom edge lengths — robust to skew.
        top_w    = ((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2) ** 0.5
        bottom_w = ((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2) ** 0.5
        target_width = max(1, int(round((top_w + bottom_w) / 2)))

    src = np.float32([tl, tr, br, bl])
    target_staff_h = target_line_spacing * 4
    dst_w = target_width + 2 * horizontal_pad
    dst_h = target_staff_h + 2 * vertical_pad

    dst = np.float32([
        (horizontal_pad,                  vertical_pad),
        (horizontal_pad + target_width,   vertical_pad),
        (horizontal_pad + target_width,   vertical_pad + target_staff_h),
        (horizontal_pad,                  vertical_pad + target_staff_h),
    ])

    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(cut, H, (dst_w, dst_h),
                                  borderValue=(255, 255, 255))
    return warped


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stitching
# ─────────────────────────────────────────────────────────────────────────────

def stitch_staves(rectified: List[np.ndarray],
                  x_offsets: Optional[List[int]] = None,
                  canvas_width: Optional[int] = None,
                  gap: int = STAVES_GAP,
                  bg=(255, 255, 255)) -> Optional[np.ndarray]:
    """
    Stack rectified staves vertically into a single image.

    If x_offsets and canvas_width are provided, each staff is placed at its
    own x position on a fixed-width canvas — this preserves layout for pages
    with partial-width or side-by-side staves.

    If they're not provided, falls back to the old behaviour: align all
    staves to x=0, canvas width = max staff width.
    """
    if not rectified:
        return None

    if x_offsets is None or canvas_width is None:
        canvas_width = max(r.shape[1] for r in rectified)
        x_offsets    = [0] * len(rectified)

    total_h = sum(r.shape[0] for r in rectified) + gap * (len(rectified) - 1)
    canvas  = np.full((total_h, canvas_width, 3), bg, dtype=np.uint8)

    y = 0
    for r, xo in zip(rectified, x_offsets):
        h, w = r.shape[:2]
        # Clip to canvas in case of off-by-one rounding
        x_end = min(xo + w, canvas_width)
        w_eff = x_end - xo
        if w_eff > 0:
            canvas[y:y+h, xo:xo+w_eff] = r[:, :w_eff]
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
        # Draw the bounding quad — clamped to the staff's real extent so
        # half-width staves don't get drawn as full-page rectangles.
        tl, tr, br, bl = get_staff_quad(staff, img_color.shape[1],
                                          margin_px=0,
                                          clamp_to_extent=True)
        quad = np.array([tl, tr, br, bl], dtype=np.int32)
        cv2.polylines(vis, [quad], True, color, 3)
        cv2.putText(vis, f'Staff {i+1}', (tl[0] + 10, tl[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
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
                  visualize: bool = True,
                  verbose: bool = True,
                  preserve_layout: bool = True,
                  use_unet: bool = True,
                  use_yolo: bool = True,
                  yolo_conf: float = 0.25,
                  unet_threshold: float = 0.5,
                  force_strategy: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Run the full curve-detect → cut → rectify → stitch pipeline.

    With visualize=True, every step writes an intermediate file into
    output_dir for inspection.

    target_width      If None (recommended), each staff is rectified at its
                      natural width (no horizontal stretching of partial-
                      width staves).  Pass a fixed integer to force every
                      staff to that width — restores the legacy behaviour.
    preserve_layout   If True, each staff is placed on the stitched canvas
                      at its original x-offset, preserving the page's
                      horizontal layout (matters when the page has partial-
                      width or side-by-side staves).
    use_unet          If True (default), use the trained U-Net for staff
                      segmentation when its checkpoint is available.
                      Most robust for handwritten input.
    use_yolo          If True (default), fall back to the OLA YOLOv8 model
                      when U-Net isn't available.
    yolo_conf         Confidence threshold for YOLO staff detections.
    unet_threshold    Probability threshold for U-Net mask binarisation.
    verbose           If True, print which detection strategy was used.
    force_strategy    Override automatic strategy selection.  Valid values:
                      'unet', 'yolo', 'printed', 'uneven_scan',
                      'handwritten', 'permissive'.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_color = cv2.imread(str(image_path))
    if img_color is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape

    # ── Step 1: detect curves ──
    staves = detect_staff_curves(img_gray, verbose=verbose,
                                  force_strategy=force_strategy,
                                  use_unet=use_unet,
                                  use_yolo=use_yolo,
                                  yolo_conf=yolo_conf,
                                  unet_threshold=unet_threshold)
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
    cut_x_offsets: List[int] = []
    for i, staff in enumerate(staves):
        cut, y_off, x_off, cut_corners = cut_staff(img_color, staff)
        cuts.append((cut, cut_corners))
        cut_x_offsets.append(x_off)
        if visualize:
            vis2 = draw_cut_with_quad(cut, cut_corners, f'Staff {i+1}')
            cv2.imwrite(str(out / f'02_cut_{i+1:02d}.png'), vis2)
    print(f'[2/4] Saved {len(cuts)} cuts')

    # ── Step 3: rectify each cut ──
    rectified = []
    for i, (cut, cut_corners) in enumerate(cuts):
        rect = rectify_cut(cut, cut_corners,
                           target_width=target_width,
                           target_line_spacing=target_line_spacing,
                           vertical_pad=vertical_pad)
        rectified.append(rect)
        if visualize:
            cv2.imwrite(str(out / f'03_rectified_{i+1:02d}.png'), rect)
            cmp = make_comparison(cut, rect, cut_corners, f'Staff {i+1}')
            cv2.imwrite(str(out / f'03_compare_{i+1:02d}.png'), cmp)
    print(f'[3/4] Rectified {len(rectified)} staves')

    # ── Step 4: stitch ──
    if preserve_layout:
        # Place each rectified staff at its source x-offset on a full-width canvas.
        # The rectified output has its own internal horizontal padding, so we
        # subtract that to land the actual staff content at the right x.
        canvas_width = img_w
        # rectify_cut adds horizontal_pad on each side; align the LEFT edge
        # of staff content with the cut's source x_offset.
        x_offsets = [max(0, x_off - 30) for x_off in cut_x_offsets]
        final = stitch_staves(rectified,
                              x_offsets=x_offsets,
                              canvas_width=canvas_width,
                              gap=gap)
    else:
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
