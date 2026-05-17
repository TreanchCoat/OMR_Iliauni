"""
symbol_detector.py — Stage 3 of the OMR pipeline.

Runs the YOLOv8 model on each cleaned staff crop produced by preprocessing.py
and returns all detections with coordinates expressed in BOTH:
    cx / cy / x1..y2          — crop-relative (within the staff bracket)
    full_cx / full_cy         — absolute, in the full rectified image

The `full_*` coordinates are what XML construction and frontend overlays use,
because they reference the rectified page that the user can actually see.

Pipeline position
-----------------
    image
      → staff_rectifier.process_image()     → rectified image
      → preprocessing.preprocess_image()    → ProcessedScore
      → symbol_detector.detect_page()       → PageDetections   ◀── this file
      → score_to_xml.build_score_xml()      → MusicXML

Output schema (JSON form)
-------------------------
    [
      {
        "part_id":         "P1",
        "staff_in_part":   0,
        "top_y":           307,
        "bot_y":           379,
        "line_positions":  [307, 324, 342, 361, 379],
        "line_spacing":    18.0,
        "crop_y1":         239,
        "total_detections": 26,
        "detections": [
          {
            "class_name": "clefG",
            "conf":       0.94,
            "cx": 280, "cy": 102,
            "x1": 257, "y1": 40, "x2": 304, "y2": 165,
            "full_cx": 280, "full_cy": 341
          },
          ...
        ]
      },
      ...
    ]
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
class Detection:
    """One YOLO detection for a single symbol."""
    class_name: str
    conf:       float

    # Crop-relative coordinates (within this staff's cleaned image)
    cx: int
    cy: int
    x1: int
    y1: int
    x2: int
    y2: int

    # Absolute coordinates in the full rectified page
    full_cx: int
    full_cy: int


@dataclass
class StaffDetections:
    """All detections for one staff, plus the staff geometry needed for XML."""
    part_id:        str
    staff_in_part:  int
    top_y:          int
    bot_y:          int
    line_positions: List[int]
    line_spacing:   float
    crop_y1:        int
    crop_x1:        int = 0    # horizontal crop offset (0 if crop is full width)
    detections:     List[Detection] = field(default_factory=list)

    @property
    def total_detections(self) -> int:
        return len(self.detections)


@dataclass
class PageDetections:
    """Detections for an entire rectified page."""
    image_path:  str
    img_h:       int
    img_w:       int
    num_parts:   int
    parts:       List[List[StaffDetections]] = field(default_factory=list)
    all_staves:  List[StaffDetections]       = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

# YOLO classes that the pipeline has a dedicated detector for (or that
# don't carry useful information at this stage).  Filtered out at the
# detect_page level so they never reach the JSON, the XML, or any of
# the visualisations.
IGNORED_YOLO_CLASSES = frozenset({
    # Staff detection is done by staff_detector_unet / score_analyzer;
    # the YOLO 'staff' / 'staffLine' classes only add noise.
    'staff', 'staffLine', 'staff-line',
})


def detect_page(processed_score,
                model_path: str,
                conf_threshold: float = 0.25,
                use_cleaned_image: bool = True,
                imgsz: int = 1280,
                ignored_classes: Optional[set] = None) -> PageDetections:
    """
    Run YOLO on every staff in a ProcessedScore.

    Parameters
    ----------
    processed_score    Output of preprocessing.preprocess_image()
    model_path         Path to YOLO .pt weights (e.g. deepscores_crops_v1.pt)
    conf_threshold     Min confidence to keep a detection (default 0.25)
    use_cleaned_image  If True, run on the staff-line-removed binary image.
                       If False, run on the original color crop.
    imgsz              Inference size for YOLO (default 1280 for thin symbols)
    ignored_classes    Class names to drop after inference.  Defaults to
                       ``IGNORED_YOLO_CLASSES`` (staff / staffLine).

    Returns
    -------
    PageDetections with all staves populated.
    """
    # Imported lazily so the rest of the pipeline still works without ultralytics
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_names = model.names   # dict[int, str]

    if ignored_classes is None:
        ignored_classes = IGNORED_YOLO_CLASSES

    num_parts = processed_score.num_parts
    parts: List[List[StaffDetections]] = [[] for _ in range(num_parts)]
    all_staves: List[StaffDetections] = []
    n_filtered = 0

    for part_idx, part_staves in enumerate(processed_score.parts):
        for pstaff in part_staves:
            img = pstaff.cleaned if use_cleaned_image else pstaff.crop

            # Ensure 3-channel input — YOLO expects BGR
            if img.ndim == 2:
                infer_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                infer_img = img

            results = model(infer_img,
                            conf=conf_threshold,
                            imgsz=imgsz,
                            verbose=False)[0]

            detections: List[Detection] = []
            if results.boxes is not None:
                xyxy   = results.boxes.xyxy.cpu().numpy()
                confs  = results.boxes.conf.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                    cls_name = class_names[int(cls_id)]
                    if cls_name in ignored_classes:
                        n_filtered += 1
                        continue
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append(Detection(
                        class_name = cls_name,
                        conf       = round(float(conf), 4),
                        cx=cx, cy=cy,
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        # Coordinates relative to the full rectified image:
                        # both x and y shift by the crop's offset.  crop_x1
                        # was always 0 in the old full-width-crop world, but
                        # is now non-zero for partial-width staves.
                        full_cx = cx + pstaff.crop_x1,
                        full_cy = cy + pstaff.crop_y1,
                    ))

            sd = StaffDetections(
                part_id        = pstaff.part_id,
                staff_in_part  = pstaff.staff_in_part,
                top_y          = pstaff.top_y,
                bot_y          = pstaff.bot_y,
                line_positions = list(pstaff.line_positions),
                line_spacing   = float(pstaff.line_spacing),
                crop_y1        = int(pstaff.crop_y1),
                crop_x1        = int(getattr(pstaff, 'crop_x1', 0)),
                detections     = detections,
            )

            parts[part_idx].append(sd)
            all_staves.append(sd)

    if n_filtered:
        print(f'      filtered {n_filtered} detection(s) of ignored '
              f'classes ({", ".join(sorted(ignored_classes))})')

    return PageDetections(
        image_path = processed_score.image_path,
        img_h      = processed_score.img_h,
        img_w      = processed_score.img_w,
        num_parts  = num_parts,
        parts      = parts,
        all_staves = all_staves,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

def assign_detection_ids(page: 'PageDetections') -> dict:
    """
    Assign a stable ``det_NNNN`` ID to every detection on the page.

    The iteration order matches ``build_musicxml_v2.assign_detection_ids``
    (all_staves, then per-staff detections), so callers that compute the
    id_map here can pass it straight into the MusicXML builder and the
    same ID will appear in both files.

    Returns a dict keyed by ``id()`` of each Detection object.  Use
    ``id_map[id(d)]`` to look up a detection's string ID.
    """
    id_map: dict = {}
    counter = 1
    for sd in page.all_staves:
        for d in sd.detections:
            id_map[id(d)] = f'det_{counter:04d}'
            counter += 1
    return id_map


def to_json_list(page: 'PageDetections',
                 id_map: Optional[dict] = None) -> list:
    """
    Build the canonical detections.json payload.

    Shape: top-level **list of staves**, each staff carries its own
    detections under a ``detections`` key.  The per-detection record is
    the SAME schema as the ``omr-coordinates`` field embedded in the
    MusicXML, so the two files cross-reference by ``id``::

        [
          {
            "part_id": "P1",
            "staff_in_part": 0,
            "top_y": 100,
            "bot_y": 180,
            "line_positions": [...],
            "line_spacing": 18.0,
            "crop_y1": 0,
            "crop_x1": 0,
            "total_detections": 36,
            "detections": [
              {"id": "det_0001", "class": "clefG", "conf": 0.95,
               "part_id": "P1", "staff_in_part": 0,
               "cx": 280, "cy": 341,
               "x1": 257, "y1": 280, "x2": 304, "y2": 405},
              ...
            ]
          },
          ...
        ]

    If ``id_map`` is None, IDs are generated by
    ``assign_detection_ids``.  Pass an existing map (e.g. the one
    produced for the XML) to keep IDs perfectly aligned.
    """
    if id_map is None:
        id_map = assign_detection_ids(page)

    out: list = []
    for sd in page.all_staves:
        crop_y1 = sd.crop_y1
        dets: list = []
        for d in sd.detections:
            dets.append({
                'id':            id_map[id(d)],
                'class':         d.class_name,
                'conf':          round(float(d.conf), 4),
                'part_id':       sd.part_id,
                'staff_in_part': sd.staff_in_part,
                'cx':            int(d.full_cx),
                'cy':            int(d.full_cy),
                'x1':            int(d.x1),
                'y1':            int(d.y1) + crop_y1,
                'x2':            int(d.x2),
                'y2':            int(d.y2) + crop_y1,
            })
        out.append({
            'part_id':          sd.part_id,
            'staff_in_part':    sd.staff_in_part,
            'top_y':            sd.top_y,
            'bot_y':            sd.bot_y,
            'line_positions':   sd.line_positions,
            'line_spacing':     sd.line_spacing,
            'crop_y1':          sd.crop_y1,
            'crop_x1':          sd.crop_x1,
            'total_detections': sd.total_detections,
            'detections':       dets,
        })
    return out


def save_json(page: PageDetections, output_path: str,
              id_map: Optional[dict] = None) -> dict:
    """
    Write a PageDetections to disk as JSON, using the per-staff schema
    (see ``to_json_list``).

    Returns the id_map that was used so the caller can pass the same
    one into ``build_musicxml_v2.build_score_xml_v2`` for matching IDs.
    """
    if id_map is None:
        id_map = assign_detection_ids(page)
    payload = to_json_list(page, id_map=id_map)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'Detections JSON -> {output_path}')
    return id_map


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

# Colors are BGR (OpenCV convention).
_CLASS_COLORS = {
    # Clefs — orange family
    'clefG':              (0, 100, 255),
    'clefF':              (0, 100, 255),
    'clefCAlto':          (0, 100, 255),
    'clefCTenor':         (0, 100, 255),
    'clef8':              (0, 200, 255),
    # Noteheads — green family
    'noteheadBlack':      (0, 200,   0),
    'noteheadHalf':       (0, 255, 100),
    'noteheadWhole':      (0, 255, 200),
    'noteheadDoubleWhole':(0, 255, 200),
    # Key signature — blue
    'keyFlat':            (255, 100,   0),
    'keySharp':           (255, 100,   0),
    'keyNatural':         (255, 100,   0),
    # Note accidentals — light blue
    'accidentalSharp':    (255, 200,   0),
    'accidentalFlat':     (255, 200,   0),
    'accidentalNatural':  (255, 200,   0),
    'accidentalDoubleSharp': (255, 200, 0),
    'accidentalDoubleFlat':  (255, 200, 0),
    # Rests — magenta
    'rest':               (200,   0, 200),
    'restWhole':          (200,   0, 200),
    'restHalf':           (200,   0, 200),
    'restQuarter':        (200,   0, 200),
    'restEighth':         (200,   0, 200),
    'rest8th':            (200,   0, 200),
    'rest16th':           (200,   0, 200),
    # Connectives — yellow / cyan
    'slur':               (0,   255, 255),
    'tie':                (0,   255, 255),
    'beam':               (255, 255,   0),
    'flag8thUp':          (255, 255, 100),
    'flag8thDown':        (255, 255, 100),
    'flag16thUp':         (255, 255, 100),
    'flag16thDown':       (255, 255, 100),
    'flag32ndUp':         (255, 255, 100),
    'flag32ndDown':       (255, 255, 100),
    # Other
    'fermataAbove':       (200, 200, 200),
    'fermataBelow':       (200, 200, 200),
    'staff':              (180, 180, 180),
    'barline':            (50,   50, 200),
    'stem':               (180,  50, 180),
}


def _color_for_class(class_name: str) -> tuple:
    """Return a deterministic BGR color for a YOLO class name."""
    if class_name in _CLASS_COLORS:
        return _CLASS_COLORS[class_name]
    # Hash-based fallback for unknown classes (deterministic across runs)
    import hashlib
    h = hashlib.md5(class_name.encode()).digest()
    return (50 + h[0] % 200, 50 + h[1] % 200, 50 + h[2] % 200)


def _draw_boxes_on(base: np.ndarray, detections,
                   show_confidence: bool, header: str,
                   show_labels: bool = False) -> np.ndarray:
    """
    Draw bounding boxes on a single image with a class-colour legend
    underneath.

    By default ``show_labels`` is False — per-detection text labels are
    skipped because they overlap badly when many detections share a
    region.  The legend strip at the bottom maps each colour to its
    class name so the boxes are still readable at a glance.
    """
    if base.ndim == 2:
        vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        vis = base.copy()

    # ── Boxes ──
    box_thickness = 1
    for det in detections:
        color = _color_for_class(det.class_name)
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2),
                      color, box_thickness)

        if show_labels:
            label = (f'{det.class_name} {det.conf:.2f}'
                     if show_confidence else det.class_name)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            ty_top = max(th + 3, det.y1)
            cv2.rectangle(vis,
                          (det.x1, ty_top - th - 3),
                          (det.x1 + tw + 3, ty_top),
                          color, -1)
            cv2.putText(vis, label,
                        (det.x1 + 2, ty_top - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # ── Header bar ──
    cv2.putText(vis, header, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA)

    # ── Legend strip below the image ──
    unique_classes = sorted({d.class_name for d in detections})
    if unique_classes:
        vis = _append_class_legend(vis, unique_classes)

    return vis


def _append_class_legend(img: np.ndarray, classes) -> np.ndarray:
    """
    Append a class→colour legend strip beneath ``img`` and return the
    combined image.  Wraps to multiple rows when the page is narrow.
    """
    cell_h = 22
    swatch_w = 18
    swatch_h = 14
    text_pad = 4
    row_pad = 4
    side_pad = 8

    # Figure out how many cells fit per row.
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.45
    cell_widths = []
    for name in classes:
        (tw, _th), _ = cv2.getTextSize(name, font, fscale, 1)
        cell_widths.append(swatch_w + text_pad + tw + side_pad)

    avail_w = max(200, img.shape[1] - 2 * side_pad)
    # Greedy wrap
    rows = [[]]
    row_w = 0
    for name, cw in zip(classes, cell_widths):
        if row_w + cw > avail_w and rows[-1]:
            rows.append([])
            row_w = 0
        rows[-1].append((name, cw))
        row_w += cw

    legend_h = row_pad + len(rows) * cell_h + row_pad
    legend = np.full((legend_h, img.shape[1], 3), 245, dtype=np.uint8)
    y = row_pad + cell_h - 6
    for row in rows:
        x = side_pad
        for name, cw in row:
            color = _color_for_class(name)
            cv2.rectangle(legend,
                          (x, y - swatch_h + 2),
                          (x + swatch_w, y + 2),
                          color, -1)
            cv2.rectangle(legend,
                          (x, y - swatch_h + 2),
                          (x + swatch_w, y + 2),
                          (0, 0, 0), 1)
            cv2.putText(legend, name,
                        (x + swatch_w + text_pad, y),
                        font, fscale, (0, 0, 0), 1, cv2.LINE_AA)
            x += cw
        y += cell_h

    return np.vstack([img, legend])


def visualize_detections(processed_score, page_detections,
                         output_dir: str,
                         mode: str = 'side_by_side',
                         show_confidence: bool = True,
                         show_labels: bool = False):
    """
    Save one annotated PNG per staff with bounding boxes and class labels.

    Parameters
    ----------
    processed_score   ProcessedScore (provides the staff crops to draw on)
    page_detections   PageDetections (provides the bboxes)
    output_dir        Directory to write *_labeled.png files into
    mode              How to render each staff:
                        'color'        — only the original color crop
                        'cleaned'      — only the staff-line-removed image
                        'side_by_side' — color on top, cleaned below (default)
                                          A black separator strip lets you
                                          confirm staff removal worked while
                                          keeping the readable color version.
    show_confidence   Whether to append "0.94" style confidence to each label.

    File naming
    -----------
        <part_id>_staff<NN>_labeled.png   e.g. P1_staff03_labeled.png
    """
    if mode not in ('color', 'cleaned', 'side_by_side'):
        raise ValueError(f'Unknown mode: {mode!r}')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for part_staves, staff_dets_list in zip(processed_score.parts,
                                              page_detections.parts):
        for pstaff, sdet in zip(part_staves, staff_dets_list):
            header_color = (f'{sdet.part_id}  staff {sdet.staff_in_part + 1}  '
                            f'({sdet.total_detections} detections)  — original')
            header_clean = (f'{sdet.part_id}  staff {sdet.staff_in_part + 1}  '
                            f'({sdet.total_detections} detections)  — cleaned')

            if mode == 'color':
                vis = _draw_boxes_on(pstaff.crop, sdet.detections,
                                      show_confidence, header_color,
                                      show_labels=show_labels)
            elif mode == 'cleaned':
                vis = _draw_boxes_on(pstaff.cleaned, sdet.detections,
                                      show_confidence, header_clean,
                                      show_labels=show_labels)
            else:   # side_by_side
                top = _draw_boxes_on(pstaff.crop, sdet.detections,
                                      show_confidence, header_color,
                                      show_labels=show_labels)
                bot = _draw_boxes_on(pstaff.cleaned, sdet.detections,
                                      show_confidence, header_clean,
                                      show_labels=show_labels)
                # Match widths just in case (they should already match)
                if top.shape[1] != bot.shape[1]:
                    w = max(top.shape[1], bot.shape[1])
                    def pad_w(img, target_w):
                        if img.shape[1] == target_w:
                            return img
                        pad = np.full((img.shape[0], target_w - img.shape[1], 3),
                                      255, dtype=np.uint8)
                        return np.hstack([img, pad])
                    top = pad_w(top, w)
                    bot = pad_w(bot, w)
                # 6-px black separator
                sep = np.full((6, top.shape[1], 3), 0, dtype=np.uint8)
                vis = np.vstack([top, sep, bot])

            fname = f'{sdet.part_id}_staff{sdet.staff_in_part+1:02d}_labeled.png'
            cv2.imwrite(str(out / fname), vis)

    print(f'Labeled crops -> {output_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append(r'S:\omr')
    from preprocessing import preprocess_image

    test_img   = r'S:\mmdetection\data\my_images\img_1.png'
    model_path = r'S:\omr\models\deepscores_crops_v1.pt'
    out_json   = r'S:\omr\detections.json'

    processed  = preprocess_image(test_img)
    detections = detect_page(processed, model_path)
    save_json(detections, out_json)

    total = sum(sd.total_detections for sd in detections.all_staves)
    print(f'\n{total} detections across {len(detections.all_staves)} staves')
