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

def detect_page(processed_score,
                model_path: str,
                conf_threshold: float = 0.25,
                use_cleaned_image: bool = True,
                imgsz: int = 1280) -> PageDetections:
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

    Returns
    -------
    PageDetections with all staves populated.
    """
    # Imported lazily so the rest of the pipeline still works without ultralytics
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_names = model.names   # dict[int, str]

    num_parts = processed_score.num_parts
    parts: List[List[StaffDetections]] = [[] for _ in range(num_parts)]
    all_staves: List[StaffDetections] = []

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
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append(Detection(
                        class_name = class_names[int(cls_id)],
                        conf       = round(float(conf), 4),
                        cx=cx, cy=cy,
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        # Coordinates relative to the full rectified image:
                        # crop spans full width, so x doesn't shift; y shifts
                        # by the crop's vertical offset.
                        full_cx = cx,
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
                detections     = detections,
            )

            parts[part_idx].append(sd)
            all_staves.append(sd)

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

def to_json_list(page: PageDetections) -> list:
    """
    Convert a PageDetections object to a flat JSON-serializable list of staff
    dicts (matches the schema of P1_staff01.json).
    """
    out = []
    for sd in page.all_staves:
        out.append({
            'part_id':          sd.part_id,
            'staff_in_part':    sd.staff_in_part,
            'top_y':            sd.top_y,
            'bot_y':            sd.bot_y,
            'line_positions':   sd.line_positions,
            'line_spacing':     sd.line_spacing,
            'crop_y1':          sd.crop_y1,
            'total_detections': sd.total_detections,
            'detections':       [asdict(d) for d in sd.detections],
        })
    return out


def save_json(page: PageDetections, output_path: str):
    """Write a PageDetections to disk as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(to_json_list(page), f, indent=2)
    print(f'Detections JSON → {output_path}')


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
