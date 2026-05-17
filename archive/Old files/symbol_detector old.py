"""
symbol_detector.py

Runs YOLO detection on preprocessed staff crops.
Returns structured detection results organized by part and staff.

Called from:
    omr_pipeline.py (xml reconstruction)
    
Standalone:
    python symbol_detector.py
    -> saves visualized images with bounding boxes
"""

from __future__ import annotations

import sys
import os
sys.path.append(r'S:\omr')

import cv2
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from ultralytics import YOLO
from preprocessing import preprocess_image, ProcessedStaff, ProcessedScore

MODEL_PATH = r'S:\saved_models\deepscores_crops_v1.pt'
TEMP_IMG = r'S:\omr\temp_detection.png'

_model = None

def load_model(model_path: str = MODEL_PATH) -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model


@dataclass
class Detection:
    """Single symbol detection."""
    class_id: int
    class_name: str
    conf: float
    # Coordinates in CROP image pixels
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    # Coordinates in FULL IMAGE pixels (crop coords + offset)
    full_x1: int = 0
    full_y1: int = 0
    full_x2: int = 0
    full_y2: int = 0
    full_cx: int = 0
    full_cy: int = 0


@dataclass
class StaffDetections:
    """All detections for a single staff."""
    staff: ProcessedStaff
    detections: List[Detection] = field(default_factory=list)

    def get(self, class_name: str) -> List[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.class_name == class_name]

    def get_noteheads(self) -> List[Detection]:
        """Get all notehead detections sorted left to right."""
        noteheads = [d for d in self.detections
                     if d.class_name in ('noteheadBlack', 'noteheadHalf',
                                         'noteheadWhole', 'noteheadDoubleWhole')]
        return sorted(noteheads, key=lambda d: d.cx)

    def get_clef(self) -> Optional[Detection]:
        """Get the leftmost clef detection."""
        clefs = [d for d in self.detections
                 if d.class_name in ('clefG', 'clefF', 'clefCAlto', 'clefCTenor')]
        return min(clefs, key=lambda d: d.cx) if clefs else None


@dataclass
class PageDetections:
    """All detections for a full page, organized by part."""
    image_path: str
    # parts[i] = list of StaffDetections for part i
    parts: List[List[StaffDetections]] = field(default_factory=list)
    # flat list sorted top to bottom
    all_staves: List[StaffDetections] = field(default_factory=list)


def _detect_on_staff(staff: ProcessedStaff, conf: float = 0.2,
                     iou: float = 0.5) -> StaffDetections:
    """Run YOLO on a single cleaned staff crop."""
    model = load_model()

    # Save cleaned image to temp file for YOLO
    cv2.imwrite(TEMP_IMG, staff.cleaned)
    results = model(TEMP_IMG, imgsz=640, conf=conf, iou=iou, verbose=False)[0]

    offset_y = staff.crop_y1
    offset_x = staff.crop_x1 if hasattr(staff, 'crop_x1') else 0

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        score = float(box.conf)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        det = Detection(
            class_id=cls_id,
            class_name=cls_name,
            conf=round(score, 4),
            x1=x1, y1=y1, x2=x2, y2=y2,
            cx=cx, cy=cy,
            # Convert to full image coordinates
            full_x1=x1 + offset_x,
            full_y1=y1 + offset_y,
            full_x2=x2 + offset_x,
            full_y2=y2 + offset_y,
            full_cx=cx + offset_x,
            full_cy=cy + offset_y
        )
        detections.append(det)

    # Sort left to right
    detections.sort(key=lambda d: d.cx)
    return StaffDetections(staff=staff, detections=detections)


def detect_page(preprocessed: ProcessedScore,
                conf: float = 0.2, iou: float = 0.5) -> PageDetections:
    """
    Run YOLO detection on all staff crops from a preprocessed page.

    Args:
        preprocessed: result from preprocess_image()
        conf:         confidence threshold
        iou:          NMS iou threshold

    Returns PageDetections with:
        .parts[i]     - list of StaffDetections for part i
        .all_staves   - all staves sorted top to bottom
        Each StaffDetections has:
            .detections        - list of Detection objects
            .get_noteheads()   - noteheads sorted left to right
            .get_clef()        - leftmost clef
            .get('className')  - detections of specific class
    """
    print(f'Running detection on {len(preprocessed.all_staves)} staves...')

    num_parts = preprocessed.num_parts
    parts: List[List[StaffDetections]] = [[] for _ in range(num_parts)]
    all_staves: List[StaffDetections] = []

    for staff in preprocessed.all_staves:
        staff_dets = _detect_on_staff(staff, conf=conf, iou=iou)
        part_idx = int(staff.part_id[1:]) - 1
        parts[part_idx].append(staff_dets)
        all_staves.append(staff_dets)
        print(f'  {staff.part_id} staff {staff.staff_in_part+1}: '
              f'{len(staff_dets.detections)} detections')

    return PageDetections(
        image_path=preprocessed.image_path,
        parts=parts,
        all_staves=all_staves
    )


def visualize_detections(page_dets: PageDetections,
                         out_dir: str, conf: float = 0.0) -> None:
    """
    Save annotated images for each staff crop with bounding boxes.
    Also saves JSON and TXT summaries.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for staff_dets in page_dets.all_staves:
        staff = staff_dets.staff

        # Draw on cleaned image converted to BGR
        if len(staff.cleaned.shape) == 2:
            vis = cv2.cvtColor(staff.cleaned, cv2.COLOR_GRAY2BGR)
        else:
            vis = staff.cleaned.copy()

        for det in staff_dets.detections:
            if det.conf < conf:
                continue
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 1)
            cv2.putText(vis, f'{det.class_name} {det.conf:.2f}',
                        (det.x1, max(0, det.y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        name = f'{staff.part_id}_staff{staff.staff_in_part+1:02d}'

        # Save image
        cv2.imwrite(os.path.join(out_dir, f'{name}.png'), vis)

        # Save JSON
        json_data = {
            'part_id': staff.part_id,
            'staff_in_part': staff.staff_in_part,
            'top_y': staff.top_y,
            'bot_y': staff.bot_y,
            'line_positions': staff.line_positions,
            'line_spacing': staff.line_spacing,
            'crop_y1': staff.crop_y1,
            'total_detections': len(staff_dets.detections),
            'detections': [
                {
                    'class_name': d.class_name,
                    'conf': d.conf,
                    'cx': d.cx, 'cy': d.cy,
                    'x1': d.x1, 'y1': d.y1, 'x2': d.x2, 'y2': d.y2,
                    'full_cx': d.full_cx, 'full_cy': d.full_cy
                }
                for d in staff_dets.detections
            ]
        }
        with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

        # Save TXT summary
        with open(os.path.join(out_dir, f'{name}.txt'), 'w') as f:
            f.write(f'{staff.part_id} staff {staff.staff_in_part+1}\n')
            f.write(f'Total: {len(staff_dets.detections)} detections\n')
            f.write('-' * 50 + '\n')
            for d in staff_dets.detections:
                f.write(f'{d.class_name:30s} conf={d.conf:.2f} '
                        f'cx={d.cx:4d} cy={d.cy:4d}\n')

    print(f'Visualizations saved to {out_dir}')


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    test_img = r'S:\mmdetection\data\my_images\img_1.png'
    out_dir = r'S:\omr\detection_test'

    # Preprocess
    preprocessed = preprocess_image(test_img)

    # Detect
    page_dets = detect_page(preprocessed)

    # Visualize and save
    visualize_detections(page_dets, out_dir)

    # Print summary
    print(f'\nDetection summary:')
    for i, part_staves in enumerate(page_dets.parts):
        print(f'\nP{i+1}:')
        for staff_dets in part_staves:
            clef = staff_dets.get_clef()
            notes = staff_dets.get_noteheads()
            print(f'  Staff {staff_dets.staff.staff_in_part+1}: '
                  f'clef={clef.class_name if clef else "none"} '
                  f'noteheads={len(notes)} '
                  f'total={len(staff_dets.detections)}')