import sys
import os
sys.path.append(r'S:\omr')
from music21 import stream, note, clef, meter, key
from staff_detector import get_staff_crops
from staff_remover import preprocess_staff_crop
from ultralytics import YOLO
import cv2
import numpy as np

# Constants
MODEL_PATH = r'S:\saved_models\deepscores_crops_v1.pt'
MM_PER_TENTH = 6.1807 / 40
DPI = 300
PIXELS_PER_MM = DPI / 25.4

DURATION_MAP = {
    'noteheadWhole': ('whole', 16),
    'noteheadHalf': ('half', 8),
    'noteheadBlack': ('quarter', 4),  # default, flags/beams refine this
    'noteheadDoubleWhole': ('breve', 32),
}

NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Treble clef: middle line (index 2) = B4
# Bass clef: middle line (index 2) = D3
CLEF_MIDDLE = {
    'clefG': ('B', 4),
    'clefF': ('D', 3),
    'clefCAlto': ('C', 4),
    'clefCTenor': ('C', 4),
}

def pixels_to_tenths(pixels):
    mm = pixels / PIXELS_PER_MM
    return round(mm / MM_PER_TENTH, 2)

def get_pitch(note_cy_in_crop, crop_offset_y, staff, clef_type):
    """
    Calculate pitch from note y position relative to staff lines.
    note_cy_in_crop: y center of note in crop image coordinates
    crop_offset_y: y offset of crop from original image
    staff: staff dict with line positions in original image coords
    """
    # Convert to original image coordinates
    note_cy = note_cy_in_crop + crop_offset_y

    spacing = staff['spacing']
    middle_line_y = staff['lines'][2]

    # Each step = half spacing (line to space = half spacing)
    steps_from_middle = (middle_line_y - note_cy) / (spacing / 2)
    steps = round(steps_from_middle)

    ref_name, ref_octave = CLEF_MIDDLE.get(clef_type, ('B', 4))
    ref_idx = NOTE_NAMES.index(ref_name)

    abs_idx = ref_idx + steps
    octave_shift = abs_idx // 7
    note_idx = abs_idx % 7
    if note_idx < 0:
        note_idx += 7
        octave_shift -= 1

    return NOTE_NAMES[note_idx], ref_octave + octave_shift

def detect_symbols_in_crop(cleaned_crop, meta, detection_model):
    """Run YOLO detection on cleaned staff crop"""
    # Save temp image for YOLO
    temp_path = r'S:\omr\temp_crop.png'
    cv2.imwrite(temp_path, cleaned_crop)

    results = detection_model(temp_path, imgsz=640, conf=0.2,
                              iou=0.5, verbose=False)[0]
    symbols = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls_name = detection_model.names[int(box.cls)]
        symbols.append({
            'class': cls_name,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cx': (x1 + x2) / 2,
            'cy': (y1 + y2) / 2,
            'conf': float(box.conf)
        })
    return symbols

def process_image(img_path, output_xml_path):
    """
    Full pipeline: image -> MusicXML
    """
    print(f'Processing: {img_path}')
    detection_model = YOLO(MODEL_PATH)

    # Step 1: Get staff crops
    crops = get_staff_crops(img_path)
    if not crops:
        print('No staves detected!')
        return

    print(f'Found {len(crops)} staves')

    # Step 2: Build score
    score = stream.Score()
    part = stream.Part()
    part.id = 'P1'
    score.append(part)

    measure_num = 1

    for crop_img, meta in crops:
        staff = meta['staff']
        offset_y = meta['offset_y']

        # Step 3: Preprocess
        cleaned = preprocess_staff_crop(crop_img)

        # Step 4: Detect symbols
        symbols = detect_symbols_in_crop(cleaned, meta, detection_model)

        # Step 5: Find clef for this staff
        clef_type = 'clefG'  # default
        clefs_found = [s for s in symbols if s['class'] in CLEF_MIDDLE]
        if clefs_found:
            # Use leftmost clef
            clef_type = min(clefs_found, key=lambda s: s['cx'])['class']

        # Step 6: Find noteheads sorted left to right
        noteheads = [s for s in symbols
                     if s['class'] in DURATION_MAP]
        noteheads.sort(key=lambda s: s['cx'])

        if not noteheads:
            continue

        # Step 7: Build measure
        m = stream.Measure(number=measure_num)
        measure_num += 1

        # Add clef
        if clef_type == 'clefG':
            m.append(clef.TrebleClef())
        elif clef_type == 'clefF':
            m.append(clef.BassClef())

        # Check for time signature
        time_sigs = [s for s in symbols if s['class'].startswith('timeSig')]
        if time_sigs:
            ts = time_sigs[0]['class'].replace('timeSig', '')
            if ts == 'Common':
                m.append(meter.TimeSignature('4/4'))
            elif ts == 'CutCommon':
                m.append(meter.TimeSignature('2/2'))
            elif ts.isdigit():
                pass  # need numerator+denominator pair, handle later

        # Add notes
        for nh in noteheads:
            pitch_name, octave = get_pitch(
                nh['cy'], offset_y, staff, clef_type)
            duration_type, _ = DURATION_MAP[nh['class']]

            n = note.Note(f'{pitch_name}{octave}')
            n.duration.type = duration_type

            # Store original pixel position as editorial data
            n.pixel_x = nh['cx']
            n.pixel_y = nh['cy'] + offset_y
            n.tenths_x = pixels_to_tenths(nh['cx'])
            n.tenths_y = pixels_to_tenths(nh['cy'] + offset_y)

            m.append(n)

        part.append(m)

    # Step 8: Save
    score.write('musicxml', fp=output_xml_path)
    print(f'Saved to {output_xml_path}')

if __name__ == '__main__':
    process_image(
        r'S:\mmdetection\data\my_images\img_1.png',
        r'S:\omr\output.xml'
    )