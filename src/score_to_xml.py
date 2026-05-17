"""
score_to_xml.py — Stage 4 of the OMR pipeline.

Takes the output of preprocessing.py and symbol_detector.py and writes a
MusicXML file using the framework in xml_builder.py.

Pipeline position
-----------------
    image
      → preprocessing.preprocess_image()  → ProcessedScore
      → symbol_detector.detect_page()     → PageDetections
      → score_to_xml.build_score_xml()    → MusicXML

Current limitation
------------------
Barline detection is not yet implemented (see handoff TODO list), so this
module currently treats **one detected staff = one measure**.  When barline
detection lands, replace the measure-assignment loop in build_score_xml()
with one that splits each staff into multiple measures based on barline x.

Assumptions about the inputs
----------------------------
ProcessedScore (from preprocessing.py):
    .num_parts                  int
    .parts[i]                   list of ProcessedStaff for part i
    ProcessedStaff exposes      .line_positions, .line_spacing,
                                .top_y, .bot_y, .left_x, .right_x

PageDetections (from symbol_detector.py):
    .parts[i][j]                StaffDetections for part i, staff j
    StaffDetections.detections  list of Detection objects
    Detection has               .class_name, .conf,
                                .cx, .cy            (crop coords)
                                .full_cx, .full_cy  (full image coords)
                                .x1, .y1, .x2, .y2  (crop coords bbox)
"""

from __future__ import annotations

import sys
import os
import json

# Self-locating bootstrap: ensure src/ is on sys.path so sibling imports work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, List, Dict
from xml_builder import XMLBuilder, pixels_to_tenths


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Pitch sitting on the MIDDLE staff line (line_positions[2]) for each clef.
# The pitch detector measures everything as half-spacings up/down from this.
#
# NOTE on octaves:
#   Music-theoretic standard: treble clef middle line = B4.  This codebase
#   currently uses B3 instead because the typical input here is a Bb-clarinet
#   score with a clef8 (treble 8vb) marker, where the SOUNDING pitch is one
#   octave lower than written.  Lowering the reference by one octave gives a
#   final XML whose <pitch> values match the sounding pitch, which is what
#   downstream MIDI converters expect.
#   If you want strict written-pitch output (and rely on <clef-octave-change>
#   being respected by your viewer), change ('B', 3) back to ('B', 4) here.
CLEF_MIDDLE_PITCH = {
    'clefG':      ('B', 3),    # treble clef (lowered for clef8 instruments)
    'clefF':      ('D', 3),    # bass clef
    'clefCAlto':  ('C', 4),    # alto clef (C on middle line)
    'clefCTenor': ('A', 3),    # tenor clef (C on 4th line, A on middle)
}

# Detected clef class → MusicXML <clef> sign + line
CLEF_TO_SIGN_LINE = {
    'clefG':      ('G', 2),
    'clefF':      ('F', 4),
    'clefCAlto':  ('C', 3),
    'clefCTenor': ('C', 4),
}

# Notehead class → (MusicXML <type>, divisions value when divisions=4)
DURATION_MAP = {
    'noteheadBlack':       ('quarter', 4),
    'noteheadHalf':        ('half',    8),
    'noteheadWhole':       ('whole',  16),
    'noteheadDoubleWhole': ('breve',  32),
}

# Detected accidental class → (xml accidental name, alter value)
ACCIDENTAL_DETECT = {
    'accidentalSharp':       ('sharp',         1),
    'accidentalFlat':        ('flat',         -1),
    'accidentalNatural':     ('natural',       0),
    'accidentalDoubleSharp': ('double-sharp',  2),
    'accidentalDoubleFlat':  ('double-flat',  -2),
}


# ─────────────────────────────────────────────────────────────────────────────
# Pitch calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_pitch(notehead_full_cy: float,
                    line_positions: List[int],
                    line_spacing: float,
                    clef_type: str = 'clefG') -> tuple:
    """
    Given a notehead's y position (in FULL IMAGE pixels) and the staff line
    geometry, return its (step, octave).

    Algorithm
    ---------
    The middle staff line (index 2) is the reference. Each half-spacing above
    is one diatonic step up, each half-spacing below is one step down. The
    step is wrapped through the diatonic alphabet; octave shifts when wrapping.

    Notes on Y axis
    ---------------
    In image coordinates Y grows DOWNWARD, so a smaller y means higher pitch.
    `(middle_y - note_y) / half_spacing` gives positive values for higher notes.
    """
    middle_line_y = line_positions[2]
    half_spacing = line_spacing / 2

    steps = round((middle_line_y - notehead_full_cy) / half_spacing)

    ref_step, ref_octave = CLEF_MIDDLE_PITCH.get(clef_type, ('B', 4))
    ref_idx = NOTE_NAMES.index(ref_step)

    abs_idx = ref_idx + steps
    octave_shift = abs_idx // 7
    note_idx = abs_idx % 7
    if note_idx < 0:
        note_idx += 7
        octave_shift -= 1

    return NOTE_NAMES[note_idx], ref_octave + octave_shift


# ─────────────────────────────────────────────────────────────────────────────
# Per-staff feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def find_main_clef(detections):
    """
    Return the leftmost clef detection (excluding clef8, which is the small
    "8" below a treble clef indicating octave-down transposition).
    Returns None if no clef detected.
    """
    clefs = [d for d in detections
             if d.class_name.startswith('clef') and d.class_name != 'clef8']
    if not clefs:
        return None
    return min(clefs, key=lambda d: d.cx)


def has_octave_marker(detections, main_clef, min_conf: float = 0.5) -> bool:
    """
    Check for a clef8 detection right under the main clef.
    Returns True for treble-8vb (clarinet, tenor voice, etc.).

    min_conf gates weak detections — clef8 is a small symbol that the model
    often hallucinates with low confidence; gating prevents spurious octave
    shifts on every measure.
    """
    if main_clef is None:
        return False
    for d in detections:
        if d.class_name == 'clef8' and d.conf >= min_conf:
            if abs(d.cx - main_clef.cx) < 50:
                return True
    return False


def detect_key_signature(detections, after_x: float) -> tuple:
    """
    Count keyFlat / keySharp accidentals appearing between the clef and the
    first notehead. Returns (fifths, mode).

    Only one type is expected per signature (all sharps or all flats), so if
    both are present the larger group wins.
    """
    flats = [d for d in detections
             if d.class_name == 'keyFlat' and d.cx > after_x]
    sharps = [d for d in detections
              if d.class_name == 'keySharp' and d.cx > after_x]

    # Filter out anything that's actually past the first notehead — those are
    # note accidentals, not key signature.
    noteheads = [d for d in detections if d.class_name.startswith('notehead')]
    if noteheads:
        first_nh_x = min(n.cx for n in noteheads)
        flats  = [f for f in flats  if f.cx < first_nh_x]
        sharps = [s for s in sharps if s.cx < first_nh_x]

    if len(flats) > len(sharps):
        return -len(flats), 'major'
    if len(sharps) > 0:
        return len(sharps), 'major'
    return 0, 'major'


def get_noteheads_sorted(detections) -> list:
    """All notehead detections in this staff, sorted left-to-right."""
    nh = [d for d in detections if d.class_name.startswith('notehead')]
    return sorted(nh, key=lambda d: d.cx)


def find_beam_at(notehead, beams, y_threshold: int = 10):
    """
    Find a beam whose horizontal span contains this notehead and which is
    vertically separated from the notehead (i.e. above the notehead for
    stems-up, or below for stems-down). Returns the beam, or None.
    """
    for beam in beams:
        if beam.x1 <= notehead.cx <= beam.x2:
            if abs(beam.cy - notehead.cy) > y_threshold:
                return beam
    return None


def find_flag_at(notehead, flags, max_dx: int = 25):
    """Find a flag (e.g. flag8thUp) close to this notehead horizontally."""
    for flag in flags:
        if abs(flag.cx - notehead.cx) < max_dx:
            return flag
    return None


def find_accidental_at(notehead, accidentals, max_dx: int = 40):
    """
    Find an accidental immediately to the LEFT of a notehead and roughly at
    the same y. Returns the closest such accidental, or None.
    """
    candidates = [a for a in accidentals
                  if 0 < notehead.cx - a.cx < max_dx
                  and abs(a.cy - notehead.cy) < 30]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.cx)   # closest = largest x


def determine_duration(notehead, all_detections) -> tuple:
    """
    Determine duration (type, divisions) for a notehead.

    Defaults from DURATION_MAP, then refines noteheadBlack via flags and beams.
    Returned divisions assume `divisions=4` at the measure level
    (so quarter=4, eighth=2, 16th=1).

    Refinement rules
    ----------------
    flag32nd*  → 32nd
    flag16th*  → 16th
    flag8th*   → eighth
    beam       → eighth   (multi-beam → 16th is TODO; needs beam-stack count)
    """
    base_type, base_divs = DURATION_MAP.get(notehead.class_name, ('quarter', 4))

    if notehead.class_name != 'noteheadBlack':
        return base_type, base_divs

    flags = [d for d in all_detections if d.class_name.startswith('flag')]
    flag = find_flag_at(notehead, flags)
    if flag is not None:
        if '32nd' in flag.class_name:
            return '32nd', 0     # divisions=4 can't represent 32nd cleanly
        if '16th' in flag.class_name:
            return '16th', 1
        if '8th' in flag.class_name:
            return 'eighth', 2

    beams = [d for d in all_detections if d.class_name == 'beam']
    if find_beam_at(notehead, beams) is not None:
        # TODO: detect double-beam (16th) by looking for two stacked beams
        return 'eighth', 2

    return base_type, base_divs


def assign_slurs(noteheads, slur_dets,
                 x_margin: int = 5) -> tuple:
    """
    Given the staff's noteheads (sorted left-to-right) and slur detections,
    return two parallel boolean lists: (slur_starts, slur_stops).

    For each slur, the leftmost notehead within its x-span is marked as the
    slur start, and the rightmost is marked as the stop.  Slurs that contain
    fewer than 2 noteheads are skipped (likely false positives).

    x_margin lets the slur extend slightly beyond a notehead's center to
    catch slurs whose end exactly matches a notehead position.
    """
    n = len(noteheads)
    starts = [False] * n
    stops  = [False] * n

    for slur in slur_dets:
        x_left  = slur.x1 - x_margin
        x_right = slur.x2 + x_margin
        under = [i for i, nh in enumerate(noteheads)
                 if x_left <= nh.cx <= x_right]
        if len(under) >= 2:
            starts[under[0]] = True
            stops[under[-1]] = True

    return starts, stops


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def assign_detection_ids(page_detections) -> Dict[int, str]:
    """
    Assign a sequential ID to every detection on the page.

    Returns a dict keyed by Python id() of each Detection object, mapping to
    a string ID like 'det_0001'.  Using id() avoids needing to mutate the
    Detection dataclass.
    """
    id_map: Dict[int, str] = {}
    counter = 1
    for sd in page_detections.all_staves:
        for d in sd.detections:
            id_map[id(d)] = f'det_{counter:04d}'
            counter += 1
    return id_map


def build_coordinate_records(page_detections, id_map: Dict[int, str]) -> list:
    """
    Build a list of dicts (one per detection) suitable for embedding in a
    <miscellaneous-field> as JSON.

    All coordinates are absolute, in the FULL RECTIFIED IMAGE.

    Schema per record:
        id              str   detection ID (e.g. 'det_0042')
        class           str   YOLO class name
        conf            float detection confidence
        part_id         str   'P1', 'P2', …
        staff_in_part   int   0-based staff index within its part
        cx, cy          int   center in full image
        x1, y1, x2, y2  int   bbox in full image
    """
    records = []
    for sd in page_detections.all_staves:
        crop_y1 = sd.crop_y1
        for d in sd.detections:
            records.append({
                'id':             id_map[id(d)],
                'class':          d.class_name,
                'conf':           round(float(d.conf), 4),
                'part_id':        sd.part_id,
                'staff_in_part':  sd.staff_in_part,
                # Center in full-image coords (already provided)
                'cx': int(d.full_cx),
                'cy': int(d.full_cy),
                # Bbox: x is unchanged (crop is full-width), y shifts by crop_y1
                'x1': int(d.x1),
                'y1': int(d.y1) + crop_y1,
                'x2': int(d.x2),
                'y2': int(d.y2) + crop_y1,
            })
    return records


def build_score_xml(processed_score,
                    page_detections,
                    output_path: str,
                    instrument_name: str = 'Clarinet',
                    midi_program: int = 72,         # GM 72 = Clarinet
                    divisions: int = 4,
                    add_bracket: bool = True,
                    include_layout: bool = True,
                    embed_coordinates: bool = True) -> XMLBuilder:
    """
    Assemble preprocessing + detection output into a MusicXML file.

    Parameters
    ----------
    processed_score    ProcessedScore from preprocessing.preprocess_image()
    page_detections    PageDetections from symbol_detector.detect_page()
    output_path        Where to write the .xml file.
    instrument_name    Name for parts (default 'Clarinet').
    midi_program       General MIDI program number (default 72 = clarinet).
    divisions          Divisions per quarter note (default 4).
    add_bracket        Whether to bracket-group all parts in part-list.
    include_layout     Whether to compute default-x/default-y for notes.
    embed_coordinates  If True, embed the absolute coordinates of every
                       detection into <identification><miscellaneous> as a
                       JSON blob, and add `id` attributes on every <note>
                       element so the consuming app can cross-reference.

    Returns the XMLBuilder so callers can inspect or extend before saving.
    """
    builder = XMLBuilder()
    builder.set_defaults()

    # Assign an ID to every detection up front.  Even when not embedding the
    # full JSON, IDs let us write `id="det_NNNN"` on each <note>.
    id_map = assign_detection_ids(page_detections)

    num_parts = processed_score.num_parts

    # ── Part list ──
    if add_bracket and num_parts > 1:
        builder.add_part_group(number=1, group_type='start', symbol='bracket')

    for i in range(num_parts):
        part_id = f'P{i+1}'
        name = f'{instrument_name} {i+1}' if num_parts > 1 else instrument_name
        builder.add_part(part_id,
                         instrument_name=name,
                         midi_channel=i + 1,
                         midi_program=midi_program)

    if add_bracket and num_parts > 1:
        builder.add_part_group(number=1, group_type='stop')

    # ── Walk each part / staff ──
    for part_idx in range(num_parts):
        part_id = f'P{part_idx + 1}'
        staves = processed_score.parts[part_idx]
        staff_dets_list = page_detections.parts[part_idx]

        for staff_num, (pstaff, sdet) in enumerate(zip(staves, staff_dets_list)):
            measure_num = staff_num + 1
            dets = sdet.detections

            _build_measure(builder, part_id, measure_num,
                           pstaff, dets, divisions,
                           is_first_measure=(staff_num == 0),
                           include_layout=include_layout,
                           id_map=id_map)

    # ── Embed coordinate metadata ──
    if embed_coordinates:
        builder.add_miscellaneous_field('omr-version', '1.0')
        builder.add_miscellaneous_field(
            'omr-source-image',
            os.path.basename(page_detections.image_path),
        )
        builder.add_miscellaneous_field(
            'omr-image-width',  str(page_detections.img_w))
        builder.add_miscellaneous_field(
            'omr-image-height', str(page_detections.img_h))

        coords = build_coordinate_records(page_detections, id_map)
        # Compact JSON (no indentation) keeps the XML file size reasonable
        builder.add_miscellaneous_field(
            'omr-coordinates',
            json.dumps(coords, separators=(',', ':')),
        )

    builder.save(output_path)
    return builder


def _build_measure(builder: XMLBuilder, part_id: str, measure_num: int,
                   pstaff, dets, divisions: int,
                   is_first_measure: bool,
                   include_layout: bool,
                   id_map: Optional[Dict[int, str]] = None):
    """
    Build a single measure for one staff.

    Currently treats the whole staff as one measure (no barline detection).

    id_map: optional mapping from id(detection) to a string ID.  When
            provided, each <note> built from a notehead gets that ID as an
            attribute, enabling cross-reference from the miscellaneous block.
    """
    # ── Identify clef ──
    clef_det = find_main_clef(dets)
    clef_type = clef_det.class_name if clef_det else 'clefG'
    clef_sign, clef_line = CLEF_TO_SIGN_LINE.get(clef_type, ('G', 2))
    clef_octave_change = -1 if has_octave_marker(dets, clef_det) else None

    # ── Key signature ──
    after_x = clef_det.cx if clef_det else 0
    key_fifths, key_mode = detect_key_signature(dets, after_x)

    # ── Create the measure element ──
    if is_first_measure:
        measure = builder.add_measure(
            part_id, measure_num,
            divisions=divisions,
            key_fifths=key_fifths, key_mode=key_mode,
            clef_sign=clef_sign, clef_line=clef_line,
            clef_octave_change=clef_octave_change,
        )
    else:
        # Subsequent measures: just a new system marker.
        # (When barline detection lands, only system-starting measures should
        # get this marker; intra-system measures should not.)
        measure = builder.add_measure(part_id, measure_num)
        builder.add_print(measure, new_system=True)

    # ── Process noteheads ──
    noteheads = get_noteheads_sorted(dets)

    if not noteheads:
        # Empty staff → whole-measure rest.
        builder.add_rest(measure,
                         duration=divisions * 4,
                         measure_rest=True,
                         voice=1)
        return

    accidentals = [d for d in dets if d.class_name in ACCIDENTAL_DETECT]
    slur_dets   = [d for d in dets if d.class_name == 'slur']
    slur_starts, slur_stops = assign_slurs(noteheads, slur_dets)

    # Reference x for default-x: leftmost notehead. Each note's default-x
    # becomes its offset from the leftmost note + a small lead-in.
    first_x = noteheads[0].full_cx if include_layout else None
    top_line_y = pstaff.line_positions[0] if include_layout else None

    for i, nh in enumerate(noteheads):
        # Pitch
        step, octave = calculate_pitch(
            nh.full_cy, pstaff.line_positions,
            pstaff.line_spacing, clef_type
        )

        # Accidental
        acc_det = find_accidental_at(nh, accidentals)
        acc_name, acc_alter = (None, None)
        if acc_det is not None:
            acc_name, acc_alter = ACCIDENTAL_DETECT[acc_det.class_name]

        # Duration
        dur_type, dur_divs = determine_duration(nh, dets)

        # Layout
        default_x = default_y = None
        if include_layout:
            default_x = pixels_to_tenths(nh.full_cx - first_x) + 15
            # MusicXML default-y is positive UP from top staff line,
            # but image y grows down — invert.
            default_y = pixels_to_tenths(top_line_y - nh.full_cy)

        # Cross-reference ID for the miscellaneous coordinate block
        nh_id = id_map.get(id(nh)) if id_map is not None else None

        builder.add_note(
            measure,
            step=step, octave=octave,
            alter=acc_alter,
            accidental=acc_name,
            duration=dur_divs if dur_divs > 0 else 1,
            duration_type=dur_type,
            voice=1,
            default_x=default_x,
            default_y=default_y,
            slur_start=slur_starts[i],
            slur_stop=slur_stops[i],
            element_id=nh_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from preprocessing import preprocess_image
    from symbol_detector import detect_page

    test_img   = r'S:\mmdetection\data\my_images\img_1.png'
    output_xml = r'S:\omr\test_output.xml'

    print('[1/3] Preprocessing image …')
    processed = preprocess_image(test_img)

    print('[2/3] Running symbol detection …')
    detections = detect_page(processed)

    print('[3/3] Building MusicXML …')
    build_score_xml(processed, detections, output_xml)

    print(f'\nDone -> {output_xml}')
