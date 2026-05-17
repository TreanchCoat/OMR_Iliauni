"""
build_musicxml_v2.py — Stage 4 (v2) of the OMR pipeline.

Drop-in replacement for ``score_to_xml.build_score_xml`` that:

* Drops barline / dynamics / lyrics / tie-vs-slur differentiation.
* Adds rests at detected x-positions (whole/half/quarter/eighth/16th/32nd).
* Groups noteheads within ±10 px on the x-axis into chords.
* Auto-configures ``<divisions>`` based on the detected symbols
  (configurable per call: ``divisions=None`` → auto, or pass an int).
* Detects fermatas (``fermataAbove`` / ``fermataBelow``) and attaches them
  to the nearest notehead.
* Detects repeat barlines (``repeat-start-barline`` /
  ``repeat-end-barline``) and emits ``<barline><repeat .../></barline>``.
* Detects trill / turn ornaments (the only DeepScores classes that match
  the user's seven melismas).
* Includes a hook (``apply_melisma_overrides``) so the consumer / final app
  can later inject the five remaining melisma types (shake, acciaccatura,
  nachshlang, long appoggiatura, glissando) once detection lands.

The output XML is a single measure per staff — measure recalculation lives
in ``measure_recalculator.py`` and is re-runnable after manual edits.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Optional, List, Dict, Iterable, Tuple

# Self-locating bootstrap: ensure src/ is on sys.path so sibling imports
# work whether this module is imported by an entry-point or run directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xml_builder import XMLBuilder, pixels_to_tenths


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (subset of score_to_xml.py, kept in sync intentionally)
# ─────────────────────────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

CLEF_MIDDLE_PITCH = {
    'clefG':      ('B', 3),    # treble (lowered for clef8 instruments)
    'clefF':      ('D', 3),    # bass
    'clefCAlto':  ('C', 4),    # alto
    'clefCTenor': ('A', 3),    # tenor
}

CLEF_TO_SIGN_LINE = {
    'clefG':      ('G', 2),
    'clefF':      ('F', 4),
    'clefCAlto':  ('C', 3),
    'clefCTenor': ('C', 4),
}

# Notehead → (xml type, multiplier in quarter notes)
# Multipliers are expressed as fractions of a quarter (1 = quarter).
NOTEHEAD_QUARTER_FRACTION = {
    'noteheadBlack':       1.0,    # quarter (refined later via flag/beam)
    'noteheadHalf':        2.0,    # half
    'noteheadWhole':       4.0,    # whole
    'noteheadDoubleWhole': 8.0,    # breve
}

NOTEHEAD_DURATION_TYPE = {
    'noteheadBlack':       'quarter',
    'noteheadHalf':        'half',
    'noteheadWhole':       'whole',
    'noteheadDoubleWhole': 'breve',
}

# Flag → (xml type, multiplier in quarter notes)
FLAG_QUARTER_FRACTION = {
    '8th':  0.5,      # eighth
    '16th': 0.25,     # 16th
    '32nd': 0.125,    # 32nd
    '64th': 0.0625,   # 64th
}
FLAG_DURATION_TYPE = {
    '8th':  'eighth',
    '16th': '16th',
    '32nd': '32nd',
    '64th': '64th',
}

# Rest class → (xml type, multiplier in quarter notes)
REST_CLASSES = {
    'restWhole':    ('whole',   4.0),
    'restHalf':     ('half',    2.0),
    'restQuarter':  ('quarter', 1.0),
    'restEighth':   ('eighth',  0.5),
    'rest8th':      ('eighth',  0.5),     # alt naming
    'rest16th':     ('16th',    0.25),
    'rest32nd':     ('32nd',    0.125),
    'rest64th':     ('64th',    0.0625),
}

ACCIDENTAL_DETECT = {
    'accidentalSharp':       ('sharp',         1),
    'accidentalFlat':        ('flat',         -1),
    'accidentalNatural':     ('natural',       0),
    'accidentalDoubleSharp': ('double-sharp',  2),
    'accidentalDoubleFlat':  ('double-flat',  -2),
}

# ── Time signature detection ────────────────────────────────────────────────
#
# DeepScores ships both *compound* time-signature classes (one box around
# the whole "3/4" stack) and *digit* classes that you can pair up by
# vertical alignment.  We accept either and fall back to "no time
# signature" when neither pattern is present.

# Compound classes — class_name → (beats, beat_type).  This is the safest,
# unambiguous form when the model detects it.
TIME_COMPOUND = {
    'timeSig2over2': (2, 2),  'timeSig2over4': (2, 4),
    'timeSig3over2': (3, 2),  'timeSig3over4': (3, 4),  'timeSig3over8': (3, 8),
    'timeSig4over4': (4, 4),
    'timeSig5over4': (5, 4),  'timeSig5over8': (5, 8),
    'timeSig6over4': (6, 4),  'timeSig6over8': (6, 8),
    'timeSig7over4': (7, 4),  'timeSig7over8': (7, 8),
    'timeSig9over8': (9, 8),
    'timeSig12over8': (12, 8),
}
# Symbolic classes
TIME_SYMBOL = {
    'timeSigCommon':    (4, 4),   # common time → 4/4
    'timeSigCutCommon': (2, 2),   # cut time     → 2/2
    'timeSigCutTime':   (2, 2),   # alt naming
}
# Single-digit classes — class_name → integer.  Used when the model emits
# the numerator and denominator separately and we must pair them by
# vertical position.
TIME_DIGIT = {
    'timeSig0': 0, 'timeSig1': 1, 'timeSig2': 2, 'timeSig3': 3,
    'timeSig4': 4, 'timeSig5': 5, 'timeSig6': 6, 'timeSig7': 7,
    'timeSig8': 8, 'timeSig9': 9, 'timeSig12': 12,
}

# Ornaments that are present in DeepScoresV2 (the only ones we can detect
# automatically today).  Other melisma types remain in the framework as
# manual-tagging hooks via ``apply_melisma_overrides``.
ORNAMENT_DETECT = {
    'ornamentTrill': 'trill',
    'ornamentTurn':  'turn',
}

# Fermata variants (DeepScores has these).
FERMATA_ABOVE_CLASSES = {'fermataAbove', 'fermata'}
FERMATA_BELOW_CLASSES = {'fermataBelow'}

# Repeat barline class names.  DeepScores does not use a single convention,
# so we accept several aliases.  Verify on first run and trim if desired.
REPEAT_FORWARD_CLASSES = {
    'repeatLeft', 'repeat-start-barline', 'repeatStart', 'repeatForward',
}
REPEAT_BACKWARD_CLASSES = {
    'repeatRight', 'repeat-end-barline', 'repeatEnd', 'repeatBackward',
}

# X-tolerance (in pixels) for grouping noteheads into chords.
CHORD_X_TOLERANCE = 10


# ─────────────────────────────────────────────────────────────────────────────
# Auto-divisions
# ─────────────────────────────────────────────────────────────────────────────

def auto_divisions(page_detections) -> int:
    """
    Pick a ``<divisions>`` value that can exactly represent every detected
    rhythmic value on the page.

    Mapping
    -------
        only halves / wholes / quarters       → divisions = 1
        + eighths                              → divisions = 2
        + 16ths                                → divisions = 4
        + 32nds                                → divisions = 8
        + 64ths                                → divisions = 16

    The result is the smallest power-of-two divisions value that keeps every
    duration an integer multiple.
    """
    seen_classes: set[str] = set()
    for sd in page_detections.all_staves:
        for d in sd.detections:
            seen_classes.add(d.class_name)

    divisions = 1
    # Beamed notes are currently always emitted as eighths, so the presence
    # of *any* beam means we need divisions ≥ 2.
    has_beam = 'beam' in seen_classes
    if has_beam or any(c in seen_classes for c in
                       ('rest8th', 'restEighth')) or \
       any('8th' in c for c in seen_classes if c.startswith('flag')):
        divisions = max(divisions, 2)
    if 'rest16th' in seen_classes or any(
            '16th' in c for c in seen_classes if c.startswith('flag')):
        divisions = max(divisions, 4)
    if 'rest32nd' in seen_classes or any(
            '32nd' in c for c in seen_classes if c.startswith('flag')):
        divisions = max(divisions, 8)
    if 'rest64th' in seen_classes or any(
            '64th' in c for c in seen_classes if c.startswith('flag')):
        divisions = max(divisions, 16)
    return divisions


# ─────────────────────────────────────────────────────────────────────────────
# Pitch calculation (identical to score_to_xml.calculate_pitch)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_pitch(notehead_full_cy: float,
                    line_positions: List[int],
                    line_spacing: float,
                    clef_type: str = 'clefG') -> Tuple[str, int]:
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
# Per-staff feature extraction helpers (slimmed copies from score_to_xml.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_main_clef(detections):
    clefs = [d for d in detections
             if d.class_name.startswith('clef') and d.class_name != 'clef8']
    if not clefs:
        return None
    return min(clefs, key=lambda d: d.cx)


def has_octave_marker(detections, main_clef, min_conf: float = 0.5) -> bool:
    if main_clef is None:
        return False
    for d in detections:
        if d.class_name == 'clef8' and d.conf >= min_conf:
            if abs(d.cx - main_clef.cx) < 50:
                return True
    return False


def detect_all_time_signatures(detections, after_x: float = 0
                               ) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Return EVERY detected time signature in left-to-right order.

    Each entry is ``(cx, (beats, beat_type))``.  Detection priority within
    one position:
      1. Compound class    (e.g. ``timeSig3over4``).
      2. Symbolic class    (``timeSigCommon`` → 4/4, ``timeSigCutCommon`` → 2/2).
      3. Stacked digits    (``timeSig3`` over ``timeSig4`` → 3/4) paired by
                           vertical alignment within ±15 px on x.

    Mid-staff time-signature changes show up as additional entries past
    the opening clef/key area.
    """
    cand = [d for d in detections if d.cx > after_x]
    found: List[Tuple[int, Tuple[int, int]]] = []

    # Compound classes (one box = full signature)
    for d in cand:
        if d.class_name in TIME_COMPOUND:
            found.append((int(d.cx), TIME_COMPOUND[d.class_name]))

    # Symbolic classes
    for d in cand:
        if d.class_name in TIME_SYMBOL:
            found.append((int(d.cx), TIME_SYMBOL[d.class_name]))

    # Stacked digits — greedy pairing by x-proximity.
    digits = sorted(
        [d for d in cand if d.class_name in TIME_DIGIT],
        key=lambda x: (x.cx, x.cy),
    )
    groups: List[list] = []
    for d in digits:
        if groups and abs(d.cx - groups[-1][0].cx) <= 15:
            groups[-1].append(d)
        else:
            groups.append([d])
    for g in groups:
        if len(g) >= 2:
            top    = min(g, key=lambda x: x.cy)
            bottom = max(g, key=lambda x: x.cy)
            beats     = TIME_DIGIT.get(top.class_name)
            beat_type = TIME_DIGIT.get(bottom.class_name)
            if (beats is not None and beat_type is not None
                    and beats > 0 and beat_type > 0):
                anchor_cx = int(min(d.cx for d in g))
                found.append((anchor_cx, (beats, beat_type)))

    found.sort(key=lambda e: e[0])

    # De-duplicate near-coincident detections (e.g. a compound class plus
    # a digit pair both describing the same signature).  Collapse entries
    # within 25 px of each other if they agree on (beats, beat_type).
    cleaned: List[Tuple[int, Tuple[int, int]]] = []
    for cx, sig in found:
        if cleaned and abs(cx - cleaned[-1][0]) <= 25 and cleaned[-1][1] == sig:
            continue
        cleaned.append((cx, sig))
    return cleaned


def detect_time_signature(detections, after_x: float = 0
                          ) -> Optional[Tuple[int, int]]:
    """
    Convenience wrapper: return the leftmost time signature in the
    detection list (or None).  Equivalent to
    ``detect_all_time_signatures(...)[0][1]``.
    """
    all_sigs = detect_all_time_signatures(detections, after_x=after_x)
    return all_sigs[0][1] if all_sigs else None


def detect_key_signature(detections, after_x: float) -> Tuple[int, str]:
    flats = [d for d in detections
             if d.class_name == 'keyFlat' and d.cx > after_x]
    sharps = [d for d in detections
              if d.class_name == 'keySharp' and d.cx > after_x]

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


def find_accidental_at(notehead, accidentals, max_dx: int = 40):
    candidates = [a for a in accidentals
                  if 0 < notehead.cx - a.cx < max_dx
                  and abs(a.cy - notehead.cy) < 30]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.cx)


def find_flag_at(notehead, flags, max_dx: int = 25):
    for flag in flags:
        if abs(flag.cx - notehead.cx) < max_dx:
            return flag
    return None


def find_beam_at(notehead, beams, y_threshold: int = 10):
    for beam in beams:
        if beam.x1 <= notehead.cx <= beam.x2:
            if abs(beam.cy - notehead.cy) > y_threshold:
                return beam
    return None


def determine_duration(notehead, all_detections,
                       divisions: int) -> Tuple[str, int]:
    """
    Return (xml_type, raw_divisions_value).

    Algorithm
    ---------
    Start from the notehead class's base value.  If the notehead is black
    (which is the only notehead that can be eighth or shorter), refine via:
      flag → exact subdivision   (flag8th* / flag16th* / flag32nd*)
      beam → eighth              (TODO: detect stacked beams for 16th/32nd)
    """
    nh_class = notehead.class_name
    base_quarter = NOTEHEAD_QUARTER_FRACTION.get(nh_class, 1.0)
    base_type = NOTEHEAD_DURATION_TYPE.get(nh_class, 'quarter')

    quarter_fraction = base_quarter
    xml_type = base_type

    if nh_class == 'noteheadBlack':
        flags = [d for d in all_detections if d.class_name.startswith('flag')]
        flag = find_flag_at(notehead, flags)
        refined = False
        if flag is not None:
            for tag, frac in FLAG_QUARTER_FRACTION.items():
                if tag in flag.class_name:
                    quarter_fraction = frac
                    xml_type = FLAG_DURATION_TYPE[tag]
                    refined = True
                    break

        if not refined:
            beams = [d for d in all_detections if d.class_name == 'beam']
            if find_beam_at(notehead, beams) is not None:
                # Single-beam → eighth (multi-beam not yet detected).
                quarter_fraction = 0.5
                xml_type = 'eighth'

    duration = max(1, int(round(quarter_fraction * divisions)))
    return xml_type, duration


# ─────────────────────────────────────────────────────────────────────────────
# Chord grouping
# ─────────────────────────────────────────────────────────────────────────────

def group_chords(noteheads, x_tolerance: int = CHORD_X_TOLERANCE
                 ) -> List[List]:
    """
    Cluster noteheads whose ``full_cx`` values fall within ``x_tolerance``
    pixels of each other into chord groups.

    The function assumes the input is sorted left-to-right.  Within each
    chord group the noteheads are sorted top-to-bottom (smaller ``full_cy``
    first) so the highest pitch is emitted first (a common MusicXML idiom).
    Returns a list of groups; a single-note "group" is a length-1 list.
    """
    groups: List[List] = []
    current: List = []
    for nh in noteheads:
        if not current or (nh.full_cx - current[-1].full_cx) <= x_tolerance:
            current.append(nh)
        else:
            groups.append(sorted(current, key=lambda n: n.full_cy))
            current = [nh]
    if current:
        groups.append(sorted(current, key=lambda n: n.full_cy))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Slur assignment (no tie/slur differentiation per user request)
# ─────────────────────────────────────────────────────────────────────────────

def assign_slurs(chord_groups, slur_dets, x_margin: int = 5
                 ) -> Tuple[List[bool], List[bool]]:
    """
    For each chord group, decide whether a slur starts/stops at the group's
    leftmost notehead position.  Returns two parallel lists indexed by chord
    group.

    Slurs that don't span at least two chord groups are skipped (likely
    false positives).
    """
    n = len(chord_groups)
    starts = [False] * n
    stops = [False] * n
    # Anchor x for each chord group = leftmost notehead's full_cx
    anchor_x = [min(nh.cx for nh in g) for g in chord_groups]

    for slur in slur_dets:
        x_left = slur.x1 - x_margin
        x_right = slur.x2 + x_margin
        under = [i for i, ax in enumerate(anchor_x)
                 if x_left <= ax <= x_right]
        if len(under) >= 2:
            starts[under[0]] = True
            stops[under[-1]] = True
    return starts, stops


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-anchor utilities (fermata / ornament attachment)
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_chord_index(symbol, chord_groups):
    """
    Return the chord group index whose leftmost notehead is nearest the
    symbol's x-coordinate (Euclidean distance using full_cx / full_cy).
    Returns None if there are no chord groups.
    """
    if not chord_groups:
        return None
    best_i = None
    best_d = float('inf')
    for i, g in enumerate(chord_groups):
        anchor = min(g, key=lambda n: n.cx)
        dx = symbol.full_cx - anchor.full_cx
        dy = symbol.full_cy - anchor.full_cy
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _nearest_chord_or_rest_index(symbol, anchors):
    """anchors is a list of (kind, idx, full_cx, full_cy) tuples."""
    if not anchors:
        return None
    best = None
    best_d = float('inf')
    for entry in anchors:
        _, _, ax, ay = entry
        dx = symbol.full_cx - ax
        dy = symbol.full_cy - ay
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best = entry
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Detection-ID and coordinate-record helpers (mirrors score_to_xml.py)
# ─────────────────────────────────────────────────────────────────────────────

def assign_detection_ids(page_detections) -> Dict[int, str]:
    id_map: Dict[int, str] = {}
    counter = 1
    for sd in page_detections.all_staves:
        for d in sd.detections:
            id_map[id(d)] = f'det_{counter:04d}'
            counter += 1
    return id_map


def build_coordinate_records(page_detections, id_map: Dict[int, str]
                             ) -> list:
    records = []
    for sd in page_detections.all_staves:
        crop_y1 = sd.crop_y1
        for d in sd.detections:
            records.append({
                'id':            id_map[id(d)],
                'class':         d.class_name,
                'conf':          round(float(d.conf), 4),
                'part_id':       sd.part_id,
                'staff_in_part': sd.staff_in_part,
                'cx': int(d.full_cx),
                'cy': int(d.full_cy),
                'x1': int(d.x1),
                'y1': int(d.y1) + crop_y1,
                'x2': int(d.x2),
                'y2': int(d.y2) + crop_y1,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Melisma override hook
# ─────────────────────────────────────────────────────────────────────────────

# Type alias for clarity.
#   { detection_id : { 'kind': 'shake' | 'acciaccatura' | 'nachshlang'
#                              | 'long_appoggiatura' | 'glissando',
#                      'partner_id': str (only for glissando) } }
MelismaOverrides = Dict[str, dict]


def apply_melisma_overrides(builder: XMLBuilder,
                            id_to_note: Dict[str, object],
                            overrides: Optional[MelismaOverrides]):
    """
    Apply user-supplied / future-detector-supplied melisma overrides.

    overrides example::

        {
          'det_0123': {'kind': 'acciaccatura',
                       'step': 'F', 'octave': 5},
          'det_0207': {'kind': 'glissando',
                       'partner_id': 'det_0208'},
        }

    The 'acciaccatura' / 'long_appoggiatura' kinds insert a grace note
    BEFORE the indicated principal note.  'shake' / 'nachshlang' attach a
    notations ornament.  'glissando' marks a pair start/stop.

    This function is intentionally tolerant: unknown ids are skipped with
    a warning rather than raising.
    """
    if not overrides:
        return
    for det_id, spec in overrides.items():
        note = id_to_note.get(det_id)
        if note is None:
            print(f"[melisma] WARN: unknown detection id {det_id}, skipped")
            continue
        kind = spec.get('kind', '').lower()
        if kind in ('shake', 'nachshlang', 'schleifer',
                    'mordent', 'inverted-mordent',
                    'turn', 'trill', 'inverted-turn'):
            try:
                builder.add_ornament(note, kind)
            except ValueError as e:
                print(f"[melisma] {det_id}: {e}")
        elif kind == 'glissando':
            partner = id_to_note.get(spec.get('partner_id'))
            if partner is None:
                print(f"[melisma] {det_id}: glissando partner missing")
                continue
            builder.add_glissando(note, partner)
        elif kind in ('acciaccatura', 'long_appoggiatura'):
            # Insert a grace note immediately before the principal.
            measure = note.getparent()
            idx = list(measure).index(note)
            slash = (kind == 'acciaccatura')
            grace = builder.add_grace_note(
                measure,
                step=spec.get('step', 'C'),
                octave=int(spec.get('octave', 5)),
                alter=spec.get('alter'),
                slash=slash,
                duration_type=spec.get('duration_type', 'eighth'),
                voice=spec.get('voice', 1),
                stem=spec.get('stem'),
            )
            # add_grace_note appends at the end of the measure; move it.
            measure.remove(grace)
            measure.insert(idx, grace)
        else:
            print(f"[melisma] WARN: unsupported kind '{kind}' for {det_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_score_xml_v2(processed_score,
                       page_detections,
                       output_path: str,
                       instrument_name: str = 'Clarinet',
                       midi_program: int = 72,
                       divisions: Optional[int] = None,
                       add_bracket: bool = True,
                       include_layout: bool = True,
                       embed_coordinates: bool = True,
                       chord_tolerance: int = CHORD_X_TOLERANCE,
                       melisma_overrides: Optional[MelismaOverrides] = None,
                       id_map: Optional[Dict[int, str]] = None,
                       ) -> XMLBuilder:
    """
    Assemble preprocessing + detection output into a MusicXML file (v2).

    Parameters
    ----------
    divisions         If None, auto-pick based on detected symbols (see
                      ``auto_divisions``).  Otherwise force the given value.
    chord_tolerance   Pixel x-tolerance for grouping noteheads into chords.
    melisma_overrides Optional dict; see ``apply_melisma_overrides``.
    id_map            Optional precomputed detection-id map.  When given,
                      reuse it so detection IDs in detections.json and
                      omr-coordinates match exactly.  If None, generate
                      one internally (legacy behaviour).
    """
    builder = XMLBuilder()
    builder.set_defaults()

    if divisions is None:
        divisions = auto_divisions(page_detections)

    if id_map is None:
        id_map = assign_detection_ids(page_detections)
    # detection_id → <note> element (filled in as we build).
    # Used by ``apply_melisma_overrides``.
    id_to_note: Dict[str, object] = {}

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

    # Pre-scan: find a page-level time signature.  YOLO typically only
    # detects a time signature glyph on the first staff of the score, but
    # the *meaning* (3/4 etc.) applies to every later staff too.  We use
    # this page-level fallback so that ``restWhole`` symbols on staves 2+
    # are sized to one measure of the active time signature instead of a
    # hard-coded 4 quarters.
    page_time_sig: Optional[Tuple[int, int]] = None
    for sd in page_detections.all_staves:
        ts = detect_time_signature(sd.detections, after_x=0)
        if ts is not None:
            page_time_sig = ts
            break

    # ── Walk each part / staff ──
    for part_idx in range(num_parts):
        part_id = f'P{part_idx + 1}'
        staves = processed_score.parts[part_idx]
        staff_dets_list = page_detections.parts[part_idx]

        for staff_num, (pstaff, sdet) in enumerate(zip(staves,
                                                       staff_dets_list)):
            measure_num = staff_num + 1
            _build_measure_v2(
                builder, part_id, measure_num,
                pstaff, sdet.detections,
                divisions=divisions,
                chord_tolerance=chord_tolerance,
                is_first_measure=(staff_num == 0),
                include_layout=include_layout,
                id_map=id_map,
                id_to_note=id_to_note,
                page_time_sig=page_time_sig,
            )

    # ── Apply melisma overrides (manual / future detectors) ──
    apply_melisma_overrides(builder, id_to_note, melisma_overrides)

    # ── Embed coordinate metadata ──
    if embed_coordinates:
        builder.add_miscellaneous_field('omr-version', '2.0')
        builder.add_miscellaneous_field(
            'omr-source-image',
            os.path.basename(page_detections.image_path),
        )
        builder.add_miscellaneous_field('omr-image-width',
                                        str(page_detections.img_w))
        builder.add_miscellaneous_field('omr-image-height',
                                        str(page_detections.img_h))
        builder.add_miscellaneous_field('omr-divisions', str(divisions))

        coords = build_coordinate_records(page_detections, id_map)
        builder.add_miscellaneous_field(
            'omr-coordinates',
            json.dumps(coords, separators=(',', ':')),
        )

    builder.save(output_path)
    return builder


# ─────────────────────────────────────────────────────────────────────────────
# Measure assembly (one full staff = one measure; recalculation is a separate
# script the user can re-run after manual edits — see measure_recalculator.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_measure_v2(builder: XMLBuilder, part_id: str, measure_num: int,
                      pstaff, dets, *,
                      divisions: int,
                      chord_tolerance: int,
                      is_first_measure: bool,
                      include_layout: bool,
                      id_map: Dict[int, str],
                      id_to_note: Dict[str, object],
                      page_time_sig: Optional[Tuple[int, int]] = None):
    # ── Clef ──
    clef_det = find_main_clef(dets)
    clef_type = clef_det.class_name if clef_det else 'clefG'
    clef_sign, clef_line = CLEF_TO_SIGN_LINE.get(clef_type, ('G', 2))
    clef_octave_change = -1 if has_octave_marker(dets, clef_det) else None

    # ── Key signature ──
    after_x = clef_det.cx if clef_det else 0
    key_fifths, key_mode = detect_key_signature(dets, after_x)

    # ── Time signature(s) ──
    # YOLO may detect MULTIPLE time signatures within a single staff
    # (e.g. piece starts in 3/4, switches to 4/4 mid-line).  We grab the
    # full ordered list; the FIRST entry sets the staff's opening time
    # signature on the measure-level <attributes>, and any further ones
    # are interleaved with the note events below so the recalculator can
    # split them into their own measures.
    all_sigs = detect_all_time_signatures(dets, after_x=after_x)
    local_sig = all_sigs[0][1] if all_sigs else None
    time_sig = local_sig or page_time_sig
    extra_time_sigs = all_sigs[1:]   # mid-staff time changes (with cx)
    # Only write <time> into <attributes> on the first measure, OR when
    # this staff carries a new signature that differs from the page-level
    # default (mid-piece time-signature change).
    write_time = is_first_measure or (
        local_sig is not None and local_sig != page_time_sig
    )
    time_beats = time_sig[0] if (time_sig and write_time) else None
    time_beat_type = time_sig[1] if (time_sig and write_time) else None

    # ── Forward repeat (start of measure) ──
    forward_repeat = any(d.class_name in REPEAT_FORWARD_CLASSES for d in dets)
    backward_repeat = any(d.class_name in REPEAT_BACKWARD_CLASSES for d in dets)

    # ── Create the measure element ──
    if is_first_measure:
        measure = builder.add_measure(
            part_id, measure_num,
            divisions=divisions,
            key_fifths=key_fifths, key_mode=key_mode,
            time_beats=time_beats, time_beat_type=time_beat_type,
            clef_sign=clef_sign, clef_line=clef_line,
            clef_octave_change=clef_octave_change,
        )
    else:
        measure = builder.add_measure(part_id, measure_num)
        builder.add_print(measure, new_system=True)

    if forward_repeat:
        builder.add_repeat(measure, direction='forward')

    # ── Pull symbol groups ──
    noteheads = sorted([d for d in dets if d.class_name.startswith('notehead')],
                       key=lambda d: d.cx)
    rests = sorted([d for d in dets if d.class_name in REST_CLASSES],
                   key=lambda d: d.cx)
    accidentals = [d for d in dets if d.class_name in ACCIDENTAL_DETECT]
    slur_dets = [d for d in dets if d.class_name == 'slur']
    fermatas = [d for d in dets
                if d.class_name in FERMATA_ABOVE_CLASSES
                or d.class_name in FERMATA_BELOW_CLASSES]
    ornaments = [d for d in dets if d.class_name in ORNAMENT_DETECT]

    # If nothing detected → whole-measure rest.
    if not noteheads and not rests:
        builder.add_rest(measure,
                         duration=divisions * 4,
                         measure_rest=True,
                         voice=1)
        if backward_repeat:
            builder.add_repeat(measure, direction='backward')
        return

    # ── Group noteheads into chord clusters ──
    chord_groups = group_chords(noteheads, x_tolerance=chord_tolerance)
    slur_starts, slur_stops = assign_slurs(chord_groups, slur_dets)

    # ── Build a unified timeline of "events" sorted by x-position ──
    # Each event is either ('chord', idx_in_chord_groups), ('rest',
    # rest_detection), or ('timesig', (beats, beat_type)).  Sorted by
    # anchor x so they appear in the same left-to-right order they were
    # detected at.
    events: List[Tuple[str, object, int]] = []
    for i, g in enumerate(chord_groups):
        anchor = min(g, key=lambda n: n.cx)
        events.append(('chord', g, anchor.cx))
    for r in rests:
        events.append(('rest', r, r.cx))
    for ts_cx, ts_value in extra_time_sigs:
        events.append(('timesig', ts_value, ts_cx))
    events.sort(key=lambda e: e[2])

    # Layout reference
    first_x = None
    top_line_y = pstaff.line_positions[0] if include_layout else None
    if include_layout and events:
        if events[0][0] == 'chord':
            first_x = min(n.full_cx for n in events[0][1])
        else:
            first_x = events[0][1].full_cx

    # Anchor table for fermata / ornament attachment (across the whole
    # measure: each chord group's first emitted note + each rest).
    # Each entry: (note_element, full_cx, full_cy)
    note_anchors: List[Tuple[object, float, float]] = []

    # ── Emit events in order ──
    chord_idx_counter = 0
    # Currently-active time signature inside this measure.  Tracks the
    # opening sig so we know when an "extra" timesig event represents a
    # real change (vs. a duplicate detection of the same value).
    active_sig = time_sig
    # Local import: we need to drop a raw <attributes><time> element into
    # the measure for mid-staff time changes.  The recalculator
    # subsequently splits on those.
    from lxml import etree as _etree

    for kind, payload, _ in events:
        if kind == 'timesig':
            new_sig = payload
            if new_sig == active_sig:
                # Duplicate detection of the same signature → ignore.
                continue
            attrs_el = _etree.SubElement(measure, 'attributes')
            t_el = _etree.SubElement(attrs_el, 'time')
            _etree.SubElement(t_el, 'beats').text = str(new_sig[0])
            _etree.SubElement(t_el, 'beat-type').text = str(new_sig[1])
            active_sig = new_sig
            continue

        if kind == 'rest':
            rest_det = payload
            xml_type, q_fraction = REST_CLASSES[rest_det.class_name]
            # A "whole rest" in real notation means "rest this measure"
            # regardless of the time signature.  If we know the time
            # signature, emit it as a measure-rest sized to one measure;
            # the recalculator then treats it as occupying exactly one
            # measure rather than the 4-quarter-note worth that a literal
            # whole note would carry.
            is_whole_rest = rest_det.class_name == 'restWhole'
            measure_rest_flag = False
            if is_whole_rest and time_sig is not None:
                dur = (time_sig[0] * divisions * 4) // time_sig[1]
                measure_rest_flag = True
                # Omit <type> on measure-rests; renderers infer it.
                xml_type = None
            else:
                dur = max(1, int(round(q_fraction * divisions)))
            r_default_x = r_default_y = None
            if include_layout and first_x is not None:
                r_default_x = pixels_to_tenths(
                    rest_det.full_cx - first_x) + 15
                r_default_y = pixels_to_tenths(
                    top_line_y - rest_det.full_cy)
            rest_el = builder.add_rest(
                measure,
                duration=dur,
                duration_type=xml_type,
                measure_rest=measure_rest_flag,
                voice=1,
                default_x=r_default_x,
                default_y=r_default_y,
                element_id=id_map.get(id(rest_det)),
            )
            rid = id_map.get(id(rest_det))
            if rid is not None:
                id_to_note[rid] = rest_el
            note_anchors.append((rest_el,
                                 float(rest_det.full_cx),
                                 float(rest_det.full_cy)))
            continue

        # kind == 'chord'
        group = payload
        gi = chord_idx_counter
        chord_idx_counter += 1

        # Determine duration from the topmost note (first in sorted group).
        ref_note = group[0]
        dur_type, dur_value = determine_duration(ref_note, dets, divisions)

        # Group anchor for layout
        anchor_full_cx = min(n.full_cx for n in group)

        first_note_in_chord = None
        for k, nh in enumerate(group):
            step, octave = calculate_pitch(
                nh.full_cy, pstaff.line_positions,
                pstaff.line_spacing, clef_type
            )
            acc_det = find_accidental_at(nh, accidentals)
            acc_name = acc_alter = None
            if acc_det is not None:
                acc_name, acc_alter = ACCIDENTAL_DETECT[acc_det.class_name]

            default_x = default_y = None
            if include_layout and first_x is not None:
                default_x = pixels_to_tenths(anchor_full_cx - first_x) + 15
                default_y = pixels_to_tenths(top_line_y - nh.full_cy)

            nh_id = id_map.get(id(nh))
            note_el = builder.add_note(
                measure,
                step=step, octave=octave,
                alter=acc_alter,
                accidental=acc_name,
                duration=dur_value,
                duration_type=dur_type,
                voice=1,
                default_x=default_x,
                default_y=default_y,
                chord=(k > 0),
                slur_start=(slur_starts[gi] if k == 0 else False),
                slur_stop=(slur_stops[gi] if k == 0 else False),
                element_id=nh_id,
            )
            if nh_id is not None:
                id_to_note[nh_id] = note_el
            if k == 0:
                first_note_in_chord = note_el

        if first_note_in_chord is not None:
            note_anchors.append((
                first_note_in_chord,
                float(anchor_full_cx),
                float(group[0].full_cy),
            ))

    # ── Post-pass: attach fermatas / ornaments to nearest anchor ──
    def _nearest(symbol):
        """Return the <note> element whose anchor is closest to symbol."""
        if not note_anchors:
            return None
        best = min(note_anchors,
                   key=lambda a: ((a[1] - symbol.full_cx) ** 2
                                  + (a[2] - symbol.full_cy) ** 2))
        return best[0]

    # Fermatas
    for f in fermatas:
        target = _nearest(f)
        if target is None:
            continue
        ftype = ('upright'
                 if f.class_name in FERMATA_ABOVE_CLASSES
                 else 'inverted')
        builder.add_fermata(target, type=ftype)

    # Ornaments (trill / turn)
    for orn in ornaments:
        target = _nearest(orn)
        if target is None:
            continue
        builder.add_ornament(target, ORNAMENT_DETECT[orn.class_name])

    # Backward repeat at end of measure
    if backward_repeat:
        builder.add_repeat(measure, direction='backward')


# ─────────────────────────────────────────────────────────────────�
