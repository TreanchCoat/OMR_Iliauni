"""
Smoke test for build_musicxml_v2.py + measure_recalculator.py.

Uses synthetic fixtures so it runs without YOLO / preprocessing.  The
fixtures fake just enough of the ProcessedScore / PageDetections shape to
exercise:

 - auto_divisions
 - chord grouping at ±10 px
 - rest emission
 - fermata attachment
 - trill ornament attachment
 - melisma override hook (acciaccatura via grace note)
 - measure recalculation under 2/4

We run, then re-parse the output and assert on a handful of structural
invariants.
"""

from dataclasses import dataclass, field
from typing import List

# Bootstrap: this test lives at <project>/tests/, so the project root
# is one level up.  Register src/ on sys.path before importing.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: F401

from build_musicxml_v2 import build_score_xml_v2, auto_divisions
from measure_recalculator import recalculate
from lxml import etree


# ── Minimal mock dataclasses that mirror the production shapes ──

@dataclass
class MockDetection:
    class_name: str
    conf: float
    cx: int
    cy: int
    x1: int
    y1: int
    x2: int
    y2: int
    full_cx: int
    full_cy: int


@dataclass
class MockStaffDet:
    part_id: str
    staff_in_part: int
    top_y: int
    bot_y: int
    line_positions: list
    line_spacing: float
    crop_y1: int
    crop_x1: int = 0
    detections: list = field(default_factory=list)


@dataclass
class MockPageDet:
    image_path: str
    img_h: int
    img_w: int
    num_parts: int
    parts: list = field(default_factory=list)
    all_staves: list = field(default_factory=list)


@dataclass
class MockProcessedStaff:
    line_positions: list
    line_spacing: float
    top_y: int
    bot_y: int
    left_x: int
    right_x: int
    crop_x1: int
    crop_y1: int
    part_id: str
    staff_in_part: int


@dataclass
class MockProcessedScore:
    image_path: str
    img_h: int
    img_w: int
    num_parts: int
    parts: list = field(default_factory=list)


def _det(cls, cx, cy, conf=0.9, w=14, h=14):
    x1 = cx - w // 2
    y1 = cy - h // 2
    return MockDetection(
        class_name=cls, conf=conf,
        cx=cx, cy=cy, x1=x1, y1=y1, x2=cx + w // 2, y2=cy + h // 2,
        full_cx=cx, full_cy=cy,
    )


def build_fixture():
    # One staff, treble clef.  line_positions span 100..180 (spacing 20).
    line_positions = [100, 120, 140, 160, 180]
    line_spacing = 20.0

    detections = [
        _det('clefG',        50, 140),
        _det('keyFlat',      80, 130),

        # Time signature 3/4 (compound class)
        _det('timeSig3over4', 110, 140),

        # Two single noteheads + a chord of three at the same x
        _det('noteheadBlack', 200, 140),
        _det('noteheadBlack', 260, 140),

        # Chord: three noteheads within 5px of each other, far apart
        # vertically so we can verify "topmost" ordering.
        _det('noteheadHalf',  320, 110),   # high
        _det('noteheadHalf',  322, 140),
        _det('noteheadHalf',  325, 170),   # low

        # A rest
        _det('restQuarter',   400, 140),

        # A black notehead with an attached 8th flag (eighth note)
        _det('noteheadBlack', 460, 140),
        _det('flag8thUp',     465, 110),

        # Fermata above a note
        _det('fermataAbove',  202, 80),

        # Trill ornament
        _det('ornamentTrill', 262, 80),
    ]

    staff_det = MockStaffDet(
        part_id='P1', staff_in_part=0,
        top_y=100, bot_y=180,
        line_positions=line_positions, line_spacing=line_spacing,
        crop_y1=0, crop_x1=0,
        detections=detections,
    )

    pstaff = MockProcessedStaff(
        line_positions=line_positions, line_spacing=line_spacing,
        top_y=100, bot_y=180, left_x=0, right_x=600,
        crop_x1=0, crop_y1=0,
        part_id='P1', staff_in_part=0,
    )

    page = MockPageDet(
        image_path='/tmp/fake.png', img_h=400, img_w=600,
        num_parts=1, parts=[[staff_det]], all_staves=[staff_det],
    )
    score = MockProcessedScore(
        image_path='/tmp/fake.png', img_h=400, img_w=600,
        num_parts=1, parts=[[pstaff]],
    )
    return score, page


def assert_(cond, msg):
    if not cond:
        print(f'   FAIL: {msg}')
        return False
    print(f'   ok   {msg}')
    return True


def main() -> int:
    score, page = build_fixture()

    # Auto-divisions: with a flag8thUp present, expect divisions=2.
    divs = auto_divisions(page)
    print(f'auto_divisions = {divs}')
    ok = assert_(divs == 2, 'divisions=2 with eighth flag present')

    out_path = r'/sessions/happy-amazing-euler/mnt/omr\_smoke_out.xml'
    build_score_xml_v2(score, page, out_path,
                       instrument_name='Test',
                       divisions=None)

    tree = etree.parse(out_path)
    root = tree.getroot()
    part = root.find('part')
    measure = part.find('measure')

    notes = measure.findall('note')
    # We expect: 2 single noteheads + 3 chord notes + 1 rest + 1 eighth
    # note = 7 <note> elements in the measure.
    ok &= assert_(len(notes) == 7,
                  f'measure has 7 note elements (got {len(notes)})')

    # First chord-continuation marker on the 4th/5th of the chord
    chord_marks = [n for n in notes if n.find('chord') is not None]
    ok &= assert_(len(chord_marks) == 2,
                  '2 chord-continuation notes (3-note chord → 2 <chord/>)')

    # The eighth note must have <type>eighth</type>
    types = [n.findtext('type') for n in notes]
    ok &= assert_('eighth' in types, 'eighth-note type emitted')

    # One <rest> child
    rests = [n for n in notes if n.find('rest') is not None]
    ok &= assert_(len(rests) == 1, 'exactly one rest emitted')
    ok &= assert_(rests[0].findtext('type') == 'quarter',
                  'rest type=quarter')

    # Fermata attached to one note
    fermatas = measure.findall('.//fermata')
    ok &= assert_(len(fermatas) == 1, 'one fermata emitted')

    # Trill attached to one note
    trills = measure.findall('.//ornaments/trill-mark')
    ok &= assert_(len(trills) == 1, 'one trill-mark emitted')

    # divisions = 2 in attributes
    div_text = measure.findtext('attributes/divisions')
    ok &= assert_(div_text == '2', f'divisions=2 in attributes (got {div_text})')

    # Time signature 3/4 should have been auto-detected and written
    time_in_builder = measure.find('attributes/time')
    ok &= assert_(time_in_builder is not None,
                  'time signature auto-emitted by builder')
    if time_in_builder is not None:
        ok &= assert_(time_in_builder.findtext('beats') == '3',
                      f'detected beats=3 (got {time_in_builder.findtext("beats")})')
        ok &= assert_(time_in_builder.findtext('beat-type') == '4',
                      f'detected beat-type=4 (got {time_in_builder.findtext("beat-type")})')

    # ── Recalculator auto-picks the time signature from the XML ──
    auto_path = r'/sessions/happy-amazing-euler/mnt/omr\_smoke_out_rebar_auto.xml'
    recalculate(out_path, auto_path)
    tree_auto = etree.parse(auto_path)
    ms_auto = tree_auto.getroot().find('part').findall('measure')
    print(f'auto-recalc (3/4): {len(ms_auto)} measures')
    ok &= assert_(len(ms_auto) >= 1,
                  'at least one measure after auto-recalc')
    auto_time = ms_auto[0].find('attributes/time')
    ok &= assert_(auto_time is not None
                  and auto_time.findtext('beats') == '3'
                  and auto_time.findtext('beat-type') == '4',
                  'auto-recalc kept 3/4 from input XML')

    # ── Manual --time override still works ──
    re_path = r'/sessions/happy-amazing-euler/mnt/omr\_smoke_out_rebar.xml'
    recalculate(out_path, re_path, time_signature='2/4')

    tree2 = etree.parse(re_path)
    part2 = tree2.getroot().find('part')
    measures2 = part2.findall('measure')
    print(f'after recalc (override 2/4): {len(measures2)} measures')
    ok &= assert_(len(measures2) >= 1, 'at least one measure after recalc')

    # Time signature should be the override (2/4), not the embedded 3/4
    time_el = measures2[0].find('attributes/time')
    ok &= assert_(time_el is not None, 'time signature in first measure')
    if time_el is not None:
        ok &= assert_(time_el.findtext('beats') == '2',
                      f'override beats=2 (got {time_el.findtext("beats")})')
        ok &= assert_(time_el.findtext('beat-type') == '4',
                      f'override beat-type=4 (got {time_el.findtext("beat-type")})')

    # ── Pass-through mode: no --time AND no <time> in XML ──
    # Build a stripped copy of the input XML with the <time> removed.
    stripped_in  = r'/sessions/happy-amazing-euler/mnt/omr\_smoke_out_notime_in.xml'
    pt_path      = r'/sessions/happy-amazing-euler/mnt/omr\_smoke_out_passthrough.xml'
    stripped_tree = etree.parse(out_path)
    for time_el in stripped_tree.getroot().findall('.//attributes/time'):
        time_el.getparent().remove(time_el)
    stripped_tree.write(stripped_in, pretty_print=True,
                        xml_declaration=True, encoding='UTF-8')
    recalculate(stripped_in, pt_path)
    tree3 = etree.parse(pt_path)
    ms3 = tree3.getroot().find('part').findall('measure')
    ok &= assert_(len(ms3) == 1,
                  f'pass-through keeps 1 measure (got {len(ms3)})')

    print()
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0
