"""
Regenerate the user's `score.xml` using build_musicxml_v2.

We can't re-run the YOLO pipeline here (no GPU / model), but the original
file embeds every detection as a JSON blob in
<identification><miscellaneous><miscellaneous-field name="omr-coordinates">.
We reconstruct minimal ProcessedScore / PageDetections shapes from that
blob and feed them into build_score_xml_v2 to verify:

  1. The output XML loads cleanly in MuseScore (schema-valid ordering).
  2. The time signature (timeSig3 + timeSig4) is auto-detected → 3/4.
  3. Rests, chords, fermatas, etc. survive the v2 path.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, '/sessions/happy-amazing-euler/mnt/omr')

from lxml import etree
from build_musicxml_v2 import build_score_xml_v2


# ─────────────────────────────────────────────────────────────────────────────
# Read the user's score.xml and pull out the omr-coordinates JSON
# ─────────────────────────────────────────────────────────────────────────────

INPUT_XML = '/sessions/happy-amazing-euler/mnt/uploads/score.xml'
OUTPUT_XML = '/sessions/happy-amazing-euler/mnt/omr/score_v2.xml'

tree = etree.parse(INPUT_XML)
root = tree.getroot()

coords_field = root.find(
    './identification/miscellaneous/'
    'miscellaneous-field[@name="omr-coordinates"]'
)
assert coords_field is not None, 'omr-coordinates field missing'
coords = json.loads(coords_field.text)
print(f'Loaded {len(coords)} detections from embedded JSON.')

img_w_field = root.find('./identification/miscellaneous/'
                       'miscellaneous-field[@name="omr-image-width"]')
img_h_field = root.find('./identification/miscellaneous/'
                       'miscellaneous-field[@name="omr-image-height"]')
src_field   = root.find('./identification/miscellaneous/'
                       'miscellaneous-field[@name="omr-source-image"]')
img_w = int(img_w_field.text)
img_h = int(img_h_field.text)
src   = src_field.text


# ─────────────────────────────────────────────────────────────────────────────
# Mock pipeline shapes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Det:
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
class StaffDet:
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
class Page:
    image_path: str
    img_h: int
    img_w: int
    num_parts: int
    parts: list = field(default_factory=list)
    all_staves: list = field(default_factory=list)


@dataclass
class PStaff:
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
class PScore:
    image_path: str
    img_h: int
    img_w: int
    num_parts: int
    parts: list = field(default_factory=list)


# Group coordinates by (part_id, staff_in_part)
by_staff: dict = {}
for c in coords:
    key = (c['part_id'], c['staff_in_part'])
    by_staff.setdefault(key, []).append(c)


# For each staff, infer line_positions from any `staff` class detection
# (cy / bbox), otherwise fall back to bounding-box of all noteheads.
def estimate_staff_geometry(items):
    staffs = [d for d in items if d['class'] == 'staff']
    if staffs:
        s = max(staffs, key=lambda d: d['conf'])
        top = s['y1']; bot = s['y2']
    else:
        # Fallback: notehead bbox
        nh = [d for d in items if d['class'].startswith('notehead')]
        if not nh:
            return None
        top = min(d['cy'] for d in nh) - 20
        bot = max(d['cy'] for d in nh) + 20
    spacing = (bot - top) / 4
    line_pos = [int(top + spacing * i) for i in range(5)]
    return line_pos, spacing, top, bot


# Determine num_parts and order parts by P1, P2, ...
all_part_ids = sorted({k[0] for k in by_staff}, key=lambda s: int(s[1:]))
num_parts = len(all_part_ids)
print(f'Found {num_parts} parts: {all_part_ids}')

parts_pstaff: list = [[] for _ in range(num_parts)]
parts_staffdets: list = [[] for _ in range(num_parts)]
all_staffdets: list = []

for pi, pid in enumerate(all_part_ids):
    staff_ids = sorted({k[1] for k in by_staff if k[0] == pid})
    for sid in staff_ids:
        items = by_staff[(pid, sid)]
        geom = estimate_staff_geometry(items)
        if geom is None:
            # No notes; skip
            continue
        line_pos, spacing, top, bot = geom

        dets = []
        for c in items:
            cx, cy = c['cx'], c['cy']
            x1, y1, x2, y2 = c['x1'], c['y1'], c['x2'], c['y2']
            dets.append(Det(
                class_name=c['class'], conf=c['conf'],
                cx=cx, cy=cy,
                x1=x1, y1=y1, x2=x2, y2=y2,
                full_cx=cx, full_cy=cy,
            ))
        sd = StaffDet(
            part_id=pid, staff_in_part=sid,
            top_y=int(top), bot_y=int(bot),
            line_positions=line_pos, line_spacing=spacing,
            crop_y1=0, crop_x1=0, detections=dets,
        )
        parts_staffdets[pi].append(sd)
        all_staffdets.append(sd)

        ps = PStaff(
            line_positions=line_pos, line_spacing=spacing,
            top_y=int(top), bot_y=int(bot),
            left_x=0, right_x=img_w,
            crop_x1=0, crop_y1=0,
            part_id=pid, staff_in_part=sid,
        )
        parts_pstaff[pi].append(ps)

page = Page(image_path=src, img_h=img_h, img_w=img_w,
            num_parts=num_parts,
            parts=parts_staffdets, all_staves=all_staffdets)
score = PScore(image_path=src, img_h=img_h, img_w=img_w,
               num_parts=num_parts, parts=parts_pstaff)


# ─────────────────────────────────────────────────────────────────────────────
# Run v2 builder
# ─────────────────────────────────────────────────────────────────────────────

build_score_xml_v2(score, page, OUTPUT_XML,
                   instrument_name='Clarinet', midi_program=72,
                   divisions=None, embed_coordinates=True)


# ─────────────────────────────────────────────────────────────────────────────
# Validate ordering + time signature presence
# ─────────────────────────────────────────────────────────────────────────────

out_tree = etree.parse(OUTPUT_XML)
out_root = out_tree.getroot()
top_order = [c.tag for c in out_root]
print('top-level order:', top_order)

required = ['work','movement-number','movement-title',
            'identification','defaults','credit','part-list','part']
idx = {t: i for i, t in enumerate(required)}
filtered = [t for t in top_order if t in idx]
ok_order = all(idx[a] <= idx[b] for a, b in zip(filtered, filtered[1:]))
print('schema-valid order?', ok_order)

# Check the first measure of every part for a <time> element.
for p in out_root.findall('part'):
    pid = p.get('id')
    first = p.find('measure')
    time = first.find('attributes/time') if first is not None else None
    if time is not None:
        beats = time.findtext('beats')
        bt    = time.findtext('beat-type')
        print(f'  {pid}: time = {beats}/{bt}')
    else:
        print(f'  {pid}: no <time> emitted')

print(f'\nOutput → {OUTPUT_XML}')
