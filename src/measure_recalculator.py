"""
measure_recalculator.py — Stage 4½ of the OMR pipeline.

A *separate*, *re-runnable* script that takes a MusicXML file produced by
``build_musicxml_v2`` (or any MusicXML really) and assigns its notes into
measures using the **time-signature-only** rule.

Why "time-signature-only"?
--------------------------
The downstream consumer of this XML does NOT care that each measure
contains exactly N beats — that arithmetic adds complexity (overflow
handling, ghost measures, drift) without benefiting the final product.
What the consumer DOES care about is: when the time signature changes
in the score, the change should land at a real bar boundary so the
renderer can show ``3/4 … 2/4 … 6/8`` correctly.

Therefore this script:

* Reads any well-formed MusicXML.
* Flattens every existing measure back into one stream per part (so a
  previous re-bar is not "sticky" if the user re-runs after manual
  edits).
* Walks the stream and dumps every note / rest / direction into the
  current measure.  **A new measure is opened only when a fresh
  ``<time>`` element is encountered** (i.e. a time-sig change).
* Carried-forward bits: ``<attributes>`` on the first measure (clef,
  key, divisions, time), per-note ``id``, ``default-x/y``, slurs, ties,
  ornaments, fermatas — all preserved verbatim.
* Forward / backward repeat barlines still migrate to the right measure.

End result
----------
* No time changes detected → one measure per part covering everything.
* N time changes detected   → N+1 measures, each holding the notes that
  fell between two consecutive time signatures.

Part padding
------------
After splitting, short parts are padded with full-measure rests so every
part has the same measure count.  MuseScore rejects scores where parts
have differing measure counts (it renders each missing one as a red
"blank" box).

Time-signature resolution
-------------------------
* If ``--time`` is given on the CLI, that value wins (manual override).
* Otherwise the script reads the first ``<time>`` already in the XML
  (written there by build_musicxml_v2 from YOLO's detection).
* If neither is present, the file is passed through unchanged.

CLI usage
---------
::

    python measure_recalculator.py in.xml out.xml             # use <time> from XML
    python measure_recalculator.py in.xml out.xml --time 2/4  # override
    python measure_recalculator.py in.xml out.xml --time 6/8  # override

The script can also be imported and invoked programmatically::

    from measure_recalculator import recalculate
    recalculate('input.xml', 'output.xml')                 # auto-detect
    recalculate('input.xml', 'output.xml', time_signature='3/4')  # override
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import List, Optional, Tuple

from lxml import etree


# ─────────────────────────────────────────────────────────────────────────────
# Time-signature parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_time_signature(spec: str) -> Tuple[int, int]:
    """
    Accept '4/4', '3/4', '6/8', '2/4', etc.  Returns (beats, beat_type).
    Raises ValueError on bad input.
    """
    if '/' not in spec:
        raise ValueError(f"Invalid time signature {spec!r}; expected 'N/M'.")
    a, b = spec.split('/', 1)
    beats = int(a.strip())
    beat_type = int(b.strip())
    if beats <= 0 or beat_type <= 0:
        raise ValueError(f"Time signature components must be positive: {spec!r}")
    return beats, beat_type


def measure_duration_in_divisions(beats: int,
                                  beat_type: int,
                                  divisions: int) -> int:
    """
    Length of one measure in raw <duration> units.

    Example: 4/4 with divisions=4 → 4 * 4 * (4/4) = 16
             6/8 with divisions=2 → 6 * 2 * (4/8) = 6
             3/4 with divisions=8 → 3 * 8 * (4/4) = 24
    """
    # In MusicXML, divisions is the number of subdivisions of a *quarter*
    # note.  One beat in time signature N/M is worth (4/M) quarter notes.
    # Therefore one beat is divisions * 4 / M raw units.
    # Total measure = beats * divisions * 4 / M.
    total = beats * divisions * 4
    if total % beat_type != 0:
        # Best-effort: warn but round.
        print(f"[warn] divisions={divisions} does not cleanly support "
              f"time {beats}/{beat_type}; measure length will be rounded.")
    return total // beat_type


# ─────────────────────────────────────────────────────────────────────────────
# MusicXML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _note_duration(note: etree._Element) -> int:
    """
    Read <duration> from a <note>.  Returns 0 if absent (grace notes have
    no duration in MusicXML).  Returns 0 also for any element that is not
    a <note>.
    """
    if note.tag != 'note':
        return 0
    if note.find('grace') is not None:
        return 0
    d = note.find('duration')
    if d is None or d.text is None:
        return 0
    try:
        return int(d.text)
    except ValueError:
        return 0


def _is_chord_continuation(note: etree._Element) -> bool:
    """True for <note><chord/>...</note> — these share their previous
    note's onset, so they don't advance the time pointer."""
    return note.tag == 'note' and note.find('chord') is not None


def _extract_attributes(part: etree._Element) -> Optional[etree._Element]:
    """
    Return the first <attributes> element under any measure of ``part``,
    or None if none exists.  Used to seed the first new measure.
    """
    for m in part.findall('measure'):
        a = m.find('attributes')
        if a is not None:
            return copy.deepcopy(a)
    return None


def _extract_divisions(part: etree._Element, default: int = 4) -> int:
    """Find the <divisions> value declared in this part (first occurrence)."""
    for d in part.iter('divisions'):
        try:
            return int(d.text)
        except (TypeError, ValueError):
            continue
    return default


def _flatten_part_children(part: etree._Element) -> List[etree._Element]:
    """
    Return every child of every <measure> in document order, dropping the
    measure wrappers themselves and ``<print>`` elements.

    ``<attributes>`` blocks are PRESERVED only if they declare a new
    ``<time>`` signature — this lets us pick up mid-piece time-signature
    changes during the split.  Other attributes (clef changes, key
    changes) are dropped because they're already captured in the carried
    first-measure attributes block.

    Barlines are kept so we can migrate repeats to the new measures.
    """
    flat: List[etree._Element] = []
    for mi, m in enumerate(part.findall('measure')):
        for child in m:
            if child.tag == 'print':
                continue
            if child.tag == 'attributes':
                # Skip the very first <attributes> (we emit it ourselves
                # on the new first measure).  Keep any later one that
                # carries a <time> — that's a time-signature change.
                if mi == 0:
                    continue
                if child.find('time') is None:
                    continue
                # Forward a stripped-down copy containing only <time>.
                ts = child.find('time')
                attrs = etree.Element('attributes')
                attrs.append(copy.deepcopy(ts))
                flat.append(attrs)
                continue
            flat.append(child)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Recalculation core
# ─────────────────────────────────────────────────────────────────────────────

def _make_full_measure_rest(divisions: int,
                            beats: int,
                            beat_type: int) -> etree._Element:
    """
    Build a single <note><rest measure="yes"/><duration>...</duration></note>
    sized for one full measure under the given time/divisions.
    """
    n = etree.Element('note')
    etree.SubElement(n, 'rest', measure='yes')
    d = etree.SubElement(n, 'duration')
    d.text = str(measure_duration_in_divisions(beats, beat_type, divisions))
    v = etree.SubElement(n, 'voice')
    v.text = '1'
    return n


def _split_part(part: etree._Element,
                beats: int,
                beat_type: int) -> None:
    """
    Rebuild the measures of ``part`` preserving the original staff
    boundaries.

    The v2 builder emits **one input measure per detected staff** in the
    page image.  That natural boundary is meaningful (one line of music
    on the page = one bar in the score) and we want to keep it.  In
    addition, when a NEW ``<time>`` signature is encountered mid-stream
    inside one of those staff-measures, we split it further so the time
    change always lands on a real bar boundary.

    Result
    ------
      * N input measures (= N staves) with no internal time-sig changes
        →  N output measures, each holding the same notes as the staff.
      * Any staff that contains a mid-stream ``<time>`` produces +1
        output measure per time change.

    Notes / rests / directions / chord continuations are copied verbatim;
    repeat barlines migrate to the appropriate adjacent measure as before.
    """
    divisions = _extract_divisions(part)

    original_attrs = _extract_attributes(part)
    if original_attrs is None:
        original_attrs = etree.Element('attributes')
        etree.SubElement(original_attrs, 'divisions').text = str(divisions)
    # Stamp the time signature onto the carried attributes block.
    existing_time = original_attrs.find('time')
    if existing_time is not None:
        original_attrs.remove(existing_time)
    time_el = etree.SubElement(original_attrs, 'time')
    etree.SubElement(time_el, 'beats').text = str(beats)
    etree.SubElement(time_el, 'beat-type').text = str(beat_type)

    # Snapshot existing input measures BEFORE we wipe them.
    input_measures = list(part.findall('measure'))
    for m in input_measures:
        part.remove(m)

    cur: Optional[etree._Element] = None
    cur_num = 0
    pending_forward_repeat = False

    def _open_measure(initial_attrs: Optional[etree._Element] = None,
                      time_only: Optional[etree._Element] = None
                      ) -> etree._Element:
        """
        Open a new <measure>.

        initial_attrs  — full <attributes> block to insert at the start
                         (used for the very first output measure).
        time_only      — a <time> element to wrap in a fresh
                         <attributes> block (mid-piece time-sig change
                         or carried time on a staff boundary).
        """
        nonlocal cur, cur_num, pending_forward_repeat
        cur_num += 1
        m = etree.SubElement(part, 'measure', number=str(cur_num))
        if initial_attrs is not None:
            m.append(initial_attrs)
        elif time_only is not None:
            attrs = etree.SubElement(m, 'attributes')
            attrs.append(copy.deepcopy(time_only))
        if pending_forward_repeat:
            bl = etree.SubElement(m, 'barline', location='left')
            etree.SubElement(bl, 'bar-style').text = 'heavy-light'
            etree.SubElement(bl, 'repeat', direction='forward')
            pending_forward_repeat = False
        cur = m
        return m

    # First output measure carries the original (time-stamped) attributes.
    _open_measure(initial_attrs=original_attrs)

    # Walk each input measure (= each detected staff).  The first input
    # measure feeds the already-opened output measure.  Every subsequent
    # input measure starts a fresh output measure.
    for mi, input_m in enumerate(input_measures):
        if mi > 0:
            # New staff → open a fresh output measure.  If this input
            # measure declares a <time> in its <attributes>, propagate
            # it so renderers show the time-sig change at this barline.
            t_here = input_m.find('attributes/time')
            if t_here is not None:
                _open_measure(time_only=t_here)
            else:
                _open_measure()

        # Walk the children of this input measure.  We've already handled
        # the leading <attributes>; mid-input-measure <attributes> with
        # a <time> still trigger an extra split.
        for ci, child in enumerate(input_m):
            if child.tag == 'print':
                continue
            if child.tag == 'attributes':
                # Skip the very first <attributes> — its content is
                # either the original attrs we already inserted (for the
                # first output measure) or the time-sig we already
                # forwarded (above).  Mid-measure <attributes> with a
                # new <time> still split.
                if ci == 0:
                    continue
                new_time = child.find('time')
                if new_time is not None:
                    _open_measure(time_only=new_time)
                continue

            if child.tag == 'barline':
                rep = child.find('repeat')
                if rep is not None:
                    direction = rep.get('direction', 'backward')
                    if direction == 'forward':
                        pending_forward_repeat = True
                    else:
                        if cur is not None:
                            cur.append(copy.deepcopy(child))
                continue

            # Everything else: notes (incl. chord continuations),
            # directions, sounds → append to current output measure.
            if cur is None:
                _open_measure()
            cur.append(copy.deepcopy(child))


def _pad_parts_to_longest(root: etree._Element,
                          beats: int,
                          beat_type: int) -> None:
    """
    After ``_split_part`` has run on every part, make sure all parts
    contain the same number of measures by appending empty
    ``<note><rest measure="yes"/></note>`` bars to the short ones.

    This is purely a cosmetic / schema-compatibility step: MuseScore
    rejects scores where parts have differing measure counts (it shows
    each missing measure as a red "blank" box).  Padding lets the user
    see whatever notes WERE detected without the render failing.
    """
    parts = root.findall('part')
    if len(parts) <= 1:
        return
    # The longest part defines the target length.  Pull <divisions>
    # from each part separately so we size the rest correctly.
    measure_counts = [len(p.findall('measure')) for p in parts]
    target = max(measure_counts)
    if target == 0:
        return
    for p in parts:
        existing = len(p.findall('measure'))
        if existing >= target:
            continue
        divs = _extract_divisions(p)
        start_num = existing + 1
        # If the part is COMPLETELY empty (no <measure> at all), give it
        # a first measure that carries the time signature so renderers
        # know what to render.
        if existing == 0:
            m = etree.SubElement(p, 'measure', number='1')
            attrs = etree.SubElement(m, 'attributes')
            etree.SubElement(attrs, 'divisions').text = str(divs)
            t = etree.SubElement(attrs, 'time')
            etree.SubElement(t, 'beats').text = str(beats)
            etree.SubElement(t, 'beat-type').text = str(beat_type)
            m.append(_make_full_measure_rest(divs, beats, beat_type))
            start_num = 2
        for n in range(start_num, target + 1):
            m = etree.SubElement(p, 'measure', number=str(n))
            m.append(_make_full_measure_rest(divs, beats, beat_type))


def _detect_time_signature_in_xml(root: etree._Element
                                  ) -> Optional[Tuple[int, int]]:
    """
    Look at the first <attributes><time> in any <part> and return
    (beats, beat-type).  Returns None if no time signature is present.
    """
    for part in root.findall('part'):
        time_el = part.find('measure/attributes/time')
        if time_el is None:
            continue
        try:
            beats = int(time_el.findtext('beats') or '')
            beat_type = int(time_el.findtext('beat-type') or '')
            if beats > 0 and beat_type > 0:
                return beats, beat_type
        except ValueError:
            continue
    return None


def recalculate(input_path: str,
                output_path: str,
                time_signature: Optional[str] = None) -> None:
    """
    Read input MusicXML and write a re-barred copy.

    Time-signature resolution:
    * If ``time_signature`` is given, it overrides everything.
    * Otherwise the function reads the time signature from the input
      XML's first ``<attributes><time>`` block.
    * If neither is present, the file is parsed and re-emitted unchanged.
    """
    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False)
    tree = etree.parse(input_path, parser)
    root = tree.getroot()

    beats_pair: Optional[Tuple[int, int]] = None
    if time_signature is not None:
        beats_pair = parse_time_signature(time_signature)
    else:
        beats_pair = _detect_time_signature_in_xml(root)
        if beats_pair is not None:
            print(f'[info] Using time signature from XML: '
                  f'{beats_pair[0]}/{beats_pair[1]}')
        else:
            print('[info] No time signature found in XML and none passed '
                  'on the CLI — passing through (single measure per staff).')

    if beats_pair is not None:
        beats, beat_type = beats_pair
        for part in root.findall('part'):
            _split_part(part, beats, beat_type)
        # Pad every part to the longest part's measure count so MuseScore
        # (and other renderers) can lay them out side-by-side.  Empty
        # parts come about when YOLO mis-assigns all staves to a single
        # part_id — the others end up with one empty measure, which
        # MuseScore renders as red "missing measure" boxes.
        _pad_parts_to_longest(root, beats, beat_type)

    doctype = (
        '<!DOCTYPE score-partwise PUBLIC '
        '"-//Recordare//DTD MusicXML 3.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/3.0/partwise.dtd">'
    )
    xml_bytes = etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding='UTF-8',
        doctype=doctype,
    )
    with open(output_path, 'wb') as f:
        f.write(xml_bytes)
    print(f'Recalculated -> {output_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog='measure_recalculator',
        description=('Re-split the measures of a MusicXML file based on a '
                     'given time signature.  Safe to re-run after manual '
                     'edits.'),
    )
    p.add_argument('input',  help='Input  .xml path')
    p.add_argument('output', help='Output .xml path')
    p.add_argument('--time', dest='time_signature', default=None,
                   help='Time signature, e.g. 2/4, 3/4, 4/4, 6/8.  Omit '
                        'to use the time signature embedded in the XML '
                        '(written there by build_musicxml_v2).')
    args = p.parse_args(argv)
    recalculate(args.input, args.output, time_signature=args.time_signature)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
