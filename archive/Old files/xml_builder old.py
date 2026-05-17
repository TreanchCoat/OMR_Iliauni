"""
xml_builder.py  —  Low-level MusicXML builder for the OMR pipeline.

Design philosophy
-----------------
Every parameter is optional.  The builder never invents or infers values you
didn't explicitly provide — it just writes the XML tags for what you give it.
This means you can call add_measure() with no clef/time/key, add_note() with no
duration type, etc. and the output will still be valid XML (whether it is valid
*MusicXML* depends on what you put in, but that's intentional — the pipeline
controls what it knows).

Direct XML via lxml
-------------------
The previous version used music21, which auto-fills many fields behind the
scenes.  This version writes XML directly so there are no hidden assumptions.

Coordinate system
-----------------
  default_x  — tenths, relative to start of measure
  default_y  — tenths, relative to top staff line (positive = up in MusicXML)

  Helper:
      from xml_builder import pixels_to_tenths
      x_tenths = pixels_to_tenths(x_pixels)

Usage example
-------------
    builder = XMLBuilder()

    builder.add_part('P1')                     # name is optional
    builder.add_part('P2', part_name='Clarinet 2')

    # First measure: set divisions, clef, key, time
    m1 = builder.add_measure('P1', 1,
                              divisions=4,
                              clef_sign='G', clef_line=2, clef_octave_change=-1,
                              key_fifths=-1, key_mode='major',
                              time_beats=4, time_beat_type=4)
    builder.add_print(m1, top_system_distance=331)
    builder.add_sound(m1, tempo=96)

    # Whole-measure rest
    builder.add_rest(m1, duration=16, measure_rest=True, voice=1)

    # Later measure: no attributes needed
    m3 = builder.add_measure('P1', 3)
    builder.add_note(m3, step='F', octave=4, duration=8, duration_type='half',
                     voice=1, stem='down', tie_start=True, default_x=15)

    # Barline
    builder.add_barline(m3, location='right', style='light-heavy')

    # New-system print marker
    m13 = builder.add_measure('P1', 13)
    builder.add_print(m13, new_system=True)

    builder.save('output.xml')
"""

from lxml import etree
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

DPI              = 300
PIXELS_PER_MM    = DPI / 25.4          # 11.811 px/mm
MM_PER_TENTH     = 6.1807 / 40         # 0.154518 mm/tenth  (from target XMLs)

def pixels_to_tenths(pixels: float) -> float:
    """Convert pixel distance (300 DPI Finale export) to MusicXML tenths."""
    return round(pixels / PIXELS_PER_MM / MM_PER_TENTH, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Accidental helpers
# ─────────────────────────────────────────────────────────────────────────────

# Map from common string names → MusicXML <accidental> text and alter value
ACCIDENTAL_MAP = {
    'sharp':        ('sharp',        1),
    'flat':         ('flat',        -1),
    'natural':      ('natural',      0),
    'double-sharp': ('double-sharp', 2),
    'double-flat':  ('double-flat', -2),
    '#':            ('sharp',        1),
    'b':            ('flat',        -1),
    'n':            ('natural',      0),
}


# ─────────────────────────────────────────────────────────────────────────────
# XMLBuilder
# ─────────────────────────────────────────────────────────────────────────────

class XMLBuilder:
    """
    Builds a MusicXML score-partwise document by direct XML construction.

    All setter methods return the element they created so callers can chain
    further operations if needed.
    """

    def __init__(self):
        self._root = etree.Element('score-partwise', version='3.0')
        self._part_list_el = etree.SubElement(self._root, 'part-list')
        # part_id -> <part> element
        self._parts: dict[str, etree._Element] = {}
        # part_id -> last <measure> element (for convenience)
        self._last_measure: dict[str, etree._Element] = {}

    # ─────────────────────────────────────────────
    # Document-level metadata  (all optional)
    # ─────────────────────────────────────────────

    def set_defaults(self,
                     millimeters: float = 6.1807,
                     tenths: float = 40,
                     page_height: Optional[float] = None,
                     page_width: Optional[float] = None):
        """
        Add a <defaults> block with scaling (and optionally page layout).
        Call once before adding parts if you want layout info preserved.
        """
        defaults = etree.SubElement(self._root, 'defaults')
        scaling  = etree.SubElement(defaults, 'scaling')
        _sub(scaling, 'millimeters', str(millimeters))
        _sub(scaling, 'tenths',      str(tenths))
        if page_height is not None or page_width is not None:
            pl = etree.SubElement(defaults, 'page-layout')
            if page_height is not None:
                _sub(pl, 'page-height', str(page_height))
            if page_width is not None:
                _sub(pl, 'page-width', str(page_width))
        return defaults

    # ─────────────────────────────────────────────
    # PART
    # ─────────────────────────────────────────────

    def add_part(self,
                 part_id:         str,
                 part_name:       Optional[str] = None,
                 instrument_name: Optional[str] = None,
                 midi_channel:    Optional[int] = None,
                 midi_program:    Optional[int] = None):
        """
        Register a part.

        part_id         — unique string, e.g. 'P1'
        part_name       — printed name (optional; omit for unnamed parts)
        instrument_name — e.g. 'Clarinet' (optional)
        midi_channel    — 1-based (optional)
        midi_program    — 1-based GM program number (optional)

        Adds a <score-part> to <part-list> and an empty <part> to the score.
        """
        sp = etree.SubElement(self._part_list_el, 'score-part', id=part_id)

        if part_name is not None:
            pn = etree.SubElement(sp, 'part-name')
            pn.set('print-object', 'no')
            pn.text = part_name

        if instrument_name is not None:
            inst_id = f'{part_id}-I1'
            si = etree.SubElement(sp, 'score-instrument', id=inst_id)
            _sub(si, 'instrument-name', instrument_name)
            if midi_channel is not None or midi_program is not None:
                mi = etree.SubElement(sp, 'midi-instrument', id=inst_id)
                if midi_channel is not None:
                    _sub(mi, 'midi-channel', str(midi_channel))
                if midi_program is not None:
                    _sub(mi, 'midi-program', str(midi_program))

        part_el = etree.SubElement(self._root, 'part', id=part_id)
        self._parts[part_id] = part_el
        return sp

    # ─────────────────────────────────────────────
    # MEASURE
    # ─────────────────────────────────────────────

    def add_measure(self,
                    part_id:            str,
                    number:             int,
                    width:              Optional[float] = None,
                    # <attributes> sub-elements — all optional
                    divisions:          Optional[int]   = None,
                    key_fifths:         Optional[int]   = None,   # neg=flats
                    key_mode:           Optional[str]   = None,   # 'major'/'minor'
                    time_beats:         Optional[int]   = None,
                    time_beat_type:     Optional[int]   = None,
                    clef_sign:          Optional[str]   = None,   # 'G','F','C','percussion'
                    clef_line:          Optional[int]   = None,   # 1-5
                    clef_octave_change: Optional[int]   = None,   # -1, 1, etc.
                    staves:             Optional[int]   = None):  # number of staves
        """
        Add a <measure> to the named part.

        Only pass the attributes that actually appear in this measure.
        Returns the <measure> element — pass it to add_note(), add_rest(), etc.

        The <attributes> block is only written when at least one attribute
        argument is provided.
        """
        attrs = {'number': str(number)}
        if width is not None:
            attrs['width'] = str(width)

        m = etree.SubElement(self._parts[part_id], 'measure', **attrs)
        self._last_measure[part_id] = m

        # Build <attributes> only if something was specified
        has_attrs = any(v is not None for v in [
            divisions, key_fifths, key_mode,
            time_beats, time_beat_type,
            clef_sign, clef_line, clef_octave_change,
            staves
        ])

        if has_attrs:
            at = etree.SubElement(m, 'attributes')
            if divisions is not None:
                _sub(at, 'divisions', str(divisions))
            if key_fifths is not None:
                k = etree.SubElement(at, 'key')
                _sub(k, 'fifths', str(key_fifths))
                if key_mode is not None:
                    _sub(k, 'mode', key_mode)
            if time_beats is not None and time_beat_type is not None:
                t = etree.SubElement(at, 'time')
                _sub(t, 'beats',     str(time_beats))
                _sub(t, 'beat-type', str(time_beat_type))
            if clef_sign is not None:
                cl = etree.SubElement(at, 'clef')
                _sub(cl, 'sign', clef_sign)
                if clef_line is not None:
                    _sub(cl, 'line', str(clef_line))
                if clef_octave_change is not None:
                    _sub(cl, 'clef-octave-change', str(clef_octave_change))
            if staves is not None:
                _sub(at, 'staves', str(staves))

        return m

    # ─────────────────────────────────────────────
    # PRINT  (layout markers)
    # ─────────────────────────────────────────────

    def add_print(self,
                  measure,
                  new_system:           bool = False,
                  new_page:             bool = False,
                  top_system_distance:  Optional[float] = None,
                  system_distance:      Optional[float] = None,
                  measure_numbering:    Optional[str]   = None):
        """
        Add a <print> element to a measure.

        new_system / new_page        — set the corresponding attribute
        top_system_distance          — tenths from top of page to top of system
        system_distance              — tenths between systems
        measure_numbering            — e.g. 'none', 'measure', 'system'
        """
        attrs = {}
        if new_system:
            attrs['new-system'] = 'yes'
        if new_page:
            attrs['new-page'] = 'yes'

        pr = etree.SubElement(measure, 'print', **attrs)

        if top_system_distance is not None or system_distance is not None:
            sl = etree.SubElement(pr, 'system-layout')
            if top_system_distance is not None:
                _sub(sl, 'top-system-distance', str(top_system_distance))
            if system_distance is not None:
                _sub(sl, 'system-distance', str(system_distance))

        if measure_numbering is not None:
            _sub(pr, 'measure-numbering', measure_numbering)

        return pr

    # ─────────────────────────────────────────────
    # SOUND / TEMPO
    # ─────────────────────────────────────────────

    def add_sound(self, measure, tempo: Optional[float] = None, **kwargs):
        """
        Add a <sound> element.  Pass tempo=96 for tempo marking.
        Any extra keyword args become XML attributes (e.g. dynamics=80).
        """
        attrs = {}
        if tempo is not None:
            attrs['tempo'] = str(tempo)
        attrs.update({k: str(v) for k, v in kwargs.items()})
        return etree.SubElement(measure, 'sound', **attrs)

    # ─────────────────────────────────────────────
    # NOTE
    # ─────────────────────────────────────────────

    def add_note(self,
                 measure,
                 # Pitch (omit all three for a rest — use add_rest instead)
                 step:          Optional[str]   = None,   # 'C'..'B'
                 octave:        Optional[int]   = None,
                 alter:         Optional[float] = None,   # semitones: -1=flat, 1=sharp
                 accidental:    Optional[str]   = None,   # 'sharp','flat','natural',…
                 # Rhythm
                 duration:      Optional[int]   = None,   # raw divisions value
                 duration_type: Optional[str]   = None,   # 'whole','half','quarter',…
                 dot:           bool            = False,
                 # Ties
                 tie_start:     bool            = False,
                 tie_stop:      bool            = False,
                 # Voice / staff
                 voice:         Optional[int]   = None,   # usually 1
                 staff:         Optional[int]   = None,   # for multi-staff parts
                 # Stem
                 stem:          Optional[str]   = None,   # 'up' or 'down'
                 stem_default_y: Optional[float] = None,
                 # Position
                 default_x:    Optional[float]  = None,
                 default_y:    Optional[float]  = None,
                 # Chord (note shares onset with previous note)
                 chord:         bool            = False,
                 # Beam  (list of ('begin'|'continue'|'end', beam_number) tuples)
                 beams:         Optional[list]  = None):
        """
        Add a <note> to a measure.

        All parameters are optional — only the ones you pass will appear in the
        output XML.

        alter       — semitone alteration on the pitch (-1 = flat, 1 = sharp).
                      Use this OR accidental, not both.
        accidental  — named string: 'sharp', 'flat', 'natural', 'double-sharp',
                      'double-flat'.  Writes both <alter> and <accidental>.
        beams       — list of tuples: [('begin', 1), ('end', 1)]
                      beam_number is 1 for eighth, 2 for 16th, etc.
        chord       — True to mark this note as part of a chord with the
                      previous note (inserts <chord/> element).

        Returns the <note> element.
        """
        note_attrs = {}
        if default_x is not None:
            note_attrs['default-x'] = str(default_x)
        if default_y is not None:
            note_attrs['default-y'] = str(default_y)

        n = etree.SubElement(measure, 'note', **note_attrs)

        if chord:
            etree.SubElement(n, 'chord')

        # Pitch
        if step is not None and octave is not None:
            pitch = etree.SubElement(n, 'pitch')
            _sub(pitch, 'step', step.upper())
            # resolve alter from accidental name if provided
            actual_alter = alter
            acc_text = None
            if accidental is not None:
                info = ACCIDENTAL_MAP.get(accidental.lower())
                if info:
                    acc_text, actual_alter = info
            if actual_alter is not None and actual_alter != 0:
                _sub(pitch, 'alter', str(int(actual_alter)
                                         if actual_alter == int(actual_alter)
                                         else actual_alter))
            _sub(pitch, 'octave', str(octave))

        # Tie (element level — also need <notations> below)
        if tie_start:
            etree.SubElement(n, 'tie', type='start')
        if tie_stop:
            etree.SubElement(n, 'tie', type='stop')

        # Duration
        if duration is not None:
            _sub(n, 'duration', str(duration))

        # Voice
        if voice is not None:
            _sub(n, 'voice', str(voice))

        # Type
        if duration_type is not None:
            _sub(n, 'type', duration_type)

        # Dot
        if dot:
            etree.SubElement(n, 'dot')

        # Accidental text
        if acc_text is not None:
            _sub(n, 'accidental', acc_text)

        # Stem
        if stem is not None:
            stem_attrs = {}
            if stem_default_y is not None:
                stem_attrs['default-y'] = str(stem_default_y)
            s = etree.SubElement(n, 'stem', **stem_attrs)
            s.text = stem

        # Staff
        if staff is not None:
            _sub(n, 'staff', str(staff))

        # Beams
        if beams:
            for beam_type, beam_num in beams:
                b = etree.SubElement(n, 'beam', number=str(beam_num))
                b.text = beam_type

        # Notations (ties, slurs, etc.)
        if tie_start or tie_stop:
            notations = etree.SubElement(n, 'notations')
            if tie_stop:
                etree.SubElement(notations, 'tied', type='stop')
            if tie_start:
                etree.SubElement(notations, 'tied', type='start')

        return n

    # ─────────────────────────────────────────────
    # REST
    # ─────────────────────────────────────────────

    def add_rest(self,
                 measure,
                 duration:      Optional[int]  = None,
                 duration_type: Optional[str]  = None,
                 dot:           bool           = False,
                 measure_rest:  bool           = False,
                 voice:         Optional[int]  = None,
                 staff:         Optional[int]  = None,
                 default_x:    Optional[float] = None,
                 default_y:    Optional[float] = None):
        """
        Add a <note><rest/></note> element.

        measure_rest=True  — adds measure="yes" attribute to <rest> (whole-bar rest).
        All other params mirror add_note().
        """
        note_attrs = {}
        if default_x is not None:
            note_attrs['default-x'] = str(default_x)
        if default_y is not None:
            note_attrs['default-y'] = str(default_y)

        n = etree.SubElement(measure, 'note', **note_attrs)
        rest_attrs = {'measure': 'yes'} if measure_rest else {}
        etree.SubElement(n, 'rest', **rest_attrs)

        if duration is not None:
            _sub(n, 'duration', str(duration))
        if voice is not None:
            _sub(n, 'voice', str(voice))
        if duration_type is not None:
            _sub(n, 'type', duration_type)
        if dot:
            etree.SubElement(n, 'dot')
        if staff is not None:
            _sub(n, 'staff', str(staff))

        return n

    # ─────────────────────────────────────────────
    # CHORD NOTE  (shorthand)
    # ─────────────────────────────────────────────

    def add_chord_note(self, measure, pitches: list, duration: Optional[int] = None,
                       duration_type: Optional[str] = None, dot: bool = False,
                       voice: Optional[int] = None, stem: Optional[str] = None,
                       default_x: Optional[float] = None):
        """
        Add multiple simultaneous notes (chord) using add_note() with chord=True.

        pitches — list of (step, octave) or (step, octave, alter) tuples.
                  e.g. [('C', 4), ('E', 4, 1), ('G', 4)]

        The first note is added normally; subsequent notes get chord=True.
        Returns list of note elements.
        """
        notes = []
        for i, p in enumerate(pitches):
            step   = p[0]
            octave = p[1]
            alter  = p[2] if len(p) > 2 else None
            notes.append(self.add_note(
                measure,
                step=step, octave=octave, alter=alter,
                duration=duration, duration_type=duration_type, dot=dot,
                voice=voice, stem=stem, default_x=default_x,
                chord=(i > 0)
            ))
        return notes

    # ─────────────────────────────────────────────
    # BARLINE
    # ─────────────────────────────────────────────

    def add_barline(self,
                    measure,
                    location: str = 'right',
                    style:    Optional[str] = None,   # 'light-heavy', 'light-light', …
                    repeat:   Optional[str] = None,   # 'forward', 'backward'
                    repeat_winged: Optional[str] = None):
        """
        Add a <barline> to a measure.

        location — 'left' or 'right' (default 'right')
        style    — bar-style text, e.g. 'light-heavy'
        repeat   — direction: 'forward' or 'backward'
        """
        bl = etree.SubElement(measure, 'barline', location=location)
        if style is not None:
            _sub(bl, 'bar-style', style)
        if repeat is not None:
            rep_attrs = {'direction': repeat}
            if repeat_winged is not None:
                rep_attrs['winged'] = repeat_winged
            etree.SubElement(bl, 'repeat', **rep_attrs)
        return bl

    # ─────────────────────────────────────────────
    # DIRECTION / TEMPO TEXT
    # ─────────────────────────────────────────────

    def add_direction(self,
                      measure,
                      text:      Optional[str]   = None,
                      tempo_bpm: Optional[float] = None,
                      placement: str             = 'above',
                      default_y: Optional[float] = None):
        """
        Add a <direction> element (e.g. tempo text like 'Maestoso').
        For a pure tempo value with no text, use add_sound() instead.
        """
        d = etree.SubElement(measure, 'direction', placement=placement)
        if text is not None:
            dt = etree.SubElement(d, 'direction-type')
            w_attrs = {}
            if default_y is not None:
                w_attrs['default-y'] = str(default_y)
            w = etree.SubElement(dt, 'words', **w_attrs)
            w.text = text
        if tempo_bpm is not None:
            etree.SubElement(d, 'sound', tempo=str(tempo_bpm))
        return d

    # ─────────────────────────────────────────────
    # PART-GROUP  (bracket over multiple parts)
    # ─────────────────────────────────────────────

    def add_part_group(self, number: int = 1, group_type: str = 'start',
                       symbol: str = 'bracket', group_barline: bool = True):
        """
        Add a <part-group> to the part-list (e.g. bracket over all clarinets).
        Call with group_type='start' before adding parts, 'stop' after.
        """
        pg = etree.SubElement(self._part_list_el, 'part-group',
                               number=str(number), type=group_type)
        if group_type == 'start':
            _sub(pg, 'group-symbol', symbol)
            _sub(pg, 'group-barline', 'yes' if group_barline else 'no')
        return pg

    # ─────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────

    def save(self, output_path: str):
        """
        Write the score to a MusicXML file with proper DOCTYPE declaration.

        output_path — full path ending in .xml, e.g. r'S:\\omr\\output.xml'
        """
        doctype = (
            '<!DOCTYPE score-partwise PUBLIC '
            '"-//Recordare//DTD MusicXML 3.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/3.0/partwise.dtd">'
        )
        xml_bytes = etree.tostring(
            self._root,
            pretty_print=True,
            xml_declaration=True,
            encoding='UTF-8',
            doctype=doctype
        )
        with open(output_path, 'wb') as f:
            f.write(xml_bytes)
        print(f'Saved → {output_path}')

    def to_string(self) -> str:
        """Return the XML as a UTF-8 string (useful for debugging)."""
        return etree.tostring(self._root, pretty_print=True).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Private helper
# ─────────────────────────────────────────────────────────────────────────────

def _sub(parent, tag: str, text: str) -> etree._Element:
    """Create a child element with text content."""
    el = etree.SubElement(parent, tag)
    el.text = text
    return el


# ─────────────────────────────────────────────────────────────────────────────
# Example — mirrors 01_Okribuli_makruli.xml structure
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    builder = XMLBuilder()

    # Optional document defaults
    builder.set_defaults(millimeters=6.1807, tenths=40,
                         page_height=1921, page_width=1358)

    # Group bracket around all three clarinets
    builder.add_part_group(number=1, group_type='start', symbol='bracket')

    # Three parts — names are optional
    builder.add_part('P1', instrument_name='Clarinet',   midi_channel=1, midi_program=72)
    builder.add_part('P2', instrument_name='Clarinet 2', midi_channel=2, midi_program=72)
    builder.add_part('P3', instrument_name='Clarinet 3', midi_channel=3, midi_program=72)

    builder.add_part_group(number=1, group_type='stop')

    # ── Measure 1 — full attributes + whole-measure rests ──
    m1_p1 = builder.add_measure('P1', 1, width=493,
                                 divisions=4,
                                 key_fifths=-1, key_mode='major',
                                 time_beats=4, time_beat_type=4,
                                 clef_sign='G', clef_line=2, clef_octave_change=-1)
    builder.add_print(m1_p1, top_system_distance=331, measure_numbering='none')
    builder.add_sound(m1_p1, tempo=96)
    builder.add_rest(m1_p1, duration=16, measure_rest=True, voice=1)

    m1_p2 = builder.add_measure('P2', 1, width=493,
                                 divisions=4,
                                 key_fifths=-1, key_mode='major',
                                 time_beats=4, time_beat_type=4,
                                 clef_sign='G', clef_line=2, clef_octave_change=-1)
    builder.add_print(m1_p2, top_system_distance=331, measure_numbering='none')
    builder.add_rest(m1_p2, duration=16, measure_rest=True, voice=1)

    m1_p3 = builder.add_measure('P3', 1, width=493,
                                 divisions=4,
                                 key_fifths=-1, key_mode='major',
                                 time_beats=4, time_beat_type=4,
                                 clef_sign='G', clef_line=2, clef_octave_change=-1)
    builder.add_print(m1_p3, top_system_distance=331, measure_numbering='none')
    builder.add_rest(m1_p3, duration=16, measure_rest=True, voice=1)

    # ── Measure 3 — notes with ties ──
    # m3 = builder.add_measure('P1', 2, width=317)
    # builder.add_note(m3, step='F', octave=4, duration=8, duration_type='half',
    #                  voice=1, stem='down', stem_default_y=-32.5,
    #                  tie_start=True, default_x=15)
    # builder.add_note(m3, step='F', octave=4, duration=2, duration_type='eighth',
    #                  voice=1, stem='down', stem_default_y=-32.5,
    #                  tie_stop=True, default_x=115)
    # builder.add_note(m3, step='F', octave=4, duration=2, duration_type='eighth',
    #                  voice=1, stem='down', stem_default_y=-32.5, default_x=165)

    # ── Measure with final barline ──
    # m_last = builder.add_measure('P1', 3, width=275)
    # builder.add_note(m_last, step='B', octave=3, alter=-1,
    #                  duration=4, duration_type='quarter',
    #                  voice=1, stem='down', stem_default_y=-28,
    #                  tie_stop=True, default_x=15)
    # builder.add_rest(m_last, duration=4, duration_type='quarter', voice=1, default_x=88)
    # builder.add_rest(m_last, duration=8, duration_type='half',    voice=1, default_x=160)
    # builder.add_barline(m_last, location='right', style='light-heavy')

    builder.save(r'S:\omr\test_output.xml')
