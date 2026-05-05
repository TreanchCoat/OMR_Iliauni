# OMR API — Usage Guide

This document covers how to call the OMR API from a frontend or backend
service. Both the **dummy** API (returns hardcoded fixtures, useful for
frontend development) and the **real** pipeline expose the same response
shapes, so code written against one works against the other unchanged.

---

## Quick reference

| Endpoint | Method | Returns | Use case |
|---|---|---|---|
| `/rectified` | GET | PNG image | Display the straightened score |
| `/detections` | GET | JSON | Per-symbol bounding boxes + classes |
| `/xml` | GET | MusicXML file | Final score document |
| `/full` | GET | JSON | All three combined in one response |
| `/health` | GET | JSON | Sanity check |

Default base URL when running locally: `http://localhost:5000`

---

## Running the server

```bash
pip install -r requirements.txt
python dummy_api.py
```

Server listens on port `5000` by default. Override host/port by editing the
`app.run(...)` call at the bottom of `dummy_api.py`.

---

## Endpoints in detail

### `GET /rectified`

Returns the rectified-page PNG (the OMR pipeline's straightened version of the
input image). This is the image whose coordinate system the detection JSON
and the MusicXML coordinates refer to.

**Response**
- `Content-Type: image/png`
- Body: raw PNG bytes
- `Content-Disposition: attachment; filename="rectified.png"`

**Example**
```bash
curl http://localhost:5000/rectified -o rectified.png
```

```javascript
// Browser
const img = new Image();
img.src = 'http://localhost:5000/rectified';
document.body.appendChild(img);
```

---

### `GET /detections`

Returns the full set of YOLO detections grouped by staff. Coordinates are
provided in two systems:

- `cx`, `cy`, `x1..y2` — within the staff crop (0,0 = top-left of crop)
- `full_cx`, `full_cy` — within the rectified page (0,0 = top-left of page)

**Response shape**
```json
[
  {
    "part_id":          "P1",
    "staff_in_part":    0,
    "top_y":            307,
    "bot_y":            379,
    "line_positions":   [307, 324, 342, 361, 379],
    "line_spacing":     18.0,
    "crop_y1":          239,
    "total_detections": 26,
    "detections": [
      {
        "class_name": "clefG",
        "conf":       0.94,
        "cx": 280, "cy": 102,
        "x1": 257, "y1": 40, "x2": 304, "y2": 165,
        "full_cx": 280, "full_cy": 341
      }
    ]
  }
]
```

**Field meanings**
- `part_id` — part identifier (`P1`, `P2`, ...) matching the `<part id>` in the MusicXML
- `staff_in_part` — 0-based staff index within the part (one staff per system)
- `line_positions` — five staff line y-coordinates in the rectified page
- `line_spacing` — vertical pixel distance between adjacent staff lines
- `crop_y1` — top edge of the staff crop in the rectified page (`full_cy = cy + crop_y1`)
- `class_name` — DeepScores class name (e.g. `clefG`, `noteheadBlack`, `slur`, `beam`)
- `conf` — detection confidence in [0, 1]

---

### `GET /xml`

Returns the assembled MusicXML document.

**Response**
- `Content-Type: application/xml`
- `Content-Disposition: attachment; filename="<source>.xml"`

The XML embeds every detection's absolute coordinates inside
`<identification><miscellaneous>`, and every `<note>` element has an `id`
attribute matching the corresponding entry. See **Cross-referencing**
below.

---

### `GET /full`

Returns all three outputs in a single JSON envelope. Useful when you want a
single round-trip and don't mind base64-encoding the image.

**Response shape**
```json
{
  "rectified_image_b64":  "<base64-encoded PNG>",
  "rectified_image_mime": "image/png",
  "detections":           [ ... same as /detections ... ],
  "xml":                  "<MusicXML string>"
}
```

**Example (browser)**
```javascript
const res = await fetch('http://localhost:5000/full');
const data = await res.json();

// Render the image
document.getElementById('score').src =
  `data:${data.rectified_image_mime};base64,${data.rectified_image_b64}`;

// Iterate detections
for (const staff of data.detections) {
  for (const det of staff.detections) {
    drawBox(det.full_cx, det.full_cy, det.class_name);
  }
}
```

---

### `GET /health`

Confirms the server is running and reports whether each fixture file exists.
Useful for monitoring and debugging fixture paths.

```json
{
  "status": "ok",
  "fixtures": {
    "rectified":  { "path": "...", "exists": true },
    "detections": { "path": "...", "exists": true },
    "xml":        { "path": "...", "exists": true }
  }
}
```

---

## Cross-referencing notes ↔ coordinates in the XML

Every `<note>` in the MusicXML has an `id` attribute (e.g. `id="det_0042"`).
The `<miscellaneous>` block contains a JSON array under the
`omr-coordinates` field with one entry per detection. Match by `id`.

```python
import xml.etree.ElementTree as ET
import json

tree = ET.parse('score.xml')
root = tree.getroot()

# Pull the coordinate JSON
misc_fields = root.findall('.//miscellaneous-field')
coords_field = next(f for f in misc_fields if f.get('name') == 'omr-coordinates')
coords_by_id = {rec['id']: rec for rec in json.loads(coords_field.text)}

# Walk every note and look up its coordinates
for note in root.findall('.//note[@id]'):
    rec = coords_by_id[note.get('id')]
    print(f"Note {note.get('id')}: full image ({rec['cx']}, {rec['cy']})")
```

The same map also includes every non-note symbol (slurs, beams, fermatas,
accidentals, etc.), so the rendering app can lay out the entire score, not
just the notes.

The XML also contains these auxiliary `<miscellaneous-field>` entries:
- `omr-version` — pipeline version string
- `omr-source-image` — original image filename
- `omr-image-width`, `omr-image-height` — rectified image dimensions

---

## CORS

The dummy API does **not** enable CORS by default. If you're calling it from
a browser on a different origin, install `flask-cors` and add this near the
top of `_create_app()` in `dummy_api.py`:

```python
from flask_cors import CORS
CORS(app)
```

---

## Switching from dummy to real

The real pipeline is `pipeline.run_pipeline()`. To convert the dummy API to a
real one, replace the body of each `get_*()` function in `dummy_api.py`:

```python
from pipeline import run_pipeline
import tempfile

def _process(image_bytes):
    with tempfile.TemporaryDirectory() as tmp:
        in_path = f'{tmp}/in.png'
        with open(in_path, 'wb') as f:
            f.write(image_bytes)
        return run_pipeline(in_path, tmp, model_path='models/.../v1.pt')
```

Add a `POST /process` endpoint that accepts an uploaded image, runs
`run_pipeline`, then serves the same three artifacts from the result dict
(`result['rectified_image']`, `result['detections_json']`, `result['xml_file']`).

The response shapes stay identical — only the source of the data changes.

---

## Errors

The dummy API does not currently return error envelopes; missing fixture
files manifest as 500-class server errors. The real-pipeline replacement
should wrap `run_pipeline()` in a try/except and return:

```json
{
  "error":   "stage_2_no_staves",
  "message": "No staves detected in input image",
  "stage":   "preprocessing"
}
```

---

## Performance notes

- The dummy API responds in <50 ms; the real pipeline takes ~2-10 s
  per page on a CPU, ~0.5-2 s on a GPU (NVIDIA 3050 reference).
- For large pages, prefer separate `/rectified` + `/detections` + `/xml`
  calls over `/full` — the base64 image in `/full` adds ~33% to payload size.
- Detection JSON for a typical 9-staff page is ~50-200 KB.
- MusicXML with embedded coordinates is ~100-400 KB.
