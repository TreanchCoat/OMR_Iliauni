"""
api_client_example.py — Minimal client showing how to call the OMR API.

Demonstrates each endpoint:
    GET  /health             — sanity check
    GET  /rectified          — download rectified PNG
    GET  /detections         — fetch detection JSON
    GET  /xml                — download MusicXML
    GET  /full               — fetch all three in one response
    POST /process            — submit an image to the real pipeline

Works against either dummy_api.py or real_api.py — they share the same
response shapes.

Usage
-----
    # Start the API in another terminal:
    python dummy_api.py        # or:  python real_api.py

    # Then run this script:
    python api_client_example.py
    python api_client_example.py --base-url http://localhost:5000
    python api_client_example.py --process path/to/score.png
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import requests   # pip install requests


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint wrappers
# ─────────────────────────────────────────────────────────────────────────────

def health(base_url: str) -> dict:
    """GET /health — returns server status + fixture info."""
    r = requests.get(f'{base_url}/health', timeout=10)
    r.raise_for_status()
    return r.json()


def get_rectified(base_url: str, save_to: str = 'rectified.png') -> str:
    """GET /rectified — saves the PNG to disk and returns the path."""
    r = requests.get(f'{base_url}/rectified', timeout=30)
    r.raise_for_status()
    Path(save_to).write_bytes(r.content)
    return save_to


def get_detections(base_url: str) -> list:
    """GET /detections — returns a list of staff dicts."""
    r = requests.get(f'{base_url}/detections', timeout=30)
    r.raise_for_status()
    return r.json()


def get_xml(base_url: str, save_to: str = 'score.xml') -> str:
    """GET /xml — saves the MusicXML and returns the path."""
    r = requests.get(f'{base_url}/xml', timeout=30)
    r.raise_for_status()
    Path(save_to).write_text(r.text, encoding='utf-8')
    return save_to


def get_full(base_url: str) -> dict:
    """GET /full — returns { rectified_image_b64, detections, xml }."""
    r = requests.get(f'{base_url}/full', timeout=60)
    r.raise_for_status()
    return r.json()


def process_image(base_url: str, image_path: str,
                  save_dir: str = 'api_output') -> dict:
    """
    POST /process — submit an image, get all three outputs back.
    Saves rectified.png, detections.json, score.xml into save_dir.
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(image_path, 'rb') as f:
        files = {'image': (Path(image_path).name, f, 'image/png')}
        r = requests.post(f'{base_url}/process', files=files, timeout=300)
    r.raise_for_status()
    data = r.json()

    # Decode and save the rectified image
    img_bytes = base64.b64decode(data['rectified_image_b64'])
    (out / 'rectified.png').write_bytes(img_bytes)

    # Save detections + xml
    (out / 'detections.json').write_text(
        json.dumps(data['detections'], indent=2), encoding='utf-8')
    (out / 'score.xml').write_text(data['xml'], encoding='utf-8')

    return {
        'rectified_image':  str(out / 'rectified.png'),
        'detections_json':  str(out / 'detections.json'),
        'xml_file':         str(out / 'score.xml'),
        'detections':       data['detections'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarize_detections(staves: list):
    """Print a compact summary of detection results."""
    total_dets = sum(s['total_detections'] for s in staves)
    print(f'  Total: {total_dets} symbols across {len(staves)} staves')

    # Class counts
    counts: dict = {}
    for staff in staves:
        for det in staff['detections']:
            counts[det['class_name']] = counts.get(det['class_name'], 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    print('  Top classes:')
    for cls, n in top:
        print(f'    {cls:30s} {n}')


def lookup_note_coordinates(xml_path: str):
    """
    Cross-reference each <note id="..."> in the MusicXML against the
    embedded coordinate JSON in <miscellaneous>.  Prints the first 10.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the omr-coordinates miscellaneous-field
    misc = root.findall('.//miscellaneous-field')
    coord_field = next((f for f in misc if f.get('name') == 'omr-coordinates'),
                       None)
    if coord_field is None:
        print('  (No omr-coordinates field found in XML)')
        return

    coords = json.loads(coord_field.text)
    coord_by_id = {rec['id']: rec for rec in coords}

    print(f'  {len(coords)} symbols in coordinate map')
    print('  Note ID → (cx, cy) in rectified image:')
    notes_with_id = root.findall('.//note[@id]')
    for note in notes_with_id[:10]:
        nid = note.get('id')
        rec = coord_by_id.get(nid)
        if rec is not None:
            pitch = note.find('pitch')
            if pitch is not None:
                step = pitch.findtext('step')
                octv = pitch.findtext('octave')
                pitch_str = f'{step}{octv}'
            else:
                pitch_str = 'rest'
            print(f'    {nid:10s}  {pitch_str:5s}  ({rec["cx"]}, {rec["cy"]})')

    if len(notes_with_id) > 10:
        print(f'    … {len(notes_with_id) - 10} more')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:5000',
                        help='API base URL (default: http://localhost:5000)')
    parser.add_argument('--out-dir', default='api_demo_out',
                        help='Where to save downloaded files')
    parser.add_argument('--process', metavar='IMAGE',
                        help='Image to submit via POST /process. '
                             'Required when talking to real_api.py; '
                             'optional for dummy_api.py.')
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Health check — also tells us which server we're talking to ──
    print('1. Health check')
    try:
        h = health(args.base_url)
        print(f'   Server status : {h.get("status")}')
        model_info = h.get('pipeline')
        is_real_api = model_info is not None
        if is_real_api:
            print(f'   Model exists  : {model_info.get("model_exists")}')
            has_cache = h.get('last_result', {}).get('has_cached_result', False)
            print(f'   Has cache     : {has_cache}')
        else:
            has_cache = True    # dummy always serves fixtures
    except requests.RequestException as e:
        print(f'   FAILED: {e}')
        print(f'   Is the server running at {args.base_url}?')
        sys.exit(1)

    # ── 2. POST /process if needed ──
    # real_api GET endpoints return 404 until /process has been called.
    # If --process was passed we always submit; otherwise warn and skip GETs.
    if args.process:
        print(f'\n2. POST /process  ({args.process})')
        try:
            result = process_image(args.base_url, args.process,
                                   save_dir=str(out / 'processed'))
            print(f'   Rectified  : {result["rectified_image"]}')
            print(f'   Detections : {result["detections_json"]}')
            print(f'   MusicXML   : {result["xml_file"]}')
            summarize_detections(result['detections'])
            has_cache = True
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print('   /process not available on this server.')
            else:
                raise
    elif is_real_api and not has_cache:
        print('\nNOTE: real_api.py has no cached result yet.')
        print('      The GET endpoints return 404 until an image is processed.')
        print('      Re-run with --process to submit one:')
        print(f'        python api_client_example.py --process score.png')
        sys.exit(0)

    # ── 3. GET /rectified ──
    print('\n3. GET /rectified')
    try:
        path = get_rectified(args.base_url, save_to=str(out / 'rectified.png'))
        print(f'   Saved to {path}')
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    # ── 4. GET /detections ──
    print('\n4. GET /detections')
    try:
        staves = get_detections(args.base_url)
        summarize_detections(staves)
        (out / 'detections.json').write_text(
            json.dumps(staves, indent=2), encoding='utf-8')
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    # ── 5. GET /xml ──
    print('\n5. GET /xml')
    try:
        xml_path = get_xml(args.base_url, save_to=str(out / 'score.xml'))
        print(f'   Saved to {xml_path}')
        print('   Cross-referencing notes ↔ coordinates:')
        lookup_note_coordinates(xml_path)
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    # ── 6. GET /full ──
    print('\n6. GET /full')
    try:
        full = get_full(args.base_url)
        img_bytes = base64.b64decode(full['rectified_image_b64'])
        print(f'   Got {len(img_bytes)} bytes of PNG, '
              f'{len(full["detections"])} staves, '
              f'{len(full["xml"])} chars of XML')
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    print(f'\nAll outputs saved under: {out}')


if __name__ == '__main__':
    main()
