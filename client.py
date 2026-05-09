"""
client.py — Minimal client showing how to call the OMR API.

Demonstrates each endpoint:
    GET  /api/v1/health             — sanity check
    GET  /api/v1/rectified          — download rectified PNG
    GET  /api/v1/detections         — fetch detection JSON
    GET  /api/v1/xml                — download MusicXML
    GET  /api/v1/full               — fetch all three in one response
    POST /api/v1/process            — submit an image to the real pipeline

Usage
-----
    # Start the API in another terminal:
    python api/dummy.py        # or:  python api/main.py

    # Then run this script:
    python api/client.py
    python api/client.py --base-url http://localhost:5000
    python api/client.py --process path/to/score.png
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
    """GET /api/v1/health — returns server status."""
    r = requests.get(f'{base_url}/api/v1/health', timeout=10)
    r.raise_for_status()
    return r.json()


def get_rectified(base_url: str, token: str, save_to: str = 'rectified.png') -> str:
    """GET /api/v1/rectified — saves the PNG to disk and returns the path."""
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(f'{base_url}/api/v1/rectified', headers=headers, timeout=30)
    r.raise_for_status()
    Path(save_to).write_bytes(r.content)
    return save_to


def get_detections(base_url: str, token: str) -> list:
    """GET /api/v1/detections — returns a list of staff dicts."""
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(f'{base_url}/api/v1/detections', headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def get_xml(base_url: str, token: str, save_to: str = 'score.xml') -> str:
    """GET /api/v1/xml — saves the MusicXML and returns the path."""
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(f'{base_url}/api/v1/xml', headers=headers, timeout=30)
    r.raise_for_status()
    Path(save_to).write_text(r.text, encoding='utf-8')
    return save_to


def get_full(base_url: str, token: str) -> dict:
    """GET /api/v1/full — returns { rectified_image_b64, detections, xml }."""
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(f'{base_url}/api/v1/full', headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def process_image(base_url: str, token: str, image_path: str,
                  save_dir: str = 'api_output') -> dict:
    """
    POST /api/v1/process — submit an image, get all three outputs back.
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    headers = {'Authorization': f'Bearer {token}'}
    with open(image_path, 'rb') as f:
        files = {'image': (Path(image_path).name, f, 'image/png')}
        r = requests.post(f'{base_url}/api/v1/process', headers=headers, files=files, timeout=300)
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
    parser.add_argument('--token', default='supersecrettoken',
                        help='API Bearer Token (default: supersecrettoken)')
    parser.add_argument('--out-dir', default='api_demo_out',
                        help='Where to save downloaded files')
    parser.add_argument('--process', metavar='IMAGE',
                        help='Image to submit via POST /api/v1/process. ')
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Health check ──
    print('1. Health check')
    try:
        h = health(args.base_url)
        print(f'   Server status : {h.get("status")}')
        print(f'   Server mode   : {h.get("mode")}')
    except requests.RequestException as e:
        print(f'   FAILED: {e}')
        print(f'   Is the server running at {args.base_url}?')
        sys.exit(1)

    # ── 2. POST /api/v1/process ──
    if args.process:
        print(f'\n2. POST /api/v1/process  ({args.process})')
        try:
            result = process_image(args.base_url, args.token, args.process,
                                   save_dir=str(out / 'processed'))
            print(f'   Rectified  : {result["rectified_image"]}')
            print(f'   Detections : {result["detections_json"]}')
            print(f'   MusicXML   : {result["xml_file"]}')
            summarize_detections(result['detections'])
        except requests.HTTPError as e:
            print(f'   FAILED: {e}')
    else:
        print('\nSkip 2. POST /api/v1/process (no --process arg)')

    # ── 3. GET /api/v1/rectified ──
    print('\n3. GET /api/v1/rectified')
    try:
        path = get_rectified(args.base_url, args.token, save_to=str(out / 'rectified.png'))
        print(f'   Saved to {path}')
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    # ── 4. GET /api/v1/detections ──
    print('\n4. GET /api/v1/detections')
    try:
        staves = get_detections(args.base_url, args.token)
        summarize_detections(staves)
        (out / 'detections.json').write_text(
            json.dumps(staves, indent=2), encoding='utf-8')
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    # ── 5. GET /api/v1/xml ──
    print('\n5. GET /api/v1/xml')
    try:
        xml_path = get_xml(args.base_url, args.token, save_to=str(out / 'score.xml'))
        print(f'   Saved to {xml_path}')
        print('   Cross-referencing notes ↔ coordinates:')
        lookup_note_coordinates(xml_path)
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    # ── 6. GET /api/v1/full ──
    print('\n6. GET /api/v1/full')
    try:
        full = get_full(args.base_url, args.token)
        img_bytes = base64.b64decode(full['rectified_image_b64'])
        print(f'   Got {len(img_bytes)} bytes of PNG, '
              f'{len(full["detections"])} staves, '
              f'{len(full["xml"])} chars of XML')
    except requests.HTTPError as e:
        print(f'   Skipped ({e.response.status_code})')

    print(f'\nAll outputs saved under: {out}')


if __name__ == '__main__':
    main()
