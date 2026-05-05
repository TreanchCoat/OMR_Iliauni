"""
dummy_api.py — Stub API for the OMR pipeline.

Returns hardcoded responses that mirror what the real pipeline (pipeline.py)
will produce, so that downstream code (frontend, integration tests, etc.) can
be developed without running the full model.

Three outputs are exposed:
    1. Rectified image  — the straightened PNG that comes out of staff_rectifier
    2. Detection JSON   — list of staff dicts, one per staff
    3. MusicXML         — the final score document

Two interfaces
--------------
    Python API:
        from dummy_api import (get_rectified_image, get_detections,
                                get_xml, get_full_response)

    HTTP API (requires Flask):
        python dummy_api.py
        GET  /rectified           → rectified image (PNG download)
        GET  /detections          → detection JSON
        GET  /xml                 → MusicXML file download
        GET  /full                → all three in one JSON response
                                     ({ rectified_image_b64, detections, xml })
        GET  /health              → fixture-existence check

Swap-out contract
-----------------
When the real pipeline lands, replace the body of get_*() with:

    from pipeline import run_pipeline
    result = run_pipeline(image_path, tmp_dir, model_path)
    return result['rectified_image'], result['xml_file'], result['detections_json']

The response shapes must stay identical.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths to the hardcoded fixture files. Point these at your sample data.
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

_RECTIFIED_PATH      = Path(r'S:\omr\sample_data\rectified.png')
_DETECTION_JSON_PATH = Path(r'S:\omr\sample_data\P1_staff01.json')
_XML_PATH            = Path(r'S:\omr\sample_data\01_Okribuli_makruli.xml')

# Fallback: look next to this script (handy for a fresh repo clone)
if not _RECTIFIED_PATH.exists():
    _RECTIFIED_PATH = _HERE / 'rectified.png'
if not _DETECTION_JSON_PATH.exists():
    _DETECTION_JSON_PATH = _HERE / 'P1_staff01.json'
if not _XML_PATH.exists():
    _XML_PATH = _HERE / '01_Okribuli_makruli.xml'


# ─────────────────────────────────────────────────────────────────────────────
# Python API
# ─────────────────────────────────────────────────────────────────────────────

def get_rectified_image() -> bytes:
    """Return the rectified-page PNG as raw bytes."""
    with open(_RECTIFIED_PATH, 'rb') as f:
        return f.read()


def get_rectified_image_path() -> str:
    """Return the disk path to the rectified PNG."""
    return str(_RECTIFIED_PATH)


def get_detections() -> list:
    """
    Return detection data as a list of staff dicts.

    Each dict matches the schema written by symbol_detector.save_json().
    If the fixture is a single staff dict, it's wrapped in a list so the
    schema is identical to a real multi-staff response.
    """
    with open(_DETECTION_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def get_xml() -> str:
    """Return the MusicXML document as a UTF-8 string."""
    with open(_XML_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def get_full_response() -> dict:
    """
    Return all three outputs in one dict:

        {
            "rectified_image_b64":  "<base64 PNG>",
            "rectified_image_mime": "image/png",
            "detections":           [ ... ],
            "xml":                  "<MusicXML string>",
        }
    """
    img_bytes = get_rectified_image()
    mime, _ = mimetypes.guess_type(str(_RECTIFIED_PATH))
    return {
        'rectified_image_b64':  base64.b64encode(img_bytes).decode('ascii'),
        'rectified_image_mime': mime or 'image/png',
        'detections':           get_detections(),
        'xml':                  get_xml(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP API (Flask only imported when running as a server)
# ─────────────────────────────────────────────────────────────────────────────

def _create_app():
    try:
        from flask import Flask, jsonify, Response, send_file
    except ImportError:
        raise ImportError(
            'Flask is required to run the HTTP server.\n'
            'Install it with:  pip install flask'
        )

    app = Flask(__name__)

    @app.route('/rectified', methods=['GET'])
    def rectified_endpoint():
        """Return the rectified-page PNG."""
        return send_file(
            _RECTIFIED_PATH,
            mimetype='image/png',
            as_attachment=True,
            download_name=_RECTIFIED_PATH.name,
        )

    @app.route('/detections', methods=['GET'])
    def detections_endpoint():
        return jsonify(get_detections())

    @app.route('/xml', methods=['GET'])
    def xml_endpoint():
        return Response(
            get_xml(),
            mimetype='application/xml',
            headers={
                'Content-Disposition':
                    f'attachment; filename="{_XML_PATH.name}"'
            }
        )

    @app.route('/full', methods=['GET'])
    def full_endpoint():
        """All three outputs in one JSON response."""
        return jsonify(get_full_response())

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'ok',
            'fixtures': {
                'rectified':  {'path': str(_RECTIFIED_PATH),
                               'exists': _RECTIFIED_PATH.exists()},
                'detections': {'path': str(_DETECTION_JSON_PATH),
                               'exists': _DETECTION_JSON_PATH.exists()},
                'xml':        {'path': str(_XML_PATH),
                               'exists': _XML_PATH.exists()},
            }
        })

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Fixture paths:')
    print(f'  rectified  : {_RECTIFIED_PATH}      (exists={_RECTIFIED_PATH.exists()})')
    print(f'  detections : {_DETECTION_JSON_PATH} (exists={_DETECTION_JSON_PATH.exists()})')
    print(f'  xml        : {_XML_PATH}            (exists={_XML_PATH.exists()})')

    app = _create_app()
    print('\nDummy OMR API server starting …')
    print('  GET http://localhost:5000/rectified')
    print('  GET http://localhost:5000/detections')
    print('  GET http://localhost:5000/xml')
    print('  GET http://localhost:5000/full')
    print('  GET http://localhost:5000/health')
    app.run(host='0.0.0.0', port=5000, debug=True)
