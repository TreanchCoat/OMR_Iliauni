"""
dummy_api.py — Dummy version of real_api.py for testing without the model.

Mirrors real_api.py's interface exactly — same endpoints, same request/
response shapes, same browser tester at /, same CORS config — but always
returns the same hardcoded fixture image and MusicXML regardless of input.

Use this when:
  • Developing a frontend before the model is trained
  • Testing UI changes without burning GPU time
  • Demoing the API with predictable output
  • Running on a machine that doesn't have torch / ultralytics installed

Endpoints
---------
    GET  /                   — browser API tester UI
    POST /process            — accepts an image, returns the fixture response
    GET  /rectified          — returns the fixture rectified PNG
    GET  /xml                — returns the fixture MusicXML
    GET  /full               — returns both in one JSON envelope
    GET  /detections         — returns the fixture detections JSON
    GET  /health             — readiness check

Configuration
-------------
Set fixture paths via environment variables:
    DUMMY_RECTIFIED_PATH    path to the PNG to always return
    DUMMY_XML_PATH          path to the MusicXML to always return
    DUMMY_DETECTIONS_PATH   path to the detection JSON to always return

If unset, defaults to files alongside this script:
    sample_data/rectified.png
    sample_data/score.xml
    sample_data/detections.json

Usage
-----
    pip install flask
    python dummy_api.py

    # Submit anything to /process — gets the fixture back:
    curl -F "image=@anything.png" http://localhost:5000/process -o response.json
"""
import env_loader  # noqa
from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env-var overridable)
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

# Fixture file paths — set via env vars or drop the files in sample_data/
RECTIFIED_PATH = Path(os.environ.get(
    'DUMMY_RECTIFIED_PATH', str(_HERE / 'sample_data' / 'rectified.png')))
XML_PATH = Path(os.environ.get(
    'DUMMY_XML_PATH', str(_HERE / 'sample_data' / 'score.xml')))
DETECTIONS_PATH = Path(os.environ.get(
    'DUMMY_DETECTIONS_PATH', str(_HERE / 'sample_data' / 'detections.json')))

# Fall back to project-root locations from earlier versions of dummy_api
if not RECTIFIED_PATH.exists():
    for candidate in [_HERE / 'rectified.png',
                       _HERE / 'sample_data' / '01_Okribuli_makruli.png']:
        if candidate.exists():
            RECTIFIED_PATH = candidate
            break

if not XML_PATH.exists():
    for candidate in [_HERE / 'sample_data' / '01_Okribuli_makruli.xml',
                       _HERE / 'score.xml']:
        if candidate.exists():
            XML_PATH = candidate
            break

if not DETECTIONS_PATH.exists():
    for candidate in [_HERE / 'sample_data' / 'P1_staff01.json',
                       _HERE / 'detections.json']:
        if candidate.exists():
            DETECTIONS_PATH = candidate
            break

MAX_IMAGE_MB = int(os.environ.get('MAX_IMAGE_MB', '50'))
HOST         = os.environ.get('HOST', '0.0.0.0')
PORT         = int(os.environ.get('PORT', '5000'))


# ─────────────────────────────────────────────────────────────────────────────
# Cached "last result" — same shape as real_api so endpoints behave identically
# ─────────────────────────────────────────────────────────────────────────────

_LAST_RESULT: Optional[Dict[str, Any]] = None
_PROCESS_LOCK = threading.Lock()


def _fixture_paths_exist() -> bool:
    return RECTIFIED_PATH.exists() and XML_PATH.exists()


def _make_result() -> Dict[str, Any]:
    """Build a result dict that points at the fixture files."""
    job_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    return {
        'job_id':            job_id,
        'rectified_image':   str(RECTIFIED_PATH),
        'xml_file':          str(XML_PATH),
        'detections_json':   str(DETECTIONS_PATH) if DETECTIONS_PATH.exists() else None,
        'labeled_crops_dir': None,
    }


def _build_full_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pack fixture artifacts into the same JSON envelope as real_api.
    Detections are loaded if available, otherwise an empty list.
    """
    rectified_path = Path(result['rectified_image'])
    xml_path       = Path(result['xml_file'])

    img_bytes = rectified_path.read_bytes()
    mime, _   = mimetypes.guess_type(str(rectified_path))

    detections = []
    det_path = result.get('detections_json')
    if det_path and Path(det_path).exists():
        with open(det_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Wrap a single staff dict in a list to match the real schema
        detections = data if isinstance(data, list) else [data]

    return {
        'job_id':               result['job_id'],
        'rectified_image_b64':  base64.b64encode(img_bytes).decode('ascii'),
        'rectified_image_mime': mime or 'image/png',
        'detections':           detections,
        'xml':                  xml_path.read_text(encoding='utf-8'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

def create_app():
    try:
        from flask import Flask, jsonify, request, send_file, Response
    except ImportError:
        raise ImportError(
            'Flask is required.  Install with:  pip install flask'
        )

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_MB * 1024 * 1024

    # ── CORS (same as real_api) ──
    @app.after_request
    def add_cors(response):
        response.headers['Access-Control-Allow-Origin']  = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response

    @app.route('/<path:p>', methods=['OPTIONS'])
    def options_handler(p):
        from flask import Response as R
        return R(status=204, headers={
            'Access-Control-Allow-Origin':  '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        })

    # ── GET / — browser API tester ──
    @app.route('/', methods=['GET'])
    def tester_ui():
        tester_path = _HERE / 'api_tester.html'
        if tester_path.exists():
            return send_file(str(tester_path), mimetype='text/html')
        return ('<p>api_tester.html not found next to dummy_api.py.</p>'), 404

    # ── POST /process ──
    @app.route('/process', methods=['POST'])
    def process_endpoint():
        """
        Accepts an image (multipart 'image' field or raw body).  Ignores the
        actual contents and always returns the fixture response.

        We still validate the upload format so the API behaves identically
        to real_api from the client's perspective.
        """
        try:
            if 'image' in request.files:
                image_bytes = request.files['image'].read()
            elif request.data:
                image_bytes = request.data
            else:
                return jsonify({
                    'error':   'no_image',
                    'message': "Send the image as multipart 'image' field "
                               "or raw body."
                }), 400

            if len(image_bytes) == 0:
                return jsonify({
                    'error':   'empty_upload',
                    'message': 'Image payload is empty.'
                }), 400

            if not _fixture_paths_exist():
                return jsonify({
                    'error':   'fixtures_missing',
                    'message': f'Fixture files not found.  '
                               f'Expected: {RECTIFIED_PATH} and {XML_PATH}'
                }), 500

            with _PROCESS_LOCK:
                global _LAST_RESULT
                _LAST_RESULT = _make_result()

            return jsonify(_build_full_response(_LAST_RESULT))

        except Exception as e:
            return jsonify({
                'error':   'pipeline_error',
                'message': str(e),
                'type':    type(e).__name__,
            }), 500

    # ── GET /rectified ──
    @app.route('/rectified', methods=['GET'])
    def rectified_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        return send_file(
            _LAST_RESULT['rectified_image'],
            mimetype='image/png',
            as_attachment=True,
            download_name='rectified.png',
        )

    # ── GET /xml ──
    @app.route('/xml', methods=['GET'])
    def xml_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        return send_file(
            _LAST_RESULT['xml_file'],
            mimetype='application/xml',
            as_attachment=True,
            download_name='score.xml',
        )

    # ── GET /detections ──
    @app.route('/detections', methods=['GET'])
    def detections_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        det_path = _LAST_RESULT.get('detections_json')
        if not det_path or not Path(det_path).exists():
            return jsonify([]), 200
        with open(det_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data if isinstance(data, list) else [data])

    # ── GET /full ──
    @app.route('/full', methods=['GET'])
    def full_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        return jsonify(_build_full_response(_LAST_RESULT))

    # ── GET /health ──
    @app.route('/health', methods=['GET'])
    def health_endpoint():
        # The `pipeline` block stays identical to real_api so test clients
        # don't have to special-case dummy mode.
        return jsonify({
            'status': 'ok',
            'mode':   'dummy',
            'pipeline': {
                'model_path':      'dummy (no model)',
                'model_exists':    True,
                'output_base_dir': str(_HERE),
                'max_image_mb':    MAX_IMAGE_MB,
            },
            'fixtures': {
                'rectified': {
                    'path':   str(RECTIFIED_PATH),
                    'exists': RECTIFIED_PATH.exists(),
                },
                'xml': {
                    'path':   str(XML_PATH),
                    'exists': XML_PATH.exists(),
                },
                'detections': {
                    'path':   str(DETECTIONS_PATH),
                    'exists': DETECTIONS_PATH.exists(),
                },
            },
            'last_result': {
                'has_cached_result': _LAST_RESULT is not None,
                'job_id':            _LAST_RESULT.get('job_id') if _LAST_RESULT else None,
            },
        })

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({
            'error':   'image_too_large',
            'message': f'Image exceeds {MAX_IMAGE_MB} MB limit.'
        }), 413

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('OMR Dummy API')
    print('-' * 60)
    print(f'  Mode             : dummy (always returns fixtures)')
    print(f'  Rectified PNG    : {RECTIFIED_PATH}  (exists={RECTIFIED_PATH.exists()})')
    print(f'  MusicXML         : {XML_PATH}        (exists={XML_PATH.exists()})')
    print(f'  Detections JSON  : {DETECTIONS_PATH} (exists={DETECTIONS_PATH.exists()})')
    print(f'  Max image size   : {MAX_IMAGE_MB} MB')
    print(f'  Listening on     : http://{HOST}:{PORT}')
    print('-' * 60)
    print('Endpoints:')
    print('  GET  /            — browser API tester')
    print('  POST /process     — accepts any image, returns fixtures')
    print('  GET  /rectified   — fixture PNG')
    print('  GET  /xml         — fixture MusicXML')
    print('  GET  /detections  — fixture detection JSON')
    print('  GET  /full        — all three combined')
    print('  GET  /health      — readiness check')

    if not _fixture_paths_exist():
        print('\nWARNING: One or more fixture files not found.')
        print('         /process will return 500 errors until the files exist.')
        print('         Set DUMMY_RECTIFIED_PATH and DUMMY_XML_PATH env vars,')
        print('         or place files at the locations above.')

    app = create_app()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
