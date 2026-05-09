"""
real_api.py — Production OMR API that runs the actual pipeline.

Wraps pipeline.run_pipeline() in a Flask service.  The response shapes are
identical to dummy_api.py so frontends can switch between them with only a
URL change.

Endpoints
---------
    POST /process            — main endpoint: upload an image, get all outputs
    GET  /rectified          — returns rectified PNG of the LAST processed image
    GET  /detections         — last detection JSON
    GET  /xml                — last MusicXML
    GET  /full               — last full response (all three combined)
    GET  /health             — server + pipeline readiness check

The GET endpoints serve a cached "last result" so a frontend that already
called /process can re-fetch a single artifact without re-running the model.
For a stateless multi-user server, replace the LAST_RESULT global with a
session-keyed cache or require /process every time.

Configuration
-------------
    MODEL_PATH         path to YOLO weights
    OUTPUT_BASE_DIR    where pipeline writes intermediate files
    MAX_IMAGE_MB       reject uploads above this size
    HOST, PORT         listen address

All can be overridden via environment variables of the same name, e.g.:
    set MODEL_PATH=S:\omr\models\v2.pt
    python real_api.py

Usage
-----
    pip install flask
    python real_api.py

    # Submit an image:
    curl -F "image=@score.png" http://localhost:5000/process -o response.json
"""
import env_loader  # noqa
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import tempfile
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Make sibling pipeline modules importable when running from any cwd
sys.path.append(str(Path(__file__).parent))

import pipeline   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env-var overridable)
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

MODEL_PATH      = Path(os.environ.get(
    'MODEL_PATH', str(_HERE / 'models' / 'deepscores_crops_v1.pt')))
OUTPUT_BASE_DIR = Path(os.environ.get(
    'OUTPUT_BASE_DIR', str(_HERE / 'api_outputs')))
MAX_IMAGE_MB    = int(os.environ.get('MAX_IMAGE_MB', '50'))
HOST            = os.environ.get('HOST', '0.0.0.0')
PORT            = int(os.environ.get('PORT', '5000'))

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


# ─────────────────────────────────────────────────────────────────────────────
# Cached "last result" — for the GET endpoints
# ─────────────────────────────────────────────────────────────────────────────
#
# A single shared cache is fine for a single-user/local-tool deployment.
# Replace with a session/UUID-keyed dict if exposing this to multiple users.

_LAST_RESULT: Optional[Dict[str, Any]] = None
_PIPELINE_LOCK = threading.Lock()    # serialize model inference


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline call
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(image_bytes: bytes, original_name: str) -> Dict[str, Any]:
    """
    Save the upload to a fresh output directory, run the pipeline, and
    return the result paths plus metadata.  Caches the result globally.
    """
    global _LAST_RESULT

    # Per-request output directory keeps multiple submissions separated
    job_id  = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    job_dir = OUTPUT_BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save the upload using its original extension (or .png as fallback)
    ext = Path(original_name).suffix.lower() or '.png'
    if ext not in ALLOWED_EXT:
        ext = '.png'
    in_path = job_dir / f'input{ext}'
    in_path.write_bytes(image_bytes)

    # Inference is not thread-safe across concurrent calls — serialize it
    with _PIPELINE_LOCK:
        result = pipeline.run_pipeline(
            image_path        = str(in_path),
            output_dir        = str(job_dir),
            model_path        = str(MODEL_PATH),
            save_debug_images = False,
            save_labeled_crops = True,
        )

    # Augment with metadata (don't include the in-memory PageDetections
    # object — it isn't JSON-serializable and the JSON file already has it)
    response = {
        'job_id':           job_id,
        'rectified_image':  result['rectified_image'],
        'detections_json':  result['detections_json'],
        'xml_file':         result['xml_file'],
        'labeled_crops_dir': result.get('labeled_crops_dir'),
    }
    _LAST_RESULT = response
    return response


def _build_full_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the pipeline's on-disk artifacts and pack them into the same
    JSON envelope as dummy_api.get_full_response().
    """
    rectified_path = Path(result['rectified_image'])
    xml_path       = Path(result['xml_file'])
    det_path       = Path(result['detections_json'])

    img_bytes = rectified_path.read_bytes()
    mime, _   = mimetypes.guess_type(str(rectified_path))

    with open(det_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

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

    # ── GET / — serve the browser API tester ──
    @app.route('/', methods=['GET'])
    def tester_ui():
        tester_path = Path(__file__).parent / 'api_tester.html'
        if tester_path.exists():
            return send_file(str(tester_path), mimetype='text/html')
        return ('<p>api_tester.html not found next to real_api.py. '
                'Place both files in the same directory.</p>'), 404

    # ── POST /process ──
    @app.route('/process', methods=['POST'])
    def process_endpoint():
        """
        Submit an image, get the full response (rectified, detections, xml).

        Accepts:
            multipart/form-data with field name 'image'
            application/octet-stream  (raw image bytes; filename in URL ?name=)
        """
        try:
            # multipart/form-data
            if 'image' in request.files:
                f = request.files['image']
                image_bytes = f.read()
                original_name = f.filename or 'upload.png'
            # raw POST body
            elif request.data:
                image_bytes = request.data
                original_name = request.args.get('name', 'upload.png')
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

            result = _run_pipeline(image_bytes, original_name)
            return jsonify(_build_full_response(result))

        except Exception as e:
            traceback.print_exc()
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

    # ── GET /detections ──
    @app.route('/detections', methods=['GET'])
    def detections_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        with open(_LAST_RESULT['detections_json'], 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='application/json')

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
        return jsonify({
            'status': 'ok',
            'pipeline': {
                'model_path':        str(MODEL_PATH),
                'model_exists':      MODEL_PATH.exists(),
                'output_base_dir':   str(OUTPUT_BASE_DIR),
                'max_image_mb':      MAX_IMAGE_MB,
            },
            'last_result': {
                'has_cached_result': _LAST_RESULT is not None,
                'job_id':            _LAST_RESULT.get('job_id') if _LAST_RESULT else None,
            },
        })

    # ── Catch-all 413 for oversized uploads ──
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
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print('OMR Real API')
    print('-' * 60)
    print(f'  Model           : {MODEL_PATH}')
    print(f'  Model exists    : {MODEL_PATH.exists()}')
    print(f'  Output base dir : {OUTPUT_BASE_DIR}')
    print(f'  Max image size  : {MAX_IMAGE_MB} MB')
    print(f'  Listening on    : http://{HOST}:{PORT}')
    print('-' * 60)
    print('Endpoints:')
    print('  POST /process     — upload image, get all outputs')
    print('  GET  /rectified   — last rectified PNG')
    print('  GET  /detections  — last detections JSON')
    print('  GET  /xml         — last MusicXML')
    print('  GET  /full        — last full response')
    print('  GET  /health      — readiness check')

    if not MODEL_PATH.exists():
        print('\nWARNING: Model file not found.  Set MODEL_PATH env variable')
        print('         or place the .pt file at the expected path.')
        print('         The server will start anyway but /process will fail.')

    app = create_app()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
