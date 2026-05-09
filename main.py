"""
real_api.py — Production OMR API (FastAPI)

Endpoints
---------
    POST /api/v1/process     — main endpoint: upload an image, get all outputs
    GET  /api/v1/rectified   — returns rectified PNG of the LAST processed image
    GET  /api/v1/detections  — last detection JSON
    GET  /api/v1/xml          — last MusicXML
    GET  /api/v1/full         — last full response (all three combined)
    GET  /api/v1/health       — server + pipeline readiness check
    GET  /admin               — Admin monitoring dashboard
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OMR-API")

# Add project root and omr directory to sys.path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
if str(_ROOT / 'omr') not in sys.path:
    sys.path.append(str(_ROOT / 'omr'))

import env_loader  # noqa
import base64
import json
import mimetypes
import os
import sys
import threading
import traceback
import uuid
import time
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import psutil
from pydantic import BaseModel, Field

import pipeline   # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class Detection(BaseModel):
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

class StaffResponse(BaseModel):
    part_id: str
    staff_in_part: int
    top_y: int
    bot_y: int
    line_positions: list[int]
    line_spacing: float
    crop_y1: int
    total_detections: int
    detections: list[Detection]

class FullResponse(BaseModel):
    job_id: str
    rectified_image_b64: str
    rectified_image_mime: str
    detections: list[StaffResponse]
    xml: str

class HealthResponse(BaseModel):
    status: str
    mode: str = "production"
    uptime_seconds: float
    requests_processed: int
    last_latency: float
    pipeline: dict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
_HERE = _ROOT

MODEL_PATH      = Path(os.environ.get('MODEL_PATH', str(_HERE / 'models' / 'deepscores_crops_v1.pt')))
OUTPUT_BASE_DIR = Path(os.environ.get('OUTPUT_BASE_DIR', str(_HERE / 'api_outputs')))
MAX_IMAGE_MB    = int(os.environ.get('MAX_IMAGE_MB', '50'))
HOST            = os.environ.get('HOST', '0.0.0.0')
PORT            = int(os.environ.get('PORT', '5000'))
API_TOKEN       = os.environ.get('API_TOKEN', 'supersecrettoken')
ADMIN_PASSWORD  = os.environ.get('ADMIN_PASSWORD', 'adminpass')

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

# Globals for stats/cache
_LAST_RESULT: Optional[Dict[str, Any]] = None
_PIPELINE_LOCK = threading.Lock()
_START_TIME = time.time()
_REQUESTS_PROCESSED = 0
_LAST_LATENCY = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Setup & Security
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None) # Hide default docs

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security_basic = HTTPBasic()
security_bearer = HTTPBearer()

def get_admin_auth(credentials: HTTPBasicCredentials = Depends(security_basic)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    if not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/docs", include_in_schema=False)
async def get_documentation(username: str = Depends(get_admin_auth)):
    return get_swagger_ui_html(openapi_url=app.openapi_url, title="API Docs")

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi(username: str = Depends(get_admin_auth)):
    return app.openapi()

# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline call
# ─────────────────────────────────────────────────────────────────────────────
def _run_pipeline(image_bytes: bytes, original_name: str) -> Dict[str, Any]:
    global _LAST_RESULT, _REQUESTS_PROCESSED, _LAST_LATENCY
    
    start_t = time.time()
    
    job_id  = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    job_dir = OUTPUT_BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_name).suffix.lower() or '.png'
    if ext not in ALLOWED_EXT:
        ext = '.png'
    in_path = job_dir / f'input{ext}'
    in_path.write_bytes(image_bytes)

    with _PIPELINE_LOCK:
        result = pipeline.run_pipeline(
            image_path        = str(in_path),
            output_dir        = str(job_dir),
            model_path        = str(MODEL_PATH),
            save_debug_images = False,
            save_labeled_crops = True,
        )

    response = {
        'job_id':           job_id,
        'rectified_image':  result['rectified_image'],
        'detections_json':  result['detections_json'],
        'xml_file':         result['xml_file'],
        'labeled_crops_dir': result.get('labeled_crops_dir'),
    }
    _LAST_RESULT = response
    
    _REQUESTS_PROCESSED += 1
    _LAST_LATENCY = time.time() - start_t
    
    logger.info(f"Job {job_id} completed in {_LAST_LATENCY:.2f}s")
    return response

def _build_full_response(result: Dict[str, Any]) -> Dict[str, Any]:
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
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/process", response_model=FullResponse, dependencies=[Depends(verify_token)])
async def process_endpoint(request: Request, image: Optional[UploadFile] = File(None)):
    """
    Submit an image, get the full response (rectified, detections, xml).
    """
    try:
        image_bytes = None
        original_name = 'upload.png'
        
        body = await request.body()
        if len(body) > MAX_IMAGE_MB * 1024 * 1024:
            logger.warning(f"Rejecting upload: {len(body) / 1024 / 1024:.2f}MB exceeds limit")
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_MB} MB limit.")
        
        if image:
            image_bytes = await image.read()
            original_name = image.filename or 'upload.png'
        elif body:
            image_bytes = body
            original_name = request.query_params.get('name', 'upload.png')
        else:
            return JSONResponse({"error": "no_image", "message": "Send the image as multipart 'image' field or raw body."}, status_code=400)

        if not image_bytes or len(image_bytes) == 0:
            return JSONResponse({"error": "empty_upload", "message": "Image payload is empty."}, status_code=400)

        logger.info(f"Processing upload: {original_name} ({len(image_bytes)} bytes)")
        result = _run_pipeline(image_bytes, original_name)
        return JSONResponse(_build_full_response(result))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse({
            'error':   'pipeline_error',
            'message': str(e),
            'type':    type(e).__name__,
        }, status_code=500)

@app.get("/api/v1/rectified", dependencies=[Depends(verify_token)])
async def rectified_endpoint():
    if _LAST_RESULT is None:
        raise HTTPException(status_code=404, detail="No cached result. Call POST /process first.")
    return FileResponse(_LAST_RESULT['rectified_image'], media_type='image/png', filename='rectified.png')

@app.get("/api/v1/detections", response_model=list[StaffResponse], dependencies=[Depends(verify_token)])
async def detections_endpoint():
    if _LAST_RESULT is None:
        raise HTTPException(status_code=404, detail="No cached result. Call POST /process first.")
    with open(_LAST_RESULT['detections_json'], 'r', encoding='utf-8') as f:
        return json.load(f)

@app.get("/api/v1/xml", dependencies=[Depends(verify_token)])
async def xml_endpoint():
    if _LAST_RESULT is None:
        raise HTTPException(status_code=404, detail="No cached result. Call POST /process first.")
    return FileResponse(_LAST_RESULT['xml_file'], media_type='application/xml', filename='score.xml')

@app.get("/api/v1/full", response_model=FullResponse, dependencies=[Depends(verify_token)])
async def full_endpoint():
    if _LAST_RESULT is None:
        raise HTTPException(status_code=404, detail="No cached result. Call POST /process first.")
    return JSONResponse(_build_full_response(_LAST_RESULT))

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_endpoint():
    return {
        'status': 'ok',
        'uptime_seconds': time.time() - _START_TIME,
        'requests_processed': _REQUESTS_PROCESSED,
        'last_latency': _LAST_LATENCY,
        'pipeline': {
            'model_path':        str(MODEL_PATH),
            'model_exists':      MODEL_PATH.exists(),
            'output_base_dir':   str(OUTPUT_BASE_DIR),
            'max_image_mb':      MAX_IMAGE_MB,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Admin Dashboard
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin", include_in_schema=False)
async def admin_dashboard(username: str = Depends(get_admin_auth)):
    uptime = time.time() - _START_TIME
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage(str(OUTPUT_BASE_DIR))
    except FileNotFoundError:
        disk = psutil.disk_usage(os.getcwd())
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OMR API Monitor</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0a0b10;
                --card-bg: rgba(255, 255, 255, 0.05);
                --accent: #0070f3;
                --text: #ffffff;
                --text-muted: #888888;
                --success: #00ff88;
            }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 40px;
                display: flex;
                justify-content: center;
            }}
            .dashboard {{
                max-width: 1000px;
                width: 100%;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 40px;
            }}
            h1 {{
                font-size: 24px;
                font-weight: 600;
                margin: 0;
                letter-spacing: -0.02em;
            }}
            .status-badge {{
                background: rgba(0, 255, 136, 0.1);
                color: var(--success);
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .status-dot {{
                width: 8px;
                height: 8px;
                background: var(--success);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--success);
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}
            .card {{
                background: var(--card-bg);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 24px;
                transition: transform 0.2s ease;
            }}
            .card:hover {{
                transform: translateY(-4px);
                border-color: rgba(255, 255, 255, 0.2);
            }}
            .card-title {{
                color: var(--text-muted);
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 16px;
                letter-spacing: 0.05em;
            }}
            .card-value {{
                font-size: 28px;
                font-weight: 600;
                font-family: 'JetBrains Mono', monospace;
            }}
            .card-sub {{
                color: var(--text-muted);
                font-size: 14px;
                margin-top: 8px;
            }}
            .progress-bar {{
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                margin-top: 16px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                background: var(--accent);
                border-radius: 3px;
            }}
            footer {{
                margin-top: 60px;
                text-align: center;
                color: var(--text-muted);
                font-size: 13px;
            }}
            code {{
                font-family: 'JetBrains Mono', monospace;
                background: rgba(0,0,0,0.3);
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <header>
                <h1>OMR Engine Monitor</h1>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    SYSTEM ONLINE
                </div>
            </header>
            
            <div class="grid">
                <div class="card">
                    <div class="card-title">Runtime Performance</div>
                    <div class="card-value">{_LAST_LATENCY:.2f}s</div>
                    <div class="card-sub">Last Job Latency</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min(100, _LAST_LATENCY * 10)}%"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">Throughput</div>
                    <div class="card-value">{_REQUESTS_PROCESSED}</div>
                    <div class="card-sub">Total Jobs Processed</div>
                </div>
                
                <div class="card">
                    <div class="card-title">System Load</div>
                    <div class="card-value">{cpu_percent}%</div>
                    <div class="card-sub">CPU Utilization</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {cpu_percent}%"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">Memory</div>
                    <div class="card-value">{memory.percent}%</div>
                    <div class="card-sub">{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB Used</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {memory.percent}%"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">Storage</div>
                    <div class="card-value">{disk.percent}%</div>
                    <div class="card-sub">{disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB Output Disk</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {disk.percent}%"></div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">Engine Health</div>
                    <div class="card-value">{'ONLINE' if MODEL_PATH.exists() else 'ERROR'}</div>
                    <div class="card-sub">Model: <code>{MODEL_PATH.name}</code></div>
                </div>
            </div>
            
            <footer>
                Uptime: {int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s • 
                Rendered at {datetime.now().strftime('%H:%M:%S')}
            </footer>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    print('OMR Real API (FastAPI)')
    print('-' * 60)
    print(f'  Model           : {{MODEL_PATH}}')
    print(f'  Model exists    : {{MODEL_PATH.exists()}}')
    print(f'  Output base dir : {{OUTPUT_BASE_DIR}}')
    print(f'  Max image size  : {{MAX_IMAGE_MB}} MB')
    print(f'  Listening on    : http://{{HOST}}:{{PORT}}')
    print('-' * 60)
    
    if not MODEL_PATH.exists():
        print('\nWARNING: Model file not found.  Set MODEL_PATH env variable')
        print('         or place the .pt file at the expected path.')
        print('         The server will start anyway but /process will fail.')

    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
