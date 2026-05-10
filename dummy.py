"""
dummy_api.py — Dummy version of real_api.py for testing without the model.

Mirrors real_api.py's interface exactly — same endpoints, same request/
response shapes, same authentication — but always returns hardcoded 
fixture image and MusicXML regardless of input.

Endpoints
---------
    GET  /api/v1/health       — sanity check
    GET  /api/v1/rectified    — download rectified PNG
    GET  /api/v1/detections   — fetch detection JSON
    GET  /api/v1/xml          — download MusicXML
    GET  /api/v1/full         — fetch all three in one response
    POST /api/v1/process      — submit an image (simulated)
    GET  /admin               — Admin monitoring dashboard
"""

import sys
import logging
import base64
import json
import mimetypes
import os
import threading
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
from pydantic import BaseModel
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OMR-DUMMY")

# Add project root to sys.path
_ROOT = Path(os.getcwd())
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

import env_loader  # noqa

# ─────────────────────────────────────────────────────────────────────────────
# Models (Shared with real_api)
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
    mode: str = "dummy"
    uptime_seconds: float
    requests_processed: int
    last_latency: float
    fixtures: dict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
_HERE = _ROOT

RECTIFIED_PATH = Path(os.environ.get('DUMMY_RECTIFIED_PATH', str(_HERE / 'sample_data' / 'rectified.png')))
XML_PATH       = Path(os.environ.get('DUMMY_XML_PATH',       str(_HERE / 'sample_data' / 'score.xml')))
DETECTIONS_PATH = Path(os.environ.get('DUMMY_DETECTIONS_PATH', str(_HERE / 'sample_data' / 'detections.json')))

MAX_IMAGE_MB   = int(os.environ.get('MAX_IMAGE_MB', '50'))
HOST           = os.environ.get('HOST', '0.0.0.0')
PORT           = int(os.environ.get('PORT', '5000'))
API_TOKEN      = os.environ.get('API_TOKEN', 'supersecrettoken')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'adminpass')

_START_TIME = time.time()
_REQUESTS_PROCESSED = 0
_LAST_LATENCY = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_full_response() -> Dict[str, Any]:
    job_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    img_bytes = RECTIFIED_PATH.read_bytes()
    mime, _   = mimetypes.guess_type(str(RECTIFIED_PATH))

    detections = []
    if DETECTIONS_PATH.exists():
        with open(DETECTIONS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        detections = data if isinstance(data, list) else [data]

    return {
        'job_id':               job_id,
        'rectified_image_b64':  base64.b64encode(img_bytes).decode('ascii'),
        'rectified_image_mime': mime or 'image/png',
        'detections':           detections,
        'xml':                  XML_PATH.read_text(encoding='utf-8'),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Setup & Security
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None)

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
    return get_swagger_ui_html(openapi_url=app.openapi_url, title="API Docs (Dummy)")

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_endpoint():
    return {
        "status": "ok",
        "mode": "dummy",
        "uptime_seconds": time.time() - _START_TIME,
        "requests_processed": _REQUESTS_PROCESSED,
        "last_latency": _LAST_LATENCY,
        "fixtures": {
            "rectified":  {"path": str(RECTIFIED_PATH),  "exists": RECTIFIED_PATH.exists()},
            "detections": {"path": str(DETECTIONS_PATH), "exists": DETECTIONS_PATH.exists()},
            "xml":        {"path": str(XML_PATH),        "exists": XML_PATH.exists()},
        }
    }

@app.post("/api/v1/process", response_model=FullResponse, dependencies=[Depends(verify_token)])
async def process_endpoint(request: Request):
    global _REQUESTS_PROCESSED, _LAST_LATENCY
    start_t = time.time()
    
    body = await request.body()
    logger.info(f"Dummy processing request: {len(body)} bytes")
    
    if not RECTIFIED_PATH.exists() or not XML_PATH.exists():
        raise HTTPException(status_code=500, detail="Dummy fixtures missing on server")
    
    _REQUESTS_PROCESSED += 1
    _LAST_LATENCY = 0.05 # Fixed low latency for dummy
    
    return JSONResponse(_build_full_response())

@app.get("/api/v1/rectified", dependencies=[Depends(verify_token)])
async def rectified_endpoint():
    if not RECTIFIED_PATH.exists():
        raise HTTPException(status_code=404, detail="Rectified fixture missing")
    return FileResponse(RECTIFIED_PATH, media_type="image/png", filename="rectified.png")

@app.get("/api/v1/detections", response_model=list[StaffResponse], dependencies=[Depends(verify_token)])
async def detections_endpoint():
    if not DETECTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="Detections fixture missing")
    with open(DETECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/v1/xml", dependencies=[Depends(verify_token)])
async def xml_endpoint():
    if not XML_PATH.exists():
        raise HTTPException(status_code=404, detail="XML fixture missing")
    return FileResponse(XML_PATH, media_type="application/xml", filename="score.xml")

@app.get("/api/v1/full", response_model=FullResponse, dependencies=[Depends(verify_token)])
async def full_endpoint():
    if not RECTIFIED_PATH.exists() or not XML_PATH.exists():
        raise HTTPException(status_code=404, detail="Fixtures missing")
    return JSONResponse(_build_full_response())

# ─────────────────────────────────────────────────────────────────────────────
# Admin Dashboard (Shared Premium UI)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin", include_in_schema=False)
async def admin_dashboard(username: str = Depends(get_admin_auth)):
    uptime = time.time() - _START_TIME
    cpu_percent = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OMR API Monitor (DUMMY)</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0a0b10;
                --card-bg: rgba(255, 255, 255, 0.05);
                --accent: #f30070;
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
            .dashboard {{ max-width: 1000px; width: 100%; }}
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; }}
            h1 {{ font-size: 24px; font-weight: 600; margin: 0; letter-spacing: -0.02em; }}
            .status-badge {{
                background: rgba(255, 0, 112, 0.1);
                color: var(--accent);
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                display: flex;
                align-items: center; gap: 8px;
            }}
            .status-dot {{ width: 8px; height: 8px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 10px var(--accent); }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{
                background: var(--card-bg);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px; padding: 24px;
            }}
            .card-title {{ color: var(--text-muted); font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 16px; }}
            .card-value {{ font-size: 28px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
            .card-sub {{ color: var(--text-muted); font-size: 14px; margin-top: 8px; }}
            footer {{ margin-top: 60px; text-align: center; color: var(--text-muted); font-size: 13px; }}
            code {{ font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <header>
                <h1>OMR Engine Monitor <span style="color:var(--accent)">[DUMMY MODE]</span></h1>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    DUMMY MODE ACTIVE
                </div>
            </header>
            
            <div class="grid">
                <div class="card">
                    <div class="card-title">Throughput</div>
                    <div class="card-value">{_REQUESTS_PROCESSED}</div>
                    <div class="card-sub">Simulated Jobs Processed</div>
                </div>
                
                <div class="card">
                    <div class="card-title">System Load</div>
                    <div class="card-value">{cpu_percent}%</div>
                    <div class="card-sub">CPU Utilization</div>
                </div>
                
                <div class="card">
                    <div class="card-title">Memory</div>
                    <div class="card-value">{memory.percent}%</div>
                    <div class="card-sub">{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB Used</div>
                </div>
                
                <div class="card">
                    <div class="card-title">Fixtures</div>
                    <div class="card-value">{'READY' if RECTIFIED_PATH.exists() else 'ERROR'}</div>
                    <div class="card-sub">PNG: <code>{RECTIFIED_PATH.name}</code></div>
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
    logger.info(f"Starting OMR Dummy API on {HOST}:{PORT}")
    uvicorn.run("dummy:app", host=HOST, port=PORT, log_level="info")
