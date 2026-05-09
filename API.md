# OMR API — Usage Guide

This document covers how to call the OMR API. The API is built with **FastAPI** and supports both a **real** pipeline and a **dummy** version for testing.

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your credentials:
```bash
API_TOKEN=your_secure_token
ADMIN_PASSWORD=your_admin_password
```

### 3. Start the Server
```bash
# Run the real pipeline API
python main.py

# OR run the dummy API for testing/frontend development
python dummy.py
```
Default base URL: `http://localhost:5000`

---

## Authentication

All data endpoints require a **Bearer Token** in the header:
`Authorization: Bearer <API_TOKEN>`

The `/docs` (Swagger) and `/admin` pages require **HTTP Basic Auth**:
- **Username**: `admin`
- **Password**: `<ADMIN_PASSWORD>`

---

## Endpoint Reference (v1)

| Endpoint | Method | Auth | Returns | Use case |
|---|---|---|---|---|
| `/api/v1/process` | POST | Bearer | JSON | **Primary**: Upload image, get all results |
| `/api/v1/rectified` | GET | Bearer | PNG | Get the straightened image from the last job |
| `/api/v1/detections` | GET | Bearer | JSON | Get symbol coordinates from the last job |
| `/api/v1/xml` | GET | Bearer | XML | Get MusicXML from the last job |
| `/api/v1/full` | GET | Bearer | JSON | All artifacts from the last job in one envelope |
| `/api/v1/health` | GET | None | JSON | Server/Pipeline status check |
| `/admin` | GET | Basic | HTML | **Premium Monitor Dashboard** |
| `/docs` | GET | Basic | HTML | Swagger UI documentation |

---

## Core Flow: `POST /api/v1/process`

This is the main entry point. You upload an image and receive the rectified image (base64), the detection JSON, and the MusicXML in a single response.

**Accepts:**
- `multipart/form-data` with an `image` field.
- `application/octet-stream` (raw bytes).

**Response Shape:**
```json
{
  "job_id": "20240509_123456_abc123",
  "rectified_image_b64": "<base64_string>",
  "rectified_image_mime": "image/png",
  "detections": [ ... ],
  "xml": "<MusicXML_content>"
}
```

---

## Monitoring Dashboard

The `/admin` endpoint provides a high-end, real-time monitoring interface showing:
- **Runtime Performance**: Latency of the last processing job.
- **Throughput**: Total number of jobs processed since start.
- **System Load**: CPU and Memory utilization.
- **Engine Health**: Status of the AI model and storage.

---

## Client Example

Check `client.py` for a full Python implementation of the authentication and processing flow.

```python
import requests

url = "http://localhost:5000/api/v1/process"
headers = {"Authorization": "Bearer your_secure_token"}
files = {"image": open("score.png", "rb")}

response = requests.post(url, headers=headers, files=files)
data = response.json()
print(f"Job ID: {data['job_id']}")
```
