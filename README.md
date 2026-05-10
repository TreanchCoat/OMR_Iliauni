

# OMR Iliauni

A robust end-to-end Optical Music Recognition (OMR) system that converts sheet music images into **MusicXML**.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials and settings.

---

### 3. Run server (Uvicorn)

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

---

### 4. Open dashboard

```
http://localhost:5000/admin
```

* Username: `admin`
* Password: value from `.env`

---

## 📂 Project Structure

* `main.py` — production FastAPI app
* `dummy.py` — lightweight mock API (no ML, for testing)
* `client.py` — example API client
* `omr/` — core pipeline:

  * rectification
  * staff detection/removal
  * YOLO symbol detection
  * MusicXML builder
* `models/` — trained YOLO / U-Net weights
* `sample_data/` — test inputs

---

## 🛠 Features

* **End-to-End Pipeline**
  Raw image → processed → structured MusicXML

* **FastAPI Backend**
  Auto-generated Swagger docs:

  ```
  /docs
  ```

* **Admin Dashboard**

  * CPU / Memory usage
  * Request latency
  * Pipeline timing

* **Dual Mode Execution**

  * `main.py` → full ML pipeline
  * `dummy.py` → API-only (no heavy deps)

---

## ⚙️ Deployment Notes (Important)

This project includes heavy ML dependencies (`torch`, `ultralytics`, `opencv`).

### For low-resource environments (e.g. Pterodactyl):

* Use:

  ```bash
  pip install --no-cache-dir
  ```
* Prefer **CPU-only torch**
* Remove training-only deps like `albumentations` if unused
* Consider running inference in a **separate service**

---

## 📖 API Documentation

See:

```
API.md
```

or open interactive docs:

```
http://localhost:5000/docs
```

---

## 🧪 Dev Tip

If you only want to test API behavior without installing ML stack:

```bash
python -m uvicorn dummy:app --host 0.0.0.0 --port 5000
```

