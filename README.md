# OMR Iliauni

A robust end-to-end Optical Music Recognition (OMR) system designed to convert sheet music images into MusicXML.

## 🚀 Quick Start

1. **Setup**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   ```
2. **Run**:
   ```bash
   python main.py
   ```
3. **Monitor**: Open `http://localhost:5000/admin` (User: `admin`, Pass: your `.env` password).

## 📂 Project Structure

- **`main.py` / `dummy.py`**: FastAPI entry points.
- **`client.py`**: Example API consumer.
- **`omr/`**: The OMR engine (rectification, staff removal, YOLO detection, XML building).
- **`models/`**: Pre-trained YOLO and U-Net models.
- **`sample_data/`**: Hardcoded fixtures for testing the API interface.

## 🛠 Features

- **End-to-End Pipeline**: From raw photo to structured MusicXML.
- **Modern API**: FastAPI-based with Swagger docs and Bearer Token security.
- **Admin Dashboard**: Real-time monitoring of CPU, Memory, and Pipeline latency.
- **Dual Mode**: `main.py` for production, `dummy.py` for interface testing.

## 📖 Documentation

See [API.md](./API.md) for detailed endpoint documentation and usage examples.