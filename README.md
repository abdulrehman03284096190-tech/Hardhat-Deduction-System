# Hardhat Detection System Using YOLOv8

Flask app for live webcam and image upload head / helmet counting with a local **`best.pt`** model.

## Project layout

Rename the project folder to **`hardhat-detection-yolov8`** if you want the exact layout from your spec. Contents:

- `app.py` — Flask routes
- `detector.py` — `HardhatDetector` (YOLO + live smoothing)
- `best.pt` — your trained weights (same folder as `app.py`, or set `MODEL_PATH`)
- `templates/index.html`
- `static/` — optional assets

## Setup

Use **Python 3.10–3.13** (64-bit). On Windows, prefer **pre-built wheels** so you do not need the Visual Studio C++ compiler.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python app.py
```

If `pip install` tries to **build NumPy from source** (Meson / “Unknown compiler”), upgrade pip and use the `requirements.txt` here: it pins **`numpy>=2`** and **`ultralytics>=8.3.78`**, which install as wheels on current Windows + Python. Older **`ultralytics==8.3.0`** required **`numpy<2`** and often triggered source builds on newer Python versions.

## Environment

See `.env.example`: `MODEL_PATH=best.pt`, `FLASK_ENV=development`.

## API

- **`GET /`** — Web UI (tabs: Live, Upload image, Upload video)
- **`GET /api/stream`** — MJPEG live webcam with overlays and detection boxes
- **`POST /api/upload_image`** — multipart **`file`** (PNG/JPG/JPEG/WebP/BMP). Response: `head_count`, `helmet_count`, `total_count`, **`result_image_url`** (annotated JPEG in `static/results/`)
- **`POST /api/upload_video`** — multipart **`file`** (MP4/AVI/MOV/MKV/WEBM). Response: **`head_count_max`**, **`helmet_count_max`**, **`total_count_max`** (peak frame), **`frames_processed`**, **`result_video_url`** (annotated MP4)

## PyTorch 2.6+

`detector.py` registers Ultralytics task classes with `add_safe_globals` and defaults `torch.load(..., weights_only=False)` for trusted local checkpoints.
