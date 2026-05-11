"""
Hardhat Detection System Using YOLOv8 — Flask API.
Image upload, video upload + annotated output, live MJPEG stream.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

load_dotenv()

from detector import HardhatDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

RESULTS_DIR = Path(app.root_path) / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

detector = HardhatDetector()
_detector_lock = threading.Lock()

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}


def _ext_ok(filename: str, allowed: set[str]) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed


def configure_camera(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() and sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap


def generate_frames():
    cap = _open_camera()
    if not cap.isOpened():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Camera not available",
            (60, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        while True:
            ok, buffer = cv2.imencode(".jpg", blank)
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.12)

    configure_camera(cap)
    fps_ema = 30.0
    alpha = 0.9

    try:
        while True:
            t0 = time.time()
            success, frame = cap.read()
            if not success:
                break

            with _detector_lock:
                frame, heads, helmets, total = detector.detect(
                    frame, live=True, fps=int(round(fps_ema))
                )
            frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)

            dt = time.time() - t0
            inst_fps = 1.0 / dt if dt > 0 else fps_ema
            fps_ema = alpha * fps_ema + (1.0 - alpha) * inst_fps

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stream")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/upload_image", methods=["POST"])
@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _ext_ok(file.filename, ALLOWED_IMAGE):
        return jsonify({"error": "Allowed images: " + ", ".join(sorted(ALLOWED_IMAGE))}), 400

    data = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    with _detector_lock:
        detector.reset_predictor()
        annotated, heads, helmets, total = detector.detect(img, live=False)

    uid = str(uuid.uuid4())
    out_name = f"{uid}_image.jpg"
    out_path = RESULTS_DIR / out_name
    if not cv2.imwrite(str(out_path), annotated):
        return jsonify({"error": "Failed to save result image"}), 500

    return jsonify(
        {
            "head_count": heads,
            "helmet_count": helmets,
            "total_count": total,
            "result_image_url": f"/static/results/{out_name}",
        }
    )


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _ext_ok(file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Allowed video: " + ", ".join(sorted(ALLOWED_VIDEO))}), 400

    uid = str(uuid.uuid4())
    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    temp_path = RESULTS_DIR / f"{uid}_in.{ext}"
    file.save(str(temp_path))

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        if temp_path.is_file():
            temp_path.unlink()
        return jsonify({"error": "Could not open video file"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        if temp_path.is_file():
            temp_path.unlink()
        return jsonify({"error": "Invalid video dimensions"}), 400

    out_mp4 = RESULTS_DIR / f"{uid}_video_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        if temp_path.is_file():
            temp_path.unlink()
        return jsonify({"error": "Could not create output video writer"}), 500

    with _detector_lock:
        detector.reset_predictor()

    max_heads = 0
    max_helmets = 0
    max_total = 0
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with _detector_lock:
                annotated, hc, hel, tot = detector.detect(frame, live=False)
            if annotated.shape[1] != w or annotated.shape[0] != h:
                annotated = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(annotated)
            if tot >= max_total:
                max_total = tot
                max_heads = hc
                max_helmets = hel
            frame_index += 1
    finally:
        writer.release()
        cap.release()
        if temp_path.is_file():
            temp_path.unlink()

    if frame_index == 0:
        if out_mp4.is_file():
            out_mp4.unlink()
        return jsonify({"error": "Video contained no readable frames"}), 400

    return jsonify(
        {
            "head_count_max": max_heads,
            "helmet_count_max": max_helmets,
            "total_count_max": max_total,
            "frames_processed": frame_index,
            "result_video_url": f"/static/results/{out_mp4.name}",
        }
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True, host="0.0.0.0", port=5000)
