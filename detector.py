"""
Hardhat / head detection using Ultralytics YOLO (best.pt).
PyTorch 2.6+ checkpoint loading + image / live inference with drawn boxes.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# PyTorch 2.6+ safe unpickling allowlist (Ultralytics checkpoints)
try:
    _safe = []
    for _name in (
        "DetectionModel",
        "SegmentationModel",
        "PoseModel",
        "ClassificationModel",
    ):
        try:
            _m = __import__(
                "ultralytics.nn.tasks", fromlist=[_name]
            )
            _safe.append(getattr(_m, _name))
        except Exception:
            pass
    if _safe:
        torch.serialization.add_safe_globals(_safe)
except Exception:
    pass

try:
    import functools

    _orig_torch_load = torch.load

    @functools.wraps(_orig_torch_load)
    def _torch_load_trusted_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_trusted_default
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
_CONF_IMAGE = 0.4
_CONF_LIVE = 0.25
_RAW = os.getenv("MODEL_PATH", "best.pt").strip()
WEIGHTS_PATH = Path(_RAW) if os.path.isabs(_RAW) else (ROOT / _RAW)


class HardhatDetector:
    """Uses predict() only — track()/ByteTrack GMC + optical flow can crash OpenCV when frame sizes vary."""

    def __init__(self) -> None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
        print(f"Loading model: {WEIGHTS_PATH}")
        self._model = YOLO(str(WEIGHTS_PATH))
        names = self._model.names
        print(f"Classes: {dict(names) if hasattr(names, 'keys') else names}")
        self._smooth_heads: deque[int] = deque(maxlen=5)
        self._smooth_helmets: deque[int] = deque(maxlen=5)

    def _ensure_plain_inference(self) -> None:
        """Turn off tracking on shared predictor so predict() does not run ByteTrack/GMC."""
        pred = getattr(self._model, "predictor", None)
        if pred is None:
            return
        args = getattr(pred, "args", None)
        if args is None:
            return
        if hasattr(args, "track"):
            setattr(args, "track", False)
        if hasattr(args, "persist"):
            setattr(args, "persist", False)

    def reset_predictor(self) -> None:
        """Drop cached predictor so image/video runs never reuse ByteTrack postprocess hooks."""
        if hasattr(self._model, "predictor"):
            setattr(self._model, "predictor", None)

    def _predict_arr(self, arr: np.ndarray, conf: float):
        """predict() with tracker disabled when supported (Ultralytics 8.x)."""
        self._ensure_plain_inference()
        try:
            return self._model.predict(arr, conf=conf, verbose=False, tracker=False)
        except TypeError:
            self._ensure_plain_inference()
            return self._model.predict(arr, conf=conf, verbose=False)

    def preprocess_live(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)

    def _class_name(self, cls_id: int) -> str:
        names = self._model.names
        if isinstance(names, dict):
            raw = names.get(cls_id, names.get(str(cls_id), ""))
        else:
            raw = names[cls_id] if 0 <= cls_id < len(names) else ""
        return str(raw).lower().strip()

    def _count_from_results(self, r0) -> tuple[int, int]:
        head_count = 0
        helmet_count = 0
        if r0.boxes is None or len(r0.boxes) == 0:
            return head_count, helmet_count
        for box in r0.boxes:
            cls = int(box.cls[0])
            name = self._class_name(cls)
            if name in ("head", "person"):
                head_count += 1
            elif name in ("helmet", "hardhat"):
                helmet_count += 1
        return head_count, helmet_count

    def _overlay_counts(
        self,
        vis: np.ndarray,
        head_count: int,
        helmet_count: int,
        total_count: int,
        fps: int | None = None,
    ) -> np.ndarray:
        cv2.putText(
            vis,
            f"Head Count: {head_count}",
            (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Helmet Count: {helmet_count}",
            (10, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Total Count: {total_count}",
            (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if fps is not None:
            cv2.putText(
                vis,
                f"FPS: {fps}",
                (10, 156),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return vis

    def detect(
        self,
        frame: np.ndarray,
        live: bool = False,
        fps: int | None = None,
    ) -> tuple[np.ndarray, int, int, int]:
        """
        Run detection; returns annotated frame (boxes + count overlay), head_count,
        helmet_count, total_count.
        """
        if frame is None or frame.size == 0:
            z = np.zeros((240, 320, 3), dtype=np.uint8)
            return z, 0, 0, 0

        h0, w0 = frame.shape[:2]

        if live:
            proc = self.preprocess_live(frame.copy())
            small = cv2.resize(proc, (640, 640), interpolation=cv2.INTER_LINEAR)
            results = self._predict_arr(small, _CONF_LIVE)
            r0 = results[0]
            head_count, helmet_count = self._count_from_results(r0)
            self._smooth_heads.append(head_count)
            self._smooth_helmets.append(helmet_count)
            head_count = int(
                round(sum(self._smooth_heads) / max(1, len(self._smooth_heads)))
            )
            helmet_count = int(
                round(sum(self._smooth_helmets) / max(1, len(self._smooth_helmets)))
            )
            vis = r0.plot()
            if vis.shape[0] != h0 or vis.shape[1] != w0:
                vis = cv2.resize(vis, (w0, h0), interpolation=cv2.INTER_LINEAR)
        else:
            self._smooth_heads.clear()
            self._smooth_helmets.clear()
            results = self._predict_arr(frame, _CONF_IMAGE)
            r0 = results[0]
            head_count, helmet_count = self._count_from_results(r0)
            vis = r0.plot()

        total_count = head_count + helmet_count
        vis = self._overlay_counts(vis, head_count, helmet_count, total_count, fps)
        return vis, head_count, helmet_count, total_count
