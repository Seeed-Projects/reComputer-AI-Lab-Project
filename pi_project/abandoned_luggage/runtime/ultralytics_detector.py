"""Ultralytics (CPU) detector — PC demo mode, same interface as HailoDetector.

Use only on a PC WITHOUT Hailo hardware, to preview the dashboard and record
demo material. The Raspberry Pi deployment always uses runtime.hailo_detector.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("runtime.ultralytics_detector")


class UltralyticsDetector:
    def __init__(self, weights_path: str | Path, imgsz: int, conf: float, iou: float,
                 input_mode: str = "uint8") -> None:
        self.weights_path = Path(weights_path)
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.input_mode = input_mode.lower()
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is unavailable; pip install ultralytics") from exc
        # Keep Ultralytics caches inside this project instead of $HOME.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self.weights_path.parent.parent))
        try:
            with open(self.weights_path, "rb") as handle:
                self._model = YOLO(str(self.weights_path), verbose=False)
        except Exception as exc:  # noqa: BLE001 — surface a clear PC-side error
            raise RuntimeError(f"failed to load Ultralytics weights: {exc}") from exc
        self._first = True
        logger.info("loaded %s (CPU demo mode, pc only)", self.weights_path.name)

    def predict(self, frame_bgr: np.ndarray):
        """Return (boxes_xyxy, confidences, class_ids) in original frame pixel space."""
        result = self._model.predict(
            frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device="cpu", verbose=False,
        )[0]
        if self._first:
            logger.info("ultralytics result: %d detections (CPU)", len(result.boxes))
            self._first = False
        boxes, confs, cids = [], [], []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, cid in zip(xyxy, scores, classes):
                boxes.append([float(v) for v in box])
                confs.append(float(score))
                cids.append(int(cid))
        return boxes, confs, cids

    def release(self) -> None:
        self._model = None