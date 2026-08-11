"""YOLO11 preprocessing and Hailo output-layout-aware post-processing."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np


def letterbox_rgb(frame_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """Create the square RGB input used when the ONNX models were exported."""
    height, width = frame_bgr.shape[:2]
    scale = min(imgsz / width, imgsz / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (imgsz - new_w) // 2
    top = (imgsz - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        imgsz - new_h - top,
        left,
        imgsz - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def _restore_xyxy(
    box: Sequence[float], img_h: int, img_w: int, imgsz: int, normalized: bool = False
) -> list[float] | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    if normalized:
        x1, x2 = x1 * imgsz, x2 * imgsz
        y1, y2 = y1 * imgsz, y2 * imgsz
    scale = min(imgsz / img_w, imgsz / img_h)
    pad_w = (imgsz - img_w * scale) / 2.0
    pad_h = (imgsz - img_h * scale) / 2.0
    restored = [
        float(np.clip((x1 - pad_w) / scale, 0, img_w)),
        float(np.clip((y1 - pad_h) / scale, 0, img_h)),
        float(np.clip((x2 - pad_w) / scale, 0, img_w)),
        float(np.clip((y2 - pad_h) / scale, 0, img_h)),
    ]
    return restored if restored[2] > restored[0] and restored[3] > restored[1] else None


def _nms_restore(
    xyxy: np.ndarray,
    scores: np.ndarray,
    img_h: int,
    img_w: int,
    imgsz: int,
    conf: float,
    iou: float,
    normalized: bool = False,
) -> tuple[list[list[float]], list[float]]:
    if not len(scores):
        return [], []
    xywh = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    indices = [int(i) for i in np.asarray(keep).reshape(-1)] if len(keep) else []
    boxes: list[list[float]] = []
    confs: list[float] = []
    for idx in indices:
        restored = _restore_xyxy(xyxy[idx], img_h, img_w, imgsz, normalized)
        if restored is not None:
            boxes.append(restored)
            confs.append(float(scores[idx]))
    return boxes, confs


def postprocess_standard(
    output: np.ndarray, img_h: int, img_w: int, conf: float, iou: float, imgsz: int
) -> tuple[list[list[float]], list[float]]:
    """Decode Ultralytics output shaped [1, 4+classes, anchors]."""
    pred = np.squeeze(np.asarray(output))
    if pred.ndim != 2:
        raise RuntimeError(f"standard YOLO output must be 2-D after squeeze, got {pred.shape}")
    if 5 <= pred.shape[0] <= 256 and pred.shape[1] > pred.shape[0]:
        pred = pred.T
    if pred.shape[1] < 5:
        raise RuntimeError(f"standard YOLO output needs at least 5 values per anchor, got {pred.shape}")
    xywh = pred[:, :4].astype(np.float32)
    scores = pred[:, 4:].astype(np.float32).max(axis=1)
    mask = scores >= conf
    xywh, scores = xywh[mask], scores[mask]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return _nms_restore(xyxy, scores, img_h, img_w, imgsz, conf, iou)


def postprocess_hailo_nms(
    output: np.ndarray, img_h: int, img_w: int, conf: float, iou: float, imgsz: int
) -> tuple[list[list[float]], list[float]]:
    """Decode Hailo NMS tensors; one-class rows are [ymin,xmin,ymax,xmax,score]."""
    value = output
    if isinstance(value, np.ndarray) and value.dtype == object:
        rows = [np.asarray(item) for item in value.reshape(-1)]
        value = np.concatenate([row.reshape(-1, row.shape[-1]) for row in rows if row.size], axis=0) if rows else np.empty((0, 5))
    pred = np.asarray(value)
    if pred.size == 0:
        return [], []
    while pred.ndim > 2 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 1 and pred.shape[0] in (5, 6):
        pred = pred.reshape(1, -1)
    if pred.ndim >= 3 and pred.shape[-1] in (5, 6):
        pred = pred.reshape(-1, pred.shape[-1])
    if pred.ndim != 2 or pred.shape[1] not in (5, 6):
        raise RuntimeError(f"unsupported Hailo NMS output shape: {pred.shape}")
    pred = pred.astype(np.float32)
    scores = pred[:, 4]
    pred = pred[scores >= conf]
    scores = scores[scores >= conf]
    if not len(pred):
        return [], []
    # Hailo NMS order is YXYX. Coordinates normally are normalized to 0..1.
    xyxy = pred[:, [1, 0, 3, 2]]
    normalized = float(np.nanmax(np.abs(xyxy))) <= 2.0
    return _nms_restore(xyxy, scores, img_h, img_w, imgsz, conf, iou, normalized)


def postprocess_auto(
    outputs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    img_h: int,
    img_w: int,
    conf: float,
    iou: float,
    imgsz: int,
) -> tuple[list[list[float]], list[float]]:
    tensors = list(outputs.values()) if isinstance(outputs, Mapping) else list(outputs)
    if len(tensors) != 1:
        raise RuntimeError(
            "HEF currently exposes multiple raw output tensors: "
            + ", ".join(str(np.asarray(item).shape) for item in tensors)
            + ". Compile with the YOLOv11 Hailo NMS post-process enabled."
        )
    tensor = np.asarray(tensors[0])
    # Hailo NMS uses [..., detections, 5/6]. Do not squeeze first: a single
    # detection shaped [1, 1, 1, 5] would collapse to [5] and be mistaken for
    # a raw Ultralytics tensor. Empty outputs [1, 1, 0, 5] are valid as well.
    if tensor.dtype == object or (tensor.ndim >= 2 and tensor.shape[-1] in (5, 6)):
        return postprocess_hailo_nms(tensor, img_h, img_w, conf, iou, imgsz)
    return postprocess_standard(tensor, img_h, img_w, conf, iou, imgsz)
