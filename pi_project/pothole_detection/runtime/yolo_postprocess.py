"""YOLOv8 preprocessing and Hailo NMS output decoding."""
from __future__ import annotations
from typing import Mapping, Sequence
import cv2
import numpy as np


def letterbox_rgb(frame_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    scale = min(imgsz / width, imgsz / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left, top = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, top, imgsz - new_h - top, left, imgsz - new_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def _restore(box, img_h: int, img_w: int, imgsz: int, normalized: bool) -> list[float] | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    if normalized:
        x1, x2, y1, y2 = x1 * imgsz, x2 * imgsz, y1 * imgsz, y2 * imgsz
    scale = min(imgsz / img_w, imgsz / img_h)
    pad_w, pad_h = (imgsz - img_w * scale) / 2, (imgsz - img_h * scale) / 2
    result = [
        float(np.clip((x1 - pad_w) / scale, 0, img_w)),
        float(np.clip((y1 - pad_h) / scale, 0, img_h)),
        float(np.clip((x2 - pad_w) / scale, 0, img_w)),
        float(np.clip((y2 - pad_h) / scale, 0, img_h)),
    ]
    return result if result[2] > result[0] and result[3] > result[1] else None


def _nms_restore(xyxy, scores, img_h, img_w, imgsz, conf, iou, normalized=False):
    if not len(scores):
        return [], []
    xywh = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    indices = [int(i) for i in np.asarray(keep).reshape(-1)] if len(keep) else []
    boxes, confs = [], []
    for index in indices:
        box = _restore(xyxy[index], img_h, img_w, imgsz, normalized)
        if box is not None:
            boxes.append(box)
            confs.append(float(scores[index]))
    return boxes, confs


def postprocess_hailo_nms(output, img_h: int, img_w: int, conf: float, iou: float, imgsz: int):
    value = output
    if isinstance(value, np.ndarray) and value.dtype == object:
        rows = [np.asarray(item) for item in value.reshape(-1)]
        value = np.concatenate(
            [row.reshape(-1, row.shape[-1]) for row in rows if row.size], axis=0
        ) if rows else np.empty((0, 5))
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
    mask = scores >= conf
    pred, scores = pred[mask], scores[mask]
    if not len(pred):
        return [], []
    xyxy = pred[:, [1, 0, 3, 2]]  # Hailo NMS is YXYX.
    normalized = float(np.nanmax(np.abs(xyxy))) <= 2.0
    return _nms_restore(xyxy, scores, img_h, img_w, imgsz, conf, iou, normalized)


def postprocess_standard(output, img_h: int, img_w: int, conf: float, iou: float, imgsz: int):
    pred = np.squeeze(np.asarray(output))
    if pred.ndim == 1 and pred.shape[0] >= 5:
        pred = pred.reshape(1, -1)
    if pred.ndim != 2:
        raise RuntimeError(f"standard YOLO output must be 2-D, got {pred.shape}")
    if 5 <= pred.shape[0] <= 256 and pred.shape[1] > pred.shape[0]:
        pred = pred.T
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


def postprocess_auto(outputs: Mapping[str, np.ndarray] | Sequence[np.ndarray], img_h, img_w, conf, iou, imgsz):
    tensors = list(outputs.values()) if isinstance(outputs, Mapping) else list(outputs)
    if len(tensors) != 1:
        raise RuntimeError("HEF must expose one Hailo NMS output; got " + str([np.asarray(x).shape for x in tensors]))
    tensor = np.asarray(tensors[0])
    if tensor.dtype == object or (tensor.ndim >= 2 and tensor.shape[-1] in (5, 6)):
        return postprocess_hailo_nms(tensor, img_h, img_w, conf, iou, imgsz)
    return postprocess_standard(tensor, img_h, img_w, conf, iou, imgsz)
