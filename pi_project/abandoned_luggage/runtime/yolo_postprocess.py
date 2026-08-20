"""YOLOv11 preprocessing and Hailo NMS output decoding (COCO-80 classes)."""
from __future__ import annotations
from typing import Mapping, Sequence
import cv2
import numpy as np

PERSON_CLASS = 0
BAG_CLASSES = {24, 26, 28}  # backpack, handbag, suitcase


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


def _nms(xyxy, scores, conf, iou):
    if not len(scores):
        return []
    xywh = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    return [int(i) for i in np.asarray(keep).reshape(-1)] if len(keep) else []


def postprocess_hailo_nms(output, img_h: int, img_w: int, conf: float, iou: float, imgsz: int):
    """Decode one Hailo NMS output tensor into (boxes_xyxy, scores, class_ids).

    Supports the layout HailoRT exposes for YOLOv8-family NMS postprocess:
      - [N, 6] / [1, N, 6]: x1,y1,x2,y2,score,class_id
      - [C, M, 5] / [1, C, M, 5]: per-class grid, row index = class id,
        each row is yxwh center-format or yxyx
      - [1, M, 5] single class
    """
    value = output
    if isinstance(value, np.ndarray) and value.dtype == object:
        rows = [np.asarray(item) for item in value.reshape(-1)]
        value = np.concatenate(
            [row.reshape(-1, row.shape[-1]) for row in rows if row.size], axis=0
        ) if rows else np.empty((0, 5))

    pred = np.asarray(value)
    if pred.size == 0:
        return [], [], []

    # Squeeze leading singleton dims, keep 2-D at most.
    while pred.ndim > 2 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 1 and pred.shape[0] in (5, 6):
        pred = pred.reshape(1, -1)
    if pred.ndim == 3 and pred.shape[-1] in (5, 6):
        # [C, M, 5] → rows of (y,x,y,x,score,class_id), matching Hailo 6-col order
        pred = np.concatenate(
            [np.column_stack((pred[c], np.full((pred.shape[1], 1), c))) for c in range(pred.shape[0])],
            axis=0,
        ) if pred.shape[0] else np.empty((0, 6))

    if pred.ndim != 2 or pred.shape[1] not in (5, 6):
        raise RuntimeError(f"unsupported Hailo NMS output shape: {pred.shape}")

    pred = pred.astype(np.float32)
    six_col = pred.shape[1] == 6
    scores = pred[:, 4]
    mask = scores >= conf
    pred, scores = pred[mask], scores[mask]
    if not len(pred):
        return [], [], []

    if six_col:
        class_ids = pred[:, 5].astype(np.int32)
        xyxy = pred[:, [1, 0, 3, 2]]  # Hailo 6-col uses yxyx order
    else:
        class_ids = np.zeros(len(pred), dtype=np.int32)
        xyxy = pred[:, [1, 0, 3, 2]]  # 5-col YXYX, single class (id 0)

    normalized = float(np.nanmax(np.abs(xyxy))) <= 2.0
    keep = _nms(xyxy, scores, conf, iou)

    boxes, confs, cids = [], [], []
    for index in keep:
        box = _restore(xyxy[index], img_h, img_w, imgsz, normalized)
        if box is not None:
            boxes.append(box)
            confs.append(float(scores[index]))
            cids.append(int(class_ids[index]))
    return boxes, confs, cids


def postprocess_auto(outputs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
                     img_h, img_w, conf, iou, imgsz):
    tensors = list(outputs.values()) if isinstance(outputs, Mapping) else list(outputs)
    if len(tensors) != 1:
        raise RuntimeError(
            "HEF must expose one Hailo NMS output; got "
            + str([np.asarray(x).shape for x in tensors])
        )
    return postprocess_hailo_nms(np.asarray(tensors[0]), img_h, img_w, conf, iou, imgsz)