"""RK3576 RKNN YOLO11 pose based pull-up counter.

This version runs ``model/yolo11n-pose.rknn`` with RKNNLite2 on RK3576.
The model has one output shaped ``(1, 56, 8400)``:

    4 box values + 1 person score + 17 * (x, y, confidence)

Example:
    python3 pullup_counter.py --source video/input/test.mp4
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "model" / "yolo11n-pose.rknn"

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
KEYPOINT_COUNT = 17
POSE_FEATURES = 4 + 1 + KEYPOINT_COUNT * 3

SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


@dataclass
class PoseFrame:
    box: tuple[int, int, int, int]
    box_conf: float
    keypoints: np.ndarray
    keypoint_conf: np.ndarray
    nose_y: Optional[float]
    shoulder_y: Optional[float]
    wrist_y: Optional[float]


class RKNNPoseModel:
    """RKNNLite2 wrapper for the converted single-output YOLO11 Pose model."""

    def __init__(
        self,
        model_path: Path,
        imgsz: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        keypoint_conf_threshold: float = 0.35,
        npu_core: str = "auto",
    ) -> None:
        if imgsz != 640:
            raise ValueError("This RKNN model has a fixed 640x640 input; --imgsz must be 640")
        if not model_path.is_file():
            raise FileNotFoundError(f"RKNN model not found: {model_path}")

        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "RKNNLite2 is not installed. Install the AArch64 wheel matching "
                "the board's Python and RKNN Runtime versions."
            ) from exc

        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.rknn = RKNNLite()

        print(f"Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            self.rknn.release()
            raise RuntimeError(f"load_rknn failed with code {ret}: {model_path}")

        core_masks = {
            "0": getattr(RKNNLite, "NPU_CORE_0", None),
            "1": getattr(RKNNLite, "NPU_CORE_1", None),
            "2": getattr(RKNNLite, "NPU_CORE_2", None),
            "auto": getattr(RKNNLite, "NPU_CORE_AUTO", None),
        }
        core_mask = core_masks[npu_core]
        ret = (
            self.rknn.init_runtime(core_mask=core_mask)
            if core_mask is not None
            else self.rknn.init_runtime()
        )
        if ret != 0:
            self.rknn.release()
            raise RuntimeError(f"init_runtime failed with code {ret}")
        print(f"RKNN runtime initialized (NPU core: {npu_core})")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        """Letterbox BGR input, convert to RGB, and return uint8 NHWC."""

        original_height, original_width = frame.shape[:2]
        scale = min(self.imgsz / original_width, self.imgsz / original_height)
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.imgsz - resized_width) // 2
        pad_y = (self.imgsz - resized_height) // 2
        right = self.imgsz - resized_width - pad_x
        bottom = self.imgsz - resized_height - pad_y
        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_y,
            bottom,
            pad_x,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb[None, ...], dtype=np.uint8)
        info = LetterboxInfo(scale, pad_x, pad_y, original_width, original_height)
        return tensor, info

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty(0, dtype=np.int64)

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size:
            index = int(order[0])
            keep.append(index)
            if order.size == 1:
                break

            remaining = order[1:]
            xx1 = np.maximum(x1[index], x1[remaining])
            yy1 = np.maximum(y1[index], y1[remaining])
            xx2 = np.minimum(x2[index], x2[remaining])
            yy2 = np.minimum(y2[index], y2[remaining])
            intersection = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
            union = areas[index] + areas[remaining] - intersection
            iou = intersection / np.maximum(union, 1e-7)
            order = remaining[iou <= threshold]

        return np.asarray(keep, dtype=np.int64)

    def postprocess(
        self,
        outputs: list[np.ndarray],
        info: LetterboxInfo,
    ) -> Optional[PoseFrame]:
        """Decode (1, 56, 8400), apply NMS, and select the best person."""

        if not outputs or outputs[0] is None:
            return None

        output = np.asarray(outputs[0])
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise RuntimeError(f"Unexpected RKNN pose output shape: {output.shape}")

        if output.shape[0] == POSE_FEATURES:
            predictions = output.T
        elif output.shape[1] == POSE_FEATURES:
            predictions = output
        else:
            raise RuntimeError(
                f"Expected 56 pose features but received RKNN output {output.shape}"
            )

        scores = predictions[:, 4]
        valid = np.isfinite(scores) & (scores >= self.conf_threshold)
        predictions = predictions[valid]
        scores = scores[valid]
        if len(predictions) == 0:
            return None

        xywh = predictions[:, :4]
        boxes = np.empty_like(xywh, dtype=np.float32)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

        keep = self._nms(boxes, scores, self.iou_threshold)
        if keep.size == 0:
            return None

        # The video has one athlete. Confidence-first selection prevents a
        # distant bystander from replacing the primary subject.
        best = int(keep[0])
        box = boxes[best].copy()
        keypoint_data = predictions[best, 5:].reshape(KEYPOINT_COUNT, 3).copy()

        box[[0, 2]] = (box[[0, 2]] - info.pad_x) / info.scale
        box[[1, 3]] = (box[[1, 3]] - info.pad_y) / info.scale
        keypoint_data[:, 0] = (keypoint_data[:, 0] - info.pad_x) / info.scale
        keypoint_data[:, 1] = (keypoint_data[:, 1] - info.pad_y) / info.scale

        box[[0, 2]] = np.clip(box[[0, 2]], 0, info.original_width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, info.original_height - 1)
        keypoint_data[:, 0] = np.clip(
            keypoint_data[:, 0], 0, info.original_width - 1
        )
        keypoint_data[:, 1] = np.clip(
            keypoint_data[:, 1], 0, info.original_height - 1
        )

        xy = keypoint_data[:, :2].astype(np.float32, copy=False)
        keypoint_conf = keypoint_data[:, 2].astype(np.float32, copy=False)
        return make_pose_frame(
            box,
            float(scores[best]),
            xy,
            keypoint_conf,
            self.keypoint_conf_threshold,
        )

    def predict(self, frame: np.ndarray) -> tuple[Optional[PoseFrame], float]:
        tensor, info = self.preprocess(frame)
        start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor])
        inference_ms = (time.perf_counter() - start) * 1000.0
        if outputs is None:
            raise RuntimeError("RKNN inference returned no outputs")
        return self.postprocess(outputs, info), inference_ms

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


class PullUpCounter:
    """Hysteresis state machine for pull-up repetitions."""

    def __init__(self, min_keypoint_conf: float = 0.35) -> None:
        self.min_keypoint_conf = min_keypoint_conf
        self.count = 0
        self.state = "WAITING"
        self.bar_y: Optional[float] = None
        self.smoothed_nose_y: Optional[float] = None
        self.smoothed_shoulder_y: Optional[float] = None
        self.stable_frames = 0
        self.last_event = ""

    @staticmethod
    def _mean_visible(values: list[Optional[float]]) -> Optional[float]:
        visible = [value for value in values if value is not None]
        return float(np.mean(visible)) if visible else None

    def update(self, pose: PoseFrame) -> None:
        if pose.wrist_y is not None:
            self.bar_y = (
                pose.wrist_y
                if self.bar_y is None
                else 0.04 * pose.wrist_y + 0.96 * self.bar_y
            )

        if (pose.nose_y is None and pose.shoulder_y is None) or self.bar_y is None:
            self.state = "DETECTING"
            self.stable_frames = 0
            return

        if pose.nose_y is not None:
            self.smoothed_nose_y = (
                pose.nose_y
                if self.smoothed_nose_y is None
                else 0.35 * pose.nose_y + 0.65 * self.smoothed_nose_y
            )
        if pose.shoulder_y is not None:
            self.smoothed_shoulder_y = (
                pose.shoulder_y
                if self.smoothed_shoulder_y is None
                else 0.35 * pose.shoulder_y + 0.65 * self.smoothed_shoulder_y
            )

        height = max(pose.box[3] - pose.box[1], 1)
        relative_nose = (
            (self.smoothed_nose_y - self.bar_y) / height
            if pose.nose_y is not None and self.smoothed_nose_y is not None
            else None
        )
        relative_shoulder = (
            (self.smoothed_shoulder_y - self.bar_y) / height
            if self.smoothed_shoulder_y is not None
            else None
        )

        active_signal = relative_nose if relative_nose is not None else relative_shoulder
        reached_top = active_signal is not None and active_signal <= 0.08
        reached_bottom = relative_nose is not None and relative_nose >= 0.12

        if self.state in ("WAITING", "DETECTING", "NO PERSON"):
            if reached_bottom:
                self.stable_frames += 1
                if self.stable_frames >= 3:
                    self.state = "HANG"
                    self.stable_frames = 0
            else:
                self.stable_frames = 0
        elif self.state == "HANG":
            self.state = "PULLING"
            if reached_top:
                self.stable_frames += 1
                if self.stable_frames >= 2:
                    self.count += 1
                    self.last_event = "REP COMPLETE"
                    self.state = "TOP"
                    self.stable_frames = 0
            else:
                self.stable_frames = 0
        elif self.state == "PULLING":
            if reached_top:
                self.stable_frames += 1
                if self.stable_frames >= 2:
                    self.count += 1
                    self.last_event = "REP COMPLETE"
                    self.state = "TOP"
                    self.stable_frames = 0
            elif reached_bottom:
                self.state = "HANG"
                self.stable_frames = 0
        elif self.state == "TOP" and reached_bottom:
            self.state = "HANG"
            self.stable_frames = 0

    def overlay_values(
        self, pose: Optional[PoseFrame]
    ) -> tuple[Optional[float], Optional[float]]:
        if pose is None or self.bar_y is None or pose.nose_y is None:
            return self.bar_y, None
        height = max(pose.box[3] - pose.box[1], 1)
        return self.bar_y, (pose.nose_y - self.bar_y) / height


def _point(
    values: np.ndarray,
    confidences: np.ndarray,
    index: int,
    threshold: float,
) -> Optional[tuple[float, float]]:
    if index >= len(values) or index >= len(confidences):
        return None
    if float(confidences[index]) < threshold:
        return None
    x, y = values[index]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def make_pose_frame(
    box: np.ndarray,
    box_conf: float,
    xy: np.ndarray,
    keypoint_conf: np.ndarray,
    keypoint_conf_threshold: float,
) -> PoseFrame:
    nose = _point(xy, keypoint_conf, NOSE, keypoint_conf_threshold)
    left_shoulder = _point(xy, keypoint_conf, LEFT_SHOULDER, keypoint_conf_threshold)
    right_shoulder = _point(xy, keypoint_conf, RIGHT_SHOULDER, keypoint_conf_threshold)
    left_wrist = _point(xy, keypoint_conf, LEFT_WRIST, keypoint_conf_threshold)
    right_wrist = _point(xy, keypoint_conf, RIGHT_WRIST, keypoint_conf_threshold)

    shoulder_y = PullUpCounter._mean_visible([
        left_shoulder[1] if left_shoulder else None,
        right_shoulder[1] if right_shoulder else None,
    ])
    wrist_y = PullUpCounter._mean_visible([
        left_wrist[1] if left_wrist else None,
        right_wrist[1] if right_wrist else None,
    ])
    x1, y1, x2, y2 = box
    return PoseFrame(
        box=(int(x1), int(y1), int(x2), int(y2)),
        box_conf=box_conf,
        keypoints=xy,
        keypoint_conf=keypoint_conf,
        nose_y=nose[1] if nose else None,
        shoulder_y=shoulder_y,
        wrist_y=wrist_y,
    )


def draw_pose(frame: np.ndarray, pose: PoseFrame, threshold: float) -> None:
    x1, y1, x2, y2 = pose.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 4)
    label = f"Person {pose.box_conf:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    )
    label_y = max(y1 - 10, text_height + 8)
    cv2.rectangle(
        frame,
        (x1, label_y - text_height - 10),
        (x1 + text_width + 12, label_y + 4),
        (40, 220, 80),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 6, label_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (10, 30, 10),
        2,
        cv2.LINE_AA,
    )

    for start, end in SKELETON:
        point_a = _point(pose.keypoints, pose.keypoint_conf, start, threshold)
        point_b = _point(pose.keypoints, pose.keypoint_conf, end, threshold)
        if point_a is not None and point_b is not None:
            cv2.line(
                frame,
                (int(point_a[0]), int(point_a[1])),
                (int(point_b[0]), int(point_b[1])),
                (255, 210, 50),
                4,
                cv2.LINE_AA,
            )

    for index, (x, y) in enumerate(pose.keypoints):
        if float(pose.keypoint_conf[index]) < threshold:
            continue
        color = (0, 80, 255) if index == NOSE else (255, 255, 255)
        cv2.circle(
            frame,
            (int(x), int(y)),
            6 if index == NOSE else 5,
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    counter: PullUpCounter,
    pose: Optional[PoseFrame],
    show_debug: bool,
) -> None:
    _, width = frame.shape[:2]
    panel_width = min(440, max(330, width // 3))
    panel_height = 165 if show_debug else 135
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (20, 20), (20 + panel_width, 20 + panel_height), (8, 12, 22), -1
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(
        frame,
        (20, 20),
        (20 + panel_width, 20 + panel_height),
        (80, 210, 255),
        2,
    )
    cv2.putText(
        frame, "PULL-UPS", (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"COUNT  {counter.count}", (40, 118), cv2.FONT_HERSHEY_SIMPLEX,
        1.7, (0, 80, 255), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame, counter.state, (230, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.85, (100, 240, 160), 2, cv2.LINE_AA
    )

    if show_debug:
        bar_y, relative_nose = counter.overlay_values(pose)
        debug = (
            "bar --  nose --"
            if bar_y is None or relative_nose is None
            else f"bar {bar_y:.0f}  nose/bar {relative_nose:.2f}"
        )
        cv2.putText(
            frame, debug, (40, 155), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (205, 215, 225), 1, cv2.LINE_AA
        )


def process_video(
    source: Path,
    output: Path,
    csv_path: Path,
    model_path: Path,
    conf: float,
    iou: float,
    keypoint_conf: float,
    imgsz: int,
    npu_core: str,
    show_debug: bool,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output}")

    model: Optional[RKNNPoseModel] = None
    counter = PullUpCounter(keypoint_conf)
    rows: list[dict[str, object]] = []
    frame_index = 0
    lost_frames = 0

    try:
        model = RKNNPoseModel(
            model_path=model_path,
            imgsz=imgsz,
            conf_threshold=conf,
            iou_threshold=iou,
            keypoint_conf_threshold=keypoint_conf,
            npu_core=npu_core,
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pose, inference_ms = model.predict(frame)
            if pose is not None:
                lost_frames = 0
                counter.update(pose)
                draw_pose(frame, pose, keypoint_conf)
            else:
                lost_frames += 1
                if lost_frames > 15:
                    counter.state = "NO PERSON"
                    counter.stable_frames = 0

            draw_hud(frame, counter, pose, show_debug)
            writer.write(frame)
            bar_y, relative_nose = counter.overlay_values(pose)
            rows.append({
                "frame": frame_index,
                "time_sec": round(frame_index / fps, 3),
                "count": counter.count,
                "state": counter.state,
                "person_conf": round(pose.box_conf, 4) if pose else "",
                "bar_y": round(bar_y, 2) if bar_y is not None else "",
                "nose_y": round(pose.nose_y, 2)
                if pose and pose.nose_y is not None else "",
                "nose_bar_ratio": round(relative_nose, 4)
                if relative_nose is not None else "",
                "inference_ms": round(inference_ms, 3),
            })

            frame_index += 1
            if frame_index % 30 == 0 or frame_index == total:
                print(
                    f"Progress: {frame_index}/{total} frames, "
                    f"count: {counter.count}, NPU: {inference_ms:.1f} ms"
                )
    finally:
        cap.release()
        writer.release()
        if model is not None:
            model.release()

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = (
            list(rows[0].keys())
            if rows
            else ["frame", "time_sec", "count", "state", "inference_ms"]
        )
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    return counter.count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count pull-ups on RK3576 with a YOLO11 Pose RKNN model"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_DIR / "video" / "input" / "test.mp4",
        help="Input video path"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output video path; defaults to video/output"
    )
    parser.add_argument("--csv", type=Path, default=None, help="Per-frame CSV path")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"RKNN model path (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--keypoint-conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640, help="Fixed RKNN input size")
    parser.add_argument(
        "--npu-core", choices=("auto", "0", "1", "2"), default="auto"
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        args.output = args.source.parent.parent / "output" / f"{args.source.stem}_counted.mp4"
    if args.csv is None:
        args.csv = args.output.with_suffix(".csv")

    print(f"Input video: {args.source}")
    print(f"Output video: {args.output}")
    print(f"Model: {args.model}")
    count = process_video(
        source=args.source,
        output=args.output,
        csv_path=args.csv,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        keypoint_conf=args.keypoint_conf,
        imgsz=args.imgsz,
        npu_core=args.npu_core,
        show_debug=args.debug,
    )
    print(f"Done. Final pull-up count: {count}")
    print(f"Counted video: {args.output}")
    print(f"Per-frame data: {args.csv}")


if __name__ == "__main__":
    main()
