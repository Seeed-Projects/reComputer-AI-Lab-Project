"""Shared Hailo pothole frame processor used by MP4 and web entry points."""
from __future__ import annotations
from pathlib import Path
import cv2
from runtime.hailo_detector import HailoDetector
from .tracker import TemporalTracker


def in_road_roi(box, width: int, height: int, cfg: dict) -> bool:
    if not cfg.get("enabled", False):
        return True
    cx = (box[0] + box[2]) / 2 / width
    bottom = box[3] / height
    horizon = float(cfg.get("horizon_y", 0.18))
    if bottom < horizon:
        return False
    progress = min(1.0, max(0.0, (bottom - horizon) / max(1e-6, 1.0 - horizon)))
    left = float(cfg.get("horizon_left", 0.22)) * (1.0 - progress)
    right = 1.0 - (1.0 - float(cfg.get("horizon_right", 0.78))) * (1.0 - progress)
    return left <= cx <= right


class PotholeProcessor:
    def __init__(self, cfg: dict, root: Path):
        model_cfg = cfg["model"]
        model_path = Path(model_cfg["path"])
        if not model_path.is_absolute():
            model_path = root / model_path
        self.detector = HailoDetector(model_path, model_cfg["imgsz"], model_cfg["confidence"],
                                      model_cfg["iou"], model_cfg.get("input_mode", "uint8"))
        self.roi_cfg = cfg.get("road_roi", {})
        self.tracker = TemporalTracker(**cfg.get("temporal", {}))

    def process(self, frame):
        boxes, confs = self.detector.predict(frame)
        height, width = frame.shape[:2]
        filtered = [(box, conf) for box, conf in zip(boxes, confs)
                    if in_road_roi(box, width, height, self.roi_cfg)]
        boxes = [item[0] for item in filtered]
        confs = [item[1] for item in filtered]
        tracks = self.tracker.update(boxes, confs)
        output = frame.copy()
        direct_count = 0
        for track in tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in track.box)
            if track.direct:
                color, label = (255, 90, 20), f"pothole {track.confidence:.2f}"
                direct_count += 1
            else:
                color, label = (0, 200, 255), "pothole tracked"
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color, 2, cv2.LINE_AA)
        return output, {
            "raw_detections": len(boxes), "visible_tracks": len(tracks),
            "direct_detections": direct_count, "tracked_only": len(tracks) - direct_count,
            "cumulative_events": self.tracker.total_events,
        }

    def release(self):
        self.detector.release()
