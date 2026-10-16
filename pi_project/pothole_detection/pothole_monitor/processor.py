"""Shared Hailo pothole frame processor used by MP4 and web entry points."""
from __future__ import annotations
from pathlib import Path
from runtime.hailo_detector import HailoDetector
from .tracker import TemporalTracker
from .overlay import SurfaceOverlay


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
    def __init__(self, cfg: dict, root: Path, detector=None):
        model_cfg = cfg["model"]
        if detector is not None:
            self.detector = detector
        else:
            model_path = Path(model_cfg["path"])
            if not model_path.is_absolute():
                model_path = root / model_path
            self.detector = HailoDetector(model_path, model_cfg["imgsz"], model_cfg["confidence"],
                                          model_cfg["iou"], model_cfg.get("input_mode", "uint8"))
        self.roi_cfg = cfg.get("road_roi", {})
        self.tracker = TemporalTracker(**cfg.get("temporal", {}))
        self.overlay = SurfaceOverlay(cfg.get("overlay", {}))

    def process(self, frame):
        boxes, confs = self.detector.predict(frame)
        height, width = frame.shape[:2]
        filtered = [(box, conf) for box, conf in zip(boxes, confs)
                    if in_road_roi(box, width, height, self.roi_cfg)]
        boxes = [item[0] for item in filtered]
        confs = [item[1] for item in filtered]
        tracks = self.tracker.update(boxes, confs)
        output, overlay_status = self.overlay.render(frame, tracks)
        direct_count = sum(track.direct for track in tracks)
        return output, {
            "raw_detections": len(boxes), "visible_tracks": len(tracks),
            "direct_detections": direct_count, "tracked_only": len(tracks) - direct_count,
            "cumulative_events": self.tracker.total_events,
            **overlay_status,
        }

    def release(self):
        self.detector.release()
