#!/usr/bin/env python3
"""Run abandoned-luggage video inference on Raspberry Pi 5 + Hailo-8."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
import sys
import time
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from abandoned_monitor.processor import AbandonedProcessor  # noqa: E402

logger = logging.getLogger("infer_video_hailo")


def resolve(path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def validate_config(cfg: dict, require_model=True) -> list[str]:
    errors = []
    for key in ("source", "output", "model"):
        if key not in cfg:
            errors.append(f"missing config key: {key}")
    model = cfg.get("model", {})
    if not {"path", "imgsz", "confidence", "iou"}.issubset(model):
        errors.append("incomplete model config")
    elif require_model and not resolve(model["path"]).exists():
        errors.append(f"missing HEF model: {resolve(model['path'])}")
    if model:
        if int(model.get("imgsz", 0)) != 640:
            errors.append("this HEF requires model.imgsz=640")
        for name in ("confidence", "iou"):
            value = float(model.get(name, -1))
            if not 0.0 <= value <= 1.0:
                errors.append(f"model.{name} must be between 0 and 1")
    roi = cfg.get("roi", {})
    if roi.get("enabled", True):
        rect = roi.get("rect", [200, 200, 1100, 800])
        if len(rect) != 4 or not all(isinstance(v, (int, float)) for v in rect):
            errors.append("roi.rect must be [x1, y1, x2, y2]")
    return errors


def source_value(value: str):
    return int(value) if value.isdigit() else str(resolve(value)) if "://" not in value else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--source", help="video file path, RTSP URL, or camera index")
    parser.add_argument("--output", help="output annotated video path")
    parser.add_argument("--confidence", type=float, help="override model.confidence")
    parser.add_argument("--iou", type=float, help="override model.iou")
    parser.add_argument("--device", default="hailo", choices=["hailo", "cpu"],
                        help="hailo = Hailo-8 (Pi); cpu = Ultralytics demo on PC")
    parser.add_argument("--no-fps-overlay", action="store_true", help="hide FPS/info overlay")
    parser.add_argument("--check-config", action="store_true", help="validate config and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    if args.confidence is not None:
        cfg["model"]["confidence"] = args.confidence
    if args.iou is not None:
        cfg["model"]["iou"] = args.iou
    errors = validate_config(cfg, require_model=not args.check_config)
    if errors:
        for error in errors:
            logger.error(error)
        return 2
    if args.check_config:
        logger.info("configuration is valid")
        return 0

    source = source_value(args.source or str(cfg["source"]))
    output_path = resolve(args.output or cfg["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {source}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*cfg.get("codec", "mp4v")),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot create output video: {output_path}")

    if args.device == "cpu":
        from runtime.ultralytics_detector import UltralyticsDetector
        mc = cfg["model"]
        detector = UltralyticsDetector(
            resolve(mc.get("pt_path", "models/source_pt/yolo11m.pt")),
            mc["imgsz"], mc["confidence"], mc["iou"], mc.get("input_mode", "uint8"))
        processor = AbandonedProcessor(cfg, ROOT, detector=detector)
    else:
        processor = AbandonedProcessor(cfg, ROOT)

    frames, total_alarms = 0, 0
    fps_ema = 0.0
    started = time.perf_counter()
    previous = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, status = processor.process(frame)
            now = time.perf_counter()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps_ema = instant if not fps_ema else 0.9 * fps_ema + 0.1 * instant
            total_alarms += int(status["abandoned"] > 0)
            if not args.no_fps_overlay:
                _draw_overlay(annotated, fps_ema, status)
            writer.write(annotated)
            frames += 1
            if frames % 30 == 0:
                logger.info("frame=%d fps=%.1f persons=%d bags=%d abandoned=%d",
                            frames, fps_ema, status["persons"], status["bags"],
                            status["abandoned"])
    finally:
        cap.release()
        writer.release()
        processor.release()
    elapsed = time.perf_counter() - started
    report = {"frames": frames, "duration_s": frames / fps, "processing_s": elapsed,
              "processing_fps": fps_ema, "alarm_frames": total_alarms,
              "output": str(output_path)}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("complete: %s", report)
    return 0


def _draw_overlay(frame, fps_ema: float, status: dict) -> None:
    abandoned = status["abandoned"]
    cv2.rectangle(frame, (8, 8), (520, 96), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS {fps_ema:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (90, 255, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, f"persons={status['persons']}  bags={status['bags']}  static={status['static_bags']}",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 60), 2, cv2.LINE_AA)
    color = (0, 0, 255) if abandoned else (200, 200, 200)
    cv2.putText(frame, f"ABANDONED={abandoned}", (18, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())