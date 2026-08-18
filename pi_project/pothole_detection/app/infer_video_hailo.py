#!/usr/bin/env python3
"""Run pothole video inference on Raspberry Pi 5 + Hailo-8."""
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
from pothole_monitor.processor import PotholeProcessor  # noqa: E402

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
        if model.get("input_mode", "uint8") not in {"uint8", "float32"}:
            errors.append("model.input_mode must be uint8 or float32")
        for name in ("confidence", "iou"):
            value = float(model.get(name, -1))
            if not 0.0 <= value <= 1.0:
                errors.append(f"model.{name} must be between 0 and 1")
    return errors


def source_value(value: str):
    return int(value) if value.isdigit() else str(resolve(value)) if "://" not in value else value


def _draw_overlay(frame, fps_ema: float, status: dict) -> None:
    """Draw FPS, visible pothole count and cumulative events on the frame."""
    cv2.rectangle(frame, (8, 8), (430, 72), (20, 20, 20), -1)
    cv2.putText(frame, f"Hailo-8 FPS {fps_ema:.1f}", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (90, 255, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, f"potholes={status['visible_tracks']}  cumulative={status['cumulative_events']}",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 210, 60), 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--source", help="video file path, RTSP URL, or camera index")
    parser.add_argument("--output", help="output annotated video path")
    parser.add_argument("--confidence", type=float, help="override model.confidence")
    parser.add_argument("--iou", type=float, help="override model.iou")
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
    processor = PotholeProcessor(cfg, ROOT)
    frames, total, cumulative = 0, 0, 0
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
            total += status["direct_detections"]
            cumulative = status["cumulative_events"]
            if not args.no_fps_overlay:
                _draw_overlay(annotated, fps_ema, status)
            writer.write(annotated)
            frames += 1
            if frames % 30 == 0:
                elapsed = time.perf_counter() - started
                logger.info("frame=%d fps=%.1f potholes=%d cumulative=%d", frames, fps_ema,
                            status["visible_tracks"], cumulative)
    finally:
        cap.release()
        writer.release()
        processor.release()
    elapsed = time.perf_counter() - started
    report = {"frames": frames, "duration_s": frames / fps, "processing_s": elapsed,
              "processing_fps": fps_ema, "direct_detections": total,
              "cumulative_events": cumulative, "output": str(output_path)}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("complete: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
