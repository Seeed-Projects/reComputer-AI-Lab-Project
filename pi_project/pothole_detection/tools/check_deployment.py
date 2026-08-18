#!/usr/bin/env python3
"""Static deployment package validation; safe to run without Hailo hardware."""
from pathlib import Path
import hashlib
import json
import py_compile
import sys

ROOT = Path(__file__).resolve().parent.parent
HEF = ROOT / "models/hef/pothole_yolov8n_hailo8.hef"
EXPECTED_HEF_SHA256 = "6acec07e677cf4ae54919e0dd7105799ea8a50f1f48bd72528a6550b611b6856"
required = ["configs/runtime.json", "configs/device_target.json", "runtime/hailo_detector.py",
            "runtime/yolo_postprocess.py", "pothole_monitor/processor.py",
            "app/infer_video_hailo.py", "web_detection.py", "input/demo.mp4",
            "models/hef/pothole_yolov8n_hailo8.hef", "Dockerfile", "README.md"]
errors = [f"missing {item}" for item in required if not (ROOT / item).exists()]
cfg = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8"))
for source in ROOT.rglob("*.py"):
    try:
        py_compile.compile(str(source), doraise=True)
    except Exception as exc:
        errors.append(f"compile failed {source.relative_to(ROOT)}: {exc}")
if int(cfg["model"]["imgsz"]) != 640:
    errors.append("HEF and runtime input size must be 640")
if cfg["model"]["path"] != "models/hef/pothole_yolov8n_hailo8.hef":
    errors.append("runtime config does not select the YOLOv8n HEF")
if HEF.is_file():
    digest = hashlib.sha256(HEF.read_bytes()).hexdigest()
    if digest != EXPECTED_HEF_SHA256:
        errors.append(f"unexpected YOLOv8n HEF SHA256: {digest}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"deployment package is structurally valid; HEF SHA256={EXPECTED_HEF_SHA256}")
