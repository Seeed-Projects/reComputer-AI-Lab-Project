#!/usr/bin/env python3
"""Check the lightweight Raspberry Pi deployment package without Hailo hardware."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "app/infer_video_hailo.py",
    "runtime/hailo_detector.py",
    "runtime/yolo_postprocess.py",
    "configs/runtime.json",
    "configs/device_target.json",
    "input/demo.mp4",
    "models/hef/shelf_product_hailo8.hef",
    "models/hef/held_product_v4_hailo8.hef",
)

missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
if missing:
    raise SystemExit("Missing deployment files:\n" + "\n".join(missing))

config = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8"))
for model_name in ("shelf_model", "held_model"):
    model_path = ROOT / config[model_name]["path"]
    if not model_path.is_file():
        raise SystemExit(f"Missing configured HEF: {model_path}")

print("Deployment package structure and configuration are valid.")
