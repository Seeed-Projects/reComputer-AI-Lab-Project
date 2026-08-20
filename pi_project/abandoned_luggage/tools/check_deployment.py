#!/usr/bin/env python3
"""Static deployment package validation; safe to run without Hailo hardware."""
from pathlib import Path
import hashlib
import json
import py_compile
import sys

ROOT = Path(__file__).resolve().parent.parent
HEF = ROOT / "models/hef/yolov11m_abandoned_hailo8.hef"
EXPECTED_HEF_SHA256 = "7d9268b3deb29c701ca7b3d1fbf51a80f662e7a8a7779d188178b42decce2ca3"
required = ["configs/runtime.json", "runtime/hailo_detector.py",
            "runtime/yolo_postprocess.py", "abandoned_monitor/processor.py",
            "abandoned_monitor/tracker.py", "app/infer_video_hailo.py",
            "web_detection.py", "input/demo.mp4", "Dockerfile", "README.md"]
errors = [f"missing {item}" for item in required if not (ROOT / item).exists()]
cfg = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8"))
for source in ROOT.rglob("*.py"):
    try:
        py_compile.compile(str(source), doraise=True)
    except Exception as exc:
        errors.append(f"compile failed {source.relative_to(ROOT)}: {exc}")
if int(cfg["model"]["imgsz"]) != 640:
    errors.append("HEF and runtime input size must be 640")
if cfg["model"]["path"] != "models/hef/yolov11m_abandoned_hailo8.hef":
    errors.append("runtime config does not select the yolov11m HEF")
if HEF.is_file():
    digest = hashlib.sha256(HEF.read_bytes()).hexdigest()
    if EXPECTED_HEF_SHA256 and digest != EXPECTED_HEF_SHA256:
        errors.append(f"unexpected HEF SHA256: {digest}")
    print(f"HEF SHA256={digest}")
else:
    print("warning: HEF not compiled yet (expected after conversion)" if not HEF.exists()
          else f"HEF SHA256={digest}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("deployment package is structurally valid")