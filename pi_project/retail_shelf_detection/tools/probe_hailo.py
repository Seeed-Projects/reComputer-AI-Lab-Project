#!/usr/bin/env python3
"""Report the board Python/Hailo binding and model readiness."""

from __future__ import annotations

import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

print(f"machine={platform.machine()} python={platform.python_version()}")
try:
    import cv2
    import numpy

    print(f"opencv={cv2.__version__} numpy={numpy.__version__}")
except ImportError as exc:
    print(f"ERROR: base Python dependency missing: {exc}")

try:
    import hailo_platform

    version = getattr(hailo_platform, "__version__", "system package")
    print(f"pyHailoRT={version}")
except ImportError as exc:
    print(f"ERROR: pyHailoRT import failed: {exc}")

for name in ("shelf_product_hailo8.hef", "held_product_v4_hailo8.hef"):
    path = ROOT / "models" / "hef" / name
    print(f"{'OK' if path.is_file() else 'MISSING'}: {path}")

