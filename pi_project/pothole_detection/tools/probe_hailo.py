#!/usr/bin/env python3
"""Verify that pyHailoRT and /dev/hailo0 are available."""
from pathlib import Path
import sys

device = Path("/dev/hailo0")
print(f"device: {device} exists={device.exists()}")
try:
    from hailo_platform import Device
    identities = Device.scan()
    print(f"Hailo devices: {identities}")
except Exception as exc:
    print(f"Hailo probe failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0 if device.exists() else 2)
