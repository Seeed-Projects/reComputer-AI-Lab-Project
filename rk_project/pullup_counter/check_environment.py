"""Validate the RK3576 environment and execute one synthetic NPU frame."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "yolo11n-pose.rknn"


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    machine = platform.machine().lower()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Architecture: {machine}")
    print(f"Platform: {platform.platform()}")

    if machine not in {"aarch64", "arm64"}:
        fail(f"RK3576 requires AArch64 Linux; detected {machine}")
    if sys.version_info[:2] != (3, 11):
        fail(
            "The bundled RKNNLite2 wheel is cp311 and requires Python 3.11; "
            f"detected {sys.version_info.major}.{sys.version_info.minor}"
        )
    if not MODEL_PATH.is_file():
        fail(f"Model is missing: {MODEL_PATH}")

    import cv2
    import numpy as np
    from rknnlite.api import RKNNLite

    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"RKNNLite2: {RKNNLite.__module__}")

    from pullup_counter import RKNNPoseModel

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detector = RKNNPoseModel(MODEL_PATH, npu_core="auto")
    try:
        _, inference_ms = detector.predict(frame)
    finally:
        detector.release()

    print(f"Synthetic-frame NPU inference: {inference_ms:.2f} ms")
    print("Environment check: PASS")


if __name__ == "__main__":
    main()
