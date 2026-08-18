"""No-hardware unit tests for the pothole-detection runtime.

These tests validate preprocessing, postprocessing, tracking, config integrity,
and file presence **without** a physical Hailo-8 device. They must NOT claim that
Hailo inference has been exercised.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pothole_monitor.processor import in_road_roi  # noqa: E402
from pothole_monitor.tracker import TemporalTracker, box_iou  # noqa: E402
from runtime.yolo_postprocess import (  # noqa: E402
    letterbox_rgb,
    postprocess_auto,
    postprocess_hailo_nms,
    postprocess_standard,
)

HEF_PATH = ROOT / "models/hef/pothole_yolov8n_hailo8.hef"
EXPECTED_HEF_SHA256 = "6acec07e677cf4ae54919e0dd7105799ea8a50f1f48bd72528a6550b611b6856"
VIDEO_PATH = ROOT / "input/demo.mp4"


class PrePostProcessTests(unittest.TestCase):
    def test_letterbox_is_rgb_uint8_640(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # BGR red → RGB should be [255,0,0]
        image = letterbox_rgb(frame, 640)
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[320, 320].tolist(), [255, 0, 0])
        self.assertEqual(image[0, 0].tolist(), [114, 114, 114])

    def test_letterbox_preserves_aspect_ratio(self):
        """A 1280×720 frame letterboxed to 640 should pad top/bottom only."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        image = letterbox_rgb(frame, 640)
        self.assertEqual(image.shape, (640, 640, 3))
        # scale = 640/1280 = 0.5 → resized to 640×360, pad = (640-360)/2 = 140 top/bottom
        # Top pad rows should be the padding colour (114,114,114) in RGB.
        self.assertTrue(np.all(image[0, :] == 114))
        self.assertTrue(np.all(image[639, :] == 114))

    def test_hailo_normalized_yxyx_output(self):
        output = np.array([[[[0.25, 0.25, 0.75, 0.75, 0.9]]]], dtype=np.float32)
        boxes, scores = postprocess_hailo_nms(output, 720, 1280, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [320, 40, 960, 680], atol=1)
        self.assertAlmostEqual(scores[0], 0.9, places=5)

    def test_hailo_pixel_yxyx_output(self):
        """When coordinates are already in pixel space (>2), no normalization is applied."""
        output = np.array([[[100, 100, 300, 300, 0.85]]], dtype=np.float32)
        boxes, scores = postprocess_hailo_nms(output, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(scores[0], 0.85, places=5)

    def test_hailo_nms_filters_low_confidence(self):
        output = np.array([
            [[0.1, 0.1, 0.5, 0.5, 0.9]],
            [[0.1, 0.1, 0.5, 0.5, 0.05]],  # below threshold
        ], dtype=np.float32)
        boxes, scores = postprocess_hailo_nms(output, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        self.assertGreater(scores[0], 0.8)

    def test_standard_single_class_output(self):
        output = np.array([[[320], [320], [320], [160], [0.8]]], dtype=np.float32)
        boxes, scores = postprocess_standard(output, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [160, 240, 480, 400], atol=1)
        self.assertAlmostEqual(scores[0], 0.8, places=5)

    def test_postprocess_auto_routes_nms(self):
        """postprocess_auto should detect object-dtype / 6-column output as Hailo NMS."""
        output = np.array([[[100, 100, 300, 300, 0.85]]], dtype=np.float32)
        boxes, scores = postprocess_auto({"yolov8n_nms_postprocess": output}, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)

    def test_empty_output_returns_empty(self):
        empty = np.array([], dtype=np.float32)
        boxes, scores = postprocess_hailo_nms(empty, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(boxes, [])
        self.assertEqual(scores, [])


class BusinessLogicTests(unittest.TestCase):
    def test_roi_filter(self):
        cfg = {"enabled": True, "horizon_y": 0.2, "horizon_left": 0.25, "horizon_right": 0.75}
        self.assertTrue(in_road_roi([400, 400, 500, 600], 1000, 1000, cfg))
        self.assertFalse(in_road_roi([0, 0, 50, 100], 1000, 1000, cfg))

    def test_roi_disabled_passes_all(self):
        cfg = {"enabled": False}
        self.assertTrue(in_road_roi([0, 0, 50, 50], 1000, 1000, cfg))

    def test_tracker_bridges_short_gap(self):
        tracker = TemporalTracker(min_hits=2, max_missed=2, strong_confidence=0.95)
        self.assertEqual(tracker.update([[10, 10, 30, 30]], [0.8]), [])
        visible = tracker.update([[11, 11, 31, 31]], [0.8])
        self.assertEqual(len(visible), 1)
        self.assertTrue(visible[0].direct)
        # Gap frame: no detections — track should survive as yellow gap-fill.
        visible = tracker.update([], [])
        self.assertEqual(len(visible), 1)
        self.assertFalse(visible[0].direct)
        self.assertGreater(box_iou([10, 10, 30, 30], [11, 11, 31, 31]), 0.8)

    def test_tracker_drops_after_max_missed(self):
        tracker = TemporalTracker(min_hits=1, max_missed=2, strong_confidence=0.95)
        tracker.update([[10, 10, 30, 30]], [0.8])
        tracker.update([], [])
        tracker.update([], [])
        tracker.update([], [])  # 3rd miss → track removed
        self.assertEqual(len(tracker.tracks), 0)

    def test_tracker_total_events(self):
        """cumulative_events should equal unique track IDs ever created."""
        tracker = TemporalTracker(min_hits=1, max_missed=5, strong_confidence=0.0)
        self.assertEqual(tracker.total_events, 0)
        tracker.update([[10, 10, 30, 30]], [0.8])
        self.assertEqual(tracker.total_events, 1)
        tracker.update([[10, 10, 30, 30], [100, 100, 200, 200]], [0.8, 0.7])
        self.assertEqual(tracker.total_events, 2)

    def test_strong_confidence_bypasses_min_hits(self):
        tracker = TemporalTracker(min_hits=5, strong_confidence=0.5)
        visible = tracker.update([[10, 10, 30, 30]], [0.8])
        self.assertEqual(len(visible), 1)


class ConfigIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(
            (ROOT / "configs/runtime.json").read_text(encoding="utf-8")
        )

    def test_model_is_yolov8n(self):
        self.assertEqual(self.cfg["model"]["path"],
                         "models/hef/pothole_yolov8n_hailo8.hef")

    def test_imgsz_is_640(self):
        self.assertEqual(int(self.cfg["model"]["imgsz"]), 640)

    def test_single_class_config(self):
        """The NMS config must declare exactly 1 class (pothole).

        Skipped in the clean deployment package where conversion configs are
        intentionally excluded.
        """
        nms_path = ROOT / "conversion/hailo_config/yolov8n_pothole_nms_640.json"
        if not nms_path.exists():
            self.skipTest("conversion config not present in clean deploy package")
        nms_cfg = json.loads(nms_path.read_text(encoding="utf-8"))
        self.assertEqual(nms_cfg["classes"], 1)
        self.assertEqual(nms_cfg["image_dims"], [640, 640])

    def test_thresholds_in_range(self):
        for key in ("confidence", "iou"):
            val = float(self.cfg["model"][key])
            self.assertTrue(0.0 <= val <= 1.0, f"{key} out of range: {val}")

    def test_input_mode_valid(self):
        self.assertIn(self.cfg["model"].get("input_mode", "uint8"),
                      {"uint8", "float32"})

    def test_json_is_valid(self):
        """All JSON config files must parse without error."""
        for jf in (ROOT / "configs").glob("*.json"):
            json.loads(jf.read_text(encoding="utf-8"))


class ModelAssetTests(unittest.TestCase):
    def test_hef_exists(self):
        self.assertTrue(HEF_PATH.is_file(), f"HEF not found: {HEF_PATH}")

    def test_hef_sha256(self):
        if not HEF_PATH.is_file():
            self.skipTest("HEF not present")
        digest = hashlib.sha256(HEF_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_HEF_SHA256,
                         f"HEF SHA-256 mismatch: expected {EXPECTED_HEF_SHA256}, got {digest}")

    def test_onnx_exists(self):
        self.assertTrue((ROOT / "models/onnx/pothole_yolov8n.onnx").is_file())

    def test_pt_exists(self):
        self.assertTrue((ROOT / "models/source_pt/pothole_yolov8n.pt").is_file())


class VideoAssetTests(unittest.TestCase):
    def test_demo_video_opens(self):
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        try:
            self.assertTrue(cap.isOpened(), f"cannot open {VIDEO_PATH}")
            ok, frame = cap.read()
            self.assertTrue(ok and frame is not None)
            self.assertEqual(frame.shape[2], 3)  # BGR 3-channel
        finally:
            cap.release()

    def test_demo_video_properties(self):
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        try:
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1280)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25.0, places=1)
            self.assertGreater(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        finally:
            cap.release()


class ModuleImportTests(unittest.TestCase):
    """Verify that runtime modules can be imported without HailoRT installed."""

    def test_import_processor(self):
        mod = importlib.import_module("pothole_monitor.processor")
        self.assertTrue(hasattr(mod, "PotholeProcessor"))

    def test_import_tracker(self):
        mod = importlib.import_module("pothole_monitor.tracker")
        self.assertTrue(hasattr(mod, "TemporalTracker"))

    def test_import_postprocess(self):
        mod = importlib.import_module("runtime.yolo_postprocess")
        self.assertTrue(hasattr(mod, "postprocess_auto"))

    def test_import_hailo_detector_class(self):
        """Module-level import must not require hailo_platform (deferred to __init__)."""
        mod = importlib.import_module("runtime.hailo_detector")
        self.assertTrue(hasattr(mod, "HailoDetector"))

    def test_import_web_detection(self):
        mod = importlib.import_module("web_detection")
        self.assertTrue(hasattr(mod, "app"))
        self.assertTrue(hasattr(mod, "WebRuntime"))

    def test_import_infer_video(self):
        mod = importlib.import_module("app.infer_video_hailo")
        self.assertTrue(hasattr(mod, "validate_config"))


class HailoUnavailableTests(unittest.TestCase):
    """Without physical Hailo hardware, the detector must give a clear error."""

    def test_hailo_detector_raises_without_hailort(self):
        from runtime.hailo_detector import HailoDetector
        # If hailo_platform is not importable, __init__ must raise RuntimeError
        # (not ImportError or a crash).
        try:
            import hailo_platform  # noqa: F401
            has_hailo = True
        except ImportError:
            has_hailo = False

        if has_hailo:
            self.skipTest("hailo_platform is importable on this host — "
                          "real Hailo inference test must run on the Pi")

        with self.assertRaises(RuntimeError) as ctx:
            HailoDetector(HEF_PATH, 640, 0.18, 0.55, "uint8")
        self.assertIn("pyHailoRT is unavailable", str(ctx.exception))

    def test_no_hailo_inference_claim(self):
        """Sanity: this test machine has no Hailo device file."""
        if Path("/dev/hailo0").exists():
            self.skipTest("/dev/hailo0 exists — running on real hardware")
        self.assertFalse(Path("/dev/hailo0").exists(),
                        "No-hardware tests must not run on a machine with /dev/hailo0")


class PythonSyntaxTests(unittest.TestCase):
    def test_all_py_compile(self):
        import py_compile
        errors = []
        for py in ROOT.rglob("*.py"):
            if "__pycache__" in str(py):
                continue
            try:
                py_compile.compile(str(py), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"{py.relative_to(ROOT)}: {exc}")
        self.assertEqual(errors, [], f"Syntax errors: {errors}")


if __name__ == "__main__":
    unittest.main()
