"""No-hardware unit tests for the abandoned-luggage runtime.

Validates preprocessing, NMS decoding, tracking, and the ported
abandonment logic without a physical Hailo-8 device.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.yolo_postprocess import (  # noqa: E402
    letterbox_rgb,
    postprocess_auto,
    postprocess_hailo_nms,
)
from abandoned_monitor.tracker import IoUTracker, box_iou  # noqa: E402
from abandoned_monitor.processor import (  # noqa: E402
    box_center_in_rect,
    point_in_rect,
)

HEF_PATH = ROOT / "models/hef/yolov11m_abandoned_hailo8.hef"
VIDEO_PATH = ROOT / "input/demo.mp4"


class PrePostProcessTests(unittest.TestCase):
    def test_letterbox_rgb_640(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        image = letterbox_rgb(frame, 640)
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[320, 320].tolist(), [255, 0, 0])  # BGR red -> RGB
        self.assertEqual(image[0, 0].tolist(), [114, 114, 114])

    def test_nms_6col_yxyx_with_class(self):
        """Hailo 6-column format: y0,x0,y1,x1,score,class_id"""
        output = np.array([[[0.25, 0.25, 0.75, 0.75, 0.9, 24.0]]], dtype=np.float32)
        boxes, scores, cids = postprocess_hailo_nms(output, 720, 1280, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], [320, 40, 960, 680], atol=1)
        self.assertAlmostEqual(scores[0], 0.9, places=5)
        self.assertEqual(cids, [24])

    def test_nms_class_grid_5col(self):
        """Per-class grid format: [C, M, 5] with normalized YXYX."""
        grid = np.zeros((2, 3, 5), dtype=np.float32)  # 2 classes, 3 proposals
        grid[0, 0] = [0.2, 0.2, 0.4, 0.4, 0.8]      # class 0
        grid[1, 1] = [0.5, 0.5, 0.7, 0.7, 0.7]      # class 1
        boxes, scores, cids = postprocess_hailo_nms(grid, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(sorted(cids), [0, 1])
        self.assertEqual(len(boxes), 2)

    def test_nms_filters_low_confidence(self):
        output = np.array([
            [0.2, 0.2, 0.4, 0.4, 0.9, 0.0],
            [0.2, 0.2, 0.4, 0.4, 0.05, 0.0],
        ], dtype=np.float32)
        boxes, scores, cids = postprocess_hailo_nms(output, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        self.assertGreater(scores[0], 0.8)
        self.assertEqual(cids, [0])

    def test_empty_output(self):
        boxes, scores, cids = postprocess_hailo_nms(np.array([]), 640, 640, 0.2, 0.5, 640)
        self.assertEqual((boxes, scores, cids), ([], [], []))

    def test_postprocess_auto_routes(self):
        output = {"yolov11m_nms_postprocess": np.array([[[0.2, 0.2, 0.4, 0.4, 0.8, 28.0]]])}
        boxes, scores, cids = postprocess_auto(output, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(cids, [28])


class GeometryTests(unittest.TestCase):
    def test_point_in_rect(self):
        self.assertTrue(point_in_rect((500, 500), [200, 200, 1100, 800]))
        self.assertFalse(point_in_rect((150, 500), [200, 200, 1100, 800]))

    def test_box_center_in_rect(self):
        self.assertTrue(box_center_in_rect([400, 400, 600, 600], [200, 200, 1100, 800]))
        self.assertFalse(box_center_in_rect([0, 0, 100, 100], [200, 200, 1100, 800]))


class TrackerTests(unittest.TestCase):
    def test_track_assigns_persistent_id(self):
        tracker = IoUTracker(min_hits=1, max_missed=30)
        tracks = tracker.update([[10, 10, 30, 30]], [0.8], [24])
        self.assertEqual(len(tracks), 1)
        first_id = tracks[0].id
        tracks = tracker.update([[11, 11, 31, 31]], [0.8], [24])
        self.assertEqual(tracks[0].id, first_id)
        self.assertEqual(tracks[0].cls, 24)

    def test_track_separates_classes(self):
        tracker = IoUTracker(min_hits=1, max_missed=30)
        tracks = tracker.update([[10, 10, 30, 30], [100, 100, 200, 200]], [0.8, 0.8], [0, 24])
        self.assertEqual(len(tracks), 2)
        self.assertEqual({t.cls for t in tracks}, {0, 24})

    def test_track_persists_through_gaps(self):
        tracker = IoUTracker(min_hits=1, max_missed=3)
        tracker.update([[10, 10, 30, 30]], [0.8], [28])
        tracks = tracker.update([], [], [])
        self.assertEqual(len(tracks), 1)  # kept despite no detection
        self.assertEqual(tracks[0].box, [10, 10, 30, 30])
        # max_missed=3 tolerates 3 lost frames; the 4th consecutive miss drops it
        for _ in range(2):   # +2 misses (total 3)
            tracks = tracker.update([], [], [])
        self.assertEqual(len(tracks), 1)
        tracks = tracker.update([], [], [])  # 4th miss
        self.assertEqual(len(tracks), 0)  # dropped after max_missed+1
        self.assertGreater(box_iou([10, 10, 30, 30], [11, 11, 31, 31]), 0.8)

    def test_tracker_reset(self):
        tracker = IoUTracker()
        tracker.update([[10, 10, 30, 30]], [0.8], [24])
        tracker.reset()
        self.assertEqual(tracker.tracks, [])


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8"))

    def test_model_config(self):
        self.assertEqual(self.cfg["model"]["path"],
                         "models/hef/yolov11m_abandoned_hailo8.hef")
        self.assertEqual(int(self.cfg["model"]["imgsz"]), 640)

    def test_abandonment_params(self):
        ab = self.cfg["abandonment"]
        self.assertEqual(ab["person_class"], 0)
        self.assertEqual(sorted(ab["bag_classes"]), [24, 26, 28])
        self.assertGreater(ab["dist_threshold"], 0)
        self.assertGreater(ab["time_threshold"], 0)

    def test_all_json_valid(self):
        for jf in (ROOT / "configs").glob("*.json"):
            json.loads(jf.read_text(encoding="utf-8"))


class HEFTests(unittest.TestCase):
    EXPECTED = "7d9268b3deb29c701ca7b3d1fbf51a80f662e7a8a7779d188178b42decce2ca3"

    def test_hef_exists(self):
        self.assertTrue(HEF_PATH.is_file(), f"HEF not found: {HEF_PATH}")

    def test_hef_sha256(self):
        if not HEF_PATH.is_file():
            self.skipTest("HEF not compiled yet")
        digest = hashlib.sha256(HEF_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, self.EXPECTED, f"SHA-256 mismatch: {digest}")


class VideoTests(unittest.TestCase):
    def test_demo_video_opens(self):
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        try:
            self.assertTrue(cap.isOpened())
            ok, frame = cap.read()
            self.assertTrue(ok and frame is not None)
            self.assertEqual(frame.shape[2], 3)
        finally:
            cap.release()


class ModuleImportTests(unittest.TestCase):
    def test_import_modules(self):
        for module in ("abandoned_monitor.processor", "abandoned_monitor.tracker",
                       "runtime.yolo_postprocess", "runtime.hailo_detector",
                       "app.infer_video_hailo", "web_detection"):
            with self.subTest(module=module):
                importlib.import_module(module)


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
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()