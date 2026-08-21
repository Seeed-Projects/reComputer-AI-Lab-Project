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
    AbandonedProcessor,
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

    def test_nms_object_array_preserves_class(self):
        """Real HailoRT NMS layout: object array (1, C), element = (N_c, 5)."""
        obj = np.empty((1, 80), dtype=object)
        obj[0, 0] = np.array([[0.10, 0.10, 0.20, 0.20, 0.9]])    # person
        obj[0, 5] = np.array([[0.50, 0.50, 0.70, 0.70, 0.7]])    # bus
        obj[0, 24] = np.array([[0.30, 0.30, 0.45, 0.45, 0.8]])   # backpack
        obj[0, 28] = np.array([[0.60, 0.20, 0.75, 0.40, 0.85]])  # suitcase
        boxes, scores, cids = postprocess_hailo_nms(obj, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(sorted(cids), [0, 5, 24, 28])
        self.assertEqual(len(boxes), 4)

    def test_nms_object_array_with_empty_classes(self):
        obj = np.empty((1, 80), dtype=object)  # unassigned slots stay None
        obj[0, 24] = np.array([[0.2, 0.2, 0.4, 0.4, 0.9]])
        boxes, scores, cids = postprocess_hailo_nms(obj, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(cids, [24])

    def test_nms_ragged_list_like_hailort(self):
        """Real HailoRT vstream output: ragged Python list of C per-class arrays."""
        ragged = [None] * 80
        ragged[0] = np.array([[0.1, 0.1, 0.2, 0.2, 0.9]])
        ragged[24] = np.array([[0.3, 0.3, 0.45, 0.45, 0.8]])
        ragged[28] = [[0.6, 0.2, 0.75, 0.4, 0.85]]
        boxes, scores, cids = postprocess_auto(
            {"yolov11m/yolov8_nms_postprocess": ragged}, 640, 640, 0.2, 0.5, 640)
        self.assertEqual(sorted(cids), [0, 24, 28])

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


class ProcessorResetTests(unittest.TestCase):
    def test_reset_clears_state_between_looped_video_passes(self):
        cfg = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8"))

        class FakeDetector:
            def predict(self, frame):
                return [], [], []

            def release(self):
                pass

        processor = AbandonedProcessor(cfg, ROOT, detector=FakeDetector())
        processor.tracker.update([[10, 10, 30, 30]], [0.9], [24])
        processor.bag_owner[1] = 2
        processor.away_start[1] = 10.0
        processor.abandoned_flags[1] = True
        processor.alarm_hold_until[1] = 20.0
        processor.static_bags[1] = {"xyxy": [10, 10, 30, 30], "cid": 24, "last_seen": 9}
        processor.frame_id = 9

        processor.reset()

        self.assertEqual(processor.tracker.tracks, [])
        self.assertEqual(processor.bag_owner, {})
        self.assertEqual(processor.away_start, {})
        self.assertEqual(processor.abandoned_flags, {})
        self.assertEqual(processor.alarm_hold_until, {})
        self.assertEqual(processor.static_bags, {})
        self.assertEqual(processor.bag_confirmation_counts, {})
        self.assertEqual(processor.frame_id, 0)


class BagFilteringTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "configs/runtime.json").read_text(encoding="utf-8"))

        class FakeDetector:
            def predict(self, frame):
                return [], [], []

            def release(self):
                pass

        self.processor = AbandonedProcessor(self.cfg, ROOT, detector=FakeDetector())

    def test_bag_zone_excludes_aircraft_and_keeps_people(self):
        boxes = [
            [800, 220, 960, 580],  # aircraft underside: bag center above bag zone
            [740, 560, 1040, 820], # white suitcase: center inside bag zone
            [220, 220, 460, 700],  # person: valid in full person ROI
        ]
        selected = self.processor._select_detections(boxes, [0.9] * 3, [26, 28, 0], 1920, 1080)
        self.assertEqual([item[2] for item in selected], [28, 0])

    def test_bag_requires_consecutive_confirmations(self):
        class FakeDetector:
            def predict(self, frame):
                return [[740, 560, 1040, 820]], [0.9], [28]

            def release(self):
                pass

        processor = AbandonedProcessor(self.cfg, ROOT, detector=FakeDetector())
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertEqual(processor.process(frame)[1]["bags"], 0)
        self.assertEqual(processor.process(frame)[1]["bags"], 0)
        self.assertEqual(processor.process(frame)[1]["bags"], 1)


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
        self.assertEqual(ab["bag_confirm_frames"], 3)
        self.assertEqual(self.cfg["roi"]["bag_rect"], [200, 500, 1100, 800])

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
