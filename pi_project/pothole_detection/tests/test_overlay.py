import cv2
import numpy as np
import pytest

from pothole_monitor.overlay import SurfaceOverlay
from pothole_monitor.tracker import Track
from pothole_monitor.processor import PotholeProcessor
from pathlib import Path


def test_estimation_selects_dark_surface_in_box():
    crop = np.full((128, 128, 3), 180, np.uint8)
    cv2.ellipse(crop, (64, 64), (40, 25), 0, 0, 360, (30, 30, 30), -1)
    mask, fallback = SurfaceOverlay()._estimate(crop)
    assert not fallback
    assert mask[64, 64] == 1
    assert mask[8, 8] == 0
    assert .1 < mask.mean() < .6


def test_overlaps_blend_once_and_input_is_unchanged():
    frame = np.full((150, 160, 3), 100, np.uint8)
    renderer = SurfaceOverlay({"opacity": .5})
    renderer._estimate = lambda crop: (np.ones((128, 128), np.float32), False)
    tracks = [Track(1, [20, 40, 130, 140], .9), Track(2, [20, 40, 130, 140], .9)]
    output, status = renderer.render(frame, tracks)
    np.testing.assert_array_equal(frame, np.full_like(frame, 100))
    np.testing.assert_array_equal(output[90, 80], [65, 105, 178])
    np.testing.assert_array_equal(output[149, 159], frame[149, 159])
    assert status["estimated_surfaces"] == 2


def test_tracked_only_is_not_filled_and_cache_is_pruned():
    frame = np.full((150, 160, 3), 100, np.uint8)
    renderer = SurfaceOverlay()
    renderer.previous[1] = np.ones((128, 128), np.float32)
    renderer.previous[2] = np.ones((128, 128), np.float32)
    output, status = renderer.render(frame, [Track(1, [20, 40, 130, 140], .9, direct=False)])
    np.testing.assert_array_equal(output[90, 80], frame[90, 80])
    assert status["estimated_surfaces"] == 0
    assert not renderer.previous


def test_clipped_and_degenerate_boxes_do_not_crash():
    frame = np.full((150, 160, 3), 100, np.uint8)
    tracks = [Track(1, [-20, 30, 50, 100], .8), Track(2, [50, 50, 50, 90], .8)]
    output, _ = SurfaceOverlay().render(frame, tracks)
    assert output.shape == frame.shape


@pytest.mark.parametrize("cfg", [{"opacity": 2}, {"mask_size": 4}, {"color_bgr": [0, 999, 0]}])
def test_invalid_config(cfg):
    with pytest.raises(ValueError):
        SurfaceOverlay(cfg)


@pytest.mark.parametrize("with_overlay_config", [True, False])
def test_shared_processor_enables_surface_for_new_and_legacy_configs(with_overlay_config):
    class Detector:
        def predict(self, frame):
            return [[20, 40, 130, 140]], [.9]

    cfg = {"model": {}}
    if with_overlay_config:
        cfg["overlay"] = {"enabled": True, "opacity": .34}
    processor = PotholeProcessor(cfg, Path(__file__).resolve().parent.parent, Detector())
    processor.overlay._estimate = lambda crop: (np.ones((128, 128), np.float32), False)
    frame = np.full((150, 160, 3), 100, np.uint8)
    output, status = processor.process(frame)
    assert status["estimated_surfaces"] == 1
    assert not np.array_equal(output[90, 80], frame[90, 80])
