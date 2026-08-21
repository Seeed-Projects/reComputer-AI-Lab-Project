"""Reference-compatible abandoned-luggage frame processing.

The RK3576 program uses Ultralytics ``model.track`` for detection and
ByteTrack IDs. On the Raspberry Pi, HailoDetector replaces only the model
inference part; this module keeps the same COCO classes, ROI presentation,
owner association, distance/time alarm logic, and box colors.

Hailo NMS returns detections but no track IDs, so IoUTracker supplies the
short-lived IDs. Missed tracks are retained internally for re-identification
but are never returned for drawing. A missed track is not a detection and must
not create a second stale box in the output.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2

from runtime.hailo_detector import HailoDetector
from runtime.yolo_postprocess import BAG_CLASSES as DEFAULT_BAG_CLASSES, PERSON_CLASS
from .tracker import IoUTracker, box_iou

COLOR_ROI = (0, 255, 255)
COLOR_NORMAL = (0, 255, 0)
COLOR_ALARM = (0, 0, 255)
COLOR_PERSON = (255, 200, 0)
COLOR_PERSIST = (0, 255, 255)
COLOR_OUTSIDE = (128, 128, 128)

BAG_NAMES = {24: "Backpack", 26: "Handbag", 28: "Suitcase"}


def point_in_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center_in_rect(xyxy, rect):
    x1, y1, x2, y2 = xyxy
    return point_in_rect(((x1 + x2) / 2, (y1 + y2) / 2), rect)


def draw_roi(frame, rect, color):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    for pt in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(frame, pt, 8, color, -1)
        cv2.circle(frame, pt, 10, (255, 255, 255), 2)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    label = "ROI"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(frame, (center[0] - tw // 2 - 10, center[1] - th // 2 - 8),
                  (center[0] + tw // 2 + 10, center[1] + th // 2 + 8), (0, 0, 0), -1)
    cv2.putText(frame, label, (center[0] - tw // 2, center[1] + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def draw_labeled_box(frame, xyxy, color, label=""):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, th + 6)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class AbandonedProcessor:
    """Process one frame using the RK reference state machine."""

    def __init__(self, cfg: dict, root: Path, detector=None):
        model_cfg = cfg["model"]
        if detector is None:
            model_path = Path(model_cfg["path"])
            if not model_path.is_absolute():
                model_path = root / model_path
            detector = HailoDetector(
                model_path,
                model_cfg["imgsz"],
                model_cfg["confidence"],
                model_cfg["iou"],
                model_cfg.get("input_mode", "uint8"),
            )
        self.detector = detector

        roi_cfg = cfg.get("roi", {})
        self.roi_enabled = bool(roi_cfg.get("enabled", True))
        self.roi_rect = [int(v) for v in roi_cfg.get("rect", [200, 200, 1100, 800])]
        # The reference draws the ROI but does not filter detections by it.
        self.roi_filter_detections = bool(roi_cfg.get("filter_detections", False))
        self.gray_outside_roi = bool(roi_cfg.get("gray_outside", True))

        abandonment = cfg.get("abandonment", {})
        self.dist_threshold = float(abandonment.get("dist_threshold", 200))
        self.time_threshold = float(abandonment.get("time_threshold", 5))
        self.alarm_hold = float(abandonment.get("alarm_hold", 0))
        self.bag_persistence_frames = max(
            0, int(abandonment.get("bag_persistence_frames", 0))
        )
        self.bag_classes = {
            int(value) for value in abandonment.get("bag_classes", sorted(DEFAULT_BAG_CLASSES))
        }
        self.person_class = int(abandonment.get("person_class", PERSON_CLASS))

        tracking = cfg.get("tracking", {})
        self.tracker = IoUTracker(
            match_iou=float(tracking.get("match_iou", 0.25)),
            min_hits=1,
            max_missed=max(0, int(tracking.get("max_missed", 30))),
        )

        # Same state as abandoned_detection.py: bag track ID -> owner ID,
        # and the time at which the owner first moved away/disappeared.
        self.bag_owner: dict[int, int] = {}
        self.away_start: dict[int, float] = {}
        self.abandoned_flags: dict[int, bool] = {}
        self.alarm_hold_until: dict[int, float] = {}
        self.static_bags: dict[int, dict] = {}
        self.frame_id = 0

    def _is_person(self, class_id: int) -> bool:
        return class_id == self.person_class

    def _is_bag(self, class_id: int) -> bool:
        return class_id in self.bag_classes

    def _select_detections(self, boxes, confs, class_ids, width, height):
        """Keep only person/bag detections, matching the RK loop's split."""
        roi = self.roi_rect if self.roi_enabled else [0, 0, width, height]
        selected = []
        for box, score, class_id in zip(boxes, confs, class_ids):
            class_id = int(class_id)
            if not (self._is_person(class_id) or self._is_bag(class_id)):
                continue
            if self.roi_filter_detections and not box_center_in_rect(box, roi):
                continue
            selected.append((box, score, class_id))
        return selected

    def _transfer_bag_state(self, source_id: int, target_id: int) -> None:
        """Keep alarm state when a new detector track replaces an old one."""
        if source_id == target_id:
            return
        for state in (
            self.bag_owner,
            self.away_start,
            self.abandoned_flags,
            self.alarm_hold_until,
        ):
            value = state.pop(source_id, None)
            if value is not None and target_id not in state:
                state[target_id] = value

    def _match_static_bag(self, track, used_static_ids: set[int]) -> int:
        """Reuse a persisted bag ID when Hailo briefly changes its track ID."""
        if not self.static_bags:
            return track.id
        tx1, ty1, tx2, ty2 = track.box
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        candidates = []
        for bag_id, info in self.static_bags.items():
            if bag_id in used_static_ids or info["cid"] != track.cls:
                continue
            old_box = info["xyxy"]
            overlap = box_iou(old_box, track.box)
            ox1, oy1, ox2, oy2 = old_box
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            old_diag = math.hypot(ox2 - ox1, oy2 - oy1)
            new_diag = math.hypot(tx2 - tx1, ty2 - ty1)
            distance = math.hypot(tcx - ocx, tcy - ocy)
            close = distance <= max(80.0, 0.5 * max(old_diag, new_diag))
            if overlap >= 0.05 or close:
                candidates.append((overlap, -distance, bag_id))
        if not candidates:
            return track.id
        _, _, matched_id = max(candidates)
        if matched_id != track.id:
            self._transfer_bag_state(track.id, matched_id)
            self.static_bags.pop(track.id, None)
        return matched_id

    def process(self, frame):
        height, width = frame.shape[:2]
        boxes, confs, class_ids = self.detector.predict(frame)
        self.frame_id += 1
        now = time.time()

        selected = self._select_detections(boxes, confs, class_ids, width, height)
        selected_boxes = [item[0] for item in selected]
        selected_confs = [item[1] for item in selected]
        selected_classes = [item[2] for item in selected]

        # Keep old tracks only for ID recovery. Only detections from this
        # frame are returned, so stale person/bag rectangles are never drawn.
        tracks = self.tracker.update(
            selected_boxes, selected_confs, selected_classes, include_missed=False
        )

        if self.bag_persistence_frames <= 0:
            self.static_bags.clear()
        else:
            for bag_id, info in list(self.static_bags.items()):
                if self.frame_id - info["last_seen"] > self.bag_persistence_frames:
                    self.static_bags.pop(bag_id, None)
                    self.bag_owner.pop(bag_id, None)
                    self.away_start.pop(bag_id, None)
                    self.abandoned_flags.pop(bag_id, None)
                    self.alarm_hold_until.pop(bag_id, None)

        persons = []  # (track_id, center_x, center_y, box)
        bags = []    # (track_id, center_x, center_y, box, class_id, persist)
        active_bag_ids = set()
        used_static_ids = set()
        for track in tracks:
            x1, y1, x2, y2 = track.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if self._is_person(track.cls):
                persons.append((track.id, center[0], center[1], track.box))
            elif self._is_bag(track.cls):
                bag_id = self._match_static_bag(track, used_static_ids)
                used_static_ids.add(bag_id)
                active_bag_ids.add(bag_id)
                if self.bag_persistence_frames > 0:
                    self.static_bags[bag_id] = {
                        "xyxy": list(track.box),
                        "cid": track.cls,
                        "last_seen": self.frame_id,
                    }
                bags.append((bag_id, center[0], center[1], track.box, track.cls, False))

        # Persist luggage only. A missed person is never synthesized.
        if self.bag_persistence_frames > 0:
            for bag_id, info in self.static_bags.items():
                if bag_id in active_bag_ids:
                    continue
                xyxy = info["xyxy"]
                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                bags.append((bag_id, cx, cy, xyxy, info["cid"], True))

        # Reference abandonment logic: first associate a bag with the nearest
        # person, then measure the owner's distance or disappearance duration.
        for bag_id, bx, by, _, bag_class, _ in bags:
            nearest_person = None
            nearest_distance = float("inf")
            for person_id, px, py, _ in persons:
                distance = math.hypot(px - bx, py - by)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_person = person_id

            if bag_id not in self.bag_owner and nearest_person is not None:
                self.bag_owner[bag_id] = nearest_person

            in_hold = now < self.alarm_hold_until.get(bag_id, 0)
            if in_hold:
                self.abandoned_flags[bag_id] = True
                continue

            abandoned = False
            owner_id = self.bag_owner.get(bag_id)
            if owner_id is not None:
                owner = next((item for item in persons if item[0] == owner_id), None)
                if owner is None:
                    self.away_start.setdefault(bag_id, now)
                else:
                    distance = math.hypot(owner[1] - bx, owner[2] - by)
                    if distance > self.dist_threshold:
                        self.away_start.setdefault(bag_id, now)
                    else:
                        self.away_start.pop(bag_id, None)

                started = self.away_start.get(bag_id)
                abandoned = started is not None and now - started > self.time_threshold

            if abandoned and not self.abandoned_flags.get(bag_id, False):
                if self.alarm_hold > 0:
                    self.alarm_hold_until[bag_id] = now + self.alarm_hold
            self.abandoned_flags[bag_id] = abandoned

        output = frame.copy()
        if self.roi_enabled:
            draw_roi(output, self.roi_rect, COLOR_ROI)

        for person_id, _, _, box in persons:
            draw_labeled_box(output, box, COLOR_PERSON, f"Person {person_id}")

        for bag_id, _, _, box, bag_class, persist in bags:
            name = BAG_NAMES.get(bag_class, f"Bag-{bag_id}")
            if self.abandoned_flags.get(bag_id, False) or now < self.alarm_hold_until.get(bag_id, 0):
                draw_labeled_box(output, box, COLOR_ALARM, "ABANDONED!")
            elif persist:
                draw_labeled_box(output, box, COLOR_PERSIST, f"{name} {bag_id} (persist)")
            elif self.gray_outside_roi and self.roi_enabled and not box_center_in_rect(box, self.roi_rect):
                draw_labeled_box(output, box, COLOR_OUTSIDE, f"{name} {bag_id}")
            else:
                draw_labeled_box(output, box, COLOR_NORMAL, f"{name} {bag_id}")

        return output, {
            "raw_detections": len(boxes),
            "persons": len(persons),
            "bags": len(bags),
            "static_bags": sum(1 for item in bags if item[5]),
            "abandoned": sum(1 for value in self.abandoned_flags.values() if value),
        }

    def release(self):
        self.detector.release()
