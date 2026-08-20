"""Shared Hailo abandoned-luggage frame processor.

Logic ported from the original PC project (abandoned_detection_v2.py):
  - ROI rectangle filtering
  - detection split: person vs luggage (backpack/handbag/suitcase)
  - owner association: each bag is owned by the nearest person
  - abandonment trigger: owner beyond DIST_THRESHOLD or gone for
    TIME_THRESHOLD seconds
  - static persistence: bags keep their last position while briefly
    undetected (BAG_PERSISTENCE_FRAMES)
  - alarm hold: alert stays for ALARM_HOLD seconds to avoid flicker

Detection backend is pluggable: HailoDetector (Pi) or UltralyticsDetector
(PC demo), both returning (boxes_xyxy, confidences, class_ids).
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import cv2
from runtime.hailo_detector import HailoDetector
from runtime.yolo_postprocess import BAG_CLASSES as DEFAULT_BAG_CLASSES, PERSON_CLASS
from .tracker import IoUTracker

COLOR_ROI = (0, 255, 255)
COLOR_NORMAL = (0, 255, 0)
COLOR_ALARM = (0, 0, 255)
COLOR_PERSON = (255, 200, 0)
COLOR_PERSIST = (0, 255, 255)

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
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)


class AbandonedProcessor:
    def __init__(self, cfg: dict, root: Path, detector=None):
        model_cfg = cfg["model"]
        if detector is not None:
            self.detector = detector
        else:
            model_path = Path(model_cfg["path"])
            if not model_path.is_absolute():
                model_path = root / model_path
            self.detector = HailoDetector(model_path, model_cfg["imgsz"],
                                          model_cfg["confidence"], model_cfg["iou"],
                                          model_cfg.get("input_mode", "uint8"))

        roi_cfg = cfg.get("roi", {})
        self.roi_enabled = bool(roi_cfg.get("enabled", True))
        self.roi_rect = [int(v) for v in roi_cfg.get("rect", [200, 200, 1100, 800])]

        ab_cfg = cfg.get("abandonment", {})
        self.dist_threshold = float(ab_cfg.get("dist_threshold", 200))
        self.time_threshold = float(ab_cfg.get("time_threshold", 5))
        self.alarm_hold = float(ab_cfg.get("alarm_hold", 5))
        self.bag_persistence_frames = int(ab_cfg.get("bag_persistence_frames", 300))
        self.bag_classes = set(int(v) for v in ab_cfg.get("bag_classes", sorted(DEFAULT_BAG_CLASSES)))
        self.person_class = int(ab_cfg.get("person_class", PERSON_CLASS))

        match_iou = float(cfg.get("tracking", {}).get("match_iou", 0.25))
        self.tracker = IoUTracker(match_iou=match_iou, min_hits=1,
                                  max_missed=self.bag_persistence_frames)

        # Ported state machines from abandoned_detection_v2.py
        self.bag_owner: dict[int, int] = {}          # bag track id -> person track id
        self.away_start: dict[int, float] = {}       # bag id -> wall time owner left
        self.abandoned_flags: dict[int, bool] = {}
        self.alarm_hold_until: dict[int, float] = {} # bag id -> wall time hold ends
        self.static_bags: dict[int, dict] = {}       # bag id -> {xyxy, cid, last_seen, owner}
        self.frame_id = 0

    # ------------------------------------------------------------------
    def _is_person(self, cid: int) -> bool:
        return cid == self.person_class

    def _is_bag(self, cid: int) -> bool:
        return cid in self.bag_classes

    # ------------------------------------------------------------------
    def process(self, frame):
        height, width = frame.shape[:2]
        boxes, confs, cids = self.detector.predict(frame)
        self.frame_id += 1
        now = time.time()

        roi = self.roi_rect if self.roi_enabled else [0, 0, width, height]

        # Current-frame detections inside the ROI
        in_roi = [i for i in range(len(boxes)) if box_center_in_rect(boxes[i], roi)]
        roi_boxes = [boxes[i] for i in in_roi]
        roi_confs = [confs[i] for i in in_roi]
        roi_cids = [cids[i] for i in in_roi]

        # Track (assigns persistent IDs per class)
        tracks = self.tracker.update(roi_boxes, roi_confs, roi_cids)
        track_by_id = {t.id: t for t in tracks}

        # ----- static persistence (ported) -----
        active_bag_ids = set()
        for t in tracks:
            if self._is_bag(t.cls):
                active_bag_ids.add(t.id)
                self.static_bags[t.id] = {
                    "xyxy": t.box, "cid": t.cls, "last_seen": self.frame_id,
                    "owner": self.bag_owner.get(t.id),
                }
        for bag_id in list(self.static_bags):
            if self.frame_id - self.static_bags[bag_id]["last_seen"] > self.bag_persistence_frames:
                self.static_bags.pop(bag_id, None)
                self.bag_owner.pop(bag_id, None)
                self.away_start.pop(bag_id, None)
                self.abandoned_flags.pop(bag_id, None)
                self.alarm_hold_until.pop(bag_id, None)

        # ----- merge current detections + persisted bags -----
        persons = []   # (track_id, cx, cy, xyxy)
        bags = []      # (track_id, cx, cy, xyxy, cid, persist)
        for t in tracks:
            cx = (t.box[0] + t.box[2]) / 2
            cy = (t.box[1] + t.box[3]) / 2
            if self._is_person(t.cls):
                persons.append((t.id, cx, cy, t.box))
            elif self._is_bag(t.cls):
                bags.append((t.id, cx, cy, t.box, t.cls, False))
        for bag_id, info in self.static_bags.items():
            if bag_id not in active_bag_ids:
                xyxy = info["xyxy"]
                bags.append((bag_id, (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2,
                             xyxy, info["cid"], True))
                if info.get("owner") is not None:
                    self.bag_owner[bag_id] = info["owner"]

        # ----- abandonment logic (ported) -----
        for bag in bags:
            bag_id, bx, by, _, bag_cls, _ = bag
            in_hold = bag_id in self.alarm_hold_until and now < self.alarm_hold_until[bag_id]
            if in_hold:
                self.abandoned_flags[bag_id] = True
                continue

            # associate owner: nearest person
            nearest_person, nearest_dist = None, 999999.0
            for pid, px, py, _ in persons:
                d = math.hypot(px - bx, py - by)
                if d < nearest_dist:
                    nearest_dist, nearest_person = d, pid
            if bag_id not in self.bag_owner and nearest_person is not None:
                self.bag_owner[bag_id] = nearest_person
                if bag_id in self.static_bags:
                    self.static_bags[bag_id]["owner"] = nearest_person

            is_abandoned = False
            if bag_id in self.bag_owner:
                owner_id = self.bag_owner[bag_id]
                owner_found = False
                for pid, px, py, _ in persons:
                    if pid == owner_id:
                        owner_found = True
                        d = math.hypot(px - bx, py - by)
                        if d > self.dist_threshold:
                            if bag_id not in self.away_start:
                                self.away_start[bag_id] = now
                            if now - self.away_start[bag_id] > self.time_threshold:
                                is_abandoned = True
                        else:
                            self.away_start.pop(bag_id, None)
                        break
                if not owner_found:
                    if bag_id not in self.away_start:
                        self.away_start[bag_id] = now
                    if now - self.away_start[bag_id] > self.time_threshold:
                        is_abandoned = True

            if is_abandoned and not self.abandoned_flags.get(bag_id, False):
                self.alarm_hold_until[bag_id] = now + self.alarm_hold
            self.abandoned_flags[bag_id] = is_abandoned

        # ----- drawing (ported colors) -----
        output = frame.copy()
        if self.roi_enabled:
            draw_roi(output, self.roi_rect, COLOR_ROI)

        for pid, _, _, xyxy in persons:
            draw_labeled_box(output, xyxy, COLOR_PERSON, f"Person {pid}")
        for bag in bags:
            bag_id, _, _, box, bag_cls, persist = bag
            name = BAG_NAMES.get(bag_cls, f"Bag-{bag_id}")
            is_abandoned = self.abandoned_flags.get(bag_id, False)
            in_hold = bag_id in self.alarm_hold_until and now < self.alarm_hold_until[bag_id]
            if is_abandoned or in_hold:
                draw_labeled_box(output, box, COLOR_ALARM, "ABANDONED!")
            elif persist:
                draw_labeled_box(output, box, COLOR_PERSIST, f"{name} {bag_id} (persist)")
            else:
                draw_labeled_box(output, box, COLOR_NORMAL, f"{name} {bag_id}")

        abandoned_count = sum(1 for v in self.abandoned_flags.values() if v)
        status = {
            "raw_detections": len(boxes), "persons": len(persons), "bags": len(bags),
            "static_bags": len(self.static_bags), "abandoned": abandoned_count,
        }
        return output, status

    def release(self):
        self.detector.release()