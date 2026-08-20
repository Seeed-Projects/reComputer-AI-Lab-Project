"""Lightweight IoU tracker for persons and luggage (no ByteTrack on Hailo)."""
from __future__ import annotations
from dataclasses import dataclass


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


@dataclass
class Track:
    id: int
    cls: int
    box: list[float]
    confidence: float
    hits: int = 1
    missed: int = 0


class IoUTracker:
    """Greedy IoU tracker with separate ID spaces per class.

    Bag tracks additionally keep a stable position while the detector
    briefly loses them (static persistence, controlled by max_missed).
    """

    def __init__(self, match_iou: float = 0.25, min_hits: int = 1,
                 max_missed: int = 30) -> None:
        self.match_iou = float(match_iou)
        self.min_hits = int(min_hits)
        self.max_missed = int(max_missed)
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, boxes, confs, class_ids) -> list[Track]:
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, (box, cid) in enumerate(zip(boxes, class_ids)):
                if cid != track.cls:
                    continue
                overlap = box_iou(track.box, box)
                if overlap >= self.match_iou:
                    pairs.append((overlap, ti, di))
        used_tracks, used_dets = set(), set()
        for _, ti, di in sorted(pairs, reverse=True):
            if ti in used_tracks or di in used_dets:
                continue
            track = self.tracks[ti]
            track.box = [float(v) for v in boxes[di]]
            track.confidence = float(confs[di])
            track.hits += 1
            track.missed = 0
            used_tracks.add(ti)
            used_dets.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missed += 1
        for di, (box, score, cid) in enumerate(zip(boxes, confs, class_ids)):
            if di not in used_dets:
                self.tracks.append(Track(self.next_id, int(cid), [float(v) for v in box],
                                         float(score)))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1