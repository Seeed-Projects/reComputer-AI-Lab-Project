"""Lightweight temporal stabilizer that bridges short Hailo detection gaps."""
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
    box: list[float]
    confidence: float
    hits: int = 1
    missed: int = 0
    direct: bool = True


class TemporalTracker:
    def __init__(self, match_iou=0.18, min_hits=2, max_missed=3, strong_confidence=0.55):
        self.match_iou = float(match_iou)
        self.min_hits = int(min_hits)
        self.max_missed = int(max_missed)
        self.strong_confidence = float(strong_confidence)
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, boxes: list[list[float]], confs: list[float]) -> list[Track]:
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, box in enumerate(boxes):
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
            track.direct = True
            used_tracks.add(ti)
            used_dets.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.missed += 1
                track.direct = False
        for di, (box, score) in enumerate(zip(boxes, confs)):
            if di not in used_dets:
                self.tracks.append(Track(self.next_id, [float(v) for v in box], float(score)))
                self.next_id += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        visible = [track for track in self.tracks
                   if track.hits >= self.min_hits or track.confidence >= self.strong_confidence]
        self.visible_ids = {track.id for track in visible}
        return visible

    @property
    def total_events(self) -> int:
        """Cumulative count of unique pothole tracks ever created."""
        return max(0, self.next_id - 1)
