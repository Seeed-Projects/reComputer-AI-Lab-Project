"""Approximate surface overlays from detection boxes (not semantic masks)."""
from __future__ import annotations

import cv2
import numpy as np


class SurfaceOverlay:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.opacity = float(cfg.get("opacity", 0.34))
        self.color = tuple(int(v) for v in cfg.get("color_bgr", [30, 110, 255]))
        self.show_boxes = bool(cfg.get("show_boxes", True))
        self.size = int(cfg.get("mask_size", 128))
        self.smoothing = float(cfg.get("smoothing", 0.4))
        if not 0 <= self.opacity <= 1 or not 0 <= self.smoothing < 1:
            raise ValueError("overlay opacity/smoothing out of range")
        if len(self.color) != 3 or not all(0 <= v <= 255 for v in self.color):
            raise ValueError("overlay color_bgr must contain three values in [0,255]")
        if not 32 <= self.size <= 256:
            raise ValueError("overlay mask_size must be between 32 and 256")
        self.previous = {}

    def _estimate(self, crop):
        # Normalized masks permit smoothing while a tracked box moves/resizes.
        small = cv2.resize(crop, (self.size, self.size))
        n = self.size
        labels = np.full((n, n), cv2.GC_PR_BGD, dtype=np.uint8)
        labels[:3] = labels[-3:] = cv2.GC_BGD
        labels[:, :3] = labels[:, -3:] = cv2.GC_BGD
        interior = np.zeros((n, n), np.uint8)
        cv2.ellipse(interior, (n // 2, n // 2), (int(n * .46), int(n * .44)),
                    0, 0, 360, 1, -1)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring = np.concatenate((lab[:5].reshape(-1, 3), lab[-5:].reshape(-1, 3),
                               lab[:, :5].reshape(-1, 3), lab[:, -5:].reshape(-1, 3)))
        # Seed high-contrast water/dark depressions, rather than always seeding
        # the box centre (which can be intact road in a large imperfect box).
        distance = np.linalg.norm(lab-np.median(ring, axis=0), axis=2)
        candidates = distance[interior > 0]
        probable = (distance >= max(8, float(np.percentile(candidates, 55)))) & (interior > 0)
        certain = (distance >= max(14, float(np.percentile(candidates, 90)))) & (interior > 0)
        if not np.any(certain):
            return np.zeros((n, n), np.float32), True
        labels[probable] = cv2.GC_PR_FGD
        labels[certain] = cv2.GC_FGD
        fallback = False
        try:
            cv2.grabCut(small, labels, None, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 2, cv2.GC_INIT_WITH_MASK)
            mask = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest = max(contours, key=cv2.contourArea) if contours else None
            if largest is None or cv2.contourArea(largest) < n * n * .08:
                fallback = True
            else:
                mask[:] = 0
                cv2.drawContours(mask, [largest], -1, 1, -1)
        except cv2.error:
            fallback = True
        if fallback:
            # Do not colour a guessed ellipse when image evidence is insufficient.
            mask = np.zeros((n, n), np.uint8)
        return mask.astype(np.float32), fallback

    def render(self, frame, tracks):
        output = frame.copy()
        h, w = frame.shape[:2]
        union = np.zeros((h, w), np.float32)
        outlines = []
        alive = set()
        estimates = fallbacks = 0
        for track in tracks:
            alive.add(track.id)
            x1, y1, x2, y2 = (int(round(v)) for v in track.box)
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            if self.enabled and track.direct:
                mask, fallback = self._estimate(frame[y1:y2, x1:x2])
                if track.id in self.previous:
                    mask = self.smoothing * self.previous[track.id] + (1-self.smoothing) * mask
                self.previous[track.id] = mask
                mask = cv2.resize((mask >= .5).astype(np.uint8), (x2-x1, y2-y1),
                                  interpolation=cv2.INTER_NEAREST)
                union[y1:y2, x1:x2] = np.maximum(union[y1:y2, x1:x2], mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outlines.extend(c + np.array([[[x1, y1]]]) for c in contours)
                estimates += 1
                fallbacks += int(fallback)
            elif not track.direct:
                self.previous.pop(track.id, None)
        if self.enabled:
            # Blend overlapping detections once, preserving underlying road texture.
            alpha = union[..., None] * self.opacity
            output[:] = np.rint(frame * (1-alpha) + np.array(self.color) * alpha).astype(np.uint8)
            cv2.drawContours(output, outlines, -1, self.color, 2, cv2.LINE_AA)
        for track in tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in track.box)
            color = self.color if track.direct else (0, 200, 255)
            if self.show_boxes or not track.direct:
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            label = f"pothole {track.confidence:.2f}" if track.direct else "pothole tracked"
            cv2.putText(output, label, (max(0, x1), max(22, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        self.previous = {key: value for key, value in self.previous.items() if key in alive}
        return output, {"estimated_surfaces": estimates - fallbacks, "unresolved_surfaces": fallbacks}
