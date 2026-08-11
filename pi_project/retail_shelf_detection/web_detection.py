#!/usr/bin/env python3
"""Web preview for retail shelf detection on Raspberry Pi 5 + Hailo-8."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.infer_video_hailo import load_json, resolve, validate_config
from app.tiled_inference import detect_tiled
from app.type_mapping import HeldTypeTracker, type_labels
from runtime.hailo_detector import HailoDetector
from shelf_monitor.drawing import (
    draw_boxes,
    draw_events,
    draw_inventory_panel,
    draw_low_stock,
    draw_regions,
)
from shelf_monitor.pipeline import ShelfPipeline
from shelf_monitor.regions import RegionConfig
from shelf_monitor.temporal_filter import EventKind, InventoryEvent

logger = logging.getLogger("web_detection")


class FrameBuffer:
    """Keep only the newest annotated frame and preview JPEG."""

    def __init__(self) -> None:
        self.annotated = None
        self.annotated_version = 0
        self.jpeg: bytes | None = None
        self.jpeg_version = 0
        self.condition = threading.Condition()

    def push_annotated(self, frame) -> None:
        with self.condition:
            self.annotated = frame
            self.annotated_version += 1
            self.condition.notify_all()

    def wait_annotated(self, version: int, timeout: float = 1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.annotated_version > version,
                timeout=timeout,
            )
            return self.annotated, self.annotated_version

    def push_jpeg(self, jpeg: bytes) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.jpeg_version += 1
            self.condition.notify_all()

    def wait_jpeg(self, version: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self.condition:
            self.condition.wait_for(lambda: self.jpeg_version > version, timeout=timeout)
            return self.jpeg, self.jpeg_version


class ShelfWebRuntime:
    """Own the two Hailo models, video loop, preview encoder, and status."""

    def __init__(
        self,
        config_path: Path,
        video_path: Path | None,
        preview_width: int,
        preview_height: int,
        jpeg_quality: int,
        target_fps: float,
        loop_video: bool,
    ) -> None:
        self.config_path = config_path
        self.video_path = video_path
        self.preview_width = max(0, int(preview_width))
        self.preview_height = max(0, int(preview_height))
        self.jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self.target_fps = max(0.0, float(target_fps))
        self.loop_video = bool(loop_video)
        self.frames = FrameBuffer()
        self.stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status: dict = {
            "state": "starting",
            "error": "",
            "frame": 0,
            "timestamp": 0.0,
            "fps": 0.0,
            "source": "",
            "loops": 0,
            "occluded": False,
            "inventory_total": 0,
            "regions": [],
            "events": [],
        }
        self._inference_thread: threading.Thread | None = None
        self._encode_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._inference_thread and self._inference_thread.is_alive():
            return
        self.stop_event.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="shelf-inference",
            daemon=True,
        )
        self._encode_thread = threading.Thread(
            target=self._encode_loop,
            name="shelf-preview-encoder",
            daemon=True,
        )
        self._inference_thread.start()
        self._encode_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.frames.condition:
            self.frames.condition.notify_all()
        for thread in (self._inference_thread, self._encode_thread):
            if thread is not None:
                thread.join(timeout=3.0)

    def status(self) -> dict:
        with self._status_lock:
            return json.loads(json.dumps(self._status))

    def update_status(self, **values) -> None:
        with self._status_lock:
            self._status.update(values)

    def mjpeg(self) -> Iterator[bytes]:
        version = -1
        while not self.stop_event.is_set():
            jpeg, version = self.frames.wait_jpeg(version, timeout=1.0)
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )

    def _encode_loop(self) -> None:
        version = -1
        while not self.stop_event.is_set():
            frame, version = self.frames.wait_annotated(version, timeout=1.0)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            if (
                self.preview_width > 0
                and self.preview_height > 0
                and (width, height) != (self.preview_width, self.preview_height)
            ):
                preview = cv2.resize(
                    frame,
                    (self.preview_width, self.preview_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview = frame
            ok, encoded = cv2.imencode(
                ".jpg",
                preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok:
                self.frames.push_jpeg(encoded.tobytes())

    @staticmethod
    def _new_session(cfg: dict, width: int, height: int) -> dict:
        region_config = RegionConfig.load(resolve(cfg["regions"]))
        type_config = RegionConfig.load(resolve(cfg["type_regions"]))
        pipeline = ShelfPipeline(
            region_config,
            event_duration=float(cfg.get("event_duration_seconds", 1.8)),
        )
        pipeline.resolve(width, height)
        type_config.resolve_for_frame(width, height)
        type_groups = load_json(cfg["type_groups"])
        type_names = load_json(cfg["type_names"])
        event_cfg = load_json(cfg["scripted_events"])
        return {
            "region_config": region_config,
            "type_config": type_config,
            "type_groups": type_groups,
            "type_names": type_names,
            "pipeline": pipeline,
            "held_tracker": HeldTypeTracker(type_config, type_groups, type_names),
            "timeline": [
                (float(item["time_seconds"]), int(item["removed"]))
                for item in event_cfg["events"]
            ],
            "active_windows": [
                (float(start), float(end))
                for start, end in cfg["held_model"].get("active_windows", [])
            ],
            "scripted_counts": {
                region.id: region.capacity for region in region_config.regions
            },
            "region_order": list(cfg.get("scripted_region_order", [])),
            "fired": set(),
            "recent_events": [],
        }

    def _inference_loop(self) -> None:
        cap = None
        shelf_detector = None
        held_detector = None
        try:
            cfg = load_json(self.config_path)
            errors = validate_config(cfg, require_models=True)
            if errors:
                raise RuntimeError("; ".join(errors))
            source = resolve(self.video_path or cfg["video"])
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video: {source}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            if width <= 0 or height <= 0:
                raise RuntimeError(f"invalid video dimensions: {width}x{height}")

            shelf_cfg = cfg["shelf_model"]
            held_cfg = cfg["held_model"]
            shelf_detector = HailoDetector(
                resolve(shelf_cfg["path"]),
                shelf_cfg["imgsz"],
                shelf_cfg["confidence"],
                shelf_cfg["iou"],
                shelf_cfg.get("input_mode", "float32"),
            )
            held_detector = HailoDetector(
                resolve(held_cfg["path"]),
                held_cfg["imgsz"],
                held_cfg["confidence"],
                held_cfg["iou"],
                held_cfg.get("input_mode", "float32"),
            )

            session = self._new_session(cfg, width, height)
            frame_index = 0
            loop_count = 0
            fps_ema = 0.0
            previous_wall = time.perf_counter()
            next_frame_at = previous_wall
            self.update_status(state="running", source=str(source), error="")

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not self.loop_video:
                        self.update_status(state="complete")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    session = self._new_session(cfg, width, height)
                    frame_index = 0
                    loop_count += 1
                    self.update_status(loops=loop_count)
                    continue

                timestamp = frame_index / source_fps
                frame_index += 1
                boxes, confidences = shelf_detector.predict(frame)
                result = session["pipeline"].process(boxes, confidences, timestamp)

                held_boxes: list[list[float]] = []
                held_confidences: list[float] = []
                active_windows = session["active_windows"]
                held_active = not active_windows or any(
                    start <= timestamp <= end for start, end in active_windows
                )
                if held_active:
                    held_boxes, held_confidences = detect_tiled(
                        held_detector,
                        frame,
                        tile_size=int(held_cfg.get("imgsz", 320)),
                        overlap=int(held_cfg.get("tile_overlap", 80)),
                        nms_iou=float(held_cfg.get("iou", 0.35)),
                    )

                shelf_labels = type_labels(
                    boxes,
                    session["type_config"],
                    session["type_groups"],
                    session["type_names"],
                )
                held_labels = session["held_tracker"].update(held_boxes, timestamp)
                for event_index, (event_time, amount) in enumerate(session["timeline"]):
                    if event_index in session["fired"] or timestamp < event_time:
                        continue
                    session["recent_events"].append(
                        InventoryEvent(
                            "__demo__",
                            "Pickup detected",
                            EventKind.REMOVED,
                            amount,
                            timestamp,
                        )
                    )
                    region_order = session["region_order"]
                    if region_order:
                        region_id = region_order[min(event_index, len(region_order) - 1)]
                        if region_id in session["scripted_counts"]:
                            session["scripted_counts"][region_id] = max(
                                0,
                                session["scripted_counts"][region_id] - amount,
                            )
                    session["fired"].add(event_index)

                event_duration = float(cfg.get("event_duration_seconds", 1.8))
                session["recent_events"] = [
                    event
                    for event in session["recent_events"]
                    if timestamp - event.timestamp <= event_duration
                ]
                inventory = session["pipeline"].inventory_state
                inventory.update(
                    session["scripted_counts"],
                    session["recent_events"],
                )

                now = time.perf_counter()
                instantaneous = 1.0 / max(1e-6, now - previous_wall)
                previous_wall = now
                fps_ema = 0.9 * fps_ema + 0.1 * instantaneous if fps_ema else instantaneous

                output = frame.copy()
                draw_regions(output, session["region_config"])
                draw_boxes(output, boxes, confidences, show_conf=False, labels=shelf_labels)
                for box, score, label in zip(held_boxes, held_confidences, held_labels):
                    x1, y1, x2, y2 = (int(round(value)) for value in box)
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    held_text = f"{label} HELD {score:.2f}" if label else f"HELD {score:.2f}"
                    cv2.putText(
                        output,
                        held_text,
                        (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                draw_inventory_panel(
                    output,
                    inventory,
                    fps_ema or source_fps,
                    timestamp,
                    result.is_occluded,
                )
                draw_events(output, session["recent_events"])
                draw_low_stock(output, inventory)
                self.frames.push_annotated(output)

                snapshots = inventory.snapshot()
                self.update_status(
                    state="running",
                    frame=frame_index,
                    timestamp=round(timestamp, 3),
                    fps=round(fps_ema, 2),
                    loops=loop_count,
                    occluded=bool(result.is_occluded),
                    inventory_total=inventory.total(),
                    regions=[
                        {
                            "id": item.id,
                            "name": item.name,
                            "count": item.count,
                            "capacity": item.capacity,
                            "status": item.status.value,
                            "last_delta": item.last_delta,
                        }
                        for item in snapshots
                    ],
                    events=[
                        {
                            "message": event.name,
                            "kind": event.kind.value,
                            "delta": event.delta,
                            "timestamp": round(event.timestamp, 2),
                        }
                        for event in session["recent_events"]
                    ],
                )

                if self.target_fps > 0:
                    next_frame_at += 1.0 / self.target_fps
                    sleep_for = next_frame_at - time.perf_counter()
                    if sleep_for > 0:
                        self.stop_event.wait(sleep_for)
                    elif sleep_for < -(1.0 / self.target_fps):
                        next_frame_at = time.perf_counter()
        except Exception as exc:
            logger.exception("web inference failed")
            self.update_status(state="error", error=str(exc))
        finally:
            self.stop_event.set()
            if cap is not None:
                cap.release()
            if shelf_detector is not None:
                shelf_detector.release()
            if held_detector is not None:
                held_detector.release()
            with self.frames.condition:
                self.frames.condition.notify_all()


app = FastAPI(title="Retail Shelf Detection · Raspberry Pi 5 + Hailo-8")
runtime: ShelfWebRuntime | None = None


@app.get("/healthz")
async def healthz() -> dict:
    status = runtime.status() if runtime else {"state": "not-started", "error": ""}
    return {"ok": status["state"] not in {"error", "not-started"}, **status}


@app.get("/api/status")
async def api_status() -> dict:
    return runtime.status() if runtime else {"state": "not-started", "error": ""}


@app.get("/api/video_feed")
async def video_feed() -> StreamingResponse:
    def empty_stream() -> Iterator[bytes]:
        if False:
            yield b""

    stream = runtime.mjpeg() if runtime else empty_stream()
    return StreamingResponse(
        stream,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Retail Shelf Detection · Hailo-8</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #101416; color: #eef3f1; }
    header { padding: 24px 32px; border-bottom: 1px solid #28302d; }
    h1 { margin: 0; color: #9de35a; font-size: clamp(22px, 4vw, 34px); }
    header p { margin: 8px 0 0; color: #aab5b0; }
    main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); gap: 20px; padding: 20px; }
    .card { background: #171d1a; border: 1px solid #2b3530; border-radius: 14px; overflow: hidden; }
    .stream { min-height: 320px; display: grid; place-items: center; background: #050706; }
    .stream img { width: 100%; height: auto; display: block; }
    .panel { padding: 18px; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .metric { background: #202824; border-radius: 9px; padding: 12px; }
    .metric strong { display: block; color: #9de35a; font-size: 22px; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { text-align: left; padding: 9px 6px; border-bottom: 1px solid #303a35; }
    th { color: #91a099; font-size: 12px; text-transform: uppercase; }
    .status { padding: 5px 9px; border-radius: 999px; background: #27322d; font-size: 12px; }
    .error { color: #ff8d8d; white-space: pre-wrap; }
    @media (max-width: 850px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Retail Shelf Detection</h1>
    <p>Raspberry Pi 5 · Hailo-8 · live MJPEG preview and inventory status</p>
  </header>
  <main>
    <section class="card stream"><img src="/api/video_feed" alt="Retail shelf detection stream"></section>
    <aside class="card panel">
      <div class="metrics">
        <div class="metric"><span>State</span><strong id="state">starting</strong></div>
        <div class="metric"><span>Inference FPS</span><strong id="fps">0.0</strong></div>
        <div class="metric"><span>Inventory</span><strong id="total">0</strong></div>
        <div class="metric"><span>Video time</span><strong id="time">0.0s</strong></div>
      </div>
      <p id="error" class="error"></p>
      <table>
        <thead><tr><th>Region</th><th>Count</th><th>Status</th></tr></thead>
        <tbody id="regions"></tbody>
      </table>
    </aside>
  </main>
  <script>
    async function refresh() {
      try {
        const s = await fetch('/api/status', {cache: 'no-store'}).then(r => r.json());
        document.getElementById('state').textContent = s.state || 'unknown';
        document.getElementById('fps').textContent = Number(s.fps || 0).toFixed(1);
        document.getElementById('total').textContent = s.inventory_total || 0;
        document.getElementById('time').textContent = Number(s.timestamp || 0).toFixed(1) + 's';
        document.getElementById('error').textContent = s.error || '';
        document.getElementById('regions').innerHTML = (s.regions || []).map(r =>
          `<tr><td>${r.name}</td><td>${r.count}/${r.capacity}</td><td><span class="status">${r.status}</span></td></tr>`
        ).join('');
      } catch (err) {
        document.getElementById('error').textContent = String(err);
      }
    }
    refresh(); setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--video_path", type=Path, help="override the video path from config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview_width", type=int, default=1280)
    parser.add_argument("--preview_height", type=int, default=720)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--target_fps", type=float, default=0.0, help="0 means uncapped")
    parser.add_argument("--no-loop", action="store_true", help="stop at the end of the video")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_json(args.config)
    errors = validate_config(cfg, require_models=not args.check_config)
    if args.video_path and not resolve(args.video_path).exists():
        errors.append(f"missing video: {resolve(args.video_path)}")
    if errors:
        for error in errors:
            logger.error(error)
        return 2
    if args.check_config:
        logger.info("web configuration is valid; HailoRT was not loaded")
        return 0

    global runtime
    runtime = ShelfWebRuntime(
        config_path=args.config,
        video_path=args.video_path,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        jpeg_quality=args.jpeg_quality,
        target_fps=args.target_fps,
        loop_video=not args.no_loop,
    )
    runtime.start()
    logger.info("web preview: http://%s:%d", args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
