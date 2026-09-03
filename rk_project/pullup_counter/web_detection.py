#!/usr/bin/env python3
"""Flask/MJPEG web inference for the RK3576 pull-up counter."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Iterator, Optional, Union

import cv2
from flask import Flask, Response, jsonify, request

from pullup_counter import (
    DEFAULT_MODEL,
    PROJECT_DIR,
    PullUpCounter,
    RKNNPoseModel,
    draw_hud,
    draw_pose,
)


LOGGER = logging.getLogger("pullup_web")
app = Flask(__name__)
runtime: Optional["WebRuntime"] = None


def resolve_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def parse_source(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    if "://" in value:
        return value
    return str(resolve_path(value))


class WebRuntime:
    """Own the video source, one RKNN inference thread, and MJPEG frames."""

    def __init__(
        self,
        source: str,
        model_path: Path,
        conf: float,
        iou: float,
        keypoint_conf: float,
        npu_core: str,
        loop_video: bool,
        show_debug: bool,
        jpeg_quality: int,
        stream_width: int,
    ) -> None:
        self.source_label = source
        self.source = parse_source(source)
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.keypoint_conf = keypoint_conf
        self.npu_core = npu_core
        self.loop_video = loop_video and not isinstance(self.source, int)
        self.show_debug = show_debug
        self.jpeg_quality = jpeg_quality
        self.stream_width = stream_width

        self.stop_event = Event()
        self.reset_event = Event()
        self.condition = Condition()
        self.status_lock = Lock()
        self.worker: Optional[Thread] = None
        self.jpeg: Optional[bytes] = None
        self.frame_version = 0
        self._started = False
        self.status_data: dict[str, object] = {
            "running": False,
            "paused": False,
            "source": source,
            "frame": 0,
            "total_frames": 0,
            "loop_count": 0,
            "count": 0,
            "state": "STARTING",
            "fps": 0.0,
            "inference_ms": 0.0,
            "person_conf": None,
            "error": None,
        }

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.worker = Thread(target=self._loop, name="rknn-web-inference", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=8)

    def status(self) -> dict[str, object]:
        with self.status_lock:
            return dict(self.status_data)

    def update_status(self, **values: object) -> None:
        with self.status_lock:
            self.status_data.update(values)

    def toggle_pause(self) -> bool:
        with self.status_lock:
            paused = not bool(self.status_data["paused"])
            self.status_data["paused"] = paused
        return paused

    def reset_count(self) -> None:
        self.reset_event.set()
        self.update_status(count=0, state="RESETTING")

    def stream(self) -> Iterator[bytes]:
        seen = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.frame_version != seen or self.stop_event.is_set(),
                    timeout=1.0,
                )
                frame = self.jpeg
                seen = self.frame_version
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

    def _publish_frame(self, frame) -> None:
        if self.stream_width > 0 and frame.shape[1] > self.stream_width:
            scale = self.stream_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (self.stream_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.frame_version += 1
            self.condition.notify_all()

    @staticmethod
    def _draw_performance(frame, fps: float, inference_ms: float) -> None:
        height, width = frame.shape[:2]
        text = f"RK3576 NPU  {fps:.1f} FPS  {inference_ms:.1f} ms"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        x1 = 20
        y2 = height - 20
        y1 = y2 - text_height - 22
        x2 = min(width - 20, x1 + text_width + 24)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 12, 22), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.putText(
            frame,
            text,
            (x1 + 12, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (90, 235, 150),
            2,
            cv2.LINE_AA,
        )

    def _loop(self) -> None:
        capture = None
        model = None
        try:
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open source: {self.source_label}")

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            model = RKNNPoseModel(
                model_path=self.model_path,
                imgsz=640,
                conf_threshold=self.conf,
                iou_threshold=self.iou,
                keypoint_conf_threshold=self.keypoint_conf,
                npu_core=self.npu_core,
            )
            counter = PullUpCounter(self.keypoint_conf)
            frame_index = 0
            loop_count = 0
            lost_frames = 0
            fps_ema = 0.0
            previous = time.perf_counter()
            self.update_status(running=True, total_frames=total_frames, state=counter.state)

            while not self.stop_event.is_set():
                if bool(self.status().get("paused")):
                    time.sleep(0.05)
                    previous = time.perf_counter()
                    continue

                if self.reset_event.is_set():
                    self.reset_event.clear()
                    counter = PullUpCounter(self.keypoint_conf)
                    lost_frames = 0

                ok, frame = capture.read()
                if not ok:
                    if self.loop_video:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        counter = PullUpCounter(self.keypoint_conf)
                        frame_index = 0
                        loop_count += 1
                        lost_frames = 0
                        previous = time.perf_counter()
                        continue
                    break

                pose, inference_ms = model.predict(frame)
                if pose is not None:
                    lost_frames = 0
                    counter.update(pose)
                    draw_pose(frame, pose, self.keypoint_conf)
                else:
                    lost_frames += 1
                    if lost_frames > 15:
                        counter.state = "NO PERSON"
                        counter.stable_frames = 0

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                fps_ema = instant_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * instant_fps

                draw_hud(frame, counter, pose, self.show_debug)
                self._draw_performance(frame, fps_ema, inference_ms)
                self._publish_frame(frame)

                frame_index += 1
                self.update_status(
                    frame=frame_index,
                    total_frames=total_frames,
                    loop_count=loop_count,
                    count=counter.count,
                    state=counter.state,
                    fps=round(fps_ema, 2),
                    inference_ms=round(inference_ms, 2),
                    person_conf=round(pose.box_conf, 4) if pose else None,
                    error=None,
                )
        except Exception as exc:
            LOGGER.exception("Web inference failed")
            self.update_status(error=str(exc), state="ERROR")
        finally:
            if capture is not None:
                capture.release()
            if model is not None:
                model.release()
            self.update_status(running=False)
            with self.condition:
                self.condition.notify_all()


@app.get("/healthz")
def healthz():
    if runtime is None:
        return jsonify({"ok": False, "running": False}), 503
    status = runtime.status()
    return jsonify({"ok": status["error"] is None, **status})


@app.get("/api/status")
def api_status():
    return jsonify(runtime.status() if runtime else {"running": False})


@app.get("/api/video_feed")
def video_feed():
    if runtime is None:
        return Response(status=503)
    return Response(
        runtime.stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/api/control/toggle-pause")
def toggle_pause():
    if runtime is None:
        return jsonify({"error": "runtime unavailable"}), 503
    return jsonify({"paused": runtime.toggle_pause()})


@app.post("/api/control/reset")
def reset_count():
    if runtime is None:
        return jsonify({"error": "runtime unavailable"}), 503
    runtime.reset_count()
    return jsonify({"ok": True})


@app.get("/")
def index():
    return Response(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RK3576 Pull-up Counter</title>
<style>
:root{color-scheme:dark;--bg:#090d14;--card:#111824;--line:#243044;--muted:#8fa0b8;--green:#63e6a2;--orange:#ff8a4c;--blue:#74b9ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#142033 0,#090d14 48%);color:#eef5ff;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1380px;margin:auto;padding:26px}.header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:18px}
h1{margin:0;font-size:clamp(24px,3vw,38px);letter-spacing:-.03em}.subtitle{color:var(--muted);margin-top:5px}.live{display:flex;align-items:center;gap:8px;color:var(--green);font-weight:700}.dot{width:10px;height:10px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green)}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:18px}.card{background:rgba(17,24,36,.92);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 55px rgba(0,0,0,.28)}
.video{padding:12px;min-height:320px;display:flex;align-items:center;justify-content:center}.video img{width:100%;max-height:78vh;object-fit:contain;border-radius:12px;background:#05070a}
.side{padding:20px}.count-label,.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em}.count{font-size:clamp(72px,9vw,126px);line-height:1;color:var(--orange);font-weight:850;margin:8px 0 18px}
.state{display:inline-flex;padding:7px 12px;border:1px solid #35506c;border-radius:999px;color:var(--blue);font-weight:750;margin-bottom:22px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.metric{padding:13px;background:#0c121c;border:1px solid #1d2a3c;border-radius:12px}.value{font-size:22px;font-weight:760;margin-top:4px}.progress{height:8px;background:#0a1018;border-radius:9px;overflow:hidden;margin:16px 0}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .3s}
.buttons{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}button{border:1px solid #33445d;background:#182337;color:#f1f6ff;border-radius:11px;padding:11px;font-weight:700;cursor:pointer}button:hover{border-color:#5f7da6}.error{color:#ff7b72;white-space:pre-wrap;margin-top:14px;font-size:13px}
@media(max-width:850px){.grid{grid-template-columns:1fr}.wrap{padding:14px}.header{align-items:flex-start;flex-direction:column}.video{padding:7px}}
</style></head><body><main class="wrap">
<header class="header"><div><h1>Pull-up Counter</h1><div class="subtitle">YOLO11 Pose · RK3576 NPU</div></div><div class="live"><span class="dot"></span><span id="connection">CONNECTING</span></div></header>
<section class="grid"><div class="card video"><img src="/api/video_feed" alt="Annotated inference stream"></div>
<aside class="card side"><div class="count-label">Pull-ups</div><div class="count" id="count">0</div><div class="state" id="state">STARTING</div>
<div class="metrics"><div class="metric"><div class="label">Stream FPS</div><div class="value" id="fps">--</div></div><div class="metric"><div class="label">NPU latency</div><div class="value" id="latency">--</div></div>
<div class="metric"><div class="label">Frame</div><div class="value" id="frame">--</div></div><div class="metric"><div class="label">Person confidence</div><div class="value" id="confidence">--</div></div></div>
<div class="progress"><div class="bar" id="bar"></div></div><div class="label" id="source">Source: --</div>
<div class="buttons"><button id="pause" onclick="togglePause()">Pause</button><button onclick="resetCount()">Reset count</button></div><div class="error" id="error"></div></aside></section></main>
<script>
const el=id=>document.getElementById(id);
async function post(url){return (await fetch(url,{method:'POST'})).json()}
async function togglePause(){const r=await post('/api/control/toggle-pause');el('pause').textContent=r.paused?'Resume':'Pause'}
async function resetCount(){await post('/api/control/reset')}
async function refresh(){try{const s=await (await fetch('/api/status',{cache:'no-store'})).json();
el('count').textContent=s.count??0;el('state').textContent=s.state??'--';el('fps').textContent=s.fps!=null?s.fps.toFixed(1):'--';
el('latency').textContent=s.inference_ms!=null?s.inference_ms.toFixed(1)+' ms':'--';el('frame').textContent=s.total_frames?`${s.frame}/${s.total_frames}`:s.frame??'--';
el('confidence').textContent=s.person_conf!=null?(s.person_conf*100).toFixed(1)+'%':'--';el('source').textContent='Source: '+(s.source??'--');
el('bar').style.width=s.total_frames?Math.min(100,100*s.frame/s.total_frames)+'%':'0%';el('pause').textContent=s.paused?'Resume':'Pause';
el('connection').textContent=s.error?'ERROR':s.running?(s.paused?'PAUSED':'LIVE'):'STOPPED';el('error').textContent=s.error??'';
}catch(e){el('connection').textContent='OFFLINE';el('error').textContent='Status unavailable'}}
setInterval(refresh,500);refresh();
</script></body></html>""",
        mimetype="text/html",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=str(PROJECT_DIR / "video" / "input" / "test.mp4"),
        help="Video path, RTSP/HTTP URL, or camera index",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--keypoint-conf", type=float, default=0.35)
    parser.add_argument("--npu-core", choices=("auto", "0", "1", "2"), default="auto")
    parser.add_argument("--no-loop", action="store_true", help="Stop at the end of a video")
    parser.add_argument("--debug", action="store_true", help="Show counter calibration values")
    parser.add_argument("--jpeg-quality", type=int, default=82, choices=range(50, 96))
    parser.add_argument("--stream-width", type=int, default=720)
    return parser


def main() -> int:
    global runtime
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    model_path = resolve_path(args.model)
    runtime = WebRuntime(
        source=args.source,
        model_path=model_path,
        conf=args.conf,
        iou=args.iou,
        keypoint_conf=args.keypoint_conf,
        npu_core=args.npu_core,
        loop_video=not args.no_loop,
        show_debug=args.debug,
        jpeg_quality=args.jpeg_quality,
        stream_width=args.stream_width,
    )
    runtime.start()
    LOGGER.info("Web inference: http://%s:%d", args.host, args.port)
    try:
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            use_reloader=False,
            debug=False,
        )
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
