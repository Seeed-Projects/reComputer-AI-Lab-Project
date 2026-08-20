#!/usr/bin/env python3
"""FastAPI/MJPEG abandoned-luggage preview for Raspberry Pi 5 + Hailo-8.

Legacy-free port of the PC web demo: same abandonment semantics
(owner association, distance/time threshold, static persistence,
alarm hold) driven by Hailo-8 inference.
"""
from __future__ import annotations
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Iterator
import argparse
import json
import logging
import sys
import time
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from abandoned_monitor.processor import AbandonedProcessor  # noqa: E402

logger = logging.getLogger("abandoned_web")
app = FastAPI(title="Raspberry Pi 5 Hailo Abandoned-Luggage Detection")
runtime = None


def resolve(path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value if "://" in value else str(resolve(value))


class WebRuntime:
    def __init__(self, cfg: dict, source: str, device: str = "hailo"):
        self.cfg, self.source = cfg, parse_source(source)
        self.device = device
        self.stop_event = Event()
        self.condition = Condition()
        self.status_lock = Lock()
        self.jpeg = None
        self.version = 0
        self.worker = None
        self._started = False
        self.status_data = {
            "running": False, "source": str(source), "device": device, "fps": 0.0,
            "frame": 0, "persons": 0, "bags": 0, "static_bags": 0,
            "abandoned": 0, "alarm_frames": 0, "error": None,
        }

    def start(self):
        if self._started:
            return
        self._started = True
        self.worker = Thread(target=self._loop, name="hailo-inference", daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=5)

    def status(self):
        with self.status_lock:
            return dict(self.status_data)

    def update(self, **values):
        with self.status_lock:
            self.status_data.update(values)

    def stream(self) -> Iterator[bytes]:
        seen = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.version != seen or self.stop_event.is_set(),
                                        timeout=1)
                frame, seen = self.jpeg, self.version
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    def _make_processor(self):
        if self.device == "cpu":
            from runtime.ultralytics_detector import UltralyticsDetector
            mc = self.cfg["model"]
            detector = UltralyticsDetector(
                resolve(mc.get("pt_path", "models/source_pt/yolo11m.pt")),
                mc["imgsz"], mc["confidence"], mc["iou"], mc.get("input_mode", "uint8"))
            return AbandonedProcessor(self.cfg, ROOT, detector=detector)
        return AbandonedProcessor(self.cfg, ROOT)

    def _loop(self):
        cap = processor = None
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open source: {self.source}")
            processor = self._make_processor()
            self.update(running=True)
            frame_id, fps_ema, alarm_frames = 0, 0.0, 0
            previous = time.perf_counter()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.cfg.get("loop_video", False) and not isinstance(self.source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                annotated, values = processor.process(frame)
                now = time.perf_counter()
                instant = 1.0 / max(now - previous, 1e-6)
                previous = now
                fps_ema = instant if not fps_ema else 0.9 * fps_ema + 0.1 * instant
                frame_id += 1
                alarm_frames += int(values["abandoned"] > 0)
                self._overlay(annotated, fps_ema, values)
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with self.condition:
                        self.jpeg = encoded.tobytes()
                        self.version += 1
                        self.condition.notify_all()
                self.update(frame=frame_id, fps=round(fps_ema, 2), alarm_frames=alarm_frames,
                            **values)
        except Exception as exc:
            logger.exception("inference failed")
            self.update(error=str(exc))
        finally:
            if cap is not None:
                cap.release()
            if processor is not None:
                processor.release()
            self.update(running=False)

    @staticmethod
    def _overlay(frame, fps_ema, values):
        label = f"{'CPU YOLO11m' if values.get('device') == 'cpu' else 'Hailo-8'} FPS {fps_ema:.1f}"
        cv2.rectangle(frame, (10, 10), (520, 100), (20, 20, 20), -1)
        cv2.putText(frame, label, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    (90, 255, 130), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"persons={values['persons']}  bags={values['bags']}  static={values['static_bags']}",
                    (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 60), 2, cv2.LINE_AA)
        cv2.putText(frame, f"ABANDONED={values['abandoned']}", (22, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if values["abandoned"] else (200, 200, 200), 2, cv2.LINE_AA)


@app.get("/healthz")
async def healthz():
    return {"ok": runtime is not None and runtime.status()["running"]}


@app.get("/api/status")
async def api_status():
    return runtime.status() if runtime else {"running": False}


@app.get("/api/video_feed")
async def video_feed():
    if runtime is None:
        return StreamingResponse(iter(()), media_type="multipart/x-mixed-replace; boundary=frame")
    return StreamingResponse(runtime.stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Abandoned Luggage Detection</title><style>
body{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}.wrap{max-width:1280px;margin:auto;padding:24px}
h1{font-size:26px;margin:0 0 18px}.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(240px,1fr);gap:18px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}img{width:100%;border-radius:8px}
.metric{font-size:26px;color:#65e695;margin:4px 0 14px}.label{color:#8b949e;font-size:12px;text-transform:uppercase}
.state{padding:6px 10px;border-radius:8px;background:#21262d;font-size:13px;margin-top:8px}
.err{color:#ff7b72;white-space:pre-wrap}.alarm{color:#ff6b6b;font-size:20px;font-weight:bold}
.legend{font-size:11px;color:#8b949e;margin-top:10px;line-height:1.7}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body><div class='wrap'><h1>Raspberry Pi 5 · Hailo-8 Abandoned Luggage Detection</h1><div class='grid'>
<div class='card'><img src='/api/video_feed'></div><div class='card'>
<div class='label'>Inference FPS</div><div class='metric' id='fps'>--</div>
<div class='label'>Persons / Bags</div><div class='metric' id='counts'>-- / --</div>
<div class='label'>Static persisted</div><div class='metric' id='static'>--</div>
<div class='label'>ABANDONED now</div><div class='metric' id='abandoned'>--</div>
<div class='label'>Frame</div><div class='metric' id='frame'>--</div>
<div class='state' id='state'>Starting…</div>
<div class='legend'>
<b style='color:#ffc800'>■</b> Yellow = ROI region<br>
<b style='color:#00ff00'>■</b> Green = luggage (normal)<br>
<b style='color:#00c8ff'>■</b> Cyan = person<br>
<b style='color:#ffff00'>■</b> Yellow box = persisted (undetected, held position)<br>
<b style='color:#ff0000'>■</b> Red = ABANDONED (owner away &gt; threshold)
</div>
</div></div></div><script>
const el=id=>document.getElementById(id);
setInterval(async()=>{try{const s=await (await fetch('/api/status')).json();
el('fps').textContent=s.fps!=null?s.fps.toFixed(1):'--';
el('counts').textContent=(s.persons!=null?s.persons:'--')+' / '+(s.bags!=null?s.bags:'--');
el('static').textContent=s.static_bags!=null?s.static_bags:'--';
const ab=el('abandoned'); ab.textContent=s.abandoned!=null?s.abandoned:'--';
ab.className='metric'+(s.abandoned>0?' alarm':'');
el('frame').textContent=s.frame!=null?s.frame:'--';
const st=el('state');
if(s.error){st.textContent='Error: '+s.error;st.className='state err';}
else if(s.running){st.textContent='Running on '+(s.device==='cpu'?'CPU':'Hailo-8');st.className='state';}
else{st.textContent='Stopped';st.className='state';}
}catch(error){el('state').textContent='Status unavailable';}},700)</script></body></html>"""


def main() -> int:
    global runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.json")
    parser.add_argument("--source", help="video file, RTSP URL, or camera index")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="hailo", choices=["hailo", "cpu"],
                        help="hailo = Hailo-8 (Pi); cpu = Ultralytics demo on PC without Hailo")
    parser.add_argument("--no-loop", action="store_true", help="stop at end of video")
    parser.add_argument("--check-config", action="store_true",
                        help="validate config without loading HailoRT")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    if args.check_config:
        model = cfg.get("model", {})
        if not {"path", "imgsz", "confidence", "iou"}.issubset(model):
            logger.error("incomplete model config")
            return 2
        hef = resolve(model["path"])
        if not hef.exists():
            logger.error(f"missing HEF model: {hef}")
            return 2
        logger.info("web configuration is valid; HailoRT was not loaded")
        return 0
    cfg["loop_video"] = cfg.get("loop_video", True) and not args.no_loop
    runtime = WebRuntime(cfg, args.source or str(cfg["source"]), device=args.device)
    runtime.start()
    logger.info("web preview (device=%s): http://%s:%d", args.device, args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())