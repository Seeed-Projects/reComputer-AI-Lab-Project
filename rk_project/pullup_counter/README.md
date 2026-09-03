# RK3576 YOLO11 Pose Pull-up Counter

Real-time pull-up counting on the RK3576 NPU with YOLO11 Pose, RKNNLite2,
OpenCV, and a Flask/MJPEG browser dashboard.

The application detects the highest-confidence person, draws a bounding box and
17 COCO pose keypoints, estimates the pull-up bar from the wrists, and counts
complete repetitions with a temporal state machine. The pretrained YOLO11n Pose
model was converted to FP16 RKNN; no additional model training was performed.

## Compatibility

| Component | Version |
| --- | --- |
| Target | Rockchip RK3576, AArch64 Linux |
| Python | 3.11 |
| RKNN-Toolkit-Lite2 | 2.3.2 |
| Tested RKNN Runtime | 2.3.0 |
| Tested RKNPU driver | 0.9.8 |
| Model input | RGB uint8, NHWC, 1 x 640 x 640 x 3 |
| Model output | 1 x 56 x 8400 |

## Project layout

```text
pullup_counter/
|-- pullup_counter.py
|-- web_detection.py
|-- check_environment.py
|-- install.sh
|-- run_demo.sh
|-- run_web.sh
|-- requirements.txt
|-- model/
|   |-- yolo11n-pose.rknn
|   `-- MODEL_INFO.md
|-- rknn-packages/
|   `-- rknn_toolkit_lite2-2.3.2-...-aarch64.whl
`-- video/
    |-- input/test.mp4
    `-- output/
```

Conversion intermediates, cached files, and generated output videos are
intentionally excluded. The final RKNN model, matching RKNNLite2 wheel, and a
short demonstration video are included because they make the project directly
reproducible.

## Install on RK3576

The board must provide an RKNN-compatible RKNPU driver and `librknnrt.so`.
The installer does not replace the runtime library supplied by the firmware.

```bash
cd rk_project/pullup_counter
bash install.sh
```

The installer creates `.venv`, installs the Python dependencies, installs the
bundled RKNNLite2 wheel, and runs one synthetic-frame NPU inference check.

## Web inference

The default source is the included demonstration video:

```bash
bash run_web.sh
```

Open `http://<RK3576_IP>:8000` on another device in the same network. The page
shows the annotated stream, pull-up count, counter state, FPS, NPU latency, and
person confidence, with pause and reset controls.

Use a camera, another video, or a network stream by overriding `--source`:

```bash
bash run_web.sh --source 0
bash run_web.sh --source /path/to/video.mp4
bash run_web.sh --source 'rtsp://camera-address/stream'
```

Only one Web inference process should own the NPU and port 8000. If the port is
already in use, open the existing service or select another port:

```bash
bash run_web.sh --port 8080
```

## Offline video inference

Run the included demonstration video:

```bash
bash run_demo.sh
```

Pass another input video as the first argument when needed:

```bash
bash run_demo.sh /path/to/input.mp4
```

The annotated MP4 and per-frame CSV are written to `video/output/`.

## Counting logic

- Wrist keypoints provide a slowly smoothed estimate of the pull-up bar.
- The nose is the primary top-position signal; shoulders are used when the face
  is occluded by the bar.
- Distances are normalized by the detected person's bounding-box height.
- Separate top and bottom thresholds provide hysteresis.
- Consecutive-frame validation prevents duplicate counts from keypoint jitter.
- The state machine uses `WAITING`, `HANG`, `PULLING`, `TOP`, and `NO PERSON`.

## Measured performance

On the tested RK3576 system, FP16 NPU inference was typically 70-80 ms per
frame. The complete browser pipeline, including decode, preprocessing,
inference, drawing, JPEG encoding, and streaming, ran at approximately 7 FPS.

## Notes

- The model is static shape. RKNN's dynamic-range warning can be ignored for
  this model.
- The counter is tuned for a mostly fixed, front-facing, single-athlete view.
- For INT8 conversion, use a representative exercise calibration set and
  compare pose and counting accuracy against this FP16 build.
