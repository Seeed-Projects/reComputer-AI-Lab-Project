# Retail Shelf Detection on Raspberry Pi 5 + Hailo-8

This project detects products on retail shelves, tracks inventory changes, and
renders pickup and low-stock events from video. It runs two custom HEF models on
a Hailo-8 accelerator attached to a Raspberry Pi 5 and provides a FastAPI/MJPEG
web preview on port 8000.

## Compatibility

| Component | Version |
| --- | --- |
| Board | Raspberry Pi 5, aarch64 |
| Accelerator | Hailo-8 M.2 Key M 2280 (`/dev/hailo0`) |
| HailoRT | 4.23.x |
| Container Python | 3.11 |
| Bare-metal test OS | Debian GNU/Linux 13 (trixie) |

The host driver, firmware, shared library, and Python wheel must use compatible
HailoRT versions. The included HEF files target Hailo-8, not Hailo-8L.

## Project layout

```text
app/                 Video inference entry point
configs/             Models, shelf regions, product names, and demo events
input/demo.mp4       Demo video
models/hef/          Shelf and held-product Hailo-8 models
runtime/             HailoRT integration and YOLO post-processing
shelf_monitor/       Inventory, event, region, and drawing logic
scripts/             Bare-metal installation, probing, and demo helpers
tools/               Deployment validation utilities
web_detection.py     Browser preview and inventory status API
```

## Run on Raspberry Pi OS

```bash
cd pi_project/retail_shelf_detection
chmod +x scripts/*.sh
./scripts/install_rpi5.sh
./scripts/probe.sh
python3 web_detection.py \
  --config configs/runtime.json \
  --video_path input/demo.mp4
```

Open `http://<PI_IP>:8000` from another device on the same network.

To generate a processed MP4 instead of starting the web preview:

```bash
./scripts/run_demo.sh
```

The processed video is written to `outputs/restock_demo_hailo8.mp4`.

To validate the configuration without loading HailoRT or the HEF files:

```bash
python3 app/infer_video_hailo.py --check-config
```

## Run with Docker

The build context includes the HailoRT 4.23.0 Python 3.11 aarch64 wheel used by
the reference Hailo-8 containers. The Raspberry Pi host driver and firmware must
remain compatible with HailoRT 4.23.x.

Pull the published image:

```bash
sudo docker pull \
  ghcr.io/seeed-projects/recomputer-ai-lab-project/retail_shelf_detection:latest
```

From the repository root, build the arm64 image:

```bash
sudo docker build \
  -f docker/hailo8/retail_shelf_detection.dockerfile \
  -t retail_shelf_detection:latest \
  pi_project/retail_shelf_detection
```

Run the bundled demo with the web interface:

```bash
sudo docker run --rm --privileged \
  --name rpi5-hailo8-retail-shelf-detection \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-ai-lab-project/retail_shelf_detection:latest \
  python web_detection.py \
    --config configs/runtime.json \
    --video_path input/demo.mp4 \
    --host 0.0.0.0 \
    --port 8000
```

Open `http://<PI_IP>:8000`. The page shows the latest annotated frame,
inference FPS, total inventory, and per-region stock state.

If the host installs the versioned HailoRT library at another path, adjust both
library mounts while keeping the host and container HailoRT versions compatible.

## Use another video

Mount the video into the container and supply a configuration whose `video` and
`output` paths point to mounted locations. When the camera position, resolution,
or shelf geometry changes, update the region files under `configs/` as well.
