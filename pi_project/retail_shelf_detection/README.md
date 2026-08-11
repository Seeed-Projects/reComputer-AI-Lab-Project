# Retail Shelf Detection on Raspberry Pi 5 + Hailo-8

This project detects products on retail shelves, tracks inventory changes, and
renders pickup and low-stock events from video. It runs two custom HEF models on
a Hailo-8 accelerator attached to a Raspberry Pi 5.

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
```

## Run on Raspberry Pi OS

```bash
cd pi_project/retail_shelf_detection
chmod +x scripts/*.sh
./scripts/install_rpi5.sh
./scripts/probe.sh
./scripts/run_demo.sh
```

The processed video is written to `outputs/restock_demo_hailo8.mp4`.

To validate the configuration without loading HailoRT or the HEF files:

```bash
python3 app/infer_video_hailo.py --check-config
```

## Run with Docker

Download a HailoRT 4.23.x Python 3.11 aarch64 wheel and place it in
`hailort-packages/`. See [hailort-packages/README.md](hailort-packages/README.md)
for compatibility details.

From the repository root, build the arm64 image:

```bash
sudo docker build \
  -f docker/hailo8/retail_shelf_detection.dockerfile \
  -t retail_shelf_detection:latest \
  pi_project/retail_shelf_detection
```

Run the bundled demo and save its result in the host `outputs/` directory:

```bash
sudo docker run --rm --privileged \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  -v "$(pwd)/pi_project/retail_shelf_detection/outputs:/app/outputs" \
  retail_shelf_detection:latest
```

If the host installs the versioned HailoRT library at another path, adjust both
library mounts while keeping the host and container HailoRT versions compatible.

## Use another video

Mount the video into the container and supply a configuration whose `video` and
`output` paths point to mounted locations. When the camera position, resolution,
or shelf geometry changes, update the region files under `configs/` as well.

