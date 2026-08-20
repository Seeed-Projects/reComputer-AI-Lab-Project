#!/usr/bin/env bash
# Run the pothole-detection Docker container on Raspberry Pi 5 + Hailo-8.
# Usage:
#   ./scripts/docker_run.sh web     # web preview (default)
#   ./scripts/docker_run.sh video   # batch video inference → outputs/
#   ./scripts/docker_run.sh shell   # interactive shell
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-pothole-detection:latest}"
MODE="${1:-web}"

# Shared device + library mounts (host HailoRT driver must match 4.23.x)
DEVICE_OPTS=(
  --privileged
  --device /dev/hailo0:/dev/hailo0
)
if [[ -f /usr/lib/libhailort.so.4.23.0 ]]; then
  DEVICE_OPTS+=(
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro
  )
fi

# Mount input and output so custom videos and results persist on the host.
VOLUME_OPTS=(
  -v "$ROOT/input:/app/input:ro"
  -v "$ROOT/output:/app/output"
  -v "$ROOT/configs:/app/configs:ro"
)

RUN_BASE=(
  --rm
  --name pothole-detection
  --net=host
  -e PYTHONUNBUFFERED=1
  "${DEVICE_OPTS[@]}"
  "${VOLUME_OPTS[@]}"
)

case "$MODE" in
  web)
    exec sudo docker run "${RUN_BASE[@]}" "$IMAGE" \
      python web_detection.py \
        --config configs/runtime.json \
        --source input/demo.mp4 \
        --host 0.0.0.0 --port 8000
    ;;
  video)
    exec sudo docker run "${RUN_BASE[@]}" "$IMAGE" \
      python app/infer_video_hailo.py \
        --config configs/runtime.json \
        --source input/demo.mp4 \
        --output output/pothole_demo_hailo8.mp4
    ;;
  shell)
    exec sudo docker run -it "${RUN_BASE[@]}" "$IMAGE" bash
    ;;
  *)
    echo "Usage: $0 {web|video|shell}" >&2
    exit 1
    ;;
esac
