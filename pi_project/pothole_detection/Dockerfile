# Raspberry Pi 5 arm64 runtime image for Hailo-8 pothole detection.
# Build ON the Pi (or any arm64 host):  docker build -t pothole-detection .
# This image does NOT include the x86_64 Dataflow Compiler; the HEF is pre-compiled.
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (opencv-headless + numpy are installed via pip inside the container)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HailoRT 4.23.0 Python 3.11 aarch64 wheel — matches the host driver/firmware.
COPY hailort-packages/hailort-4.23.0-cp311-cp311-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/hailort-4.23.0-cp311-cp311-linux_aarch64.whl \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/*.whl

# Copy runtime files only (.dockerignore strips conversion artefacts)
COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# Default: web preview. Override CMD for video inference.
CMD ["python", "web_detection.py", \
     "--config", "configs/runtime.json", \
     "--host", "0.0.0.0", "--port", "8000"]
