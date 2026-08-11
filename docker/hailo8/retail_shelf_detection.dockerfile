# Raspberry Pi 5 arm64 image for retail shelf detection with Hailo-8.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Supply one HailoRT 4.23.x cp311 aarch64 wheel in hailort-packages/.
COPY hailort-packages/*.whl /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev \
    && pip install --no-cache-dir /tmp/hailort-*.whl \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/*.whl

COPY . .

EXPOSE 8000

CMD ["python", "web_detection.py", "--config", "configs/runtime.json", "--host", "0.0.0.0", "--port", "8000"]
