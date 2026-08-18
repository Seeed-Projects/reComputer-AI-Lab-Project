#!/usr/bin/env bash
# Raspberry Pi 5 + Hailo-8 pothole-detection installer.
# Does NOT upgrade HailoRT/DFC or download unrelated packages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
WHEEL="$ROOT/hailort-packages/hailort-4.23.0-cp311-cp311-linux_aarch64.whl"
HEF="$ROOT/models/hef/pothole_yolov8n_hailo8.hef"

# ---- 1. architecture check ----
arch="$(uname -m)"
if [[ "$arch" != "aarch64" ]]; then
  echo "ERROR: this installer targets aarch64 (Raspberry Pi 5 64-bit). Got: $arch" >&2
  exit 1
fi

# ---- 2. Python version check (3.11 required by the bundled wheel) ----
py_ver="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [[ "$py_ver" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (bundled wheel is cp311). Got: $py_ver" >&2
  echo "Install Raspberry Pi OS 64-bit Bookworm (python3.11) or use pyenv." >&2
  exit 1
fi
echo "OK: arch=$arch  python=$py_ver"

# ---- 3. system packages (no hailo-all / no DFC) ----
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-dev ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  >/dev/null

# ---- 4. venv with system-site-packages (for OS OpenCV/NumPy if present) ----
echo "Creating virtual environment..."
"$PY" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip -q

# ---- 5. project Python dependencies ----
echo "Installing project dependencies..."
.venv/bin/python -m pip install -r requirements-rpi5.txt -q

# ---- 6. HailoRT wheel (install, do not upgrade) ----
if [[ ! -f "$WHEEL" ]]; then
  echo "ERROR: HailoRT wheel not found: $WHEEL" >&2
  exit 1
fi
echo "Installing HailoRT from local wheel..."
if ! .venv/bin/python -c "import hailo_platform" 2>/dev/null; then
  .venv/bin/python -m pip install "$WHEEL" -q
else
  echo "HailoRT already importable — skipped (no upgrade)."
fi

# ---- 7. /dev/hailo0 device check ----
chmod +x scripts/*.sh
if [[ -c /dev/hailo0 ]]; then
  echo "OK: /dev/hailo0 present"
else
  echo "WARNING: /dev/hailo0 not found." >&2
  echo "  Install the Hailo PCIe driver (hailo-all) and reboot, or check the M.2 slot." >&2
  echo "  You can still proceed — run scripts/probe.sh after fixing the driver." >&2
fi

# ---- 8. HEF presence ----
if [[ -f "$HEF" ]]; then
  echo "OK: HEF present: $HEF"
else
  echo "WARNING: HEF not found at $HEF — copy it before running." >&2
fi

cat <<EOF

============================================================
 Installation complete.
   venv:  $ROOT/.venv
   python: $ROOT/.venv/bin/python

 Next steps:
   1. If the Hailo driver was newly installed, reboot.
   2. $ROOT/scripts/probe.sh
   3. $ROOT/scripts/run_demo.sh         (video inference)
   4. .venv/bin/python web_detection.py  (web preview)
============================================================
EOF
