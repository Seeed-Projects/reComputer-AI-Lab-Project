#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "warning: this installer is intended for 64-bit Raspberry Pi OS (aarch64)" >&2
fi

sudo apt update
sudo apt install -y hailo-all python3-hailort python3-opencv python3-numpy python3-venv
python3 -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-rpi5.txt"

echo "Installation complete. Reboot if hailo-all or the PCIe driver was newly installed."
echo "Then run: $ROOT/scripts/probe.sh"

