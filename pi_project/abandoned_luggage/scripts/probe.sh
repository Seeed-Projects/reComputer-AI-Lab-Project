#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
hailortcli scan
hailortcli fw-control identify
hailortcli parse-hef models/hef/pothole_yolov8n_hailo8.hef
"${PYTHON:-.venv/bin/python}" tools/probe_hailo.py
