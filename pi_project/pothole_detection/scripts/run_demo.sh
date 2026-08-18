#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-.venv/bin/python}" app/infer_video_hailo.py --config configs/runtime.json "$@"
