#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

exec "${PYTHON_BIN}" "${PROJECT_DIR}/web_detection.py" \
    --host 0.0.0.0 \
    --port 8000 \
    --source "${PROJECT_DIR}/video/input/test.mp4" \
    "$@"
