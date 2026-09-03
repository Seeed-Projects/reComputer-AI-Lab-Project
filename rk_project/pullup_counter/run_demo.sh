#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ $# -ge 1 && "$1" != --* ]]; then
    SOURCE="$1"
    shift
else
    SOURCE="${PROJECT_DIR}/video/input/test.mp4"
fi
SOURCE_NAME="$(basename -- "${SOURCE}")"
SOURCE_STEM="${SOURCE_NAME%.*}"

exec "${PYTHON_BIN}" "${PROJECT_DIR}/pullup_counter.py" \
    --source "${SOURCE}" \
    --output "${PROJECT_DIR}/video/output/${SOURCE_STEM}_counted.mp4" \
    "$@"
