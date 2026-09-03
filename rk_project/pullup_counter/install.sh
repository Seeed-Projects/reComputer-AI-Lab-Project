#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
LITE_WHEEL="${PROJECT_DIR}/rknn-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

case "$(uname -m)" in
    aarch64|arm64) ;;
    *)
        echo "ERROR: this project targets RK3576 AArch64 Linux; current architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} was not found. Install Python 3.11 first." >&2
    exit 1
fi

PYTHON_TAG="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_TAG}" != "3.11" ]]; then
    echo "ERROR: the bundled RKNNLite2 wheel requires Python 3.11; found ${PYTHON_TAG}." >&2
    exit 1
fi

if [[ ! -f "${LITE_WHEEL}" ]]; then
    echo "ERROR: RKNNLite2 wheel is missing: ${LITE_WHEEL}" >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements.txt"
"${VENV_PYTHON}" -m pip install --no-index --no-deps "${LITE_WHEEL}"

echo
echo "Installation complete. Running the RK3576 environment check..."
"${VENV_PYTHON}" "${PROJECT_DIR}/check_environment.py"
