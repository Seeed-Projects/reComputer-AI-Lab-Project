#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "OS: $(. /etc/os-release && echo "$PRETTY_NAME")"
uname -a
python3 --version
hailortcli --version
hailortcli scan
hailortcli fw-control identify
python3 "$ROOT/tools/probe_hailo.py"

