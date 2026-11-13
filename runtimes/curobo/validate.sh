#!/usr/bin/env bash
# Start an isolated real acceptance server and always stop our own process.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${CUROBO_DATA_DIR:-${ROBOT_TOOLS_MODELS_ROOT:-/data/${USER}/robot-tools-models}/curobo}"
export CUROBO_DATA_DIR="$DATA_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PORT="${CUROBO_TEST_PORT:-15562}"
mkdir -p "$DATA_DIR/logs"
LOG="$DATA_DIR/logs/acceptance-server.log"
"$HERE/.venv/bin/python" "$HERE/server.py" --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
"$HERE/.venv/bin/python" - "$PORT" "$SERVER_PID" <<'PY'
import os
import sys
import time
import urllib.request

port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError("cuRobo did not become ready within 180 seconds; see acceptance-server.log")
PY
"$HERE/.venv/bin/python" "$HERE/smoke.py" --port "$PORT"
