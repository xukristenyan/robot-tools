#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
REAL_PROJECT="$HERE/real"
if [[ ! -f "$HERE/upstream/depth_anything_v2/dpt.py" ]]; then
    echo "Initialize selected upstream first: robot-tools install <profile>" >&2
    exit 1
fi
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --project "$REAL_PROJECT" --locked
uv pip check --python "$VENV_DIR/bin/python"
"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"
