#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
REAL_PROJECT="$HERE/real"
DATA_DIR="${CUROBO_DATA_DIR:-${ROBOT_TOOLS_MODELS_ROOT:-/data/${USER}/robot-tools-models}/curobo}"
export CUROBO_DATA_DIR="$DATA_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DIR/cache/uv}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$DATA_DIR/cache/cuda}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$DATA_DIR/cache/warp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DIR/cache}"
if [[ ! -f "$HERE/upstream/curobo/motion_planner.py" ]]; then
    echo "Initialize the selected upstream: robot-tools install <profile>" >&2
    exit 1
fi
if [[ "$(git -C "$HERE/upstream" rev-parse HEAD)" != "42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83" ]]; then
    echo "cuRobo upstream does not match the tested pin; restore the selected submodule revision." >&2
    exit 1
fi
mkdir -p "$DATA_DIR" "$UV_CACHE_DIR"
# Preserve a light environment created by sync. Move it once to the data disk,
# keeping the conventional .venv path used by every launcher command.
if [[ -d "$VENV_DIR" && ! -L "$VENV_DIR" ]]; then
    if [[ -e "$DATA_DIR/venv" ]]; then
        echo "Both local .venv and data/venv exist; choose one before installing." >&2
        exit 1
    fi
    mv "$VENV_DIR" "$DATA_DIR/venv"
fi
if [[ ! -e "$VENV_DIR" && ! -L "$VENV_DIR" ]]; then
    ln -s "$DATA_DIR/venv" "$VENV_DIR"
fi
if [[ -L "$VENV_DIR" && ! -e "$VENV_DIR" ]]; then
    uv venv --python 3.12 "$DATA_DIR/venv"
fi
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --project "$REAL_PROJECT" --locked
uv pip check --python "$VENV_DIR/bin/python"
"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"
