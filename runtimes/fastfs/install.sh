#!/usr/bin/env bash
# Sync the locked real FastFoundationStereo environment into this runtime's
# shared venv. CUDA 12.8 wheels are required for Blackwell GPUs such as the
# RTX 5090 (sm_120).
# External host, storage, and verification requirements: PREREQUISITES.md.
#
# Invoked by `robot-tools install` or `bash install.sh` directly.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist." >&2
    echo "        Run \`robot-tools sync <services.yaml>\` first to create it." >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

if [[ ! -f "$HERE/upstream/core/foundation_stereo.py" ]]; then
    echo "[error] FastFoundationStereo upstream is not initialized." >&2
    echo "        Run \`robot-tools install <services.yaml>\` so only the selected" >&2
    echo "        upstream submodule is initialized before this script runs." >&2
    exit 1
fi

REAL_PROJECT="$HERE/real"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

echo "[fastfs install] Syncing locked inference-only environment..."
uv sync --project "$REAL_PROJECT" --locked

echo "[fastfs verify] Checking dependency graph and upstream imports..."
uv pip check --python "$VENV_DIR/bin/python"
PYTHONPATH="$HERE/upstream${PYTHONPATH:+:$PYTHONPATH}" "$VENV_DIR/bin/python" - <<'PY'
import cv2
import imageio
import omegaconf
import timm
import torch
import torchvision

from core.foundation_stereo import FastFoundationStereo

assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torchvision.__version__ == "0.25.0+cu128", torchvision.__version__
print(
    "FastFoundationStereo imports passed:",
    torch.__version__,
    torchvision.__version__,
    timm.__version__,
    cv2.__version__,
    imageio.__version__,
    omegaconf.__version__,
    FastFoundationStereo.__name__,
)
PY

"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"

echo
echo "[fastfs install done]"
echo "  FFS deps installed in: $VENV_DIR"
echo "  Next: robot-tools download"
