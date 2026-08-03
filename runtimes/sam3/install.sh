#!/usr/bin/env bash
# Sync the locked real SAM3 environment into this runtime's shared venv.
# CUDA 12.8 wheels are required for Blackwell GPUs such as the RTX 5090
# (sm_120), and match the FastFS environment so uv can share cached artifacts.
# External host, access, storage, and verification requirements: PREREQUISITES.md.
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

if [[ ! -f "$HERE/upstream/pyproject.toml" ]]; then
    echo "[error] SAM3 upstream is not initialized." >&2
    echo "        Run \`robot-tools install <services.yaml>\` so only the selected" >&2
    echo "        upstream submodule is initialized before this script runs." >&2
    exit 1
fi

REAL_PROJECT="$HERE/real"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

echo "[sam3 install] Syncing locked SAM3 + PyTorch 2.10.0 cu128 environment..."
uv sync --project "$REAL_PROJECT" --locked

echo "[sam3 verify] Checking the installed dependency graph..."
uv pip check --python "$VENV_DIR/bin/python"
"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"

echo
echo "[sam3 install done]"
echo "  SAM 3 deps installed in: $VENV_DIR"
echo
echo "  Next, gain access to the gated weights (one-time, system-wide):"
echo "    1. Visit https://huggingface.co/facebook/sam3 and request access"
echo "    2. Once granted, run: hf auth login   (paste an HF access token)"
echo "    3. Pre-fetch weights: robot-tools download"
echo "       Otherwise, weights download lazily on the first launch."
