#!/usr/bin/env bash
# Manual install steps for Fast-FoundationStereo.
# FFS upstream has no setup.py, so we install torch + xformers from the cu124
# index and then `pip install -r requirements.txt`.
#
# Invoked by `robot-tools setup fastfs` or `bash setup.sh` directly.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist." >&2
    echo "        Run \`robot-tools sync <services.yaml>\` first to create it." >&2
    exit 1
fi

# Pin uv pip to this server's venv regardless of cwd.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

echo "[fastfs setup 1/3] Installing PyTorch 2.6.0 + xformers (cu124)..."
uv pip install torch==2.6.0 torchvision==0.21.0 xformers \
    --index-url https://download.pytorch.org/whl/cu124

echo "[fastfs setup 2/3] Installing FFS requirements.txt..."
uv pip install -r "$HERE/upstream/requirements.txt"

echo "[fastfs setup 3/3] Installing gdown for download.sh..."
uv pip install gdown

echo
echo "[fastfs setup done]"
echo "  FFS deps installed in: $VENV_DIR"
echo "  Next: bash download.sh   (or: robot-tools download fastfs)"
