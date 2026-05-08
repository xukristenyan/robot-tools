#!/usr/bin/env bash
# Install SAM 3 (torch 2.7 cu126 + sam3 package) into this server's venv.
# Invoked by `robot-tools setup sam3` or `bash setup.sh` directly.

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

echo "[sam3 setup 1/3] Installing PyTorch 2.7.0 (cu126)..."
uv pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

echo "[sam3 setup 2/3] Installing SAM 3 (editable)..."
( cd "$HERE/upstream" && uv pip install -e . )

# SAM 3 upstream's pyproject.toml `dependencies` is incomplete — runtime needs
# many packages that live in [notebooks]/[train] extras. Pull the inference
# subset directly. Use opencv-headless (no X11/Qt) since this is a server.
echo "[sam3 setup 3/3] Installing missing runtime deps..."
uv pip install \
    einops \
    hydra-core \
    opencv-python-headless \
    pycocotools \
    scipy \
    scikit-image \
    scikit-learn \
    psutil

echo
echo "[sam3 setup done]"
echo "  SAM 3 deps installed in: $VENV_DIR"
echo
echo "  Next, gain access to the gated weights (one-time, system-wide):"
echo "    1. Visit https://huggingface.co/facebook/sam3 and request access"
echo "    2. Once granted, run: hf auth login   (paste an HF access token)"
echo "    3. (Optional) Pre-fetch weights:  robot-tools download sam3"
echo "       Otherwise, weights download lazily on the first launch."
