#!/usr/bin/env bash
# Clone the official GraspGenModels checkpoints from HuggingFace.
# Idempotent: skips if already present.
#
# Usage:
#     bash download.sh           # download into ./GraspGenModels
#     DEST=/some/where bash download.sh
#
# After download, point GraspGen at it:
#     export GRASPGEN_MODELS_DIR="$(pwd)/GraspGenModels"

set -euo pipefail

# Non-interactive SSH/launcher shells may not source the user's shell rc. The
# official user-local Git LFS installer places its executable here.
export PATH="${HOME}/.local/bin:${PATH}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-${GRASPGEN_MODELS_DIR:-$HERE/GraspGenModels}}"
HF_REPO="https://huggingface.co/adithyamurali/GraspGenModels"

if [[ -d "$DEST/.git" ]]; then
    echo "[skip] $DEST already a git repo. To update: cd '$DEST' && git pull"
    exit 0
fi

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[error] git-lfs is required (HuggingFace repos use LFS). Install it:" >&2
    echo "    sudo apt install git-lfs && git lfs install" >&2
    exit 1
fi

echo "[clone] $HF_REPO -> $DEST  (~7.8 GB including Git LFS cache)"
git lfs install --skip-repo
git clone "$HF_REPO" "$DEST"

echo
echo "[done] Models at: $DEST"
echo "       Add to your shell rc:"
echo "           export GRASPGEN_MODELS_DIR='$DEST'"
