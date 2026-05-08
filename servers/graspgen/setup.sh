#!/usr/bin/env bash
# Manual install steps for the real GraspGen model — what `uv sync` can't do.
#
# Idempotent-ish (re-running re-installs editable, recompiles pointnet2_ops).
# Invoked by `robot-tools setup graspgen` or `bash setup.sh` directly.

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

# Run from upstream/ so uv reads upstream's [tool.uv] (find-links for PyG
# wheels + prerelease allow). VIRTUAL_ENV still pins the install target.
echo "[graspgen setup 1/2] Installing GraspGen package (editable)..."
( cd "$HERE/upstream" && uv pip install -e . )

# GraspGen pins numpy==1.26.4 (torch 2.1.0 was compiled against numpy 1.x).
# The robot-tools light deps allow numpy>=1.26 so a 2.x might already be
# installed by the prior `uv sync` and won't get auto-downgraded above.
uv pip install --reinstall-package numpy 'numpy==1.26.4'

echo "[graspgen setup 2/2] Compiling pointnet2_ops..."
( cd "$HERE/upstream" && bash install_uv_pointnet.sh )

echo
echo "[graspgen setup done]"
echo "  GraspGen installed in: $VENV_DIR"
echo "  Next: bash download.sh   (or: robot-tools download graspgen)"
