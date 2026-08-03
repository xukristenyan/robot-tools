#!/usr/bin/env bash
# Fetch GraspGenX checkpoints + gripper assets from HuggingFace.
#
# Clones into upstream/ext/ — the default locations resolved by the pinned
# graspgenx source, so no env vars are needed afterwards. robot-tools disables
# upstream's import-time auto-clone: this explicit command is the only normal
# path that downloads these repositories.
#
# Overrides, if you keep the checkouts elsewhere:
#     export GRASPGENX_CHECKPOINT_DIR=/path/to/graspgenx_checkpoints
#     export GRASPGENX_GRIPPER_CFG_DIR=/path/to/gripper_descriptions
#
# Idempotent: skips a repo if already present.

set -euo pipefail

# Non-interactive SSH/launcher shells may not source the user's shell rc. The
# official user-local Git LFS installer places its executable here.
export PATH="${HOME}/.local/bin:${PATH}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$HERE/upstream/ext"

CKPT_REPO="https://huggingface.co/adithyamurali/GraspGenXModel"
CKPT_DEST="${GRASPGENX_CHECKPOINT_DIR:-$EXT_DIR/graspgenx_checkpoints}"

GRIPPER_REPO="https://huggingface.co/datasets/adithyamurali/gripper_descriptions"
GRIPPER_DEST="${GRASPGENX_GRIPPER_CFG_DIR:-$EXT_DIR/gripper_descriptions}"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[error] git-lfs is required (HuggingFace repos use LFS). Install it:" >&2
    echo "    sudo apt install git-lfs && git lfs install" >&2
    exit 1
fi
git lfs install --skip-repo

mkdir -p "$EXT_DIR"

clone_if_missing() {
    local url="$1" dest="$2" label="$3"
    if [[ -d "$dest/.git" ]]; then
        echo "[skip] $label already at $dest. To update: cd '$dest' && git pull"
        return 0
    fi
    echo "[clone] $url -> $dest  ($label, LFS assets may take a while)"
    git clone --depth 1 "$url" "$dest"
}

clone_if_missing "$CKPT_REPO"    "$CKPT_DEST"    "graspgenx checkpoints"
clone_if_missing "$GRIPPER_REPO" "$GRIPPER_DEST" "gripper_descriptions"

echo
echo "[done] checkpoints:          $CKPT_DEST"
echo "       gripper_descriptions: $GRIPPER_DEST"
if [[ -n "${GRASPGENX_CHECKPOINT_DIR:-}" || -n "${GRASPGENX_GRIPPER_CFG_DIR:-}" ]]; then
    echo "       (custom paths — keep the GRASPGENX_* env vars exported when launching)"
fi
