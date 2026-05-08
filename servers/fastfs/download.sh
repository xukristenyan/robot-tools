#!/usr/bin/env bash
# Fetch Fast-FoundationStereo weights from the official Google Drive folder.
# Idempotent: skips checkpoints already on disk.
#
# Requires `gdown` in this server's venv (installed by setup.sh).
#
# Usage:
#     bash download.sh                    # download into ./weights
#     DEST=/some/where bash download.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$HERE/weights}"
VENV_DIR="$HERE/.venv"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist. Run \`robot-tools sync\` + \`setup fastfs\` first." >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

if ! command -v gdown >/dev/null 2>&1; then
    echo "[error] gdown not in venv. Run \`robot-tools setup fastfs\` first." >&2
    exit 1
fi

mkdir -p "$DEST"
echo "[clone] $GDRIVE_FOLDER -> $DEST"
gdown --folder "$GDRIVE_FOLDER" -O "$DEST"

echo
echo "[done] Weights at: $DEST"
echo "       Subfolders (one per checkpoint id):"
ls -1 "$DEST" | head
