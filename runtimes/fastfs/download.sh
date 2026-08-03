#!/usr/bin/env bash
# Fetch Fast-FoundationStereo weights from the official Google Drive folder.
# gdown runs as an isolated, pinned uv tool and does not pollute the inference
# environment.
#
# Usage:
#     bash download.sh                    # download into ./weights
#     DEST=/some/where bash download.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-${FFS_WEIGHTS_DIR:-$HERE/weights}}"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"
GDOWN_VERSION="6.1.0"

mkdir -p "$DEST"
echo "[clone] $GDRIVE_FOLDER -> $DEST"
uvx --from "gdown==$GDOWN_VERSION" gdown --folder "$GDRIVE_FOLDER" -O "$DEST"

echo
echo "[done] Weights at: $DEST"
echo "       Subfolders (one per checkpoint id):"
ls -1 "$DEST" | head
