#!/usr/bin/env bash
# Official Large relative-depth checkpoint at a fixed upstream revision.
set -euo pipefail
DEST="${DEST:-${DEPTH_ANYTHING_V2_WEIGHTS_DIR:-/data/${USER}/robot-tools-models/depth_anything_v2}}"
REVISION="cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9"
NAME="depth_anything_v2_vitl.pth"
mkdir -p "$DEST"
TARGET="$DEST/$NAME"
if [[ -s "$TARGET" ]]; then
    echo "[skip] Existing Large checkpoint: $TARGET"
    exit 0
fi
PARTIAL="$(mktemp "$DEST/$NAME.XXXXXX.partial")"
trap 'rm -f "$PARTIAL"' EXIT
curl --fail --location --retry 3 \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/$REVISION/$NAME" \
    --output "$PARTIAL"
test -s "$PARTIAL"
mv "$PARTIAL" "$TARGET"
echo "[done] Large checkpoint: $TARGET"
