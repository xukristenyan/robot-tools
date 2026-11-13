#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${POINTSAM_DATA_DIR:-${ROBOT_TOOLS_MODELS_ROOT:-/data/${USER}/robot-tools-models}/pointsam}"
DEST="$DATA_DIR/weights"
REVISION="fb2cd5cd8047e3681491d82fc41b880a9a914fc8"
mkdir -p "$DEST"
TARGET="$DEST/model.safetensors"
EXPECTED_BYTES=1244245632
# The full safetensors structure and tensor names are validated at model load.
if [[ -f "$TARGET" && "$(stat -c %s "$TARGET")" == "$EXPECTED_BYTES" ]]; then
    echo "[skip] Point-SAM checkpoint: $TARGET"
    exit 0
fi
PARTIAL="$(mktemp "$DEST/model.XXXXXX.partial")"
trap 'rm -f "$PARTIAL"' EXIT
curl --fail --location --retry 3 \
    "https://huggingface.co/yuchen0187/Point-SAM/resolve/$REVISION/model.safetensors" \
    --output "$PARTIAL"
if [[ "$(stat -c %s "$PARTIAL")" != "$EXPECTED_BYTES" ]]; then
    echo "Point-SAM checkpoint is incomplete or differs from the pinned artifact." >&2
    exit 1
fi
mv "$PARTIAL" "$TARGET"
echo "[done] Point-SAM checkpoint: $TARGET"
