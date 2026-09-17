#!/usr/bin/env bash
# Official public checkpoint archive, pinned by repository revision.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ANYPLACE_WEIGHTS_DIR:-/data/xkyan3/robot-tools-models/anyplace}"
REVISION="669f1b0ebcbe2ae3a72970ff31e911e8af73b2d6"
mkdir -p "$DEST"
ARCHIVE="${ANYPLACE_CHECKPOINT_ARCHIVE:-$DEST/anyplace_ckpts.zip}"
if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 3 \
        "https://huggingface.co/datasets/yuchiallanzhao/anyplace/resolve/$REVISION/anyplace_ckpts.zip" \
        --output "$ARCHIVE.partial"
    mv "$ARCHIVE.partial" "$ARCHIVE"
fi
uv run --project "$HERE" --no-sync python "$HERE/extract_checkpoint.py" "$ARCHIVE" "$DEST"
