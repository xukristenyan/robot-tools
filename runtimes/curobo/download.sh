#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${CUROBO_DATA_DIR:-${ROBOT_TOOLS_MODELS_ROOT:-/data/${USER}/robot-tools-models}/curobo}"
test -f "$HERE/upstream/curobo/content/configs/robot/franka.yml"
mkdir -p "$DATA_DIR/assets" "$DATA_DIR/cache"
echo "cuRobo has no learned weights. Robot assets are included in the pinned upstream."
echo "Uploaded meshes and runtime caches: $DATA_DIR"
