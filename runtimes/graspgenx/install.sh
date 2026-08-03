#!/usr/bin/env bash
# Sync the real GraspGenX inference environment from one exact uv lock.
# External host, asset, storage, and verification requirements:
# PREREQUISITES.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
UPSTREAM_DIR="$HERE/upstream"
COMPAT_PATCH="$HERE/compat/no-auto-download.patch"
LAZY_ZMQ_PATCH="$HERE/compat/lazy-zmq.patch"
REAL_PROJECT="$HERE/real"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist." >&2
    echo "        Run \`robot-tools sync <services.yaml>\` first to create it." >&2
    exit 1
fi

if [[ ! -f "$UPSTREAM_DIR/pyproject.toml" ]]; then
    echo "[error] GraspGenX upstream is not initialized at $UPSTREAM_DIR." >&2
    echo "        Run \`robot-tools install <services.yaml>\` so only the selected" >&2
    echo "        upstream submodule is initialized before this script runs." >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

# Importing the official package otherwise clones checkpoints and gripper
# assets as a side effect. robot-tools makes network/storage mutations explicit
# through `robot-tools download` instead.
apply_compat_patch() {
    local patch_file="$1"
    local label="$2"
    if patch --dry-run --forward --silent -d "$UPSTREAM_DIR" -p1 < "$patch_file" >/dev/null 2>&1; then
        patch --forward --batch -d "$UPSTREAM_DIR" -p1 < "$patch_file"
    elif patch --dry-run --reverse --silent -d "$UPSTREAM_DIR" -p1 < "$patch_file" >/dev/null 2>&1; then
        echo "[skip] $label patch already applied."
    else
        echo "[error] $label patch does not match the pinned GraspGenX upstream." >&2
        echo "        Restore submodule commit b9429097 and retry." >&2
        exit 1
    fi
}

echo "[graspgenx prepare] Removing implicit download and ZMQ coupling..."
apply_compat_patch "$COMPAT_PATCH" "No-auto-download"
apply_compat_patch "$LAZY_ZMQ_PATCH" "Lazy-ZMQ"

echo "[graspgenx install] Syncing locked inference-only environment..."
uv sync --project "$REAL_PROJECT" --locked

echo "[graspgenx verify] Checking imports, dependency graph, and CUDA..."
uv pip check --python "$VENV_DIR/bin/python"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
PYGLET_HEADLESS="${PYGLET_HEADLESS:-true}" \
PYTHONPATH="$UPSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$VENV_DIR/bin/python" - <<'PY'
import torch

import graspgenx
from graspgenx.grasp_server import GraspGenXSampler, load_grasp_gen_model
from graspgenx.samplers import run_planner_on_batch, run_planner_on_object
from graspgenx.utils.scene_loaders import depth_to_camera_xyz

assert torch.__version__ == "2.10.0+cu128", torch.__version__
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available after installing torch cu128")

left = torch.randn(256, 256, device="cuda")
right = torch.randn(256, 256, device="cuda")
result = left @ right
torch.cuda.synchronize()
assert result.shape == (256, 256)
print(
    "GraspGenX imports/CUDA passed:",
    torch.__version__,
    torch.cuda.get_device_name(0),
    GraspGenXSampler.__name__,
    load_grasp_gen_model.__name__,
    run_planner_on_object.__name__,
    run_planner_on_batch.__name__,
    depth_to_camera_xyz.__name__,
)
PY

"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"

echo
echo "[graspgenx install done]"
echo "  GraspGenX deps installed in: $VENV_DIR"
echo "  Next: robot-tools download"
