#!/usr/bin/env bash
# Sync the original GraspGen diffusion runtime from one exact uv lock.
# External host, compiler, storage, and verification requirements:
# PREREQUISITES.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
UPSTREAM_DIR="$HERE/upstream"
COMPAT_PATCH="$HERE/compat/lazy-ptv3.patch"
REAL_PROJECT="$HERE/real"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[error] $VENV_DIR does not exist." >&2
    echo "        Run \`robot-tools sync <services.yaml>\` first to create it." >&2
    exit 1
fi

if [[ ! -f "$UPSTREAM_DIR/pyproject.toml" ]]; then
    echo "[error] GraspGen upstream is not initialized at $UPSTREAM_DIR." >&2
    echo "        Run \`robot-tools install <services.yaml>\` so only the selected" >&2
    echo "        upstream submodule is initialized before this script runs." >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[error] nvcc is required to compile pointnet2_ops." >&2
    echo "        Install a CUDA toolkit (12.0 or newer), then retry." >&2
    exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

# The released 2f_140 model uses PointNet++, while the released Franka and
# suction models use PTV3. Keep PointNet++ and load PTV3 lazily so either
# backbone can start without coupling its imports to the other.
echo "[graspgen prepare] Applying pinned-upstream compatibility patch..."
if patch --dry-run --forward --silent -d "$UPSTREAM_DIR" -p1 < "$COMPAT_PATCH" >/dev/null 2>&1; then
    patch --forward --batch -d "$UPSTREAM_DIR" -p1 < "$COMPAT_PATCH"
elif patch --dry-run --reverse --silent -d "$UPSTREAM_DIR" -p1 < "$COMPAT_PATCH" >/dev/null 2>&1; then
    echo "[skip] Compatibility patch already applied."
else
    echo "[error] Compatibility patch does not match the pinned GraspGen upstream." >&2
    echo "        Restore submodule commit a56d518f and retry." >&2
    exit 1
fi

# robot-tools maintains the missing 2f_85 config and collision asset as a
# clearly separated overlay. It retargets the released Franka checkpoint.
if [[ -d "$HERE/patches" ]]; then
    cp -r "$HERE/patches/." "$UPSTREAM_DIR/"
fi

# CUDA 12.0 cannot emit sm_120 cubins, but it can emit compute_89 PTX, which
# the Blackwell driver JITs. Callers can override these defaults on another
# verified toolchain.
export CC="${CC:-/usr/bin/gcc-12}"
export CXX="${CXX:-/usr/bin/g++-12}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
NVCC_PATH="$(command -v nvcc)"
DEFAULT_CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export CUDA_HOME="${CUDA_HOME:-$DEFAULT_CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"

if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
    echo "[error] gcc-12/g++-12 are required by the verified CUDA 12.x toolchain." >&2
    echo "        Set CC, CXX, and CUDAHOSTCXX explicitly if installed elsewhere." >&2
    exit 1
fi

echo "[graspgen install] Syncing locked inference environment and CUDA extension..."
uv sync --project "$REAL_PROJECT" --locked

echo "[graspgen verify] Checking dependency graph and exercising CUDA backbones..."
uv pip check --python "$VENV_DIR/bin/python"
PYTHONPATH="$UPSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}" "$VENV_DIR/bin/python" - <<'PY'
import spconv
import spconv.pytorch as spconv_torch
import torch
import torch_scatter
from pointnet2_ops import pointnet2_utils

assert torch.__version__ == "2.10.0+cu128", torch.__version__
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available after installing torch cu128")

xyz = torch.rand(1, 64, 3, device="cuda")
indices = pointnet2_utils.furthest_point_sample(xyz, 8)

source = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
pointer = torch.tensor([0, 2, 4], device="cuda")
segments = torch_scatter.segment_csr(source, pointer, reduce="mean")

features = torch.randn(4, 3, device="cuda")
sparse_indices = torch.tensor(
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
    dtype=torch.int32,
    device="cuda",
)
sparse = spconv_torch.SparseConvTensor(features, sparse_indices, [4, 4, 4], 1)
conv = spconv_torch.SubMConv3d(3, 8, 3, padding=1, bias=False).cuda()
conv_output = conv(sparse)
torch.cuda.synchronize()
assert indices.shape == (1, 8)
assert segments.shape == (2, 3)
assert conv_output.features.shape == (4, 8)
print(
    "GraspGen CUDA probes passed:",
    torch.__version__,
    torch.cuda.get_device_name(0),
    torch_scatter.__version__,
    spconv.__version__,
)
PY

"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"

echo
echo "[graspgen install done]"
echo "  GraspGen deps installed in: $VENV_DIR"
echo "  Next: robot-tools download"
