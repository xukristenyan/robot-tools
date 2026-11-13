#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv"
REAL_PROJECT="$HERE/real"
DATA_DIR="${POINTSAM_DATA_DIR:-${ROBOT_TOOLS_MODELS_ROOT:-/data/${USER}/robot-tools-models}/pointsam}"
export POINTSAM_DATA_DIR="$DATA_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DIR/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$DATA_DIR/python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DIR/cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$DATA_DIR/cache/cuda}"
export TMPDIR="${POINTSAM_TMPDIR:-$DATA_DIR/tmp}"
mkdir -p "$DATA_DIR/source" "$UV_CACHE_DIR" "$TMPDIR"

# Downloaded sources stay on the model disk, rather than inside the checkout.
# Only Point-SAM and its pinned torkit3d submodule are fetched; Apex is not used.
UPSTREAM_DIR="$DATA_DIR/source/Point-SAM"
REVISION="25f4fd9675479f3c951e830771595855ff7ed466"
if [[ ! -e "$UPSTREAM_DIR" ]]; then
    git clone --no-checkout https://github.com/zyc00/Point-SAM.git "$UPSTREAM_DIR"
    git -C "$UPSTREAM_DIR" checkout --detach "$REVISION"
fi
if [[ "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" != "$REVISION" ]]; then
    echo "Point-SAM source differs from the pinned revision; select a fresh POINTSAM_DATA_DIR." >&2
    exit 1
fi

git -C "$UPSTREAM_DIR" submodule update --init -- third_party/torkit3d
if [[ ! -e "$HERE/upstream" && ! -L "$HERE/upstream" ]]; then
    ln -s "$UPSTREAM_DIR" "$HERE/upstream"
fi
if [[ "$(readlink -f "$HERE/upstream")" != "$UPSTREAM_DIR" ]]; then
    echo "Existing upstream link points outside POINTSAM_DATA_DIR; choose matching storage." >&2
    exit 1
fi
COMPAT_PATCH="$HERE/compat/torch-build.patch"
TORKIT_DIR="$UPSTREAM_DIR/third_party/torkit3d"
if patch --dry-run --forward --silent -d "$TORKIT_DIR" -p1 < "$COMPAT_PATCH" >/dev/null 2>&1; then
    patch --forward --batch -d "$TORKIT_DIR" -p1 < "$COMPAT_PATCH"
elif ! patch --dry-run --reverse --silent -d "$TORKIT_DIR" -p1 < "$COMPAT_PATCH" >/dev/null 2>&1; then
    echo "torkit3d build patch does not match the pinned source." >&2
    exit 1
fi

# Same shared light/real .venv contract as the other runtimes, on the data disk.
if [[ -d "$VENV_DIR" && ! -L "$VENV_DIR" ]]; then
    if [[ -e "$DATA_DIR/venv" ]]; then
        echo "Both local .venv and data/venv exist; choose one before installing." >&2
        exit 1
    fi
    mv "$VENV_DIR" "$DATA_DIR/venv"
fi
if [[ ! -e "$VENV_DIR" && ! -L "$VENV_DIR" ]]; then
    ln -s "$DATA_DIR/venv" "$VENV_DIR"
fi
if [[ -L "$VENV_DIR" && ! -e "$VENV_DIR" ]]; then
    uv venv --python 3.12 "$DATA_DIR/venv"
fi

# Match the existing GraspGen CUDA 12.0 / RTX 5090 build convention.
# compute_89 PTX is JIT-compiled by the driver on Blackwell.
export CC="${CC:-/usr/bin/gcc-12}"
export CXX="${CXX:-/usr/bin/g++-12}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC_PATH")")}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-4}"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --project "$REAL_PROJECT" --locked
uv pip check --python "$VENV_DIR/bin/python"
"$VENV_DIR/bin/python" "$HERE/doctor.py"
"$VENV_DIR/bin/python" -m robot_tools.runtime_state record "$HERE"
