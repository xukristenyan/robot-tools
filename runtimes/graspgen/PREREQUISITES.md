# Original GraspGen external prerequisites

This file is the definitive checklist for requirements outside `install.sh`.
`INSTALL.md` explains the CUDA/PTX rationale in more detail. The install script
syncs the locked model environment and compiles PointNet++ inside an existing
venv; it does
not initialize source, install the system CUDA compiler/toolchain, install Git
LFS, create the venv, or download checkpoints.

## Verified baseline

| Item | Verified value |
|---|---|
| Platform | Ubuntu 24.04.4 LTS, x86_64 |
| GPU / driver | RTX 5090 / 580.173.02 |
| Python | uv-managed 3.10.20 |
| uv | 0.11.21 |
| upstream commit | `a56d518f3b76ea2a432b5b838b3c68027d29be49` |
| GPU packages | Torch 2.10.0+cu128, torch-scatter pt210cu128, SpConv cu126 |
| system CUDA compiler | nvcc 12.0 |
| host compiler | `/usr/bin/gcc-12`, `/usr/bin/g++-12`, version 12.4.0 |
| checkpoints | Hugging Face `adithyamurali/GraspGenModels` |

The install compiles PointNet++ with `TORCH_CUDA_ARCH_LIST=8.9+PTX`. The RTX
5090 driver JITs that forward-compatible PTX for Blackwell. A different
compiler/GPU combination must rerun the CUDA probes and is not covered by the
jarjar result.

## System-level requirements

These must exist before install:

```bash
command -v uv git git-lfs patch nvcc
test -x /usr/bin/gcc-12
test -x /usr/bin/g++-12
nvidia-smi
git lfs install
```

If the compiler paths differ, export `CC`, `CXX`, and `CUDAHOSTCXX` explicitly.
Do not point `CUDA_HOME` at a directory that does not contain the selected
`nvcc`.

The model checkout is about 7.8 GiB including its local Git LFS object store.
The complete environment and build artifacts need additional space.

## Model storage

```bash
export ROBOT_TOOLS_MODELS_ROOT="/data/${USER}/robot-tools-models"
export GRASPGEN_MODELS_DIR="$ROBOT_TOOLS_MODELS_ROOT/graspgen/GraspGenModels"
mkdir -p "$(dirname "$GRASPGEN_MODELS_DIR")"
```

`download.sh` uses `DEST`, then `GRASPGEN_MODELS_DIR`, then the runtime-local
default, in that order. Keep `GRASPGEN_MODELS_DIR` exported when launching from
shared storage.

## Clean-machine sequence

```bash
git submodule update --init runtimes/graspgen/upstream
test "$(git -C runtimes/graspgen/upstream rev-parse HEAD)" = \
  "a56d518f3b76ea2a432b5b838b3c68027d29be49"

uv sync --locked
uv run robot-tools sync
uv run robot-tools install

DEST="$GRASPGEN_MODELS_DIR" uv run robot-tools download
git -C "$GRASPGEN_MODELS_DIR" lfs fsck

# Keep GRASPGEN_MODELS_DIR exported and remove --fake before launching.
uv run robot-tools launch
```

`services.yaml` must include `graspgen`. The install is idempotent: it recognizes
an already-applied compatibility patch, and uv reuses the exact locked
PointNet++ build when its inputs have not changed.

## Required verification

The final install phase already executes real CUDA operations for PointNet++,
torch-scatter, and SpConv. Also run:

```bash
uv pip check --python runtimes/graspgen/.venv/bin/python
test -s "$GRASPGEN_MODELS_DIR/checkpoints/graspgen_franka_panda_gen.pth"
test -s "$GRASPGEN_MODELS_DIR/checkpoints/graspgen_franka_panda_dis.pth"
```

Then launch real gripper configurations and run typed inference requests. The
jarjar verification covers Franka/PTV3, Robotiq 2F-140/PointNet++, and the
collision-free action. The service supports the original diffusion path;
GraspMoE belongs to the separate GraspGenX service.

## Conditions that require revalidation

- upstream no longer at the pinned commit or its worktree has unrelated edits;
- different system nvcc/host compiler, GPU architecture, or driver family;
- incomplete Git LFS objects despite a successful-looking clone;
- moving `GRASPGEN_MODELS_DIR` without exporting the new value at launch;
- running plain `uv sync` in `runtimes/graspgen` after install, which restores
  the lightweight fake/server environment; rerun `install.sh` afterward;
- changing the real lock, PointNet++ source, or CUDA build variables.
