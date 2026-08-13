# GraspGenX external prerequisites

This file lists everything required outside `install.sh`. The script syncs an
exact inference-only Torch/cu128 lock, patches packaging side effects, and runs
import/CUDA probes in an existing venv; it does not initialize source, create the venv,
install Git LFS, download checkpoints/gripper assets, or preserve custom model
paths for future launches.

## Verified baseline

| Item | Verified value |
|---|---|
| Platform | Ubuntu 24.04.4 LTS, x86_64 |
| GPU / driver | RTX 5090 / 580.173.02 |
| Python | uv-managed 3.10.20 |
| uv | 0.11.21 |
| upstream commit | `b9429097728cb1c430dd78b92edf17ba318aad03` |
| GPU packages | Torch 2.10.0+cu128, TorchVision 0.25.0+cu128 |
| checkpoint | Hugging Face `adithyamurali/GraspGenXModel`, `release` |
| gripper assets | Hugging Face dataset `adithyamurali/gripper_descriptions` |

The released `ptv3_vanilla` backbone is pure PyTorch. System CUDA toolkit,
nvcc, PointNet++, SpConv, and torch-scatter are not required. A working NVIDIA
driver, `git`, Git LFS, `uv`, outbound HTTPS, and enough disk space are
required.

The verified checkpoint checkout is about 3.2 GiB. The selected gripper
checkout is about 129 MiB, and the locked service venv on jarjar is about 7.2
GiB.

## Model storage

```bash
export ROBOT_TOOLS_MODELS_ROOT="/data/${USER}/robot-tools-models"
export GRASPGENX_CHECKPOINT_DIR="$ROBOT_TOOLS_MODELS_ROOT/graspgenx/checkpoints"
export GRASPGENX_GRIPPER_CFG_DIR="$ROBOT_TOOLS_MODELS_ROOT/graspgenx/gripper_descriptions"
mkdir -p "$GRASPGENX_CHECKPOINT_DIR" "$GRASPGENX_GRIPPER_CFG_DIR"
```

Both `GRASPGENX_*` variables must remain exported during download and every
real launch. If they are absent, the runtime checks
`runtimes/graspgenx/upstream/ext`; robot-tools disables upstream's import-time
auto-clone, so missing data causes launch to fail rather than creating another
copy on the home filesystem.

## Clean-machine sequence

```bash
git submodule update --init runtimes/graspgenx/upstream
test "$(git -C runtimes/graspgenx/upstream rev-parse HEAD)" = \
  "b9429097728cb1c430dd78b92edf17ba318aad03"
command -v git-lfs
git lfs install

uv sync --locked
uv run robot-tools sync
uv run robot-tools install
uv run robot-tools download
```

The current deployment only needs `franka_panda` and `robotiq_2f_85`. Ensure
their LFS objects are present without fetching every supported gripper:

```bash
git -C "$GRASPGENX_GRIPPER_CFG_DIR" lfs pull \
  --include="gripper_descriptions/assets/x_grippers/franka_panda/**,gripper_descriptions/assets/x_grippers/robotiq_2f_85/**" \
  --exclude=""
```

Keep the environment variables exported, remove `--fake`, and optionally add
`--default-gripper robotiq_2f_85` before launching.

## Required verification

```bash
uv pip check --python runtimes/graspgenx/.venv/bin/python
git -C "$GRASPGENX_CHECKPOINT_DIR" lfs fsck
test -s "$GRASPGENX_CHECKPOINT_DIR/release/gen/epoch_736.pth"
test -s "$GRASPGENX_CHECKPOINT_DIR/release/dis/epoch_1056.pth"
runtimes/graspgenx/.venv/bin/python -c \
  'import torch; assert torch.__version__ == "2.10.0+cu128"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

A full `git lfs fsck` of the selectively populated gripper repository will
report the intentionally absent grippers. Validate the two required folders
by loading both named grippers or by running a real named-gripper HTTP request.
The jarjar baseline covered named Franka/Robotiq and scene-depth inference
before service API v4. The current `generate_safe_grasps` and
`generate_safe_grasps_for_all` actions require a new real GPU run before they
can be treated as end-to-end verified.

## Conditions that require revalidation

- upstream no longer at the pinned commit or its compatibility patch no longer
  applies cleanly;
- different GPU architecture, Python minor version, or NVIDIA driver family;
- missing/incomplete Git LFS objects;
- launching without the same `GRASPGENX_*` environment variables used for
  download;
- enabling TensorRT, which is intentionally outside the tested baseline;
- running plain `uv sync` in `runtimes/graspgenx` after install, which restores
  the lightweight fake/server environment; rerun `install.sh` afterward;
- changing the real lock or either compatibility patch.
