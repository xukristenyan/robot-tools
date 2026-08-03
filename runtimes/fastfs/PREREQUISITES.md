# Fast-FoundationStereo external prerequisites

This file lists everything required outside `install.sh`. The script exact-syncs
the locked, inference-only real project into an existing venv; it does not
initialize source, create the venv, guarantee Google Drive access, download a
checkpoint, or preserve the model path for later launches.

## Verified baseline

| Item | Verified value |
|---|---|
| Platform | Ubuntu 24.04.4 LTS, x86_64 |
| GPU / driver | RTX 5090 / 580.173.02 |
| Python | uv-managed 3.12.13 |
| uv | 0.11.21 |
| upstream commit | `f8442a5f406d3058e060c48acbd019963e54f490` |
| GPU packages | Torch 2.10.0, TorchVision 0.25.0, cu128 |
| checkpoint | official Google Drive folder, model `23-36-37` |
| runtime setting | `valid_iters=8`, `max_disp=192` |

System CUDA toolkit/nvcc is not required for the prebuilt cu128 wheels. A
working NVIDIA driver, `git`, `uv>=0.11.21`, outbound HTTPS to
PyPI/PyTorch/Google Drive, and a standard C/C++ build toolchain are required.
uv-managed Python is intentional because Triton JIT needs Python development
headers.

The verified `23-36-37` checkpoint directory is about 68 MiB and contains
`model_best_bp2_serialize.pth` (71,098,210 bytes) plus `cfg.yaml`.

## Model storage

```bash
export ROBOT_TOOLS_MODELS_ROOT="/data/${USER}/robot-tools-models"
export FFS_WEIGHTS_DIR="$ROBOT_TOOLS_MODELS_ROOT/fastfs"
mkdir -p "$FFS_WEIGHTS_DIR"
```

`download.sh` uses `DEST`, then `FFS_WEIGHTS_DIR`, then local `weights/` in that
order. Keep `FFS_WEIGHTS_DIR` exported for every real launch when using external
model storage.

## Clean-machine sequence

```bash
git submodule update --init runtimes/fastfs/upstream
test "$(git -C runtimes/fastfs/upstream rev-parse HEAD)" = \
  "f8442a5f406d3058e060c48acbd019963e54f490"

uv sync --locked
uv run robot-tools sync
uv run robot-tools install

DEST="$FFS_WEIGHTS_DIR" uv run robot-tools download
test -s "$FFS_WEIGHTS_DIR/23-36-37/model_best_bp2_serialize.pth"

# Keep FFS_WEIGHTS_DIR exported and remove --fake before launching.
# Use extra_args: ["--model", "23-36-37", "--valid-iters", "8"]
uv run robot-tools launch
```

`services.yaml` must include `fastfs`. Google Drive can rate-limit or change
sharing behavior; for reliable multi-machine deployment, copy the verified
checkpoint from an internal artifact store instead of depending indefinitely
on gdown.

## Required verification

```bash
uv pip check --python runtimes/fastfs/.venv/bin/python
runtimes/fastfs/.venv/bin/python -c \
  'import torch; assert torch.__version__ == "2.10.0+cu128"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Then launch the real server and run one stereo pair through `FastFSClient`.
Verify finite disparity with the expected image shape; package imports do not
verify checkpoint compatibility or Triton execution.

## Conditions that require revalidation

- a different upstream commit, checkpoint ID, `valid_iters`, or `max_disp`;
- different GPU architecture, Python minor version, or driver family;
- Google Drive returning an HTML/error file instead of the checkpoint;
- `DEST` overriding `FFS_WEIGHTS_DIR` with an unintended directory;
- running plain `uv sync` in `runtimes/fastfs` after install (it intentionally
  restores the lightweight fake environment; rerun `install` for real mode).
