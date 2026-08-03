# SAM3 external prerequisites

This file lists everything required outside `install.sh`. The launcher
initializes only the selected SAM3 source and creates the service venv; the
script then exact-syncs `real/uv.lock`. Neither grants Hugging Face access,
authenticates, chooses persistent cache storage, nor downloads the gated
checkpoint.

## Verified baseline

| Item | Verified value |
|---|---|
| Platform | Ubuntu 24.04.4 LTS, x86_64 |
| GPU / driver | RTX 5090 / 580.173.02 |
| Python | uv-managed 3.12.13 |
| uv | >=0.11.21 (verified 0.11.21) |
| upstream commit | `ea46ebca5ee9c4edf0b7c7d44de069d4241a03d5` |
| Torch family | Torch 2.10.0, TorchVision 0.25.0, cu128 |
| model | gated Hugging Face repo `facebook/sam3`, file `sam3.pt` |

System CUDA toolkit/nvcc is not required for the prebuilt cu128 wheels. A
working NVIDIA driver, `git`, `uv`, outbound HTTPS, and a normal C/C++ build
toolchain are still required. uv-managed Python is intentional because it
includes headers needed by Torch/Triton JIT paths.

## Access and storage

Before installing on a new machine:

1. Request and receive access at <https://huggingface.co/facebook/sam3>.
2. Decide where the Hugging Face cache will live. The verified checkpoint
   cache is about 3.3 GiB, excluding the Python environment and uv cache.
3. Export `HF_HOME` before login, download, and every real launch. Changing it
   later changes both token and cache lookup locations.

Recommended `/data` layout:

```bash
export ROBOT_TOOLS_MODELS_ROOT="/data/${USER}/robot-tools-models"
export HF_HOME="$ROBOT_TOOLS_MODELS_ROOT/sam3/huggingface"
mkdir -p "$HF_HOME"
```

Never print or commit the Hugging Face token.

## Clean-machine sequence

```bash
uv sync --locked
# services.yaml contains sam3 and only the other services wanted on this host.
uv run robot-tools install

test "$(git -C runtimes/sam3/upstream rev-parse HEAD)" = \
  "ea46ebca5ee9c4edf0b7c7d44de069d4241a03d5"
runtimes/sam3/.venv/bin/hf auth login
uv run robot-tools download

# Keep HF_HOME exported and remove --fake before launching.
uv run robot-tools launch
```

`services.yaml` must include `sam3`. The login is performed after install so the
known service-local `hf` executable is used.

## Required verification

```bash
uv pip check --python runtimes/sam3/.venv/bin/python
runtimes/sam3/.venv/bin/hf auth whoami
runtimes/sam3/.venv/bin/python -c \
  'import torch; assert torch.__version__ == "2.10.0+cu128"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Then launch the real server and perform one typed client segmentation call;
import success alone does not verify gated checkpoint access or GPU inference.

## Conditions that require revalidation

- different SAM3 upstream commit or robot-tools install script;
- different GPU architecture, Python minor version, or NVIDIA driver family;
- moving `HF_HOME` without moving/re-downloading both token and checkpoint;
- revoked/expired Hugging Face token or gated-model approval;
- running light `robot-tools sync` after install without running `install`
  again before a real launch.
