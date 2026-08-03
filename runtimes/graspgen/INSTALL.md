# Install the original GraspGen on an RTX 5090 server

This is the compatibility recipe used by `install.sh`. It keeps the pinned,
original diffusion GraspGen and does not add GraspMoE. GraspMoE is
provided by the separate GraspGenX service.

For the definitive clean-machine checklist—including Git LFS, compiler
commands, persistent model storage, and verification—read
[`PREREQUISITES.md`](PREREQUISITES.md). This file explains why the unusual
CUDA/PTX choices are necessary.

Do **not** use the old torch 2.1/cu121 + `TORCH_CUDA_ARCH_LIST=8.6` recipe on an
RTX 5090. Blackwell is compute capability 12.0, and that combination cannot
execute its CUDA kernels.

## Prerequisites

- NVIDIA driver capable of running CUDA 12.8 PyTorch wheels (jarjar uses 580.x)
- `uv`
- a system CUDA toolkit with `nvcc` 12.0 or newer
- `gcc-12` and `g++-12`
- the pinned GraspGen git submodule

## Steps

From the robot-tools checkout:

```bash
git submodule update --init runtimes/graspgen/upstream
uv run robot-tools sync
uv run robot-tools install
uv run robot-tools download
```

`install.sh` applies the tracked `compat/lazy-ptv3.patch`, then performs one
locked `uv sync` from `real/`. That lock selects torch 2.10.0/torchvision 0.25.0
from the cu128 index, the released checkpoint's PTV3 runtime (`torch-scatter`
cu128 + SpConv cu126), and the local PointNet++ CUDA extension. It finishes by
running PointNet++, `segment_csr`, and sparse-conv CUDA operations, so a
successful install validates more than package imports.

GraspGen itself is not installed as a distribution. `backend.py` imports the
pinned source tree directly, avoiding upstream metadata that otherwise installs
training, visualization, USD, and demo packages unused by this HTTP service.

## Why this is needed

| Problem | Cause | Fix |
|---------|-------|-----|
| `no kernel image is available` | torch 2.1/cu121 or an sm_86-only extension on a 5090 | Use the install script's torch 2.10/cu128 path |
| CUDA 12.0 cannot compile `sm_120` | native Blackwell code generation arrived later | Build `8.9+PTX`; the 580 driver JITs it for sm_120 |
| `unsupported GNU version` | CUDA 12.0 does not support the system gcc-13 | Use gcc-12/g++-12 |
| a PointNet-only config asks for spconv/scatter | upstream imports PTV3 eagerly | The compatibility patch makes PTV3 imports lazy |
| released checkpoint selects PTV3 | the real config needs scatter and sparse convolution | Install the torch 2.10/cu128 PyG wheel and tested SpConv cu126 wheel |

The `8.9+PTX` fallback was tested on jarjar by compiling a CUDA extension with
system nvcc 12.0 and executing it on an NVIDIA GeForce RTX 5090. If a native
CUDA 12.8+ toolkit is later installed, `TORCH_CUDA_ARCH_LIST=12.0` may be used
instead.
