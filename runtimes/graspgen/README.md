# GraspGen runtime

Independent uv project that wraps NVlabs/GraspGen as an HTTP server.

Before a real-model installation, read [`PREREQUISITES.md`](PREREQUISITES.md).
It is the definitive checklist for everything outside `install.sh`.
[`INSTALL.md`](INSTALL.md) supplements it with the RTX 5090 CUDA/PTX rationale.

## Layout

```
runtimes/graspgen/
├── pyproject.toml         lightweight fake/server project (own .venv)
├── real/                  exact real-model pyproject + uv.lock
├── PREREQUISITES.md       external host, compiler, storage, and verification contract
├── server.py              BaseServer subclass; @handler-decorated methods
├── backend.py             headless adapter around GraspGenSampler
├── doctor.py              read-only real/CUDA-extension import probe
├── upstream/              git submodule of NVlabs/GraspGen
├── patches/               files overlaid into upstream/ during install (e.g. 2f_85 support)
├── install.sh             syncs real/ into the shared service venv and runs CUDA probes
├── download.sh            fetch checkpoints from HuggingFace (~7.8 GB with LFS cache)
├── INSTALL.md             R2D2-specific CUDA hack (fallback for weird setups)
└── GraspGenModels/        gitignored; created by download.sh
```

## Fake mode (no model, fast)

From any project that depends on robot-tools:
```bash
uv run robot-tools sync        # light deps only
uv run robot-tools launch      # services.yaml has extra_args: ["--fake"]
```

Or directly inside this dir:
```bash
uv sync
uv run python server.py --port 5557 --fake
```

## Real mode (the painful one)

The abbreviated commands below assume the system compiler, Git LFS, and model
storage requirements are already satisfied. Follow
[`PREREQUISITES.md`](PREREQUISITES.md) for a new machine.

The top-level runtime project intentionally stays fake-capable and small.
`install.sh` syncs the separate, locked `real/` project into the same venv;
there are no follow-up `uv pip install` mutations:

```bash
# From any consuming project:
uv run robot-tools sync          # creates the venv
uv run robot-tools install       # installs grasp_gen + pointnet2_ops
uv run robot-tools download      # fetches checkpoints (~7.8GB with LFS cache)
# (default GRASPGEN_MODELS_DIR is <robot-tools>/runtimes/graspgen/GraspGenModels;
#  set it explicitly only if you put the weights elsewhere.)
uv run robot-tools launch        # remove --fake from extra_args first
```

The pinned upstream is imported directly from `upstream/` rather than installed
as a package, so its training, visualization, USD, and demo dependencies do not
leak into the service. The real lock keeps the original diffusion model on the
tested torch 2.10/cu128 path, includes PTV3 scatter/SpConv, and builds PointNet++
as forward-compatible PTX on Blackwell. See `INSTALL.md` for the rationale.
GraspMoE is intentionally handled by `graspgenx`, not here.

## Status

`server.py` switches between fake and real via `--fake`. `backend.py` is the
real-model wrapper invoked when `--fake` is not set.
