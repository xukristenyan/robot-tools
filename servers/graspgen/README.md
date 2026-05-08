# graspgen-server

Independent uv project that wraps NVlabs/GraspGen as a ZMQ server.

## Layout

```
servers/graspgen/
├── pyproject.toml         this server's uv project (own .venv)
├── server.py              BaseServer subclass; @handler-decorated methods
├── pipeline.py            headless wrapper around GraspGenSampler
├── upstream/              git submodule of NVlabs/GraspGen
├── setup.sh               manual install steps (uv pip install -e ./upstream + pointnet)
├── download.sh            fetch checkpoints from HuggingFace (~9.3 GB)
├── INSTALL.md             R2D2-specific CUDA hack (fallback for weird setups)
└── GraspGenModels/        gitignored; created by download.sh
```

## Mock mode (no model, fast)

From any project that depends on robot-tools:
```bash
uv run robot-tools sync   services.yaml             # light deps only
uv run robot-tools launch services.yaml             # services.yaml has extra_args: ["--mock"]
```

Or directly inside this dir:
```bash
uv sync
uv run python server.py --port 5557 --mock
```

## Real mode (the painful one)

`grasp_gen` is **not declared in pyproject.toml** — its torch / pointnet2_ops
build can't be expressed via `uv sync`. Use the helper scripts:

```bash
# From any consuming project:
uv run robot-tools sync     services.yaml          # creates the venv
uv run robot-tools setup    graspgen               # installs grasp_gen + pointnet2_ops
uv run robot-tools download graspgen               # fetches the 9.3GB checkpoints
# (default GRASPGEN_MODELS_DIR is <robot-tools>/servers/graspgen/GraspGenModels;
#  set it explicitly only if you put the weights elsewhere.)
uv run robot-tools launch   services.yaml          # remove --mock from extra_args first
```

`setup.sh` follows GraspGen's official uv path. If your system has the CUDA-13
vs CUDA-12.1 mismatch (R2D2-style), see `INSTALL.md` for the conda-based recipe.

## Status

`server.py` switches between mock and real via `--mock`. `pipeline.py` is the
real-model wrapper invoked when `--mock` is not set.
