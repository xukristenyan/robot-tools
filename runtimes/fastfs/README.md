# FastFS runtime

Independent uv project that wraps NVlabs/Fast-FoundationStereo as an HTTP server.

Returns **raw disparity** (the model's native output). Convert to depth on the
client with `disparity_to_depth(disp, K, baseline)` — keeps the server free of
per-camera intrinsics.

Before a real-model installation, read [`PREREQUISITES.md`](PREREQUISITES.md).
It records everything intentionally outside `install.sh`, including the tested
host baseline, persistent checkpoint path, clean-machine order, and required
verification.

## Layout

```
runtimes/fastfs/
├── pyproject.toml         this runtime's uv project (own .venv, Python 3.12)
├── real/                  exact, locked real-inference dependency project
├── PREREQUISITES.md       external host, storage, and verification contract
├── server.py              BaseServer subclass; @handler-decorated `reconstruct`
├── backend.py             headless adapter around FFS model + InputPadder
├── doctor.py              read-only real dependency and upstream import probe
├── upstream/              git submodule of NVlabs/Fast-FoundationStereo
├── install.sh             exact-syncs real/ into the shared runtime .venv
├── download.sh            fetches checkpoints with an isolated pinned gdown
└── weights/               gitignored; created by download.sh
```

The real project includes only the fixed PyTorch inference path: Torch,
TorchVision, timm, OmegaConf (required by the serialized checkpoint), imageio,
and headless OpenCV. The official full requirements also cover demo UI,
point-cloud, and analysis paths. We intentionally exclude open3d, SciPy,
scikit-image, einops, and xformers because the pinned `pytorch1` forward path
does not import or call them; the real checkpoint E2E test guards that boundary.

## Fake mode (no model, fast)

```bash
uv run robot-tools sync        # light deps only
uv run robot-tools launch      # services.yaml has extra_args: ["--fake"]
```

## Real mode

The abbreviated commands below assume the host and storage prerequisites are
already satisfied. Follow [`PREREQUISITES.md`](PREREQUISITES.md) for a new
machine.

```bash
uv run robot-tools sync          # creates the venv
uv run robot-tools install       # exact locked real environment
uv run robot-tools download      # official GDrive checkpoints
# edit services.yaml → drop --fake
uv run robot-tools launch
```

## Available checkpoints (after download)

| ID | Notes |
|---|---|
| `23-36-37` | Default in our server (largest, most accurate, slowest) |
| `20-30-48` | Mid |
| `20-26-39` | Smaller |
| `15-44-51` | Smallest, fastest |

Pass `--model <id>` to `server.py` (or via `services.yaml` `extra_args:
["--model", "20-30-48"]`) to switch.
