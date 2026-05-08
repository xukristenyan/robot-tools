# fastfs-server

Independent uv project that wraps NVlabs/Fast-FoundationStereo as a ZMQ server.

Returns **raw disparity** (the model's native output). Convert to depth on the
client with `disparity_to_depth(disp, K, baseline)` — keeps the server free of
per-camera intrinsics.

## Layout

```
servers/fastfs/
├── pyproject.toml         this server's uv project (own .venv, Python 3.12)
├── server.py              BaseServer subclass; @handler-decorated `reconstruct`
├── pipeline.py            headless wrapper around FFS model + InputPadder
├── upstream/              git submodule of NVlabs/Fast-FoundationStereo
├── setup.sh               installs torch+xformers (cu124) + FFS requirements + gdown
├── download.sh            fetches checkpoints from Google Drive
└── weights/               gitignored; created by download.sh
```

## Mock mode (no model, fast)

```bash
uv run robot-tools sync   services.yaml      # light deps only
uv run robot-tools launch services.yaml      # services.yaml has extra_args: ["--mock"]
```

## Real mode

```bash
uv run robot-tools sync     services.yaml          # creates the venv
uv run robot-tools setup    fastfs                 # torch + xformers + requirements.txt
uv run robot-tools download fastfs                 # gdown the GDrive folder
# edit services.yaml → drop --mock
uv run robot-tools launch   services.yaml
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
