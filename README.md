# robot-tools

Shared client SDK + server launcher for vision/manipulation model services.

Each model (GraspGen, Fast-FoundationStereo, SAM 3, …) runs as an independent
ZMQ server in its own `uv` venv. Downstream projects pip-install this SDK and
pick which clients they need via a YAML profile.

## Prerequisites

- **uv** (manages all the venvs). Install once, system-wide:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc          # or restart your shell
  ```
- **NVIDIA GPU + driver** for any server that needs CUDA (everything we ship
  so far does).
- **git-lfs** for fetching some weights:
  ```bash
  sudo apt install git-lfs && git lfs install
  ```

## Layout

```
robot-tools/                  ← this repo (Python package inside is robot_tools)
├── pyproject.toml            ← root SDK package (light client deps only)
├── NOTES.md                  ← how to add a new server (read this when adding one)
├── services.example.yaml     ← copy into your project as `services.yaml`
├── src/robot_tools/          ← Python package: contracts/clients/server_base/launcher
└── servers/                  ← one independent uv subproject per model
    ├── graspgen/             ← NVlabs/GraspGen     (Python 3.10, torch 2.1)
    ├── fastfs/               ← NVlabs/Fast-FoundationStereo (Python 3.12, torch 2.6)
    └── sam3/                 ← facebookresearch/sam3        (Python 3.12, torch 2.7)
```

## Quickstart

Clone with submodules so you get the upstream model repos:
```bash
git clone --recurse-submodules <robot-tools url>
cd robot-tools
```

Then in your downstream project:
```bash
cd ~/path/to/your-project
uv add --editable ~/CodeLib/robot-tools     # install the client SDK

# Copy the service profile and edit which servers you want
cp ~/CodeLib/robot-tools/services.example.yaml services.yaml

uv run robot-tools list                     # see registered servers
uv run robot-tools sync   services.yaml     # build chosen servers' venvs
uv run robot-tools launch services.yaml     # start them (mock mode by default)
```

Real-model bring-up is per-server (each pulls weights / has its own setup
script). See:
- `servers/graspgen/README.md`
- `servers/fastfs/README.md`
- `servers/sam3/README.md`
- `~/CodeLib/projects/test-proj-a/` for a working reference consumer.

## Adding a new server

See [NOTES.md](NOTES.md) — step-by-step recipe with code templates and the
known pitfalls (~10 min per new server).

## Status

| Server | Status | Notes |
|---|---|---|
| `graspgen` | ✅ working (real) | franka / robotiq / suction grippers |
| `fastfs` | ✅ working (real) | stereo → disparity; client converts to depth |
| `sam3` | ✅ working (real) | text & point prompt segmentation; gated weights |
