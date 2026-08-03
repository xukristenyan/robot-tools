# GPU runtime installation contract

`install.sh` is only the model environment installer. A reproducible deployment
also needs the correct robot-tools revision, selected pinned upstream submodules, system
tools, model artifacts, credentials, storage environment, and post-install
verification. The service-specific requirements are documented next to each
runtime:

- [`sam3/PREREQUISITES.md`](sam3/PREREQUISITES.md)
- [`fastfs/PREREQUISITES.md`](fastfs/PREREQUISITES.md)
- [`graspgen/PREREQUISITES.md`](graspgen/PREREQUISITES.md)
- [`graspgenx/PREREQUISITES.md`](graspgenx/PREREQUISITES.md)

## Verified jarjar baseline

| Component | Verified value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, x86_64, kernel 7.0.0-28-generic |
| GPU | NVIDIA GeForce RTX 5090 |
| NVIDIA driver | 580.173.02 |
| uv | 0.11.21 |
| Git | 2.43.0 |
| Git LFS | 3.5.1 |
| SAM3 / FastFS Python | 3.12.13, uv-managed |
| GraspGen / GraspGenX Python | 3.10.20, uv-managed |
| GPU wheels | Torch 2.10.0 + CUDA 12.8 |

An equal or newer compatible driver may work, but the table above is the
tested baseline. A different GPU architecture, OS, upstream revision, Python
minor version, or package index state is a new installation target and must be
revalidated.

## Reproducible order of operations

From an exact, committed robot-tools revision:

```bash
uv sync --locked
uv run robot-tools sync
uv run robot-tools install
uv run robot-tools download
uv run robot-tools doctor
uv run robot-tools launch
```

`services.yaml` is the selection boundary. `install` initializes only those
services' upstream submodules. `sync` creates exact locked environments and may
remove undeclared packages installed by `install.sh`; `launch` uses
`uv run --no-sync` and never changes an existing environment.

If `sync` is run after a real installation, rerun `install services.yaml`
before real launch. `sync` clears the runtime's real-install marker; `install`
records the current `real/uv.lock` only after its dependency and import probes
pass. `doctor` rejects a real profile when that marker is missing or stale, when
`uv pip check` fails, or when the runtime-specific `doctor.py` imports fail.

## Model storage convention

On machines with a large `/data` filesystem:

```bash
export ROBOT_TOOLS_MODELS_ROOT="/data/${USER}/robot-tools-models"
mkdir -p "$ROBOT_TOOLS_MODELS_ROOT"/{sam3,fastfs,graspgen,graspgenx}
```

`ROBOT_TOOLS_MODELS_ROOT` is a documentation convention, not a variable read
directly by the Python package. Each service maps it to its own supported
environment variables as described in its prerequisites file. Export the same
variables both when downloading and when launching.

For launches after logout/reboot, persist these non-secret path variables in
the shell profile, scheduler job, or service-manager environment that starts
robot-tools. The launcher inherits its environment; it does not remember
variables from an earlier interactive shell. Keep access tokens out of Git.

## What “reproducible” means here

The current install scripts have been run successfully on jarjar with real
models. This gives a tested clean-machine path, not an unconditional guarantee:

- the robot-tools changes and submodule pointers must be committed and
  available to the other machine;
- PyPI, PyTorch, Hugging Face, Google Drive, and Git endpoints must remain
  reachable, unless artifacts are mirrored internally;
- credentials and gated-model approval are machine/user specific;
- large checkpoint files must be verified after transfer;
- some upstream transitive dependencies are not represented by a fully
  offline lock, so a future registry change can still require refreshing the
  tested constraints.

For stronger, offline-grade reproducibility, archive the four completed venv
package manifests, mirror all wheels/checkpoints, record artifact SHA256 sums,
and install from that immutable artifact store.
