# cuRobo deployment prerequisites

## v0.9.0 upgrade on jarjar (2026-11-09)

Current upstream is cuRoboV2-Fast **v0.9.0**, commit
`42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83`. The runtime uses GPU BiRRT,
the upstream default cubic B-spline optimizer, fixed-path TimeOpt and CUDA graphs.
Torch remains 2.10.0+cu128 and Warp remains 1.13.0 in the existing Python 3.12.13
environment. Dependency consistency and real imports passed.

The isolated GPU service passed the complete smoke test, including full-pose
and position-only Cartesian waypoints, six chained stages, unreachable IK,
collision rejection, contact-aware planning, attachments, release and scene/mesh
isolation. Native versus adapted collision costs and gradients also passed
for static/swept queries, summed costs and robot layer masks.

Evidence: `/home/xkyan3/projects/beyond-pick/outputs/curobo-v090-upgrade-20261109/`.
These are service acceptance checks, not a task-success or speedup benchmark.
Startup compilation and wrapper checks are included in some first-call timings.

## Previous verified baseline (historical)

| Component | Value |
|---|---|
| Acceptance date | 2026-10-16 |
| GPU | NVIDIA RTX 5090, device 1, 32 GB |
| Driver | 580.178.04 |
| Python | uv-managed CPython 3.12.13 |
| Torch | 2.10.0+cu128 |
| Warp | 1.13.0 |
| cuda-core | 1.2.2; CUDA 12 dependencies in `real/uv.lock` |
| cuRobo | V2 / 0.8.0.post1.dev43 |
| Upstream | `78fd485fa82d9b9a063fb4985e371814587e666a` |
| Robot | official `franka.yml`, seven active joints, tool `panda_hand` |

Use a compatible NVIDIA GPU/driver and Linux x86_64. The official V2
[installation guide](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/docs/getting-started/installation.rst)
requires driver 580.65.06 or newer. The lock uses CUDA 12 wheels and the
default runtime-compiled `cuda.core` backend. No system CUDA extension build,
simulator, display server, account token or gated model download is needed.
Git, uv and access to GitHub/PyPI/the PyTorch wheel index are needed to install.

## Storage and installation

```bash
export ROBOT_TOOLS_MODELS_ROOT=/data/xkyan3/robot-tools-models
export CUROBO_DATA_DIR="$ROBOT_TOOLS_MODELS_ROOT/curobo"
export UV_CACHE_DIR="$CUROBO_DATA_DIR/cache/uv"

uv run robot-tools install runtimes/curobo/services.yaml
uv run robot-tools download runtimes/curobo/services.yaml
uv run robot-tools doctor runtimes/curobo/services.yaml
uv run robot-tools launch runtimes/curobo/services.yaml
```

The standard runtime `.venv` is a symlink to `$CUROBO_DATA_DIR/venv`.
An existing light `.venv` is moved there by `install.sh` once; its conventional
path remains the same. Torch/CUDA wheels, uv cache, Warp/CUDA compilation cache,
uploaded meshes and validation logs all live below the data directory. The
initial installation occupied about 8 GB with uv hardlinks; allow at least
15 GB plus uploaded geometry. The pinned source submodule remains in the
checkout as in the other services. There are no model checkpoint weights.

`CUROBO_DATA_DIR` defaults to `/data/$USER/robot-tools-models/curobo`.
Nonstandard hosts can set it to another writable absolute directory. Persist
the same path variables in the launcher environment. Explicit standard cache
overrides (`UV_CACHE_DIR`, `WARP_CACHE_PATH`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`,
`TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`) take precedence.

The light and real environments share `.venv`, matching the repository
contract. `sync` restores light dependencies and clears the real marker;
`install` checks the lock, package graph, real module imports and CUDA device
before writing the marker. `doctor` does not plan, download or compile kernels.
Server readiness is published only after IK, collision and motion warmup.

## Acceptance evidence

`bash runtimes/curobo/validate.sh` passed on the baseline above. It creates a
temporary service on port 15562 and stops it even on failure. Set
`CUDA_VISIBLE_DEVICES` and `CUROBO_TEST_PORT` to choose another device/port.
Logs are under `$CUROBO_DATA_DIR/logs/acceptance-server.log`.

The recorded smoke verified reachable and unreachable IK, world collision
rejection, explicit collision-disabled IK, six chained motion stages,
obstacle/link-specific contact exemptions, attached-body collision, release
pose changes, and isolation between scenes and meshes with reused names.
An empty-world six-stage motion returned float32 joint arrays with shapes
`(41,7), (21,7), (21,7), (21,7), (21,7), (21,7)` and 0.025 s sample spacing.
One warm run measured approximately 5.0 ms for IK and 399 ms for all six stages,
including final sampled collision checks. These are one smoke case's server
timings, not a benchmark or a task success-rate estimate.

Changing the upstream commit, GPU, driver, robot YAML, sphere geometry,
gripper configuration, collision margin, or real lock requires revalidation.
The acceptance uses synthetic scenes; calibration to PyRep/GraspGenX frames,
reconstructed geometry quality, grasp execution and real robot performance
remain downstream responsibilities.
