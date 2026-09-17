# AnyPlace placement service

Low-level placement from supplied pick/place masks and RGBD. Uses the official
[AnyPlace](https://github.com/ac-rad/anyplace) multitask diffusion checkpoint,
`NSMTransformerSingleTransformationRegression`, `multistep_regression_scene`,
configuration loading, HEALPix rotation grid, FPS wrappers, and pose utilities.
No AnyGrasp, Molmo, SAM, IsaacLab, robot controller, PyBullet, or dataset is
installed or launched. RGB accompanies the observation; the model consumes XYZ.

## uv deployment

Environment management is exclusively uv. Light and real projects have separate
lockfiles and share `runtimes/anyplace/.venv`. See [prerequisites](PREREQUISITES.md).

From the repository root:

```bash
uv run robot-tools install runtimes/anyplace/services.yaml
uv run robot-tools download runtimes/anyplace/services.yaml
uv run robot-tools doctor runtimes/anyplace/services.yaml
uv run robot-tools launch runtimes/anyplace/services.yaml
```

The supplied profile selects only AnyPlace, binds `127.0.0.1:5560`, and uses GPU 0.
Change its GPU/port for your host. The default artifact location is
`/data/xkyan3/robot-tools-models/anyplace/anyplace_multitask/model.pth`.
Set `ANYPLACE_WEIGHTS_DIR=/your/model/root/anyplace` for both download and launch
to override the model directory. `--checkpoint /absolute/path/model.pth` selects
another compatible official diffusion checkpoint at server startup.

Download retrieves the official checkpoint ZIP at fixed Hugging Face revision
`669f1b0ebcbe2ae3a72970ff31e911e8af73b2d6` and extracts only the multitask model.
`ANYPLACE_CHECKPOINT_ARCHIVE=/path/anyplace_ckpts.zip` reuses a local archive.
Files are written via temporary paths; repeating download is safe. There is no
download at import or launch. No training/evaluation dataset is downloaded.

For fake mode, add `extra_args: ["--fake"]` to a copy of the profile and use
`robot-tools sync` followed by `launch`. Fake mode returns identity transforms;
it needs no weights, GPU, or initialized upstream. A light sync invalidates the
real installation, so rerun `install` before returning to real mode.

## Input and output

Action: `predict_placements`, service ID: `anyplace`, API version: `1`.

Each `MaskedRGBD` observation contains:

| Field | Meaning |
|---|---|
| `rgb` | `(H,W,3)` uint8 RGB, aligned with depth |
| `depth` | `(H,W)` float32/64, optical-axis Z in **metres**, not ray range |
| `mask` | `(H,W)` bool; the client helper accepts binary integer 0/1 or 0/255 |
| `intrinsics` | `(3,3)` zero-skew pinhole matrix for these exact image dimensions |
| `camera_pose` | Optional `(4,4)` camera-to-common-frame rigid transform; identity by default |

Invalid/nonpositive depth is ignored. At least four valid selected pixels and
a non-collinear surface are required. Images must already be rectified; do not
pass distortion parameters or depth in millimetres. Pick/place can be separate
views and resolutions, but their camera poses must map into the **same frame**.
Camera coordinates are X right, Y down, Z forward. Prefer a gravity-aligned
world frame for inputs matching the upstream training/evaluation convention.
No hidden scale normalization or extra translation is applied by the adapter.

`pick.mask` selects the movable object (upstream **child**).
`place.mask` selects the local receiving geometry (upstream **parent**). This
mask supplies the region normally selected around Molmo's location proposal.
For a rack with multiple slots, select the geometry surrounding one desired slot
and call separately for other regions; a full-rack mask does not specify which
slot to use. Include surfaces around an opening, not only missing-depth pixels
inside it. Original low-level sampling and refinement operate on that region.

| Per-call option | Default | Range |
|---|---:|---|
| `num_samples` | 5 | 1–32 initial placement candidates |
| `n_refine_iters` | 50 | 5–100 refinement steps |
| `seed` | 0 | 0–2³²−1; repeatable on the tested host |

The response preserves the original output key and meaning:

```python
relative = response.relative  # (N, 4, 4), float64
p_final = relative[i] @ p_initial          # homogeneous points in common frame
T_place_gripper = relative[i] @ T_pick_gripper
```

These are **object displacement transforms**, not absolute object poses and not
grasp poses. All candidates are returned in upstream order, without invented
confidence scores: the released diffusion evaluation disables the success
classifier and assigns all candidates the same internal score. Collision
checking, IK, reachability and execution remain downstream responsibilities.

## Client example

```python
from robot_tools.services.anyplace import AnyPlaceClient, masked_rgbd

pick = masked_rgbd(pick_rgb, pick_depth_m, pick_mask, pick_K,
                   camera_pose=T_world_pick_camera)
place = masked_rgbd(place_rgb, place_depth_m, place_mask, place_K,
                    camera_pose=T_world_place_camera)
with AnyPlaceClient(host="127.0.0.1", port=5560, timeout_ms=120_000) as client:
    result = client.predict_placements(pick, place, num_samples=5, seed=0)
print(result.relative.shape)
```

For files, use [the runnable example](../../examples/anyplace_placement.py):

```bash
uv run python examples/anyplace_placement.py --pick data/pick.npz --place data/place.npz
```

Each input NPZ has `rgb`, `depth`, `mask`, `intrinsics`, and optional
`camera_pose`; output NPZ has `relative`. Loading needs no pickle.

## Upstream compatibility

Pinned upstream: `3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e` (MIT).
The install script applies [a reviewed patch](compat/placement-only.patch)
idempotently; it fails if the patch no longer matches. The network architecture,
learned parameters, diffusion/noise schedule, and relative-transform assembly
are retained. All checkpoint parameters load strictly.

The upstream README targets Python 3.8 / Torch 1.13 / CUDA 11.7. This runtime
uses Python 3.10 and Torch 2.10 / CUDA 12.8 for RTX 5090 support. Changes are:

1. Use standard Python logging instead of importing `airobot` for two log calls.
2. Honor headless visualization and omit unused success-classifier preparation.
3. Use [batched Torch FPS](anyplace_fps.py) through upstream FPS wrappers instead
   of legacy `torch-cluster`, `torch-scatter`, and `KNN_CUDA` binaries. Sampling
   starts at index zero and uses the same farthest-first algorithm. Ties can
   choose different points; sparse clouds repeat samples to fill 1024 points.
   Bitwise equivalence to the old CUDA extension is not claimed.
4. Preserve the batch dimension for a single candidate. Initialize exactly
   planar receiving regions at their centroid because their OBB has no volume.

Meshcat's Python types remain an upstream import dependency; no viewer/server is
created. Unused KNN/interpolation imports are lazy and never entered by this
placement path. No source package installation runs the training-oriented
upstream `setup.py`.

## Acceptance

On the tested RTX 5090, the official multitask model loaded and produced finite
rigid `(5,4,4)` float64 transforms from synthetic masked RGBD, using 50 refinement
steps in about 5.5 seconds (peak Torch allocation about 379 MiB, excluding CUDA
context/other processes). A one-candidate, five-step call also passed.
The selected launcher install/doctor/launch flow and real HTTP client were
verified on 2026-09-16: five candidates / 50 steps took about 5.5 seconds over
HTTP; an exactly planar receiving region / one candidate / five steps passed,
including agreement of repeated seeded calls. The full SDK test suite passed
232 tests; lint, formatting, shell syntax and uv lock checks passed.

These are integration checks, not a placement-success benchmark. Real camera
noise, object geometry, choice of receiving region, and robot execution need
task-specific evaluation. Fake HTTP tests verify wire shape/dtype, service
identity, option validation and input rejection without model dependencies.

Reproduce the real test after launching the service:

```bash
uv run --project runtimes/anyplace --no-sync python runtimes/anyplace/smoke.py \
  --host 127.0.0.1 --num-samples 5 --n-refine-iters 50
uv run --project runtimes/anyplace --no-sync python runtimes/anyplace/smoke.py \
  --host 127.0.0.1 --planar --num-samples 1 --n-refine-iters 5 --repeat
```
