# Usage examples

These scripts show the downstream client API for every service and one
multi-service pipeline:

| Script | Flow |
|---|---|
| `curobo_motion.py` | Robot-base pose targets → IK/motion planning and sampled trajectory collision checks |
| `depth_anything_v2_image.py` | Single RGB image → raw relative inverse depth (arbitrary units) |
| `anyplace_placement.py` | Pick/place mask + RGBD + camera calibration → native relative placement transforms |
| `fastfs_stereo.py` | Stereo images → disparity → metric depth |
| `sam3_segment.py` | Image + text or click → masks |
| `graspgen_basic.py` | Object/scene point clouds → grasp poses |
| `graspgenx_native.py` | Object point cloud + named gripper → grasp poses, optionally filtered against a surrounding scene |
| `pipeline_stereo_to_grasp.py` | FastFS → SAM3 → backprojection → GraspGen |

## Before running an example

From the repository root, start the services you want to exercise. The example
launch profile uses fake mode by default:

```bash
uv run robot-tools launch examples/services.yaml
```

In another terminal, verify the addresses consumed by the scripts:

```bash
uv run robot-tools status examples/endpoints.yaml
```

Edit `examples/endpoints.yaml` when a service runs on another machine. Edit
`examples/services.yaml` only when this checkout is responsible for launching
the runtimes.

> [!NOTE]
> Fake mode is useful for checking each client's transport and contract. The
> stereo-to-grasp pipeline needs meaningful model outputs, so use real
> runtimes for an end-to-end result.

The image examples use Pillow without adding it to the lightweight SDK. Their
commands provide it as a temporary uv dependency.

## Run

```bash
# cuRobo; six illustrative stages in one request. No robot execution.
uv run python examples/curobo_motion.py --six-stage

# Depth Anything V2 Large; saves raw float32 values, not metric depth.
uv run --with pillow python examples/depth_anything_v2_image.py \
  --image data/scene.png --out relative_inverse_depth.npy

# AnyPlace; each NPZ contains rgb, depth in metres, mask, intrinsics,
# and optionally camera_pose (camera-to-common-frame).
uv run python examples/anyplace_placement.py \
  --pick data/pick.npz --place data/place.npz

# Stereo reconstruction; saves depth.npy by default.
uv run --with pillow python examples/fastfs_stereo.py \
  --left data/left.png --right data/right.png

# Text segmentation and optional point-prompt segmentation.
uv run --with pillow python examples/sam3_segment.py \
  --image data/scene.png --text "the red mug" --click 640 480

# GraspGen; add --scene for collision filtering.
uv run python examples/graspgen_basic.py \
  --object data/object_pc.npy --scene data/scene_pc.npy

# GraspGenX named-gripper inference; add --scene for collision filtering.
uv run python examples/graspgenx_native.py \
  --object data/object_pc.npy --gripper robotiq_2f_85 \
  --scene data/scene_pc_without_object.npy

# Full stereo → segmentation → collision-free grasp pipeline.
uv run --with pillow python examples/pipeline_stereo_to_grasp.py \
  --left data/left.png --right data/right.png \
  --target "the red mug"
```

Use each script's `--help` output for its input and output arguments.

## Configuration

The examples keep three different kinds of values separate:

| Location | Owns |
|---|---|
| `examples/services.yaml` | Runtime selection, ports, GPUs, fake/real mode, server startup arguments |
| `examples/endpoints.yaml` | Client-side host and port for each service |
| `examples/configs/<service>.yaml` | Per-call model and camera parameters used by these scripts |

Server startup choices such as a default gripper live in
`services.yaml`. Values that can change for each request, such as thresholds
and result counts, live in the per-service example config and are passed as
client method arguments.

## Data conventions

- Images are rectified RGB or grayscale arrays; a stereo pair must have the
  same resolution.
- Point clouds are `.npy` arrays with shape `(N, 3)`, `float32`, in camera
  frame meters.
- The GraspGenX `--scene` point cloud must use the same frame as `--object`
  and must already exclude the target object's points. The server downsamples
  it when needed; it does not remove the target object for this action.
- Camera intrinsics use pixel units for `fx`, `fy`, `cx`, and `cy`;
  stereo baseline is in meters.
- Grasp poses are `(K, 4, 4)` transforms in the coordinate frame documented
  by the selected service.

## Point-SAM colored point cloud

Save `xyz` (N,3 float metres), `rgb` (N,3 uint8 RGB), `prompt_points` (Q,3,
same coordinate frame) and `prompt_labels` (Q, 1 foreground / 0 background)
in an NPZ. Optional `prompt_mask` is a previous (N,) logits row.

```bash
uv run --no-sync python examples/pointsam_segment.py input.npz \
  --endpoints examples/endpoints.yaml --config examples/configs/pointsam.yaml \
  --output outputs/pointsam/masks.npz
```

The example encodes once, segments and releases the cache in `finally`.
Output masks and logits retain the input point order. See the
[Point-SAM runtime](../runtimes/pointsam/README.md) for iterative refinement.
