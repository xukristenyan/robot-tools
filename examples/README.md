# robot-tools usage examples

Minimal scripts showing how a downstream project consumes each client, plus
one end-to-end pipeline that chains all three.

All examples assume the **real** servers are already running — i.e. the user
has done `setup` + `download` for each server and removed `--mock` from
`services.yaml`.

```bash
# In the robot-tools repo:
robot-tools launch services.yaml
```

## Service config

`examples/services.yaml` is a copy of the repo-root `services.example.yaml`,
shipped here so the demo scripts have ports to read without you setting up
anything. Every script calls `load_service("<name>")` from `_config.py`,
which reads this file. Edit it to change ports/hosts and the scripts follow.

This mirrors how a real downstream project works: it has its own
`services.yaml`, copied from `services.example.yaml` and tuned per machine.
The launcher (`robot-tools launch services.yaml`) reads the same format,
so config and clients stay in sync.

## Install in your downstream project

```bash
uv add --editable /path/to/robot-tools
uv add pillow                 # used by the image-loading examples
```

## Per-module parameters

Tuning knobs (camera intrinsics, score thresholds, num_grasps, …) live in
`examples/configs/<name>.yaml` instead of CLI args. Edit those files to
re-tune; the scripts call `load_params("<name>")` to read them.

```
examples/configs/
├── fastfs.yaml      camera intrinsics + valid_iters / max_disp
├── sam3.yaml        score_threshold + multimask_output
└── graspgen.yaml    grasp_threshold, num_grasps, topk,
                     collision_threshold, max_scene_points
```

CLI args are reserved for **per-run inputs** — image paths, point cloud
paths, the text prompt, click coords. Everything else is config.

## Run

```bash
# 1. Stereo reconstruction → disparity (+ depth via configs/fastfs.yaml)
uv run python examples/fastfs_stereo.py \
    --left data/left.png --right data/right.png

# 2. Open-vocab + click segmentation
uv run python examples/sam3_segment.py \
    --image data/scene.png --text "the red mug" --click 640 480

# 3. Grasp generation from an object point cloud (+ optional scene PC)
uv run python examples/graspgen_basic.py \
    --object data/object_pc.npy --scene data/scene_pc.npy

# 4. End-to-end: stereo → SAM3 mask → backproject → GraspGen
uv run python examples/pipeline_stereo_to_grasp.py \
    --left data/left.png --right data/right.png \
    --target "the red mug"
```

## Data conventions

- Stereo images: rectified pair, same resolution, RGB or grayscale.
- Point clouds: `.npy` of shape `(N, 3)` float32, in **camera frame meters**.
- Camera intrinsics: `fx/fy` in pixels, `cx/cy` in pixels, `baseline` in meters.
