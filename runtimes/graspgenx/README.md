# GraspGenX runtime

Independent robot-tools HTTP service for
[NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX). It follows the
official GraspGenX README, demos, and `graspgenx.serving` semantics while using
robot-tools FastAPI/MessagePack transport, readiness checks, bounded admission,
and typed synchronous clients.

GraspGenX is not a renamed GraspGen server. It loads one gripper-independent
model and conditions inference on a named gripper resolved from the official
assets.

Before a real-model installation, read [`PREREQUISITES.md`](PREREQUISITES.md).
It records everything intentionally outside `install.sh`, including Git LFS,
persistent checkpoint/gripper paths, the tested host baseline, clean-machine
order, and required verification.

## Native API

GraspGenX service API version 4 uses named-gripper object and scene inputs.
Use `GraspGenXClient`; `GraspGenClient` only connects to the original GraspGen
service.

| HTTP action | Input |
|---|---|
| `get_metadata` | No inference payload |
| `generate_grasps` | Segmented object point cloud `(N,3)` |
| `generate_safe_grasps` | Object point cloud `(N,3)` plus surrounding scene point cloud `(M,3)` |
| `generate_grasps_for_all` | Metric depth + intrinsics + integer instance mask |
| `generate_safe_grasps_for_all` | The same scene-depth input, with collision filtering |

`graspmoe` is the native default planner: diffusion candidates are combined
with OBB-generated candidates and all are scored by the discriminator.
`diffusion` selects only the learned diffusion branch. Native sweep-volume
responses include a `diff` or `obb` tag for every grasp.

For planner-based actions, the server returns all candidates. The synchronous
client applies `grasp_threshold` and `topk_num_grasps` locally, matching the
official GraspGenX client. This preserves raw results and avoids confusing
upstream's per-retry internal top-k with a final global top-k.

## Client examples

Named gripper from official `gripper_descriptions` assets:

```python
from robot_tools.services.graspgenx import GraspGenXClient

with GraspGenXClient(host="127.0.0.1", port=5559) as client:
    result = client.generate_grasps(
        object_xyz,
        gripper_name="robotiq_2f_85",
        num_grasps=200,
        grasp_threshold=0.7,
        topk_num_grasps=100,
    )
    print(result.grasps.shape, result.confidences.shape)
```

Collision-filtered grasps for one selected object:

```python
from robot_tools.services.graspgenx import GraspGenXClient

with GraspGenXClient(host="127.0.0.1", port=5559) as client:
    result = client.generate_safe_grasps(
        object_xyz,              # selected object, camera frame, meters
        scene_xyz_without_object,  # surrounding collision scene, same frame
        gripper_name="robotiq_2f_85",
        planner="graspmoe",
        collision_threshold=0.005,
        grasp_threshold=0.7,
        topk_num_grasps=100,
    )
    print(result.grasps.shape, result.confidences.shape)
```

Scene modes return MessagePack-safe parallel lists. Convert them to an
instance-id mapping when useful:

```python
result = client.generate_grasps_for_all(
    depth_meters,
    intrinsics_3x3,
    instance_mask,
    gripper_name="robotiq_2f_85",
)
by_instance = client.scene_results_by_id(result)
```

Depth must be floating point in meters. Instance mask value `0` means
background/ignore. All inputs and returned poses stay in the input camera or
point-cloud frame. `generate_safe_grasps` expects the caller to exclude the
target object from `scene_xyz_without_object`; the server caps that collision
scene at 8192 points and checks the named gripper collision mesh. This is a
static gripper-versus-scene clearance test, not arm, path, or robot motion
planning.

## Layout

```text
runtimes/graspgenx/
├── pyproject.toml         lightweight fake/server environment
├── real/                  exact real-model pyproject + uv.lock
├── PREREQUISITES.md       external host, assets, storage, and verification contract
├── server.py              robot-tools handlers and deterministic fake mode
├── backend.py            official shared-model/sampler/planner integration
├── doctor.py             read-only real imports plus compat-patch checks
├── upstream/              pinned NVlabs/GraspGenX submodule
│   └── ext/               downloaded checkpoints and gripper assets
├── compat/                no-auto-download and lazy-ZMQ source patches
├── install.sh             one locked real sync plus import/CUDA probes
└── download.sh            Hugging Face checkpoint/assets clone via Git LFS
```

The released model uses the pure-PyTorch `ptv3_vanilla` backbone. Unlike the
original GraspGen environment, this service does not build PointNet++, SpConv,
or torch-scatter. The pinned source is imported directly from `upstream/` and
the separate real lock selects Torch 2.10/cu128 plus only the imports used by
the HTTP inference paths. Upstream training, TensorBoard, Viser,
scene-synthesizer, PyG, ZMQ, and end-to-end simulation packages stay outside
the service environment.

Two small source patches remove packaging side effects rather than changing
model math: importing `graspgenx` no longer clones Hugging Face repositories,
and importing `graspgenx.serving.types` no longer imports the optional ZMQ
transport. The upstream ZMQ exports remain lazily available to users who
explicitly install and request them; robot-tools uses HTTP + MessagePack only.

The backend also applies the official headless runtime defaults
`PYOPENGL_PLATFORM=egl` and `PYGLET_HEADLESS=true` before importing GraspGenX.
Explicit environment values are preserved.

## Fake mode

```bash
uv run robot-tools sync
uv run robot-tools launch
```

Or directly:

```bash
cd runtimes/graspgenx
uv sync
uv run python server.py --port 5559 --fake \
  --default-gripper robotiq_2f_85
```

Fake mode imports no Torch or upstream package and exercises all five native
HTTP actions with deterministic arrays.

## Real mode

The abbreviated commands below assume the Git LFS and storage prerequisites
are already satisfied. Follow [`PREREQUISITES.md`](PREREQUISITES.md) for a new
machine.

```bash
git submodule update --init runtimes/graspgenx/upstream
uv run robot-tools sync
uv run robot-tools install

# Recommended on jarjar: keep model data off the quota-limited home filesystem.
export GRASPGENX_CHECKPOINT_DIR=/data/xkyan3/robot-tools-models/graspgenx/checkpoints
export GRASPGENX_GRIPPER_CFG_DIR=/data/xkyan3/robot-tools-models/graspgenx/gripper_descriptions
uv run robot-tools download
# remove --fake from this service in services.yaml
uv run robot-tools launch
```

Useful startup arguments:

- `--config /path/to/release` — checkpoint root containing `gen/` and `dis/`;
  omitted uses the official current release.
- `--assets-dir /path/to/assets` — fallback root containing `x_grippers/` and
  `proc_grippers/`; official `gripper_descriptions` is resolved first.
- `--default-gripper robotiq_2f_85` — optional preload and fallback for
  name-based `infer` requests. Other named grippers remain request-selectable
  and are cached lazily.
- `--tensorrt --tensorrt-precision fp32|fp16` — opt-in upstream acceleration;
  not part of the currently tested jarjar baseline.

Weights and real-gripper assets default to `upstream/ext/`. Override them with
`GRASPGENX_CHECKPOINT_DIR` and `GRASPGENX_GRIPPER_CFG_DIR`, keeping the same
environment variables exported when launching. Assets are downloaded only by
the explicit `robot-tools download` command; a launch with missing assets fails
instead of starting an unexpected multi-gigabyte clone.

The tested jarjar storage convention is:

```text
/data/xkyan3/robot-tools-models/
├── graspgen/
├── graspgenx/
│   ├── checkpoints/
│   └── gripper_descriptions/
├── fastfs/
└── sam3/
```

Only `franka_panda` and `robotiq_2f_85` assets are required by the current
deployment. Their tracked LFS files can be repaired without fetching every
supported gripper:

```bash
git -C "$GRASPGENX_GRIPPER_CFG_DIR" lfs pull \
  --include="gripper_descriptions/assets/x_grippers/franka_panda/**,gripper_descriptions/assets/x_grippers/robotiq_2f_85/**" \
  --exclude=""
```

## RTX 5090 verification

Verified on jarjar with two RTX 5090 GPUs, driver 580.173.02, Python 3.10.20,
Torch 2.10.0+cu128, and the official `release` checkpoints. The locked
inference environment contains 101 resolved packages and occupies about 7.2
GiB (most of that is PyTorch/CUDA). The temporary test server used GPU 1 and
was stopped afterwards.

| Path | Official sample/result | Server inference |
|---|---|---:|
| named `robotiq_2f_85` | 30 grasps, confidence 0.725–0.908 | 1.160 s cold/warm-up |
| named `franka_panda` | 30 grasps, confidence 0.656–0.912 | 0.895 s |
| raw sweep + GraspMoE | client final top-10: 7 `obb`, 3 `diff` | 0.217 s |
| scene depth, IDs 101/102 | 5 grasps per instance, no skips | 0.270 s |
| organized scene PC, IDs 101/102 | 5 grasps per instance, no skips | 0.276 s |

During a 200-candidate inference, `/health` responded in about 0.93 ms and GPU
memory peaked at about 1,769 MiB. Admission behaved as configured: one running,
one waiting, and the third request returned typed `ServiceBusyError` in about
1.66 ms. A 100 ms client timeout returned `InferenceTimeoutError` without
cancelling the running GPU work or releasing its capacity early; the same
client remained reusable.

## Adding a new named gripper

The official wizard creates the URDF/mesh/config bundle, including open and
half-open sweep volumes:

```bash
cd runtimes/graspgenx
uv run --no-sync python upstream/scripts/gripper_config_wizard.py \
  --urdf /path/to/gripper.urdf \
  --name my_gripper \
  --port 8081
```

For inference-only integrations that already know the two sweep boxes, use
`SweepVolumeParams` instead; it needs no named server asset.
