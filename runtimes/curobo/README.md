# cuRobo service

Use a lightweight `CuroboClient` to solve IK, check an existing joint path, or
plan an entire pick/place sequence in one HTTP call. The GPU runtime stays warm
and owns cuRobo, Torch, CUDA, collision geometry and planner caches. It never
executes trajectories or controls a gripper.

The service reuses unchanged observed scenes and contact partitions across IK,
batch IK, motion planning and trajectory checks. It compares actual obstacle
poses, geometry IDs and contact permissions. Attachment spheres are compared and
updated separately for each kinematics consumer. Every request prepares its full
context before computation, including detaching a previously carried object.
Changed scenes still use the original update path; collision decisions are never
cached. Failed partial scene updates invalidate the reusable state.

Uploaded mesh arrays use a bounded in-memory LRU cache keyed by immutable mesh IDs
(the existing `SceneStore.capacity`, 32 by default). GPU geometry capacity is
unchanged. Request-end reset clears the active context; physical geometry updates
are deferred to the next prepare. Solver budgets, tolerances and collision checks
remain unchanged. Restart or replace a running worker to load these code changes.

## Run

The checked-in `services.yaml` selects only cuRobo, loopback port **5562**, GPU 1,
and the official Franka Panda configuration. Change `gpu` for another host.

```bash
uv run robot-tools install runtimes/curobo/services.yaml
uv run robot-tools download runtimes/curobo/services.yaml
uv run robot-tools doctor runtimes/curobo/services.yaml
uv run robot-tools launch runtimes/curobo/services.yaml
```

In another terminal: `uv run robot-tools status runtimes/curobo/endpoints.yaml`.

`install` initializes only the selected pinned submodule and syncs
`real/uv.lock`. The conventional `runtimes/curobo/.venv` points into
`/data/$USER/robot-tools-models/curobo/venv`. Assets and caches live alongside it.
Override with `CUROBO_DATA_DIR`; see [prerequisites](PREREQUISITES.md).
There are no learned weights to download. The upstream includes robot assets.

For a CPU-only wiring check, create a profile with `name: curobo` and
`extra_args: ["--fake"]`, then use the normal `sync` and `launch` commands.
Fake results explicitly carry `fake=true` and contain synthetic trajectories.
As for other services, `sync` replaces the real environment with its light
environment; run `install` again before real use.

## Downstream usage

Install only the root SDK in the downstream environment:

```bash
uv add --editable /path/to/robot-tools
```

```yaml
# endpoints.yaml
endpoints:
  curobo: {host: 127.0.0.1, port: 5562}
```

```python
import numpy as np
from robot_tools.core.endpoints import get_endpoint
from robot_tools.services.curobo import CuroboClient, Cuboid, MotionStage, World

with CuroboClient.from_endpoint(get_endpoint("curobo", "endpoints.yaml"), timeout_ms=120_000) as client:
    robot = client.get_robot_info()
    world = World(obstacles=[Cuboid(
        name="table", dims=(1.5, 1.5, 0.1),
        pose=np.array([0.5, 0, -0.1, 1, 0, 0, 0], dtype=np.float32),
    )])
    registration = client.put_scene("episode-42", world)
    assert registration.failure_code is None
    scene = registration.scene
    goal = np.array([0.45, 0, 0.45, 0, 1, 0, 0], dtype=np.float32)
    # Use measured current joints in an application. default_position is a demo.
    ik = client.solve_ik(goal, robot.default_position, scene=scene, joint_names=robot.joint_names)
    plan = client.plan_motion(
        robot.default_position, [MotionStage(name="approach", goal_pose=goal)],
        scene=scene, joint_names=robot.joint_names,
    )
    if plan.success:
        check = client.check_trajectory(plan.stages[0].positions, scene=scene, joint_names=robot.joint_names)
        print(check.valid, check.first_invalid_index)
    else:
        print(plan.failed_stage, plan.failure_code)
```

The runnable [example](../../examples/curobo_motion.py) also supports
`--six-stage`. Its poses demonstrate the API; they are not a measured grasp task.

## Semantics

| Action | Result and limits |
|---|---|
| `get_robot_info` | Active joint names/order, limits, default joints, base/tool/attachment frames, tolerances, margin, fake flag and upstream commit |
| `put_mesh` | Upload `(V,3)` vertices and integer `(F,3)` triangles once; returns an opaque mesh ID |
| `put_scene` | Register a complete immutable world; returns `(scene_id, version)` |
| `solve_ik` | Up to 256 independent goals per HTTP request; one solution and success flag per goal; no path guarantee |
| `check_trajectory` | Densifies every joint edge, checks world/self collisions and joint limits; returns aligned masks and first invalid sample |
| `plan_motion` | Up to 16 sequential pose stages; returns timed joint trajectories or the first failed stage |

All poses are **`[x,y,z,qw,qx,qy,qz]`**, metres, unit quaternions, in the
robot base frame. For the default Panda they target **`panda_hand`** relative to
**`panda_link0`**. Convert camera/world coordinates and GraspGenX/PyRep TCP
frames in downstream code; these frames are not interchangeable. Joint vectors
use radians and the explicit `joint_names` order. The service accepts reordered
names and returns that same order. The default has seven active arm joints and
locks both finger joints at 0.04 m, as specified by the official robot YAML.
Match the configured robot/gripper geometry to the robot being checked.

`collision_aware` uses official `MotionPlanner.plan_pose`, then independently
checks the returned path. `ik_only` solves each stage endpoint and joins it to
the previous solution with linear joint interpolation. This mode explicitly
skips world/self collision checks and returns `dt=None`; it is an untimed
reachability/ablation path, not equivalent to PyRep's path algorithm. It still
enforces joint limits. There is **no collision fallback** in collision-aware mode.

Default validation spacing is 0.02 rad per joint, including both endpoints and
all original knots. This is sampled geometric validation, not continuous swept
collision certification, dynamic obstacle prediction, or controller validation.
The returned trajectories use the planner's 0.025 s interpolation interval;
each stage plans from rest and should be treated as a stop at the boundary.
`--collision-margin` inflates robot spheres by 0.01 m by default, including
attached geometry. Position/orientation tolerances are 0.005 m / 0.05 rad.

## Six-stage tasks, carrying and contact

Send `approach`, `grasp`, `lift`, `transfer`, `release`, `retreat` as six
`MotionStage` objects in one `plan_motion` request. The previous stage's last
joint state becomes the next stage's start. Candidate ranking, grasp selection,
contact policy, gripper actuation and execution remain in the downstream project.

Each stage provides a **complete** context relative to its scene snapshot:

```python
from robot_tools.services.curobo import AllowedContact, Attachment, MotionStage

carry = Attachment(
    object_name="picked_object",
    # Covering spheres expressed in attached_object coordinates, x/y/z/r metres.
    spheres=np.array([[0, 0, 0.12, 0.05]], dtype=np.float32),
)
lift = MotionStage(name="lift", goal_pose=lift_pose, attachment=carry)
transfer = MotionStage(name="transfer", goal_pose=transfer_pose, attachment=carry)
release = MotionStage(
    name="release", goal_pose=release_pose, attachment=carry,
    allowed_contacts=[AllowedContact(obstacle="receiver", robot_links=["attached_object"])],
)
retreat = MotionStage(
    name="retreat", goal_pose=retreat_pose,
    obstacle_poses={"picked_object": released_object_pose},
    allowed_contacts=[AllowedContact(obstacle="picked_object", robot_links=["panda_leftfinger", "panda_rightfinger"])],
)
```

`attachment` removes the named world object for that stage and adds up to 64
covering spheres to the robot. In the default YAML, `attached_object` coincides
with `panda_hand`. Compute its sphere centres using the calibrated grasp/object
transform. Geometry fitting is caller-owned; a sparse under-covering sphere set
will miss collisions. Release is expressed by omitting the attachment and
supplying the object's released world pose. Include that override in subsequent
stages as well. No attachment or pose update silently persists across stages.

`allowed_contacts` exempts exactly the named obstacle/link pairs for the **whole
stage**. Split a stage if contact is allowed only near its end. Unknown names
fail rather than becoming wildcards. The optimizer retains contact objects and
computes costs and gradients with only the named links exempted for each object.
IK, trajectory optimization and graph search use the same pair permissions.
The independent checker tests those objects against every
non-exempt link, including the carried object. A proposal that violates another
pair is rejected, so contact tasks can have conservative planning failures.
Robot self-collision exclusions remain those in the robot YAML.

## Cartesian waypoint planning (v0.9.0)

Existing stages still call `plan_pose`. Add `cartesian_path` to a `MotionStage`
to call `plan_pose_waypoints` with collision checking and the same contact and
attachment context. `CartesianPath.poses` contains 1–72 ordered `(xyz, qwxyz)`
tool poses in the robot base frame, including `goal_pose` as the final target
and excluding the current FK pose. The limit matches the default 16-control
B-spline; oversized paths fail validation instead of being truncated.

```python
from robot_tools.services.curobo import CartesianPath, MotionStage

stage = MotionStage(
    name="follow-path", goal_pose=targets[-1],
    cartesian_path=CartesianPath(
        poses=targets,  # float32 array (K, 7), unit wxyz quaternions
        path_tolerance=(0.02, 0.174533),  # metres, radians
        final_tolerance=(0.005, 0.05),
        track_orientation=True,
    ),
)
```

`track_orientation=False` tracks positions only, including at the final target.
`ik_only` rejects Cartesian paths. Fake mode rejects them explicitly and does
not claim to validate a Cartesian trajectory. Real planning uses GPU BiRRT,
B-spline trajectory optimization and TimeOpt; existing independent trajectory
checks remain enabled. No fixed end-to-end speedup is promised.

## Scene ownership and geometry

Every IK/check/plan request binds an explicit `SceneRef`. Updating an existing
scene ID creates a new opaque version and does not alter the old snapshot.
The last 32 referenced/registered snapshots are retained in memory. After
eviction or restart, `scene_not_found` requires re-registering the world.
Mesh uploads persist as NPZ files with opaque IDs under `assets/` and can be
reused by many scene versions without retransmitting triangles. No caller-local
file paths are interpreted by the server. Mesh BVHs include mesh IDs in
their cache identity; a reused object name cannot select another scene's mesh.
At most 128 distinct named mesh geometries can be resident between cleanup calls.
Use `CuroboClient(cleanup_on_close=True)` for a trial: its context manager or
`close()` releases the exact scene versions and mesh IDs uploaded by that client,
including partial uploads when a later operation raises. Shared mesh files are
deleted only after their last retained scene reference is released. Other clients'
scene references and mesh files remain valid. Ordinary clients retain the previous
persistent-resource behavior unless they explicitly call `release_resources()`.

Cleanup is serialized with inference. It synchronizes the GPU, detaches carried
geometry, invalidates the active world and graph-search nodes, and releases Warp
mesh BVHs in the planner, collision checker and contact partitions. Fixed tensor
buffers and CUDA captures remain allocated; the next request reloads its explicit
scene before computing. Cleanup does not run between requests within a trial.
Client opt-in requires the `release_resources` action from the server, so an old
worker fails capability negotiation rather than silently leaking resources.

Experiment artifacts remain with the caller. SIGKILL, host failure and a lost
upload response cannot execute client teardown; abandoned resources then require
worker recovery. Cleanup failures propagate to the caller rather than reporting
success. The HTTP transport still closes on a cleanup error.

Blender geometry should be exported to triangle vertices/faces in metres.
For observed point clouds, the NumPy-only `occupied_voxel_mesh(points,
voxel_size)` helper builds the exposed faces of occupied cells for `put_mesh`.
It does not fuse RGB-D, infer unseen geometry, or treat unobserved space as
verified free space. A fused/reconstructed mesh can be uploaded the same way.
Use actual current object poses, not a desired future Blender arrangement.

## Validation

```bash
uv run --extra server pytest -q tests/test_curobo.py
# Start/stop its own real service on 15562, default GPU 1:
bash runtimes/curobo/validate.sh
# Or check an already running server:
uv run python runtimes/curobo/smoke.py --port 5562
```

The real smoke covers reachable/unreachable IK, blocked planning, six-stage
planning, contacts, attachments, release updates and mesh/scene isolation.
No robot motion is executed. See [prerequisites](PREREQUISITES.md) for recorded
hardware and acceptance evidence.

## Official API sources

This runtime pins cuRoboV2-Fast v0.9.0 commit `42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83`.
The older `curobo.org` V1 `MotionGen` examples do not match this version.

- [Official installation](https://github.com/NVlabs/curobo/blob/42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83/docs/getting-started/installation.rst)
- [Official IK example](https://github.com/NVlabs/curobo/blob/42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83/curobo/examples/getting_started/inverse_kinematics.py)
- [Official motion planning example](https://github.com/NVlabs/curobo/blob/42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83/curobo/examples/getting_started/motion_planning.py)
- [Official collision checking API](https://github.com/NVlabs/curobo/blob/42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83/curobo/collision_checking.py)
