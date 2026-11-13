"""Real GPU acceptance through the lightweight HTTP client; no robot execution."""

from __future__ import annotations

import argparse
import json

import numpy as np

from robot_tools.services.curobo import (
    AllowedContact,
    Attachment,
    CartesianPath,
    Cuboid,
    CuroboClient,
    Mesh,
    MotionStage,
    StageContext,
    World,
    occupied_voxel_mesh,
)


def pose(x, y=0, z=0, quaternion=(1, 0, 0, 0)):
    return np.array([x, y, z, *quaternion], dtype=np.float32)


def run(client):
    info = client.get_robot_info()
    assert not info.fake, "Real smoke requires a real server"
    names = info.joint_names
    goal = pose(0.45, 0, 0.45, (0, 1, 0, 0))
    timings = {}

    def register(name, world):
        result = client.put_scene(f"smoke-{name}", world)
        assert result.failure_code is None, result.failure_code
        return result.scene

    empty = register("empty", World())
    initial = client.check_trajectory(info.default_position[None], scene=empty, joint_names=names)
    assert initial.valid, initial
    ik = client.solve_ik(goal, info.default_position, scene=empty, joint_names=names)
    assert ik.success.all(), ik
    timings["ik_ms"] = ik.timing_ms
    solved = ik.positions[0]
    check = client.check_trajectory(solved[None], scene=empty, joint_names=names)
    assert check.valid, check
    unreachable = client.solve_ik(pose(10, 0, 10)[None], solved, scene=empty, joint_names=names)
    assert not unreachable.success.any()
    print("PASS: reachable/unreachable IK and valid robot configurations", flush=True)

    blocked = register("blocked", World(obstacles=[Cuboid(name="wall", pose=pose(0, 0, 0.5), dims=(4, 4, 3))]))
    bad = client.check_trajectory(solved[None], scene=blocked, joint_names=names)
    assert not bad.valid and bad.world_collision.all(), bad
    raw = client.solve_ik(goal, solved, scene=blocked, joint_names=names, collision_check=False)
    assert raw.success.all() and not raw.collision_checked
    denied = client.plan_motion(solved, [MotionStage(name="blocked", goal_pose=goal)], scene=blocked, joint_names=names)
    assert not denied.success and denied.failure_code == "start_in_collision", denied
    assert client.check_trajectory(solved[None], scene=empty, joint_names=names).valid
    print("PASS: collision rejection, explicit raw IK and scene isolation", flush=True)

    stages = []
    for name, dy, dz in [
        ("approach", 0, 0.05),
        ("grasp", 0, 0),
        ("lift", 0, 0.10),
        ("transfer", 0.08, 0.10),
        ("release", 0.08, 0),
        ("retreat", 0.08, 0.10),
    ]:
        target = goal.copy()
        target[1] += dy
        target[2] += dz
        stages.append(MotionStage(name=name, goal_pose=target))
    plan = client.plan_motion(info.default_position, stages, scene=empty, joint_names=names)
    assert plan.success, plan
    assert len(plan.stages) == 6
    for previous, following in zip(plan.stages, plan.stages[1:]):
        np.testing.assert_allclose(previous.positions[-1], following.positions[0], atol=1e-3)
    assert all(s.dt > 0 and s.collision_checked for s in plan.stages)
    timings["six_stage_ms"] = plan.timing_ms
    timings["stages_ms"] = {s.name: s.timing_ms for s in plan.stages}
    print("PASS: six-stage collision-aware trajectory planning", flush=True)

    # Ordered Cartesian targets: fixed orientation, then a position-only path.
    for track_orientation in (True, False):
        targets = np.stack([goal.copy(), goal.copy()])
        targets[0, 1] += 0.025
        targets[1, 1] += 0.05
        followed = client.plan_motion(solved, [MotionStage(
            name="cartesian", goal_pose=targets[-1],
            cartesian_path=CartesianPath(poses=targets, track_orientation=track_orientation),
        )], scene=empty, joint_names=names)
        assert followed.success, followed
        assert followed.stages[0].collision_checked and followed.stages[0].dt > 0
        timings[f"cartesian_{track_orientation}_ms"] = followed.timing_ms
    print("PASS: full-pose and position-only Cartesian waypoint planning", flush=True)

    hand_scene = register(
        "hand-contact", World(obstacles=[Cuboid(name="contact", pose=pose(*goal[:3]), dims=(0.1, 0.1, 0.1))])
    )
    assert not client.check_trajectory(solved[None], scene=hand_scene, joint_names=names).valid
    all_contact = StageContext(allowed_contacts=[AllowedContact(obstacle="contact", robot_links=info.collision_links)])
    assert client.check_trajectory(solved[None], scene=hand_scene, joint_names=names, context=all_contact).valid
    finger_only = StageContext(allowed_contacts=[AllowedContact(obstacle="contact", robot_links=["panda_leftfinger"])])
    assert not client.check_trajectory(solved[None], scene=hand_scene, joint_names=names, context=finger_only).valid
    contact_plan = client.plan_motion(solved, [MotionStage(
        name="contact-plan", goal_pose=goal,
        allowed_contacts=all_contact.allowed_contacts,
    )], scene=hand_scene, joint_names=names)
    assert contact_plan.success, contact_plan
    assert not client.check_trajectory(solved[None], scene=hand_scene, joint_names=names).valid
    print("PASS: contact exemptions are specific to obstacle/link pairs", flush=True)

    carried_world = World(
        obstacles=[
            Cuboid(name="pickup", pose=pose(5), dims=(0.05, 0.05, 0.05)),
            Cuboid(name="receiver", pose=pose(0.45, 0, 0.15), dims=(0.06, 0.06, 0.06)),
        ]
    )
    carried_scene = register("attachment", carried_world)
    assert client.check_trajectory(solved[None], scene=carried_scene, joint_names=names).valid
    attachment = Attachment(object_name="pickup", spheres=np.array([[0, 0, 0.30, 0.06]], dtype=np.float32))
    held = StageContext(attachment=attachment)
    attached_check = client.check_trajectory(solved[None], scene=carried_scene, joint_names=names, context=held)
    assert not attached_check.valid and attached_check.world_collision.all(), attached_check
    held_contact = StageContext(
        attachment=attachment, allowed_contacts=[AllowedContact(obstacle="receiver", robot_links=["attached_object"])]
    )
    assert client.check_trajectory(solved[None], scene=carried_scene, joint_names=names, context=held_contact).valid
    assert client.check_trajectory(solved[None], scene=carried_scene, joint_names=names).valid
    released = StageContext(obstacle_poses={"pickup": pose(*goal[:3])})
    assert not client.check_trajectory(solved[None], scene=carried_scene, joint_names=names, context=released).valid
    print("PASS: carried geometry, attachment contact, release pose and state reset", flush=True)

    # Same logical object name, different mesh geometry: never reuse a stale BVH.
    vertices, faces = occupied_voxel_mesh(np.array([[0.01, 0.01, 0.01]], dtype=np.float32), 2.0)
    near = client.put_mesh(vertices - 1, faces).mesh_id
    far = client.put_mesh(vertices + 5, faces).mesh_id
    near_scene = register("mesh-near", World(obstacles=[Mesh(name="reused-name", mesh_id=near, pose=pose(0))]))
    far_scene = register("mesh-far", World(obstacles=[Mesh(name="reused-name", mesh_id=far, pose=pose(0))]))
    assert not client.check_trajectory(solved[None], scene=near_scene, joint_names=names).valid
    assert client.check_trajectory(solved[None], scene=far_scene, joint_names=names).valid
    assert not client.check_trajectory(solved[None], scene=near_scene, joint_names=names).valid
    print("PASS: uploaded meshes, occupied voxel conversion and mesh cache isolation", flush=True)
    return {
        "robot": info.robot_config,
        "upstream_commit": info.upstream_commit,
        "timings": timings,
        "stage_shapes": [list(s.positions.shape) for s in plan.stages],
        "checks": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
    args = parser.parse_args()
    with CuroboClient(host=args.host, port=args.port, timeout_ms=180_000) as client:
        print(json.dumps(run(client), indent=2))


if __name__ == "__main__":
    main()
