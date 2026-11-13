"""Plan/check a pose, or send a complete six-stage pick/place task in one call."""

import argparse

import numpy as np
from _config import load_endpoint, load_params

from robot_tools.services.curobo import CuroboClient, MotionStage, World


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--six-stage", action="store_true", help="Demonstrate six pose targets; no physical grasp/execution"
    )
    args = parser.parse_args()
    options = load_params("curobo")
    goal = np.asarray(options.pop("goal_pose"), dtype=np.float32)
    stages = [MotionStage(name="reach", goal_pose=goal)]
    if args.six_stage:
        # Illustrative robot-base targets. Real downstream code supplies its
        # measured scene, calibrated TCP conversion, attachments and contacts.
        offsets = [0.10, 0.0, 0.10, 0.15, 0.05, 0.15]
        names = ["approach", "grasp", "lift", "transfer", "release", "retreat"]
        stages = []
        for name, dz in zip(names, offsets):
            pose = goal.copy()
            pose[2] += dz
            stages.append(MotionStage(name=name, goal_pose=pose))
    with CuroboClient.from_endpoint(load_endpoint("curobo"), timeout_ms=120_000) as client:
        info = client.get_robot_info()
        registered = client.put_scene("example", World())
        if registered.failure_code:
            raise RuntimeError(registered.failure_code)
        result = client.plan_motion(
            info.default_position,
            stages,
            scene=registered.scene,
            joint_names=info.joint_names,
            **options,
        )
        print(
            f"success={result.success}, fake={result.fake}, failed_stage={result.failed_stage}, reason={result.failure_code}"
        )
        for stage in result.stages:
            print(
                f"{stage.name}: success={stage.success}, shape={stage.positions.shape}, dt={stage.dt}, ms={stage.timing_ms:.1f}"
            )
        if result.success:
            checked = client.check_trajectory(
                result.stages[0].positions,
                scene=registered.scene,
                joint_names=info.joint_names,
            )
            print(f"first-stage check: valid={checked.valid}, samples={len(checked.sampled_positions)}")
        else:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
