"""Native GraspGenX named-gripper inference on a segmented object PC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _config import load_endpoint, load_params

from robot_tools.services.graspgenx import GraspGenXClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object",
        type=Path,
        required=True,
        help="Segmented object point cloud: (N,3) float32 .npy in meters",
    )
    parser.add_argument(
        "--gripper",
        default="robotiq_2f_85",
        help="Name from the official gripper_descriptions assets",
    )
    args = parser.parse_args()

    object_pc = np.load(args.object).astype(np.float32)
    params = load_params("graspgenx")
    with GraspGenXClient.from_endpoint(load_endpoint("graspgenx")) as client:
        response = client.generate_grasps(
            object_pc,
            gripper_name=args.gripper,
            num_grasps=params["num_grasps"],
            grasp_threshold=params["grasp_threshold"],
            topk_num_grasps=params["topk_num_grasps"],
        )

    print(
        f"gripper={response.gripper_name} grasps={len(response.grasps)} server_infer_ms={response.timing.infer_ms:.1f}"
    )
    if len(response.grasps):
        best = int(np.argmax(response.confidences))
        print(f"best confidence={response.confidences[best]:.3f}")
        print(response.grasps[best])


if __name__ == "__main__":
    main()
