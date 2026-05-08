"""GraspGen — grasps for an object point cloud.

Two flavors: `generate_grasps` ignores collisions; `generate_collision_free_grasps`
also takes a full scene point cloud and filters poses against it. Pass
`--scene` to use the second flavor.

Server-startup args (gripper choice, model id) live in `services.yaml`'s
`extra_args`, not here. This script just consumes whichever gripper the
running server was launched with.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_tools.clients.graspgen import GraspGenClient

from _config import load_params, load_service


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--object", type=Path, required=True, help="(N,3) float32 .npy")
    p.add_argument("--scene", type=Path, default=None, help="(M,3) float32 .npy; enables collision-free")
    args = p.parse_args()

    params = load_params("graspgen")
    object_pc = np.load(args.object).astype(np.float32)
    print(f"object PC: {object_pc.shape}")

    host, port = load_service("graspgen")
    client = GraspGenClient(host=host, port=port)

    if args.scene is not None:
        scene_pc = np.load(args.scene).astype(np.float32)
        print(f"scene PC: {scene_pc.shape}")
        resp = client.generate_collision_free_grasps(
            object_pc, scene_pc,
            grasp_threshold=params["grasp_threshold"],
            num_grasps=params["num_grasps"],
            topk=params["topk"],
            collision_threshold=params["collision_threshold"],
            max_scene_points=params["max_scene_points"],
        )
        free = int(resp.collision_free_mask.sum())
        print(f"got {len(resp.poses)} poses ({free} collision-free)")
    else:
        resp = client.generate_grasps(
            object_pc,
            grasp_threshold=params["grasp_threshold"],
            num_grasps=params["num_grasps"],
            topk=params["topk"],
        )
        print(f"got {len(resp.poses)} poses")

    if len(resp.poses):
        best = int(np.argmax(resp.confidences))
        print(f"best grasp #{best}: confidence={resp.confidences[best]:.3f}")
        print(resp.poses[best])


if __name__ == "__main__":
    main()
