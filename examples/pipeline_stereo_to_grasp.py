"""End-to-end pipeline: stereo pair → grasp pose.

  1. FastFS  — left/right images   → disparity → depth
  2. SAM 3   — left image + text   → object mask
  3. backproject masked depth pixels via intrinsics → object point cloud
  4. GraspGen — object PC + scene PC → collision-free grasps

This is the canonical "downstream consumer" shape: each model is one
HTTP call; per-call knobs are kwargs; per-camera math (depth conversion,
backprojection) lives on the client side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _config import load_endpoint, load_params
from PIL import Image

from robot_tools.services.fastfs import FastFSClient, disparity_to_depth
from robot_tools.services.graspgen import GraspGenClient
from robot_tools.services.sam3 import SAM3Client


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(H,W) depth + 3x3 K → (N,3) point cloud in camera frame; optional mask filter."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return pts.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--target", required=True, help="text prompt for the object to grasp")
    args = p.parse_args()

    ffs_params = load_params("fastfs")
    cam = ffs_params["camera"]
    sam_params = load_params("sam3")
    gg_params = load_params("graspgen")

    left = np.asarray(Image.open(args.left).convert("RGB"))
    right = np.asarray(Image.open(args.right).convert("RGB"))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float32)

    # 1. stereo → depth
    with FastFSClient.from_endpoint(load_endpoint("fastfs")) as client:
        disp = client.reconstruct(
            left,
            right,
            valid_iters=ffs_params["valid_iters"],
            max_disp=ffs_params["max_disp"],
        ).disparity
    depth = disparity_to_depth(disp, K, cam["baseline"])
    print(f"depth: {depth.shape}, finite={np.isfinite(depth).sum()}")

    # 2. text → object mask
    with SAM3Client.from_endpoint(load_endpoint("sam3")) as client:
        det_resp = client.segment_by_text(left, args.target, score_threshold=sam_params["score_threshold"])
    if not det_resp.detections:
        raise SystemExit(f"no detection for '{args.target}'")
    best = det_resp.detections[0]
    print(f"target '{args.target}' score={best.score:.3f} pixels={int(best.mask.sum())}")

    # 3. backproject — full scene PC, plus the masked-only object PC
    scene_pc = backproject(depth, K)
    object_pc = backproject(depth, K, mask=best.mask)
    print(f"scene PC={scene_pc.shape} object PC={object_pc.shape}")
    if len(object_pc) < 64:
        raise SystemExit("object point cloud too sparse — check mask / depth")

    # 4. grasp
    with GraspGenClient.from_endpoint(load_endpoint("graspgen")) as client:
        resp = client.generate_collision_free_grasps(
            object_pc,
            scene_pc,
            grasp_threshold=gg_params["grasp_threshold"],
            num_grasps=gg_params["num_grasps"],
            topk=gg_params["topk"],
            collision_threshold=gg_params["collision_threshold"],
            max_scene_points=gg_params["max_scene_points"],
        )
    free = int(resp.collision_free_mask.sum())
    print(f"got {len(resp.poses)} grasps ({free} collision-free)")

    if len(resp.poses):
        best_idx = int(np.argmax(resp.confidences * resp.collision_free_mask.astype(np.float32)))
        print(f"best collision-free grasp #{best_idx}: conf={resp.confidences[best_idx]:.3f}")
        print(resp.poses[best_idx])


if __name__ == "__main__":
    main()
