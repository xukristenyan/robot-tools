"""Predict placements from two masked RGBD observations stored as NPZ files."""

import argparse
from pathlib import Path

import numpy as np

from robot_tools.core.endpoints import get_endpoint
from robot_tools.services.anyplace import AnyPlaceClient, masked_rgbd


def load_observation(path):
    # NPZ keys: rgb, depth (metres), mask, intrinsics, optional camera_pose.
    with np.load(path, allow_pickle=False) as data:
        return masked_rgbd(
            data["rgb"],
            data["depth"],
            data["mask"],
            data["intrinsics"],
            camera_pose=data.get("camera_pose"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick", required=True, help="Pick-object observation NPZ")
    parser.add_argument("--place", required=True, help="Local receiving-region observation NPZ")
    parser.add_argument("--endpoints", default=str(Path(__file__).with_name("endpoints.yaml")))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--n-refine-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="anyplace_relative.npz")
    args = parser.parse_args()
    pick, place = load_observation(args.pick), load_observation(args.place)
    endpoint = get_endpoint("anyplace", args.endpoints)
    with AnyPlaceClient.from_endpoint(endpoint, timeout_ms=120_000) as client:
        response = client.predict_placements(
            pick,
            place,
            num_samples=args.num_samples,
            n_refine_iters=args.n_refine_iters,
            seed=args.seed,
        )
    # Keep the upstream output field without pickle-dependent object arrays.
    np.savez(args.output, relative=response.relative)
    print(f"Saved relative {response.relative.shape} {response.relative.dtype} to {args.output}")


if __name__ == "__main__":
    main()
