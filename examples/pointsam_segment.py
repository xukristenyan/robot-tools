"""Segment a saved colored cloud using points and optional prior logits.

Input NPZ: xyz (N,3) float metres, rgb (N,3) uint8, prompt_points (Q,3)
in the same frame, prompt_labels (Q,) 0/1, optional prompt_mask (N,) logits.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from robot_tools.core.endpoints import get_endpoint
from robot_tools.services.pointsam import PointSAMClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--endpoints", type=Path, default=Path("examples/endpoints.yaml"))
    parser.add_argument("--config", type=Path, default=Path("examples/configs/pointsam.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if set(config) != {"num_groups", "group_size", "multimask_output"}:
        raise ValueError("config must contain num_groups, group_size, multimask_output")
    with np.load(args.input, allow_pickle=False) as data:
        xyz, rgb = data["xyz"], data["rgb"]
        prompts, labels = data["prompt_points"], data["prompt_labels"]
        mask = data.get("prompt_mask", None)
    with PointSAMClient.from_endpoint(get_endpoint("pointsam", args.endpoints)) as client:
        cloud = client.encode_cloud(xyz, rgb, num_groups=config["num_groups"], group_size=config["group_size"])
        try:
            result = client.segment(
                cloud.cloud_id, prompts, labels, prompt_mask=mask, multimask_output=config["multimask_output"]
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, masks=result.masks, logits=result.logits, scores=result.scores)
            print(f"{len(result.masks)} masks for {cloud.num_points} points -> {args.output}")
        finally:
            client.release_cloud(cloud.cloud_id)


if __name__ == "__main__":
    main()
