"""Single RGB image to raw relative inverse depth (arbitrary units, not meters)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _config import load_endpoint, load_params
from PIL import Image

from robot_tools.services.depth_anything_v2 import DepthAnythingV2Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("relative_inverse_depth.npy"))
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    params = load_params("depth_anything_v2")
    with DepthAnythingV2Client.from_endpoint(load_endpoint("depth_anything_v2")) as client:
        result = client.estimate_depth(image, input_size=params["input_size"])
    depth = result.relative_inverse_depth
    np.save(args.out, depth)
    print(f"relative inverse depth: {depth.shape}, {depth.dtype}, range=[{depth.min():.4f}, {depth.max():.4f}]")
    print(f"saved {args.out}; arbitrary units, larger means nearer (not metric depth)")


if __name__ == "__main__":
    main()
