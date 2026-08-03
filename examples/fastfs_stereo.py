"""Stereo pair → disparity (server) → depth (client-side helper).

The server returns raw disparity in pixels; conversion to metric depth is
camera-specific so the client SDK ships a `disparity_to_depth` helper that
takes your intrinsics + baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _config import load_endpoint, load_params
from PIL import Image

from robot_tools.services.fastfs import FastFSClient, disparity_to_depth


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("depth.npy"))
    args = p.parse_args()

    params = load_params("fastfs")
    cam = params["camera"]
    left = load_image(args.left)
    right = load_image(args.right)

    with FastFSClient.from_endpoint(load_endpoint("fastfs")) as client:
        resp = client.reconstruct(
            left,
            right,
            valid_iters=params["valid_iters"],
            max_disp=params["max_disp"],
        )

    disp = resp.disparity
    print(f"disparity: shape={disp.shape} min={disp.min():.2f} max={disp.max():.2f}")

    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float32)
    depth = disparity_to_depth(disp, K, cam["baseline"])
    finite = depth[np.isfinite(depth)]
    print(f"depth (m): min={finite.min():.3f} max={finite.max():.3f} mean={finite.mean():.3f}")

    np.save(args.out, depth)
    print(f"saved depth → {args.out}")


if __name__ == "__main__":
    main()
