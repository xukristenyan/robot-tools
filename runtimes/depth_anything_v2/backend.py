"""Headless adapter; preserve upstream preprocessing and raw model output."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINT_NAME = "depth_anything_v2_vitl.pth"
MODEL_CONFIG = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}


def checkpoint_path(checkpoint: str | Path | None = None) -> Path:
    if checkpoint is not None:
        return Path(checkpoint).expanduser().resolve()
    root = os.environ.get(
        "DEPTH_ANYTHING_V2_WEIGHTS_DIR", f"/data/{getpass.getuser()}/robot-tools-models/depth_anything_v2"
    )
    return Path(root).expanduser().resolve() / CHECKPOINT_NAME


def upstream_model_class():
    upstream = HERE / "upstream"
    if not (upstream / "depth_anything_v2" / "dpt.py").is_file():
        raise FileNotFoundError("Depth Anything V2 upstream is missing; run robot-tools install <profile>")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from depth_anything_v2.dpt import DepthAnythingV2

    return DepthAnythingV2


class DepthAnythingV2Backend:
    def __init__(self, *, checkpoint: str | Path | None = None):
        import torch

        path = checkpoint_path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Large checkpoint: {path}; run robot-tools download <profile>")
        model_cls = upstream_model_class()
        # Match upstream infer_image's device selection. To force CPU, launch
        # with CUDA_VISIBLE_DEVICES="" on Linux.
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = model_cls(**MODEL_CONFIG)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model = self.model.to(self.device).eval()

    def infer(self, image: np.ndarray, *, input_size: int = 518) -> np.ndarray:
        # infer_image expects OpenCV BGR and internally converts it to RGB.
        # Keep official resizing, normalization, interpolation, and precision.
        bgr = np.ascontiguousarray(image[..., ::-1])
        depth = self.model.infer_image(bgr, input_size=input_size)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        if depth.shape != image.shape[:2]:
            raise RuntimeError("Upstream returned a different image resolution")
        return depth
