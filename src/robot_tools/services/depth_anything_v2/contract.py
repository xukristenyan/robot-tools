from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, field_validator

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "depth_anything_v2"
API_VERSION = "1"
ACTIONS = frozenset({"estimate_depth"})


class EstimateDepthRequest(BaseRequest):
    action: Literal["estimate_depth"] = "estimate_depth"
    image: np.ndarray  # (H, W, 3), uint8 RGB; no implicit float scaling or BGR conversion
    input_size: int | None = Field(default=None, ge=14, strict=True)

    @field_validator("image")
    @classmethod
    def validate_image(cls, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("image must be a nonempty (H, W, 3) RGB array")
        if image.dtype != np.uint8:
            raise ValueError("image must have dtype uint8 with values in [0, 255]")
        return np.ascontiguousarray(image)


class EstimateDepthResponse(BaseResponse):
    # Raw affine-invariant inverse depth, at the input image's (H, W).
    # Larger means nearer. Arbitrary units; not meters, disparity pixels,
    # or a min/max-normalized visualization. Scales can differ across images.
    relative_inverse_depth: np.ndarray

    @field_validator("relative_inverse_depth")
    @classmethod
    def validate_depth(cls, depth: np.ndarray) -> np.ndarray:
        if depth.ndim != 2 or min(depth.shape) < 1 or depth.dtype != np.float32:
            raise ValueError("relative_inverse_depth must be nonempty (H, W) float32")
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("relative_inverse_depth must contain finite, nonnegative values")
        return depth
