"""Point-SAM point order is preserved; coordinates and prompts share one frame in metres."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "pointsam"
API_VERSION = "1"
ACTIONS = frozenset({"encode_cloud", "segment", "release_cloud"})


def finite_float_array(value: np.ndarray, name: str, ndim: int) -> None:
    if value.ndim != ndim or value.dtype not in (np.float32, np.float64) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite {ndim}D float32/float64 array")


class EncodeCloudRequest(BaseRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    action: Literal["encode_cloud"] = "encode_cloud"
    xyz: np.ndarray  # (N, 3), float32/64, metres, common caller-defined frame
    rgb: np.ndarray  # (N, 3), uint8 RGB, same order as xyz
    num_groups: int = Field(default=512, ge=3, le=2048, strict=True)
    group_size: int = Field(default=64, ge=3, le=512, strict=True)

    @model_validator(mode="after")
    def validate_cloud(self):
        finite_float_array(self.xyz, "xyz", 2)
        if self.xyz.shape[1:] != (3,) or len(self.xyz) < max(self.num_groups, self.group_size):
            raise ValueError("xyz must be (N, 3), with N >= num_groups and group_size")
        if self.rgb.shape != self.xyz.shape or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be (N, 3) uint8 RGB matching xyz")
        if not np.any(self.xyz != self.xyz[0]):
            raise ValueError("point cloud must have nonzero spatial extent")
        return self


class EncodeCloudResponse(BaseResponse):
    cloud_id: str = Field(min_length=1)
    num_points: int = Field(gt=0, strict=True)


class SegmentRequest(BaseRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    action: Literal["segment"] = "segment"
    cloud_id: str = Field(min_length=1, max_length=128)
    prompt_points: np.ndarray  # (Q, 3) float32/64 in the ORIGINAL xyz frame, metres
    prompt_labels: np.ndarray  # (Q,) bool or integer: 1 foreground, 0 background
    prompt_mask: np.ndarray | None = None  # (N,) finite float logits, NOT a binary mask
    multimask_output: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def validate_prompts(self):
        finite_float_array(self.prompt_points, "prompt_points", 2)
        if self.prompt_points.shape[1:] != (3,) or not 1 <= len(self.prompt_points) <= 64:
            raise ValueError("prompt_points must have shape (Q, 3), 1 <= Q <= 64")
        labels = self.prompt_labels
        if labels.shape != (len(self.prompt_points),) or labels.dtype.kind not in "biu":
            raise ValueError("prompt_labels must be a matching boolean/integer (Q,) array")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("prompt_labels must contain only 0 and 1")
        if self.prompt_mask is not None:
            finite_float_array(self.prompt_mask, "prompt_mask", 1)
            if not len(self.prompt_mask):
                raise ValueError("prompt_mask must not be empty")
        return self


class SegmentResponse(BaseResponse):
    cloud_id: str = Field(min_length=1)
    masks: np.ndarray  # (M, N) bool, original input point order; logits > 0
    scores: np.ndarray  # (M,) float32 predicted IoU, descending; not calibrated probability
    logits: np.ndarray  # (M, N) float32; pass one row as the next prompt_mask

    @model_validator(mode="after")
    def validate_result(self):
        if self.masks.dtype != np.bool_ or self.masks.ndim != 2:
            raise ValueError("masks must be (M, N) bool")
        count, points = self.masks.shape
        if count not in (1, 3) or points < 1:
            raise ValueError("expected one or three nonempty masks")
        for name, value, shape in (("scores", self.scores, (count,)), ("logits", self.logits, self.masks.shape)):
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 with shape {shape}")
        if np.any(np.diff(self.scores) > 0) or not np.array_equal(self.masks, self.logits > 0):
            raise ValueError("scores must be descending and masks must equal logits > 0")
        return self


class ReleaseCloudRequest(BaseRequest):
    model_config = ConfigDict(extra="forbid")
    action: Literal["release_cloud"] = "release_cloud"
    cloud_id: str = Field(min_length=1, max_length=128)


class ReleaseCloudResponse(BaseResponse):
    released: bool
