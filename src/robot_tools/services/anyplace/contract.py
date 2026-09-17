"""AnyPlace low-level placement; all geometry uses metres in one common frame."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "anyplace"
API_VERSION = "1"
ACTIONS = frozenset({"predict_placements"})


def validate_rigid(transforms: np.ndarray) -> None:
    if not np.issubdtype(transforms.dtype, np.floating) or not np.isfinite(transforms).all():
        raise ValueError("transforms must contain finite floating-point values")
    if not np.allclose(transforms[..., 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("transforms must have homogeneous last row [0, 0, 0, 1]")
    rotation = transforms[..., :3, :3]
    if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(3), atol=1e-3):
        raise ValueError("transform rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("transform rotations must have determinant +1")


class MaskedRGBD(BaseModel):
    """Aligned RGB/depth/mask from one rectified pinhole camera.

    depth is optical-axis Z in metres (not range). Zero/negative/nonfinite
    depth is ignored. camera_pose maps camera XYZ (right, down, forward) into
    the common output frame; identity means the camera frame is that frame.
    Pick and place may use different cameras if both poses share that frame.
    RGB is part of the observation but the upstream model uses geometry only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    rgb: np.ndarray  # (H, W, 3), uint8 RGB
    depth: np.ndarray  # (H, W), float32/64 metres
    mask: np.ndarray  # (H, W), bool; client also accepts integer binary masks
    intrinsics: np.ndarray  # (3, 3), zero-skew pinhole K
    camera_pose: np.ndarray = Field(default_factory=lambda: np.eye(4, dtype=np.float64))

    @model_validator(mode="after")
    def validate_observation(self) -> MaskedRGBD:
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be (H, W, 3) uint8 RGB")
        shape = self.rgb.shape[:2]
        if min(shape) == 0 or self.depth.shape != shape or self.mask.shape != shape:
            raise ValueError("rgb, depth and mask must have matching nonempty image dimensions")
        if self.depth.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("depth must be float32 or float64 in metres")
        if self.mask.dtype != np.bool_:
            raise ValueError("mask must be boolean")
        valid = self.mask & np.isfinite(self.depth) & (self.depth > 0)
        if np.count_nonzero(valid) < 4:
            raise ValueError("mask must select at least four pixels with finite positive depth")
        k = self.intrinsics
        if k.shape != (3, 3) or not np.issubdtype(k.dtype, np.floating) or not np.isfinite(k).all():
            raise ValueError("intrinsics must be a finite floating-point (3, 3) matrix")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("intrinsics must have positive focal lengths and last row [0, 0, 1]")
        if not np.allclose([k[0, 1], k[1, 0]], 0):
            raise ValueError("intrinsics must describe a zero-skew rectified camera")
        if self.camera_pose.shape != (4, 4):
            raise ValueError("camera_pose must be (4, 4)")
        validate_rigid(self.camera_pose)
        return self


class PredictPlacementsRequest(BaseRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    action: Literal["predict_placements"] = "predict_placements"
    pick: MaskedRGBD
    place: MaskedRGBD  # mask selects the local receiving region (upstream parent)
    num_samples: int = Field(default=5, ge=1, le=32, strict=True)
    n_refine_iters: int = Field(default=50, ge=5, le=100, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)


class PredictPlacementsResponse(BaseResponse):
    """Native upstream `relative`: (N, 4, 4) float64, unsorted candidates.

    p_final = relative[i] @ p_initial for homogeneous pick-object points in
    the common frame. T_place_gripper = relative[i] @ T_pick_gripper.
    No learned success scores or collision/IK guarantees are supplied.
    """

    relative: np.ndarray

    @model_validator(mode="after")
    def validate_relative(self) -> PredictPlacementsResponse:
        if self.relative.ndim != 3 or self.relative.shape[1:] != (4, 4) or len(self.relative) == 0:
            raise ValueError("relative must have nonempty shape (N, 4, 4)")
        if self.relative.dtype != np.float64:
            raise ValueError("relative must use upstream float64 dtype")
        validate_rigid(self.relative)
        return self
