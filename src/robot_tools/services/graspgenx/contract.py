"""Typed contracts for the native GraspGenX serving API.

The action names and inference semantics mirror NVlabs/GraspGenX's official
``graspgenx.serving`` layer.  Transport, readiness, errors, and admission are
provided by robot-tools rather than the upstream ZMQ wrapper.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "graspgenx"
API_VERSION = "2"
ACTIONS = frozenset(
    {
        "infer",
        "infer_object",
        "infer_scene_depth",
        "infer_scene_pc",
        "metadata",
    }
)

Planner = Literal["graspmoe", "diffusion"]
BranchTag = Literal["diff", "obb"]
OBBMode = Literal["advanced", "pca"]
OBBSkipRule = Literal["auto", "never"]
OBBDensity = Literal["sparse", "dense", "dense-topandside"]

FLAT_SWEEP_VOLUME_FIELDS = (
    "extents_open",
    "offset_open",
    "extents_mid",
    "offset_mid",
)


def _vector3(value: object, *, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not np.all(array > 0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def _object_point_cloud(value: object) -> np.ndarray:
    point_cloud = np.asarray(value, dtype=np.float32)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3 or len(point_cloud) == 0:
        raise ValueError("point_cloud must be a non-empty (N, 3) array")
    if not np.all(np.isfinite(point_cloud)):
        raise ValueError("point_cloud must contain only finite values")
    return np.ascontiguousarray(point_cloud)


def _grasps(value: object) -> np.ndarray:
    grasps = np.asarray(value, dtype=np.float32)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError("grasps must have shape (K, 4, 4)")
    return grasps


def _confidences(value: object) -> np.ndarray:
    confidences = np.asarray(value, dtype=np.float32)
    if confidences.ndim != 1:
        raise ValueError("confidences must have shape (K,)")
    return confidences


class SweepVolumeParams(BaseModel):
    """Gripper conditioning used by the released sweep-volume-v2 model.

    All distances are meters in the gripper base frame. ``+Z`` is the
    approach direction and ``+X`` is the closing direction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    extents_open: np.ndarray
    offset_open: np.ndarray
    extents_mid: np.ndarray
    offset_mid: np.ndarray
    gripper_type: Literal[0, 1, 2] = 0
    fingertip_depth: float | None = Field(default=None, gt=0)

    @field_validator("extents_open", "extents_mid", mode="before")
    @classmethod
    def _validate_extents(cls, value: object, info) -> np.ndarray:
        return _vector3(value, name=info.field_name, positive=True)

    @field_validator("offset_open", "offset_mid", mode="before")
    @classmethod
    def _validate_offsets(cls, value: object, info) -> np.ndarray:
        return _vector3(value, name=info.field_name)

    @property
    def resolved_fingertip_depth(self) -> float:
        if self.fingertip_depth is not None:
            return float(self.fingertip_depth)
        return float(self.offset_open[2] + self.extents_open[2] / 2.0)

    @property
    def jaw_width(self) -> float:
        return float(self.extents_open[0])

    def to_flat(self) -> np.ndarray:
        return np.concatenate([getattr(self, field) for field in FLAT_SWEEP_VOLUME_FIELDS]).astype(np.float32)

    def to_wire(self) -> dict:
        return self.model_dump()

    def cache_key(self) -> str:
        rounded = np.round(self.to_flat().astype(np.float64), 6)
        payload = (
            rounded.tobytes()
            + int(self.gripper_type).to_bytes(1, byteorder="big")
            + np.float64(round(self.resolved_fingertip_depth, 6)).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_flat(
        cls,
        flat: object,
        *,
        gripper_type: Literal[0, 1, 2] = 0,
        fingertip_depth: float | None = None,
    ) -> SweepVolumeParams:
        values = np.asarray(flat, dtype=np.float32).reshape(-1)
        if values.shape != (12,):
            raise ValueError("flat sweep-volume conditioning must contain 12 values")
        return cls(
            extents_open=values[0:3],
            offset_open=values[3:6],
            extents_mid=values[6:9],
            offset_mid=values[9:12],
            gripper_type=gripper_type,
            fingertip_depth=fingertip_depth,
        )

    @classmethod
    def coerce(cls, value: object) -> SweepVolumeParams:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, (np.ndarray, list, tuple)):
            return cls.from_flat(value)
        raise TypeError("sweep_volume_params must be SweepVolumeParams, a mapping, or a flat 12-value array")


class _GraspGenXRequest(BaseRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class MetadataRequest(_GraspGenXRequest):
    action: Literal["metadata"] = "metadata"


class ModelMetadata(BaseModel):
    generator_backbone: str
    discriminator_backbone: str
    grasp_repr: str
    num_diffusion_iters_eval: int


class PrecisionMetadata(BaseModel):
    weights: str = "fp32"
    tensorrt: bool = False
    tensorrt_precision: Literal["fp32", "fp16"] | None = None


class MetadataResponse(BaseResponse):
    default_gripper: str | None
    loaded_grippers: list[str]
    model: ModelMetadata
    assets_dir: str
    actions: list[str]
    precision: PrecisionMetadata


class InferRequest(_GraspGenXRequest):
    """Official name-based inference path for a configured gripper asset."""

    action: Literal["infer"] = "infer"
    point_cloud: np.ndarray
    gripper_name: str | None = None
    num_grasps: int = Field(default=200, ge=1)
    grasp_threshold: float = Field(default=-1.0, allow_inf_nan=False)
    topk_num_grasps: int = 100

    @field_validator("point_cloud", mode="before")
    @classmethod
    def _validate_point_cloud(cls, value: object) -> np.ndarray:
        return _object_point_cloud(value)

    @field_validator("grasp_threshold", "topk_num_grasps", mode="before")
    @classmethod
    def _reject_boolean_selection(cls, value: object, info):
        if isinstance(value, (bool, np.bool_)):
            # Pydantic converts ValueError, but not TypeError, into a ValidationError.
            raise ValueError(f"{info.field_name} must not be boolean")  # noqa: TRY004
        return value

    @field_validator("gripper_name")
    @classmethod
    def _validate_gripper_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("gripper_name must not be empty")
        return value

    @field_validator("topk_num_grasps")
    @classmethod
    def _validate_topk(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("topk_num_grasps must be -1 or a positive integer")
        return value


class InferenceTiming(BaseModel):
    infer_ms: float = Field(ge=0)


class InferResponse(BaseResponse):
    grasps: np.ndarray
    confidences: np.ndarray
    gripper_name: str
    timing: InferenceTiming

    @field_validator("grasps", mode="before")
    @classmethod
    def _validate_grasps(cls, value: object) -> np.ndarray:
        return _grasps(value)

    @field_validator("confidences", mode="before")
    @classmethod
    def _validate_confidences(cls, value: object) -> np.ndarray:
        return _confidences(value)

    @model_validator(mode="after")
    def _validate_lengths(self) -> InferResponse:
        if len(self.grasps) != len(self.confidences):
            raise ValueError("grasps and confidences lengths must match")
        return self


class PlannerRequest(_GraspGenXRequest):
    """Planner controls shared by the official sweep-volume actions."""

    action: str
    sweep_volume_params: SweepVolumeParams
    planner: Planner = "graspmoe"
    num_grasps: int = Field(default=200, ge=1)
    moe_num_yaws: int = Field(default=36, ge=1)
    moe_z_offsets_cm: tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, -1.0, 0.0)
    moe_outlier_threshold: float = Field(default=0.014, gt=0)
    moe_outlier_k: int = Field(default=20, ge=1)
    moe_obb_mode: OBBMode = "advanced"
    moe_skip_obb_rule: OBBSkipRule = "auto"
    moe_obb_density: OBBDensity = "sparse"
    moe_obb_position_spacing_cm: float = Field(default=1.0, gt=0)

    @field_validator("sweep_volume_params", mode="before")
    @classmethod
    def _validate_sweep_volume(cls, value: object) -> SweepVolumeParams:
        return SweepVolumeParams.coerce(value)

    @field_validator("moe_z_offsets_cm")
    @classmethod
    def _validate_z_offsets(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or not np.all(np.isfinite(value)):
            raise ValueError("moe_z_offsets_cm must contain finite values")
        return tuple(float(item) for item in value)

    def planner_kwargs(self) -> dict:
        """Arguments passed verbatim to upstream planner functions.

        Native actions deliberately return all candidates. Score thresholding
        and final top-k selection belong to the client.
        """

        return {
            "planner": self.planner,
            "num_grasps": self.num_grasps,
            "moe_num_yaws": self.moe_num_yaws,
            "moe_z_offsets_cm": self.moe_z_offsets_cm,
            "moe_outlier_threshold": self.moe_outlier_threshold,
            "moe_outlier_k": self.moe_outlier_k,
            "moe_obb_mode": self.moe_obb_mode,
            "moe_skip_obb_rule": self.moe_skip_obb_rule,
            "moe_obb_density": self.moe_obb_density,
            "moe_obb_position_spacing_cm": self.moe_obb_position_spacing_cm,
            "grasp_threshold": -1.0,
            "topk_num_grasps": -1 if self.planner == "graspmoe" else self.num_grasps,
        }


class InferObjectRequest(PlannerRequest):
    action: Literal["infer_object"] = "infer_object"
    point_cloud: np.ndarray

    @field_validator("point_cloud", mode="before")
    @classmethod
    def _validate_point_cloud(cls, value: object) -> np.ndarray:
        return _object_point_cloud(value)


class InferObjectResponse(BaseResponse):
    grasps: np.ndarray
    confidences: np.ndarray
    branch_tags: list[BranchTag]
    timing: InferenceTiming

    @field_validator("grasps", mode="before")
    @classmethod
    def _validate_grasps(cls, value: object) -> np.ndarray:
        return _grasps(value)

    @field_validator("confidences", mode="before")
    @classmethod
    def _validate_confidences(cls, value: object) -> np.ndarray:
        return _confidences(value)

    @model_validator(mode="after")
    def _validate_lengths(self) -> InferObjectResponse:
        count = len(self.grasps)
        if len(self.confidences) != count or len(self.branch_tags) != count:
            raise ValueError("grasps, confidences, and branch_tags lengths must match")
        return self


class InferSceneDepthRequest(PlannerRequest):
    action: Literal["infer_scene_depth"] = "infer_scene_depth"
    depth: np.ndarray
    intrinsics: np.ndarray
    instance_mask: np.ndarray
    min_object_points: int = Field(default=100, ge=1)

    @field_validator("depth", mode="before")
    @classmethod
    def _validate_depth(cls, value: object) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 2:
            raise ValueError("depth must have shape (H, W)")
        if not np.issubdtype(raw.dtype, np.floating):
            raise ValueError("depth must use floating-point meters")
        return np.asarray(raw, dtype=np.float32)

    @field_validator("intrinsics", mode="before")
    @classmethod
    def _validate_intrinsics(cls, value: object) -> np.ndarray:
        intrinsics = np.asarray(value, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise ValueError("intrinsics must be a finite (3, 3) matrix")
        return intrinsics

    @field_validator("instance_mask", mode="before")
    @classmethod
    def _validate_mask(cls, value: object) -> np.ndarray:
        mask = np.asarray(value)
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("instance_mask must use an integer dtype")
        return mask.astype(np.int32, copy=False)

    @model_validator(mode="after")
    def _validate_shapes(self) -> InferSceneDepthRequest:
        if self.instance_mask.shape != self.depth.shape:
            raise ValueError("instance_mask shape must match depth")
        return self


class InferScenePointCloudRequest(PlannerRequest):
    action: Literal["infer_scene_pc"] = "infer_scene_pc"
    point_cloud: np.ndarray
    instance_mask: np.ndarray
    min_object_points: int = Field(default=100, ge=1)

    @field_validator("point_cloud", mode="before")
    @classmethod
    def _validate_point_cloud(cls, value: object) -> np.ndarray:
        point_cloud = np.asarray(value, dtype=np.float32)
        if not (
            (point_cloud.ndim == 2 and point_cloud.shape[1] == 3)
            or (point_cloud.ndim == 3 and point_cloud.shape[-1] == 3)
        ):
            raise ValueError("point_cloud must have shape (N, 3) or (H, W, 3)")
        return np.ascontiguousarray(point_cloud)

    @field_validator("instance_mask", mode="before")
    @classmethod
    def _validate_mask(cls, value: object) -> np.ndarray:
        mask = np.asarray(value)
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("instance_mask must use an integer dtype")
        return mask.astype(np.int32, copy=False)

    @model_validator(mode="after")
    def _validate_shapes(self) -> InferScenePointCloudRequest:
        point_count = int(np.prod(self.point_cloud.shape[:-1]))
        if self.instance_mask.size != point_count:
            raise ValueError("instance_mask size must match the number of point-cloud points")
        return self


class InferSceneResponse(BaseResponse):
    instance_ids: np.ndarray
    grasps: list[np.ndarray]
    confidences: list[np.ndarray]
    branch_tags: list[list[BranchTag]]
    skipped_instance_ids: np.ndarray
    timing: InferenceTiming

    @field_validator("instance_ids", "skipped_instance_ids", mode="before")
    @classmethod
    def _validate_ids(cls, value: object) -> np.ndarray:
        ids = np.asarray(value, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError("instance id arrays must have shape (K,)")
        return ids

    @field_validator("grasps", mode="before")
    @classmethod
    def _validate_grasp_lists(cls, value: object) -> list[np.ndarray]:
        return [_grasps(item) for item in value]

    @field_validator("confidences", mode="before")
    @classmethod
    def _validate_confidence_lists(cls, value: object) -> list[np.ndarray]:
        return [_confidences(item) for item in value]

    @model_validator(mode="after")
    def _validate_parallel_lists(self) -> InferSceneResponse:
        count = len(self.instance_ids)
        if not (len(self.grasps) == len(self.confidences) == len(self.branch_tags) == count):
            raise ValueError("scene response parallel-list lengths must match")
        for grasps, confidences, tags in zip(
            self.grasps,
            self.confidences,
            self.branch_tags,
            strict=True,
        ):
            if len(grasps) != len(confidences) or len(grasps) != len(tags):
                raise ValueError("per-instance grasp, confidence, and tag lengths must match")
        return self


__all__ = [
    "ACTIONS",
    "API_VERSION",
    "SERVICE_ID",
    "BranchTag",
    "InferObjectRequest",
    "InferObjectResponse",
    "InferRequest",
    "InferResponse",
    "InferSceneDepthRequest",
    "InferScenePointCloudRequest",
    "InferSceneResponse",
    "InferenceTiming",
    "MetadataRequest",
    "MetadataResponse",
    "ModelMetadata",
    "OBBDensity",
    "OBBMode",
    "OBBSkipRule",
    "Planner",
    "PlannerRequest",
    "PrecisionMetadata",
    "SweepVolumeParams",
]
