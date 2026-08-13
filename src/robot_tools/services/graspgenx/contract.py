"""Typed contracts for the robot-tools GraspGenX service."""


from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "graspgenx"
API_VERSION = "4"
ACTIONS = frozenset(
    {
        "generate_grasps",
        "generate_safe_grasps",
        "generate_grasps_for_all",
        "generate_safe_grasps_for_all",
        "get_metadata",
    }
)

Planner = Literal["graspmoe", "diffusion"]
BranchTag = Literal["diff", "obb"]
OBBMode = Literal["advanced", "pca"]
OBBSkipRule = Literal["auto", "never"]
OBBDensity = Literal["sparse", "dense", "dense-topandside"]

def _point_cloud(value: object, *, name: str, allow_empty: bool = False) -> np.ndarray:
    point_cloud = np.asarray(value, dtype=np.float32)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not allow_empty and len(point_cloud) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(point_cloud)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(point_cloud)


def _object_point_cloud(value: object) -> np.ndarray:
    return _point_cloud(value, name="point_cloud")


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


class _GraspGenXRequest(BaseRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class GetMetadataRequest(_GraspGenXRequest):
    action: Literal["get_metadata"] = "get_metadata"


class ModelMetadata(BaseModel):
    generator_backbone: str
    discriminator_backbone: str
    grasp_repr: str
    num_diffusion_iters_eval: int


class PrecisionMetadata(BaseModel):
    weights: str = "fp32"
    tensorrt: bool = False
    tensorrt_precision: Literal["fp32", "fp16"] | None = None


class GetMetadataResponse(BaseResponse):
    default_gripper: str | None
    loaded_grippers: list[str]
    model: ModelMetadata
    assets_dir: str
    actions: list[str]
    precision: PrecisionMetadata


class GenerateGraspsRequest(_GraspGenXRequest):
    """Official name-based inference path for a configured gripper asset."""

    action: Literal["generate_grasps"] = "generate_grasps"
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


class GenerateGraspsResponse(BaseResponse):
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
    def _validate_lengths(self) -> GenerateGraspsResponse:
        if len(self.grasps) != len(self.confidences):
            raise ValueError("grasps and confidences lengths must match")
        return self


class PlannerRequest(_GraspGenXRequest):
    """Planner controls shared by scene grasp actions."""

    action: str
    gripper_name: str | None = None
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

    @field_validator("gripper_name")
    @classmethod
    def _validate_gripper_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("gripper_name must not be empty")
        return value

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


class _SceneDepthRequest(PlannerRequest):
    depth: np.ndarray
    intrinsics: np.ndarray
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


class GenerateGraspsForAllRequest(_SceneDepthRequest):
    action: Literal["generate_grasps_for_all"] = "generate_grasps_for_all"
    instance_mask: np.ndarray

    @field_validator("instance_mask", mode="before")
    @classmethod
    def _validate_mask(cls, value: object) -> np.ndarray:
        mask = np.asarray(value)
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("instance_mask must use an integer dtype")
        return mask.astype(np.int32, copy=False)

    @model_validator(mode="after")
    def _validate_shapes(self) -> GenerateGraspsForAllRequest:
        if self.instance_mask.shape != self.depth.shape:
            raise ValueError("instance_mask shape must match depth")
        return self


class GenerateSafeGraspsRequest(PlannerRequest):
    action: Literal["generate_safe_grasps"] = "generate_safe_grasps"
    object_point_cloud: np.ndarray
    scene_point_cloud: np.ndarray
    collision_threshold: float = Field(default=0.02, gt=0)

    @field_validator("object_point_cloud", mode="before")
    @classmethod
    def _validate_object_point_cloud(cls, value: object) -> np.ndarray:
        return _point_cloud(value, name="object_point_cloud")

    @field_validator("scene_point_cloud", mode="before")
    @classmethod
    def _validate_scene_point_cloud(cls, value: object) -> np.ndarray:
        return _point_cloud(value, name="scene_point_cloud", allow_empty=True)


class GenerateSafeGraspsForAllRequest(GenerateGraspsForAllRequest):
    action: Literal["generate_safe_grasps_for_all"] = "generate_safe_grasps_for_all"
    collision_threshold: float = Field(default=0.02, gt=0)


class GenerateGraspsForAllResponse(BaseResponse):
    instance_ids: np.ndarray
    grasps: list[np.ndarray]
    confidences: list[np.ndarray]
    branch_tags: list[list[BranchTag]]
    skipped_instance_ids: np.ndarray
    gripper_name: str
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
    def _validate_parallel_lists(self) -> GenerateGraspsForAllResponse:
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
    "GenerateGraspsForAllRequest",
    "GenerateGraspsForAllResponse",
    "GenerateGraspsRequest",
    "GenerateGraspsResponse",
    "GenerateSafeGraspsForAllRequest",
    "GenerateSafeGraspsRequest",
    "GetMetadataRequest",
    "GetMetadataResponse",
    "InferenceTiming",
    "ModelMetadata",
    "OBBDensity",
    "OBBMode",
    "OBBSkipRule",
    "Planner",
    "PlannerRequest",
    "PrecisionMetadata",
]
