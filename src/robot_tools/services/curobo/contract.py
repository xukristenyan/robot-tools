"""Task-level cuRobo API. Metres, radians, robot-base frame, quaternions wxyz.

Joint arrays always follow the explicit joint_names field. Poses target the
configured tool frame, not an implicit grasp/TCP frame. Collision results are
sampled geometric checks, not execution or continuous-time safety guarantees.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "curobo"
API_VERSION = "1"
ACTIONS = frozenset({"get_robot_info", "put_mesh", "put_scene", "solve_ik", "solve_ik_batch", "check_trajectory", "plan_motion", "release_resources"})
Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
OpaqueID = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def floats(value: np.ndarray, name: str) -> np.ndarray:
    if value.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite float32/float64 values")
    return value


def poses(value: np.ndarray) -> np.ndarray:
    floats(value, "pose")
    if value.shape[-1:] != (7,) or not np.allclose(np.linalg.norm(value[..., 3:], axis=-1), 1, atol=1e-4):
        raise ValueError("pose must be [..., 7]: xyz metres, unit quaternion qw,qx,qy,qz")
    return value


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", allow_inf_nan=False)


class Request(BaseRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", allow_inf_nan=False)


class SceneRef(Model):
    scene_id: Name
    version: OpaqueID


class Cuboid(Model):
    kind: Literal["cuboid"] = "cuboid"
    name: Name
    pose: np.ndarray
    dims: tuple[float, float, float]

    @model_validator(mode="after")
    def valid(self):
        if self.pose.shape != (7,):
            raise ValueError("pose must have shape (7,)")
        poses(self.pose)
        if min(self.dims) <= 0:
            raise ValueError("cuboid dimensions must be positive metres")
        return self


class Mesh(Model):
    kind: Literal["mesh"] = "mesh"
    name: Name
    mesh_id: OpaqueID
    pose: np.ndarray

    @field_validator("pose")
    @classmethod
    def valid_pose(cls, value):
        if value.shape != (7,):
            raise ValueError("pose must have shape (7,)")
        return poses(value)


Obstacle = Annotated[Cuboid | Mesh, Field(discriminator="kind")]


class World(Model):
    obstacles: list[Obstacle] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def unique_names(self):
        if len({o.name for o in self.obstacles}) != len(self.obstacles):
            raise ValueError("obstacle names must be unique")
        return self


class AllowedContact(Model):
    """Ignore ONLY this obstacle/link pair, for this entire stage/check."""

    obstacle: Name
    robot_links: list[Name] = Field(min_length=1, max_length=32)


class Attachment(Model):
    """Conservative collision spheres (N,4) in the configured attachment frame.

    The named world object is removed during this stage only. The caller
    supplies a covering sphere approximation and the correct grasp transform.
    """

    object_name: Name
    spheres: np.ndarray

    @field_validator("spheres")
    @classmethod
    def valid_spheres(cls, value):
        floats(value, "spheres")
        if value.ndim != 2 or value.shape[1] != 4 or not 1 <= len(value) <= 64 or (value[:, 3] <= 0).any():
            raise ValueError("spheres must be (1..64, 4) with positive radii")
        return value


class StageContext(Model):
    """Complete stage state; omitted values never carry over from earlier calls.

    obstacle_poses updates objects at release/other known state transitions.
    Each override is relative to the immutable scene, not the previous stage.
    """

    obstacle_poses: dict[Name, np.ndarray] = Field(default_factory=dict)
    attachment: Attachment | None = None
    allowed_contacts: list[AllowedContact] = Field(default_factory=list, max_length=32)

    @field_validator("obstacle_poses")
    @classmethod
    def valid_overrides(cls, values):
        for value in values.values():
            if value.shape != (7,):
                raise ValueError("obstacle pose must have shape (7,)")
            poses(value)
        return values


class CartesianPath(Model):
    """Ordered tool poses (K,7), including the final goal, excluding current FK.

    v0.9.0's default 16-control spline retains at most 72 targets. Positions
    interpolate linearly; orientations use SLERP. Tolerances are metres/radians.
    """

    poses: np.ndarray
    path_tolerance: tuple[float, float] = (0.02, 0.174533)
    final_tolerance: tuple[float, float] = (0.005, 0.05)
    track_orientation: bool = True

    @model_validator(mode="after")
    def valid_path(self):
        if self.poses.ndim != 2 or not 1 <= len(self.poses) <= 72:
            raise ValueError("Cartesian poses must be (1..72, 7)")
        poses(self.poses)
        for tolerance in (self.path_tolerance, self.final_tolerance):
            if min(tolerance) <= 0:
                raise ValueError("Cartesian tolerances must be positive metres/radians")
        return self


class MotionStage(StageContext):
    name: Name
    goal_pose: np.ndarray
    cartesian_path: CartesianPath | None = None

    @model_serializer(mode="wrap")
    def serialize_stage(self, handler):
        # Ordinary pose requests must still work with an older API-v1 server.
        result = handler(self)
        if self.cartesian_path is None:
            result.pop("cartesian_path", None)
        return result

    @field_validator("goal_pose")
    @classmethod
    def valid_goal(cls, value):
        if value.shape != (7,):
            raise ValueError("goal_pose must have shape (7,)")
        return poses(value)

    @model_validator(mode="after")
    def valid_path_goal(self):
        if self.cartesian_path is not None:
            last = self.cartesian_path.poses[-1]
            same_position = np.allclose(last[:3], self.goal_pose[:3], atol=1e-6, rtol=0)
            same_rotation = np.isclose(abs(np.dot(last[3:], self.goal_pose[3:])), 1, atol=1e-4)
            if not same_position or (self.cartesian_path.track_orientation and not same_rotation):
                raise ValueError("Cartesian path must end at goal_pose")
        return self


class GetRobotInfoRequest(Request):
    action: Literal["get_robot_info"] = "get_robot_info"


class ReleaseResourcesRequest(Request):
    """Release exact scene versions and uploaded meshes owned by a caller."""

    action: Literal["release_resources"] = "release_resources"
    scenes: list[SceneRef] = Field(default_factory=list)
    mesh_ids: list[OpaqueID] = Field(default_factory=list)


class ReleaseResourcesResponse(BaseResponse):
    released_scenes: int
    released_meshes: int
    deferred_meshes: int
    cached_meshes: int


class GetRobotInfoResponse(BaseResponse):
    robot_config: str
    base_frame: str
    tool_frame: str
    joint_names: list[str]
    joint_limits: np.ndarray  # (2, D), lower/upper radians
    default_position: np.ndarray  # (D,)
    collision_links: list[str]
    attachment_frame: str | None
    collision_margin_m: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    fake: bool
    upstream_commit: str


class PutMeshRequest(Request):
    action: Literal["put_mesh"] = "put_mesh"
    vertices: np.ndarray  # (V,3), metres, mesh-local frame
    faces: np.ndarray  # (F,3), integer triangle indices; closed meshes recommended

    @model_validator(mode="after")
    def valid_mesh(self):
        floats(self.vertices, "vertices")
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3:
            raise ValueError("vertices must have shape (V, 3), V >= 3")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or len(self.faces) < 1:
            raise ValueError("faces must have shape (F, 3), F >= 1")
        if (
            not np.issubdtype(self.faces.dtype, np.integer)
            or self.faces.min() < 0
            or self.faces.max() >= len(self.vertices)
        ):
            raise ValueError("faces must contain valid integer vertex indices")
        return self


class PutMeshResponse(BaseResponse):
    mesh_id: OpaqueID


class PutSceneRequest(Request):
    action: Literal["put_scene"] = "put_scene"
    scene_id: Name
    world: World


class PutSceneResponse(BaseResponse):
    scene: SceneRef | None = None
    failure_code: str | None = None


class JointRequest(Request):
    scene: SceneRef
    joint_names: list[Name] = Field(min_length=1, max_length=64)

    @field_validator("joint_names")
    @classmethod
    def unique_joints(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("joint_names must be unique")
        return value


class SolveIKRequest(JointRequest):
    action: Literal["solve_ik"] = "solve_ik"
    goal_poses: np.ndarray  # (N,7)
    seed_position: np.ndarray  # (D,)
    collision_check: bool = True
    context: StageContext = Field(default_factory=StageContext)

    @model_validator(mode="after")
    def valid_ik(self):
        if self.goal_poses.ndim != 2 or not 1 <= len(self.goal_poses) <= 256:
            raise ValueError("goal_poses must be (1..256, 7)")
        poses(self.goal_poses)
        if self.seed_position.shape != (len(self.joint_names),):
            raise ValueError("seed_position must match joint_names")
        floats(self.seed_position, "seed_position")
        return self


class SolveIKResponse(BaseResponse):
    success: np.ndarray  # (N,) bool; failed positions are zeros
    positions: np.ndarray  # (N,D) float32, requested joint order
    position_error_m: np.ndarray  # (N,) float32; -1 means unavailable
    rotation_error_rad: np.ndarray  # (N,) float32; -1 means unavailable
    collision_checked: bool
    failure_code: str | None = None
    fake: bool = False

    @model_validator(mode="after")
    def valid_result(self):
        n = len(self.success)
        if self.success.shape != (n,) or self.success.dtype != np.bool_:
            raise ValueError("success must be a bool vector")
        if self.positions.ndim != 2 or self.positions.shape[0] != n:
            raise ValueError("positions must align with success")
        for value in (self.positions, self.position_error_m, self.rotation_error_rad):
            floats(value, "IK result")
        if self.position_error_m.shape != (n,) or self.rotation_error_rad.shape != (n,):
            raise ValueError("errors must align with success")
        return self


class SolveIKBatchRequest(Request):
    action: Literal["solve_ik_batch"] = "solve_ik_batch"
    requests: list[SolveIKRequest] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def one_goal_per_candidate(self):
        if any(len(request.goal_poses) != 1 for request in self.requests):
            raise ValueError("Batch IK requires exactly one goal per candidate")
        return self


class SolveIKBatchResponse(BaseResponse):
    results: list[SolveIKResponse]
    group_sizes: list[int]


class CheckTrajectoryRequest(JointRequest):
    action: Literal["check_trajectory"] = "check_trajectory"
    positions: np.ndarray  # (T,D); adjacent positions are densified
    context: StageContext = Field(default_factory=StageContext)
    max_joint_step_rad: float = Field(default=0.02, gt=0, le=0.1)

    @model_validator(mode="after")
    def valid_trajectory(self):
        floats(self.positions, "positions")
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != len(self.joint_names)
            or not 1 <= len(self.positions) <= 20000
        ):
            raise ValueError("positions must be (1..20000, len(joint_names))")
        return self


class CheckTrajectoryResponse(BaseResponse):
    valid: bool
    sampled_positions: np.ndarray  # (S,D), includes original knots and both endpoints
    world_collision: np.ndarray  # (S,) bool
    self_collision: np.ndarray  # (S,) bool
    joint_limit_violation: np.ndarray  # (S,) bool
    first_invalid_index: int | None = None
    failure_code: str | None = None
    fake: bool = False

    @model_validator(mode="after")
    def valid_result(self):
        floats(self.sampled_positions, "sampled_positions")
        if self.sampled_positions.ndim != 2:
            raise ValueError("sampled_positions must be a matrix")
        n = len(self.sampled_positions)
        for mask in (self.world_collision, self.self_collision, self.joint_limit_violation):
            if mask.shape != (n,) or mask.dtype != np.bool_:
                raise ValueError("collision masks must align with sampled positions")
        bad = self.world_collision | self.self_collision | self.joint_limit_violation
        first = int(np.flatnonzero(bad)[0]) if bad.any() else None
        if self.first_invalid_index != first or self.valid != (n > 0 and first is None and self.failure_code is None):
            raise ValueError("valid/first_invalid_index must agree with masks and failure_code")
        return self


class PlanMotionRequest(JointRequest):
    action: Literal["plan_motion"] = "plan_motion"
    start_position: np.ndarray
    stages: list[MotionStage] = Field(min_length=1, max_length=16)
    mode: Literal["collision_aware", "ik_only"] = "collision_aware"
    max_attempts: int = Field(default=3, ge=1, le=10, strict=True)
    max_joint_step_rad: float = Field(default=0.02, gt=0, le=0.1)

    @model_validator(mode="after")
    def valid_plan(self):
        floats(self.start_position, "start_position")
        if self.start_position.shape != (len(self.joint_names),):
            raise ValueError("start_position must match joint_names")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        if self.mode == "ik_only" and any(stage.cartesian_path is not None for stage in self.stages):
            raise ValueError("Cartesian paths require collision_aware planning")
        return self


class StageResult(Model):
    name: str
    success: bool
    positions: np.ndarray  # (T,D); failed stages have no trajectory
    dt: float | None = Field(default=None, gt=0)  # None for untimed IK interpolation
    collision_checked: bool
    failure_code: str | None = None
    timing_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def valid_result(self):
        floats(self.positions, "positions")
        if self.positions.ndim != 2 or self.success != (len(self.positions) > 0 and self.failure_code is None):
            raise ValueError("stage success must agree with positions and failure_code")
        return self


class PlanMotionResponse(BaseResponse):
    success: bool
    stages: list[StageResult]
    failed_stage: str | None = None
    failure_code: str | None = None
    fake: bool = False

    @model_validator(mode="after")
    def valid_result(self):
        if self.success and (not self.stages or not all(s.success for s in self.stages) or self.failure_code):
            raise ValueError("successful plans require successful stages")
        if self.success and self.failed_stage is not None:
            raise ValueError("successful plan cannot have a failed_stage")
        return self
