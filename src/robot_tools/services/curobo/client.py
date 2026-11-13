"""Lightweight, synchronous cuRobo client; no Torch/cuRobo imports."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from robot_tools.core.client import BaseClient
from robot_tools.services.curobo.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    CheckTrajectoryRequest,
    CheckTrajectoryResponse,
    GetRobotInfoRequest,
    GetRobotInfoResponse,
    MotionStage,
    PlanMotionRequest,
    PlanMotionResponse,
    PutMeshRequest,
    PutMeshResponse,
    PutSceneRequest,
    PutSceneResponse,
    SceneRef,
    SolveIKRequest,
    SolveIKBatchRequest,
    SolveIKBatchResponse,
    SolveIKResponse,
    StageContext,
    World,
    ReleaseResourcesRequest,
    ReleaseResourcesResponse,
)


class CuroboClient(BaseClient):
    SUPPORTED_SERVICE_APIS: ClassVar = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS - {"solve_ik_batch", "release_resources"}

    def __init__(self, *args, cleanup_on_close=False, **kwargs):
        self._cleanup_on_close = cleanup_on_close
        self._owned_mesh_ids = []
        self._owned_scenes = []
        self._closed = False
        self.cleanup_result = None
        if cleanup_on_close:
            self.REQUIRED_ACTIONS = self.REQUIRED_ACTIONS | {"release_resources"}
        super().__init__(*args, **kwargs)

    def release_resources(self) -> ReleaseResourcesResponse:
        result = self._request(ReleaseResourcesRequest(
            scenes=self._owned_scenes, mesh_ids=self._owned_mesh_ids), ReleaseResourcesResponse)
        self._owned_scenes.clear()
        self._owned_mesh_ids.clear()
        self.cleanup_result = result
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._cleanup_on_close and (self._owned_scenes or self._owned_mesh_ids):
                self.release_resources()
        finally:
            self._closed = True
            super().close()

    def get_robot_info(self) -> GetRobotInfoResponse:
        return self._request(GetRobotInfoRequest(), GetRobotInfoResponse)

    def put_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> PutMeshResponse:
        result = self._request(PutMeshRequest(vertices=np.asarray(vertices), faces=np.asarray(faces)), PutMeshResponse)
        self._owned_mesh_ids.append(result.mesh_id)
        return result

    def put_scene(self, scene_id: str, world: World) -> PutSceneResponse:
        result = self._request(PutSceneRequest(scene_id=scene_id, world=world), PutSceneResponse)
        if result.scene is not None:
            self._owned_scenes.append(result.scene)
        return result

    def solve_ik(
        self,
        goal_poses: np.ndarray,
        seed_position: np.ndarray,
        *,
        scene: SceneRef,
        joint_names: list[str],
        collision_check: bool = True,
        context: StageContext | None = None,
    ) -> SolveIKResponse:
        request = SolveIKRequest(
            goal_poses=np.atleast_2d(np.asarray(goal_poses, dtype=np.float32)),
            seed_position=np.asarray(seed_position, dtype=np.float32),
            scene=scene,
            joint_names=joint_names,
            collision_check=collision_check,
            **({"context": context} if context is not None else {}),
        )
        return self._request(request, SolveIKResponse)

    def solve_ik_batch(self, requests: list[SolveIKRequest]) -> SolveIKBatchResponse:
        return self._request(SolveIKBatchRequest(requests=requests), SolveIKBatchResponse)

    def check_trajectory(
        self,
        positions: np.ndarray,
        *,
        scene: SceneRef,
        joint_names: list[str],
        context: StageContext | None = None,
        max_joint_step_rad: float = 0.02,
    ) -> CheckTrajectoryResponse:
        request = CheckTrajectoryRequest(
            positions=np.asarray(positions, dtype=np.float32),
            scene=scene,
            joint_names=joint_names,
            max_joint_step_rad=max_joint_step_rad,
            **({"context": context} if context is not None else {}),
        )
        return self._request(request, CheckTrajectoryResponse)

    def plan_motion(
        self,
        start_position: np.ndarray,
        stages: list[MotionStage],
        *,
        scene: SceneRef,
        joint_names: list[str],
        mode: Literal["collision_aware", "ik_only"] = "collision_aware",
        max_attempts: int = 3,
        max_joint_step_rad: float = 0.02,
    ) -> PlanMotionResponse:
        request = PlanMotionRequest(
            start_position=np.asarray(start_position, dtype=np.float32),
            stages=stages,
            scene=scene,
            joint_names=joint_names,
            mode=mode,
            max_attempts=max_attempts,
            max_joint_step_rad=max_joint_step_rad,
        )
        return self._request(request, PlanMotionResponse)
