from __future__ import annotations

from typing import ClassVar

import numpy as np

from robot_tools.core.client import BaseClient
from robot_tools.services.graspgen.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    GenerateCollisionFreeGraspsRequest,
    GenerateCollisionFreeGraspsResponse,
    GenerateGraspsRequest,
    GenerateGraspsResponse,
)


class GraspGenClient(BaseClient):
    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {
        SERVICE_ID: frozenset({API_VERSION}),
    }
    REQUIRED_ACTIONS = ACTIONS

    def generate_grasps(
        self,
        point_cloud: np.ndarray,
        *,
        grasp_threshold: float = 0.8,
        num_grasps: int = 200,
        topk: int = -1,
    ) -> GenerateGraspsResponse:
        req = GenerateGraspsRequest(
            point_cloud=np.asarray(point_cloud, dtype=np.float32),
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk=topk,
        )
        return self._request(req, GenerateGraspsResponse)

    def generate_collision_free_grasps(
        self,
        point_cloud: np.ndarray,
        scene_pc: np.ndarray,
        *,
        grasp_threshold: float = 0.8,
        num_grasps: int = 200,
        topk: int = -1,
        collision_threshold: float = 0.02,
        max_scene_points: int = 8192,
    ) -> GenerateCollisionFreeGraspsResponse:
        req = GenerateCollisionFreeGraspsRequest(
            point_cloud=np.asarray(point_cloud, dtype=np.float32),
            scene_pc=np.asarray(scene_pc, dtype=np.float32),
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk=topk,
            collision_threshold=collision_threshold,
            max_scene_points=max_scene_points,
        )
        return self._request(req, GenerateCollisionFreeGraspsResponse)
