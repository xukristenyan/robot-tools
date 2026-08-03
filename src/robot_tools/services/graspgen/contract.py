from __future__ import annotations

from typing import Literal

import numpy as np

from robot_tools.core.wire import BaseRequest, BaseResponse

SERVICE_ID = "graspgen"
API_VERSION = "1"
ACTIONS = frozenset({"generate_grasps", "generate_collision_free_grasps"})


class GenerateGraspsRequest(BaseRequest):
    action: Literal["generate_grasps"] = "generate_grasps"
    point_cloud: np.ndarray  # (N, 3) float32, object points
    grasp_threshold: float = 0.8
    num_grasps: int = 200
    topk: int = -1


class GenerateGraspsResponse(BaseResponse):
    poses: np.ndarray  # (K, 4, 4) float32
    confidences: np.ndarray  # (K,) float32


class GenerateCollisionFreeGraspsRequest(BaseRequest):
    action: Literal["generate_collision_free_grasps"] = "generate_collision_free_grasps"
    point_cloud: np.ndarray  # (N, 3) object
    scene_pc: np.ndarray  # (M, 3) full scene
    grasp_threshold: float = 0.8
    num_grasps: int = 200
    topk: int = -1
    collision_threshold: float = 0.02
    max_scene_points: int = 8192


class GenerateCollisionFreeGraspsResponse(BaseResponse):
    poses: np.ndarray  # (K, 4, 4)
    confidences: np.ndarray  # (K,)
    collision_free_mask: np.ndarray  # (K,) bool
