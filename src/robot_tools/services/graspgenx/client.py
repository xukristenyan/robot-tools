"""Synchronous typed client for the robot-tools GraspGenX API."""

from __future__ import annotations

import operator
from typing import ClassVar

import numpy as np

from robot_tools.core.client import BaseClient
from robot_tools.services.graspgenx.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    BranchTag,
    GenerateGraspsForAllRequest,
    GenerateGraspsForAllResponse,
    GenerateGraspsRequest,
    GenerateGraspsResponse,
    GenerateSafeGraspsForAllRequest,
    GenerateSafeGraspsRequest,
    GetMetadataRequest,
    GetMetadataResponse,
    OBBDensity,
    OBBMode,
    OBBSkipRule,
    Planner,
)


class GraspGenXClient(BaseClient):
    """Drive the GraspGenX service without importing Torch or upstream code."""

    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS

    def __init__(self, *args, **kwargs):
        self._metadata_cache: GetMetadataResponse | None = None
        super().__init__(*args, **kwargs)

    def get_metadata(self, *, refresh: bool = False) -> GetMetadataResponse:
        if refresh or self._metadata_cache is None:
            self._metadata_cache = self._request(GetMetadataRequest(), GetMetadataResponse)
        return self._metadata_cache

    @property
    def server_metadata(self) -> GetMetadataResponse:
        return self.get_metadata()

    def generate_grasps(
        self,
        point_cloud: np.ndarray,
        *,
        gripper_name: str | None = None,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
    ) -> GenerateGraspsResponse:
        """Generate grasps for one segmented object point cloud."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = GenerateGraspsRequest(
            point_cloud=point_cloud,
            gripper_name=gripper_name,
            num_grasps=num_grasps,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )
        return self._request(request, GenerateGraspsResponse)

    def generate_safe_grasps(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        target_mask: np.ndarray,
        *,
        gripper_name: str | None = None,
        planner: Planner = "graspmoe",
        min_object_points: int = 100,
        collision_threshold: float = 0.02,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        moe_num_yaws: int = 36,
        moe_z_offsets_cm: tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, -1.0, 0.0),
        moe_outlier_threshold: float = 0.014,
        moe_outlier_k: int = 20,
        moe_obb_mode: OBBMode = "advanced",
        moe_skip_obb_rule: OBBSkipRule = "auto",
        moe_obb_density: OBBDensity = "sparse",
        moe_obb_position_spacing_cm: float = 1.0,
    ) -> GenerateGraspsResponse:
        """Generate grasps for one mask and remove collisions with the rest of the scene."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = GenerateSafeGraspsRequest(
            depth=depth,
            intrinsics=intrinsics,
            target_mask=target_mask,
            gripper_name=gripper_name,
            min_object_points=min_object_points,
            collision_threshold=collision_threshold,
            **_planner_fields(
                planner=planner,
                num_grasps=num_grasps,
                moe_num_yaws=moe_num_yaws,
                moe_z_offsets_cm=moe_z_offsets_cm,
                moe_outlier_threshold=moe_outlier_threshold,
                moe_outlier_k=moe_outlier_k,
                moe_obb_mode=moe_obb_mode,
                moe_skip_obb_rule=moe_skip_obb_rule,
                moe_obb_density=moe_obb_density,
                moe_obb_position_spacing_cm=moe_obb_position_spacing_cm,
            ),
        )
        response = self._request(request, GenerateGraspsResponse)
        grasps, confidences = _apply_selection(
            response.grasps,
            response.confidences,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )
        return GenerateGraspsResponse(
            grasps=grasps,
            confidences=confidences,
            gripper_name=response.gripper_name,
            timing=response.timing,
            timing_ms=response.timing_ms,
        )

    def generate_grasps_for_all(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        *,
        gripper_name: str | None = None,
        planner: Planner = "graspmoe",
        min_object_points: int = 100,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        moe_num_yaws: int = 36,
        moe_z_offsets_cm: tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, -1.0, 0.0),
        moe_outlier_threshold: float = 0.014,
        moe_outlier_k: int = 20,
        moe_obb_mode: OBBMode = "advanced",
        moe_skip_obb_rule: OBBSkipRule = "auto",
        moe_obb_density: OBBDensity = "sparse",
        moe_obb_position_spacing_cm: float = 1.0,
    ) -> GenerateGraspsForAllResponse:
        """Generate per-instance grasps from metric depth and an integer mask."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = GenerateGraspsForAllRequest(
            depth=depth,
            intrinsics=intrinsics,
            instance_mask=instance_mask,
            gripper_name=gripper_name,
            min_object_points=min_object_points,
            **_planner_fields(
                planner=planner,
                num_grasps=num_grasps,
                moe_num_yaws=moe_num_yaws,
                moe_z_offsets_cm=moe_z_offsets_cm,
                moe_outlier_threshold=moe_outlier_threshold,
                moe_outlier_k=moe_outlier_k,
                moe_obb_mode=moe_obb_mode,
                moe_skip_obb_rule=moe_skip_obb_rule,
                moe_obb_density=moe_obb_density,
                moe_obb_position_spacing_cm=moe_obb_position_spacing_cm,
            ),
        )
        response = self._request(request, GenerateGraspsForAllResponse)
        return _filter_scene_response(
            response,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )

    def generate_safe_grasps_for_all(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        *,
        gripper_name: str | None = None,
        planner: Planner = "graspmoe",
        min_object_points: int = 100,
        collision_threshold: float = 0.02,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        moe_num_yaws: int = 36,
        moe_z_offsets_cm: tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, -1.0, 0.0),
        moe_outlier_threshold: float = 0.014,
        moe_outlier_k: int = 20,
        moe_obb_mode: OBBMode = "advanced",
        moe_skip_obb_rule: OBBSkipRule = "auto",
        moe_obb_density: OBBDensity = "sparse",
        moe_obb_position_spacing_cm: float = 1.0,
    ) -> GenerateGraspsForAllResponse:
        """Generate per-instance grasps and remove collisions with each object's surroundings."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = GenerateSafeGraspsForAllRequest(
            depth=depth,
            intrinsics=intrinsics,
            instance_mask=instance_mask,
            gripper_name=gripper_name,
            min_object_points=min_object_points,
            collision_threshold=collision_threshold,
            **_planner_fields(
                planner=planner,
                num_grasps=num_grasps,
                moe_num_yaws=moe_num_yaws,
                moe_z_offsets_cm=moe_z_offsets_cm,
                moe_outlier_threshold=moe_outlier_threshold,
                moe_outlier_k=moe_outlier_k,
                moe_obb_mode=moe_obb_mode,
                moe_skip_obb_rule=moe_skip_obb_rule,
                moe_obb_density=moe_obb_density,
                moe_obb_position_spacing_cm=moe_obb_position_spacing_cm,
            ),
        )
        response = self._request(request, GenerateGraspsForAllResponse)
        return _filter_scene_response(
            response,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )

    @staticmethod
    def scene_results_by_id(response: GenerateGraspsForAllResponse) -> dict[int, tuple]:
        """Convert the wire-safe parallel lists to an instance-keyed mapping."""

        return {
            int(instance_id): (grasps, confidences, tags)
            for instance_id, grasps, confidences, tags in zip(
                response.instance_ids,
                response.grasps,
                response.confidences,
                response.branch_tags,
                strict=True,
            )
        }


def _planner_fields(
    *,
    planner: Planner,
    num_grasps: int,
    moe_num_yaws: int,
    moe_z_offsets_cm: tuple[float, ...],
    moe_outlier_threshold: float,
    moe_outlier_k: int,
    moe_obb_mode: OBBMode,
    moe_skip_obb_rule: OBBSkipRule,
    moe_obb_density: OBBDensity,
    moe_obb_position_spacing_cm: float,
) -> dict:
    return {
        "planner": planner,
        "num_grasps": num_grasps,
        "moe_num_yaws": moe_num_yaws,
        "moe_z_offsets_cm": moe_z_offsets_cm,
        "moe_outlier_threshold": moe_outlier_threshold,
        "moe_outlier_k": moe_outlier_k,
        "moe_obb_mode": moe_obb_mode,
        "moe_skip_obb_rule": moe_skip_obb_rule,
        "moe_obb_density": moe_obb_density,
        "moe_obb_position_spacing_cm": moe_obb_position_spacing_cm,
    }


def _validate_selection(grasp_threshold: float, topk_num_grasps: int) -> tuple[float, int]:
    if isinstance(grasp_threshold, (bool, np.bool_, str, bytes)):
        raise TypeError("grasp_threshold must be a finite number")
    try:
        normalized_threshold = float(grasp_threshold)
    except TypeError as exc:
        raise TypeError("grasp_threshold must be a finite number") from exc
    except ValueError as exc:
        raise ValueError("grasp_threshold must be a finite number") from exc
    if not np.isfinite(normalized_threshold):
        raise ValueError("grasp_threshold must be finite")

    if isinstance(topk_num_grasps, (bool, np.bool_)):
        raise TypeError("topk_num_grasps must be -1 or a positive integer")
    try:
        normalized_topk = operator.index(topk_num_grasps)
    except TypeError as exc:
        raise TypeError("topk_num_grasps must be -1 or a positive integer") from exc
    if normalized_topk == 0 or normalized_topk < -1:
        raise ValueError("topk_num_grasps must be -1 or a positive integer")
    return normalized_threshold, normalized_topk


def _selection_indices(
    confidences: np.ndarray,
    *,
    grasp_threshold: float,
    topk_num_grasps: int,
) -> np.ndarray:
    grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
    indices = np.arange(len(confidences))
    if grasp_threshold > 0.0:
        indices = indices[confidences >= grasp_threshold]
    if topk_num_grasps > 0:
        order = np.argsort(-confidences[indices], kind="stable")[:topk_num_grasps]
        indices = indices[order]
    return indices


def _apply_selection(
    grasps: np.ndarray,
    confidences: np.ndarray,
    *,
    grasp_threshold: float,
    topk_num_grasps: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = _selection_indices(
        confidences,
        grasp_threshold=grasp_threshold,
        topk_num_grasps=topk_num_grasps,
    )
    return grasps[indices], confidences[indices]


def _apply_selection_with_tags(
    grasps: np.ndarray,
    confidences: np.ndarray,
    branch_tags: list[BranchTag],
    *,
    grasp_threshold: float,
    topk_num_grasps: int,
) -> tuple[np.ndarray, np.ndarray, list[BranchTag]]:
    indices = _selection_indices(
        confidences,
        grasp_threshold=grasp_threshold,
        topk_num_grasps=topk_num_grasps,
    )
    return grasps[indices], confidences[indices], [branch_tags[int(index)] for index in indices]


def _filter_scene_response(
    response: GenerateGraspsForAllResponse,
    *,
    grasp_threshold: float,
    topk_num_grasps: int,
) -> GenerateGraspsForAllResponse:
    grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
    instance_ids: list[int] = []
    grasps_list: list[np.ndarray] = []
    confidence_list: list[np.ndarray] = []
    tag_list: list[list[BranchTag]] = []
    skipped = response.skipped_instance_ids.tolist()

    for instance_id, grasps, confidences, tags in zip(
        response.instance_ids,
        response.grasps,
        response.confidences,
        response.branch_tags,
        strict=True,
    ):
        selected_grasps, selected_confidences, selected_tags = _apply_selection_with_tags(
            grasps,
            confidences,
            tags,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )
        if len(selected_grasps) == 0:
            skipped.append(int(instance_id))
            continue
        instance_ids.append(int(instance_id))
        grasps_list.append(selected_grasps)
        confidence_list.append(selected_confidences)
        tag_list.append(selected_tags)

    return GenerateGraspsForAllResponse(
        instance_ids=np.asarray(instance_ids, dtype=np.int32),
        grasps=grasps_list,
        confidences=confidence_list,
        branch_tags=tag_list,
        skipped_instance_ids=np.asarray(sorted(set(skipped)), dtype=np.int32),
        gripper_name=response.gripper_name,
        timing=response.timing,
        timing_ms=response.timing_ms,
    )


__all__ = ["GraspGenXClient"]
