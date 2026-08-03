"""Synchronous typed client for the native GraspGenX API."""

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
    InferObjectRequest,
    InferObjectResponse,
    InferRequest,
    InferResponse,
    InferSceneDepthRequest,
    InferScenePointCloudRequest,
    InferSceneResponse,
    MetadataRequest,
    MetadataResponse,
    OBBDensity,
    OBBMode,
    OBBSkipRule,
    Planner,
    SweepVolumeParams,
)


class GraspGenXClient(BaseClient):
    """Drive GraspGenX without importing Torch or the upstream package.

    The API mirrors the official GraspGenX serving modes while retaining the
    robot-tools convention of returning complete Pydantic response objects.
    Native sweep-volume modes apply score thresholding and final top-k locally,
    exactly as the official lightweight client does.
    """

    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS

    def __init__(self, *args, **kwargs):
        self._metadata_cache: MetadataResponse | None = None
        super().__init__(*args, **kwargs)

    def metadata(self, *, refresh: bool = False) -> MetadataResponse:
        if refresh or self._metadata_cache is None:
            self._metadata_cache = self._request(MetadataRequest(), MetadataResponse)
        return self._metadata_cache

    @property
    def server_metadata(self) -> MetadataResponse:
        return self.metadata()

    def infer(
        self,
        point_cloud: np.ndarray,
        *,
        gripper_name: str | None = None,
        num_grasps: int = 200,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
    ) -> InferResponse:
        """Name-based inference using a gripper from official assets."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = InferRequest(
            point_cloud=point_cloud,
            gripper_name=gripper_name,
            num_grasps=num_grasps,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )
        return self._request(request, InferResponse)

    def infer_object(
        self,
        point_cloud: np.ndarray,
        sweep_volume_params: SweepVolumeParams | dict | np.ndarray | list | tuple,
        *,
        planner: Planner = "graspmoe",
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
    ) -> InferObjectResponse:
        """Plan grasps for one segmented object point cloud.

        The HTTP service returns every candidate. ``grasp_threshold`` and
        ``topk_num_grasps`` are intentionally applied in this client process.
        """

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = InferObjectRequest(
            point_cloud=point_cloud,
            sweep_volume_params=sweep_volume_params,
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
        response = self._request(request, InferObjectResponse)
        grasps, confidences, tags = _apply_threshold_topk(
            response.grasps,
            response.confidences,
            response.branch_tags,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )
        return InferObjectResponse(
            grasps=grasps,
            confidences=confidences,
            branch_tags=tags,
            timing=response.timing,
            timing_ms=response.timing_ms,
        )

    def infer_scene_depth(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        sweep_volume_params: SweepVolumeParams | dict | np.ndarray | list | tuple,
        *,
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
    ) -> InferSceneResponse:
        """Plan per-instance grasps from metric depth in the camera frame."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = InferSceneDepthRequest(
            depth=depth,
            intrinsics=intrinsics,
            instance_mask=instance_mask,
            sweep_volume_params=sweep_volume_params,
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
        response = self._request(request, InferSceneResponse)
        return _filter_scene_response(
            response,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )

    def infer_scene_pc(
        self,
        point_cloud: np.ndarray,
        instance_mask: np.ndarray,
        sweep_volume_params: SweepVolumeParams | dict | np.ndarray | list | tuple,
        *,
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
    ) -> InferSceneResponse:
        """Plan per-instance grasps in the input point-cloud frame."""

        grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
        request = InferScenePointCloudRequest(
            point_cloud=point_cloud,
            instance_mask=instance_mask,
            sweep_volume_params=sweep_volume_params,
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
        response = self._request(request, InferSceneResponse)
        return _filter_scene_response(
            response,
            grasp_threshold=grasp_threshold,
            topk_num_grasps=topk_num_grasps,
        )

    @staticmethod
    def scene_results_by_id(response: InferSceneResponse) -> dict[int, tuple]:
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


def _apply_threshold_topk(
    grasps: np.ndarray,
    confidences: np.ndarray,
    branch_tags: list[BranchTag],
    *,
    grasp_threshold: float,
    topk_num_grasps: int,
) -> tuple[np.ndarray, np.ndarray, list[BranchTag]]:
    grasp_threshold, topk_num_grasps = _validate_selection(grasp_threshold, topk_num_grasps)
    tags = list(branch_tags)
    if grasp_threshold > 0.0:
        keep = confidences >= grasp_threshold
        grasps = grasps[keep]
        confidences = confidences[keep]
        tags = [tag for tag, selected in zip(tags, keep, strict=True) if selected]
    if topk_num_grasps > 0:
        order = np.argsort(-confidences, kind="stable")[:topk_num_grasps]
        grasps = grasps[order]
        confidences = confidences[order]
        tags = [tags[int(index)] for index in order]
    return grasps, confidences, tags


def _filter_scene_response(
    response: InferSceneResponse,
    *,
    grasp_threshold: float,
    topk_num_grasps: int,
) -> InferSceneResponse:
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
        selected_grasps, selected_confidences, selected_tags = _apply_threshold_topk(
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

    return InferSceneResponse(
        instance_ids=np.asarray(instance_ids, dtype=np.int32),
        grasps=grasps_list,
        confidences=confidence_list,
        branch_tags=tag_list,
        skipped_instance_ids=np.asarray(sorted(set(skipped)), dtype=np.int32),
        timing=response.timing,
        timing_ms=response.timing_ms,
    )


__all__ = ["GraspGenXClient"]
