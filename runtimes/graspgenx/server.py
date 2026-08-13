"""robot-tools HTTP server for the native GraspGenX inference API."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.graspgenx.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    GenerateGraspsForAllRequest,
    GenerateGraspsForAllResponse,
    GenerateGraspsRequest,
    GenerateGraspsResponse,
    GenerateSafeGraspsForAllRequest,
    GenerateSafeGraspsRequest,
    GetMetadataRequest,
    GetMetadataResponse,
)

logger = logging.getLogger(__name__)


class GraspGenXServer(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5559,
        *,
        fake: bool = False,
        config_path: str | None = None,
        assets_dir: str | None = None,
        default_gripper: str | None = None,
        use_tensorrt: bool = False,
        tensorrt_precision: str = "fp32",
    ):
        super().__init__(
            service_id=SERVICE_ID,
            api_version=API_VERSION,
            host=host,
            port=port,
        )
        self._fake = fake
        self._default_gripper = default_gripper
        if fake:
            logger.warning("FAKE mode: returning deterministic test data; no model or assets loaded")
            self.backend = None
        else:
            from backend import GraspGenXBackend

            self.backend = GraspGenXBackend(
                config_path=config_path,
                assets_dir=assets_dir,
                default_gripper=default_gripper,
                use_tensorrt=use_tensorrt,
                tensorrt_precision=tensorrt_precision,
            )
        self.mark_ready()

    @handler(GetMetadataRequest)
    def get_metadata(self, req: GetMetadataRequest) -> GetMetadataResponse:
        if self._fake:
            loaded = [self._default_gripper] if self._default_gripper else []
            data = {
                "default_gripper": self._default_gripper,
                "loaded_grippers": loaded,
                "model": {
                    "generator_backbone": "ptv3_vanilla",
                    "discriminator_backbone": "ptv3_vanilla",
                    "grasp_repr": "fake_se3",
                    "num_diffusion_iters_eval": 0,
                },
                "assets_dir": "fake://gripper-assets",
                "precision": {
                    "weights": "fp32",
                    "tensorrt": False,
                    "tensorrt_precision": None,
                },
            }
        else:
            data = self.backend.get_metadata()
        return GetMetadataResponse(actions=sorted(ACTIONS), **data)

    @handler(GenerateGraspsRequest)
    def generate_grasps(self, req: GenerateGraspsRequest) -> GenerateGraspsResponse:
        if self._fake:
            gripper_name = _resolve_gripper_name(req.gripper_name, self._default_gripper)
            grasps, confidences, _tags = _fake_candidates(
                req.num_grasps,
                planner="diffusion",
            )
            if req.grasp_threshold > 0:
                keep = confidences >= req.grasp_threshold
                grasps, confidences = grasps[keep], confidences[keep]
            if req.topk_num_grasps > 0:
                order = np.argsort(-confidences, kind="stable")[: req.topk_num_grasps]
                grasps, confidences = grasps[order], confidences[order]
            return GenerateGraspsResponse(
                grasps=grasps,
                confidences=confidences,
                gripper_name=gripper_name,
                timing={"infer_ms": 0.0},
            )

        result = self.backend.generate_grasps(
            req.point_cloud,
            gripper_name=req.gripper_name,
            num_grasps=req.num_grasps,
            grasp_threshold=req.grasp_threshold,
            topk_num_grasps=req.topk_num_grasps,
        )
        return GenerateGraspsResponse(**result)

    @handler(GenerateSafeGraspsRequest)
    def generate_safe_grasps(self, req: GenerateSafeGraspsRequest) -> GenerateGraspsResponse:
        if self._fake:
            gripper_name = _resolve_gripper_name(req.gripper_name, self._default_gripper)
            valid_target = (req.depth > 0) & np.isfinite(req.depth) & req.target_mask
            point_count = int(np.count_nonzero(valid_target))
            if point_count < req.min_object_points:
                raise ValueError(
                    f"target_mask contains {point_count} valid points; at least {req.min_object_points} are required"
                )
            grasps, confidences, _tags = _fake_candidates(
                req.num_grasps,
                planner=req.planner,
            )
            return GenerateGraspsResponse(
                grasps=grasps,
                confidences=confidences,
                gripper_name=gripper_name,
                timing={"infer_ms": 0.0},
            )

        result = self.backend.generate_safe_grasps(
            req.depth,
            req.intrinsics,
            req.target_mask,
            gripper_name=req.gripper_name,
            min_object_points=req.min_object_points,
            collision_threshold=req.collision_threshold,
            planner_kwargs=req.planner_kwargs(),
        )
        return GenerateGraspsResponse(**result)

    @handler(GenerateGraspsForAllRequest)
    def generate_grasps_for_all(
        self,
        req: GenerateGraspsForAllRequest,
    ) -> GenerateGraspsForAllResponse:
        if self._fake:
            gripper_name = _resolve_gripper_name(req.gripper_name, self._default_gripper)
            valid = ((req.depth > 0) & np.isfinite(req.depth)).reshape(-1)
            return _fake_scene(
                req.instance_mask.reshape(-1),
                valid,
                gripper_name=gripper_name,
                min_object_points=req.min_object_points,
                num_grasps=req.num_grasps,
                planner=req.planner,
            )

        result = self.backend.generate_grasps_for_all(
            req.depth,
            req.intrinsics,
            req.instance_mask,
            gripper_name=req.gripper_name,
            min_object_points=req.min_object_points,
            planner_kwargs=req.planner_kwargs(),
        )
        return GenerateGraspsForAllResponse(**result)

    @handler(GenerateSafeGraspsForAllRequest)
    def generate_safe_grasps_for_all(
        self,
        req: GenerateSafeGraspsForAllRequest,
    ) -> GenerateGraspsForAllResponse:
        if self._fake:
            gripper_name = _resolve_gripper_name(req.gripper_name, self._default_gripper)
            valid = ((req.depth > 0) & np.isfinite(req.depth)).reshape(-1)
            return _fake_scene(
                req.instance_mask.reshape(-1),
                valid,
                gripper_name=gripper_name,
                min_object_points=req.min_object_points,
                num_grasps=req.num_grasps,
                planner=req.planner,
            )

        result = self.backend.generate_safe_grasps_for_all(
            req.depth,
            req.intrinsics,
            req.instance_mask,
            gripper_name=req.gripper_name,
            min_object_points=req.min_object_points,
            collision_threshold=req.collision_threshold,
            planner_kwargs=req.planner_kwargs(),
        )
        return GenerateGraspsForAllResponse(**result)


def _resolve_gripper_name(requested: str | None, default: str | None) -> str:
    gripper_name = requested or default
    if not gripper_name:
        raise ValueError("request omitted gripper_name and the server has no default gripper")
    return gripper_name


def _fake_candidates(
    num_grasps: int,
    *,
    planner: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    count = min(num_grasps, 32)
    grasps = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    if count:
        grasps[:, 0, 3] = np.arange(count, dtype=np.float32) / 1000.0
    confidences = np.linspace(0.05, 0.95, count, dtype=np.float32)
    tags = (
        ["diff"] * count if planner == "diffusion" else ["diff" if index % 2 == 0 else "obb" for index in range(count)]
    )
    return grasps, confidences, tags


def _fake_scene(
    instance_mask: np.ndarray,
    valid: np.ndarray,
    *,
    gripper_name: str,
    min_object_points: int,
    num_grasps: int,
    planner: str,
) -> GenerateGraspsForAllResponse:
    started = time.monotonic()
    instance_ids: list[int] = []
    grasps_list: list[np.ndarray] = []
    confidence_list: list[np.ndarray] = []
    tag_list: list[list[str]] = []
    skipped_ids: list[int] = []

    candidate_ids = np.unique(instance_mask[valid & (instance_mask > 0)])
    for instance_id in candidate_ids.tolist():
        point_count = int(np.count_nonzero(valid & (instance_mask == instance_id)))
        if point_count < min_object_points:
            skipped_ids.append(int(instance_id))
            continue
        grasps, confidences, tags = _fake_candidates(
            num_grasps,
            planner=planner,
        )
        instance_ids.append(int(instance_id))
        grasps_list.append(grasps)
        confidence_list.append(confidences)
        tag_list.append(tags)

    return GenerateGraspsForAllResponse(
        instance_ids=np.asarray(instance_ids, dtype=np.int32),
        grasps=grasps_list,
        confidences=confidence_list,
        branch_tags=tag_list,
        skipped_instance_ids=np.asarray(skipped_ids, dtype=np.int32),
        gripper_name=gripper_name,
        timing={"infer_ms": (time.monotonic() - started) * 1000.0},
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5559)
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Checkpoint root containing gen/ and dis/; defaults to the official release",
    )
    parser.add_argument(
        "--assets-dir",
        "--assets_dir",
        dest="assets_dir",
        default=None,
        help="Fallback root containing x_grippers/ and proc_grippers/",
    )
    parser.add_argument(
        "--default-gripper",
        "--default_gripper",
        dest="default_gripper",
        default=None,
        help="Optionally preload this named gripper and use it when a request omits gripper_name",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Opt in to upstream TensorRT acceleration; falls back to eager PyTorch",
    )
    parser.add_argument(
        "--tensorrt-precision",
        "--tensorrt_precision",
        dest="tensorrt_precision",
        choices=["fp32", "fp16"],
        default="fp32",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Skip all model imports and return deterministic protocol test data",
    )
    args = parser.parse_args()

    GraspGenXServer(
        host=args.host,
        port=args.port,
        fake=args.fake,
        config_path=args.config_path,
        assets_dir=args.assets_dir,
        default_gripper=args.default_gripper,
        use_tensorrt=args.tensorrt,
        tensorrt_precision=args.tensorrt_precision,
    ).serve_forever()


if __name__ == "__main__":
    main()
