"""Headless named-gripper backend for the robot-tools GraspGenX service."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MAX_COLLISION_SCENE_POINTS = 8192
NUM_COLLISION_SAMPLES = 2000

# The official GraspGenX headless instructions require EGL; pyglet also needs
# its headless mode before pyrender is imported transitively by dataset.py.
# Respect an explicit deployment override while making the HTTP server safe to
# start over SSH without an X display.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "true")

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "upstream"

# Keep the pinned upstream as source rather than installing its broad package
# metadata, which includes training, visualization, ZMQ, and demo tooling.
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))


class GraspGenXBackend:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        assets_dir: str | None = None,
        default_gripper: str | None = None,
        use_tensorrt: bool = False,
        tensorrt_precision: str = "fp32",
    ):
        from graspgenx import get_checkpoints_version_dir
        from graspgenx.grasp_server import load_grasp_gen_model
        from graspgenx.utils.checkpoint_io import load_model_cfg

        checkpoint_root = Path(config_path or get_checkpoints_version_dir()).expanduser().resolve()
        if checkpoint_root.is_file() and checkpoint_root.name == "config.yaml":
            if checkpoint_root.parent.name != "gen":
                raise ValueError("a config.yaml path must point to the checkpoint gen/ directory")
            checkpoint_root = checkpoint_root.parent.parent
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(f"checkpoint root does not exist: {checkpoint_root}")

        generator_dir = checkpoint_root / "gen"
        discriminator_dir = checkpoint_root / "dis"
        if not generator_dir.is_dir() or not discriminator_dir.is_dir():
            raise FileNotFoundError(f"checkpoint root {checkpoint_root} must contain gen/ and dis/")

        self.assets_dir = str(
            Path(assets_dir).expanduser().resolve() if assets_dir is not None else (_UPSTREAM / "assets").resolve()
        )
        self.default_gripper = default_gripper
        self.cfg = load_model_cfg(str(generator_dir), str(discriminator_dir))
        self._samplers: dict[str, object] = {}
        self._collision_surface_points: dict[str, np.ndarray] = {}
        self._samplers_lock = threading.Lock()
        self._tensorrt_applied = False
        self._tensorrt_precision = tensorrt_precision

        logger.info("loading shared GraspGenX model from %s", checkpoint_root)
        self._shared_model = load_grasp_gen_model(self.cfg)
        logger.info("shared GraspGenX model loaded")

        if use_tensorrt:
            self._enable_tensorrt(tensorrt_precision)

        if default_gripper is not None:
            self._get_named_sampler(default_gripper)

    def _enable_tensorrt(self, precision: str) -> None:
        from types import SimpleNamespace

        from graspgenx.models.tensorrt_utils import accelerate_sampler
        from graspgenx.samplers.graspmoe import set_gpu_obb

        self._tensorrt_applied = bool(
            accelerate_sampler(
                SimpleNamespace(model=self._shared_model),
                precision=precision,
            )
        )
        if self._tensorrt_applied:
            set_gpu_obb(True)
            logger.info("GraspGenX TensorRT enabled (%s)", precision)
        else:
            logger.warning("TensorRT was requested but unavailable; using eager PyTorch")

    def _get_named_sampler(self, gripper_name: str):
        from graspgenx.grasp_server import GraspGenXSampler

        key = gripper_name
        with self._samplers_lock:
            sampler = self._samplers.get(key)
            if sampler is None:
                logger.info("creating GraspGenX sampler for gripper %s", gripper_name)
                sampler = GraspGenXSampler(
                    self.cfg,
                    gripper_name=gripper_name,
                    assets_dir=self.assets_dir,
                    model=self._shared_model,
                )
                self._samplers[key] = sampler
            return sampler

    def _resolve_sampler(self, gripper_name: str | None) -> tuple[str, object]:
        selected_gripper = gripper_name or self.default_gripper
        if not selected_gripper:
            raise ValueError("request omitted gripper_name and the server has no default gripper")
        return selected_gripper, self._get_named_sampler(selected_gripper)

    def _get_collision_surface_points(self, gripper_name: str, sampler: object) -> np.ndarray:
        import trimesh

        with self._samplers_lock:
            surface_points = self._collision_surface_points.get(gripper_name)
            if surface_points is None:
                sampled, _ = trimesh.sample.sample_surface(
                    sampler.get_gripper_info().collision_mesh,
                    NUM_COLLISION_SAMPLES,
                )
                surface_points = np.asarray(sampled, dtype=np.float32)
                self._collision_surface_points[gripper_name] = surface_points
            return surface_points

    def get_metadata(self) -> dict:
        diffusion = self.cfg.diffusion
        discriminator = self.cfg.discriminator
        return {
            "default_gripper": self.default_gripper,
            "loaded_grippers": sorted(self._samplers),
            "model": {
                "generator_backbone": str(diffusion.object_backbone),
                "discriminator_backbone": str(discriminator.object_backbone),
                "grasp_repr": str(diffusion.grasp_repr),
                "num_diffusion_iters_eval": int(diffusion.num_diffusion_iters_eval),
            },
            "assets_dir": self.assets_dir,
            "precision": {
                "weights": "fp32",
                "tensorrt": self._tensorrt_applied,
                "tensorrt_precision": (self._tensorrt_precision if self._tensorrt_applied else None),
            },
        }

    def generate_grasps(
        self,
        point_cloud: np.ndarray,
        *,
        gripper_name: str | None,
        num_grasps: int,
        grasp_threshold: float,
        topk_num_grasps: int,
    ) -> dict:
        from graspgenx.grasp_server import GraspGenXSampler

        selected_gripper, sampler = self._resolve_sampler(gripper_name)
        started = time.monotonic()
        grasps, confidences = GraspGenXSampler.run_inference(
            point_cloud,
            sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
        )
        infer_ms = (time.monotonic() - started) * 1000.0
        return {
            "grasps": _to_grasps(grasps),
            "confidences": _to_confidences(confidences),
            "gripper_name": selected_gripper,
            "timing": {"infer_ms": infer_ms},
        }

    @staticmethod
    def _scene_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from graspgenx.utils.scene_loaders import depth_to_camera_xyz

        point_cloud = np.asarray(depth_to_camera_xyz(depth, intrinsics), dtype=np.float32).reshape(-1, 3)
        valid = ((depth > 0) & np.isfinite(depth)).reshape(-1)
        valid &= np.isfinite(point_cloud).all(axis=1)
        return point_cloud, valid

    def generate_safe_grasps(
        self,
        object_point_cloud: np.ndarray,
        scene_point_cloud: np.ndarray,
        *,
        gripper_name: str | None,
        collision_threshold: float,
        planner_kwargs: dict,
    ) -> dict:
        from graspgenx.samplers import run_planner_on_object
        from graspgenx.utils.collision_filter import filter_colliding_grasps

        selected_gripper, sampler = self._resolve_sampler(gripper_name)
        object_points = np.ascontiguousarray(object_point_cloud, dtype=np.float32)

        started = time.monotonic()
        grasps, confidences, _tags, _obb = run_planner_on_object(
            np.ascontiguousarray(object_points, dtype=np.float32),
            sampler,
            **planner_kwargs,
        )
        grasps = _to_grasps(grasps)
        confidences = _to_confidences(confidences)
        scene_without_target = _downsample_scene(scene_point_cloud)
        keep = filter_colliding_grasps(
            scene_pc=scene_without_target,
            grasp_poses=grasps,
            collision_threshold=collision_threshold,
            gripper_surface_points=self._get_collision_surface_points(selected_gripper, sampler),
        )
        infer_ms = (time.monotonic() - started) * 1000.0
        return {
            "grasps": grasps[keep],
            "confidences": confidences[keep],
            "gripper_name": selected_gripper,
            "timing": {"infer_ms": infer_ms},
        }

    def generate_grasps_for_all(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        *,
        gripper_name: str | None,
        min_object_points: int,
        planner_kwargs: dict,
    ) -> dict:
        selected_gripper, sampler = self._resolve_sampler(gripper_name)
        point_cloud, valid = self._scene_from_depth(depth, intrinsics)
        result = self._generate_for_instances(
            point_cloud,
            instance_mask.reshape(-1),
            valid,
            sampler=sampler,
            min_object_points=min_object_points,
            planner_kwargs=planner_kwargs,
        )
        result["gripper_name"] = selected_gripper
        return result

    def generate_safe_grasps_for_all(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        *,
        gripper_name: str | None,
        min_object_points: int,
        collision_threshold: float,
        planner_kwargs: dict,
    ) -> dict:
        selected_gripper, sampler = self._resolve_sampler(gripper_name)
        point_cloud, valid = self._scene_from_depth(depth, intrinsics)
        result = self._generate_for_instances(
            point_cloud,
            instance_mask.reshape(-1),
            valid,
            sampler=sampler,
            min_object_points=min_object_points,
            planner_kwargs=planner_kwargs,
            collision_threshold=collision_threshold,
            gripper_surface_points=self._get_collision_surface_points(selected_gripper, sampler),
        )
        result["gripper_name"] = selected_gripper
        return result

    def _generate_for_instances(
        self,
        point_cloud: np.ndarray,
        instance_mask: np.ndarray,
        valid: np.ndarray,
        *,
        sampler: object,
        min_object_points: int,
        planner_kwargs: dict,
        collision_threshold: float | None = None,
        gripper_surface_points: np.ndarray | None = None,
    ) -> dict:
        from graspgenx.samplers import run_planner_on_batch
        from graspgenx.utils.collision_filter import filter_colliding_grasps

        candidate_ids = np.unique(instance_mask[valid & (instance_mask > 0)])
        instance_ids: list[int] = []
        instance_point_clouds: list[np.ndarray] = []
        skipped_ids: list[int] = []

        for instance_id in candidate_ids.tolist():
            points = point_cloud[valid & (instance_mask == instance_id)]
            if len(points) < min_object_points:
                skipped_ids.append(int(instance_id))
                continue
            instance_ids.append(int(instance_id))
            instance_point_clouds.append(np.ascontiguousarray(points, dtype=np.float32))

        started = time.monotonic()
        batch_results = run_planner_on_batch(
            instance_point_clouds,
            sampler,
            **planner_kwargs,
        )
        output_ids: list[int] = []
        output_grasps: list[np.ndarray] = []
        output_confidences: list[np.ndarray] = []
        output_tags: list[list[str]] = []
        for instance_id, (grasps, confidences, tags, _obb) in zip(
            instance_ids,
            batch_results,
            strict=True,
        ):
            grasps = _to_grasps(grasps)
            confidences = _to_confidences(confidences)
            tags = [str(tag) for tag in tags]
            if collision_threshold is not None:
                scene_without_target = _downsample_scene(point_cloud[valid & (instance_mask != instance_id)])
                keep = filter_colliding_grasps(
                    scene_pc=scene_without_target,
                    grasp_poses=grasps,
                    collision_threshold=collision_threshold,
                    gripper_surface_points=gripper_surface_points,
                )
                grasps = grasps[keep]
                confidences = confidences[keep]
                tags = [tag for tag, is_free in zip(tags, keep, strict=True) if is_free]
            if len(grasps) == 0:
                skipped_ids.append(instance_id)
                continue
            output_ids.append(instance_id)
            output_grasps.append(grasps)
            output_confidences.append(confidences)
            output_tags.append(tags)

        infer_ms = (time.monotonic() - started) * 1000.0

        return {
            "instance_ids": np.asarray(output_ids, dtype=np.int32),
            "grasps": output_grasps,
            "confidences": output_confidences,
            "branch_tags": output_tags,
            "skipped_instance_ids": np.asarray(
                sorted(skipped_ids),
                dtype=np.int32,
            ),
            "timing": {"infer_ms": infer_ms},
        }


def _downsample_scene(point_cloud: np.ndarray) -> np.ndarray:
    point_cloud = np.ascontiguousarray(point_cloud, dtype=np.float32)
    if len(point_cloud) <= MAX_COLLISION_SCENE_POINTS:
        return point_cloud
    indices = np.linspace(0, len(point_cloud) - 1, MAX_COLLISION_SCENE_POINTS, dtype=np.int64)
    return np.ascontiguousarray(point_cloud[indices])


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_grasps(value: object) -> np.ndarray:
    array = _to_numpy(value)
    if array.size == 0:
        return np.empty((0, 4, 4), dtype=np.float32)
    return np.asarray(array, dtype=np.float32).reshape(-1, 4, 4)


def _to_confidences(value: object) -> np.ndarray:
    array = _to_numpy(value)
    if array.size == 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(array, dtype=np.float32).reshape(-1)
