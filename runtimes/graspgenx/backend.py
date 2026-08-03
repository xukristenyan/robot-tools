"""Headless wrapper around the official GraspGenX inference paths.

The gripper-independent model is loaded once. Named grippers and raw
sweep-volume conditionings each get a lazily constructed sampler that shares
those model weights, matching ``graspgenx.serving.zmq_server``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from robot_tools.services.graspgenx.contract import SweepVolumeParams

logger = logging.getLogger(__name__)

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

    def _get_sweep_sampler(self, params: SweepVolumeParams):
        from graspgenx.grasp_server import GraspGenXSampler

        key = f"sweep:{params.cache_key()}"
        with self._samplers_lock:
            sampler = self._samplers.get(key)
            if sampler is None:
                logger.info("creating GraspGenX sampler from sweep volume %s", key)
                sampler = GraspGenXSampler.from_sweep_volume(
                    self.cfg,
                    params.to_wire(),
                    model=self._shared_model,
                )
                self._samplers[key] = sampler
            return sampler

    def metadata(self) -> dict:
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

    def infer(
        self,
        point_cloud: np.ndarray,
        *,
        gripper_name: str | None,
        num_grasps: int,
        grasp_threshold: float,
        topk_num_grasps: int,
    ) -> dict:
        from graspgenx.grasp_server import GraspGenXSampler

        selected_gripper = gripper_name or self.default_gripper
        if not selected_gripper:
            raise ValueError("request omitted gripper_name and the server has no default gripper")
        sampler = self._get_named_sampler(selected_gripper)
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

    def infer_object(
        self,
        point_cloud: np.ndarray,
        *,
        sweep_volume_params: SweepVolumeParams,
        planner_kwargs: dict,
    ) -> dict:
        from graspgenx.samplers import run_planner_on_object

        sampler = self._get_sweep_sampler(sweep_volume_params)
        started = time.monotonic()
        grasps, confidences, tags, _obb = run_planner_on_object(
            point_cloud,
            sampler,
            **planner_kwargs,
        )
        infer_ms = (time.monotonic() - started) * 1000.0
        return {
            "grasps": _to_grasps(grasps),
            "confidences": _to_confidences(confidences),
            "branch_tags": [str(tag) for tag in tags],
            "timing": {"infer_ms": infer_ms},
        }

    def infer_scene_depth(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        *,
        sweep_volume_params: SweepVolumeParams,
        min_object_points: int,
        planner_kwargs: dict,
    ) -> dict:
        from graspgenx.utils.scene_loaders import depth_to_camera_xyz

        xyz = depth_to_camera_xyz(depth, intrinsics).reshape(-1, 3)
        valid = ((depth > 0) & np.isfinite(depth)).reshape(-1)
        return self._infer_instances(
            xyz,
            instance_mask.reshape(-1),
            valid,
            sweep_volume_params=sweep_volume_params,
            min_object_points=min_object_points,
            planner_kwargs=planner_kwargs,
        )

    def infer_scene_pc(
        self,
        point_cloud: np.ndarray,
        instance_mask: np.ndarray,
        *,
        sweep_volume_params: SweepVolumeParams,
        min_object_points: int,
        planner_kwargs: dict,
    ) -> dict:
        point_cloud = point_cloud.reshape(-1, 3)
        valid = np.isfinite(point_cloud).all(axis=1)
        return self._infer_instances(
            point_cloud,
            instance_mask.reshape(-1),
            valid,
            sweep_volume_params=sweep_volume_params,
            min_object_points=min_object_points,
            planner_kwargs=planner_kwargs,
        )

    def _infer_instances(
        self,
        point_cloud: np.ndarray,
        instance_mask: np.ndarray,
        valid: np.ndarray,
        *,
        sweep_volume_params: SweepVolumeParams,
        min_object_points: int,
        planner_kwargs: dict,
    ) -> dict:
        from graspgenx.samplers import run_planner_on_batch

        sampler = self._get_sweep_sampler(sweep_volume_params)
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
        infer_ms = (time.monotonic() - started) * 1000.0

        output_ids: list[int] = []
        output_grasps: list[np.ndarray] = []
        output_confidences: list[np.ndarray] = []
        output_tags: list[list[str]] = []
        for instance_id, (grasps, confidences, tags, _obb) in zip(
            instance_ids,
            batch_results,
            strict=True,
        ):
            if len(grasps) == 0:
                skipped_ids.append(instance_id)
                continue
            output_ids.append(instance_id)
            output_grasps.append(_to_grasps(grasps))
            output_confidences.append(_to_confidences(confidences))
            output_tags.append([str(tag) for tag in tags])

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
