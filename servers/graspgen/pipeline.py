"""High-level GraspGen inference wrapper, headless.

Models directory resolution order:
  1. $GRASPGEN_MODELS_DIR (explicit override)
  2. <this server dir>/GraspGenModels  (default, populated by download.sh)

Visualization helpers (viser) are intentionally NOT included here — server-side
rendering is the wrong layer. Visualize on the client if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "GraspGenModels"


def _models_dir() -> Path:
    raw = os.getenv("GRASPGEN_MODELS_DIR")
    candidate = Path(raw).expanduser().resolve() if raw else _DEFAULT_MODELS_DIR
    if not candidate.exists():
        raise RuntimeError(
            f"GraspGen models not found at {candidate}. "
            f"Run `robot-tools download graspgen` (or `bash download.sh`)."
        )
    return candidate


def _resolve_gripper_config(gripper_config: str) -> str:
    p = Path(gripper_config)
    if p.is_file():
        return str(p)

    cfg_dir = _models_dir() / "checkpoints"
    candidate = cfg_dir / f"graspgen_{gripper_config}.yml"
    if candidate.is_file():
        return str(candidate)

    available = sorted(f.stem for f in cfg_dir.glob("graspgen_*.yml"))
    raise FileNotFoundError(
        f"Gripper config {gripper_config!r} not found in {cfg_dir}. "
        f"Available: {available}"
    )


class GraspPipeline:
    """Wraps GraspGenSampler with grasp generation + collision filtering."""

    def __init__(
        self,
        gripper_config: str,
        grasp_threshold: float = 0.8,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
    ):
        # Imports are deferred so this module is importable without the heavy
        # GraspGen deps installed (useful for type-checking / tests).
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info

        config_path = _resolve_gripper_config(gripper_config)
        self.cfg = load_grasp_cfg(config_path)
        self.sampler = GraspGenSampler(self.cfg)
        self.gripper_name = self.cfg.data.gripper_name
        self.gripper_info = get_gripper_info(self.gripper_name)

        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps

    def generate_grasps(self, object_pc: np.ndarray) -> dict:
        from grasp_gen.grasp_server import GraspGenSampler

        grasps, conf = GraspGenSampler.run_inference(
            object_pc, self.sampler,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            topk_num_grasps=self.topk_num_grasps,
        )

        if len(grasps) == 0:
            return {"poses": np.empty((0, 4, 4), dtype=np.float32),
                    "confidences": np.empty((0,), dtype=np.float32)}

        grasps = grasps.cpu().numpy()
        conf = conf.cpu().numpy()
        grasps[:, 3, 3] = 1
        return {"poses": grasps, "confidences": conf}

    def generate_collision_free_grasps(
        self,
        object_pc: np.ndarray,
        scene_pc: np.ndarray,
        collision_threshold: float = 0.02,
        max_scene_points: int = 8192,
    ) -> dict:
        from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

        result = self.generate_grasps(object_pc)
        if result["poses"].shape[0] == 0:
            result["collision_free_mask"] = np.empty((0,), dtype=bool)
            return result

        # Remove the object points from the scene by exact-byte match.
        void_dtype = np.dtype((np.void, object_pc.dtype.itemsize * 3))
        obj_set = np.ascontiguousarray(object_pc).view(void_dtype).ravel()
        scene_view = np.ascontiguousarray(scene_pc).view(void_dtype).ravel()
        scene_pc = scene_pc[~np.isin(scene_view, obj_set)]

        if len(scene_pc) > max_scene_points:
            idx = np.random.choice(len(scene_pc), max_scene_points, replace=False)
            scene_pc = scene_pc[idx]

        result["collision_free_mask"] = filter_colliding_grasps(
            scene_pc=scene_pc,
            grasp_poses=result["poses"],
            gripper_collision_mesh=self.gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
        )
        return result
