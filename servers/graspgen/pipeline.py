"""GraspGen inference wrapper, headless. Visualization is intentionally client-side only.

GRASPGEN_MODELS_DIR overrides the default GraspGenModels/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "GraspGenModels"

# Grippers with no native checkpoint -> source model to load and retarget from.
# Strokes: franka ~105mm, 2f_85 = 85mm, 2f_140 = 136mm.
_RETARGET_SOURCE = {
    "robotiq_2f_85": "franka_panda",
}


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
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_depth, get_gripper_info

        # Custom yml paths bypass retargeting.
        input_path = Path(gripper_config)
        if input_path.is_file():
            target_name = input_path.stem.removeprefix("graspgen_")
            source_name = target_name
            config_path = str(input_path)
        else:
            target_name = gripper_config
            source_name = _RETARGET_SOURCE.get(target_name, target_name)
            config_path = _resolve_gripper_config(source_name)

        self.cfg = load_grasp_cfg(config_path)
        self.sampler = GraspGenSampler(self.cfg)

        # Both refer to TARGET, not the source model whose weights we loaded.
        self.gripper_name = target_name
        self.gripper_info = get_gripper_info(target_name)

        # Z-offset retargeting
        if source_name != target_name:
            self._z_offset = (
                float(get_gripper_depth(source_name))
                - float(get_gripper_depth(target_name))
            )
        else:
            self._z_offset = 0.0

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

        if self._z_offset != 0.0:
            T_off = np.eye(4, dtype=grasps.dtype)
            T_off[2, 3] = self._z_offset
            grasps = grasps @ T_off

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
