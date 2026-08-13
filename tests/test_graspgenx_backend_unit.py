from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_backend_module():
    path = Path(__file__).resolve().parents[1] / "runtimes" / "graspgenx" / "backend.py"
    spec = importlib.util.spec_from_file_location(
        "robot_tools_test_graspgenx_backend",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_upstream(monkeypatch, checkpoint_root: Path):
    calls = SimpleNamespace(
        model_loads=0,
        named_samplers=[],
        batch_point_clouds=None,
        batch_kwargs=None,
    )
    shared_model = object()

    root = types.ModuleType("graspgenx")
    root.__path__ = []
    root.get_checkpoints_version_dir = lambda: checkpoint_root

    grasp_server = types.ModuleType("graspgenx.grasp_server")

    class FakeSampler:
        def __init__(self, cfg, gripper_name=None, assets_dir=None, model=None):
            self.cfg = cfg
            self.gripper_name = gripper_name
            self.assets_dir = assets_dir
            self.model = model
            calls.named_samplers.append(self)

        @staticmethod
        def run_inference(point_cloud, sampler, *, grasp_threshold, num_grasps, topk_num_grasps):
            del point_cloud, sampler, grasp_threshold
            count = min(num_grasps, topk_num_grasps)
            grasps = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
            confidences = np.linspace(0.5, 0.9, count, dtype=np.float32)
            return grasps, confidences

    def load_grasp_gen_model(cfg):
        calls.model_loads += 1
        calls.cfg = cfg
        return shared_model

    grasp_server.GraspGenXSampler = FakeSampler
    grasp_server.load_grasp_gen_model = load_grasp_gen_model

    checkpoint_io = types.ModuleType("graspgenx.utils.checkpoint_io")
    checkpoint_io.load_model_cfg = lambda gen, dis: SimpleNamespace(
        diffusion=SimpleNamespace(
            object_backbone="ptv3_vanilla",
            grasp_repr="se3",
            num_diffusion_iters_eval=8,
        ),
        discriminator=SimpleNamespace(object_backbone="ptv3_vanilla"),
    )

    samplers = types.ModuleType("graspgenx.samplers")

    def run_planner_on_batch(point_clouds, sampler, **kwargs):
        calls.batch_sampler = sampler
        calls.batch_point_clouds = point_clouds
        calls.batch_kwargs = kwargs
        output = []
        for _point_cloud in point_clouds:
            grasps = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))
            output.append((grasps, np.array([0.75], dtype=np.float32), ["diff"], None))
        return output

    samplers.run_planner_on_batch = run_planner_on_batch

    scene_loaders = types.ModuleType("graspgenx.utils.scene_loaders")

    def depth_to_camera_xyz(depth, intrinsics):
        del intrinsics
        y, x = np.indices(depth.shape)
        return np.stack([x, y, depth], axis=-1).astype(np.float32)

    scene_loaders.depth_to_camera_xyz = depth_to_camera_xyz

    collision_filter = types.ModuleType("graspgenx.utils.collision_filter")
    collision_filter.filter_colliding_grasps = lambda **kwargs: np.ones(len(kwargs["grasp_poses"]), dtype=bool)
    utils = types.ModuleType("graspgenx.utils")
    utils.__path__ = []

    for name, module in {
        "graspgenx": root,
        "graspgenx.grasp_server": grasp_server,
        "graspgenx.samplers": samplers,
        "graspgenx.utils": utils,
        "graspgenx.utils.checkpoint_io": checkpoint_io,
        "graspgenx.utils.collision_filter": collision_filter,
        "graspgenx.utils.scene_loaders": scene_loaders,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return calls, shared_model


def test_backend_shares_model_and_caches_named_grippers(monkeypatch, tmp_path):
    checkpoint_root = tmp_path / "release"
    (checkpoint_root / "gen").mkdir(parents=True)
    (checkpoint_root / "dis").mkdir()
    calls, shared_model = _install_fake_upstream(monkeypatch, checkpoint_root)
    module = _load_backend_module()

    backend = module.GraspGenXBackend(
        config_path=str(checkpoint_root),
        default_gripper="franka_panda",
    )
    first_named = backend._get_named_sampler("franka_panda")
    second_named = backend._get_named_sampler("robotiq_2f_85")

    assert calls.model_loads == 1
    assert len(calls.named_samplers) == 2
    assert first_named is backend._get_named_sampler("franka_panda")
    assert first_named.model is shared_model
    assert second_named.model is shared_model
    assert backend.get_metadata()["loaded_grippers"] == ["franka_panda", "robotiq_2f_85"]


def test_backend_batches_named_gripper_scene_instances(monkeypatch, tmp_path):
    checkpoint_root = tmp_path / "release"
    (checkpoint_root / "gen").mkdir(parents=True)
    (checkpoint_root / "dis").mkdir()
    calls, _shared_model = _install_fake_upstream(monkeypatch, checkpoint_root)
    module = _load_backend_module()
    backend = module.GraspGenXBackend(config_path=str(checkpoint_root))
    planner_kwargs = {
        "planner": "graspmoe",
        "grasp_threshold": -1.0,
        "num_grasps": 12,
        "topk_num_grasps": -1,
        "moe_num_yaws": 18,
    }
    depth = np.ones((2, 4), dtype=np.float32)
    depth[-1, -1] = np.nan
    instance_mask = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int32)

    result = backend.generate_grasps_for_all(
        depth,
        np.eye(3),
        instance_mask,
        gripper_name="robotiq_2f_85",
        min_object_points=4,
        planner_kwargs=planner_kwargs,
    )

    assert result["instance_ids"].tolist() == [1]
    assert result["skipped_instance_ids"].tolist() == [2]
    assert result["gripper_name"] == "robotiq_2f_85"
    assert len(calls.batch_point_clouds) == 1
    assert calls.batch_point_clouds[0].shape == (4, 3)
    assert calls.batch_sampler.gripper_name == "robotiq_2f_85"
    assert calls.batch_kwargs == planner_kwargs
