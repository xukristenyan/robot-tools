from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_tools.services.graspgenx.contract import SweepVolumeParams


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
        sweep_samplers=[],
        object_kwargs=None,
        batch_point_clouds=None,
        batch_kwargs=None,
    )
    shared_model = object()

    root = types.ModuleType("graspgenx")
    root.__path__ = []
    root.get_checkpoints_version_dir = lambda: checkpoint_root

    grasp_server = types.ModuleType("graspgenx.grasp_server")

    class FakeSampler:
        def __init__(
            self,
            cfg,
            gripper_name=None,
            assets_dir=None,
            model=None,
        ):
            self.cfg = cfg
            self.gripper_name = gripper_name
            self.assets_dir = assets_dir
            self.model = model
            calls.named_samplers.append(self)

        @classmethod
        def from_sweep_volume(cls, cfg, params, model=None):
            sampler = cls.__new__(cls)
            sampler.cfg = cfg
            sampler.gripper_name = "custom_sweep_volume"
            sampler.assets_dir = None
            sampler.model = model
            sampler.params = params
            calls.sweep_samplers.append(sampler)
            return sampler

        @staticmethod
        def run_inference(
            point_cloud,
            sampler,
            *,
            grasp_threshold,
            num_grasps,
            topk_num_grasps,
        ):
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

    def run_planner_on_object(point_cloud, sampler, **kwargs):
        calls.object_sampler = sampler
        calls.object_kwargs = kwargs
        grasps = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        return grasps, np.array([0.7, 0.8], dtype=np.float32), ["diff", "obb"], None

    def run_planner_on_batch(point_clouds, sampler, **kwargs):
        calls.batch_sampler = sampler
        calls.batch_point_clouds = point_clouds
        calls.batch_kwargs = kwargs
        output = []
        for _point_cloud in point_clouds:
            grasps = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))
            output.append((grasps, np.array([0.75], dtype=np.float32), ["diff"], None))
        return output

    samplers.run_planner_on_object = run_planner_on_object
    samplers.run_planner_on_batch = run_planner_on_batch

    scene_loaders = types.ModuleType("graspgenx.utils.scene_loaders")

    def depth_to_camera_xyz(depth, intrinsics):
        del intrinsics
        y, x = np.indices(depth.shape)
        return np.stack([x, y, depth], axis=-1).astype(np.float32)

    scene_loaders.depth_to_camera_xyz = depth_to_camera_xyz

    utils = types.ModuleType("graspgenx.utils")
    utils.__path__ = []

    for name, module in {
        "graspgenx": root,
        "graspgenx.grasp_server": grasp_server,
        "graspgenx.samplers": samplers,
        "graspgenx.utils": utils,
        "graspgenx.utils.checkpoint_io": checkpoint_io,
        "graspgenx.utils.scene_loaders": scene_loaders,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return calls, shared_model


def _sweep() -> SweepVolumeParams:
    return SweepVolumeParams.from_flat([0.08, 0.04, 0.12, 0.0, 0.0, 0.06, 0.04, 0.04, 0.12, 0.0, 0.0, 0.06])


def test_backend_shares_model_and_caches_conditioning(monkeypatch, tmp_path):
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
    first_sweep = backend._get_sweep_sampler(_sweep())
    same_sweep = backend._get_sweep_sampler(_sweep())

    assert calls.model_loads == 1
    assert len(calls.named_samplers) == 2
    assert len(calls.sweep_samplers) == 1
    assert first_named is backend._get_named_sampler("franka_panda")
    assert first_sweep is same_sweep
    assert first_named.model is shared_model
    assert second_named.model is shared_model
    assert first_sweep.model is shared_model
    assert backend.metadata()["loaded_grippers"] == [
        "franka_panda",
        "robotiq_2f_85",
        f"sweep:{_sweep().cache_key()}",
    ]


def test_backend_forwards_planner_and_batches_scene_instances(monkeypatch, tmp_path):
    checkpoint_root = tmp_path / "release"
    (checkpoint_root / "gen").mkdir(parents=True)
    (checkpoint_root / "dis").mkdir()
    calls, _shared_model = _install_fake_upstream(monkeypatch, checkpoint_root)
    module = _load_backend_module()
    backend = module.GraspGenXBackend(config_path=str(checkpoint_root))
    sweep = _sweep()
    planner_kwargs = {
        "planner": "graspmoe",
        "grasp_threshold": -1.0,
        "num_grasps": 12,
        "topk_num_grasps": -1,
        "moe_num_yaws": 18,
    }

    object_result = backend.infer_object(
        np.zeros((20, 3), dtype=np.float32),
        sweep_volume_params=sweep,
        planner_kwargs=planner_kwargs,
    )
    assert object_result["grasps"].shape == (2, 4, 4)
    assert object_result["branch_tags"] == ["diff", "obb"]
    assert calls.object_kwargs == planner_kwargs

    scene_pc = np.zeros((8, 3), dtype=np.float32)
    scene_pc[-1] = np.nan
    instance_mask = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    scene_result = backend.infer_scene_pc(
        scene_pc,
        instance_mask,
        sweep_volume_params=sweep,
        min_object_points=4,
        planner_kwargs=planner_kwargs,
    )

    assert scene_result["instance_ids"].tolist() == [1]
    assert scene_result["skipped_instance_ids"].tolist() == [2]
    assert len(calls.batch_point_clouds) == 1
    assert calls.batch_point_clouds[0].shape == (4, 3)
    assert calls.batch_kwargs == planner_kwargs
