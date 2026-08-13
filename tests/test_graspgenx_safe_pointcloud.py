from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from robot_tools.services.graspgenx import GraspGenXClient
from robot_tools.services.graspgenx.contract import (
    API_VERSION,
    GenerateGraspsResponse,
    GenerateSafeGraspsRequest,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(**kwargs) -> GenerateSafeGraspsRequest:
    fields = {
        "object_point_cloud": np.zeros((128, 3), dtype=np.float32),
        "scene_point_cloud": np.zeros((256, 3), dtype=np.float32),
        "gripper_name": "robotiq_2f_85",
        "collision_threshold": 0.005,
    }
    fields.update(kwargs)
    return GenerateSafeGraspsRequest(**fields)


def test_safe_request_uses_two_point_clouds() -> None:
    request = _request(
        object_point_cloud=np.zeros((4, 3), dtype=np.float64),
        scene_point_cloud=np.empty((0, 3), dtype=np.float64),
    )

    assert API_VERSION == "4"
    assert request.object_point_cloud.shape == (4, 3)
    assert request.object_point_cloud.dtype == np.float32
    assert request.scene_point_cloud.shape == (0, 3)
    assert request.scene_point_cloud.dtype == np.float32
    assert request.object_point_cloud.flags.c_contiguous
    assert request.scene_point_cloud.flags.c_contiguous
    dumped = request.model_dump()
    assert "depth" not in dumped
    assert "intrinsics" not in dumped
    assert "target_mask" not in dumped


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("object_point_cloud", np.empty((0, 3)), "object_point_cloud must not be empty"),
        ("object_point_cloud", np.zeros((4, 4)), r"object_point_cloud must have shape \(N, 3\)"),
        ("scene_point_cloud", np.array([[np.nan, 0.0, 0.0]]), "scene_point_cloud must contain only finite"),
    ],
)
def test_safe_request_rejects_invalid_point_clouds(field, value, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _request(**{field: value})


def test_client_sends_point_clouds_and_filters_response(monkeypatch) -> None:
    object_point_cloud = np.arange(12, dtype=np.float32).reshape(4, 3)
    scene_point_cloud = np.arange(18, dtype=np.float32).reshape(6, 3)
    grasps = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    response = GenerateGraspsResponse(
        grasps=grasps,
        confidences=np.array([0.6, 0.9, 0.8], dtype=np.float32),
        gripper_name="robotiq_2f_85",
        timing={"infer_ms": 1.0},
    )
    captured = []

    with GraspGenXClient(wait=False) as client:
        monkeypatch.setattr(client, "_request", lambda request, response_cls: captured.append(request) or response)
        result = client.generate_safe_grasps(
            object_point_cloud,
            scene_point_cloud,
            gripper_name="robotiq_2f_85",
            collision_threshold=0.005,
            grasp_threshold=0.7,
            topk_num_grasps=1,
        )

    request = captured[0]
    np.testing.assert_array_equal(request.object_point_cloud, object_point_cloud)
    np.testing.assert_array_equal(request.scene_point_cloud, scene_point_cloud)
    assert request.collision_threshold == pytest.approx(0.005)
    assert result.confidences.tolist() == pytest.approx([0.9])


def test_server_forwards_point_clouds_to_backend() -> None:
    module = _load_module("robot_tools_test_graspgenx_server_v4", ROOT / "runtimes/graspgenx/server.py")
    server = module.GraspGenXServer(fake=True, default_gripper="robotiq_2f_85")
    captured = {}

    def generate_safe_grasps(object_point_cloud, scene_point_cloud, **kwargs):
        captured["object_point_cloud"] = object_point_cloud
        captured["scene_point_cloud"] = scene_point_cloud
        captured["kwargs"] = kwargs
        return {
            "grasps": np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
            "confidences": np.array([0.8], dtype=np.float32),
            "gripper_name": "robotiq_2f_85",
            "timing": {"infer_ms": 1.0},
        }

    server._fake = False
    server.backend = SimpleNamespace(generate_safe_grasps=generate_safe_grasps)
    request = _request()
    response = server.generate_safe_grasps(request)

    assert response.confidences.tolist() == pytest.approx([0.8])
    np.testing.assert_array_equal(captured["object_point_cloud"], request.object_point_cloud)
    np.testing.assert_array_equal(captured["scene_point_cloud"], request.scene_point_cloud)
    assert captured["kwargs"]["collision_threshold"] == pytest.approx(0.005)
    assert "min_object_points" not in captured["kwargs"]


def test_backend_downsamples_scene_and_filters_candidates(monkeypatch) -> None:
    calls = SimpleNamespace(object_point_cloud=None, scene_point_cloud=None)
    samplers = types.ModuleType("graspgenx.samplers")
    collision_filter = types.ModuleType("graspgenx.utils.collision_filter")
    utils = types.ModuleType("graspgenx.utils")
    utils.__path__ = []

    def run_planner_on_object(point_cloud, sampler, **kwargs):
        del sampler, kwargs
        calls.object_point_cloud = point_cloud
        grasps = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        return grasps, np.array([0.9, 0.6], dtype=np.float32), ["diff", "obb"], None

    def filter_colliding_grasps(*, scene_pc, grasp_poses, **kwargs):
        del grasp_poses, kwargs
        calls.scene_point_cloud = scene_pc
        return np.array([True, False])

    samplers.run_planner_on_object = run_planner_on_object
    collision_filter.filter_colliding_grasps = filter_colliding_grasps
    monkeypatch.setitem(sys.modules, "graspgenx.samplers", samplers)
    monkeypatch.setitem(sys.modules, "graspgenx.utils", utils)
    monkeypatch.setitem(sys.modules, "graspgenx.utils.collision_filter", collision_filter)

    module = _load_module("robot_tools_test_graspgenx_backend_v4", ROOT / "runtimes/graspgenx/backend.py")
    backend = module.GraspGenXBackend.__new__(module.GraspGenXBackend)
    backend._resolve_sampler = lambda gripper_name: (gripper_name, object())
    backend._get_collision_surface_points = lambda gripper_name, sampler: np.zeros((8, 3), dtype=np.float32)

    object_point_cloud = np.arange(384, dtype=np.float32).reshape(128, 3)
    scene_point_cloud = np.arange(27_000, dtype=np.float32).reshape(9_000, 3)
    result = backend.generate_safe_grasps(
        object_point_cloud,
        scene_point_cloud,
        gripper_name="robotiq_2f_85",
        collision_threshold=0.005,
        planner_kwargs={"planner": "graspmoe"},
    )

    np.testing.assert_array_equal(calls.object_point_cloud, object_point_cloud)
    assert calls.scene_point_cloud.shape == (module.MAX_COLLISION_SCENE_POINTS, 3)
    np.testing.assert_array_equal(calls.scene_point_cloud[0], scene_point_cloud[0])
    np.testing.assert_array_equal(calls.scene_point_cloud[-1], scene_point_cloud[-1])
    assert result["grasps"].shape == (1, 4, 4)
    assert result["confidences"].tolist() == pytest.approx([0.9])
