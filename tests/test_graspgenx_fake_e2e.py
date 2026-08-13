from __future__ import annotations

import importlib.util
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from pydantic import ValidationError

from robot_tools.services.graspgenx import GraspGenXClient
from robot_tools.services.graspgenx.contract import (
    ACTIONS,
    API_VERSION,
    GenerateGraspsForAllRequest,
    GenerateGraspsForAllResponse,
    GenerateSafeGraspsRequest,
)


def _load_graspgenx_server_class():
    path = Path(__file__).resolve().parents[1] / "runtimes" / "graspgenx" / "server.py"
    spec = importlib.util.spec_from_file_location(
        "robot_tools_test_graspgenx_server",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraspGenXServer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def graspgenx_fake_port():
    port = _free_port()
    server_cls = _load_graspgenx_server_class()
    server = server_cls(
        host="127.0.0.1",
        port=port,
        fake=True,
        default_gripper="robotiq_2f_85",
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            server._build_app(),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not uvicorn_server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not uvicorn_server.started:
        raise RuntimeError("GraspGenX fake uvicorn did not start")
    yield port
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


def test_public_contract_rejects_raw_sweep_volume() -> None:
    with pytest.raises(ValidationError, match="sweep_volume_params"):
        GenerateSafeGraspsRequest.model_validate(
            {
                "object_point_cloud": np.zeros((10, 3), dtype=np.float32),
                "scene_point_cloud": np.zeros((20, 3), dtype=np.float32),
                "gripper_name": "robotiq_2f_85",
                "sweep_volume_params": np.zeros(12, dtype=np.float32),
            }
        )


def test_scene_contract_validates_depth_dtype_and_mask_shape() -> None:
    with pytest.raises(ValidationError, match="floating-point meters"):
        GenerateGraspsForAllRequest(
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=np.eye(3),
            instance_mask=np.ones((2, 2), dtype=np.int32),
        )
    with pytest.raises(ValidationError, match="shape must match depth"):
        GenerateGraspsForAllRequest(
            depth=np.ones((2, 2), dtype=np.float32),
            intrinsics=np.eye(3),
            instance_mask=np.ones((2, 3), dtype=np.int32),
        )


def test_scene_response_rejects_misaligned_branch_tags() -> None:
    with pytest.raises(ValidationError, match="lengths must match"):
        GenerateGraspsForAllResponse(
            instance_ids=np.array([1], dtype=np.int32),
            grasps=[np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))],
            confidences=[np.ones(2, dtype=np.float32)],
            branch_tags=[["diff"]],
            skipped_instance_ids=np.empty((0,), dtype=np.int32),
            gripper_name="robotiq_2f_85",
            timing={"infer_ms": 1.0},
        )


def test_health_and_metadata_expose_v4_actions(graspgenx_fake_port) -> None:
    with GraspGenXClient(host="127.0.0.1", port=graspgenx_fake_port, timeout_ms=2_000) as client:
        health = client.health()
        metadata = client.server_metadata

    assert set(health["actions"]) == set(ACTIONS)
    assert health["api_version"] == API_VERSION
    assert metadata.default_gripper == "robotiq_2f_85"
    assert metadata.loaded_grippers == ["robotiq_2f_85"]
    assert metadata.actions == sorted(ACTIONS)
    assert metadata.precision.tensorrt is False


def test_generate_grasps_uses_named_gripper(graspgenx_fake_port) -> None:
    point_cloud = np.zeros((2048, 3), dtype=np.float32)
    with GraspGenXClient(host="127.0.0.1", port=graspgenx_fake_port, timeout_ms=2_000) as client:
        response = client.generate_grasps(
            point_cloud,
            num_grasps=10,
            grasp_threshold=0.5,
            topk_num_grasps=3,
        )

    assert response.gripper_name == "robotiq_2f_85"
    assert response.grasps.shape == (3, 4, 4)
    assert response.confidences.shape == (3,)
    assert np.all(response.confidences >= 0.5)
    assert np.all(response.confidences[:-1] >= response.confidences[1:])


@pytest.mark.parametrize("planner", ["diffusion", "graspmoe"])
def test_generate_safe_grasps_filters_client_side(graspgenx_fake_port, planner) -> None:
    object_point_cloud = np.zeros((2048, 3), dtype=np.float32)
    scene_point_cloud = np.ones((1024, 3), dtype=np.float32)
    with GraspGenXClient(host="127.0.0.1", port=graspgenx_fake_port, timeout_ms=2_000) as client:
        response = client.generate_safe_grasps(
            object_point_cloud,
            scene_point_cloud,
            planner=planner,
            num_grasps=8,
            grasp_threshold=0.4,
            topk_num_grasps=3,
            moe_obb_density="dense-topandside",
        )

    assert response.gripper_name == "robotiq_2f_85"
    assert response.grasps.shape == (3, 4, 4)
    assert response.confidences.shape == (3,)
    assert np.all(response.confidences[:-1] >= response.confidences[1:])


@pytest.mark.parametrize("method_name", ["generate_grasps_for_all", "generate_safe_grasps_for_all"])
def test_scene_actions_batch_instances_and_report_skips(graspgenx_fake_port, method_name) -> None:
    depth = np.ones((3, 4), dtype=np.float32)
    mask = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    intrinsics = np.array(
        [[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with GraspGenXClient(host="127.0.0.1", port=graspgenx_fake_port, timeout_ms=2_000) as client:
        method = getattr(client, method_name)
        response = method(
            depth,
            intrinsics,
            mask,
            min_object_points=3,
            num_grasps=6,
            topk_num_grasps=2,
        )
        by_id = client.scene_results_by_id(response)

    assert response.gripper_name == "robotiq_2f_85"
    assert response.instance_ids.tolist() == [1]
    assert response.skipped_instance_ids.tolist() == [2]
    assert set(by_id) == {1}
    assert by_id[1][0].shape == (2, 4, 4)
    assert len(by_id[1][2]) == 2
