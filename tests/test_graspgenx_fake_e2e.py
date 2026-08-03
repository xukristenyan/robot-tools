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
    InferObjectRequest,
    InferObjectResponse,
    InferSceneDepthRequest,
    InferScenePointCloudRequest,
    SweepVolumeParams,
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


@pytest.fixture()
def sweep_volume() -> SweepVolumeParams:
    return SweepVolumeParams(
        extents_open=np.array([0.085, 0.04, 0.12], dtype=np.float32),
        offset_open=np.array([0.0, 0.0, 0.06], dtype=np.float32),
        extents_mid=np.array([0.045, 0.04, 0.12], dtype=np.float32),
        offset_mid=np.array([0.0, 0.0, 0.06], dtype=np.float32),
        gripper_type=0,
        fingertip_depth=0.12,
    )


def test_sweep_volume_roundtrip_and_validation(sweep_volume):
    flat = sweep_volume.to_flat()
    restored = SweepVolumeParams.from_flat(
        flat,
        gripper_type=sweep_volume.gripper_type,
        fingertip_depth=sweep_volume.fingertip_depth,
    )

    assert flat.shape == (12,)
    assert flat.dtype == np.float32
    np.testing.assert_array_equal(restored.to_flat(), flat)
    assert restored.cache_key() == sweep_volume.cache_key()
    assert restored.jaw_width == pytest.approx(0.085)

    with pytest.raises(ValidationError, match="positive"):
        SweepVolumeParams(
            extents_open=[0.0, 0.04, 0.12],
            offset_open=[0.0, 0.0, 0.06],
            extents_mid=[0.04, 0.04, 0.12],
            offset_mid=[0.0, 0.0, 0.06],
        )
    with pytest.raises(ValueError, match="12 values"):
        SweepVolumeParams.from_flat(np.zeros(11, dtype=np.float32))


def test_native_requests_reject_server_side_selection_fields(sweep_volume):
    with pytest.raises(ValidationError, match="grasp_threshold"):
        InferObjectRequest.model_validate(
            {
                "action": "infer_object",
                "point_cloud": np.zeros((10, 3), dtype=np.float32),
                "sweep_volume_params": sweep_volume.model_dump(),
                "grasp_threshold": 0.5,
            }
        )


def test_scene_contracts_validate_depth_dtype_and_mask_shape(sweep_volume):
    with pytest.raises(ValidationError, match="floating-point meters"):
        InferSceneDepthRequest(
            depth=np.ones((2, 2), dtype=np.uint16),
            intrinsics=np.eye(3),
            instance_mask=np.ones((2, 2), dtype=np.int32),
            sweep_volume_params=sweep_volume,
        )
    with pytest.raises(ValidationError, match="instance_mask size"):
        InferScenePointCloudRequest(
            point_cloud=np.zeros((5, 3), dtype=np.float32),
            instance_mask=np.ones(4, dtype=np.int32),
            sweep_volume_params=sweep_volume,
        )


def test_response_rejects_misaligned_branch_tags():
    with pytest.raises(ValidationError, match="lengths must match"):
        InferObjectResponse(
            grasps=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
            confidences=np.ones(2, dtype=np.float32),
            branch_tags=["diff"],
            timing={"infer_ms": 1.0},
        )


def test_health_and_metadata_expose_native_actions(graspgenx_fake_port):
    with GraspGenXClient(
        host="127.0.0.1",
        port=graspgenx_fake_port,
        timeout_ms=2_000,
    ) as client:
        health = client.health()
        metadata = client.server_metadata

    assert set(health["actions"]) == set(ACTIONS)
    assert health["api_version"] == "2"
    assert metadata.default_gripper == "robotiq_2f_85"
    assert metadata.loaded_grippers == ["robotiq_2f_85"]
    assert metadata.actions == sorted(ACTIONS)
    assert metadata.precision.tensorrt is False


def test_named_gripper_infer_uses_official_fields(graspgenx_fake_port):
    point_cloud = np.zeros((2048, 3), dtype=np.float32)
    with GraspGenXClient(
        host="127.0.0.1",
        port=graspgenx_fake_port,
        timeout_ms=2_000,
    ) as client:
        response = client.infer(
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
def test_object_infer_preserves_tags_and_filters_client_side(
    graspgenx_fake_port,
    sweep_volume,
    planner,
):
    point_cloud = np.zeros((2048, 3), dtype=np.float32)
    with GraspGenXClient(
        host="127.0.0.1",
        port=graspgenx_fake_port,
        timeout_ms=2_000,
    ) as client:
        response = client.infer_object(
            point_cloud,
            sweep_volume,
            planner=planner,
            num_grasps=8,
            grasp_threshold=0.4,
            topk_num_grasps=3,
            moe_obb_density="dense-topandside",
        )

    assert response.grasps.shape == (3, 4, 4)
    assert response.confidences.shape == (3,)
    assert len(response.branch_tags) == 3
    assert np.all(response.confidences[:-1] >= response.confidences[1:])
    if planner == "diffusion":
        assert set(response.branch_tags) == {"diff"}
    else:
        assert set(response.branch_tags) == {"diff", "obb"}


def test_scene_depth_batches_instances_and_reports_skips(
    graspgenx_fake_port,
    sweep_volume,
):
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

    with GraspGenXClient(
        host="127.0.0.1",
        port=graspgenx_fake_port,
        timeout_ms=2_000,
    ) as client:
        response = client.infer_scene_depth(
            depth,
            intrinsics,
            mask,
            sweep_volume,
            min_object_points=3,
            num_grasps=6,
            topk_num_grasps=2,
        )
        by_id = client.scene_results_by_id(response)

    assert response.instance_ids.tolist() == [1]
    assert response.skipped_instance_ids.tolist() == [2]
    assert set(by_id) == {1}
    assert by_id[1][0].shape == (2, 4, 4)
    assert len(by_id[1][2]) == 2


def test_scene_point_cloud_ignores_nonfinite_points(
    graspgenx_fake_port,
    sweep_volume,
):
    point_cloud = np.zeros((2, 4, 3), dtype=np.float32)
    point_cloud[1, 0] = np.nan
    mask = np.array([[3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.int32)

    with GraspGenXClient(
        host="127.0.0.1",
        port=graspgenx_fake_port,
        timeout_ms=2_000,
    ) as client:
        response = client.infer_scene_pc(
            point_cloud,
            mask,
            sweep_volume,
            planner="diffusion",
            min_object_points=4,
            num_grasps=4,
            topk_num_grasps=-1,
        )

    assert response.instance_ids.tolist() == [3]
    assert response.skipped_instance_ids.tolist() == [4]
    assert response.grasps[0].shape == (4, 4, 4)
    assert set(response.branch_tags[0]) == {"diff"}
