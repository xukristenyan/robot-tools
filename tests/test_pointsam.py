from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn
from pydantic import ValidationError

from robot_tools import launcher
from robot_tools.core.exceptions import ProtocolMismatchError, RemoteInferenceError
from robot_tools.core.wire import CONTENT_TYPE, pack
from robot_tools.registry import get_service_runtime
from robot_tools.services.pointsam import PointSAMClient
from robot_tools.services.pointsam.contract import (
    ACTIONS,
    EncodeCloudRequest,
    SegmentRequest,
    SegmentResponse,
)

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes/pointsam"


def load_server():
    spec = importlib.util.spec_from_file_location("pointsam_test_server", RUNTIME / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PointSAMServer


def cloud():
    xyz = np.random.default_rng(12).uniform(-0.1, 0.1, (512, 3)).astype(np.float32)
    rgb = np.full((512, 3), 120, dtype=np.uint8)
    return xyz, rgb


@pytest.mark.parametrize(
    "change",
    [
        {"xyz": np.zeros((512, 3), dtype=np.float32)},
        {"xyz": np.ones((512, 3), dtype=np.int64)},
        {"xyz": np.full((512, 3), np.nan, dtype=np.float32)},
        {"xyz": np.zeros((2, 3), dtype=np.float32)},
        {"rgb": np.zeros((512, 3), dtype=np.float32)},
        {"rgb": np.zeros((511, 3), dtype=np.uint8)},
        {"num_groups": True},
        {"group_size": 513},
        {"num_groups": 513},
    ],
)
def test_cloud_rejection_before_connecting(change):
    xyz, rgb = cloud()
    with PointSAMClient(wait=False) as client, pytest.raises(ValidationError):
        client.encode_cloud(**({"xyz": xyz, "rgb": rgb} | change))


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_labels": np.array([2])},
        {"prompt_labels": np.array([1.0])},
        {"prompt_labels": np.array([1, 0])},
        {"prompt_points": np.zeros((0, 3), np.float32)},
        {"prompt_points": np.full((1, 3), np.inf, np.float32)},
        {"prompt_mask": np.ones(512, dtype=bool)},
        {"prompt_mask": np.full(512, np.nan, np.float32)},
        {"multimask_output": "true"},
    ],
)
def test_prompt_rejection_before_connecting(change):
    kwargs = {"cloud_id": "example", "prompt_points": np.zeros((1, 3), np.float32), "prompt_labels": np.array([1])}
    with PointSAMClient(wait=False) as client, pytest.raises(ValidationError):
        client.segment(**(kwargs | change))


def test_response_validates_shapes_order_and_logits():
    logits = np.ones((3, 512), np.float32)
    valid = {"cloud_id": "x", "masks": logits > 0, "logits": logits, "scores": np.array([0.9, 0.8, 0.7], np.float32)}
    SegmentResponse(**valid)
    for change in (
        {"scores": np.array([0.7, 0.9, 0.8], np.float32)},
        {"logits": logits.astype(np.float64)},
        {"masks": np.zeros_like(logits, dtype=bool)},
        {"scores": np.array([np.nan, 0.8, 0.7], np.float32)},
    ):
        with pytest.raises(ValidationError):
            SegmentResponse(**(valid | change))


@pytest.fixture()
def fake_service():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    service = load_server()(port=port, fake=True, max_clouds=2)
    server = uvicorn.Server(uvicorn.Config(service._build_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield port, service
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_fake_http_refinement_lifecycle_and_status(fake_service, tmp_path):
    port, service = fake_service
    xyz, rgb = cloud()
    with PointSAMClient(host="127.0.0.1", port=port) as client:
        info = client.check_compatibility()
        assert info["service_id"] == "pointsam"
        assert info["api_version"] == "1"
        assert set(info["actions"]) == ACTIONS
        encoded = client.encode_cloud(xyz, rgb)
        first = client.segment(encoded.cloud_id, xyz[:1], np.array([1]))
        assert first.masks.shape == (3, 512)
        assert first.timing_ms >= 0
        refined = client.segment(encoded.cloud_id, xyz[:2], np.array([1, 0]), prompt_mask=first.logits[0])
        assert refined.masks.shape == (1, 512)
        repeated = client.segment(encoded.cloud_id, xyz[:1], np.array([1]))
        np.testing.assert_array_equal(first.logits, repeated.logits)  # prompts are per call
        assert len(service.clouds) == 1
        assert client.release_cloud(encoded.cloud_id).released
        assert not client.release_cloud(encoded.cloud_id).released
        with pytest.raises(RemoteInferenceError):
            client.segment(encoded.cloud_id, xyz[:1], np.array([1]))
    assert not service.clouds
    profile = tmp_path / "endpoints.yaml"
    profile.write_text(f"endpoints:\n  pointsam: {{host: 127.0.0.1, port: {port}}}\n")
    assert launcher.status(profile) == 0
    response = httpx.post(
        f"http://127.0.0.1:{port}/encode_cloud",
        content=pack({"action": "encode_cloud", "xyz": xyz, "rgb": rgb.astype(np.float32)}),
        headers={"Content-Type": CONTENT_TYPE},
    )
    assert response.status_code == 422


def test_frame_invariance_and_point_order():
    service = load_server()(fake=True)
    xyz, rgb = cloud()
    xyz = xyz.astype(np.float64)
    perm = np.random.default_rng(3).permutation(len(xyz))
    encoded = service.encode_cloud(EncodeCloudRequest(xyz=xyz, rgb=rgb))
    transformed = service.encode_cloud(EncodeCloudRequest(xyz=xyz[perm] * 3 + [1, -2, 4], rgb=rgb[perm]))
    first = service.segment(
        SegmentRequest(cloud_id=encoded.cloud_id, prompt_points=xyz[:2], prompt_labels=np.array([1, 0]))
    )
    second = service.segment(
        SegmentRequest(
            cloud_id=transformed.cloud_id, prompt_points=xyz[:2] * 3 + [1, -2, 4], prompt_labels=np.array([1, 0])
        )
    )
    np.testing.assert_allclose(first.logits[:, perm], second.logits, atol=1e-6)
    np.testing.assert_array_equal(first.masks[:, perm], second.masks)


def test_cache_capacity_invalid_prompts_and_recovery(fake_service):
    port, service = fake_service
    xyz, rgb = cloud()
    with PointSAMClient(host="127.0.0.1", port=port) as client:
        first, second = client.encode_cloud(xyz, rgb), client.encode_cloud(xyz, rgb)
        with pytest.raises(RemoteInferenceError):
            client.encode_cloud(xyz, rgb)
        assert len(service.clouds) == 2  # no silent eviction
        with pytest.raises(RemoteInferenceError):
            client.segment(first.cloud_id, xyz[:1], np.array([1]), prompt_mask=np.ones(3, np.float32))
        with pytest.raises(RemoteInferenceError):
            client.segment(first.cloud_id, np.array([[20.0, 20.0, 20.0]]), np.array([1]))
        assert client.health_check()
        client.release_cloud(first.cloud_id)
        client.release_cloud(second.cloud_id)
        third = client.encode_cloud(xyz, rgb)
        assert third.cloud_id not in (first.cloud_id, second.cloud_id)


def test_fake_runtime_has_no_heavy_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from server import PointSAMServer; PointSAMServer(fake=True); "
                "assert not ({'torch','timm','pc_sam','torkit3d','backend'} & set(sys.modules))"
            ),
        ],
        cwd=RUNTIME,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_external_source_does_not_initialize_other_submodules(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    launcher._init_upstream(get_service_runtime("pointsam"))
    assert calls == []


def test_client_rejects_changed_cloud_count(monkeypatch):
    xyz, rgb = cloud()
    with PointSAMClient(wait=False) as client:
        monkeypatch.setattr(client, "_request", lambda req, cls: cls(cloud_id="x", num_points=511))
        with pytest.raises(ProtocolMismatchError):
            client.encode_cloud(xyz, rgb)
