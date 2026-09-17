from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_tools import launcher
from robot_tools.services.anyplace import AnyPlaceClient, masked_rgbd
from robot_tools.services.anyplace.contract import PredictPlacementsRequest, PredictPlacementsResponse

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes/anyplace"


def observation(**overrides):
    values = {
        "rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "depth": np.full((8, 10), 0.5, dtype=np.float32),
        "mask": np.ones((8, 10), dtype=bool),
        "intrinsics": np.array([[100.0, 0, 4], [0, 100.0, 3], [0, 0, 1]]),
    }
    values.update(overrides)
    return masked_rgbd(**values)


def load_runtime(name):
    spec = importlib.util.spec_from_file_location(f"test_anyplace_{name}", RUNTIME / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "override",
    [
        {"rgb": np.zeros((8, 10, 3), dtype=np.float32)},
        {"depth": np.ones((8, 10), dtype=np.uint16)},
        {"depth": np.full((8, 10), np.nan, dtype=np.float32)},
        {"depth": np.zeros((8, 10), dtype=np.float32)},
        {"mask": np.zeros((8, 10), dtype=bool)},
        {"mask": np.ones((2, 2), dtype=bool)},
        {"mask": np.full((8, 10), 2, dtype=np.uint8)},
        {"intrinsics": np.zeros((3, 3))},
        {"intrinsics": np.array([[100, 1, 4], [0, 100, 3], [0, 0, 1]])},
        {"camera_pose": np.diag([1.0, 1.0, -1.0, 1.0])},
    ],
)
def test_reject_invalid_observations(override):
    with pytest.raises(ValueError):
        observation(**override)


def test_rgbd_projection_and_frame_transform():
    depth = np.full((8, 10), 0.5, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0
    pose = np.array([[0.0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    obs = observation(depth=depth, mask=np.full((8, 10), 255, dtype=np.uint8), camera_pose=pose)
    cloud = load_runtime("geometry").point_cloud(obs)
    assert cloud.shape == (78, 3)
    # First valid pixel is (u=2,v=0): camera XYZ=(-.01,-.015,.5).
    np.testing.assert_allclose(cloud[0], [1.015, 1.99, 3.5])
    assert obs.mask.dtype == np.bool_


@pytest.mark.parametrize(
    "field,value", [("num_samples", 0), ("num_samples", 33), ("num_samples", 1.5), ("n_refine_iters", 1), ("seed", -1)]
)
def test_local_options_rejected_before_http(monkeypatch, field, value):
    client = AnyPlaceClient(wait=False)
    monkeypatch.setattr(client, "_request", lambda *_: pytest.fail("invalid input reached HTTP"))
    with client, pytest.raises(ValueError):
        client.predict_placements(observation(), observation(), **{field: value})


def test_client_preserves_request_semantics(monkeypatch):
    response = PredictPlacementsResponse(relative=np.eye(4)[None])
    with AnyPlaceClient(wait=False) as client:

        def request(req, response_type):
            assert req.num_samples == 2 and req.n_refine_iters == 10 and req.seed == 7
            assert req.action == "predict_placements"
            assert response_type is PredictPlacementsResponse
            return response

        monkeypatch.setattr(client, "_request", request)
        assert (
            client.predict_placements(observation(), observation(), num_samples=2, n_refine_iters=10, seed=7)
            is response
        )


@pytest.mark.parametrize(
    "relative",
    [
        np.eye(4),
        np.zeros((0, 4, 4)),
        np.zeros((1, 4, 4)),
        np.eye(4, dtype=np.float32)[None],
        np.diag([1.0, 1.0, -1.0, 1.0])[None],
    ],
)
def test_response_requires_native_rigid_transforms(relative):
    with pytest.raises(ValueError):
        PredictPlacementsResponse(relative=relative)


def test_fake_import_does_not_load_upstream_or_torch():
    script = f"""
import runpy, sys
module = runpy.run_path({str(RUNTIME / "server.py")!r})
server = module['AnyPlaceServer'](fake=True)
assert 'torch' not in sys.modules
assert not any(k == 'anyplace' or k.startswith('anyplace.') for k in sys.modules)
assert server.backend is None
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_fake_handler_deterministic():
    server = load_runtime("server").AnyPlaceServer(fake=True)
    request = PredictPlacementsRequest(pick=observation(), place=observation(), num_samples=3)
    first = server.predict_placements(request)
    second = server.predict_placements(request)
    np.testing.assert_array_equal(first.relative, np.tile(np.eye(4), (3, 1, 1)))
    np.testing.assert_array_equal(first.relative, second.relative)


def test_selection_initializes_only_anyplace(tmp_path, monkeypatch):
    profile = tmp_path / "services.yaml"
    profile.write_text("services:\n  - name: anyplace\n")
    calls = []
    monkeypatch.setattr(launcher, "_init_upstream", lambda spec: calls.append(spec.service_id))
    monkeypatch.setattr(launcher, "_sync_one", lambda *_: None)
    monkeypatch.setattr(launcher, "_run_runtime_script", lambda *_: None)
    launcher.install(profile)
    assert calls == ["anyplace"]


def test_status_registers_anyplace():
    assert launcher._status_client_types()["anyplace"] is AnyPlaceClient
