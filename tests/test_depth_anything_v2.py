from __future__ import annotations

import importlib.util
import os
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
from robot_tools.core.exceptions import ProtocolMismatchError
from robot_tools.core.wire import CONTENT_TYPE, pack
from robot_tools.services.depth_anything_v2 import DepthAnythingV2Client
from robot_tools.services.depth_anything_v2.contract import EstimateDepthRequest, EstimateDepthResponse

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes" / "depth_anything_v2"


def load_runtime(name):
    spec = importlib.util.spec_from_file_location(f"depth_anything_v2_test_{name}", RUNTIME / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 5), dtype=np.uint8),
        np.zeros((4, 5, 4), dtype=np.uint8),
        np.zeros((0, 5, 3), dtype=np.uint8),
        np.zeros((4, 5, 3), dtype=np.float32),
        np.zeros((4, 5, 3), dtype=np.uint16),
    ],
)
def test_client_rejects_invalid_images_before_connecting(image):
    with DepthAnythingV2Client(wait=False) as client, pytest.raises(ValidationError):
        client.estimate_depth(image)


@pytest.mark.parametrize("size", [0, -1, 13, 518.5, "518", True])
def test_input_size_rejects_invalid_values(size):
    with pytest.raises(ValidationError):
        EstimateDepthRequest(image=np.zeros((4, 5, 3), dtype=np.uint8), input_size=size)


@pytest.mark.parametrize(
    "depth",
    [
        np.ones((3, 4), dtype=np.float64),
        np.ones((3, 4, 1), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.full((3, 4), np.nan, dtype=np.float32),
        np.full((3, 4), np.inf, dtype=np.float32),
        np.full((3, 4), -1, dtype=np.float32),
    ],
)
def test_invalid_predictions_rejected(depth):
    with pytest.raises(ValidationError):
        EstimateDepthResponse(relative_inverse_depth=depth)


def test_client_preserves_rgb_and_rejects_wrong_output_resolution(monkeypatch):
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[:, ::-1]
    with DepthAnythingV2Client(wait=False) as client:

        def respond(request, response_cls):
            assert request.image.flags.c_contiguous
            np.testing.assert_array_equal(request.image, image)
            assert request.input_size == 700
            return response_cls(relative_inverse_depth=np.ones((5, 4), dtype=np.float32))

        monkeypatch.setattr(client, "_request", respond)
        with pytest.raises(ProtocolMismatchError, match="resolution"):
            client.estimate_depth(image, input_size=700)


@pytest.fixture()
def fake_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    service = load_runtime("server").DepthAnythingV2Server(port=port, fake=True)
    server = uvicorn.Server(uvicorn.Config(service._build_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_fake_http_roundtrip_status_and_server_validation(fake_port, tmp_path):
    image = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
    with DepthAnythingV2Client(host="127.0.0.1", port=fake_port, timeout_ms=2000) as client:
        health = client.check_compatibility()
        assert health["service_id"] == "depth_anything_v2"
        assert health["api_version"] == "1"
        assert health["actions"] == ["estimate_depth"]
        first = client.estimate_depth(image)
        second = client.estimate_depth(image, input_size=700)
        assert first.relative_inverse_depth.shape == (6, 10)
        assert first.relative_inverse_depth.dtype == np.float32
        assert first.timing_ms >= 0
        np.testing.assert_array_equal(first.relative_inverse_depth, second.relative_inverse_depth)

    profile = tmp_path / "endpoints.yaml"
    profile.write_text(f"endpoints:\n  depth_anything_v2: {{host: 127.0.0.1, port: {fake_port}}}\n")
    assert launcher.status(profile) == 0
    for payload in (
        {"image": image.astype(np.float32)},
        {"image": image, "input_size": 0},
    ):
        response = httpx.post(
            f"http://127.0.0.1:{fake_port}/estimate_depth",
            content=pack({"action": "estimate_depth", **payload}),
            headers={"Content-Type": CONTENT_TYPE},
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == "validation_failed"


def test_backend_rgb_conversion_and_unmodified_output():
    backend_cls = load_runtime("backend").DepthAnythingV2Backend
    backend = backend_cls.__new__(backend_cls)
    image = np.array([[[255, 13, 2], [4, 5, 6]]], dtype=np.uint8)
    expected = np.array([[42.5, 3.25]], dtype=np.float32)

    class Model:
        def infer_image(self, bgr, input_size):
            assert bgr.flags.c_contiguous
            np.testing.assert_array_equal(bgr, image[..., ::-1])
            assert input_size == 700
            return expected

    backend.model = Model()
    np.testing.assert_array_equal(backend.infer(image, input_size=700), expected)


def test_backend_checkpoint_paths(monkeypatch, tmp_path):
    backend = load_runtime("backend")
    monkeypatch.setenv("DEPTH_ANYTHING_V2_WEIGHTS_DIR", str(tmp_path))
    assert backend.checkpoint_path() == tmp_path / "depth_anything_v2_vitl.pth"
    explicit = tmp_path / "custom.pth"
    assert backend.checkpoint_path(explicit) == explicit


def test_server_forwards_input_size_override():
    service = load_runtime("server").DepthAnythingV2Server(fake=True, input_size=700)
    service._fake = False
    seen = []

    class Backend:
        def infer(self, image, *, input_size):
            seen.append(input_size)
            return np.full(image.shape[:2], 42, dtype=np.float32)

    service.backend = Backend()
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    service.estimate_depth(EstimateDepthRequest(image=image))
    result = service.estimate_depth(EstimateDepthRequest(image=image, input_size=518))
    assert seen == [700, 518]
    np.testing.assert_array_equal(result.relative_inverse_depth, np.full((3, 4), 42, dtype=np.float32))


def test_fake_imports_do_not_load_model_or_upstream():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from server import DepthAnythingV2Server; "
                "DepthAnythingV2Server(fake=True); "
                "assert all(m not in sys.modules for m in ('torch', 'cv2', 'backend', 'depth_anything_v2'))"
            ),
        ],
        cwd=RUNTIME,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_install_selection_only_initializes_this_upstream(monkeypatch, tmp_path):
    profile = tmp_path / "services.yaml"
    profile.write_text("services:\n  - name: depth_anything_v2\n")
    commands = []
    monkeypatch.setattr(launcher, "clear_real_install", lambda runtime_dir: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: commands.append(command))
    launcher.install(profile)
    upstream_calls = [command for command in commands if command[:2] == ["git", "submodule"]]
    assert upstream_calls == [
        ["git", "submodule", "update", "--init", "--recursive", "--", "runtimes/depth_anything_v2/upstream"]
    ]


@pytest.mark.parametrize("fail", [False, True])
def test_download_atomicity_and_existing_checkpoint_skip(tmp_path, fail):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text("#!/bin/bash\nprintf 'mock checkpoint' > \"${@: -1}\"\n" + ("exit 22\n" if fail else ""))
    curl.chmod(0o755)
    dest = tmp_path / "weights"
    env = {**os.environ, "DEST": str(dest), "PATH": f"{bindir}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(RUNTIME / "download.sh")], env=env, capture_output=True, text=True, check=False
    )
    target = dest / "depth_anything_v2_vitl.pth"
    assert (result.returncode != 0) == fail
    assert target.exists() != fail
    assert not list(dest.glob("*.partial"))
    if not fail:
        curl.write_text("#!/bin/bash\nexit 99\n")
        second = subprocess.run(
            ["bash", str(RUNTIME / "download.sh")], env=env, capture_output=True, text=True, check=False
        )
        assert second.returncode == 0
        assert "[skip]" in second.stdout
