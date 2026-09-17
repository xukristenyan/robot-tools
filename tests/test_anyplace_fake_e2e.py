from __future__ import annotations

import importlib.util
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn

from robot_tools.core.wire import CONTENT_TYPE, pack
from robot_tools.services.anyplace import AnyPlaceClient, masked_rgbd


@pytest.fixture()
def anyplace_port():
    path = Path(__file__).resolve().parents[1] / "runtimes/anyplace/server.py"
    spec = importlib.util.spec_from_file_location("anyplace_e2e_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    service = module.AnyPlaceServer(port=port, fake=True)
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


def test_masked_rgbd_typed_http_roundtrip(anyplace_port):
    obs = masked_rgbd(
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.ones((4, 6), dtype=np.float32),
        np.ones((4, 6), dtype=bool),
        np.array([[100, 0, 3], [0, 100, 2], [0, 0, 1]]),
    )
    with AnyPlaceClient(host="127.0.0.1", port=anyplace_port, timeout_ms=2000) as client:
        health = client.check_compatibility()
        assert health["service_id"] == "anyplace"
        assert health["api_version"] == "1"
        assert health["actions"] == ["predict_placements"]
        result = client.predict_placements(obs, obs, num_samples=3)
    np.testing.assert_array_equal(result.relative, np.tile(np.eye(4), (3, 1, 1)))
    assert result.relative.dtype == np.float64

    # Bypass client validation to verify the HTTP boundary independently.
    payload = {"action": "predict_placements", "pick": obs.model_dump(), "place": obs.model_dump()}
    payload["pick"]["mask"] = np.zeros((4, 6), dtype=bool)
    response = httpx.post(
        f"http://127.0.0.1:{anyplace_port}/predict_placements",
        content=pack(payload),
        headers={"Content-Type": CONTENT_TYPE},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_failed"
