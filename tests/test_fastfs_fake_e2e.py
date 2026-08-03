from __future__ import annotations

import importlib.util
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from robot_tools.services.fastfs import FastFSClient


def _load_fastfs_server_class():
    path = Path(__file__).resolve().parents[1] / "runtimes" / "fastfs" / "server.py"
    spec = importlib.util.spec_from_file_location(
        "robot_tools_test_fastfs_server",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FastFSServer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def fastfs_fake_port():
    port = _free_port()
    server_cls = _load_fastfs_server_class()
    server = server_cls(host="127.0.0.1", port=port, fake=True)
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
        raise RuntimeError("FastFS fake uvicorn did not start")
    yield port
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


def test_real_fastfs_fake_server_and_typed_client_roundtrip(
    fastfs_fake_port,
):
    left = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    right = np.flip(left, axis=1)
    with FastFSClient(
        host="127.0.0.1",
        port=fastfs_fake_port,
        timeout_ms=2_000,
    ) as client:
        response = client.reconstruct(left, right)

    assert response.disparity.shape == (6, 8)
    assert response.disparity.dtype == np.float32
    np.testing.assert_array_equal(
        response.disparity,
        np.zeros((6, 8), dtype=np.float32),
    )
