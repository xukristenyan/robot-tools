"""Integration tests for the HTTP/MessagePack server transport."""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import ClassVar, Literal

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from robot_tools import __version__
from robot_tools.core import wire
from robot_tools.core.client import BaseClient
from robot_tools.core.exceptions import (
    InferenceTimeoutError,
    ServiceBusyError,
)
from robot_tools.core.server import BaseServer, handler
from robot_tools.core.wire import BaseRequest, BaseResponse

DUMMY_SERVICE_ID = "dummy"
DUMMY_API_VERSION = "1"
SECRET = "secret-model-path=/private/models/weights.pt"


class EchoRequest(BaseRequest):
    action: Literal["echo"] = "echo"
    image: np.ndarray
    gain: float = 1.0


class EchoResponse(BaseResponse):
    image: np.ndarray


class BoomRequest(BaseRequest):
    action: Literal["boom"] = "boom"


class HoldRequest(BaseRequest):
    action: Literal["hold"] = "hold"
    label: str


class BadResponseRequest(BaseRequest):
    action: Literal["bad_response"] = "bad_response"


class BadResponse(BaseResponse):
    payload: object


class DummyServer(BaseServer):
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        max_request_bytes: int = 64 * 1024 * 1024,
    ):
        self.hold_started = threading.Event()
        self.release_hold = threading.Event()
        self.hold_executions: list[str] = []
        super().__init__(
            service_id=DUMMY_SERVICE_ID,
            api_version=DUMMY_API_VERSION,
            host=host,
            port=port,
            max_request_bytes=max_request_bytes,
        )
        self.mark_ready()

    @handler(EchoRequest)
    def echo(self, req: EchoRequest) -> EchoResponse:
        return EchoResponse(image=req.image * req.gain)

    @handler(BoomRequest)
    def boom(self, req: BoomRequest) -> BaseResponse:
        raise ValueError(SECRET)

    @handler(HoldRequest)
    def hold(self, req: HoldRequest) -> BaseResponse:
        self.hold_executions.append(req.label)
        self.hold_started.set()
        if not self.release_hold.wait(timeout=10):
            raise TimeoutError("test did not release held inference")
        return BaseResponse()

    @handler(BadResponseRequest)
    def bad_response(self, req: BadResponseRequest) -> BadResponse:
        return BadResponse(payload=object())


class EchoClient(BaseClient):
    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {DUMMY_SERVICE_ID: frozenset({DUMMY_API_VERSION})}
    REQUIRED_ACTIONS = frozenset({"echo"})

    def echo(self, image: np.ndarray, gain: float = 1.0) -> EchoResponse:
        return self._request(
            EchoRequest(image=image, gain=gain),
            EchoResponse,
        )


@pytest.fixture()
def server_and_client():
    server = DummyServer()
    with TestClient(server._build_app()) as client:
        yield server, client


def _post(client: TestClient, action: str, payload: object):
    return client.post(
        f"/{action}",
        content=wire.pack(payload),
        headers={"Content-Type": wire.CONTENT_TYPE},
    )


def _wait_for_admitted(server: BaseServer, expected: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if server.admitted_request_count == expected:
            return
        time.sleep(0.005)
    raise AssertionError(f"expected {expected} admitted requests, got {server.admitted_request_count}")


def test_array_roundtrip_preserves_shape_dtype_and_values(
    server_and_client,
):
    _server, client = server_and_client
    image = np.random.default_rng(7).random((4, 6, 3)).astype(np.float32)
    response = _post(
        client,
        "echo",
        {"action": "echo", "image": image, "gain": 2.0},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(wire.CONTENT_TYPE)
    assert response.headers["X-Request-ID"]
    parsed = EchoResponse.model_validate(wire.unpack(response.content))
    assert parsed.image.shape == image.shape
    assert parsed.image.dtype == np.float32
    np.testing.assert_allclose(parsed.image, image * 2.0, rtol=1e-6)
    assert parsed.timing_ms is not None


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Content-Type": "application/json"},
        {"Content-Type": "application/octet-stream"},
    ],
)
def test_content_type_is_required(server_and_client, headers):
    _server, client = server_and_client
    response = client.post(
        "/echo",
        content=wire.pack({"action": "echo", "image": np.zeros((1, 1))}),
        headers=headers,
    )
    assert response.status_code == 415
    assert response.json()["error_code"] == "unsupported_media_type"


def test_content_type_parameter_is_accepted(server_and_client):
    _server, client = server_and_client
    response = client.post(
        "/echo",
        content=wire.pack({"action": "echo", "image": np.zeros((1, 1))}),
        headers={"Content-Type": f"{wire.CONTENT_TYPE}; version=1"},
    )
    assert response.status_code == 200


def test_declared_oversized_body_is_413():
    server = DummyServer(max_request_bytes=16)
    with TestClient(server._build_app()) as client:
        response = client.post(
            "/echo",
            content=b"x" * 17,
            headers={"Content-Type": wire.CONTENT_TYPE},
        )
    assert response.status_code == 413
    assert response.json()["error_code"] == "request_too_large"


def test_streamed_oversized_body_is_413():
    server = DummyServer(max_request_bytes=16)
    with TestClient(server._build_app()) as client:
        response = client.post(
            "/echo",
            content=iter([b"x" * 8, b"y" * 9]),
            headers={"Content-Type": wire.CONTENT_TYPE},
        )
    assert response.status_code == 413
    assert response.json()["error_code"] == "request_too_large"


@pytest.mark.parametrize(
    "raw",
    [b"\xc1\xc1\xc1", wire.pack(["not", "a", "map"])],
)
def test_invalid_msgpack_is_400(server_and_client, raw):
    _server, client = server_and_client
    response = client.post(
        "/echo",
        content=raw,
        headers={"Content-Type": wire.CONTENT_TYPE},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_msgpack"


def test_pydantic_validation_failure_is_422(server_and_client):
    _server, client = server_and_client
    response = _post(client, "echo", {"action": "echo"})
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_failed"
    assert "image" not in response.json()["message"]


def test_handler_exception_hides_traceback_and_returns_request_id(
    server_and_client,
    caplog,
):
    _server, client = server_and_client
    with caplog.at_level(logging.ERROR):
        response = _post(client, "boom", {"action": "boom"})

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "inference_failed"
    assert body["message"] == "inference failed"
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert SECRET not in response.text
    assert "Traceback" not in response.text
    assert SECRET in caplog.text
    assert body["request_id"] in caplog.text


def test_response_serialization_failure_is_safe(
    server_and_client,
    caplog,
):
    _server, client = server_and_client
    with caplog.at_level(logging.ERROR):
        response = _post(
            client,
            "bad_response",
            {"action": "bad_response"},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "response_serialization_failed"
    assert body["message"] == "response serialization failed"
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert "TypeError" not in response.text
    assert body["request_id"] in caplog.text


def test_unknown_action_is_404(server_and_client):
    _server, client = server_and_client
    assert _post(client, "nope", {"action": "nope"}).status_code == 404


def test_health_reports_readiness_and_compatibility_metadata(
    server_and_client,
):
    server, client = server_and_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service_id": DUMMY_SERVICE_ID,
        "wire_protocol_version": wire.WIRE_PROTOCOL_VERSION,
        "api_version": DUMMY_API_VERSION,
        "package_version": __version__,
        "actions": ["bad_response", "boom", "echo", "hold"],
    }

    server.mark_not_ready()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    inference = _post(
        client,
        "echo",
        {"action": "echo", "image": np.zeros((1, 1))},
    )
    assert inference.status_code == 503
    assert inference.json()["error_code"] == "service_not_ready"


def test_health_and_bounded_pending_capacity_are_deterministic():
    server = DummyServer()
    results: dict[str, object] = {}

    with TestClient(server._build_app()) as client:

        def request(label: str) -> None:
            results[label] = _post(
                client,
                "hold",
                {"action": "hold", "label": label},
            )

        first = threading.Thread(target=request, args=("first",))
        second = threading.Thread(target=request, args=("second",))
        first.start()
        assert server.hold_started.wait(timeout=2)

        health_started = time.monotonic()
        health = client.get("/health")
        assert health.status_code == 200
        assert time.monotonic() - health_started < 0.5

        second.start()
        _wait_for_admitted(server, 2)
        busy_started = time.monotonic()
        busy = _post(
            client,
            "hold",
            {"action": "hold", "label": "third"},
        )
        assert busy.status_code == 503
        assert busy.json()["error_code"] == "service_busy"
        assert time.monotonic() - busy_started < 0.5

        server.release_hold.set()
        first.join(timeout=3)
        second.join(timeout=3)

        assert not first.is_alive()
        assert not second.is_alive()
        assert results["first"].status_code == 200
        assert results["second"].status_code == 200
        assert server.hold_executions == ["first", "second"]
        assert server.admitted_request_count == 0


def test_busy_state_is_isolated_between_server_instances():
    sam = DummyServer()
    fastfs = DummyServer()
    sam_results: list[object] = []

    with (
        TestClient(sam._build_app()) as sam_client,
        TestClient(fastfs._build_app()) as fastfs_client,
    ):

        def hold_sam(label: str) -> None:
            sam_results.append(
                _post(
                    sam_client,
                    "hold",
                    {"action": "hold", "label": label},
                )
            )

        first = threading.Thread(target=hold_sam, args=("first",))
        second = threading.Thread(target=hold_sam, args=("second",))
        first.start()
        assert sam.hold_started.wait(timeout=2)
        second.start()
        _wait_for_admitted(sam, 2)

        response = _post(
            fastfs_client,
            "echo",
            {
                "action": "echo",
                "image": np.ones((2, 2), dtype=np.uint8),
            },
        )
        assert response.status_code == 200
        assert fastfs.admitted_request_count == 0

        sam.release_hold.set()
        first.join(timeout=3)
        second.join(timeout=3)
        assert len(sam_results) == 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def live_server():
    port = _free_port()
    server = DummyServer(host="127.0.0.1", port=port)
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
        raise RuntimeError("uvicorn did not start")
    yield server, port
    server.release_hold.set()
    uvicorn_server.should_exit = True
    thread.join(timeout=5)


def test_timeout_keeps_capacity_until_handler_finishes_and_client_reuses(
    live_server,
):
    server, port = live_server
    endpoint = {"host": "127.0.0.1", "port": port}

    with (
        EchoClient(**endpoint, timeout_ms=100) as timed_client,
        EchoClient(**endpoint, timeout_ms=5_000) as waiting_client,
        EchoClient(**endpoint, timeout_ms=1_000) as busy_client,
    ):
        with pytest.raises(InferenceTimeoutError):
            timed_client._request(
                HoldRequest(label="timed-out"),
                BaseResponse,
            )
        assert server.hold_started.wait(timeout=2)
        _wait_for_admitted(server, 1)

        waiting_result: list[BaseResponse] = []

        def submit_waiter() -> None:
            waiting_result.append(
                waiting_client._request(
                    HoldRequest(label="waiting"),
                    BaseResponse,
                )
            )

        waiter = threading.Thread(target=submit_waiter)
        waiter.start()
        _wait_for_admitted(server, 2)

        with pytest.raises(ServiceBusyError):
            busy_client._request(
                HoldRequest(label="busy"),
                BaseResponse,
            )

        server.release_hold.set()
        waiter.join(timeout=5)
        assert not waiter.is_alive()
        assert len(waiting_result) == 1
        _wait_for_admitted(server, 0)

        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        response = timed_client.echo(image, gain=2)
        np.testing.assert_array_equal(response.image, image * 2)
        assert server.hold_executions == ["timed-out", "waiting"]
