from __future__ import annotations

import logging
from importlib.metadata import version

import httpx
import numpy as np
import pytest

from robot_tools import __version__
from robot_tools.core import wire
from robot_tools.core.exceptions import (
    ProtocolMismatchError,
    RemoteInferenceError,
    ServiceBusyError,
)
from robot_tools.services.fastfs import FastFSClient
from robot_tools.services.fastfs.contract import API_VERSION as FASTFS_API_VERSION
from robot_tools.services.graspgen import GraspGenClient
from robot_tools.services.graspgen.contract import API_VERSION as GRASPGEN_API_VERSION
from robot_tools.services.graspgenx import GraspGenXClient
from robot_tools.services.graspgenx.contract import API_VERSION as GRASPGENX_API_VERSION
from robot_tools.services.sam3 import SAM3Client
from robot_tools.services.sam3.contract import API_VERSION as SAM3_API_VERSION

SERVICE_ACTIONS = {
    "sam3": ["segment_by_point", "segment_by_text"],
    "fastfs": ["reconstruct"],
    "graspgen": [
        "generate_collision_free_grasps",
        "generate_grasps",
    ],
    "graspgenx": [
        "infer",
        "infer_object",
        "infer_scene_depth",
        "infer_scene_pc",
        "metadata",
    ],
}

SERVICE_API_VERSIONS = {
    "sam3": SAM3_API_VERSION,
    "fastfs": FASTFS_API_VERSION,
    "graspgen": GRASPGEN_API_VERSION,
    "graspgenx": GRASPGENX_API_VERSION,
}


def test_package_version_comes_from_installed_project_metadata():
    assert __version__ == version("robot-tools")


def _health(
    service_id: str,
    *,
    wire_version: str = wire.WIRE_PROTOCOL_VERSION,
    api_version: str | None = None,
    package_version: str = __version__,
    actions: list[str] | None = None,
) -> dict:
    return {
        "status": "ready",
        "service_id": service_id,
        "wire_protocol_version": wire_version,
        "api_version": api_version or SERVICE_API_VERSIONS[service_id],
        "package_version": package_version,
        "actions": SERVICE_ACTIONS[service_id] if actions is None else actions,
    }


def _check(client_cls, health: dict) -> None:
    client = client_cls(wait=False)
    try:
        client.health = lambda timeout_s=2.0: health
        client._ensure_compatible()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("client_cls", "service_id"),
    [
        (SAM3Client, "sam3"),
        (FastFSClient, "fastfs"),
        (GraspGenClient, "graspgen"),
        (GraspGenXClient, "graspgenx"),
    ],
)
def test_explicit_service_compatibility_is_accepted(
    client_cls,
    service_id,
):
    _check(client_cls, _health(service_id))


@pytest.mark.parametrize(
    ("client_cls", "service_id"),
    [
        (SAM3Client, "fastfs"),
        (FastFSClient, "sam3"),
        (GraspGenClient, "graspgenx"),
        (GraspGenXClient, "graspgen"),
    ],
)
def test_wrong_service_is_rejected_before_inference(
    client_cls,
    service_id,
):
    with pytest.raises(ProtocolMismatchError, match="cannot connect"):
        _check(client_cls, _health(service_id))


def test_wire_protocol_mismatch_is_rejected():
    with pytest.raises(ProtocolMismatchError, match="wire protocol"):
        _check(FastFSClient, _health("fastfs", wire_version="999"))


def test_service_api_mismatch_is_rejected():
    with pytest.raises(ProtocolMismatchError, match="FastFS|fastfs|API"):
        _check(FastFSClient, _health("fastfs", api_version="999"))


def test_missing_action_is_rejected():
    with pytest.raises(ProtocolMismatchError, match="missing required actions"):
        _check(FastFSClient, _health("fastfs", actions=[]))


def test_package_version_mismatch_only_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _check(
            FastFSClient,
            _health("fastfs", package_version="999.0.0"),
        )
    assert "wire/API checks passed" in caplog.text


def test_missing_health_field_is_rejected():
    info = _health("sam3")
    del info["actions"]
    with pytest.raises(ProtocolMismatchError, match="missing fields"):
        _check(SAM3Client, info)


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "service_id",
        "wire_protocol_version",
        "api_version",
        "package_version",
    ],
)
@pytest.mark.parametrize("malformed", [[], {}])
def test_non_string_health_fields_are_protocol_mismatches(field, malformed):
    info = _health("fastfs")
    info[field] = malformed

    with pytest.raises(ProtocolMismatchError, match=rf"invalid {field} field"):
        _check(FastFSClient, info)


def test_503_busy_maps_to_service_busy_error():
    request = httpx.Request("POST", "http://test/reconstruct")
    response = httpx.Response(
        503,
        request=request,
        headers={"X-Request-ID": "req-1"},
        json={
            "error_code": "service_busy",
            "message": "service is busy",
            "request_id": "req-1",
        },
    )
    error = FastFSClient._remote_error(
        response,
        "http://test/reconstruct",
    )
    assert isinstance(error, ServiceBusyError)
    assert error.request_id == "req-1"
    assert error.status_code == 503


def test_other_remote_error_maps_to_remote_inference_error():
    request = httpx.Request("POST", "http://test/reconstruct")
    response = httpx.Response(
        500,
        request=request,
        json={
            "error_code": "inference_failed",
            "message": "inference failed",
            "request_id": "req-2",
        },
    )
    error = FastFSClient._remote_error(
        response,
        "http://test/reconstruct",
    )
    assert type(error) is RemoteInferenceError
    assert error.error_code == "inference_failed"


def test_success_with_wrong_content_type_is_protocol_mismatch():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"Content-Type": "application/json"},
            json={"disparity": []},
        )

    client = FastFSClient(wait=False)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(respond),
    )
    client._compatible = True
    client._server_info = _health("fastfs")
    try:
        with pytest.raises(
            ProtocolMismatchError,
            match="Content-Type",
        ):
            client.reconstruct(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
            )
    finally:
        client.close()
