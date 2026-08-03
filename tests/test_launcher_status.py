from __future__ import annotations

import httpx
import pytest

from robot_tools import __version__, launcher
from robot_tools.core import client as client_module
from robot_tools.core import wire
from robot_tools.services.fastfs.contract import ACTIONS as FASTFS_ACTIONS
from robot_tools.services.fastfs.contract import API_VERSION as FASTFS_API_VERSION
from robot_tools.services.sam3.contract import ACTIONS as SAM3_ACTIONS
from robot_tools.services.sam3.contract import API_VERSION as SAM3_API_VERSION


def _health(
    service_id: str,
    *,
    wire_version: str = wire.WIRE_PROTOCOL_VERSION,
    api_version: str | None = None,
    actions: set[str] | frozenset[str] | None = None,
) -> dict:
    defaults = {
        "fastfs": (FASTFS_API_VERSION, FASTFS_ACTIONS),
        "sam3": (SAM3_API_VERSION, SAM3_ACTIONS),
    }
    default_api, default_actions = defaults[service_id]
    return {
        "status": "ready",
        "service_id": service_id,
        "wire_protocol_version": wire_version,
        "api_version": api_version or default_api,
        "package_version": __version__,
        "actions": sorted(default_actions if actions is None else actions),
    }


def _endpoints(tmp_path, entries: str):
    path = tmp_path / "endpoints.yaml"
    path.write_text(f"endpoints:\n{entries}")
    return path


def _mock_health_by_port(monkeypatch, health_by_port: dict[int, dict]) -> None:
    real_client = httpx.Client

    def make_client(*args, **kwargs):
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health"
            return httpx.Response(
                200,
                request=request,
                json=health_by_port[request.url.port],
            )

        kwargs["transport"] = httpx.MockTransport(respond)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(client_module.httpx, "Client", make_client)


def test_status_accepts_compatible_typed_service(tmp_path, monkeypatch, capsys):
    path = _endpoints(tmp_path, "  fastfs: {host: 127.0.0.1, port: 5556}\n")
    _mock_health_by_port(monkeypatch, {5556: _health("fastfs")})

    assert launcher.status(path) == 0
    output = capsys.readouterr().out
    assert "fastfs" in output
    assert "✓" in output
    assert "service=fastfs" in output


def test_status_rejects_swapped_service_ports(tmp_path, monkeypatch, capsys):
    path = _endpoints(
        tmp_path,
        "  fastfs: {host: 127.0.0.1, port: 5558}\n  sam3: {host: 127.0.0.1, port: 5556}\n",
    )
    _mock_health_by_port(
        monkeypatch,
        {
            5556: _health("fastfs"),
            5558: _health("sam3"),
        },
    )

    assert launcher.status(path) == 1
    output = capsys.readouterr().out
    assert output.count("✗") == 2
    assert output.count("incompatible") == 2
    assert "✓" not in output


@pytest.mark.parametrize(
    "health",
    [
        _health("fastfs", wire_version="999"),
        _health("fastfs", api_version="999"),
        _health("fastfs", actions=set()),
    ],
)
def test_status_rejects_wire_api_and_action_mismatches(
    tmp_path,
    monkeypatch,
    capsys,
    health,
):
    path = _endpoints(tmp_path, "  fastfs: {host: 127.0.0.1, port: 5556}\n")
    _mock_health_by_port(monkeypatch, {5556: health})

    assert launcher.status(path) == 1
    output = capsys.readouterr().out
    assert "fastfs" in output
    assert "✗" in output
    assert "incompatible" in output
    assert "✓" not in output


@pytest.mark.parametrize(
    "field",
    ["service_id", "wire_protocol_version", "api_version"],
)
@pytest.mark.parametrize("malformed", [[], {}])
def test_status_reports_malformed_health_and_continues(
    tmp_path,
    monkeypatch,
    capsys,
    field,
    malformed,
):
    path = _endpoints(
        tmp_path,
        "  fastfs: {host: 127.0.0.1, port: 5556}\n  sam3: {host: 127.0.0.1, port: 5558}\n",
    )
    malformed_health = _health("fastfs")
    malformed_health[field] = malformed
    _mock_health_by_port(
        monkeypatch,
        {
            5556: malformed_health,
            5558: _health("sam3"),
        },
    )

    assert launcher.status(path) == 1
    output = capsys.readouterr().out
    assert "fastfs" in output
    assert "incompatible" in output
    assert "sam3" in output
    assert "sam3         ✓" in output
