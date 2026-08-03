"""Registry of deployable service runtimes.

Each entry tells the launcher where a service's runtime project lives, which
command starts its HTTP server, and which port it uses by default.

Add a service here after creating its ``runtimes/<service>/`` uv project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServiceRuntimeSpec:
    service_id: str
    runtime_dir: Path
    server_command: tuple[str, ...]  # command after ``uv run``
    default_port: int
    gpu_required: bool = True


SERVICE_REGISTRY: dict[str, ServiceRuntimeSpec] = {
    "graspgen": ServiceRuntimeSpec(
        service_id="graspgen",
        runtime_dir=REPO_ROOT / "runtimes" / "graspgen",
        server_command=("python", "server.py"),
        default_port=5557,
        gpu_required=True,
    ),
    "graspgenx": ServiceRuntimeSpec(
        service_id="graspgenx",
        runtime_dir=REPO_ROOT / "runtimes" / "graspgenx",
        server_command=("python", "server.py"),
        default_port=5559,
        gpu_required=True,
    ),
    "fastfs": ServiceRuntimeSpec(
        service_id="fastfs",
        runtime_dir=REPO_ROOT / "runtimes" / "fastfs",
        server_command=("python", "server.py"),
        default_port=5556,
        gpu_required=True,
    ),
    "sam3": ServiceRuntimeSpec(
        service_id="sam3",
        runtime_dir=REPO_ROOT / "runtimes" / "sam3",
        server_command=("python", "server.py"),
        default_port=5558,
        gpu_required=True,
    ),
}


def get_service_runtime(service_id: str) -> ServiceRuntimeSpec:
    if service_id not in SERVICE_REGISTRY:
        raise KeyError(f"unknown service: {service_id!r}. registered: {list(SERVICE_REGISTRY)}")
    return SERVICE_REGISTRY[service_id]


def service_ids() -> list[str]:
    return list(SERVICE_REGISTRY)
