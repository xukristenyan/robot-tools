"""Registry of available model servers.

Each entry tells the launcher how to start a server: where its uv project lives,
which command to run inside that project, and the default port.

Add a new server here after creating its `servers/<name>/` uv project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServerSpec:
    name: str
    project_dir: Path  # absolute path to the uv project for this server
    entry: list[str]  # command (after `uv run`) to start the server
    default_port: int
    gpu_required: bool = True


SERVER_REGISTRY: dict[str, ServerSpec] = {
    "graspgen": ServerSpec(
        name="graspgen",
        project_dir=REPO_ROOT / "servers" / "graspgen",
        entry=["python", "server.py"],
        default_port=5557,
        gpu_required=True,
    ),
    "fastfs": ServerSpec(
        name="fastfs",
        project_dir=REPO_ROOT / "servers" / "fastfs",
        entry=["python", "server.py"],
        default_port=5556,
        gpu_required=True,
    ),
    "sam3": ServerSpec(
        name="sam3",
        project_dir=REPO_ROOT / "servers" / "sam3",
        entry=["python", "server.py"],
        default_port=5558,
        gpu_required=True,
    ),
}


def get(name: str) -> ServerSpec:
    if name not in SERVER_REGISTRY:
        raise KeyError(
            f"unknown server: {name!r}. registered: {list(SERVER_REGISTRY)}"
        )
    return SERVER_REGISTRY[name]


def names() -> list[str]:
    return list(SERVER_REGISTRY)
