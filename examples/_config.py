"""Tiny helper to read host/port from services.yaml by service name.

Mirrors what a real downstream project does: keep one services.yaml, look
up `(host, port)` per service. Scripts in this folder use it instead of
`--host`/`--port` flags so the examples and the launcher stay in sync —
edit services.yaml once and both follow.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_SERVICES_PATH = _HERE / "services.yaml"
_CONFIGS_DIR = _HERE / "configs"


def load_service(name: str, path: Path | None = None) -> tuple[str, int]:
    cfg = yaml.safe_load((path or _SERVICES_PATH).read_text())
    for entry in cfg.get("services", []):
        if entry["name"] == name:
            return entry.get("host", "127.0.0.1"), int(entry["port"])
    raise KeyError(f"service '{name}' not found in {path or _SERVICES_PATH}")


def load_params(name: str) -> dict:
    """Read configs/<name>.yaml as a dict of per-module tuning parameters."""
    return yaml.safe_load((_CONFIGS_DIR / f"{name}.yaml").read_text())
