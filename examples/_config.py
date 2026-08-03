"""Tiny helpers for the example scripts.

Mirrors what a real downstream project does: connection info comes from
`endpoints.yaml` (client side), per-service tuning from `configs/<name>.yaml`.
Scripts use these instead of `--host`/`--port` flags so editing the yaml once
is enough.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_tools.core.endpoints import Endpoint, get_endpoint

_HERE = Path(__file__).resolve().parent
_ENDPOINTS_PATH = _HERE / "endpoints.yaml"
_CONFIGS_DIR = _HERE / "configs"


def load_endpoint(name: str, path: Path | None = None) -> Endpoint:
    """Load one typed client endpoint from the example connection profile."""
    return get_endpoint(name, path or _ENDPOINTS_PATH)


def load_params(name: str) -> dict:
    """Read configs/<name>.yaml as a dict of per-service tuning parameters."""
    return yaml.safe_load((_CONFIGS_DIR / f"{name}.yaml").read_text())
