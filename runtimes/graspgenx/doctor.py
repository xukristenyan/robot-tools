"""Read-only import probe for the real GraspGenX runtime."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
UPSTREAM = RUNTIME_DIR / "upstream"
MODULES = (
    "torch",
    "graspgenx.grasp_server",
    "graspgenx.samplers",
    "graspgenx.utils.scene_loaders",
)


def _require_read_only_import_patches() -> None:
    package_init = (UPSTREAM / "graspgenx" / "__init__.py").read_text()
    active_lines = {line.strip() for line in package_init.splitlines()}
    forbidden_calls = {"ensure_checkpoints()", "ensure_gripper_descriptions()"}
    if active_lines.intersection(forbidden_calls):
        raise RuntimeError("GraspGenX no-auto-download compatibility patch is not applied; rerun install")

    serving_init = (UPSTREAM / "graspgenx" / "serving" / "__init__.py").read_text()
    if "def __getattr__(name):" not in serving_init:
        raise RuntimeError("GraspGenX lazy-ZMQ compatibility patch is not applied; rerun install")


def main() -> None:
    _require_read_only_import_patches()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    sys.path.insert(0, str(UPSTREAM))
    for module in MODULES:
        importlib.import_module(module)
    print(f"GraspGenX real-runtime imports passed: {', '.join(MODULES)}")


if __name__ == "__main__":
    main()
