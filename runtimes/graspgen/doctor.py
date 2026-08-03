"""Read-only import probe for the real GraspGen runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
UPSTREAM = RUNTIME_DIR / "upstream"
MODULES = (
    "diffusers",
    "pointnet2_ops.pointnet2_utils",
    "spconv.pytorch",
    "torch",
    "torch_scatter",
    "grasp_gen.grasp_server",
)


def _require_lazy_ptv3_patch() -> None:
    eager_import = "from grasp_gen.models.ptv3.ptv3 import PointTransformerV3"
    for relative_path in (
        "grasp_gen/models/discriminator.py",
        "grasp_gen/models/generator.py",
    ):
        source = (UPSTREAM / relative_path).read_text()
        if any(line == eager_import for line in source.splitlines()):
            raise RuntimeError("GraspGen lazy-PTV3 compatibility patch is not applied; rerun install")


def main() -> None:
    _require_lazy_ptv3_patch()
    sys.path.insert(0, str(UPSTREAM))
    for module in MODULES:
        importlib.import_module(module)
    print(f"GraspGen real-runtime imports passed: {', '.join(MODULES)}")


if __name__ == "__main__":
    main()
