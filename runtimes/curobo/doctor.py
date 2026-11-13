"""Read-only dependency/import and driver probe; no planning or downloads."""

from __future__ import annotations

import importlib

from storage import configure_cache


def main() -> None:
    configure_cache()
    for name in (
        "backend",
        "curobo.motion_planner",
        "curobo.inverse_kinematics",
        "curobo.collision_checking",
        "curobo.scene",
    ):
        importlib.import_module(name)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo requires an accessible NVIDIA CUDA device")
    print(f"cuRobo real imports passed: torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
