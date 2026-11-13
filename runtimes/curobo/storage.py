"""Keep runtime caches and uploaded geometry on the model filesystem."""

from __future__ import annotations

import getpass
import os
from pathlib import Path


def storage_root() -> Path:
    models = os.environ.get("ROBOT_TOOLS_MODELS_ROOT", f"/data/{getpass.getuser()}/robot-tools-models")
    return Path(os.environ.get("CUROBO_DATA_DIR", str(Path(models) / "curobo"))).expanduser()


def configure_cache() -> Path:
    root = storage_root()
    for name, folder in {
        "XDG_CACHE_HOME": "cache",
        "CUDA_CACHE_PATH": "cache/cuda",
        "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "WARP_CACHE_PATH": "cache/warp",
    }.items():
        os.environ.setdefault(name, str(root / folder))
    # The public runtime module re-exports immutable values. The pinned
    # upstream reads this defining module, so set its cache before any solver.
    from curobo._src import runtime

    runtime.cache_dir = str(root / "cache" / "curobo")
    return root
