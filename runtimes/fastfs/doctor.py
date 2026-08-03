"""Read-only import probe for the real FastFoundationStereo runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parent / "upstream"
MODULES = (
    "cv2",
    "imageio",
    "omegaconf",
    "timm",
    "torch",
    "torchvision",
    "core.foundation_stereo",
)


def main() -> None:
    sys.path.insert(0, str(UPSTREAM))
    for module in MODULES:
        importlib.import_module(module)
    print(f"FastFoundationStereo real-runtime imports passed: {', '.join(MODULES)}")


if __name__ == "__main__":
    main()
