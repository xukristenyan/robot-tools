"""Read-only import probe for the real SAM3 runtime."""

from __future__ import annotations

import importlib

MODULES = (
    "PIL",
    "torch",
    "torchvision",
    "sam3",
    "sam3.model_builder",
    "sam3.model.sam3_image_processor",
)


def main() -> None:
    for module in MODULES:
        importlib.import_module(module)
    print(f"SAM3 real-runtime imports passed: {', '.join(MODULES)}")


if __name__ == "__main__":
    main()
