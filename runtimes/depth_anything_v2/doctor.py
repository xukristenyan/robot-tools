"""Read-only real imports; checkpoint download is a separate lifecycle step."""

from backend import upstream_model_class


def main():
    import cv2
    import torch
    import torchvision

    model = upstream_model_class()
    print(
        f"Depth Anything V2 imports passed: {model.__name__}, "
        f"torch={torch.__version__}, torchvision={torchvision.__version__}, cv2={cv2.__version__}"
    )


if __name__ == "__main__":
    main()
