"""Read-only import and tiny native CUDA probe; does not download weights."""

from backend import import_upstream


def main():
    import hydra
    import safetensors
    import timm
    import torch
    from torkit3d.ops.sample_farthest_points import sample_farthest_points

    model = import_upstream()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    points = torch.arange(192, dtype=torch.float32, device="cuda").reshape(1, 64, 3) / 192
    indices = sample_farthest_points(points, 8)
    torch.cuda.synchronize()
    assert indices.shape == (1, 8)
    assert torch.unique(indices).numel() == 8
    print(
        f"Point-SAM imports + CUDA FPS OK: {model.__name__}, torch={torch.__version__}, "
        f"timm={timm.__version__}, hydra={hydra.__version__}, safetensors={safetensors.__version__}, "
        f"gpu={torch.cuda.get_device_name(0)}"
    )


if __name__ == "__main__":
    main()
