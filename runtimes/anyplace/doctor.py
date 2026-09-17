"""Read-only import and CUDA probe; installing does not require checkpoints."""

from backend import upstream_imports


def main():
    import torch
    from anyplace_fps import native_fps

    model, _, util, inference = upstream_imports()
    assert torch.__version__ == "2.10.0+cu128", torch.__version__
    if not torch.cuda.is_available():
        raise RuntimeError("AnyPlace requires a visible CUDA GPU")
    sample = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], device="cuda")
    _, indices = native_fps(sample, 2, return_idx=True)
    assert indices.tolist() == [[0, 2]]
    # Ensure the pinned headless patch is installed, without opening a viewer.
    util.meshcat_pcd_show(None, sample[0].cpu().numpy(), (255, 0, 0), "probe")
    print(f"AnyPlace real imports/CUDA passed: {model.__name__}, {inference.__name__}, {torch.__version__}")


if __name__ == "__main__":
    main()
