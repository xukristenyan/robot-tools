# Depth Anything V2 Large prerequisites

## Pinned inputs

| Component | Selection |
|---|---|
| Runtime | Linux x86_64, uv-managed Python 3.12 |
| Model packages | Torch 2.10.0 + CUDA 12.8, torchvision 0.25.0, headless OpenCV |
| Upstream | `https://github.com/DepthAnything/Depth-Anything-V2.git` |
| Upstream commit | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| Hugging Face repo | `depth-anything/Depth-Anything-V2-Large` |
| Weight revision | `cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9` |
| Weight filename | `depth_anything_v2_vitl.pth` (1,341,395,338 bytes) |

Only relative-depth Large is supported. The metric weights use a different
head and output semantics. Neither metric weights nor other model sizes are
downloaded by this runtime.

## Host requirements

- `uv >= 0.11.21`, Git, Bash, and curl; outbound access to GitHub, Hugging Face,
  PyPI, and the PyTorch wheel index for installation.
- Persistent writable model storage (at least 1.34 GB for this checkpoint),
  plus space for the independent environment and shared uv cache.
- A CUDA-capable GPU and compatible driver for GPU inference; the repository
  baseline is RTX 5090 / driver 580.173.02. CPU inference is also supported.
- No Hugging Face login, Gradio, xformers, custom CUDA compilation, or upstream
  package installation is required for this inference path. The optional
  upstream "xFormers not available" messages are expected.

For this host, models live at
`/data/xkyan3/robot-tools-models/depth_anything_v2/depth_anything_v2_vitl.pth`.
Use `DEPTH_ANYTHING_V2_WEIGHTS_DIR` on download and launch to override storage.
The launch profile binds loopback port 5561 and selects physical GPU 1.

## Acceptance evidence

Verified on 2026-09-23 with Python 3.12.13, Torch 2.10.0+cu128,
torchvision 0.25.0+cu128, OpenCV 5.0.0, physical GPU 1 (RTX 5090), and NVIDIA
driver 580.173.02. The downloaded Large checkpoint loaded strictly.

`smoke.py` used the pinned upstream's `assets/examples/demo01.jpg`:

| Check | Result |
|---|---|
| Input / output image dimensions | 1362 × 2048 |
| Model input size | 518 |
| Typed HTTP response | `float32`, finite, range [0, 816.8045] |
| Difference from direct official inference | Maximum absolute difference **0.0** |
| First HTTP handler inference time | 372.6 ms (one sample, excludes model loading) |
| Peak Torch allocated GPU memory | 2105.5 MiB (includes model; excludes non-Torch allocations) |

The root test suite passed 261 tests under Python 3.10; this includes fake
HTTP/MessagePack round trips, RGB/dtype validation, input-size forwarding,
checkpoint download behavior, service selection, and light/runtime layout.
The dedicated real profile also passed `robot-tools doctor`, launched through
`robot-tools launch` on port 5561, and served `examples/depth_anything_v2_image.py`
successfully, saving the original-resolution float32 prediction as `.npy`.
CPU fallback follows the official implementation but has not been benchmarked.
Fake tests verify API behavior, not model quality. Latency and memory depend
on image aspect ratio, input size, and other GPU workloads. The real sample
checks adapter fidelity, not accuracy on edited images.
