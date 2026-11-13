# Point-SAM prerequisites and verification

## Pinned artifacts

- Source: `zyc00/Point-SAM`, Git revision `25f4fd9675479f3c951e830771595855ff7ed466`.
- CUDA point operations: `Jiayuan-Gu/torkit3d`, revision `235ecf60497271136f5552cb45bb7cf75ab1cb09`, matching the upstream submodule.
- Weights: `yuchen0187/Point-SAM`, revision `fb2cd5cd8047e3681491d82fc41b880a9a914fc8`, `model.safetensors` (1,244,245,632 bytes).
- Python 3.12, Torch 2.10.0 cu128, TorchVision 0.25.0, timm 0.9.16; all resolved packages in `real/uv.lock`.

Models and source are public; no gated-model approval is needed. Source code and
model weights have their own upstream license files/cards.

## Host and storage

Requires Linux x86_64, uv >= 0.11.21, Git, curl, a CUDA-capable NVIDIA GPU/driver,
nvcc, patch, and a compatible C++ compiler. Installation follows the existing GraspGen
CUDA 12.0 / gcc-12 / g++-12 build convention, emitting `8.9+PTX` for the Blackwell
driver to JIT. Other hardware/toolchains require revalidation; override CC, CXX,
CUDAHOSTCXX, CUDA_HOME and TORCH_CUDA_ARCH_LIST explicitly when appropriate.

Use `/data/xkyan3/robot-tools-models/pointsam` on jarjar. `POINTSAM_DATA_DIR`
controls weights, downloaded source, runtime venv, and default caches. Export
UV_CACHE_DIR and UV_PYTHON_INSTALL_DIR before invoking the root launcher so its
initial light sync also downloads to the data filesystem. Downloads use atomic
replacement after size verification; model loading validates tensor names/shapes.

The real install records the existing shared robot-tools runtime state only
after dependency checks and `doctor.py` succeed. Doctor checks imports and a
small CUDA FPS call, without downloading or loading model weights.

## Acceptance record

Verified on jarjar, 2026-10-17 UTC:

- NVIDIA RTX 5090 (GPU 1), driver 580.178.04; uv-managed Python 3.12.13.
- Torch 2.10.0+cu128, TorchVision 0.25.0+cu128, timm 0.9.16, torkit3d 0.1.2.
- `uv pip check` and actual CUDA farthest-point sampling passed.
- Root SDK test suite: **309 passed**, one existing FastAPI/Starlette deprecation warning.
- Real HTTP encode/segment/refine/release passed with 2,048 synthetic colored points.
  Initial output `(3, 2048)` bool; refined output `(1, 2048)` bool; logits float32.
- Cached decoding and the upstream public `predict_masks` produced identical logits
  on that input (maximum absolute difference 0); masks also matched exactly.
- One cold acceptance run: model load 2.13 s, encode 163.9 ms, first decode 30.8 ms,
  refinement 1.9 ms. These are observed single-run timings, not a benchmark.
  Peak Torch-allocated memory during the test was 1,434,661,376 bytes (about 1.34 GiB),
  excluding driver/context and other processes.

The test starts and stops its own HTTP server. Results validate installation,
transport, cache reuse and adapter equivalence. Robot-scene segmentation quality,
large-cloud VRAM/latency and downstream task success remain untested.

The local acceptance record is `outputs/pointsam-integration/real-smoke.json`.
