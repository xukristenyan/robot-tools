# Point-SAM service

Promptable 3D segmentation using the official pretrained ViT-L model from
[zyc00/Point-SAM](https://github.com/zyc00/Point-SAM).

## Public boundary

| Setting | Value |
|---|---|
| Service / API / default port | `pointsam` / `1` / `5563` |
| Actions | `encode_cloud`, `segment`, `release_cloud` |
| Input | Registered XYZ `(N,3)` float32/64 in metres; RGB `(N,3)` uint8 |
| Prompts | XYZ `(Q,3)` in the original cloud frame; labels `(Q,)`, 1 positive / 0 negative |
| Output | Masks `(M,N)` bool, logits `(M,N)` float32, scores `(M,)` float32 descending |
| Refinement | Supply all current prompts and optionally one previous logits row `(N,)` |
| Fake mode | Deterministic distance-based masks; no weights, upstream, Torch or GPU |

The service preserves input point order. It does not calibrate, merge, resample,
complete geometry or choose a target from language. Multi-mask prediction returns
three masks only for a single point without a mask prompt, otherwise one mask.
Predicted IoU scores are raw model estimates, not calibrated probabilities.

`encode_cloud` normalizes coordinates internally and caches encoder output.
Prompts use that same transform. The default `num_groups=512`, `group_size=64`
follows the upstream README; both can be selected on each encode. N must be at
least both values. No coordinate normalization is required in the caller.

Cached cloud IDs are independent, process-local and expire on release or restart.
Prompts are never implicitly accumulated. Default limits are two cached clouds
and 65,536 input points per cloud. The cache refuses additional clouds rather
than evicting another caller's cloud. Always release in `finally`. The limits
are resource bounds, not guarantees that every point/group combination fits VRAM.
Unknown IDs, full cache and cloud-dependent validation failures use the shared
runtime's sanitized `500 inference_failed` envelope; schema errors use 422.

## Deployment

Read [PREREQUISITES.md](PREREQUISITES.md) first. Light and real environments use
the same `.venv`, matching the other runtimes. Installation places that environment
on the data disk through a `.venv` symlink. Downloads, source and caches stay there.

```bash
export POINTSAM_DATA_DIR=/data/$USER/robot-tools-models/pointsam
export UV_CACHE_DIR="$POINTSAM_DATA_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$POINTSAM_DATA_DIR/python"
CUDA_VISIBLE_DEVICES=1 uv run --no-sync robot-tools install runtimes/pointsam/services.yaml
uv run --no-sync robot-tools download runtimes/pointsam/services.yaml
CUDA_VISIBLE_DEVICES=1 uv run --no-sync robot-tools doctor runtimes/pointsam/services.yaml
uv run --no-sync robot-tools launch runtimes/pointsam/services.yaml
```

The included profile selects GPU 1; change it for another host. The existing
launcher does not propagate the profile GPU to install/doctor, so set
`CUDA_VISIBLE_DEVICES` explicitly for their GPU probes when needed.

The source is stored at `$POINTSAM_DATA_DIR/source/Point-SAM`, pinned by the
installer. Unlike services with checkout-local submodules, this runtime has
`upstream_submodule=False` in the registry. Only its install script fetches its
source; no other service is initialized. The installer initializes only the pinned torkit3d submodule inside that source.
An ignored `upstream` symlink exposes it to the real uv project; the source and
build outputs remain on the data filesystem.

Default checkpoint: `$POINTSAM_DATA_DIR/weights/model.safetensors`.
`POINTSAM_DATA_DIR` defaults to `$ROBOT_TOOLS_MODELS_ROOT/pointsam`, or
`/data/$USER/robot-tools-models/pointsam` when neither is set. Export the same
storage setting for install, download, doctor and launch.

Startup arguments: `--host`, `--port`, `--device`, `--checkpoint`, `--max-clouds`,
`--max-points`, `--fake`. Per-cloud grouping and per-call prompts belong in the API.

For fake mode, use `extra_args: ["--fake"]` in a pointsam-only profile and run
`robot-tools sync <profile>` followed by `robot-tools launch <profile>`.
A light sync removes real dependencies and invalidates the real-install marker;
run install again before real launch.

## Client

```python
import numpy as np
from robot_tools.services.pointsam import PointSAMClient

# xyz: registered (N,3) float metres; rgb: (N,3) uint8 RGB
with PointSAMClient(host="127.0.0.1", port=5563) as client:
    cloud = client.encode_cloud(xyz, rgb)
    try:
        result = client.segment(cloud.cloud_id, xyz[[positive_index]], np.array([1]))
        refined = client.segment(
            cloud.cloud_id,
            xyz[[positive_index, negative_index]],
            np.array([1, 0]),
            prompt_mask=result.logits[0],
        )
        selected_xyz = xyz[refined.masks[0]]
    finally:
        client.release_cloud(cloud.cloud_id)
```

For saved NPZ input, see `examples/pointsam_segment.py` and its tuning config.

## Upstream adaptation and acceptance

The released demo and `evaluation/inference.py` do not match the current model
method signatures. This adapter uses `configs/model/default.yaml`, strict
safetensors loading, and the current `PointCloudSAM.predict_masks` computation.
It separates the encoder from prompt decoding to reuse its output and cached
interpolation weights. Ordinary PyTorch LayerNorm replaces the optional Apex
optimization; normalization parameters and epsilon are unchanged. No Apex,
Flask, training dataset, or pretrained Uni3D download is needed for inference.
`compat/torch-build.patch` removes torkit3d's incompatible `no_python_abi_suffix`
constructor option. The extension uses the normal Python ABI filename; its CUDA
operations are unchanged.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync --project runtimes/pointsam python runtimes/pointsam/smoke.py
```

`smoke.py` runs real HTTP/MessagePack calls, iterative refinement, release, and
numerical comparison against the public upstream `predict_masks` method. It
uses a synthetic 2,048-point cloud; passing validates execution and adapter
equivalence, not segmentation quality on robot observations.

Validation results are recorded in [PREREQUISITES.md](PREREQUISITES.md).
