# Depth Anything V2 Large

Single RGB image → official **raw relative inverse depth**, using the Large
(`vitl`, 335.3M parameters) checkpoint. Larger values mean nearer surfaces.
There are no metric units, and values from separate images need not have the
same scale or offset. This service does not perform calibration, RGBD fusion,
metric inference, or image-edit completion.

## Client contract

```python
from robot_tools.core.endpoints import get_endpoint
from robot_tools.services.depth_anything_v2 import DepthAnythingV2Client

with DepthAnythingV2Client.from_endpoint(
    get_endpoint("depth_anything_v2", "endpoints.yaml")
) as client:
    result = client.estimate_depth(rgb, input_size=518)
    inverse_depth = result.relative_inverse_depth  # (H, W), float32
```

`rgb` must be nonempty `(H, W, 3)` **RGB uint8**. Floats, grayscale, and RGBA
are rejected rather than silently reinterpreted. The output has the original
image dimensions and contains finite, nonnegative values. It is not an 8-bit
visualization, normalized to [0, 1], or directly usable as metric point-cloud Z.

The only action is `estimate_depth`, service API version `1`, default port
`5561`. `input_size=None` uses the startup default, `518`. An override must be
an integer >= 14. The upstream preserves aspect ratio and rounds inference
dimensions to multiples of 14; increasing the size increases memory and
compute. RGB is converted to BGR only at the official `infer_image` boundary.

## Installation and model storage

Read [PREREQUISITES.md](PREREQUISITES.md). From the repository root:

```bash
export DEPTH_ANYTHING_V2_WEIGHTS_DIR=/data/$USER/robot-tools-models/depth_anything_v2
uv run robot-tools install runtimes/depth_anything_v2/services.yaml
uv run robot-tools download runtimes/depth_anything_v2/services.yaml
uv run robot-tools doctor runtimes/depth_anything_v2/services.yaml
uv run robot-tools launch runtimes/depth_anything_v2/services.yaml
```

The dedicated profile selects only this service and uses **physical GPU 1**.
Change its `gpu` field for another host. The default model directory is
`/data/<current-user>/robot-tools-models/depth_anything_v2`, even when the
environment variable is unset. Custom directories must be exported both for
download and launch. A direct `--checkpoint /absolute/path/...pth` overrides
the default file on launch; it must be the relative-depth Large architecture.

Only `depth_anything_v2_vitl.pth` is downloaded (about 1.34 GB). The downloader
uses a fixed official Hugging Face revision, writes to a temporary file,
then renames it after a successful nonempty download. Existing nonempty files
are reused. No checkpoint hash verification is performed. Strict model loading
checks checkpoint architecture; delete an incomplete/corrupt file before retrying.

Installation initializes only this upstream, syncs `real/uv.lock` into this
runtime's `.venv`, probes imports, and records the existing repository's real
installation state. A later `robot-tools sync` restores the light environment;
rerun `install` before real launch.

## Fake mode and examples

Add `extra_args: ["--fake"]` to a profile to exercise transport without Torch,
the upstream, weights, or a GPU. Fake output is a deterministic horizontal
gradient in [1, 2], with no geometric meaning.

Client `endpoints.yaml`:

```yaml
endpoints:
  depth_anything_v2: {host: 127.0.0.1, port: 5561}
```

The examples profile already includes this endpoint. To save a real prediction:

```bash
uv run --with pillow python examples/depth_anything_v2_image.py \
  --image path/to/rgb.png --out relative_inverse_depth.npy
```

Per-call settings live in `examples/configs/depth_anything_v2.yaml`.

## Real acceptance

`doctor` checks actual model imports and dependency/installation state; it
does not allocate Large or prove that a checkpoint or GPU has sufficient memory.
The separate smoke command loads the checkpoint, performs a typed HTTP request,
and compares the response with direct official inference:

```bash
CUDA_VISIBLE_DEVICES=1 runtimes/depth_anything_v2/.venv/bin/python \
  runtimes/depth_anything_v2/smoke.py \
  --image runtimes/depth_anything_v2/upstream/assets/examples/demo01.jpg
```

It reports shape, dtype, range, first-request latency, peak allocated CUDA
memory, and maximum absolute difference from upstream. To check CPU on Linux,
use `CUDA_VISIBLE_DEVICES=""` (slower). The model follows the upstream's automatic
CUDA/MPS/CPU selection, using its default float32 inference without AMP or
additional postprocessing. See prerequisites for recorded acceptance evidence.

The official Large weights use CC-BY-NC-4.0; see the
[upstream license information](https://github.com/DepthAnything/Depth-Anything-V2#license).
