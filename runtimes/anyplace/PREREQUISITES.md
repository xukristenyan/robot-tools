# AnyPlace prerequisites

| Component | Tested value |
|---|---|
| OS / architecture | Ubuntu 24.04, Linux x86_64 |
| GPU / driver | NVIDIA RTX 5090 / 580.173.02 |
| Python | uv-managed 3.10.20 |
| uv | 0.12.3; real project requires >=0.11.21 |
| Torch | 2.10.0+cu128 |
| NumPy | 1.26.4 in real environment |
| Upstream | ac-rad/anyplace commit 3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e |
| Checkpoint | Official anyplace_multitask/model.pth, 81,354,343 bytes |

Use Git, `patch`, `curl`, a working NVIDIA driver, and uv. No conda, nvcc,
AnyGrasp license, Hugging Face token, VLM service, simulator or GUI is required.
Network access is needed for GitHub, PyPI, PyTorch's CUDA wheel index, and the
public `yuchiallanzhao/anyplace` Hugging Face dataset repository.

Allow roughly 7 GB for the environment plus uv's cache, 355 MiB for the official
checkpoint ZIP, and 78 MiB for the extracted multitask model. Only AnyPlace's
selected upstream is initialized. Unselected model services are not installed.

Default persistent storage:

```bash
export ANYPLACE_WEIGHTS_DIR=/data/xkyan3/robot-tools-models/anyplace
```

The default already uses this path. On another host, set this variable for
**both** downloading and launching. Startup `--checkpoint` overrides it.
The downloader extracts only the multitask model from the release ZIP.

`install.sh` syncs `real/uv.lock`, checks dependencies, exercises actual model
imports and CUDA FPS, then records the repository's standard real-install
state. `doctor.py` is read-only and does not load checkpoints or download files.
Model existence and architecture are validated at real server startup.
Launching uses `uv run --no-sync`; a later light sync requires reinstalling.

Revalidate after changing GPU architecture, driver, Python, upstream revision,
compatibility patch, dependency lock, or checkpoint. See [README](README.md) for
the tested input/output, semantic limitations, and deployment commands.
