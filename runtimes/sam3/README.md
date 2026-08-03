# SAM3 runtime

Independent uv project that wraps Meta's SAM 3 as an HTTP server.
Image-mode segmentation (single frame). Two prompt types:

- **Text**: `client.segment_by_text(rgb, "the red mug")` → list of detections
  (`mask` + absolute `box_xyxy` + score) sorted by confidence.
- **Point**: `client.segment_by_point(rgb, (x, y))` → up to 3 candidate masks
  at increasing scales, sorted by score.

Video tracking, multiple positive/negative points, and box prompts are outside
the current downstream scope and therefore are not exposed.

Before a real-model installation, read [`PREREQUISITES.md`](PREREQUISITES.md).
It records the requirements intentionally outside `install.sh`, including the
tested host baseline, gated Hugging Face access, persistent cache location,
clean-machine order, and required verification.

## Layout

```
runtimes/sam3/
├── pyproject.toml         locked light/fake project (no upstream or Torch)
├── uv.lock               exact light/fake dependency graph
├── real/
│   ├── pyproject.toml     Linux x86_64 + cu128 real-model project
│   └── uv.lock            exact SAM3, Torch, and transitive dependency graph
├── PREREQUISITES.md       external host, access, storage, and verification contract
├── server.py              BaseServer subclass; @handler-decorated methods
├── backend.py             adapts Sam3Processor + build_sam3_image_model
├── doctor.py              read-only real dependency and SAM3 import probe
├── upstream/              git submodule of facebookresearch/sam3
├── install.sh             exact-syncs real/uv.lock into the shared .venv
├── download.sh            HF auth check + pre-fetch weights
└── (no local weights/)    — SAM 3 weights live in ~/.cache/huggingface/
```

The two projects intentionally share `runtimes/sam3/.venv`, but they never
share a lockfile. `sync` makes that environment exactly light/fake;
`install` makes it exactly real. This keeps an uninitialized `upstream/` out of
fake resolution while retaining reproducible real installations.

The pinned upstream also imports `einops`, `psutil`, and `pycocotools` while
loading `model_builder.py`, despite not declaring them as core dependencies.
They are explicit real-project dependencies; Notebook, training, evaluation,
and delayed video-I/O dependencies remain excluded.

## Fake mode (no model, fast)

```bash
uv run robot-tools sync        # light deps only
uv run robot-tools launch      # services.yaml has extra_args: ["--fake"]
```

## Real mode

The abbreviated commands below assume the host and storage prerequisites are
already satisfied. Follow [`PREREQUISITES.md`](PREREQUISITES.md) for a new
machine.

⚠️ **SAM 3 weights are gated.** You must:

1. **Request access:** visit <https://huggingface.co/facebook/sam3>,
   click "Request access", wait for approval (usually fast).
2. **Authenticate locally** (one-time, system-wide):
   ```bash
   hf auth login                 # paste an HF access token
   ```
3. Then:
   ```bash
   uv run robot-tools sync
   uv run robot-tools install       # selected upstream + locked real env
   uv run robot-tools download      # auth check + cache weights
   # edit services.yaml → drop --fake
   uv run robot-tools launch
   ```

If `download` fails with "gated" or "403", access wasn't granted yet.
If it fails with "not authenticated", run `hf auth login`.

## Status

`server.py` switches between fake and real via `--fake`. `backend.py`
is the real-model wrapper; only image-mode segmentation is implemented.
