# sam3-server

Independent uv project that wraps Meta's SAM 3 as a ZMQ server.
Image-mode segmentation (single frame). Two prompt types:

- **Text**: `client.segment_by_text(rgb, "the red mug")` → list of detections
  (mask + box + score) sorted by confidence.
- **Point**: `client.segment_by_point(rgb, (x, y))` → up to 3 candidate masks
  at increasing scales, sorted by score.

Video / point-tracking is **not** wired up here yet — add a third action
later if needed (SAM 3 supports it via `build_sam3_video_predictor`).

## Layout

```
servers/sam3/
├── pyproject.toml         this server's uv project (own .venv, Python 3.12 managed)
├── server.py              BaseServer subclass; @handler-decorated methods
├── pipeline.py            wraps Sam3Processor + build_sam3_image_model
├── upstream/              git submodule of facebookresearch/sam3
├── setup.sh               installs torch 2.7 (cu126) + sam3 (editable)
├── download.sh            HF auth check + pre-fetch weights
└── (no local weights/)    — SAM 3 weights live in ~/.cache/huggingface/
```

## Mock mode (no model, fast)

```bash
uv run robot-tools sync   services.yaml      # light deps only
uv run robot-tools launch services.yaml      # services.yaml has extra_args: ["--mock"]
```

## Real mode

⚠️ **SAM 3 weights are gated.** You must:

1. **Request access:** visit <https://huggingface.co/facebook/sam3>,
   click "Request access", wait for approval (usually fast).
2. **Authenticate locally** (one-time, system-wide):
   ```bash
   hf auth login                 # paste an HF access token
   ```
3. Then:
   ```bash
   uv run robot-tools sync     services.yaml
   uv run robot-tools setup    sam3            # torch + sam3 editable
   uv run robot-tools download sam3            # auth check + cache weights
   # edit services.yaml → drop --mock
   uv run robot-tools launch   services.yaml
   ```

If `download` fails with "gated" or "403", access wasn't granted yet.
If it fails with "not authenticated", run `hf auth login`.

## Status

`server.py` switches between mock and real via `--mock`. `pipeline.py`
is the real-model wrapper; only image-mode segmentation is implemented.
