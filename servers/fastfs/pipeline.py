"""Headless wrapper around Fast-FoundationStereo for ZMQ-server use.

Resolution order for weights:
  1. $FFS_WEIGHTS_DIR
  2. <this server dir>/weights  (default, populated by download.sh)

Adapted from user's old `my_server_client/my_server.py` `_load_model` / `_infer_numpy`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "upstream"
_DEFAULT_WEIGHTS_DIR = _HERE / "weights"

# Make the FFS upstream repo importable.
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from Utils import AMP_DTYPE, set_logging_format, set_seed  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


def _weights_dir() -> Path:
    raw = os.getenv("FFS_WEIGHTS_DIR")
    candidate = Path(raw).expanduser().resolve() if raw else _DEFAULT_WEIGHTS_DIR
    if not candidate.exists():
        raise RuntimeError(
            f"FFS weights not found at {candidate}. "
            f"Run `robot-tools download fastfs` (or `bash download.sh`)."
        )
    return candidate


def _resolve_model(model_id: str) -> Path:
    """Accept a checkpoint ID (e.g. '23-36-37') or absolute path to a .pth."""
    p = Path(model_id)
    if p.is_file():
        return p
    base = _weights_dir() / model_id
    preferred = base / "model_best_bp2_serialize.pth"
    if preferred.is_file():
        return preferred
    pths = sorted(base.glob("*.pth")) if base.is_dir() else []
    if pths:
        return pths[0]
    raise FileNotFoundError(
        f"FFS checkpoint {model_id!r} not found. "
        f"Tried {preferred} and any *.pth under {base}."
    )


class FastFSPipeline:
    """Loads the FFS model once, runs inference on numpy stereo pairs."""

    def __init__(
        self,
        model_id: str = "23-36-37",
        valid_iters: int = 8,
        max_disp: int = 192,
        scale: float = 1.0,
    ):
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)

        path = _resolve_model(model_id)
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.args.valid_iters = int(valid_iters)
        model.args.max_disp = int(max_disp)
        model.cuda().eval()

        self.model = model
        self.model_path = str(path)
        self.default_valid_iters = int(valid_iters)
        self.default_max_disp = int(max_disp)
        self.scale = float(scale)

    def warmup(self, h: int = 480, w: int = 640) -> None:
        """One-shot forward to amortize the first-call compile."""
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self.infer(dummy, dummy)

    def infer(
        self,
        left_u8: np.ndarray,
        right_u8: np.ndarray,
        valid_iters: int | None = None,
        max_disp: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run one forward pass. Returns (disparity (H,W) float32, timings_ms dict)."""
        t_start = time.time()
        if left_u8.shape != right_u8.shape:
            raise ValueError(
                f"left/right shape mismatch: {left_u8.shape} vs {right_u8.shape}"
            )
        if left_u8.dtype != np.uint8 or right_u8.dtype != np.uint8:
            raise ValueError("left/right must be uint8")
        if left_u8.ndim == 2:
            left_u8 = np.stack([left_u8] * 3, axis=-1)
            right_u8 = np.stack([right_u8] * 3, axis=-1)
        left_u8 = np.ascontiguousarray(left_u8[..., :3])
        right_u8 = np.ascontiguousarray(right_u8[..., :3])

        if self.scale != 1.0:
            left_u8 = cv2.resize(left_u8, None, fx=self.scale, fy=self.scale)
            right_u8 = cv2.resize(right_u8, (left_u8.shape[1], left_u8.shape[0]))
        H, W = left_u8.shape[:2]
        t_pre = time.time()

        t0 = torch.as_tensor(left_u8).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(right_u8).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        vi = int(valid_iters) if valid_iters is not None else self.default_valid_iters
        md = int(max_disp) if max_disp is not None else self.default_max_disp
        self.model.args.valid_iters = vi
        self.model.args.max_disp = md

        t_inf0 = time.time()
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            disp = self.model.forward(
                t0, t1, iters=vi, test_mode=True, optimize_build_volume="pytorch1"
            )
        disp = padder.unpad(disp.float())
        torch.cuda.synchronize()
        t_inf1 = time.time()
        disp_np = disp.data.cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)

        t_end = time.time()
        return disp_np, {
            "preprocess": round((t_pre - t_start) * 1000, 3),
            "infer": round((t_inf1 - t_inf0) * 1000, 3),
            "postprocess": round((t_end - t_inf1) * 1000, 3),
            "total": round((t_end - t_start) * 1000, 3),
        }
