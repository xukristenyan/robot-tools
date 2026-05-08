from __future__ import annotations

from typing import Literal

import numpy as np

from robot_tools.contracts.base import BaseRequest, BaseResponse


class ReconstructRequest(BaseRequest):
    action: Literal["reconstruct"] = "reconstruct"
    left: np.ndarray   # (H, W, 3) uint8 or (H, W) gray; client coerces to HxWx3 uint8
    right: np.ndarray  # same shape as left
    # Per-call refinement knobs; None => use server's startup defaults.
    valid_iters: int | None = None
    max_disp: int | None = None


class ReconstructResponse(BaseResponse):
    disparity: np.ndarray  # (H, W) float32, in pixels — depth = K_fx * baseline / disp
