from __future__ import annotations

import numpy as np

from robot_tools.clients.base import BaseClient
from robot_tools.contracts.fastfs import ReconstructRequest, ReconstructResponse


class FastFSClient(BaseClient):
    def reconstruct(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        valid_iters: int | None = None,
        max_disp: int | None = None,
    ) -> ReconstructResponse:
        """Run FFS on a stereo pair, get back raw disparity.

        left, right: HxW (gray) or HxWx3/HxWx4. Coerced to uint8 HxWx3.
        valid_iters / max_disp: optional per-call overrides; None uses server default.

        For depth, do `disparity_to_depth(resp.disparity, K, baseline)` client-side.
        """
        req = ReconstructRequest(
            left=_as_hwc3_u8(left),
            right=_as_hwc3_u8(right),
            valid_iters=valid_iters,
            max_disp=max_disp,
        )
        return self._request(req, ReconstructResponse)


def disparity_to_depth(
    disparity: np.ndarray,
    K: np.ndarray,
    baseline: float,
) -> np.ndarray:
    """depth (m) = K[0,0] * baseline / disparity, with safe guard against zero."""
    safe = np.where(disparity > 1e-6, disparity, np.inf)
    return (float(K[0, 0]) * float(baseline) / safe).astype(np.float32)


def _as_hwc3_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")
    img = img[..., :3]
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return np.ascontiguousarray(img)
