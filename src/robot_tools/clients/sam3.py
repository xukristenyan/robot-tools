from __future__ import annotations

import numpy as np

from robot_tools.clients.base import BaseClient
from robot_tools.contracts.sam3 import (
    SegmentByPointRequest,
    SegmentByPointResponse,
    SegmentByTextRequest,
    SegmentByTextResponse,
)


class SAM3Client(BaseClient):
    def segment_by_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        *,
        score_threshold: float = 0.0,
    ) -> SegmentByTextResponse:
        """Open-vocabulary segmentation: text describes the target.

        Returns one or more `TextDetection`s, sorted by score. Each has a
        boolean mask (H,W), a [x1,y1,x2,y2] box, score, and label (echoes
        the text). Use `score_threshold` to drop weak hits server-side.
        """
        req = SegmentByTextRequest(
            image=_as_hwc3_u8(image),
            text_prompt=text_prompt,
            score_threshold=score_threshold,
        )
        return self._request(req, SegmentByTextResponse)

    def segment_by_point(
        self,
        image: np.ndarray,
        point_xy: tuple[float, float],
        *,
        multimask_output: bool = True,
    ) -> SegmentByPointResponse:
        """Click-prompted segmentation: a single foreground point.

        With `multimask_output=True` (default) returns up to 3 candidate
        masks at increasing scales, sorted by score. Pick the first one
        for the most-likely segment.
        """
        req = SegmentByPointRequest(
            image=_as_hwc3_u8(image),
            point_xy=tuple(float(v) for v in point_xy),
            multimask_output=multimask_output,
        )
        return self._request(req, SegmentByPointResponse)


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
