from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from robot_tools.contracts.base import BaseRequest, BaseResponse


# ---------- segment_by_text: open-vocabulary text → masks ----------

class SegmentByTextRequest(BaseRequest):
    action: Literal["segment_by_text"] = "segment_by_text"
    image: np.ndarray         # (H, W, 3) uint8 RGB
    text_prompt: str           # e.g. "the red mug"
    score_threshold: float = 0.0    # filter low-confidence detections client-side


class TextDetection(BaseModel):
    """One detected instance from a text-prompt query."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mask: np.ndarray   # (H, W) bool
    box: list[float]   # [x1, y1, x2, y2] in pixels
    score: float
    label: str         # echoes the text_prompt


class SegmentByTextResponse(BaseResponse):
    detections: list[TextDetection]   # sorted by score, descending


# ---------- segment_by_point: pixel-click → masks ----------

class SegmentByPointRequest(BaseRequest):
    action: Literal["segment_by_point"] = "segment_by_point"
    image: np.ndarray            # (H, W, 3) uint8 RGB
    point_xy: tuple[float, float]  # foreground point in pixel coords
    multimask_output: bool = True   # return up to 3 masks (small / med / large) sorted by score


class SegmentByPointResponse(BaseResponse):
    masks: np.ndarray   # (N, H, W) bool, sorted by score descending
    scores: list[float]
