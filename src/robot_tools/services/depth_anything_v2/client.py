from __future__ import annotations

from typing import ClassVar

import numpy as np

from robot_tools.core.client import BaseClient
from robot_tools.core.exceptions import ProtocolMismatchError
from robot_tools.services.depth_anything_v2.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    EstimateDepthRequest,
    EstimateDepthResponse,
)


class DepthAnythingV2Client(BaseClient):
    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS

    def estimate_depth(self, image: np.ndarray, *, input_size: int | None = None) -> EstimateDepthResponse:
        """Infer raw relative inverse depth from one uint8 RGB image.

        Returns float32 (H, W), larger means nearer, with no metric units.
        input_size overrides the server default (518); upstream preserves
        aspect ratio and rounds inference dimensions to multiples of 14.
        """
        request = EstimateDepthRequest(image=np.asarray(image), input_size=input_size)
        response = self._request(request, EstimateDepthResponse)
        if response.relative_inverse_depth.shape != request.image.shape[:2]:
            raise ProtocolMismatchError("Depth Anything V2 returned a different image resolution")
        return response
