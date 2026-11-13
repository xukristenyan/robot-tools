from __future__ import annotations

from typing import ClassVar

import numpy as np

from robot_tools.core.client import BaseClient
from robot_tools.core.exceptions import ProtocolMismatchError
from robot_tools.services.pointsam.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    EncodeCloudRequest,
    EncodeCloudResponse,
    ReleaseCloudRequest,
    ReleaseCloudResponse,
    SegmentRequest,
    SegmentResponse,
)


class PointSAMClient(BaseClient):
    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS

    def encode_cloud(self, xyz: np.ndarray, rgb: np.ndarray, *, num_groups=512, group_size=64) -> EncodeCloudResponse:
        """Encode a registered RGB point cloud once. No resampling or alignment is performed.

        xyz is (N, 3) floating point, in metres; rgb is (N, 3) uint8 RGB.
        Release the returned cloud_id after use. IDs expire on server restart.
        """
        request = EncodeCloudRequest(
            xyz=np.ascontiguousarray(xyz),
            rgb=np.ascontiguousarray(rgb),
            num_groups=num_groups,
            group_size=group_size,
        )
        response = self._request(request, EncodeCloudResponse)
        if response.num_points != len(request.xyz):
            raise ProtocolMismatchError("Point-SAM changed the input point count")
        return response

    def segment(
        self,
        cloud_id: str,
        prompt_points: np.ndarray,
        prompt_labels: np.ndarray,
        *,
        prompt_mask: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> SegmentResponse:
        """Supply ALL current positive/negative prompts in the original cloud frame.

        Optional prompt_mask is one previous logits row in original point order.
        Three masks are returned only for one point without a mask prompt;
        multiple points or a mask prompt yield one refined mask.
        No previous prompt state is implicitly shared between callers.
        """
        request = SegmentRequest(
            cloud_id=cloud_id,
            prompt_points=np.ascontiguousarray(prompt_points),
            prompt_labels=np.ascontiguousarray(prompt_labels),
            prompt_mask=None if prompt_mask is None else np.ascontiguousarray(prompt_mask),
            multimask_output=multimask_output,
        )
        response = self._request(request, SegmentResponse)
        expected_masks = 3 if multimask_output and len(request.prompt_points) == 1 and prompt_mask is None else 1
        if response.cloud_id != cloud_id or len(response.masks) != expected_masks:
            raise ProtocolMismatchError("Point-SAM returned a different cloud or mask count")
        if prompt_mask is not None and response.masks.shape[1] != len(prompt_mask):
            raise ProtocolMismatchError("Point-SAM returned a different point count")
        return response

    def release_cloud(self, cloud_id: str) -> ReleaseCloudResponse:
        """Free cached tensors; repeated release is harmless and returns released=False."""
        return self._request(ReleaseCloudRequest(cloud_id=cloud_id), ReleaseCloudResponse)
