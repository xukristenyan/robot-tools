from __future__ import annotations

from typing import ClassVar

import numpy as np

from robot_tools.core.client import BaseClient
from robot_tools.services.anyplace.contract import (
    ACTIONS,
    API_VERSION,
    SERVICE_ID,
    MaskedRGBD,
    PredictPlacementsRequest,
    PredictPlacementsResponse,
)


def masked_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    camera_pose: np.ndarray | None = None,
) -> MaskedRGBD:
    """Build an observation; integer masks may use either 0/1 or 0/255.

    Depth units are never guessed or rescaled: convert millimetres to metres
    before calling. RGB dtype must already be uint8.
    """
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.integer) or not (
            np.isin(mask, [0, 1]).all() or np.isin(mask, [0, 255]).all()
        ):
            raise ValueError("mask must be boolean or a binary 0/1 or 0/255 integer image")
        mask = mask != 0
    return MaskedRGBD(
        rgb=np.ascontiguousarray(rgb),
        depth=np.ascontiguousarray(depth),
        mask=np.ascontiguousarray(mask),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        camera_pose=np.eye(4) if camera_pose is None else np.asarray(camera_pose, dtype=np.float64),
    )


class AnyPlaceClient(BaseClient):
    SUPPORTED_SERVICE_APIS: ClassVar[dict[str, frozenset[str]]] = {SERVICE_ID: frozenset({API_VERSION})}
    REQUIRED_ACTIONS = ACTIONS

    def predict_placements(
        self,
        pick: MaskedRGBD,
        place: MaskedRGBD,
        *,
        num_samples: int = 5,
        n_refine_iters: int = 50,
        seed: int = 0,
    ) -> PredictPlacementsResponse:
        """Predict relative pick-object transforms using the supplied place region."""
        request = PredictPlacementsRequest(
            pick=pick,
            place=place,
            num_samples=num_samples,
            n_refine_iters=n_refine_iters,
            seed=seed,
        )
        return self._request(request, PredictPlacementsResponse)
