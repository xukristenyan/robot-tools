"""AnyPlace placement-only HTTP runtime; fake mode imports no model code."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.anyplace.contract import (
    API_VERSION,
    SERVICE_ID,
    PredictPlacementsRequest,
    PredictPlacementsResponse,
)


class AnyPlaceServer(BaseServer):
    def __init__(self, host="127.0.0.1", port=5560, *, fake=False, checkpoint=None):
        super().__init__(service_id=SERVICE_ID, api_version=API_VERSION, host=host, port=port)
        self._fake = fake
        self.backend = None
        if not fake:
            from backend import AnyPlaceBackend

            self.backend = AnyPlaceBackend(checkpoint=checkpoint)
        self.mark_ready()

    @handler(PredictPlacementsRequest)
    def predict_placements(self, request: PredictPlacementsRequest) -> PredictPlacementsResponse:
        if self._fake:
            # Protocol fixture only: identity poses are not model predictions.
            return PredictPlacementsResponse(relative=np.tile(np.eye(4), (request.num_samples, 1, 1)))
        return self.backend.predict_placements(request)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to multitask in ANYPLACE_WEIGHTS_DIR")
    args = parser.parse_args()
    AnyPlaceServer(**vars(args)).serve_forever()


if __name__ == "__main__":
    main()
