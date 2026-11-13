"""Single RGB image to raw relative inverse depth, using the Large model."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.depth_anything_v2.contract import (
    API_VERSION,
    SERVICE_ID,
    EstimateDepthRequest,
    EstimateDepthResponse,
)

logger = logging.getLogger(__name__)


class DepthAnythingV2Server(BaseServer):
    def __init__(self, host="127.0.0.1", port=5561, *, fake=False, checkpoint=None, input_size=518):
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size < 14:
            raise ValueError("input_size must be an integer >= 14")
        super().__init__(service_id=SERVICE_ID, api_version=API_VERSION, host=host, port=port)
        self._fake = fake
        self._input_size = input_size
        self.backend = None
        if not fake:
            from backend import DepthAnythingV2Backend

            self.backend = DepthAnythingV2Backend(checkpoint=checkpoint)
            logger.info("Loaded Depth Anything V2 Large on %s", self.backend.device)
        else:
            logger.warning("FAKE mode: deterministic synthetic inverse depth; no model loaded")
        self.mark_ready()

    @handler(EstimateDepthRequest)
    def estimate_depth(self, request: EstimateDepthRequest) -> EstimateDepthResponse:
        start = time.perf_counter()
        if self._fake:
            h, w = request.image.shape[:2]
            depth = np.broadcast_to(np.linspace(1, 2, w, dtype=np.float32), (h, w)).copy()
        else:
            depth = self.backend.infer(request.image, input_size=request.input_size or self._input_size)
        return EstimateDepthResponse(relative_inverse_depth=depth, timing_ms=(time.perf_counter() - start) * 1000)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5561)
    parser.add_argument("--checkpoint", help="Path to an official relative-depth Large (vitl) checkpoint")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    if args.input_size < 14:
        parser.error("--input-size must be >= 14")
    DepthAnythingV2Server(**vars(args)).serve_forever()


if __name__ == "__main__":
    main()
