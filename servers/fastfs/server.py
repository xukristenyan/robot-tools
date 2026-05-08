"""Fast-FoundationStereo ZMQ server.

Run from this directory:
    uv run python server.py --port 5556                                   # real model
    uv run python server.py --port 5556 --mock                            # protocol-only

Real mode requires:
    - FFS deps installed (see README or `robot-tools setup fastfs`)
    - FFS_WEIGHTS_DIR or default ./weights/ populated (`robot-tools download fastfs`)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.contracts.fastfs import ReconstructRequest, ReconstructResponse
from robot_tools.server_base import BaseServer, handler

logger = logging.getLogger(__name__)


class FastFSServer(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        *,
        mock: bool = False,
        model_id: str = "23-36-37",
        valid_iters: int = 8,
        max_disp: int = 192,
    ):
        super().__init__(host=host, port=port)
        self._mock = mock
        if mock:
            logger.warning("MOCK mode: returning zero disparity, no model loaded")
            self.pipeline = None
        else:
            from pipeline import FastFSPipeline
            logger.info("loading FFS pipeline (model=%s)", model_id)
            self.pipeline = FastFSPipeline(
                model_id=model_id, valid_iters=valid_iters, max_disp=max_disp,
            )
            logger.info("warming up...")
            self.pipeline.warmup()

    @handler(ReconstructRequest)
    def reconstruct(self, req: ReconstructRequest) -> ReconstructResponse:
        if self._mock:
            H, W = req.left.shape[:2]
            return ReconstructResponse(disparity=np.zeros((H, W), dtype=np.float32))

        disp, _timings = self.pipeline.infer(
            req.left, req.right,
            valid_iters=req.valid_iters,
            max_disp=req.max_disp,
        )
        return ReconstructResponse(disparity=disp)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--model", default="23-36-37",
                        help="Checkpoint id under weights/ or absolute .pth path")
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--mock", action="store_true",
                        help="Skip real model loading; return zero disparity. "
                             "Useful before FFS is installed.")
    args = parser.parse_args()

    server = FastFSServer(
        host=args.host, port=args.port, mock=args.mock,
        model_id=args.model, valid_iters=args.valid_iters, max_disp=args.max_disp,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
