"""Fast-FoundationStereo HTTP server.

Run from this directory:
    uv run python server.py --port 5556                                   # real model
    uv run python server.py --port 5556 --fake                            # protocol-only

Real mode requires:
    - FFS deps installed (`robot-tools install`)
    - FFS_WEIGHTS_DIR or default ./weights/ populated (`robot-tools download`)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.fastfs.contract import (
    API_VERSION,
    SERVICE_ID,
    ReconstructRequest,
    ReconstructResponse,
)

logger = logging.getLogger(__name__)


class FastFSServer(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        *,
        fake: bool = False,
        model_id: str = "23-36-37",
        valid_iters: int = 8,
        max_disp: int = 192,
    ):
        super().__init__(
            service_id=SERVICE_ID,
            api_version=API_VERSION,
            host=host,
            port=port,
        )
        self._fake = fake
        if fake:
            logger.warning("FAKE mode: returning deterministic test data; no model loaded")
            self.backend = None
        else:
            from backend import FastFSBackend

            logger.info("loading FFS backend (model=%s)", model_id)
            self.backend = FastFSBackend(
                model_id=model_id,
                valid_iters=valid_iters,
                max_disp=max_disp,
            )
            logger.info("warming up...")
            self.backend.warmup()
        self.mark_ready()

    @handler(ReconstructRequest)
    def reconstruct(self, req: ReconstructRequest) -> ReconstructResponse:
        if self._fake:
            H, W = req.left.shape[:2]
            return ReconstructResponse(disparity=np.zeros((H, W), dtype=np.float32))

        disp, _timings = self.backend.infer(
            req.left,
            req.right,
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
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 only on a trusted, firewalled LAN)")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--model", default="23-36-37", help="Checkpoint id under weights/ or absolute .pth path")
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Skip real model loading; return zero disparity. Useful before FFS is installed.",
    )
    args = parser.parse_args()

    server = FastFSServer(
        host=args.host,
        port=args.port,
        fake=args.fake,
        model_id=args.model,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
