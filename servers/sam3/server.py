"""SAM 3 ZMQ server.

Run from this directory:
    uv run python server.py --port 5558                    # real model
    uv run python server.py --port 5558 --mock             # protocol-only

Real mode requires:
    - SAM 3 + torch installed (`robot-tools setup sam3`)
    - HuggingFace auth + access to `facebook/sam3` granted (see README)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.contracts.sam3 import (
    SegmentByPointRequest,
    SegmentByPointResponse,
    SegmentByTextRequest,
    SegmentByTextResponse,
    TextDetection,
)
from robot_tools.server_base import BaseServer, handler

logger = logging.getLogger(__name__)


class SAM3Server(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5558,
        *,
        mock: bool = False,
        device: str = "cuda",
    ):
        super().__init__(host=host, port=port)
        self._mock = mock
        if mock:
            logger.warning("MOCK mode: returning empty results, no model loaded")
            self.pipeline = None
        else:
            from pipeline import SAM3Pipeline
            self.pipeline = SAM3Pipeline(device=device)

    @handler(SegmentByTextRequest)
    def segment_by_text(self, req: SegmentByTextRequest) -> SegmentByTextResponse:
        if self._mock:
            return SegmentByTextResponse(detections=[])

        results = self.pipeline.segment_by_text(req.image, req.text_prompt)
        thr = req.score_threshold
        return SegmentByTextResponse(detections=[
            TextDetection(
                mask=r["mask"], box=r["box"], score=r["score"], label=r["label"],
            )
            for r in results if r["score"] >= thr
        ])

    @handler(SegmentByPointRequest)
    def segment_by_point(self, req: SegmentByPointRequest) -> SegmentByPointResponse:
        if self._mock:
            H, W = req.image.shape[:2]
            return SegmentByPointResponse(
                masks=np.zeros((1, H, W), dtype=bool),
                scores=[0.0],
            )

        masks, scores = self.pipeline.segment_by_point(
            req.image, req.point_xy, multimask_output=req.multimask_output,
        )
        return SegmentByPointResponse(masks=masks, scores=scores)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mock", action="store_true",
                        help="Skip real model loading; return empty results.")
    args = parser.parse_args()

    server = SAM3Server(
        host=args.host, port=args.port, mock=args.mock, device=args.device,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
