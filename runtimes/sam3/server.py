"""SAM 3 HTTP server.

Run from this directory:
    uv run python server.py --port 5558                    # real model
    uv run python server.py --port 5558 --fake             # protocol-only

Real mode requires:
    - SAM 3 + torch installed (`robot-tools install`)
    - HuggingFace auth + access to `facebook/sam3` granted (see README)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.sam3.contract import (
    API_VERSION,
    SERVICE_ID,
    SegmentByPointRequest,
    SegmentByPointResponse,
    SegmentByTextRequest,
    SegmentByTextResponse,
    TextDetection,
)

logger = logging.getLogger(__name__)


class SAM3Server(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5558,
        *,
        fake: bool = False,
        device: str = "cuda",
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
            from backend import SAM3Backend

            self.backend = SAM3Backend(device=device)
        self.mark_ready()

    @handler(SegmentByTextRequest)
    def segment_by_text(self, req: SegmentByTextRequest) -> SegmentByTextResponse:
        if self._fake:
            return _fake_text_response(req)

        results = self.backend.segment_by_text(req.image, req.text_prompt)
        thr = req.score_threshold
        return SegmentByTextResponse(
            detections=[
                TextDetection(
                    mask=r["mask"],
                    box_xyxy=r["box_xyxy"],
                    score=r["score"],
                    label=r["label"],
                )
                for r in results
                if r["score"] >= thr
            ]
        )

    @handler(SegmentByPointRequest)
    def segment_by_point(self, req: SegmentByPointRequest) -> SegmentByPointResponse:
        if self._fake:
            return _fake_point_response(req)

        masks, scores = self.backend.segment_by_point(
            req.image,
            req.point_xy,
            multimask_output=req.multimask_output,
        )
        return SegmentByPointResponse(masks=masks, scores=scores)


def _fake_text_response(req: SegmentByTextRequest) -> SegmentByTextResponse:
    """Return one deterministic instance so downstream pipelines stay testable."""
    if req.score_threshold > 1.0:
        return SegmentByTextResponse(detections=[])

    height, width = req.image.shape[:2]
    x1, x2 = width // 4, max(width // 4 + 1, 3 * width // 4)
    y1, y2 = height // 4, max(height // 4 + 1, 3 * height // 4)
    x2, y2 = min(x2, width), min(y2, height)
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return SegmentByTextResponse(
        detections=[
            TextDetection(
                mask=mask,
                box_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                score=1.0,
                label=req.text_prompt,
            )
        ]
    )


def _fake_point_response(req: SegmentByPointRequest) -> SegmentByPointResponse:
    """Return ranked circular masks centered on the requested pixel."""
    height, width = req.image.shape[:2]
    x = float(np.clip(req.point_xy[0], 0, max(0, width - 1)))
    y = float(np.clip(req.point_xy[1], 0, max(0, height - 1)))
    base_radius = max(1, min(height, width) // 16)
    radii = (base_radius, base_radius * 2, base_radius * 3) if req.multimask_output else (base_radius,)
    yy, xx = np.ogrid[:height, :width]
    masks = np.stack([((xx - x) ** 2 + (yy - y) ** 2) <= radius**2 for radius in radii])
    scores = [0.95, 0.8, 0.65][: len(radii)]
    return SegmentByPointResponse(masks=masks.astype(bool), scores=scores)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 only on a trusted, firewalled LAN)")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fake", action="store_true", help="Skip real model loading; return deterministic test data.")
    args = parser.parse_args()

    server = SAM3Server(
        host=args.host,
        port=args.port,
        fake=args.fake,
        device=args.device,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
