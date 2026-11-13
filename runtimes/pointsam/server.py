"""Point-SAM HTTP runtime. Fake mode imports no model libraries."""

from __future__ import annotations

import argparse
import logging
import uuid

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.pointsam.contract import (
    API_VERSION,
    SERVICE_ID,
    EncodeCloudRequest,
    EncodeCloudResponse,
    ReleaseCloudRequest,
    ReleaseCloudResponse,
    SegmentRequest,
    SegmentResponse,
)


def normalize_cloud(xyz):
    # Float64 subtraction preserves small local structure in translated frames.
    center = xyz.astype(np.float64).mean(axis=0)
    centered = xyz.astype(np.float64) - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cloud must have a finite, nonzero normalization radius")
    return (centered / scale).astype(np.float32), center, scale


class PointSAMServer(BaseServer):
    def __init__(
        self, host="127.0.0.1", port=5563, *, fake=False, device="cuda", checkpoint=None, max_clouds=2, max_points=65536
    ):
        super().__init__(service_id=SERVICE_ID, api_version=API_VERSION, host=host, port=port)
        if max_clouds < 1 or max_points < 3:
            raise ValueError("max_clouds >= 1 and max_points >= 3 are required")
        self._fake = fake
        self.max_clouds, self.max_points = max_clouds, max_points
        # BaseServer serializes all handlers, including cache changes.
        self.clouds = {}
        self.backend = None
        if not fake:
            from backend import PointSAMBackend

            self.backend = PointSAMBackend(checkpoint=checkpoint, device=device)
        self.mark_ready()

    @handler(EncodeCloudRequest)
    def encode_cloud(self, req: EncodeCloudRequest) -> EncodeCloudResponse:
        if len(self.clouds) >= self.max_clouds:
            raise ValueError("cloud cache is full; release_cloud before encoding another cloud")
        if len(req.xyz) > self.max_points:
            raise ValueError("point count exceeds the configured max_points")
        xyz, center, scale = normalize_cloud(req.xyz)
        encoded = xyz.copy() if self._fake else self.backend.encode(xyz, req.rgb, req.num_groups, req.group_size)
        cloud_id = uuid.uuid4().hex
        self.clouds[cloud_id] = {"encoded": encoded, "center": center, "scale": scale, "count": len(xyz)}
        return EncodeCloudResponse(cloud_id=cloud_id, num_points=len(xyz))

    @handler(SegmentRequest)
    def segment(self, req: SegmentRequest) -> SegmentResponse:
        cloud = self.clouds.get(req.cloud_id)
        if cloud is None:
            raise ValueError("unknown cloud_id; encode the cloud again after release or server restart")
        if req.prompt_mask is not None and len(req.prompt_mask) != cloud["count"]:
            raise ValueError("prompt_mask length must match the cached cloud")
        prompts = ((req.prompt_points.astype(np.float64) - cloud["center"]) / cloud["scale"]).astype(np.float32)
        if not np.isfinite(prompts).all() or np.any(np.abs(prompts) > 1 + 1e-6):
            raise ValueError("prompt_points must lie within the cloud's normalized coordinate bounds")
        multiple = req.multimask_output and len(prompts) == 1 and req.prompt_mask is None
        if self._fake:
            logits, scores = fake_segment(cloud["encoded"], prompts, req.prompt_labels, req.prompt_mask, multiple)
        else:
            logits, scores = self.backend.segment(
                cloud["encoded"], prompts, req.prompt_labels, req.prompt_mask, multiple
            )
        order = np.argsort(-scores, kind="stable")
        logits = np.ascontiguousarray(logits[order], dtype=np.float32)
        return SegmentResponse(
            cloud_id=req.cloud_id, masks=logits > 0, logits=logits, scores=np.asarray(scores[order], dtype=np.float32)
        )

    @handler(ReleaseCloudRequest)
    def release_cloud(self, req: ReleaseCloudRequest) -> ReleaseCloudResponse:
        return ReleaseCloudResponse(released=self.clouds.pop(req.cloud_id, None) is not None)


def fake_segment(xyz, prompts, labels, mask, multiple):
    distances = np.linalg.norm(xyz[:, None, :] - prompts[None, :, :], axis=-1)
    pos = distances[:, labels == 1].min(axis=1) if np.any(labels == 1) else np.ones(len(xyz))
    neg = distances[:, labels == 0].min(axis=1) if np.any(labels == 0) else np.ones(len(xyz))
    radii = (0.2, 0.4, 0.6) if multiple else (0.3,)
    logits = np.stack([np.minimum(radius - pos, neg - pos) for radius in radii]).astype(np.float32)
    if mask is not None:
        logits += 0.1 * mask[None, :]
    return logits, np.asarray([0.95, 0.8, 0.65][: len(radii)], dtype=np.float32)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-clouds", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=65536)
    parser.add_argument("--fake", action="store_true")
    PointSAMServer(**vars(parser.parse_args())).serve_forever()


if __name__ == "__main__":
    main()
