"""GraspGen HTTP server.

Run from this directory:
    uv run python server.py --port 5557                          # real model
    uv run python server.py --port 5557 --fake                   # protocol-only

Real mode requires:
    - GraspGen installed in this venv (see INSTALL.md)
    - GRASPGEN_MODELS_DIR pointing at the downloaded checkpoints (see download.sh)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.graspgen.contract import (
    API_VERSION,
    SERVICE_ID,
    GenerateCollisionFreeGraspsRequest,
    GenerateCollisionFreeGraspsResponse,
    GenerateGraspsRequest,
    GenerateGraspsResponse,
)

logger = logging.getLogger(__name__)


class GraspGenServer(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5557,
        *,
        fake: bool = False,
        gripper: str = "franka_panda",
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
            from backend import GraspGenBackend

            logger.info("loading GraspGen backend (gripper=%s)", gripper)
            self.backend = GraspGenBackend(gripper)
        self.mark_ready()

    @handler(GenerateGraspsRequest)
    def generate_grasps(self, req: GenerateGraspsRequest) -> GenerateGraspsResponse:
        if self._fake:
            return _fake_grasps(req.num_grasps, req.grasp_threshold)

        self._apply_overrides(req)
        result = self.backend.generate_grasps(req.point_cloud)
        return GenerateGraspsResponse(
            poses=np.asarray(result["poses"], dtype=np.float32),
            confidences=np.asarray(result["confidences"], dtype=np.float32),
        )

    @handler(GenerateCollisionFreeGraspsRequest)
    def generate_collision_free_grasps(
        self, req: GenerateCollisionFreeGraspsRequest
    ) -> GenerateCollisionFreeGraspsResponse:
        if self._fake:
            base = _fake_grasps(req.num_grasps, req.grasp_threshold)
            return GenerateCollisionFreeGraspsResponse(
                poses=base.poses,
                confidences=base.confidences,
                collision_free_mask=np.ones(len(base.poses), dtype=bool),
            )

        self._apply_overrides(req)
        result = self.backend.generate_collision_free_grasps(
            req.point_cloud,
            req.scene_pc,
            collision_threshold=req.collision_threshold,
            max_scene_points=req.max_scene_points,
        )
        return GenerateCollisionFreeGraspsResponse(
            poses=np.asarray(result["poses"], dtype=np.float32),
            confidences=np.asarray(result["confidences"], dtype=np.float32),
            collision_free_mask=np.asarray(result["collision_free_mask"], dtype=bool),
        )

    def _apply_overrides(self, req) -> None:
        # BaseServer's per-service execution lock serializes this mutation.
        self.backend.grasp_threshold = req.grasp_threshold
        self.backend.num_grasps = req.num_grasps
        self.backend.topk_num_grasps = req.topk


def _fake_grasps(num_grasps: int, threshold: float) -> GenerateGraspsResponse:
    k = min(num_grasps, 16)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    confidences = np.linspace(1.0, threshold, k, dtype=np.float32)
    return GenerateGraspsResponse(poses=poses, confidences=confidences)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 only on a trusted, firewalled LAN)")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument(
        "--gripper",
        default="franka_panda",
        help="Gripper name (franka_panda, robotiq_2f_85, robotiq_2f_140, single_suction_cup_30mm).",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Skip real model loading; return fake grasps. Useful before GraspGen is installed.",
    )
    args = parser.parse_args()

    server = GraspGenServer(
        host=args.host,
        port=args.port,
        fake=args.fake,
        gripper=args.gripper,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
