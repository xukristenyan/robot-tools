"""GraspGen ZMQ server.

Run from this directory:
    uv run python server.py --port 5557                          # real model
    uv run python server.py --port 5557 --mock                   # protocol-only

Real mode requires:
    - GraspGen installed in this venv (see INSTALL.md)
    - GRASPGEN_MODELS_DIR pointing at the downloaded checkpoints (see download_models.sh)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from robot_tools.contracts.graspgen import (
    GenerateCollisionFreeGraspsRequest,
    GenerateCollisionFreeGraspsResponse,
    GenerateGraspsRequest,
    GenerateGraspsResponse,
)
from robot_tools.server_base import BaseServer, handler

logger = logging.getLogger(__name__)


class GraspGenServer(BaseServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5557,
        *,
        mock: bool = False,
        gripper: str = "franka_panda",
    ):
        super().__init__(host=host, port=port)
        self._mock = mock
        if mock:
            logger.warning("MOCK mode: returning fake grasps, no model loaded")
            self.pipeline = None
        else:
            from pipeline import GraspPipeline
            logger.info("loading GraspGen pipeline (gripper=%s)", gripper)
            self.pipeline = GraspPipeline(gripper)

    @handler(GenerateGraspsRequest)
    def generate_grasps(self, req: GenerateGraspsRequest) -> GenerateGraspsResponse:
        if self._mock:
            return _mock_grasps(req.num_grasps, req.grasp_threshold)

        self._apply_overrides(req)
        result = self.pipeline.generate_grasps(req.point_cloud)
        return GenerateGraspsResponse(
            poses=np.asarray(result["poses"], dtype=np.float32),
            confidences=np.asarray(result["confidences"], dtype=np.float32),
        )

    @handler(GenerateCollisionFreeGraspsRequest)
    def generate_collision_free_grasps(
        self, req: GenerateCollisionFreeGraspsRequest
    ) -> GenerateCollisionFreeGraspsResponse:
        if self._mock:
            base = _mock_grasps(req.num_grasps, req.grasp_threshold)
            return GenerateCollisionFreeGraspsResponse(
                poses=base.poses,
                confidences=base.confidences,
                collision_free_mask=np.ones(len(base.poses), dtype=bool),
            )

        self._apply_overrides(req)
        result = self.pipeline.generate_collision_free_grasps(
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
        # GPU lock in BaseServer serializes calls, so mutating shared state is safe.
        self.pipeline.grasp_threshold = req.grasp_threshold
        self.pipeline.num_grasps = req.num_grasps
        self.pipeline.topk_num_grasps = req.topk


def _mock_grasps(num_grasps: int, threshold: float) -> GenerateGraspsResponse:
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--gripper", default="franka_panda",
                        help="Gripper name (franka_panda, robotiq_2f_85/140, single_suction_cup_30mm)")
    parser.add_argument("--mock", action="store_true",
                        help="Skip real model loading; return fake grasps. "
                             "Useful before GraspGen is installed.")
    args = parser.parse_args()

    server = GraspGenServer(
        host=args.host, port=args.port, mock=args.mock, gripper=args.gripper,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
