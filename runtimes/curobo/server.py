"""cuRobo task-level IK, collision checking and multi-stage planning service."""

from __future__ import annotations

import argparse
import logging

from engine import Engine, FakeBackend, SceneStore

from robot_tools.core.server import BaseServer, handler
from robot_tools.services.curobo.contract import (
    API_VERSION,
    SERVICE_ID,
    CheckTrajectoryRequest,
    GetRobotInfoRequest,
    PlanMotionRequest,
    PutMeshRequest,
    PutSceneRequest,
    SolveIKRequest,
    SolveIKBatchRequest,
    ReleaseResourcesRequest,
)


class CuroboServer(BaseServer):
    def __init__(self, host="127.0.0.1", port=5562, *, fake=False, robot_config="franka.yml", collision_margin=0.01):
        super().__init__(service_id=SERVICE_ID, api_version=API_VERSION, host=host, port=port)
        if fake:
            store = SceneStore(None)
            backend = FakeBackend()
        else:
            from storage import configure_cache

            store = SceneStore(configure_cache() / "assets")
            from backend import CuroboBackend

            backend = CuroboBackend(store, robot_config=robot_config, collision_margin=collision_margin)
        self.engine = Engine(backend, store)
        self.mark_ready()

    @handler(GetRobotInfoRequest)
    def get_robot_info(self, request):
        return self.engine.info.model_copy(deep=True)

    @handler(ReleaseResourcesRequest)
    def release_resources(self, request):
        return self.engine.release_resources(request)

    @handler(PutMeshRequest)
    def put_mesh(self, request):
        return self.engine.store.put_mesh(request)

    @handler(PutSceneRequest)
    def put_scene(self, request):
        return self.engine.store.put_scene(request)

    @handler(SolveIKRequest)
    def solve_ik(self, request):
        return self.engine.solve_ik(request)

    @handler(SolveIKBatchRequest)
    def solve_ik_batch(self, request):
        return self.engine.solve_ik_batch(request)

    @handler(CheckTrajectoryRequest)
    def check_trajectory(self, request):
        return self.engine.check_trajectory(request)

    @handler(PlanMotionRequest)
    def plan_motion(self, request):
        return self.engine.plan_motion(request)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--robot-config", default="franka.yml")
    parser.add_argument("--collision-margin", type=float, default=0.01, help="Robot sphere inflation in metres")
    args = parser.parse_args()
    if not 0 <= args.collision_margin <= 0.1:
        parser.error("--collision-margin must be in [0, 0.1] metres")
    CuroboServer(**vars(args)).serve_forever()


if __name__ == "__main__":
    main()
