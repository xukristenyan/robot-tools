from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from pydantic import ValidationError

from robot_tools.services.curobo import Cuboid, CuroboClient, MotionStage, StageContext, World, occupied_voxel_mesh
from robot_tools.services.curobo.contract import (
    ACTIONS,
    CheckTrajectoryRequest,
    CheckTrajectoryResponse,
    PlanMotionRequest,
    PutMeshRequest,
    PutSceneRequest,
    SceneRef,
    SolveIKRequest,
)

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes" / "curobo"


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNTIME))
    spec = importlib.util.spec_from_file_location("curobo_test_server", RUNTIME / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine(runtime):
    return runtime.CuroboServer(fake=True).engine


@pytest.fixture
def client(runtime):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    service = runtime.CuroboServer(fake=True, port=port)
    server = uvicorn.Server(uvicorn.Config(service._build_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        with CuroboClient(host="127.0.0.1", port=port, timeout_ms=5000) as result:
            yield result
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def pose(x=0.4):
    return np.array([x, 0, 0.4, 0, 1, 0, 0], dtype=np.float32)


def scene(engine, scene_id="test", world=None):
    return engine.store.put_scene(PutSceneRequest(scene_id=scene_id, world=world or World())).scene


def test_fake_http_roundtrip_all_actions_and_joint_reordering(client):
    assert set(client.health()["actions"]) == ACTIONS
    info = client.get_robot_info()
    assert info.fake and len(info.joint_names) == 7
    vertices, faces = occupied_voxel_mesh(np.zeros((1, 3), dtype=np.float32))
    assert len(client.put_mesh(vertices, faces).mesh_id) == 64
    reference = client.put_scene("one", World()).scene
    ik = client.solve_ik(pose(), info.default_position, scene=reference, joint_names=info.joint_names)
    assert ik.success.tolist() == [True] and ik.positions.dtype == np.float32
    stages = [MotionStage(name=f"stage{i}", goal_pose=pose(0.4 + i * 0.01)) for i in range(6)]
    plan = client.plan_motion(info.default_position, stages, scene=reference, joint_names=info.joint_names)
    assert plan.success and len(plan.stages) == 6
    for a, b in zip(plan.stages, plan.stages[1:]):
        np.testing.assert_array_equal(a.positions[-1], b.positions[0])
    checked = client.check_trajectory(plan.stages[0].positions, scene=reference, joint_names=info.joint_names)
    assert checked.valid and checked.first_invalid_index is None
    reverse = client.solve_ik(pose(), info.default_position[::-1], scene=reference, joint_names=info.joint_names[::-1])
    np.testing.assert_allclose(reverse.positions[:, ::-1], ik.positions)


def test_failed_stage_stops_task_and_ik_only_is_untimed(client):
    info = client.get_robot_info()
    reference = client.put_scene("task", World()).scene
    result = client.plan_motion(
        info.default_position,
        [
            MotionStage(name="reach", goal_pose=pose()),
            MotionStage(name="unreachable", goal_pose=pose(100)),
            MotionStage(name="must_not_run", goal_pose=pose()),
        ],
        scene=reference,
        joint_names=info.joint_names,
        mode="ik_only",
    )
    assert not result.success
    assert result.failed_stage == "unreachable" and result.failure_code == "ik_failed"
    assert len(result.stages) == 2 and result.stages[1].positions.shape == (0, 7)
    assert result.stages[0].dt is None and not result.stages[0].collision_checked


def test_scene_versions_are_immutable_and_missing_version_fails_closed(engine):
    a = scene(engine, "same-id")
    world = World(obstacles=[Cuboid(name="block", pose=pose(), dims=(1, 1, 1))])
    b = scene(engine, "same-id", world)
    assert a.version != b.version
    assert not engine.store.get(a).obstacles
    assert len(engine.store.get(b).obstacles) == 1
    request = CheckTrajectoryRequest(
        scene=SceneRef(scene_id="same-id", version="0" * 64),
        joint_names=engine.info.joint_names,
        positions=engine.info.default_position[None],
    )
    result = engine.check_trajectory(request)
    assert not result.valid and result.failure_code == "scene_not_found"
    assert len(result.sampled_positions) == 0


def test_densification_detects_collision_between_valid_knots(engine):
    def collision(positions, world, context):
        return np.abs(positions[:, 0]) < 0.01, np.zeros(len(positions), dtype=bool)

    engine.backend.collisions = collision
    q = np.tile(engine.info.default_position, (2, 1))
    q[:, 0] = [-0.2, 0.2]
    result = engine.check_trajectory(
        CheckTrajectoryRequest(
            scene=scene(engine),
            joint_names=engine.info.joint_names,
            positions=q,
        )
    )
    assert not result.valid and result.first_invalid_index not in (0, len(result.sampled_positions) - 1)
    assert np.max(np.abs(np.diff(result.sampled_positions, axis=0))) <= 0.020001
    np.testing.assert_array_equal(result.sampled_positions[[0, -1]], q)


def test_sample_budget_fails_closed_before_gpu_work(engine):
    q = np.tile(engine.info.default_position, (2, 1))
    q[1, 0] = 1000
    result = engine.check_trajectory(
        CheckTrajectoryRequest(
            scene=scene(engine),
            joint_names=engine.info.joint_names,
            positions=q,
            max_joint_step_rad=0.001,
        )
    )
    assert not result.valid and result.failure_code == "sample_limit_exceeded"


def test_attachment_and_contact_context_does_not_mutate_scene(engine):
    from robot_tools.services.curobo import AllowedContact, Attachment

    world = World(obstacles=[Cuboid(name="object", pose=pose(), dims=(0.1, 0.1, 0.1))])
    ref = scene(engine, world=world)
    ctx = StageContext(
        attachment=Attachment(object_name="object", spheres=np.array([[0, 0, 0, 0.1]], dtype=np.float32))
    )
    assert not engine._context(engine.store.get(ref), ctx).obstacles
    assert len(engine._context(engine.store.get(ref), StageContext()).obstacles) == 1
    with pytest.raises(ValueError, match="unknown_collision_link"):
        engine._context(world, StageContext(allowed_contacts=[AllowedContact(obstacle="object", robot_links=["typo"])]))


def test_invalid_joint_set_and_start_limits_fail_closed(engine):
    request = PlanMotionRequest(
        scene=scene(engine),
        joint_names=["wrong"],
        start_position=np.zeros(1, dtype=np.float32),
        stages=[MotionStage(name="reach", goal_pose=pose())],
    )
    assert engine.plan_motion(request).failure_code == "joint_names_mismatch"
    q = engine.info.default_position.copy()
    q[3] = 0.5
    request = request.model_copy(update={"joint_names": engine.info.joint_names, "start_position": q})
    assert engine.plan_motion(request).failure_code == "start_joint_limits"


def test_backend_state_reset_even_after_unexpected_failure(engine):
    reset = []

    def broken(*args):
        raise RuntimeError("GPU failed")

    engine.backend.ik = broken
    engine.backend.reset = lambda: reset.append(True)
    request = SolveIKRequest(
        scene=scene(engine),
        joint_names=engine.info.joint_names,
        goal_poses=pose()[None],
        seed_position=engine.info.default_position,
    )
    with pytest.raises(RuntimeError, match="GPU failed"):
        engine.solve_ik(request)
    assert reset == [True]


def test_plan_rejects_midpath_collision_without_returning_failed_trajectory(engine):
    def collision(positions, world, context):
        return (positions[:, 0] > 0.01) & (positions[:, 0] < 0.03), np.zeros(len(positions), dtype=bool)

    engine.backend.collisions = collision
    request = PlanMotionRequest(
        scene=scene(engine),
        joint_names=engine.info.joint_names,
        start_position=engine.info.default_position,
        stages=[MotionStage(name="reach", goal_pose=pose())],
    )
    result = engine.plan_motion(request)
    assert not result.success and result.failure_code == "trajectory_invalid"
    assert result.stages[0].positions.shape == (0, 7)


def test_extreme_sampling_request_has_bounded_allocation(engine):
    request = CheckTrajectoryRequest(
        scene=scene(engine),
        joint_names=engine.info.joint_names,
        positions=np.stack([engine.info.default_position, engine.info.default_position + 1]),
        max_joint_step_rad=1e-300,
    )
    result = engine.check_trajectory(request)
    assert not result.valid and result.failure_code == "sample_limit_exceeded"


def test_client_rejects_bad_input_before_transport(monkeypatch):
    def transport(*args):
        raise AssertionError("invalid input reached the transport")

    monkeypatch.setattr(CuroboClient, "_request", transport)
    with CuroboClient(wait=False) as client, pytest.raises(ValidationError):
        client.solve_ik(
            np.zeros((1, 7), dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            scene=SceneRef(scene_id="one", version="0" * 64),
            joint_names=[f"j{i}" for i in range(7)],
        )


def test_curobo_registration_and_selected_install(tmp_path, monkeypatch):
    from robot_tools import launcher

    profile = tmp_path / "services.yaml"
    profile.write_text("services:\n  - name: curobo\n")
    initialized, scripts = [], []
    monkeypatch.setattr(launcher, "_init_upstream", lambda spec: initialized.append(spec.service_id))
    monkeypatch.setattr(launcher, "_sync_one", lambda spec, extras: None)
    monkeypatch.setattr(launcher, "_run_runtime_script", lambda spec, name: scripts.append((spec.service_id, name)))
    launcher.install(profile)
    assert initialized == ["curobo"] and scripts == [("curobo", "install.sh")]
    assert launcher._status_client_types()["curobo"] is CuroboClient


@pytest.mark.parametrize("bad", [np.zeros(7, dtype=np.float32), np.array([0, 0, 0, 1, 0, 0, np.nan], dtype=np.float32)])
def test_contract_rejects_nonunit_or_nonfinite_pose(bad):
    with pytest.raises(ValidationError):
        MotionStage(name="reach", goal_pose=bad)


def test_contract_rejects_misaligned_response():
    with pytest.raises(ValidationError, match="align"):
        CheckTrajectoryResponse(
            valid=True,
            sampled_positions=np.zeros((2, 7), dtype=np.float32),
            world_collision=np.zeros(1, dtype=bool),
            self_collision=np.zeros(2, dtype=bool),
            joint_limit_violation=np.zeros(2, dtype=bool),
        )


def test_mesh_validation_and_opaque_storage_ids(engine, tmp_path):
    vertices, faces = occupied_voxel_mesh(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValidationError):
        PutMeshRequest(vertices=vertices, faces=faces.astype(np.float32))
    from engine import SceneStore

    store = SceneStore(tmp_path)
    request = PutMeshRequest(vertices=vertices, faces=faces)
    a, b = store.put_mesh(request), store.put_mesh(request)
    assert a.mesh_id != b.mesh_id and len(list(tmp_path.glob("*.npz"))) == 2
    for actual, expected in zip(store.mesh(a.mesh_id), (vertices, faces)):
        np.testing.assert_array_equal(actual, expected)


def test_cartesian_path_contract_and_http_failure(client):
    from robot_tools.services.curobo import CartesianPath

    targets = np.stack([pose(0.42), pose(0.44)])
    path = CartesianPath(poses=targets)
    stage = MotionStage(name="follow", goal_pose=targets[-1], cartesian_path=path)
    assert "cartesian_path" not in MotionStage(name="pose", goal_pose=pose()).model_dump()
    assert "cartesian_path" in stage.model_dump()
    info = client.get_robot_info()
    reference = client.put_scene("cartesian", World()).scene
    result = client.plan_motion(info.default_position, [stage], scene=reference, joint_names=info.joint_names)
    assert not result.success and result.failure_code == "cartesian_requires_real_backend"
    with pytest.raises(ValidationError, match="collision_aware"):
        PlanMotionRequest(start_position=info.default_position, stages=[stage],
            scene=reference, joint_names=info.joint_names, mode="ik_only")
    with pytest.raises(ValidationError, match="end at goal_pose"):
        MotionStage(name="wrong-end", goal_pose=pose(0.6), cartesian_path=path)
    for invalid in (targets[:0], np.tile(targets, (37, 1)), targets[:, :3]):
        with pytest.raises(ValidationError):
            CartesianPath(poses=invalid)
    with pytest.raises(ValidationError):
        CartesianPath(poses=targets, path_tolerance=(0.0, 0.1))
    # q and -q encode the same final orientation.
    opposite = targets[-1].copy()
    opposite[3:] *= -1
    MotionStage(name="same-rotation", goal_pose=opposite, cartesian_path=path)


def test_voxel_mesh_has_outward_faces_and_removes_internal_faces():
    vertices, faces = occupied_voxel_mesh(np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float32), 1)
    assert vertices.shape == (12, 3) and faces.shape == (20, 3)
    triangles = vertices[faces]
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    assert volume == pytest.approx(2)


def test_fake_starts_without_backend_or_upstream(tmp_path):
    for filename in ("server.py", "engine.py"):
        (tmp_path / filename).write_text((RUNTIME / filename).read_text())
    code = "from server import CuroboServer; import sys; s=CuroboServer(fake=True); assert s.engine.info.fake; assert 'torch' not in sys.modules; assert 'curobo' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr
