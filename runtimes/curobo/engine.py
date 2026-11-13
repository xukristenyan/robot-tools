"""Request isolation, stage orchestration and bounded geometry storage.

This module has no GPU dependencies. Backends implement only robot operations;
the same orchestration and failure handling runs in fake and real modes.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np

from robot_tools.services.curobo.contract import (
    CheckTrajectoryResponse,
    Mesh,
    PlanMotionResponse,
    PutMeshResponse,
    PutSceneResponse,
    SceneRef,
    SolveIKResponse,
    SolveIKBatchResponse,
    StageResult,
    ReleaseResourcesResponse,
)


class InputError(ValueError):
    """Stable task failure, safe to expose over the service boundary."""


def same_contents(a, b):
    """Compare actual scene/context values without a content digest."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_contents(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_contents(x, y) for x, y in zip(a, b))
    return a == b


def densify(positions, step, max_samples=20000):
    """Interpolate every edge in joint space, retaining both ends and all knots."""
    positions = np.asarray(positions, dtype=np.float32)
    with np.errstate(over="ignore", divide="ignore"):
        counts = np.maximum(1, np.ceil(np.max(np.abs(np.diff(positions.astype(np.float64), axis=0)), axis=1) / step))
    if not np.isfinite(counts).all() or counts.sum() + 1 > max_samples:
        raise InputError("sample_limit_exceeded")
    counts = counts.astype(int)
    parts = [positions[:1]]
    for a, b, count in zip(positions[:-1], positions[1:], counts):
        parts.append(np.linspace(a, b, count + 1, dtype=np.float32)[1:])
    return np.concatenate(parts)


class SceneStore:
    def __init__(self, asset_dir: Path | None, capacity=32):
        self.asset_dir = asset_dir
        self.capacity = capacity
        self.scenes = OrderedDict()
        self.meshes = OrderedDict()
        self._released_mesh_ids = set()
        if asset_dir is not None:
            asset_dir.mkdir(parents=True, exist_ok=True)

    def put_mesh(self, request):
        vertices = np.ascontiguousarray(request.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(request.faces, dtype=np.int32)
        mesh_id = uuid4().hex + uuid4().hex
        if self.asset_dir is not None:
            target = self.asset_dir / f"{mesh_id}.npz"
            if not target.exists():
                with tempfile.NamedTemporaryFile(dir=self.asset_dir, suffix=".npz", delete=False) as file:
                    temporary = Path(file.name)
                    try:
                        np.savez(file, vertices=vertices, faces=faces)
                        file.flush()
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
        else:
            self.meshes[mesh_id] = (vertices.copy(), faces.copy())
            while len(self.meshes) > self.capacity:
                self.meshes.popitem(last=False)
        return PutMeshResponse(mesh_id=mesh_id)

    def mesh(self, mesh_id):
        if self.asset_dir is None:
            if mesh_id not in self.meshes:
                raise InputError("mesh_not_found")
            return self.meshes[mesh_id]
        path = self.asset_dir / f"{mesh_id}.npz"
        if not path.is_file():
            self.meshes.pop(mesh_id, None)
            raise InputError("mesh_not_found")
        if mesh_id in self.meshes:
            self.meshes.move_to_end(mesh_id)
            return self.meshes[mesh_id]
        with np.load(path, allow_pickle=False) as data:
            # Uploaded IDs are immutable. Consumers treat these arrays as read-only.
            arrays = data["vertices"], data["faces"]
        self.meshes[mesh_id] = arrays
        while len(self.meshes) > self.capacity:
            self.meshes.popitem(last=False)
        return arrays

    def put_scene(self, request):
        try:
            for obstacle in request.world.obstacles:
                if isinstance(obstacle, Mesh):
                    self.mesh(obstacle.mesh_id)
        except InputError as exc:
            return PutSceneResponse(failure_code=str(exc))
        world = request.world.model_copy(deep=True)
        version = uuid4().hex + uuid4().hex
        key = (request.scene_id, version)
        self.scenes[key] = world
        self.scenes.move_to_end(key)
        while len(self.scenes) > self.capacity:
            self.scenes.popitem(last=False)
        return PutSceneResponse(scene=SceneRef(scene_id=request.scene_id, version=version))

    def get(self, reference):
        key = (reference.scene_id, reference.version)
        if key not in self.scenes:
            raise InputError("scene_not_found")
        self.scenes.move_to_end(key)
        return self.scenes[key].model_copy(deep=True)

    def release_resources(self, scenes, mesh_ids):
        released_scenes = 0
        for scene in scenes:
            released_scenes += self.scenes.pop((scene.scene_id, scene.version), None) is not None
        self._released_mesh_ids.update(mesh_ids)
        referenced = {obstacle.mesh_id for world in self.scenes.values()
                      for obstacle in world.obstacles if isinstance(obstacle, Mesh)}
        released_meshes = 0
        for mesh_id in self._released_mesh_ids - referenced:
            existed = self.meshes.pop(mesh_id, None) is not None
            if self.asset_dir is not None:
                path = self.asset_dir / f'{mesh_id}.npz'
                existed = path.is_file() or existed
                path.unlink(missing_ok=True)
            released_meshes += existed
            self._released_mesh_ids.remove(mesh_id)
        return released_scenes, released_meshes, len(self._released_mesh_ids)


class Engine:
    def __init__(self, backend, store):
        self.backend = backend
        self.store = store
        self.info = backend.info

    def release_resources(self, request):
        # The server serializes this handler with all IK/check/plan requests.
        # Invalidate shared GPU workspace before dropping backing mesh arrays.
        cached_meshes = self.backend.clear_geometry_cache()
        scenes, meshes, deferred = self.store.release_resources(request.scenes, request.mesh_ids)
        return ReleaseResourcesResponse(released_scenes=scenes, released_meshes=meshes,
                                        deferred_meshes=deferred, cached_meshes=cached_meshes)

    def _order(self, names):
        expected = self.info.joint_names
        if set(names) != set(expected):
            raise InputError("joint_names_mismatch")
        return [names.index(n) for n in expected], [expected.index(n) for n in names]

    def _context(self, world, context):
        names = {o.name for o in world.obstacles}
        used = set(context.obstacle_poses) | {c.obstacle for c in context.allowed_contacts}
        if context.attachment is not None:
            used.add(context.attachment.object_name)
            if self.info.attachment_frame is None:
                raise InputError("attachment_not_supported")
        if not used <= names:
            raise InputError("unknown_obstacle")
        links = {link for c in context.allowed_contacts for link in c.robot_links}
        if not links <= set(self.info.collision_links):
            raise InputError("unknown_collision_link")
        if context.attachment and context.attachment.object_name in {c.obstacle for c in context.allowed_contacts}:
            raise InputError("contact_with_attached_world_object")
        result = world.model_copy(deep=True)
        for obstacle in result.obstacles:
            if obstacle.name in context.obstacle_poses:
                obstacle.pose = context.obstacle_poses[obstacle.name].copy()
        if context.attachment:
            result.obstacles = [o for o in result.obstacles if o.name != context.attachment.object_name]
        return result

    def solve_ik(self, request):
        n, d = len(request.goal_poses), len(request.joint_names)
        success = np.zeros(n, dtype=bool)
        positions = np.zeros((n, d), dtype=np.float32)
        pos_error = np.full(n, -1, dtype=np.float32)
        rot_error = pos_error.copy()
        failure = None
        try:
            order, inverse = self._order(request.joint_names)
            world = self._context(self.store.get(request.scene), request.context)
            seed = request.seed_position[order]
            self.backend.prepare(world, request.context, request.collision_check)
            for i, goal in enumerate(request.goal_poses):
                q, pe, re = self.backend.ik(goal, seed, request.collision_check)
                pos_error[i], rot_error[i] = pe, re
                if q is not None:
                    valid = not self._bounds(q[None]).any()
                    if valid and request.collision_check:
                        masks = self.backend.collisions(q[None], world, request.context)
                        valid = not (masks[0] | masks[1]).any()
                    if valid:
                        success[i], positions[i] = True, q[inverse]
        except InputError as exc:
            failure = str(exc)
        finally:
            self.backend.reset()
        return SolveIKResponse(
            success=success,
            positions=positions,
            position_error_m=pos_error,
            rotation_error_rad=rot_error,
            collision_checked=request.collision_check,
            failure_code=failure,
            fake=self.info.fake,
        )

    def solve_ik_batch(self, request):
        # Same world/contact policy can use native per-env attachment geometry.
        # Validate each row before grouping, so one bad input cannot poison peers.
        groups, results, sizes = [], [None] * len(request.requests), []
        for index, item in enumerate(request.requests):
            try:
                self._order(item.joint_names)
                self._context(self.store.get(item.scene), item.context)
            except InputError as error:
                results[index] = SolveIKResponse(success=np.zeros(1, bool),
                    positions=np.zeros((1,len(item.joint_names)),np.float32),
                    position_error_m=np.full(1,-1,np.float32),rotation_error_rad=np.full(1,-1,np.float32),
                    collision_checked=item.collision_check,failure_code=str(error),fake=self.info.fake)
                continue
            key = (item.scene.scene_id, item.scene.version, item.collision_check)
            context = item.context.model_dump()
            if context['attachment'] is not None:
                context['attachment'] = {'object_name':context['attachment']['object_name']}
            for group_key, group_context, indices in groups:
                if key == group_key and same_contents(context, group_context) and len(indices) < 16:
                    indices.append(index)
                    break
            else:
                groups.append((key, context, [index]))
        for _, _, indices in groups:
            first = request.requests[indices[0]]
            sizes.append(len(indices))
            try:
                orders = [self._order(request.requests[i].joint_names) for i in indices]
                world = self._context(self.store.get(first.scene), first.context)
                goals = np.stack([request.requests[i].goal_poses[0] for i in indices])
                seeds = np.stack([request.requests[i].seed_position[order]
                                  for i, (order, _) in zip(indices, orders)])
                contexts = [request.requests[i].context for i in indices]
                success, positions, pe, re = self.backend.ik_batch_contexts(
                    goals, seeds, world, contexts, first.collision_check)
                success &= ~self._bounds(positions)
                for row, (index, (_, inverse)) in enumerate(zip(indices, orders)):
                    q = positions[row, inverse] if success[row] else np.zeros(len(inverse), np.float32)
                    results[index] = SolveIKResponse(success=success[row:row+1], positions=q[None],
                        position_error_m=pe[row:row+1], rotation_error_rad=re[row:row+1],
                        collision_checked=first.collision_check, fake=self.info.fake)
            except InputError as error:
                for index in indices:
                    results[index] = SolveIKResponse(success=np.zeros(1, bool),
                        positions=np.zeros((1, len(request.requests[index].joint_names)), np.float32),
                        position_error_m=np.full(1, -1, np.float32), rotation_error_rad=np.full(1, -1, np.float32),
                        collision_checked=first.collision_check, failure_code=str(error), fake=self.info.fake)
            finally:
                self.backend.reset()
        return SolveIKBatchResponse(results=results, group_sizes=sizes)

    def _bounds(self, positions):
        lower, upper = self.info.joint_limits
        return ((positions < lower) | (positions > upper)).any(axis=1)

    def _check(self, positions, world, context, step):
        samples = densify(positions, step)
        world_bad, self_bad = self.backend.collisions(samples, world, context)
        bounds = self._bounds(samples)
        bad = world_bad | self_bad | bounds
        first = int(np.flatnonzero(bad)[0]) if bad.any() else None
        return samples, world_bad, self_bad, bounds, first

    def check_trajectory(self, request):
        failure = None
        samples = np.empty((0, len(request.joint_names)), dtype=np.float32)
        w = s = b = np.empty(0, dtype=bool)
        first = None
        try:
            order, inverse = self._order(request.joint_names)
            world = self._context(self.store.get(request.scene), request.context)
            self.backend.prepare(world, request.context, True)
            samples, w, s, b, first = self._check(
                request.positions[:, order], world, request.context, request.max_joint_step_rad
            )
            samples = samples[:, inverse]
        except InputError as exc:
            failure = str(exc)
        finally:
            self.backend.reset()
        return CheckTrajectoryResponse(
            valid=failure is None and first is None,
            sampled_positions=samples,
            world_collision=w,
            self_collision=s,
            joint_limit_violation=b,
            first_invalid_index=first,
            failure_code=failure,
            fake=self.info.fake,
        )

    def plan_motion(self, request):
        results = []
        stage_name = request.stages[0].name
        failure = None
        collision = request.mode == "collision_aware"
        try:
            order, inverse = self._order(request.joint_names)
            world = self.store.get(request.scene)
            current = request.start_position[order].copy()
            if self._bounds(current[None]).any():
                raise InputError("start_joint_limits")
            for stage in request.stages:
                stage_name = stage.name
                start = time.perf_counter()
                stage_world = self._context(world, stage)
                self.backend.prepare(stage_world, stage, collision)
                if collision:
                    w, s = self.backend.collisions(current[None], stage_world, stage)
                    if (w | s).any():
                        raise InputError("start_in_collision")
                    trajectory, dt = self.backend.plan(stage.goal_pose, current, stage, request.max_attempts)
                    if trajectory is None:
                        raise InputError("planning_failed")
                    if not np.allclose(trajectory[0], current, atol=1e-3):
                        raise InputError("trajectory_start_mismatch")
                    _, _, _, _, first = self._check(trajectory, stage_world, stage, request.max_joint_step_rad)
                    if first is not None:
                        raise InputError("trajectory_invalid")
                else:
                    goal, _, _ = self.backend.ik(stage.goal_pose, current, False)
                    if goal is None:
                        raise InputError("ik_failed")
                    trajectory = densify(np.stack([current, goal]), request.max_joint_step_rad)
                    dt = None
                    if self._bounds(trajectory).any():
                        raise InputError("joint_limits")
                results.append(
                    StageResult(
                        name=stage.name,
                        success=True,
                        positions=trajectory[:, inverse],
                        dt=dt,
                        collision_checked=collision,
                        timing_ms=(time.perf_counter() - start) * 1000,
                    )
                )
                current = trajectory[-1].copy()
        except InputError as exc:
            failure = str(exc)
            results.append(
                StageResult(
                    name=stage_name,
                    success=False,
                    positions=np.empty((0, len(request.joint_names)), dtype=np.float32),
                    collision_checked=collision,
                    failure_code=failure,
                    timing_ms=0,
                )
            )
        finally:
            self.backend.reset()
        return PlanMotionResponse(
            success=failure is None,
            stages=results,
            failed_stage=stage_name if failure else None,
            failure_code=failure,
            fake=self.info.fake,
        )


class FakeBackend:
    """Deterministic protocol fixture; never a substitute for robot geometry."""

    def __init__(self):
        from robot_tools.services.curobo.contract import GetRobotInfoResponse

        self.info = GetRobotInfoResponse(
            robot_config="fake-franka",
            base_frame="panda_link0",
            tool_frame="panda_hand",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            joint_limits=np.array(
                [[-2.9, -1.8, -2.9, -3.1, -2.9, -0.1, -2.9], [2.9, 1.8, 2.9, -0.01, 2.9, 3.8, 2.9]], dtype=np.float32
            ),
            default_position=np.array([0, -1.3, 0, -2.5, 0, 1.5, 0.8], dtype=np.float32),
            collision_links=["panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"],
            attachment_frame="attached_object",
            collision_margin_m=0.01,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=0.05,
            fake=True,
            upstream_commit="fake",
        )

    def prepare(self, world, context, collision):
        pass

    def reset(self):
        pass

    def clear_geometry_cache(self):
        return 0

    def ik(self, goal, seed, collision):
        if np.linalg.norm(goal[:3]) > 2:
            return None, 1.0, 1.0
        q = seed.astype(np.float32).copy()
        q[0] = goal[0] * 0.1
        return q, 0.0, 0.0

    def ik_batch(self, goals, seeds, collision):
        # Deterministic protocol-test backend; the real backend uses GPU batches.
        success = np.linalg.norm(goals[:, :3], axis=1) <= 2
        positions = seeds.astype(np.float32).copy()
        positions[:, 0] = goals[:, 0] * .1
        error = (~success).astype(np.float32)
        return success, positions, error, error.copy()

    def ik_batch_contexts(self, goals, seeds, world, contexts, collision):
        success, positions, pe, re = self.ik_batch(goals,seeds,collision)
        if collision:
            for i in np.flatnonzero(success):
                self.prepare(world,contexts[i],collision)
                w,s=self.collisions(positions[i:i+1],world,contexts[i])
                success[i] &= not (w|s).any()
        return success,positions,pe,re

    def plan(self, goal, current, context, attempts):
        if context.cartesian_path is not None:
            raise InputError("cartesian_requires_real_backend")
        q, _, _ = self.ik(goal, current, True)
        return (None, None) if q is None else (densify(np.stack([current, q]), 0.02), 0.025)

    def collisions(self, positions, world, context):
        return np.zeros(len(positions), dtype=bool), np.zeros(len(positions), dtype=bool)
