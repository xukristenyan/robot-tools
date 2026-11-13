"""Headless adapter for the pinned official cuRobo V2 Python API."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
from engine import InputError, same_contents
from storage import configure_cache

configure_cache()

import torch
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.config_io import load_yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid as CuroboCuboid
from curobo.scene import Mesh as CuroboMesh
from curobo.scene import Scene
from curobo.types import ContentPath, GoalToolPose, JointState, Pose

from robot_tools.services.curobo.contract import Cuboid, GetRobotInfoResponse, World, StageContext
from contact_collision import ContactCollisionChecker

UPSTREAM_COMMIT = "42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83"
logger = logging.getLogger(__name__)


class CuroboBackend:
    def __init__(self, store, *, robot_config="franka.yml", collision_margin=0.01):
        if not np.isfinite(collision_margin) or not 0 <= collision_margin <= 0.1:
            raise ValueError("collision_margin must be between 0 and 0.1 metres")
        self._batch_ik_solvers = {}
        self._context_ik_banks = {}
        self._prepared_scene = None
        self._attachment_states = {}
        self._active_context = None
        self.store = store
        self.margin = collision_margin
        self._seen_meshes = set()
        config_path = ContentPath(robot_config_file=robot_config).get_robot_configuration_path()
        robot = load_yaml(config_path)["robot_cfg"]
        kin = robot["kinematics"]
        if len(kin["tool_frames"]) != 1:
            raise ValueError("This service requires a single tool frame in the robot configuration")
        kin["collision_sphere_buffer"] = float(kin.get("collision_sphere_buffer", 0)) + collision_margin
        self.attachment_frame = "attached_object" if "attached_object" in kin.get("extra_links", {}) else None
        if self.attachment_frame:
            kin.setdefault("extra_collision_spheres", {})[self.attachment_frame] = 64
        self._robot_config_input = copy.deepcopy(robot)
        # Each component owns its mutable geometry. The planner itself shares
        # its robot geometry between IK, TrajOpt and graph planning upstream.
        self.planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=copy.deepcopy(robot),
                scene_model=Scene(),
                collision_cache={"mesh": 128, "cuboid": 128},
                use_cuda_graph=True,
                self_collision_check=True,
                # v0.9.0 Fast: GPU BiRRT, B-spline TrajOpt and fixed-path TimeOpt.
                graph_planner_algorithm="rrt",
                enable_timeopt=True,
                interpolation_dt=0.025,
                position_tolerance=0.005,
                orientation_tolerance=0.05,
            )
        )
        self.raw_ik = InverseKinematics(
            InverseKinematicsCfg.create(
                robot=copy.deepcopy(robot),
                self_collision_check=False,
                use_cuda_graph=True,
                optimizer_configs=["ik/lbfgs_ik.yml"],
                num_seeds=32,
            )
        )
        self.checker = RobotCollisionChecker(
            RobotCollisionCheckerCfg.load_from_config(
                robot_config=copy.deepcopy(robot),
                scene_model=Scene(),
                n_meshes=128,
                n_cuboids=128,
                collision_activation_distance=0.0,
            )
        )
        self.info = GetRobotInfoResponse(
            robot_config=Path(robot_config).name,
            base_frame=kin["base_link"],
            tool_frame=self.planner.tool_frames[0],
            joint_names=self.planner.joint_names,
            joint_limits=self._numpy(self.planner.kinematics.get_joint_limits().position),
            default_position=self._numpy(self.planner.default_joint_state.position).reshape(-1),
            collision_links=kin["collision_link_names"],
            attachment_frame=self.attachment_frame,
            collision_margin_m=collision_margin,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=0.05,
            fake=False,
            upstream_commit=UPSTREAM_COMMIT,
        )
        scene_checker = self.planner.scene_collision_checker
        params = self.planner.kinematics.kinematics_config
        sphere_count = params.num_spheres
        self.contact_checker = ContactCollisionChecker(scene_checker.checker, scene_checker.device_cfg, sphere_count)
        scene_checker.checker = self.contact_checker
        logger.info("Warming up cuRobo on %s", torch.cuda.get_device_name())
        # Warm both collision kernels, IK modes and the planner before readiness.
        initial = self.info.default_position
        js = self._js(initial)
        tool_pose = self.planner.compute_kinematics(js).tool_poses.get_link_pose(self.info.tool_frame)
        goal = GoalToolPose.from_poses({self.info.tool_frame: tool_pose}, num_goalset=1)
        self.raw_ik.solve_pose(goal, current_state=js)
        self.planner.ik_solver.solve_pose(goal, current_state=js)
        if not self.planner.warmup(enable_graph=True, num_warmup_iterations=2):
            raise RuntimeError("cuRobo planner warmup failed")
        self.checker.validate(self._tensor(initial).reshape(1, 1, -1))
        batch_goal = np.r_[self._numpy(tool_pose.position).ravel(), self._numpy(tool_pose.quaternion).ravel()]
        for capacity in (1,4,8,16):
            self.ik_batch_contexts(np.repeat(batch_goal[None],capacity,axis=0),
                np.repeat(initial[None],capacity,axis=0), World(), [StageContext() for _ in range(capacity)], True)
        torch.cuda.synchronize()
        logger.info("cuRobo v0.9.0 ready: %s, tool=%s, graph=rrt, B-spline + TimeOpt, commit=%s",
                    robot_config, self.info.tool_frame, UPSTREAM_COMMIT)

    @staticmethod
    def _tensor(value):
        return torch.as_tensor(np.array(value, dtype=np.float32, copy=True), device="cuda")

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=True)

    def _js(self, position):
        return JointState.from_position(self._tensor(position).reshape(1, -1), joint_names=self.info.joint_names)

    def _goal(self, pose):
        pose = self._tensor(pose).reshape(1, 7)
        return GoalToolPose.from_poses(
            {self.info.tool_frame: Pose(position=pose[:, :3], quaternion=pose[:, 3:])},
            num_goalset=1,
        )

    def _scene(self, world, *, names=None, exclude=()):
        cuboids, meshes = [], []
        for obs in world.obstacles:
            if obs.name in exclude or (names is not None and obs.name not in names):
                continue
            if isinstance(obs, Cuboid):
                cuboids.append(CuroboCuboid(name=obs.name, dims=list(obs.dims), pose=obs.pose.tolist()))
            else:
                # Upstream caches BVHs by mesh name. Include geometry identity
                # so another caller's identically named object cannot reuse it.
                key = f"{obs.name}:{obs.mesh_id}"
                if key not in self._seen_meshes and len(self._seen_meshes) >= 128:
                    raise InputError("mesh_cache_capacity_exceeded")
                vertices, faces = self.store.mesh(obs.mesh_id)
                meshes.append(CuroboMesh(name=key, pose=obs.pose.tolist(), vertices=vertices, faces=faces))
                self._seen_meshes.add(key)
        return Scene(cuboid=cuboids, mesh=meshes)

    def _collision_kinematics(self):
        models = [self.planner.kinematics, self.checker.kinematics]
        if True in self._batch_ik_solvers:
            models.append(self._batch_ik_solvers[True].kinematics)
        return models

    def _attach(self, context):
        if self.attachment_frame is None:
            return
        desired = None if context.attachment is None else context.attachment.spheres
        # Each consumer owns geometry, including a batch solver created later.
        # Compare sphere values, not only the object's name or attachment flag.
        for kinematics in self._collision_kinematics():
            key = id(kinematics)
            if key in self._attachment_states and same_contents(self._attachment_states[key], desired):
                continue
            # A failed update must not leave a reusable record of the old state.
            self._attachment_states.pop(key, None)
            params = kinematics.kinematics_config
            params.reset_link_spheres(self.attachment_frame)
            if desired is not None:
                spheres = self._tensor(desired)
                spheres[:, 3] += self.margin
                params.update_link_spheres(self.attachment_frame, spheres)
            self._attachment_states[key] = None if desired is None else desired.copy()

    def prepare(self, world, context, collision):
        self._active_context = context
        self._attach(context)
        if collision:
            scene_state = (world.model_dump(), [c.model_dump() for c in context.allowed_contacts])
            if self._prepared_scene is not None and same_contents(scene_state, self._prepared_scene):
                return
            # The planner or contact partitions may be partly updated on error.
            # Invalidate before modifying GPU storage so a later request rebuilds it.
            self._prepared_scene = None
            contacts = {}
            for contact in context.allowed_contacts:
                contacts.setdefault(contact.obstacle, set()).update(contact.robot_links)
            partitions = {}
            for obstacle in world.obstacles:
                links = tuple(sorted(contacts.get(obstacle.name, ())))
                partitions.setdefault(links, set()).add(obstacle.name)
            params = self.planner.kinematics.kinematics_config
            groups = []
            for links, names in partitions.items():
                indices = torch.cat([params.get_sphere_index_from_link_name(link).reshape(-1)
                    for link in links]) if links else torch.empty(0, dtype=torch.long, device='cuda')
                groups.append((self._scene(world, names=names), indices.long()))
            # Keep the entire observed world. Only specific sphere/object pairs
            # are exempted, identically for IK, graph search and trajectory costs.
            self.planner.update_world(self._scene(world))
            if self.contact_checker.configure(groups):
                self.planner.ik_solver.reset_cuda_graph()
                if True in self._batch_ik_solvers:
                    self._batch_ik_solvers[True].reset_cuda_graph()
                self.planner.trajopt_solver.reset_cuda_graph()
                if self.planner.graph_planner is not None:
                    self.planner.graph_planner.reset_cuda_graph()
            self._prepared_scene = copy.deepcopy(scene_state)

    def reset(self):
        # Engine serializes requests and always prepares their complete context
        # before solving/checking. Defer physical detach to that next prepare;
        # it also removes a previous attachment when the new context has none.
        self._active_context = None

    def clear_geometry_cache(self):
        """Release trial geometry while keeping fixed solver/graph buffers valid."""
        torch.cuda.synchronize()
        self._prepared_scene = None
        self._active_context = None
        for model in self._collision_kinematics():
            if self.attachment_frame:
                model.kinematics_config.reset_link_spheres(self.attachment_frame)
        self._attachment_states.clear()
        self.planner.clear_scene_cache()  # Also discards graph-search nodes.
        scenes = [self.planner.scene_collision_checker, self.checker.scene_model]
        for scene, ignored in self.contact_checker.partitions:
            ignored.zero_()
            scenes.append(scene)
        for scene in scenes:
            scene.load_collision_model(Scene())
            meshes = scene.data.meshes
            if meshes is not None:
                # Public clear_cache only disables obstacles; it keeps Warp BVHs.
                # Zero captured handle buffers before releasing the BVHs. Keep
                # tensor allocations and CUDA captures for the next trial.
                meshes.mesh_ids.zero_()
        torch.cuda.synchronize()
        for scene in scenes:
            if scene.data.meshes is not None:
                scene.data.meshes.clear(clear_warp_cache=True)
        self._seen_meshes.clear()
        remaining = sum(bank.clear_geometry_cache() for bank in self._context_ik_banks.values())
        return remaining + sum(len(scene.data.meshes.wp_cache) for scene in scenes if scene.data.meshes is not None)

    def ik_batch_contexts(self, goals, seeds, world, contexts, collision):
        from batch_ik import BatchIK
        capacity = next(size for size in (1,4,8,16) if len(goals) <= size)
        key = (bool(collision),capacity)
        if key not in self._context_ik_banks:
            self._context_ik_banks[key] = BatchIK(self,capacity,collision)
        return self._context_ik_banks[key].solve(world,contexts,goals,seeds)

    def ik(self, goal, seed, collision):
        solver = self.planner.ik_solver if collision else self.raw_ik
        # cuRobo optimizes its costs with autograd even during inference.
        result = solver.solve_pose(self._goal(goal), current_state=self._js(seed), return_seeds=1)
        pe = float(result.position_error.reshape(-1)[0].item())
        re = float(result.rotation_error.reshape(-1)[0].item())
        pe = pe if np.isfinite(pe) else -1.0
        re = re if np.isfinite(re) else -1.0
        if not bool(result.success.reshape(-1)[0].item()):
            return None, pe, re
        q = self._numpy(solver.get_active_js(result.js_solution).position).reshape(-1, len(self.info.joint_names))[0]
        if not np.isfinite(q).all():
            return None, pe, re
        return q, pe, re

    def ik_batch(self, goals, seeds, collision):
        count = len(goals)
        if collision not in self._batch_ik_solvers:
            original = self.planner.ik_solver if collision else self.raw_ik
            # Independent optimizer/graphs preserve scalar-call performance.
            # Share the live scene checker so exact contact masks stay identical.
            # Build fresh optimizer configs: live configs contain mutable batch
            # sizes and CUDA pointers and cannot be safely copied after warmup.
            config = InverseKinematicsCfg.create(
                robot=copy.deepcopy(self._robot_config_input),
                optimizer_configs=["ik/lbfgs_ik.yml"],
                self_collision_check=collision,
                num_seeds=original.config.num_seeds,
                position_tolerance=original.config.position_tolerance,
                orientation_tolerance=original.config.orientation_tolerance,
                optimizer_collision_activation_distance=original.config.optimizer_collision_activation_distance,
                position_rank_weight=original.config.position_rank_weight,
                orientation_rank_weight=original.config.orientation_rank_weight,
                cspace_rank_weight=original.config.cspace_rank_weight,
                use_cuda_graph=True, max_batch_size=16)
            solver = InverseKinematics(config,
                scene_collision_checker=original.core.scene_collision_checker)
            self._batch_ik_solvers[collision] = solver
            if collision:
                self._attach(self._active_context)
        solver = self._batch_ik_solvers[collision]
        # Fixed capacity: this pinned runtime cannot recapture initialized CUDA
        # graphs. Upstream pads smaller batches and slices their results.
        poses = self._tensor(goals)
        goal = GoalToolPose.from_poses({self.info.tool_frame:
            Pose(position=poses[:, :3], quaternion=poses[:, 3:])}, num_goalset=1)
        current = JointState.from_position(self._tensor(seeds), joint_names=self.info.joint_names)
        result = solver.solve_pose(goal, current_state=current, return_seeds=1)
        positions = self._numpy(solver.get_active_js(result.js_solution).position).reshape(count, -1)
        success = result.success.detach().cpu().numpy().reshape(count).copy()
        success &= np.isfinite(positions).all(axis=1)
        positions[~success] = 0
        errors = []
        for value in (result.position_error, result.rotation_error):
            error = self._numpy(value).reshape(count)
            error[~np.isfinite(error)] = -1
            errors.append(error)
        return success, positions, *errors

    def plan(self, goal, current, context, attempts):
        if context.cartesian_path is None:
            # Keep the normal solve and TimeOpt; skip additional TrajOpt passes.
            result = self.planner.plan_pose(
                self._goal(goal), self._js(current), max_attempts=attempts, finetune_attempts=0
            )
        else:
            path = context.cartesian_path
            targets = self._tensor(path.poses)
            # [batch=1, waypoints, links=1, goalset=1, xyz/quaternion].
            waypoints = GoalToolPose(
                self.planner.tool_frames,
                targets[None, :, None, None, :3].contiguous(),
                targets[None, :, None, None, 3:].contiguous(),
            )
            result = self.planner.plan_pose_waypoints(
                waypoints, self._js(current), max_attempts=attempts, finetune_attempts=0,
                path_tolerance=path.path_tolerance,
                final_tolerance=path.final_tolerance,
                track_orientation=path.track_orientation,
            )
        if result is None or not bool(result.success.any().item()):
            return None, None
        trajectory = result.get_interpolated_plan()
        q = self._numpy(self.planner.trajopt_solver.get_active_js(trajectory).position).reshape(
            -1, len(self.info.joint_names)
        )
        if len(q) == 0 or not np.isfinite(q).all():
            return None, None
        return q, float(self.planner.trajopt_solver.config.interpolation_dt)

    @staticmethod
    def _violations(value, count):
        # cuRobo collision outputs are nonnegative penetration costs, NOT
        # signed clearance distances. Non-finite output fails closed.
        array = value.detach().cpu().numpy().reshape(count, -1)
        return ((array > 0) | ~np.isfinite(array)).any(axis=1)

    def collisions(self, positions, world, context):
        contacts = {}
        for contact in context.allowed_contacts:
            contacts.setdefault(contact.obstacle, set()).update(contact.robot_links)
        world_bad = np.zeros(len(positions), dtype=bool)
        self_bad = np.zeros(len(positions), dtype=bool)
        normal_scene = self._scene(world, exclude=set(contacts))
        contact_scenes = {name: self._scene(world, names={name}) for name in contacts}
        with torch.no_grad():
            for start in range(0, len(positions), 512):
                end = min(start + 512, len(positions))
                count = end - start
                q = self._tensor(positions[start:end]).reshape(count, 1, -1)
                state = self.checker.get_kinematics(q)
                spheres = state.robot_spheres.clone()
                self_bad[start:end] = self._violations(self.checker.get_self_collision(spheres), count)
                self.checker.update_world(normal_scene)
                world_bad[start:end] = self._violations(self.checker.get_collision_constraint(spheres), count)
                for name, links in contacts.items():
                    query = spheres.clone()
                    for link in links:
                        indices = self.checker.kinematics.kinematics_config.get_sphere_index_from_link_name(link)
                        query[:, :, indices, 3] = -100.0
                    self.checker.update_world(contact_scenes[name])
                    world_bad[start:end] |= self._violations(self.checker.get_collision_constraint(query), count)
        return world_bad, self_bad
