"""Fixed-capacity native multi-environment IK with per-grasp link geometry.

Official IKSolverCfg.multi_env and KinematicsParams.update_link_spheres(config_idx)
keep every candidate's attachment separate. Contact permissions and world poses
must be identical inside one engine group; other policies use separate calls.
"""
import copy
import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose
from contact_collision import ContactCollisionChecker
from engine import same_contents


class BatchIK:
    def __init__(self, backend, capacity, collision):
        self.backend, self.capacity, self.collision = backend, capacity, collision
        self.scene_state = None
        self.checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=copy.deepcopy(backend._robot_config_input), scene_model=Scene(),
            num_envs=capacity,n_meshes=128,n_cuboids=128,collision_activation_distance=0.))
        native = self.checker.scene_model
        self.contacts = ContactCollisionChecker(native.checker,native.device_cfg,
            self.checker.kinematics.kinematics_config.num_spheres,num_envs=capacity)
        native.checker = self.contacts
        original = backend.planner.ik_solver if collision else backend.raw_ik
        config = InverseKinematicsCfg.create(robot=copy.deepcopy(backend._robot_config_input),
            optimizer_configs=['ik/lbfgs_ik.yml'],self_collision_check=collision,
            num_seeds=original.config.num_seeds,
            position_tolerance=original.config.position_tolerance,
            orientation_tolerance=original.config.orientation_tolerance,
            optimizer_collision_activation_distance=original.config.optimizer_collision_activation_distance,
            position_rank_weight=original.config.position_rank_weight,
            orientation_rank_weight=original.config.orientation_rank_weight,
            cspace_rank_weight=original.config.cspace_rank_weight,
            use_cuda_graph=True,max_batch_size=capacity,multi_env=capacity>1)
        self.solver = InverseKinematics(config,scene_collision_checker=native if collision else None)
        if self.solver.kinematics.kinematics_config.num_envs != capacity:
            raise RuntimeError('Native IK did not allocate per-environment robot geometry')

    def prepare(self,world,contexts):
        first=contexts[0]
        if self.collision:
            state=(world.model_dump(),[c.model_dump() for c in first.allowed_contacts])
            if self.scene_state is None or not same_contents(state,self.scene_state):
                self.scene_state=None
                contacts={}
                for c in first.allowed_contacts:contacts.setdefault(c.obstacle,set()).update(c.robot_links)
                partitions={}
                for obstacle in world.obstacles:
                    partitions.setdefault(tuple(sorted(contacts.get(obstacle.name,()))),set()).add(obstacle.name)
                params=self.solver.kinematics.kinematics_config
                groups=[]
                for links,names in partitions.items():
                    indices=torch.cat([params.get_sphere_index_from_link_name(link).reshape(-1) for link in links]) if links else torch.empty(0,dtype=torch.long,device='cuda')
                    groups.append((self.backend._scene(world,names=names),indices.long()))
                if self.contacts.configure(groups):self.solver.reset_cuda_graph()
                self.scene_state=copy.deepcopy(state)
        params=self.solver.kinematics.kinematics_config
        if self.backend.attachment_frame:
            params.reset_link_spheres(self.backend.attachment_frame)
            # Native variable-batch padding repeats the first row. Match it exactly.
            for i,context in enumerate(contexts+[first]*(self.capacity-len(contexts))):
                if context.attachment is not None:
                    spheres=self.backend._tensor(context.attachment.spheres)
                    spheres[:,3]+=self.backend.margin
                    params.update_link_spheres(self.backend.attachment_frame,spheres,config_idx=i)

    def solve(self,world,contexts,goals,seeds):
        self.prepare(world,contexts)
        poses=self.backend._tensor(goals)
        target=GoalToolPose.from_poses({self.backend.info.tool_frame:Pose(position=poses[:,:3],quaternion=poses[:,3:])},num_goalset=1)
        state=JointState.from_position(self.backend._tensor(seeds),joint_names=self.backend.info.joint_names)
        result=self.solver.solve_pose(target,current_state=state,return_seeds=1)
        count=len(goals)
        positions=self.backend._numpy(self.solver.get_active_js(result.js_solution).position).reshape(count,-1)
        success=result.success.detach().cpu().numpy().reshape(count).copy()
        success &= np.isfinite(positions).all(axis=1)
        # Independent exact endpoint collision check, with the same env-to-row map.
        if self.collision and success.any():
            selected=np.flatnonzero(success)
            envs=torch.as_tensor(selected,dtype=torch.int32,device='cuda')
            q=JointState.from_position(self.backend._tensor(positions[selected]).reshape(len(selected),1,-1),joint_names=self.backend.info.joint_names)
            with torch.no_grad():
                kin=self.solver.kinematics.compute_kinematics(q,idxs_env=envs)
                bad_world=self.backend._violations(self.checker.get_collision_constraint(kin,env_query_idx=envs),len(selected))
                bad_self=self.backend._violations(self.checker.get_self_collision(kin.robot_spheres),len(selected))
                success[selected] &= ~(bad_world | bad_self)
        positions[~success]=0
        errors=[]
        for value in (result.position_error,result.rotation_error):
            error=self.backend._numpy(value).reshape(count);error[~np.isfinite(error)]=-1;errors.append(error)
        return success,positions,*errors

    def clear_geometry_cache(self):
        self.scene_state=None
        if self.backend.attachment_frame:self.solver.kinematics.kinematics_config.reset_link_spheres(self.backend.attachment_frame)
        scenes=[self.checker.scene_model]+[scene for scene,_ in self.contacts.partitions]
        for scene in scenes:
            for env in range(self.capacity):scene.load_collision_model(Scene(),env_idx=env)
            if scene.data.meshes is not None:scene.data.meshes.mesh_ids.zero_()
        torch.cuda.synchronize()
        for scene in scenes:
            if scene.data.meshes is not None:scene.data.meshes.clear(clear_warp_cache=True)
        return sum(len(s.data.meshes.wp_cache) for s in scenes if s.data.meshes is not None)
