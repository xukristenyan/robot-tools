"""Per-link world contacts for the pinned cuRobo planner, including gradients.

Partition obstacles by their exempt links. Query every partition with only those
links masked, then sum its native cuRobo costs and analytic gradients. Robot
self-collision geometry is never modified. Four reusable partitions cover the
usual normal/base/gripper/attached-body contexts; extra partitions grow on demand.
"""
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo.scene import Scene


class _ContactCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spheres, owner, buffer, weight, activation, env_idx,
                return_loss, dt, speed_metric, sum_distance, sphere_mask):
        buffer.zero_()
        parts = owner.buffers(buffer, spheres.shape)
        for (scene, ignored), part in zip(owner.partitions, parts):
            query = spheres.clone()
            query[..., 3] = torch.where(ignored, -100., spheres[..., 3])
            if dt is None:
                scene.checker.get_sphere_distance(scene.data, query, part, weight,
                    activation, env_query_idx=env_idx, return_loss=False,
                    sphere_collision_mask=sphere_mask)
            else:
                scene.checker.get_swept_sphere_distance(scene.data, query, part, weight,
                    activation, dt, enable_speed_metric=speed_metric,
                    env_query_idx=env_idx, return_loss=False,
                    sphere_collision_mask=sphere_mask)
            buffer.distance.add_(part.distance)
            buffer.gradient.add_(part.gradient)
        ctx.return_loss = return_loss
        ctx.save_for_backward(buffer.gradient)
        return buffer.distance.sum(dim=-1, keepdim=True) if sum_distance else buffer.distance

    @staticmethod
    def backward(ctx, grad_output):
        gradient, = ctx.saved_tensors
        if ctx.return_loss:
            gradient = gradient * grad_output.unsqueeze(-1)
        return gradient, None, None, None, None, None, None, None, None, None, None


class ContactCollisionChecker:
    def __init__(self, native, device_cfg, num_spheres, *, num_envs=1):
        self.native, self.device_cfg, self.num_spheres = native, device_cfg, num_spheres
        self.num_envs = num_envs
        self.partitions, self._buffers = [], {}
        self.grow(4)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def grow(self, count):
        changed = count > len(self.partitions)
        while len(self.partitions) < count:
            scene = SceneCollision.from_config(SceneCollisionCfg(device_cfg=self.device_cfg,
                scene_model=Scene(), num_envs=self.num_envs, cache={'mesh':128, 'cuboid':128}))
            mask = torch.zeros(self.num_spheres, dtype=torch.bool, device=self.device_cfg.device)
            self.partitions.append((scene, mask))
        if changed:
            self._buffers.clear()
        return changed

    def configure(self, groups):
        """Update existing GPU storage in place so captured graphs see new contacts."""
        grew = self.grow(len(groups))
        for index, (scene, ignored) in enumerate(self.partitions):
            ignored.zero_()
            if index < len(groups):
                model, indices = groups[index]
                for env in range(self.num_envs):
                    scene.load_collision_model(model, env_idx=env)
                ignored[indices] = True
            else:
                for env in range(self.num_envs):
                    scene.load_collision_model(Scene(), env_idx=env)
        return grew

    def buffers(self, output, shape):
        key = id(output)
        record = self._buffers.get(key)
        if record is None or record[1] != tuple(shape):
            parts = [CollisionBuffer.from_shape(shape, self.device_cfg) for _ in self.partitions]
            self._buffers[key] = (output, tuple(shape), parts)
        return self._buffers[key][2]

    def get_sphere_distance(self, scene, query_sphere, collision_buffer, weight,
                            activation_distance, env_query_idx=None, return_loss=False,
                            sum_distance=False, sphere_collision_mask=None):
        return _ContactCost.apply(query_sphere, self, collision_buffer, weight,
            activation_distance, env_query_idx, return_loss, None, False,
            sum_distance, sphere_collision_mask)

    get_sphere_collision = get_sphere_distance

    def get_swept_sphere_distance(self, scene, query_sphere, collision_buffer, weight,
            activation_distance, trajectory_dt, enable_speed_metric=False,
            env_query_idx=None, return_loss=False, sum_distance=False, sphere_collision_mask=None):
        return _ContactCost.apply(query_sphere, self, collision_buffer, weight,
            activation_distance, env_query_idx, return_loss, trajectory_dt, enable_speed_metric,
            sum_distance, sphere_collision_mask)

    get_swept_sphere_collision = get_swept_sphere_distance
