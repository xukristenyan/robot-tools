import torch
import numpy as np
import trimesh
from pathlib import Path
from grasp_gen.robot import load_control_points_core, load_default_gripper_config


class GripperModel(object):
    def __init__(self, data_root_dir=None):
        if data_root_dir is None:
            data_root_dir = f'{Path(__file__).parent.parent.parent}/assets/robotiq'
        self.mesh = trimesh.load(f"{data_root_dir}/robotiq_85_collision.obj")

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh


def get_gripper_offset_bins():
    """Standardizes 10 bins for the 85mm stroke."""
    max_stroke = 0.085
    offset_bins = np.linspace(0.0, max_stroke, 11).tolist()

    # Carried over from 2f_140; only used during training, not inference.
    offset_bin_weights = [
        0.16652107, 0.21488856, 0.37031708, 0.55618503, 0.75124664,
        0.93943357, 1.07824539, 1.19423112, 1.55731375, 3.17161779
    ]
    return offset_bins, offset_bin_weights


def load_control_points() -> torch.Tensor:
    """
    Load the control points for the gripper, used for training.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T


def load_control_points_for_visualization():
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    mid_point = (control_points[0] + control_points[1]) / 2
    control_points = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0], mid_point, control_points[1], control_points[-1]
    ]
    return [control_points, ]
