"""Lightweight, deterministic RGBD lifting shared by fake and real handlers."""

import numpy as np

from robot_tools.services.anyplace.contract import MaskedRGBD


def point_cloud(observation: MaskedRGBD) -> np.ndarray:
    valid = observation.mask & np.isfinite(observation.depth) & (observation.depth > 0)
    v, u = np.nonzero(valid)
    z = observation.depth[v, u].astype(np.float64)
    k = observation.intrinsics
    xyz = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    pose = observation.camera_pose
    return np.ascontiguousarray(xyz @ pose[:3, :3].T + pose[:3, 3])
