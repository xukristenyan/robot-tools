"""Optional NumPy-only conversion of observed points to occupied voxel meshes."""

from __future__ import annotations

import numpy as np


def occupied_voxel_mesh(points: np.ndarray, voxel_size: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Build the exterior triangles of occupied cells in the point-cloud frame.

    Does not infer surfaces, fill unseen space, or create an ESDF. The caller
    owns filtering, calibration and the policy for unobserved regions.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 1_000_000:
        raise ValueError("points must have shape (1..1000000, 3)")
    if not np.issubdtype(points.dtype, np.floating) or not np.isfinite(points).all():
        raise ValueError("points must contain finite floating-point metres")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive metres")
    scaled = np.floor(points.astype(np.float64) / voxel_size)
    if np.max(np.abs(scaled)) > 1e9:
        raise ValueError("point coordinates exceed voxel indexing range")
    cells = np.unique(scaled.astype(np.int64), axis=0)
    if len(cells) > 10000:
        raise ValueError("at most 10000 occupied voxels; increase voxel_size or crop the cloud")
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    sides = [
        ((0, 0, -1), [0, 3, 2, 1]),
        ((0, 0, 1), [4, 5, 6, 7]),
        ((0, -1, 0), [0, 1, 5, 4]),
        ((1, 0, 0), [1, 2, 6, 5]),
        ((0, 1, 0), [2, 3, 7, 6]),
        ((-1, 0, 0), [3, 0, 4, 7]),
    ]
    vertices, remap = np.unique((cells[:, None] + corners).reshape(-1, 3), axis=0, return_inverse=True)
    remap = remap.reshape(-1, 8)
    occupied = {tuple(cell) for cell in cells}
    faces = []
    for cell, indices in zip(cells, remap):
        for offset, face in sides:
            if tuple(cell + offset) not in occupied:
                a, b, c, d = indices[face]
                faces.extend(((a, b, c), (a, c, d)))
    return (vertices * voxel_size).astype(np.float32), np.asarray(faces, dtype=np.int32)
