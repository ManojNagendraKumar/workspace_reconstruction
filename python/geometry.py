"""Camera geometry used across the pipeline: projection, back projection and rigid transforms."""
from __future__ import annotations

import numpy as np


def project(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Pinhole projection of (N, 3) camera frame points to (N, 2) pixels (u, v)."""
    z = pts_cam[:, 2:3]
    uv = (K[:2, :2] @ (pts_cam[:, :2] / z).T).T + K[:2, 2]
    return uv


def backproject(K: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Inverse of project(): pixels plus depth along the optical axis -> (N, 3) camera points."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def depth_to_points(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None):
    """Back project every valid depth pixel (optionally only inside mask). Returns points and pixel indices."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    valid = (depth > 0) & np.isfinite(depth)
    if mask is not None:
        valid &= mask
    pts = backproject(K, uu[valid].astype(float), vv[valid].astype(float), depth[valid].astype(float))
    return pts, (vv[valid], uu[valid])


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix, used to compare an estimated rotation with ground truth."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
