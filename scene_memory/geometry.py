"""Pinhole and SIMPLE_RADIAL geometry.

Camera convention matches COLMAP: X_c = R @ X_w + t, with +Z forward.
Camera center is C = -R.T @ t. Focal length is never inferred from an HFOV guess.
"""

from __future__ import annotations

import numpy as np


def qvec_to_R(q) -> np.ndarray:
    """COLMAP qw, qx, qy, qz to a world-to-camera rotation."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def as_R(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return R


def as_t(t) -> np.ndarray:
    return np.asarray(t, dtype=np.float64).reshape(3)


def camera_center(R, t) -> np.ndarray:
    R = as_R(R)
    t = as_t(t)
    return (-R.T @ t).astype(np.float64)


def project(Xw, R, t, fx, fy, cx, cy, k: float = 0.0):
    """Project world points. k is the SIMPLE_RADIAL term; 0 is a pinhole."""
    Xw = np.asarray(Xw, dtype=np.float64).reshape(-1, 3)
    Xc = (as_R(R) @ Xw.T).T + as_t(t)
    z = Xc[:, 2]
    safe = np.clip(z, 1e-8, None)
    x = Xc[:, 0] / safe
    y = Xc[:, 1] / safe
    r2 = x * x + y * y
    scale = 1.0 + float(k) * r2
    u = float(fx) * x * scale + float(cx)
    v = float(fy) * y * scale + float(cy)
    return u, v, z


def unproject(u, v, z, R, t, fx, fy, cx, cy, k: float = 0.0) -> np.ndarray:
    """Inverse of project. Radial distortion is inverted by fixed-point iteration."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    xd = (u - float(cx)) / float(fx)
    yd = (v - float(cy)) / float(fy)
    xu = xd.copy()
    yu = yd.copy()
    if abs(float(k)) > 0:
        for _ in range(8):
            r2 = xu * xu + yu * yu
            scale = 1.0 + float(k) * r2
            xu = xd / scale
            yu = yd / scale
    Xc = np.stack([xu * z, yu * z, z], axis=1)
    R = as_R(R)
    Xw = (R.T @ (Xc - as_t(t)).T).T
    return Xw


def rot_deg(Ra, Rb) -> float:
    Rd = as_R(Ra) @ as_R(Rb).T
    c = float(np.clip((np.trace(Rd) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def umeyama(src, dst):
    """dst ~= s * R @ src + t. src and dst are Nx3."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(src) < 3:
        raise ValueError("umeyama needs at least 3 points")
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = float(np.mean(np.sum(X * X, axis=1)))
    s = float(np.trace(np.diag(S) @ D) / max(var, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


def sim3_points(xyz, s, R, t) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return s * (as_R(R) @ xyz.T).T + as_t(t)


def sim3_camera(R_w2c, t_w2c, s, R_align, t_align):
    """Map a w2c camera through X_dst = s * R_align @ X_src + t_align."""
    C_src = camera_center(R_w2c, t_w2c)
    C_dst = s * (as_R(R_align) @ C_src) + as_t(t_align)
    R_c2w_dst = as_R(R_align) @ as_R(R_w2c).T
    R_dst = R_c2w_dst.T
    t_dst = -R_dst @ C_dst
    return R_dst, t_dst


def alignment_rmse(src, dst, s, R, t) -> float:
    pred = sim3_points(src, s, R, t)
    err = pred - np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def rigid_from_rt(R, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = as_R(R)
    T[:3, 3] = as_t(t)
    return T


def invert_rigid(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out
