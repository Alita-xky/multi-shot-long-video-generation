"""Nominal SE3 camera poses from CameraCtrl waypoints.

Note:
H3 has no real camera extrinsics. These poses are interpolated from
waypoint look-at controls (nominal SE3), not estimated from video.
Together with monocular depth they produce trajectory-aligned pseudo
geometry, not COLMAP or NeRF reconstruction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


WIDTH = 960
HEIGHT = 544
HFOV_DEG = 60.0
FPS = 24.0
NUM_FRAMES = 243


def camera_intrinsics(width: int = WIDTH, height: int = HEIGHT, hfov_deg: float = HFOV_DEG) -> dict[str, float]:
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) * 0.5)
    fy = fx
    return {"fx": fx, "fy": fy, "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0, "width": width, "height": height}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def look_at_w2c(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera: X_c = R @ X_w + t, OpenCV (+Z forward, Y down)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = _unit(up)
    forward = _unit(target - eye)
    right = np.cross(up, forward)
    if float(np.linalg.norm(right)) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(up, forward)
    right = _unit(right)
    down = _unit(np.cross(forward, right))
    r_c2w = np.stack([right, down, forward], axis=1)
    r_w2c = r_c2w.T
    t = -r_w2c @ eye
    return r_w2c.astype(np.float64), t.astype(np.float64)


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-np.asarray(R).T @ np.asarray(t).reshape(3)).astype(np.float64)


def _slerp_R(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    """Rotation interpolation via quaternion slerp."""
    q0 = _rot_to_quat(R0)
    q1 = _rot_to_quat(R1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        q = q / (np.linalg.norm(q) + 1e-8)
        return _quat_to_rot(q)
    theta = math.acos(dot)
    q = (math.sin((1 - a) * theta) * q0 + math.sin(a * theta) * q1) / math.sin(theta)
    return _quat_to_rot(q)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-8)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def interpolate_clip_poses(
    waypoints: list[dict[str, Any]],
    num_frames: int = NUM_FRAMES,
    fps: float = FPS,
) -> list[dict[str, Any]]:
    wps = sorted(waypoints, key=lambda w: int(w["frame"]))
    frames: list[dict[str, Any]] = []
    eyes = [np.asarray(w["eye"], dtype=np.float64).reshape(3) for w in wps]
    targets = [np.asarray(w["look_at"], dtype=np.float64).reshape(3) for w in wps]
    ups = [np.asarray(w.get("up", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3) for w in wps]
    ids = [int(w["frame"]) for w in wps]
    for f in range(num_frames):
        if f <= ids[0]:
            eye, target, up = eyes[0], targets[0], ups[0]
        elif f >= ids[-1]:
            eye, target, up = eyes[-1], targets[-1], ups[-1]
        else:
            i = 0
            while i + 1 < len(ids) and ids[i + 1] < f:
                i += 1
            a = (f - ids[i]) / max(ids[i + 1] - ids[i], 1)
            eye = (1.0 - a) * eyes[i] + a * eyes[i + 1]
            target = (1.0 - a) * targets[i] + a * targets[i + 1]
            up = (1.0 - a) * ups[i] + a * ups[i + 1]
        R, t = look_at_w2c(eye, target, up)
        frames.append(
            {
                "frame_id": f,
                "timestamp": f / fps,
                "R": R,
                "t": t,
            }
        )
    return frames


def load_scene(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def poses_for_clip(scene: dict[str, Any], clip_id: str, num_frames: int = NUM_FRAMES) -> list[dict[str, Any]]:
    return interpolate_clip_poses(scene["clips"][clip_id]["waypoints"], num_frames=num_frames)


def midpoint_pose(poses: list[dict[str, Any]]) -> dict[str, Any]:
    return poses[len(poses) // 2]


def pose_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ca = camera_center(a["R"], a["t"])
    cb = camera_center(b["R"], b["t"])
    trans = float(np.linalg.norm(ca - cb))
    rel = a["R"] @ b["R"].T
    ang = math.acos(float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)))
    return trans + 0.5 * ang
