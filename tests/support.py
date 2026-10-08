"""Shared synthetic cameras and reconstructions for the scene-memory tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_memory.types import Camera, Reconstruction  # noqa: E402


POSE = {
    "R": np.eye(3),
    "t": np.zeros(3),
    "fx": 50.0,
    "fy": 50.0,
    "cx": 16.0,
    "cy": 12.0,
    "width": 32,
    "height": 24,
    "k": 0.0,
}


def camera(name, center, frame_index, path=None, fx=50.0, width=32, height=24, cx=16.0, cy=12.0):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    return Camera(
        image_name=name,
        image_path=None if path is None else str(path),
        registered=True,
        width=width,
        height=height,
        R_w2c=np.eye(3),
        t_w2c=-center,
        model="SIMPLE_PINHOLE",
        fx=fx,
        fy=fx,
        cx=cx,
        cy=cy,
        k=0.0,
        intrinsics_refined=False,
        principal_point_refined=False,
        frame_index=frame_index,
        reconstructor="test",
    )


def reconstruction(cameras, xyz, rgb, src_name, *, reproj=None, pair_inliers=None, pose_prior_count=0, input_image_count=None):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    src_name = np.asarray(src_name, dtype="U256").reshape(-1)
    src = np.arange(n, dtype=np.int32)
    features = np.concatenate([rgb, src.astype(np.float32).reshape(-1, 1)], axis=1)
    if reproj is None:
        reproj = np.zeros((n,), np.float32)
    return Reconstruction(
        reconstructor="test",
        cameras=list(cameras),
        xyz=xyz,
        rgb=rgb,
        features=features,
        src=src,
        src_name=src_name,
        reproj=np.asarray(reproj, np.float32).reshape(-1),
        mean_reproj_error=float(np.mean(reproj)) if n else None,
        median_reproj_error=float(np.median(reproj)) if n else None,
        pair_inliers=dict(pair_inliers or {}),
        pose_prior_count=pose_prior_count,
        input_image_count=len(cameras) if input_image_count is None else input_image_count,
    )
