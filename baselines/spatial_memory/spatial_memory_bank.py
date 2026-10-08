"""Trajectory-conditioned 3D spatial memory (GEN3C-lite).

Note:
H3 has no real camera extrinsics. camera_pose is nominal SE3 from
CameraCtrl waypoint interpolation. Depth is monocular (Depth Anything V2).
The point cloud is trajectory-aligned pseudo geometry, not COLMAP or NeRF.
The goal is long-horizon world consistency, not true 3D reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spatial_memory.poses import (
    HEIGHT,
    WIDTH,
    camera_center,
    camera_intrinsics,
    pose_distance,
)


KEYFRAME_STRIDE = 24
PIXEL_STRIDE = 8
Z_NEAR = 0.5
Z_FAR = 10.0
# Anchor: the look-at target (table) sits at the image center, so the
# center-patch disparity is pinned to the camera->target distance. This
# gives monocular depth a shared metric scale across frames.
ANCHOR_PATCH = 20
ANCHOR_TARGET = np.array([0.0, 0.95, 0.0])


def _metric_z(depth01: np.ndarray) -> np.ndarray:
    """near=high in [0,1] -> metric Z increasing with distance."""
    return Z_NEAR + (1.0 - np.clip(depth01, 0.0, 1.0)) * (Z_FAR - Z_NEAR)


def _anchor_scale(raw: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """k such that z = k / disparity puts the image-center patch at the
    camera->look-at distance. Per-frame anchor keeps scale consistent."""
    h, w = raw.shape[:2]
    a = ANCHOR_PATCH
    center = raw[h // 2 - a : h // 2 + a, w // 2 - a : w // 2 + a]
    disp = float(np.median(center))
    C = camera_center(R, t)
    dist = float(np.linalg.norm(ANCHOR_TARGET - C))
    return dist * max(disp, 1e-3)


def _metric_z_from_raw(raw: np.ndarray, k: float) -> np.ndarray:
    """Inverse-depth: z = k / disparity (disparity near=high)."""
    disp = np.asarray(raw, dtype=np.float32)
    z = k / np.clip(disp, 1e-3, None)
    return np.clip(z, Z_NEAR, Z_FAR)


@dataclass
class FrameRecord:
    frame_id: int
    image_path: str | None
    rgb: np.ndarray
    depth: np.ndarray
    feature: np.ndarray
    R: np.ndarray
    t: np.ndarray
    depth_raw: np.ndarray | None = None


@dataclass
class SpatialMemoryBank:
    width: int = WIDTH
    height: int = HEIGHT
    keyframe_stride: int = KEYFRAME_STRIDE
    pixel_stride: int = PIXEL_STRIDE
    frames: list[FrameRecord] = field(default_factory=list)
    xyz: np.ndarray | None = None
    rgb: np.ndarray | None = None
    feat: np.ndarray | None = None
    src: np.ndarray | None = None
    K: dict[str, float] = field(default_factory=camera_intrinsics)

    def add_frame(
        self,
        image,
        depth: np.ndarray,
        camera_pose: dict[str, Any],
        feature: np.ndarray,
        frame_id: int,
        image_path: str | None = None,
        depth_raw: np.ndarray | None = None,
    ) -> None:
        if isinstance(image, (str, Path)):
            image_path = str(image)
            rgb = np.asarray(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = np.asarray(image)
        self.frames.append(
            FrameRecord(
                frame_id=int(frame_id),
                image_path=image_path,
                rgb=rgb,
                depth=np.asarray(depth, dtype=np.float32),
                feature=np.asarray(feature, dtype=np.float32).reshape(-1),
                R=np.asarray(camera_pose["R"], dtype=np.float64),
                t=np.asarray(camera_pose["t"], dtype=np.float64).reshape(3),
                depth_raw=None if depth_raw is None else np.asarray(depth_raw, dtype=np.float32),
            )
        )
        self.xyz = None

    def build_point_cloud(self, out_npz: Path | None = None) -> dict[str, np.ndarray]:
        K = self.K
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        stride = self.pixel_stride
        xs_all, rgb_all, ft_all, src_all = [], [], [], []
        ids = [fr.frame_id for fr in self.frames]
        lo, hi = min(ids), max(ids)
        use_raw = all(fr.depth_raw is not None for fr in self.frames) and len(self.frames) > 0
        for rec in self.frames:
            if rec.frame_id % self.keyframe_stride != 0 and rec.frame_id not in (lo, hi):
                continue
            h, w = rec.rgb.shape[:2]
            if use_raw:
                k = _anchor_scale(rec.depth_raw, rec.R, rec.t)
                z = _metric_z_from_raw(rec.depth_raw, k)
                if z.shape[:2] != (h, w):
                    z = np.asarray(
                        Image.fromarray(z.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR),
                        dtype=np.float32,
                    )
            else:
                depth = rec.depth
                if depth.shape[:2] != (h, w):
                    depth = np.asarray(
                        Image.fromarray((np.clip(depth, 0, 1) * 255).astype(np.uint8), mode="L").resize(
                            (w, h), Image.BILINEAR
                        ),
                        dtype=np.float32,
                    ) / 255.0
                z = _metric_z(depth)
            vs = np.arange(0, h, stride)
            us = np.arange(0, w, stride)
            uu, vv = np.meshgrid(us, vs)
            zz = z[vv, uu]
            x = (uu - cx) / fx * zz
            y = (vv - cy) / fy * zz
            Xc = np.stack([x, y, zz], axis=-1).reshape(-1, 3)
            R_c2w = rec.R.T
            C = camera_center(rec.R, rec.t)
            Xw = (R_c2w @ Xc.T).T + C
            col = rec.rgb[vv, uu].reshape(-1, 3).astype(np.float32) / 255.0
            feat = np.repeat(rec.feature[None, :], Xw.shape[0], axis=0)
            xs_all.append(Xw.astype(np.float32))
            rgb_all.append(col)
            ft_all.append(feat.astype(np.float32))
            src_all.append(np.full((Xw.shape[0],), rec.frame_id, dtype=np.int32))
        if not xs_all:
            raise RuntimeError("no keyframes to back-project")
        self.xyz = np.concatenate(xs_all, axis=0)
        self.rgb = np.concatenate(rgb_all, axis=0)
        self.feat = np.concatenate(ft_all, axis=0)
        self.src = np.concatenate(src_all, axis=0)
        blob = {"xyz": self.xyz, "rgb": self.rgb, "feat": self.feat, "src": self.src}
        if out_npz is not None:
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **blob)
        return blob

    def _project(self, pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.xyz is None:
            self.build_point_cloud()
        R = np.asarray(pose["R"])
        t = np.asarray(pose["t"]).reshape(3)
        Xc = (R @ self.xyz.T).T + t
        z = Xc[:, 2]
        fx, fy, cx, cy = self.K["fx"], self.K["fy"], self.K["cx"], self.K["cy"]
        u = fx * Xc[:, 0] / np.clip(z, 1e-4, None) + cx
        v = fy * Xc[:, 1] / np.clip(z, 1e-4, None) + cy
        vis = (z > 0.05) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, vis

    def query(self, target_camera_pose: dict[str, Any], topk: int = 6) -> list[tuple[FrameRecord, dict[str, float]]]:
        """Retrieve source frames for a future target pose (not last-frame CLIP)."""
        if self.xyz is None:
            self.build_point_cloud()
        _, _, vis = self._project(target_camera_pose)
        vis_idx = np.where(vis)[0]
        n_vis = max(len(vis_idx), 1)
        tgt_feat = (
            self.feat[vis_idx].mean(axis=0)
            if len(vis_idx)
            else np.zeros_like(self.frames[0].feature)
        )
        tgt_feat = tgt_feat / (np.linalg.norm(tgt_feat) + 1e-8)
        scored = []
        for rec in self.frames:
            src_mask = self.src == rec.frame_id
            vis_src = int((src_mask & vis).sum()) if vis_idx.size else 0
            visibility = vis_src / n_vis
            prox = 1.0 / (1.0 + pose_distance({"R": rec.R, "t": rec.t}, target_camera_pose))
            f = rec.feature / (np.linalg.norm(rec.feature) + 1e-8)
            feat_sim = float(np.dot(f, tgt_feat))
            score = 0.5 * visibility + 0.3 * prox + 0.2 * max(feat_sim, 0.0)
            scored.append((rec, {"score": score, "visibility": visibility, "pose": prox, "feat": feat_sim}))
        scored.sort(key=lambda x: x[1]["score"], reverse=True)
        # Prefer distinct keyframes.
        picked = []
        used = set()
        for rec, s in scored:
            bin_id = rec.frame_id // self.keyframe_stride
            if bin_id in used:
                continue
            used.add(bin_id)
            picked.append((rec, s))
            if len(picked) >= topk:
                break
        return picked

    def query_pose_fov(self, target_camera_pose: dict[str, Any], topk: int = 6) -> list[tuple[FrameRecord, dict[str, float]]]:
        """2D context-memory retrieve: target pose/FOV only, no last-frame CLIP."""
        return self.query(target_camera_pose, topk=topk)

    def render_view(
        self,
        target_camera_pose: dict[str, Any],
        out_path: Path | str | None = None,
        splat: int = 3,
    ) -> np.ndarray:
        if self.xyz is None:
            self.build_point_cloud()
        u, v, vis = self._project(target_camera_pose)
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        zbuf = np.full((self.height, self.width), np.inf, dtype=np.float32)
        R = np.asarray(target_camera_pose["R"])
        t = np.asarray(target_camera_pose["t"]).reshape(3)
        z = ((R @ self.xyz.T).T + t)[:, 2]
        idx = np.where(vis)[0]
        order = idx[np.argsort(-z[idx])]  # far to near
        r = max(int(splat), 1) // 2
        for i in order:
            uu = int(u[i])
            vv = int(v[i])
            if z[i] >= zbuf[vv, uu]:
                continue
            zbuf[vv, uu] = z[i]
            y0, y1 = max(vv - r, 0), min(vv + r + 1, self.height)
            x0, x1 = max(uu - r, 0), min(uu + r + 1, self.width)
            img[y0:y1, x0:x1] = self.rgb[i]
        vis_img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(vis_img).save(out_path)
        return vis_img
