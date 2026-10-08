"""Z-buffer point render. Pixels with no point stay empty. Nothing is inpainted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scene_memory.geometry import project


@dataclass
class MemoryView:
    rgb: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    hit_frame_indices: list[int]
    hit_image_names: list[str]
    valid_pixels: int

    @property
    def empty(self) -> bool:
        return self.valid_pixels == 0


def render_points(
    xyz,
    rgb,
    src,
    src_name,
    pose: dict,
    *,
    z_min: float = 0.05,
) -> MemoryView:
    """Project points with the given w2c pose. One pixel per point, nearer wins.

    ``pose`` must carry width, height, fx, fy, cx, cy. Missing intrinsics raise
    rather than assuming a field of view.
    """
    width, height, fx, fy, cx, cy, k = _intrinsics(pose)
    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) == 0:
        return MemoryView(rgb_out, mask, depth, [], [], 0)
    colors = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if colors.size and float(colors.max()) > 1.0:
        colors = colors / 255.0
    src = np.asarray(src, dtype=np.int32).reshape(-1)
    names = np.asarray(src_name).astype(str).reshape(-1)
    u, v, z = project(xyz, pose["R"], pose["t"], fx, fy, cx, cy, k)
    # Points behind the camera project to non-finite pixels. Round only the ones near the image.
    near = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > z_min)
    near &= (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    ui = np.zeros(len(u), dtype=np.int32)
    vi = np.zeros(len(v), dtype=np.int32)
    ui[near] = np.rint(u[near]).astype(np.int32)
    vi[near] = np.rint(v[near]).astype(np.int32)
    vis = near & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(vis):
        return MemoryView(rgb_out, mask, depth, [], [], 0)
    ui, vi, z = ui[vis], vi[vis], z[vis]
    colors, src, names = colors[vis], src[vis], names[vis]
    order = np.argsort(z, kind="mergesort")
    ui, vi, z = ui[order], vi[order], z[order]
    colors, src, names = colors[order], src[order], names[order]
    lin = vi.astype(np.int64) * width + ui.astype(np.int64)
    _, first = np.unique(lin, return_index=True)
    ui, vi, z = ui[first], vi[first], z[first]
    colors, src, names = colors[first], src[first], names[first]
    rgb_out[vi, ui] = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    mask[vi, ui] = True
    depth[vi, ui] = z.astype(np.float32)
    hit_frames = sorted({int(v) for v in src.tolist()})
    hit_names = sorted({str(n) for n in names.tolist() if str(n)})
    return MemoryView(rgb_out, mask, depth, hit_frames, hit_names, int(mask.sum()))


def _intrinsics(pose: dict):
    missing = [key for key in ("width", "height", "fx", "fy", "cx", "cy", "R", "t") if key not in pose or pose[key] is None]
    if missing:
        raise ValueError(
            "query camera is missing "
            + ", ".join(missing)
            + "; refusing to guess image size or HFOV"
        )
    return (
        int(pose["width"]),
        int(pose["height"]),
        float(pose["fx"]),
        float(pose["fy"]),
        float(pose["cx"]),
        float(pose["cy"]),
        float(pose.get("k") or 0.0),
    )
