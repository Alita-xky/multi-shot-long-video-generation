"""Debug visualizations for trajectory-aligned pseudo geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spatial_memory.poses import camera_center
from spatial_memory.spatial_memory_bank import FrameRecord, SpatialMemoryBank


def save_pointcloud_png(bank: SpatialMemoryBank, path: Path | str, max_points: int = 20000) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bank.xyz is None:
        bank.build_point_cloud()
    xyz = bank.xyz
    rgb = np.clip(bank.rgb, 0, 1)
    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    centers = np.array([camera_center(rec.R, rec.t) for rec in bank.frames])

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(15, 7))
    # Top-down: X (right) vs Z (forward). Table at origin, camera orbits in X-Z.
    ax_top.scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=1)
    ax_top.plot(centers[:, 0], centers[:, 2], "-", color="lime", lw=1.5, alpha=0.9)
    ax_top.scatter([0], [0], c="red", marker="*", s=250, edgecolors="k")
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Z")
    ax_top.set_title("top-down (X-Z)")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, alpha=0.3)
    # Side: Z (forward) vs Y (up).
    ax_side.scatter(xyz[:, 2], xyz[:, 1], c=rgb, s=1)
    ax_side.plot(centers[:, 2], centers[:, 1], "-", color="lime", lw=1.5, alpha=0.9)
    ax_side.scatter([0], [0.95], c="red", marker="*", s=250, edgecolors="k")
    ax_side.set_xlabel("Z")
    ax_side.set_ylabel("Y")
    ax_side.set_title("side (Z-Y)")
    ax_side.grid(True, alpha=0.3)
    fig.suptitle("trajectory-aligned pseudo geometry")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_trajectory_png(bank: SpatialMemoryBank, path: Path | str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    xs, zs = [], []
    for rec in bank.frames:
        c = camera_center(rec.R, rec.t)
        xs.append(c[0])
        zs.append(c[2])
    ax.plot(xs, zs, "-o", color="tab:blue", markersize=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("nominal camera trajectory")
    ax.grid(True, alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_retrieved_png(items: list[tuple[FrameRecord, dict]], path: Path | str) -> Path:
    tiles = []
    for rec, meta in items:
        im = Image.fromarray(rec.rgb)
        im.thumbnail((320, 180))
        canvas = Image.new("RGB", (320, 200), (20, 20, 20))
        canvas.paste(im, ((320 - im.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 182), f"f{rec.frame_id} s={meta['score']:.2f}", fill=(230, 230, 230))
        tiles.append(canvas)
    if not tiles:
        Image.new("RGB", (320, 200), (0, 0, 0)).save(path)
        return Path(path)
    w = 320 * min(3, len(tiles))
    rows = (len(tiles) + 2) // 3
    out = Image.new("RGB", (w, rows * 200), (10, 10, 10))
    for i, t in enumerate(tiles):
        r, c = divmod(i, 3)
        out.paste(t, (c * 320, r * 200))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path
