#!/usr/bin/env python3
"""Reconstruct the locally rigid orbit prefix. Nominal poses are not an input.

Frames 0, 24, ..., 240 of C_trajectory/final.mp4. Frame 728 is not in this set.
A failed registration is reported; it is not replaced with waypoint geometry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scene_memory.backends.colmap_backend import ColmapBackend  # noqa: E402
from scene_memory.conditioning import ablation, write_memory_view  # noqa: E402
from scene_memory.evaluate import object_persistence, return_report, spatial_report  # noqa: E402
from scene_memory.memory import SceneMemory  # noqa: E402


FRAMES = list(range(0, 241, 24))
OUT = ROOT / "runs" / "orbit_prefix"


def extract_frames(video: Path, image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in FRAMES:
        dest = image_dir / f"frame_{frame:06d}.png"
        paths.append(dest)
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        subprocess.check_call(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"select=eq(n\\,{frame})",
                "-frames:v",
                "1",
                str(dest),
            ]
        )
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError(f"failed to extract frame {frame}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="COLMAP the orbit prefix. Nominal poses are not an input.")
    parser.add_argument("--video", type=Path, required=True)
    video = parser.parse_args().video
    if not video.is_file():
        raise SystemExit(f"missing video {video}")
    paths = extract_frames(video, OUT / "images")
    memory = SceneMemory()
    memory.build(paths, ColmapBackend(), work_root=OUT / "colmap")
    memory.save(OUT / "memory")
    spatial = spatial_report(memory)
    returning = return_report(memory, "frame_000000.png", "frame_000728.png")
    along_arc = return_report(memory, "frame_000000.png", "frame_000240.png")
    group = memory.groups["default"]
    source = next(cam for cam in group.registered() if cam.frame_index == 0)
    R, t = group.world_camera(source)
    pose = {
        "R": R,
        "t": t,
        "fx": source.fx,
        "fy": source.fy,
        "cx": source.cx,
        "cy": source.cy,
        "width": source.width,
        "height": source.height,
        "k": source.k,
    }
    written = write_memory_view(memory, pose, OUT / "memory_view")
    spec = ablation(
        "A continuous shot that stays in the same room.",
        paths[0],
        written["rgb"],
        written["mask"],
    )
    (OUT / "conditioning.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    report = {
        "video": str(video),
        "frames": FRAMES,
        "nominal_poses_used_in_mapping": False,
        "spatial": spatial,
        "return_frame_728": returning,
        "arc_0_to_240": along_arc,
        "objects": object_persistence(memory, None),
        "conditioning_changes_weights": spec["proposed"]["changes_weights"],
        "conditioning_references": spec["proposed"]["references"],
        "memory_view_valid_pixels": written["valid_pixels"],
    }
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: spatial[k] for k in ("registered", "registration_ratio", "num_points", "mean_reproj_error", "pose_prior_count")}))
    errors = []
    if spatial["registration_ratio"] < 0.8:
        errors.append(f"registration ratio {spatial['registration_ratio']} < 0.8")
    if spatial["mean_reproj_error"] is None or spatial["mean_reproj_error"] >= 1.0:
        errors.append(f"mean reprojection {spatial['mean_reproj_error']} is not sub-pixel")
    if spatial["num_points"] <= 0:
        errors.append("point cloud is empty")
    if spatial["pose_prior_count"] != 0:
        errors.append("pose priors were present")
    if returning["closure_supported"] or returning["registered_b"]:
        errors.append("frame 728 must stay unregistered in this prefix")
    if spec["proposed"]["changes_weights"]:
        errors.append("conditioning claims a weight change")
    if "memory_mask" in " ".join(spec["proposed"]["argv"]):
        errors.append("mask was passed as a generator reference")
    if written["valid_pixels"] <= 0:
        errors.append("source-camera memory view is empty")
    holes = ~np.asarray(Image.open(written["mask"]).convert("L")).astype(bool)
    rgb = np.asarray(Image.open(written["rgb"]).convert("RGB"))
    if holes.any() and int(rgb[holes].max()) != 0:
        errors.append("memory view filled a hole")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
