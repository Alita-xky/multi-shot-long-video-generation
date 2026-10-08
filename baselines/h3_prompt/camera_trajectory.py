"""Inference-time camera trajectory condition (CameraCtrl idea, no encoder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Waypoint:
    frame: int
    motion: str
    view: str


@dataclass
class CameraTrajectory:
    """Explicit per-clip orbit around a central table.

    CameraCtrl trains a pose encoder over Plücker embeddings. H3 has no such
    encoder here, so the same information is serialized as timestamped view
    waypoints and injected into the prompt.
    """

    clip_id: str
    duration_sec: float
    waypoints: tuple[Waypoint, ...]
    start_view: str
    end_view: str
    path_summary: str


TABLE_ORBIT = {
    "clip_01": CameraTrajectory(
        clip_id="clip_01",
        duration_sec=10.125,
        start_view="front of the central table, sofa on room-left, door on room-right",
        end_view="left side of the table; table remains the orbit center",
        path_summary="clockwise quarter-orbit: front → left side",
        waypoints=(
            Waypoint(0, "hold", "front view of the table"),
            Waypoint(60, "move left / clockwise", "front-left three-quarter view"),
            Waypoint(120, "continue clockwise", "left-front of the table"),
            Waypoint(180, "continue clockwise", "true left side of the table"),
            Waypoint(242, "arrive and hold", "left side of the table, ready to continue"),
        ),
    ),
    "clip_02": CameraTrajectory(
        clip_id="clip_02",
        duration_sec=10.125,
        start_view="left side of the central table (same pose as previous last frame)",
        end_view="behind the table",
        path_summary="clockwise quarter-orbit: left side → behind the table",
        waypoints=(
            Waypoint(0, "hold", "left-side view, matching the previous last frame"),
            Waypoint(60, "move clockwise", "left-back three-quarter view"),
            Waypoint(120, "continue clockwise", "back-left of the table"),
            Waypoint(180, "continue clockwise", "directly behind the table"),
            Waypoint(242, "arrive and hold", "behind the table, ready to continue"),
        ),
    ),
    "clip_03": CameraTrajectory(
        clip_id="clip_03",
        duration_sec=10.125,
        start_view="behind the central table (same pose as previous last frame)",
        end_view="original front view: sofa room-left, door room-right",
        path_summary="clockwise half-orbit: behind → original front viewpoint",
        waypoints=(
            Waypoint(0, "hold", "behind the table, matching the previous last frame"),
            Waypoint(80, "move clockwise", "back-right of the table"),
            Waypoint(140, "continue clockwise", "right side of the table"),
            Waypoint(200, "continue clockwise toward original front", "front-right three-quarter view"),
            Waypoint(242, "arrive and hold", "original front view, sofa left, door right"),
        ),
    ),
}


def trajectory_for(clip_id: str) -> CameraTrajectory:
    return TABLE_ORBIT[clip_id]


def to_prompt_block(traj: CameraTrajectory) -> str:
    lines = [
        "Camera trajectory (must be followed; do not reverse, reset, or return early):",
        f"Start: {traj.start_view}",
        f"Path: {traj.path_summary}",
        f"End: {traj.end_view}",
        "Keyframes (24 fps, frame index is exact):",
    ]
    for wp in traj.waypoints:
        t = wp.frame / 24.0
        lines.append(f"- frame {wp.frame} ({t:05.2f}s): {wp.motion}; {wp.view}")
    lines.append(
        "The camera only translates/orbits. It never teleports, never cuts, "
        "and never returns to an earlier waypoint before the later ones."
    )
    return "\n".join(lines)


def vague_camera_line() -> str:
    return "The camera moves around the table."
