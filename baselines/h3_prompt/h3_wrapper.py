"""H3 inference-time wrapper: memory refs + optional trajectory prompt. No weight changes."""

from __future__ import annotations

from pathlib import Path

from camera_trajectory import CameraTrajectory, to_prompt_block, vague_camera_line, trajectory_for
from memory_bank import MemoryFrame

ROOM_ANCHORS = (
    "The scene is one indoor room. The room geometry stays the same for the whole clip. "
    "Three fixed spatial anchors define the room: a table in the center; a sofa on the "
    "left side of the room when the camera faces the front of the table; a door on the "
    "right side of the room when the camera faces the front of the table. All objects "
    "stay static. There are no people."
)


def build_clip1_prompt(*, use_trajectory: bool) -> str:
    traj = trajectory_for("clip_01")
    camera = to_prompt_block(traj) if use_trajectory else vague_camera_line()
    if use_trajectory:
        summary = (
            "[reference generation] A continuous 10.125-second one-take. "
            "Follow the camera trajectory exactly: front of the table to the left side."
        )
        detail = (
            "[Shot 1] Medium-wide eye-level opening at the front of the central table. "
            "Sofa visible room-left, door visible room-right. Execute the camera "
            "trajectory keyframes. Keep the table as the orbit center. Only the camera "
            "moves. End at the left side of the table."
        )
    else:
        summary = (
            "[reference generation] A continuous 10.125-second one-take inside the same room. "
            "The camera moves around the table."
        )
        detail = (
            "[Shot 1] Medium-wide eye-level view of a central table in a living room. "
            "The camera moves around the table. Only the camera moves. No people. "
            "No object rearrangement. No cuts."
        )
    return "\n".join(
        [
            "subject_definitions:",
            ROOM_ANCHORS,
            "",
            "summary:",
            summary,
            "",
            "retention_analysis:",
            "Room layout (throughout [Shot 1]): fully_preserved - table, sofa, and door keep physical positions.",
            "",
            "detailed_description:",
            detail,
            "",
            camera,
            "",
            "overall_soundscape:",
            "Quiet indoor room ambience.",
            "",
            "non_diegetic_music:",
            "N/A",
            "",
        ]
    )


def build_continuation_prompt(
    clip_id: str,
    memory_frames: list[MemoryFrame],
    *,
    use_trajectory: bool,
    ablation: str,
) -> str:
    traj = trajectory_for(clip_id)
    n_mem = len(memory_frames)
    camera = to_prompt_block(traj) if use_trajectory else vague_camera_line()
    if use_trajectory:
        summary_motion = traj.path_summary
        end = f"End at: {traj.end_view}."
        motion = f"execute the camera trajectory ({traj.path_summary})"
    else:
        summary_motion = "the camera moves around the table"
        end = "The camera moves around the table."
        motion = "the camera moves around the table"

    lines = [
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.",
        "",
        "subject_definitions:",
        "<Picture 1> is the exact final frame of the previous clip and the exact "
        f"first frame of this clip. Opening view: {traj.start_view}.",
        ROOM_ANCHORS,
    ]
    for i, fr in enumerate(memory_frames, start=2):
        lines.append(
            f"<Picture {i}> is a retrieved spatial-memory frame from {fr.clip_id} "
            f"at frame {fr.frame_idx} ({fr.t_sec:.2f}s). Layout reference only "
            "(table/sofa/door geometry). It must not replace <Picture 1> and must "
            "not jump the camera to that past viewpoint."
        )
    lines += [
        "",
        "summary:",
        f"[continuation generation] A continuous 10.125-second one-take from <Picture 1>. "
        f"{summary_motion}. No reset of the room.",
        "",
        "retention_analysis:",
        "<Picture 1> ([Shot 1] first frame): fully_preserved - opening matches the previous last frame.",
        "Room layout (throughout [Shot 1]): fully_preserved - table, sofa, door stay fixed; only the camera moves.",
    ]
    if n_mem:
        pics = ", ".join(f"<Picture {i}>" for i in range(2, n_mem + 2))
        lines.append(
            f"{pics} (spatial memory, ablation={ablation}): reference - layout cues only."
        )
    lines += [
        "",
        "detailed_description:",
        f"[Shot 1] First frame is exactly <Picture 1>. From 00:00.000 to 00:10.125, "
        f"{motion}, keeping the table in frame as the orbit center. Only the camera "
        f"moves. No people, no object motion, no cuts, freeze, or teleport. {end}",
        "",
        camera,
        "",
        "overall_soundscape:",
        "Quiet indoor room ambience.",
        "",
        "non_diegetic_music:",
        "N/A",
        "",
    ]
    return "\n".join(lines)


def reference_flags(last_frame: Path, memory_frames: list[MemoryFrame]) -> list[str]:
    flags = [f"image:{last_frame}"]
    for fr in memory_frames:
        flags.append(f"image:{fr.path}")
    if len(flags) > 9:
        raise ValueError(f"Ref2VA image cap is 9, got {len(flags)}")
    return flags
