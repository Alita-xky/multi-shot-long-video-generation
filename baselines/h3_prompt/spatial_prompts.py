"""Scene-aware H3 prompts for the spatial-memory experiment. Does not edit h3_wrapper.py."""

from __future__ import annotations

from pathlib import Path

from spatial_memory.spatial_memory_bank import FrameRecord


def waypoint_block(clip: dict) -> str:
    lines = [
        "Camera trajectory (must be followed; do not reverse, reset, or return early):",
        f"Start: {clip['start_view']}",
        f"Path: {clip['path_summary']}",
        f"End: {clip['end_view']}",
        "Keyframes (24 fps):",
    ]
    for wp in clip["waypoints"]:
        t = int(wp["frame"]) / 24.0
        lines.append(f"- frame {wp['frame']} ({t:05.2f}s): {wp['motion']}; {wp['view']}")
    lines.append("The camera never teleports, never cuts, and never returns early.")
    return "\n".join(lines)


def clip1_prompt(scene: dict, clip: dict, *, use_trajectory: bool) -> str:
    camera = waypoint_block(clip) if use_trajectory else clip.get("vague", "The camera moves.")
    if use_trajectory:
        summary = f"[reference generation] 10.125s one-take. Follow the trajectory: {clip['path_summary']}."
        detail = (
            f"[Shot 1] Open at {clip['start_view']}. Execute the camera keyframes. "
            f"End at {clip['end_view']}. Only the camera moves."
        )
    else:
        summary = "[reference generation] 10.125s one-take. " + clip.get("vague", "The camera moves.")
        detail = f"[Shot 1] {clip.get('vague', 'The camera moves.')} Only the camera moves. No people. No cuts."
    return "\n".join(
        [
            "subject_definitions:",
            scene["anchors"],
            "",
            "summary:",
            summary,
            "",
            "retention_analysis:",
            "Layout (throughout [Shot 1]): fully_preserved.",
            "",
            "detailed_description:",
            detail,
            "",
            camera,
            "",
            "overall_soundscape:",
            "Quiet ambience.",
            "",
            "non_diegetic_music:",
            "N/A",
            "",
        ]
    )


def continuation_prompt(
    scene: dict,
    clip: dict,
    retrieved: list[FrameRecord],
    *,
    has_render: bool,
    use_trajectory: bool,
    ablation: str,
) -> str:
    camera = waypoint_block(clip) if use_trajectory else clip.get("vague", "The camera moves.")
    n_mem = len(retrieved)
    render_idx = 2 + n_mem if has_render else None
    lines = [
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.",
        "",
        "subject_definitions:",
        f"<Picture 1> is the exact final frame of the previous clip and the exact first frame of this clip. Opening: {clip['start_view']}.",
        scene["anchors"],
    ]
    for i, rec in enumerate(retrieved, start=2):
        lines.append(
            f"<Picture {i}> is a retrieved spatial-memory frame (src frame {rec.frame_id}). "
            "Layout reference only. It must not replace <Picture 1> and must not jump the camera."
        )
    if has_render and render_idx is not None:
        lines.append(
            f"<Picture {render_idx}> is a 3D-memory projected layout constraint rendered from the "
            "next-clip target camera pose (GEN3C-lite point projection). It is not a real photograph. "
            "Do not overwrite <Picture 1>. Do not teleport the camera to this rendered viewpoint."
        )
    lines += [
        "",
        "summary:",
        f"[continuation generation] 10.125s one-take from <Picture 1>. {clip['path_summary'] if use_trajectory else clip.get('vague')}.",
        "",
        "retention_analysis:",
        "<Picture 1> ([Shot 1] first frame): fully_preserved.",
        "Layout: fully_preserved.",
    ]
    if n_mem:
        pics = ", ".join(f"<Picture {i}>" for i in range(2, 2 + n_mem))
        lines.append(f"{pics} (ablation={ablation}): history layout cues only.")
    if has_render and render_idx is not None:
        lines.append(
            f"<Picture {render_idx}> (spatial render): reference - projected layout constraint only."
        )
    lines += [
        "",
        "detailed_description:",
        f"[Shot 1] First frame is exactly <Picture 1>. Then {clip['path_summary'] if use_trajectory else clip.get('vague')}. "
        f"End at {clip['end_view'] if use_trajectory else 'a continued viewpoint'}. Only the camera moves. No cuts.",
        "",
        camera,
        "",
        "overall_soundscape:",
        "Quiet ambience.",
        "",
        "non_diegetic_music:",
        "N/A",
        "",
    ]
    return "\n".join(lines)


def reference_flags(
    last_frame: Path,
    retrieved_paths: list[Path],
    render_path: Path | None,
    max_images: int = 9,
) -> list[str]:
    flags = [f"image:{last_frame}"]
    for p in retrieved_paths:
        flags.append(f"image:{p}")
    if render_path is not None:
        flags.append(f"image:{render_path}")
    if len(flags) > max_images:
        keep = max_images - (1 if render_path is not None else 0)
        flags = [flags[0]] + flags[1:keep] + ([flags[-1]] if render_path is not None else [])
    if len(flags) > max_images:
        raise ValueError(f"Ref2VA image cap is {max_images}, got {len(flags)}")
    return flags
