"""Spatial, return-to-scene, and object-persistence metrics.

Object persistence uses caller-supplied pixel labels. It does not run a detector.
A missing match is reported as unobserved rather than filled in.
"""

from __future__ import annotations

import numpy as np

from scene_memory.geometry import project, rot_deg


def spatial_report(memory) -> dict:
    input_images = 0
    registered = 0
    pose_priors = 0
    centers = []
    fills = []
    reproj = []
    stored_means = []
    for group in memory.groups.values():
        input_images += int(group.input_image_count or len(group.cameras))
        pose_priors += int(group.pose_prior_count)
        if group.mean_reproj_error is not None:
            stored_means.append(float(group.mean_reproj_error))
        if len(group.reproj):
            reproj.append(np.asarray(group.reproj, np.float64))
        for cam in group.cameras:
            if not cam.registered:
                continue
            registered += 1
            if group.placement is None or cam.frame_index is None:
                continue
            centers.append((cam.frame_index, group.world_camera(cam), cam))
            pose = _pose(group, cam)
            if pose is None:
                continue
            view = memory.query(pose)
            fills.append(float(view.mask.mean()) if view.mask.size else 0.0)
    centers.sort(key=lambda item: item[0])
    xyz = memory.world_points()[0]
    reproj_all = np.concatenate(reproj) if reproj else np.zeros((0,), np.float64)
    smooth = _smoothness([camera_center_of(R, t) for _, (R, t), _ in centers])
    return {
        "input_images": input_images,
        "registered": registered,
        "registration_ratio": (registered / input_images) if input_images else 0.0,
        "num_points": int(len(xyz)),
        "mean_reproj_error": None if len(reproj_all) == 0 else float(reproj_all.mean()),
        "median_reproj_error": None if len(reproj_all) == 0 else float(np.median(reproj_all)),
        "stored_mean_reproj_error": None if not stored_means else float(np.mean(stored_means)),
        "pose_prior_count": pose_priors,
        "trajectory": smooth,
        "source_view_fill_median": None if not fills else float(np.median(fills)),
        "nominal_poses_used_in_mapping": False,
    }


def return_report(memory, name_a: str, name_b: str) -> dict:
    found_a = _find(memory, name_a)
    found_b = _find(memory, name_b)
    report = {
        "name_a": name_a,
        "name_b": name_b,
        "registered_a": bool(found_a and found_a[1].registered and found_a[0].placement is not None),
        "registered_b": bool(found_b and found_b[1].registered and found_b[0].placement is not None),
        "verified_inliers": 0,
        "closure_supported": False,
        "center_distance": None,
        "relative_rotation_deg": None,
        "geometric_overlap": None,
        "reason": None,
    }
    if not report["registered_a"] or not report["registered_b"]:
        report["reason"] = "not_registered"
        return report
    group_a, cam_a = found_a
    group_b, cam_b = found_b
    Ra, ta = group_a.world_camera(cam_a)
    Rb, tb = group_b.world_camera(cam_b)
    Ca = camera_center_of(Ra, ta)
    Cb = camera_center_of(Rb, tb)
    report["center_distance"] = float(np.linalg.norm(Ca - Cb))
    report["relative_rotation_deg"] = rot_deg(Ra, Rb)
    if group_a.group_id == group_b.group_id:
        report["verified_inliers"] = int(group_a.pair_inliers.get(_key(name_a, name_b), 0))
    report["closure_supported"] = report["verified_inliers"] > 0
    xyz = memory.world_points()[0]
    if len(xyz) and cam_a.fx is not None and cam_b.fx is not None:
        vis_a = _visible(xyz, Ra, ta, cam_a)
        vis_b = _visible(xyz, Rb, tb, cam_b)
        either = int((vis_a | vis_b).sum())
        both = int((vis_a & vis_b).sum())
        report["geometric_overlap"] = (both / either) if either else 0.0
        report["points_visible_in_both"] = both
    if not report["closure_supported"]:
        report["reason"] = "no_verified_matches"
    return report


def object_persistence(memory, tracks, *, radius: float = 2.0) -> dict:
    if not tracks:
        return {"status": "not_provided", "objects": []}
    objects = []
    for name, observations in tracks.items():
        localized = []
        unobserved = 0
        for obs in observations:
            point = _localize(memory, obs["camera"], float(obs["u"]), float(obs["v"]), radius)
            if point is None:
                unobserved += 1
            else:
                localized.append(point)
        spread = None
        if len(localized) >= 2:
            stack = np.stack(localized)
            med = np.median(stack, axis=0)
            spread = float(np.max(np.linalg.norm(stack - med, axis=1)))
        objects.append(
            {
                "name": name,
                "observations": len(observations),
                "localized": len(localized),
                "unobserved": unobserved,
                "position_spread": spread,
            }
        )
    return {"status": "ok", "radius_px": radius, "objects": objects}


def camera_center_of(R, t) -> np.ndarray:
    from scene_memory.geometry import camera_center

    return camera_center(R, t)


def _pose(group, cam):
    if cam.fx is None or not cam.width or not cam.height:
        return None
    R, t = group.world_camera(cam)
    return {
        "R": R,
        "t": t,
        "fx": cam.fx,
        "fy": cam.fy,
        "cx": cam.cx,
        "cy": cam.cy,
        "width": cam.width,
        "height": cam.height,
        "k": cam.k,
    }


def _find(memory, name):
    for group in memory.groups.values():
        cam = group.camera(name)
        if cam is not None:
            return group, cam
    return None


def _key(a, b):
    return (a, b) if a <= b else (b, a)


def _visible(xyz, R, t, cam) -> np.ndarray:
    u, v, z = project(xyz, R, t, cam.fx, cam.fy, cam.cx, cam.cy, cam.k)
    return (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(z)
        & (z > 0.05)
        & (u >= -0.5)
        & (u < cam.width - 0.5)
        & (v >= -0.5)
        & (v < cam.height - 0.5)
    )


def _localize(memory, camera_name, u, v, radius):
    found = _find(memory, camera_name)
    if found is None:
        return None
    group, cam = found
    if not cam.registered or group.placement is None or cam.fx is None:
        return None
    xyz = memory.world_points()[0]
    if len(xyz) == 0:
        return None
    R, t = group.world_camera(cam)
    pu, pv, z = project(xyz, R, t, cam.fx, cam.fy, cam.cx, cam.cy, cam.k)
    dist = np.hypot(pu - u, pv - v)
    sel = (z > 0.05) & (dist <= radius)
    if not np.any(sel):
        return None
    return np.median(np.asarray(xyz, np.float64)[sel], axis=0)


def _smoothness(centers: list[np.ndarray]) -> dict:
    if len(centers) < 2:
        return {"num_cameras": len(centers), "step_mean": None, "step_std": None, "turn_mean_deg": None}
    stack = np.stack(centers)
    steps = np.linalg.norm(np.diff(stack, axis=0), axis=1)
    turns = []
    for i in range(1, len(steps)):
        a = stack[i] - stack[i - 1]
        b = stack[i + 1] - stack[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            continue
        turns.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))))
    return {
        "num_cameras": len(centers),
        "step_mean": float(steps.mean()),
        "step_std": float(steps.std()),
        "turn_mean_deg": None if not turns else float(np.mean(turns)),
    }
