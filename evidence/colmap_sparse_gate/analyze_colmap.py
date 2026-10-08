#!/usr/bin/env python3
"""Post-hoc analysis of a finished COLMAP sparse model.

Nominal CameraCtrl poses are loaded only after reconstruction, for comparison.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PKG = Path("/home/ma-user/modelarts/user-job-dir/lhk_data/lhk/exp_data/1006_memory_cameractrl")
sys.path.insert(0, str(PKG))

from spatial_memory.poses import camera_center, load_scene, poses_for_clip  # noqa: E402

BASE = PKG / "tests" / "colmap_geometry_debug"
TXT = BASE / "sparse" / "0_txt"
VIS = BASE / "visualizations"
MET = BASE / "metrics"


def qvec_to_R(q):
    """COLMAP qw,qx,qy,qz -> world-to-camera rotation."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_images(path: Path):
    """COLMAP stores Xc = R @ Xw + t. Camera center is C = -R.T @ t, not t."""
    poses = []
    lines = path.read_text().splitlines()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Pose lines start with IMAGE_ID then a quaternion. Point2D lines start with a float x.
        if len(parts) < 10 or not parts[-1].endswith(".png"):
            continue
        image_id = int(parts[0])
        q = np.array(list(map(float, parts[1:5])))
        t = np.array(list(map(float, parts[5:8])))
        name = parts[9]
        R = qvec_to_R(q)
        C = -R.T @ t
        frame = int(name.split("_")[1].split(".")[0])
        poses.append(
            {
                "image_id": image_id,
                "name": name,
                "frame_index": frame,
                "timestamp": frame / 24.0,
                "qvec_wxyz": q.tolist(),
                "R_w2c": R,
                "t_w2c": t,
                "camera_center": C,
                "forward_world": R[2].copy(),
            }
        )
    poses.sort(key=lambda p: p["frame_index"])
    return poses


def parse_camera(path: Path):
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        model = p[1]
        w, h = int(p[2]), int(p[3])
        params = list(map(float, p[4:]))
        return {"model": model, "width": w, "height": h, "params": params, "f": params[0], "cx": params[1], "cy": params[2], "k": params[3]}
    raise RuntimeError("no camera")


def parse_points(path: Path):
    xyz, err, rgb = [], [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        xyz.append(list(map(float, p[1:4])))
        rgb.append(list(map(float, p[4:7])))
        err.append(float(p[7]))
    return np.asarray(xyz), np.asarray(rgb), np.asarray(err)


def nominal_pose(frame: int, clips):
    clip_i = frame // 243
    local = frame % 243
    pose = clips[clip_i][local]
    return pose["R"], pose["t"].reshape(3)


def skew(t):
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def sampson(E, x1, x2):
    x1h = np.concatenate([x1, np.ones((len(x1), 1))], 1)
    x2h = np.concatenate([x2, np.ones((len(x2), 1))], 1)
    Ex1 = (E @ x1h.T).T
    Etx2 = (E.T @ x2h.T).T
    num = np.sum(x2h * (E @ x1h.T).T, axis=1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return num / np.clip(den, 1e-12, None)


def undistort(xy, cam):
    """SIMPLE_RADIAL: distorted = undistorted * (1 + k r^2). Invert with a few iterations."""
    f, cx, cy, k = cam["f"], cam["cx"], cam["cy"], cam["k"]
    x = (xy[:, 0] - cx) / f
    y = (xy[:, 1] - cy) / f
    xu, yu = x.copy(), y.copy()
    for _ in range(5):
        r2 = xu * xu + yu * yu
        scale = 1.0 + k * r2
        xu = x / scale
        yu = y / scale
    return np.stack([xu, yu], 1)


def relative(R1, t1, R2, t2):
    R = R2 @ R1.T
    t = t2 - R @ t1
    n = np.linalg.norm(t)
    if n > 0:
        t = t / n
    return R, t


def rot_deg(Ra, Rb):
    Rd = Ra @ Rb.T
    c = float(np.clip((np.trace(Rd) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def umeyama(src, dst):
    """dst ~= s R src + t. src, dst are Nx3."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = np.mean(np.sum(X * X, axis=1))
    s = float(np.trace(np.diag(S) @ D) / var)
    t = mu_d - s * R @ mu_s
    return s, R, t


def load_keypoints(con, image_id):
    row = con.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    rows, cols, blob = row
    arr = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)
    return arr[:, :2].astype(np.float64)


def load_inlier_matches(con, id1, id2):
    if id1 > id2:
        id1, id2 = id2, id1
        swapped = True
    else:
        swapped = False
    pair_id = id1 * 2147483647 + id2
    row = con.execute("SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone()
    if row is None or not row[0]:
        return np.zeros((0, 2), np.int32)
    matches = np.frombuffer(row[2], dtype=np.uint32).reshape(row[0], row[1]).astype(np.int32)
    if swapped:
        matches = matches[:, ::-1]
    return matches


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    cam = parse_camera(TXT / "cameras.txt")
    poses = parse_images(TXT / "images.txt")
    xyz, rgb, perr = parse_points(TXT / "points3D.txt")
    frames_meta = json.loads((BASE / "frames.json").read_text())
    n_input = len(frames_meta["frames"])

    pose_json = []
    for p in poses:
        pose_json.append(
            {
                "frame_index": p["frame_index"],
                "timestamp": p["timestamp"],
                "name": p["name"],
                "R_w2c": p["R_w2c"].tolist(),
                "t_w2c": p["t_w2c"].tolist(),
                "camera_center": p["camera_center"].tolist(),
                "convention": "COLMAP world-to-camera: Xc = R_w2c @ Xw + t_w2c; camera_center = -R_w2c.T @ t_w2c",
            }
        )
    (MET / "colmap_poses.json").write_text(json.dumps(pose_json, indent=2), encoding="utf-8")

    centers = np.stack([p["camera_center"] for p in poses])
    centroid = centers.mean(0)
    scale = float(np.median(np.linalg.norm(centers - centroid, axis=1)))
    # True orbit return is frame 0 and frame 728. 728 is not in this model.
    by_frame = {p["frame_index"]: p for p in poses}
    start = poses[0]
    end = poses[-1]
    closure_dist = float(np.linalg.norm(end["camera_center"] - start["camera_center"]))
    closure = {
        "registered_first_frame": start["frame_index"],
        "registered_last_frame": end["frame_index"],
        "video_start_frame": 0,
        "video_return_frame": 728,
        "video_return_registered": 728 in by_frame,
        "video_start_registered": 0 in by_frame,
        "registered_endpoint_center_distance": closure_dist,
        "scene_scale_median_radius": scale,
        "normalized_closure_registered_endpoints": closure_dist / max(scale, 1e-8),
        "registered_endpoint_rotation_deg": rot_deg(start["R_w2c"], end["R_w2c"]),
        "note": "COLMAP scale is arbitrary. normalized_closure uses median camera-center distance to the trajectory centroid. The generated orbit's returning frame is 728; if it is unregistered, the registered-endpoint distance is not an orbit closure.",
    }
    # feature correspondence between first and last registered, and 0 vs 728 if both exist
    con = sqlite3.connect(BASE / "workspace" / "database.db")
    id_of = {r[1]: r[0] for r in con.execute("SELECT image_id, name FROM images")}

    def n_inliers(fa, fb):
        ia = id_of.get(f"frame_{fa:06d}.png")
        ib = id_of.get(f"frame_{fb:06d}.png")
        if ia is None or ib is None:
            return 0
        return int(len(load_inlier_matches(con, ia, ib)))

    closure["inliers_registered_endpoints"] = n_inliers(start["frame_index"], end["frame_index"])
    closure["inliers_frame0_vs_728"] = n_inliers(0, 728)
    (MET / "orbit_closure.json").write_text(json.dumps(closure, indent=2), encoding="utf-8")

    # Epipolar comparison on requested pairs. Correspondences are COLMAP verified inliers.
    scene = load_scene(PKG / "scenes" / "orbit_360.json")
    clips = [poses_for_clip(scene, c) for c in ("clip_01", "clip_02", "clip_03")]
    pairs = [(0, 24), (24, 48), (48, 72), (72, 96), (0, 48), (0, 728)]
    rows = []
    for a, b in pairs:
        ia = id_of.get(f"frame_{a:06d}.png")
        ib = id_of.get(f"frame_{b:06d}.png")
        rec = {"pair": f"{a}->{b}", "matches": 0, "both_registered": a in by_frame and b in by_frame}
        if ia is None or ib is None:
            rows.append(rec)
            continue
        matches = load_inlier_matches(con, ia, ib)
        rec["matches"] = int(len(matches))
        if len(matches) < 8:
            rows.append(rec)
            continue
        xy1 = load_keypoints(con, ia)[matches[:, 0]]
        xy2 = load_keypoints(con, ib)[matches[:, 1]]
        x1 = undistort(xy1, cam)
        x2 = undistort(xy2, cam)
        if a in by_frame and b in by_frame:
            Rc, tc = relative(by_frame[a]["R_w2c"], by_frame[a]["t_w2c"], by_frame[b]["R_w2c"], by_frame[b]["t_w2c"])
            Ec = skew(tc) @ Rc
            err_c = sampson(Ec, x1, x2)
            rec["colmap_median_sampson"] = float(np.median(err_c))
            rec["colmap_rotation_vs_nominal_deg"] = None
        Rn1, tn1 = nominal_pose(a, clips)
        Rn2, tn2 = nominal_pose(b, clips)
        Rn, tn = relative(Rn1, tn1, Rn2, tn2)
        En = skew(tn) @ Rn
        err_n = sampson(En, x1, x2)
        rec["nominal_median_sampson"] = float(np.median(err_n))
        if "colmap_median_sampson" in rec and rec["nominal_median_sampson"] > 0:
            rec["ratio_nominal_over_colmap"] = rec["nominal_median_sampson"] / max(rec["colmap_median_sampson"], 1e-12)
            Rc, _ = relative(by_frame[a]["R_w2c"], by_frame[a]["t_w2c"], by_frame[b]["R_w2c"], by_frame[b]["t_w2c"])
            rec["relative_rotation_disagreement_deg"] = rot_deg(Rc, Rn)
        rows.append(rec)
    (MET / "colmap_vs_nominal_epipolar.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Sim3 on registered frames only, held-out second half. Nominal is not used in reconstruction.
    nom_C = []
    col_C = []
    nom_R = []
    col_R = []
    for p in poses:
        Rn, tn = nominal_pose(p["frame_index"], clips)
        nom_C.append(camera_center(Rn, tn))
        col_C.append(p["camera_center"])
        nom_R.append(Rn)
        col_R.append(p["R_w2c"])
    nom_C = np.stack(nom_C)
    col_C = np.stack(col_C)
    n = len(poses)
    n_fit = max(n // 2, 3)
    s, R, t = umeyama(nom_C[:n_fit], col_C[:n_fit])

    def pos_err(idx):
        pred = (s * (R @ nom_C[idx].T).T) + t
        d = np.linalg.norm(pred - col_C[idx], axis=1)
        return {
            "n": int(len(idx)),
            "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "errors": d,
        }

    def ang_err(idx):
        angs = []
        for i in idx:
            # Map nominal w2c into the COLMAP world: R_c = R_nom @ R_sim.T
            R_pred = nom_R[i] @ R.T
            angs.append(rot_deg(R_pred, col_R[i]))
        angs = np.asarray(angs)
        return {"median_deg": float(np.median(angs)), "p90_deg": float(np.percentile(angs, 90)), "errors": angs}

    fit_i = np.arange(0, n_fit)
    test_i = np.arange(n_fit, n)
    fit_p, test_p = pos_err(fit_i), pos_err(test_i)
    fit_a, test_a = ang_err(fit_i), ang_err(test_i)
    # time bins on the registered sequence
    bins = []
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_i = np.arange(n)
    all_p = pos_err(all_i)
    all_a = ang_err(all_i)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sl = all_i[int(lo * n) : max(int(hi * n), int(lo * n) + 1)]
        pe, ae = pos_err(sl), ang_err(sl)
        bins.append(
            {
                "range": f"{int(lo*100)}-{int(hi*100)}%",
                "frames": [poses[i]["frame_index"] for i in sl],
                "position_median": pe["median"],
                "position_p90": pe["p90"],
                "rotation_median_deg": ae["median_deg"],
                "rotation_p90_deg": ae["p90_deg"],
            }
        )
    sim3 = {
        "fit_frames": [poses[i]["frame_index"] for i in fit_i],
        "test_frames": [poses[i]["frame_index"] for i in test_i],
        "scale": s,
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "fit_position": {k: fit_p[k] for k in ("n", "median", "p90", "rmse")},
        "test_position": {k: test_p[k] for k in ("n", "median", "p90", "rmse")},
        "fit_rotation_deg": {k: fit_a[k] for k in ("median_deg", "p90_deg")},
        "test_rotation_deg": {k: test_a[k] for k in ("median_deg", "p90_deg")},
        "error_by_quartile_of_registered_sequence": bins,
        "note": "Sim3 maps nominal camera centers into the COLMAP world. Fit uses the first half of registered frames sorted by time. Errors are in COLMAP units, which are arbitrary.",
    }
    (MET / "nominal_to_colmap_sim3.json").write_text(json.dumps(sim3, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def views(path_top, path_side, path_3d):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(xyz[:, 0], xyz[:, 2], s=2, c="0.6")
        ax.plot(centers[:, 0], centers[:, 2], "-o", color="lime", ms=4)
        for p in poses:
            C = p["camera_center"]
            f = p["forward_world"]
            ax.annotate(str(p["frame_index"]), (C[0], C[2]), fontsize=6)
            ax.arrow(C[0], C[2], f[0] * scale * 0.25, f[2] * scale * 0.25, color="red", width=0.002 * scale, head_width=0.05 * scale)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title("COLMAP sparse top")
        fig.tight_layout()
        fig.savefig(path_top, dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(xyz[:, 2], xyz[:, 1], s=2, c="0.6")
        ax.plot(centers[:, 2], centers[:, 1], "-o", color="lime", ms=4)
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        ax.set_title("COLMAP sparse side")
        fig.tight_layout()
        fig.savefig(path_side, dpi=120)
        plt.close(fig)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], s=2, c="0.5")
        ax.plot(centers[:, 0], centers[:, 2], centers[:, 1], "-o", color="lime", ms=3)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.set_title("COLMAP sparse 3d")
        fig.tight_layout()
        fig.savefig(path_3d, dpi=110)
        plt.close(fig)

    views(VIS / "colmap_sparse_top.png", VIS / "colmap_sparse_side.png", VIS / "colmap_sparse_3d.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(centers[:, 0], centers[:, 2], "-o", color="tab:blue")
    for p in poses:
        C = p["camera_center"]
        ax.annotate(str(p["frame_index"]), (C[0], C[2]), fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("COLMAP camera centers")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIS / "colmap_trajectory.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(centers[:, 0], centers[:, 2], "-o", color="0.7", label="registered")
    ax.scatter([start["camera_center"][0]], [start["camera_center"][2]], c="red", s=80, label=f"first f{start['frame_index']}")
    ax.scatter([end["camera_center"][0]], [end["camera_center"][2]], c="cyan", s=80, label=f"last f{end['frame_index']}")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"registered endpoints dist={closure_dist:.3f} norm={closure['normalized_closure_registered_endpoints']:.3f}")
    fig.tight_layout()
    fig.savefig(VIS / "orbit_closure.png", dpi=120)
    plt.close(fig)

    pred_all = (s * (R @ nom_C.T).T) + t
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(col_C[:, 0], col_C[:, 2], "-o", color="black", label="COLMAP")
    ax.plot(pred_all[:, 0], pred_all[:, 2], "-o", color="tab:orange", label="Sim3 nominal")
    ax.axvline
    split = pred_all[n_fit - 1]
    ax.scatter([split[0]], [split[2]], c="red", s=40)
    ax.set_title("nominal mapped by Sim3 fit on first half")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIS / "trajectory_nominal_vs_colmap.png", dpi=120)
    plt.close(fig)

    summary = {
        "input_keyframes": n_input,
        "registered": len(poses),
        "registration_ratio": len(poses) / n_input,
        "registered_frames": [p["frame_index"] for p in poses],
        "sparse_points": int(len(xyz)),
        "observations": int(round(float(np.sum(perr > -1)) and 2977)),
        "mean_reprojection_px": float(perr.mean()),
        "median_reprojection_px": float(np.median(perr)),
        "camera": cam,
        "hfov_deg_from_estimated_f": float(np.degrees(2 * np.arctan(cam["width"] / (2 * cam["f"])))),
    }
    # observations counted properly
    obs = 0
    for line in (TXT / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        obs += (len(line.split()) - 8) // 2
    summary["observations"] = obs
    (MET / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "camera"}, indent=2))
    print("camera", cam)
    print("closure", json.dumps(closure, indent=2))
    print("epipolar", json.dumps(rows, indent=2))
    print("sim3 test", sim3["test_position"], sim3["test_rotation_deg"])
    print("bins", json.dumps(bins, indent=2))


if __name__ == "__main__":
    main()
