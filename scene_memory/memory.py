"""Persistent scene memory.

Local clouds come from a reconstruction backend. A shared world frame exists
only where an explicit placement is set. A single reconstructed group is placed
at the identity. Disjoint groups are not stacked into one rigid model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scene_memory.geometry import (
    alignment_rmse,
    camera_center,
    invert_rigid,
    sim3_camera,
    sim3_points,
    umeyama,
)
from scene_memory.render import MemoryView, render_points
from scene_memory.scene_graph import T_to_json, as_T, eye4
from scene_memory.types import Camera, Reconstruction, UpdateResult


NOMINAL_SOURCES = {"nominal", "waypoint", "cameractrl", "camera_ctrl"}
MIN_OVERLAP_CAMERAS = 3
DEFAULT_MIN_INLIERS = 15


def pose_source(camera_pose) -> str | None:
    if not camera_pose:
        return None
    raw = camera_pose.get("source", camera_pose.get("provenance"))
    if raw is None:
        return None
    return str(raw).lower()


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def inlier_count(pairs: dict, a: str, b: str) -> int:
    return int(pairs.get(pair_key(a, b), 0))


@dataclass
class GroupCloud:
    group_id: str
    reconstructor: str
    cameras: list[Camera]
    xyz: np.ndarray
    rgb: np.ndarray
    features: np.ndarray
    src: np.ndarray
    src_name: np.ndarray
    reproj: np.ndarray
    mean_reproj_error: float | None
    median_reproj_error: float | None
    pair_inliers: dict[tuple[str, str], int]
    pose_prior_count: int
    input_image_count: int
    focal_prior_px: float | None
    intrinsics_refined: bool
    principal_point_refined: bool
    placement: np.ndarray | None = None

    def camera(self, name: str) -> Camera | None:
        for cam in self.cameras:
            if cam.image_name == name:
                return cam
        return None

    def registered(self) -> list[Camera]:
        return [cam for cam in self.cameras if cam.registered and cam.R_w2c is not None]

    def world_xyz(self) -> np.ndarray | None:
        if self.placement is None or len(self.xyz) == 0:
            return None if self.placement is None else self.xyz
        T = self.placement
        return sim3_points(self.xyz, 1.0, T[:3, :3], T[:3, 3]).astype(np.float32)

    def world_camera(self, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
        if self.placement is None:
            raise RuntimeError(f"group {self.group_id} has no world placement")
        T = self.placement
        return sim3_camera(cam.R_w2c, cam.t_w2c, 1.0, T[:3, :3], T[:3, 3])


def _empty_arrays():
    return (
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.float32),
        np.zeros((0, 4), np.float32),
        np.zeros((0,), np.int32),
        np.array([], dtype="U256"),
        np.zeros((0,), np.float32),
    )


@dataclass
class ChunkEvidence:
    """A joint reconstruction of overlap images plus new frames, in its own frame."""

    source: str
    reconstructor: str
    reconstruction: Reconstruction
    overlap_names: list[str]
    new_names: list[str]
    group_id: str
    return_frame: str | None = None
    anchor_frame: str | None = None


@dataclass
class SceneMemory:
    groups: dict[str, GroupCloud] = field(default_factory=dict)
    update_log: list[dict] = field(default_factory=list)

    def build(self, images, backend, *, placements=None, work_root: Path | str | None = None) -> "SceneMemory":
        """Reconstruct each overlap group separately. Do not register disjoint groups together."""
        groups = _as_groups(images)
        placements = placements or {}
        root = Path(work_root) if work_root is not None else None
        single = len(groups) == 1
        for gid, paths in groups.items():
            work = None if root is None else root / gid
            recon = backend.reconstruct(list(paths), work)
            if int(getattr(recon, "pose_prior_count", 0) or 0) != 0:
                raise RuntimeError(f"{gid}: reconstruction contains pose priors")
            place = as_T(placements.get(gid))
            if place is None and single and gid not in placements:
                place = eye4()
            self._add_reconstruction(gid, recon, place)
        return self

    def _add_reconstruction(self, gid: str, recon: Reconstruction, placement) -> None:
        cams = []
        for cam in recon.cameras:
            cam.group_id = gid
            cam.reconstructor = recon.reconstructor
            cams.append(cam)
        refined = any(cam.intrinsics_refined for cam in cams)
        pp = any(cam.principal_point_refined for cam in cams)
        self.groups[gid] = GroupCloud(
            group_id=gid,
            reconstructor=recon.reconstructor,
            cameras=cams,
            xyz=np.asarray(recon.xyz, np.float32).reshape(-1, 3),
            rgb=np.asarray(recon.rgb, np.float32).reshape(-1, 3),
            features=np.asarray(recon.features, np.float32),
            src=np.asarray(recon.src, np.int32).reshape(-1),
            src_name=np.asarray(recon.src_name).astype("U256").reshape(-1),
            reproj=np.asarray(recon.reproj, np.float32).reshape(-1),
            mean_reproj_error=recon.mean_reproj_error,
            median_reproj_error=recon.median_reproj_error,
            pair_inliers={pair_key(*k): int(v) for k, v in recon.pair_inliers.items()},
            pose_prior_count=int(recon.pose_prior_count),
            input_image_count=int(recon.input_image_count or len(cams)),
            focal_prior_px=recon.focal_prior_px,
            intrinsics_refined=refined,
            principal_point_refined=pp,
            placement=as_T(placement),
        )

    def set_placement(self, group_id: str, world_from_group) -> None:
        self.groups[group_id].placement = as_T(world_from_group)

    def world_points(self):
        """Placed points in the world frame. Unplaced groups are left out."""
        return self._world_points()

    def query(self, camera_pose: dict) -> MemoryView:
        """Render the placed world from a w2c pose. Unplaced groups are omitted."""
        xyz, rgb, src, names = self._world_points()
        return render_points(xyz, rgb, src, names, camera_pose)

    def update(
        self,
        new_frames,
        camera_pose=None,
        *,
        evidence: ChunkEvidence | None = None,
        backend=None,
        work_dir: Path | str | None = None,
        group_id: str | None = None,
        return_frame: str | None = None,
        anchor_frame: str | None = None,
        min_inliers: int = DEFAULT_MIN_INLIERS,
        max_alignment_rmse: float | None = None,
    ) -> UpdateResult:
        """Merge new observations into an existing group.

        ``camera_pose`` is not a measurement. A nominal or waypoint pose is rejected.
        Without reconstruction evidence the memory is left unchanged.
        """
        source = pose_source(camera_pose)
        if source in NOMINAL_SOURCES:
            return self._reject("nominal_pose_rejected", group_id=group_id, closure=_closure_flag(return_frame, False))
        if evidence is None:
            if backend is None:
                return self._reject("missing_reconstruction_evidence", group_id=group_id, closure=_closure_flag(return_frame, False))
            gid = group_id or self._only_placed_group()
            if gid is None:
                return self._reject("group_not_in_world", closure=_closure_flag(return_frame, False))
            built = self._evidence_from_backend(
                gid,
                list(new_frames),
                backend,
                None if work_dir is None else Path(work_dir),
                return_frame=return_frame,
                anchor_frame=anchor_frame,
            )
            if isinstance(built, UpdateResult):
                return built
            evidence = built
        if evidence.source.lower() in NOMINAL_SOURCES:
            return self._reject("nominal_pose_rejected", group_id=evidence.group_id, closure=_closure_flag(evidence.return_frame, False))
        return self._apply_evidence(evidence, min_inliers=min_inliers, max_alignment_rmse=max_alignment_rmse)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        chunks_xyz, chunks_rgb, chunks_feat, chunks_src, chunks_name, chunks_err, chunks_gid = [], [], [], [], [], [], []
        cameras = {}
        placements = {}
        for gid, group in self.groups.items():
            cameras[gid] = {
                "reconstructor": group.reconstructor,
                "intrinsics_refined": group.intrinsics_refined,
                "principal_point_refined": group.principal_point_refined,
                "mean_reproj_error": group.mean_reproj_error,
                "median_reproj_error": group.median_reproj_error,
                "input_image_count": group.input_image_count,
                "pose_prior_count": group.pose_prior_count,
                "focal_prior_px": group.focal_prior_px,
                "nominal_poses_used_in_mapping": False,
                "cameras": [cam.to_json() for cam in group.cameras],
            }
            placements[gid] = T_to_json(group.placement)
            n = len(group.xyz)
            if n == 0:
                continue
            chunks_xyz.append(group.xyz)
            chunks_rgb.append(group.rgb)
            chunks_feat.append(group.features)
            chunks_src.append(group.src)
            chunks_name.append(group.src_name.astype("U256"))
            chunks_err.append(group.reproj)
            chunks_gid.append(np.full((n,), gid, dtype="U256"))
        if chunks_xyz:
            blob = dict(
                xyz=np.concatenate(chunks_xyz),
                rgb=np.concatenate(chunks_rgb),
                features=np.concatenate(chunks_feat),
                src=np.concatenate(chunks_src),
                src_name=np.concatenate(chunks_name),
                reproj=np.concatenate(chunks_err),
                group_id=np.concatenate(chunks_gid),
            )
        else:
            xyz, rgb, feat, src, names, err = _empty_arrays()
            blob = dict(xyz=xyz, rgb=rgb, features=feat, src=src, src_name=names, reproj=err, group_id=np.array([], dtype="U256"))
        np.savez_compressed(path / "points.npz", **blob)
        (path / "cameras.json").write_text(json.dumps({"groups": cameras}, indent=2), encoding="utf-8")
        (path / "scene_graph.json").write_text(
            json.dumps({"nodes": [{"id": gid} for gid in self.groups], "placements": placements}, indent=2),
            encoding="utf-8",
        )
        pairs = {}
        for gid, group in self.groups.items():
            pairs[gid] = {f"{a}|{b}": int(n) for (a, b), n in sorted(group.pair_inliers.items())}
        (path / "pair_inliers.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
        (path / "update_log.json").write_text(json.dumps(self.update_log, indent=2), encoding="utf-8")
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "nominal_poses_used_in_mapping": False,
                    "depth_anything_used": False,
                    "groups": list(self.groups),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "SceneMemory":
        path = Path(path)
        cameras = json.loads((path / "cameras.json").read_text(encoding="utf-8"))["groups"]
        graph = json.loads((path / "scene_graph.json").read_text(encoding="utf-8"))
        raw_pairs = json.loads((path / "pair_inliers.json").read_text(encoding="utf-8")) if (path / "pair_inliers.json").is_file() else {}
        log = json.loads((path / "update_log.json").read_text(encoding="utf-8")) if (path / "update_log.json").is_file() else []
        with np.load(path / "points.npz") as blob:
            xyz = np.asarray(blob["xyz"], np.float32)
            rgb = np.asarray(blob["rgb"], np.float32)
            features = np.asarray(blob["features"], np.float32)
            src = np.asarray(blob["src"], np.int32)
            src_name = np.asarray(blob["src_name"]).astype("U256")
            reproj = np.asarray(blob["reproj"], np.float32)
            group_ids = np.asarray(blob["group_id"]).astype("U256")
        memory = cls(update_log=log)
        for gid, spec in cameras.items():
            sel = group_ids == gid if len(group_ids) else np.zeros((0,), dtype=bool)
            pairs = {}
            for key, count in raw_pairs.get(gid, {}).items():
                a, b = key.split("|", 1)
                pairs[pair_key(a, b)] = int(count)
            recon = Reconstruction(
                reconstructor=spec["reconstructor"],
                cameras=[Camera.from_json(item) for item in spec["cameras"]],
                xyz=xyz[sel] if len(xyz) else np.zeros((0, 3), np.float32),
                rgb=rgb[sel] if len(rgb) else np.zeros((0, 3), np.float32),
                features=features[sel] if len(features) else np.zeros((0, features.shape[1] if features.ndim == 2 else 4), np.float32),
                src=src[sel] if len(src) else np.zeros((0,), np.int32),
                src_name=src_name[sel] if len(src_name) else np.array([], dtype="U256"),
                reproj=reproj[sel] if len(reproj) else np.zeros((0,), np.float32),
                mean_reproj_error=spec.get("mean_reproj_error"),
                median_reproj_error=spec.get("median_reproj_error"),
                pair_inliers=pairs,
                pose_prior_count=int(spec.get("pose_prior_count") or 0),
                input_image_count=int(spec.get("input_image_count") or 0),
                focal_prior_px=spec.get("focal_prior_px"),
            )
            memory._add_reconstruction(gid, recon, as_T(graph["placements"].get(gid)))
        return memory

    def _world_points(self):
        xs, cs, ss, ns = [], [], [], []
        for group in self.groups.values():
            if group.placement is None:
                continue
            world = group.world_xyz()
            if world is None or len(world) == 0:
                continue
            xs.append(np.asarray(world, np.float64))
            cs.append(np.asarray(group.rgb, np.float64))
            ss.append(group.src)
            ns.append(group.src_name)
        if not xs:
            return (
                np.zeros((0, 3), np.float64),
                np.zeros((0, 3), np.float64),
                np.zeros((0,), np.int32),
                np.array([], dtype="U256"),
            )
        return np.concatenate(xs), np.concatenate(cs), np.concatenate(ss), np.concatenate(ns)

    def _only_placed_group(self) -> str | None:
        placed = [gid for gid, group in self.groups.items() if group.placement is not None]
        if len(placed) == 1:
            return placed[0]
        return None

    def _evidence_from_backend(self, gid, new_frames, backend, work_dir, *, return_frame, anchor_frame):
        group = self.groups.get(gid)
        if group is None or group.placement is None:
            return self._reject("group_not_in_world", group_id=gid, closure=_closure_flag(return_frame, False))
        overlap = [cam for cam in group.registered() if cam.image_path]
        if len(overlap) < MIN_OVERLAP_CAMERAS:
            return self._reject("insufficient_overlap", group_id=gid, closure=_closure_flag(return_frame, False))
        paths = [Path(cam.image_path) for cam in overlap] + [Path(p) for p in new_frames]
        recon = backend.reconstruct(paths, work_dir if work_dir is not None else Path("."))
        if int(recon.pose_prior_count or 0) != 0:
            return self._reject("pose_prior_present", group_id=gid, closure=_closure_flag(return_frame, False))
        return ChunkEvidence(
            source="reconstruction",
            reconstructor=recon.reconstructor,
            reconstruction=recon,
            overlap_names=[cam.image_name for cam in overlap],
            new_names=[Path(p).name for p in new_frames],
            group_id=gid,
            return_frame=return_frame,
            anchor_frame=anchor_frame,
        )

    def _apply_evidence(self, evidence: ChunkEvidence, *, min_inliers: int, max_alignment_rmse: float | None) -> UpdateResult:
        gid = evidence.group_id
        group = self.groups.get(gid)
        closure_requested = evidence.return_frame is not None
        if group is None or group.placement is None:
            return self._reject("group_not_in_world", group_id=gid, closure=_closure_flag(evidence.return_frame, False))
        if evidence.source.lower() not in {"reconstruction", "explicit_alignment"}:
            return self._reject("missing_reconstruction_evidence", group_id=gid, closure=_closure_flag(evidence.return_frame, False))
        old = {cam.image_name: cam for cam in group.registered()}
        new_model = {cam.image_name: cam for cam in evidence.reconstruction.cameras if cam.registered}
        missing = [name for name in evidence.overlap_names if name not in old or name not in new_model]
        if missing or len(evidence.overlap_names) < MIN_OVERLAP_CAMERAS:
            return self._reject(
                "insufficient_overlap" if len(evidence.overlap_names) < MIN_OVERLAP_CAMERAS else "alignment_failed",
                group_id=gid,
                closure=_closure_flag(evidence.return_frame, False),
                skipped=list(evidence.new_names),
            )
        old_c = np.stack([self._world_center(group, old[name]) for name in evidence.overlap_names])
        new_c = np.stack([new_model[name].center() for name in evidence.overlap_names])
        try:
            s, R, t = umeyama(new_c, old_c)
            rmse = alignment_rmse(new_c, old_c, s, R, t)
        except ValueError:
            return self._reject("alignment_failed", group_id=gid, closure=_closure_flag(evidence.return_frame, False))
        limit = max_alignment_rmse
        if limit is None:
            radius = _median_radius(old_c)
            limit = max(0.25 * radius, 1e-4)
        if rmse > limit:
            return self._reject(
                "alignment_failed",
                group_id=gid,
                closure=_closure_flag(evidence.return_frame, False),
                alignment_rmse=rmse,
                skipped=list(evidence.new_names),
            )
        pairs = evidence.reconstruction.pair_inliers
        anchor = evidence.anchor_frame or _anchor_name(evidence.overlap_names, old)
        qualified = []
        skipped = []
        inliers_for_return = None
        for name in evidence.new_names:
            if name not in new_model:
                skipped.append(name)
                continue
            best = max((inlier_count(pairs, name, other) for other in evidence.overlap_names), default=0)
            if evidence.return_frame == name:
                inliers_for_return = inlier_count(pairs, name, anchor) if anchor else 0
                if inliers_for_return == 0:
                    return self._reject(
                        "closure_rejected_no_verified_matches",
                        group_id=gid,
                        closure="rejected",
                        alignment_rmse=rmse,
                        verified_inliers=0,
                        skipped=list(evidence.new_names),
                    )
                if inliers_for_return < min_inliers:
                    return self._reject(
                        "closure_rejected_insufficient_inliers",
                        group_id=gid,
                        closure="rejected",
                        alignment_rmse=rmse,
                        verified_inliers=inliers_for_return,
                        skipped=list(evidence.new_names),
                    )
            if best < min_inliers:
                skipped.append(name)
                continue
            qualified.append(name)
        if closure_requested and evidence.return_frame not in qualified:
            return self._reject(
                "closure_rejected_no_verified_matches",
                group_id=gid,
                closure="rejected",
                alignment_rmse=rmse,
                verified_inliers=inliers_for_return or 0,
                skipped=list(evidence.new_names),
            )
        if not qualified:
            return self._reject(
                "no_verified_overlap",
                group_id=gid,
                closure="not_requested",
                alignment_rmse=rmse,
                skipped=skipped,
            )
        recon = evidence.reconstruction
        keep = np.array([str(n) in set(qualified) for n in recon.src_name.tolist()], dtype=bool) if len(recon.src_name) else np.zeros((0,), bool)
        xyz_new = sim3_points(recon.xyz[keep], s, R, t) if keep.any() else np.zeros((0, 3), np.float64)
        T_inv = invert_rigid(group.placement)
        xyz_g = sim3_points(xyz_new, 1.0, T_inv[:3, :3], T_inv[:3, 3]).astype(np.float32)
        rgb_g = np.asarray(recon.rgb[keep], np.float32) if keep.any() else np.zeros((0, 3), np.float32)
        feat_g = np.asarray(recon.features[keep], np.float32) if keep.any() else np.zeros((0, group.features.shape[1]), np.float32)
        src_g = np.asarray(recon.src[keep], np.int32) if keep.any() else np.zeros((0,), np.int32)
        name_g = np.asarray(recon.src_name[keep]).astype("U256") if keep.any() else np.array([], dtype="U256")
        err_g = np.asarray(recon.reproj[keep], np.float32) if keep.any() else np.zeros((0,), np.float32)
        cameras_to_add = []
        for name in qualified:
            cam = new_model[name]
            Rw, tw = sim3_camera(cam.R_w2c, cam.t_w2c, s, R, t)
            Rg, tg = sim3_camera(Rw, tw, 1.0, T_inv[:3, :3], T_inv[:3, 3])
            cameras_to_add.append(
                (
                    name,
                    Camera(
                        image_name=cam.image_name,
                        image_path=cam.image_path,
                        registered=True,
                        width=cam.width,
                        height=cam.height,
                        R_w2c=Rg,
                        t_w2c=tg,
                        model=cam.model,
                        fx=cam.fx,
                        fy=cam.fy,
                        cx=cam.cx,
                        cy=cam.cy,
                        k=cam.k,
                        intrinsics_refined=cam.intrinsics_refined,
                        principal_point_refined=cam.principal_point_refined,
                        frame_index=cam.frame_index,
                        group_id=gid,
                        reconstructor=evidence.reconstructor,
                    ),
                )
            )
        if len(xyz_g):
            group.xyz = np.concatenate([group.xyz, xyz_g])
            group.rgb = np.concatenate([group.rgb, rgb_g])
            group.features = np.concatenate([group.features, feat_g])
            group.src = np.concatenate([group.src, src_g])
            group.src_name = np.concatenate([group.src_name, name_g])
            group.reproj = np.concatenate([group.reproj, err_g])
        for name, stored in cameras_to_add:
            existing = group.camera(name)
            if existing is None:
                group.cameras.append(stored)
            else:
                existing.registered = True
                existing.R_w2c = stored.R_w2c
                existing.t_w2c = stored.t_w2c
                existing.image_path = stored.image_path or existing.image_path
        for key, count in pairs.items():
            group.pair_inliers[pair_key(*key)] = int(count)
        return_distance = None
        closure = "not_requested"
        if closure_requested:
            closure = "supported"
            anchor_cam = old.get(anchor) if anchor else None
            ret_cam = new_model.get(evidence.return_frame)
            if anchor_cam is not None and ret_cam is not None:
                C_ret = sim3_points(ret_cam.center()[None, :], s, R, t)[0]
                return_distance = float(np.linalg.norm(C_ret - self._world_center(group, anchor_cam)))
        result = UpdateResult(
            accepted=True,
            reason="merged",
            added_points=int(len(xyz_g)),
            skipped_frames=skipped,
            closure=closure,
            alignment_rmse=rmse,
            return_center_distance=return_distance,
            verified_inliers=inliers_for_return,
            group_id=gid,
        )
        self.update_log.append(result.to_json())
        return result

    def _world_center(self, group: GroupCloud, cam: Camera) -> np.ndarray:
        R, t = group.world_camera(cam)
        return camera_center(R, t)

    def _reject(self, reason: str, *, group_id=None, closure="not_requested", alignment_rmse=None, verified_inliers=None, skipped=None) -> UpdateResult:
        result = UpdateResult(
            accepted=False,
            reason=reason,
            added_points=0,
            skipped_frames=list(skipped or []),
            closure=closure,
            alignment_rmse=alignment_rmse,
            verified_inliers=verified_inliers,
            group_id=group_id,
        )
        self.update_log.append(result.to_json())
        return result


def _as_groups(images) -> dict[str, list]:
    if isinstance(images, dict):
        return {str(k): list(v) for k, v in images.items()}
    return {"default": list(images)}


def _closure_flag(return_frame, supported: bool) -> str:
    if return_frame is None:
        return "not_requested"
    return "supported" if supported else "rejected"


def _median_radius(centers: np.ndarray) -> float:
    mu = np.asarray(centers, np.float64).mean(0)
    dist = np.linalg.norm(np.asarray(centers, np.float64) - mu, axis=1)
    return float(np.median(dist))


def _anchor_name(overlap_names: list[str], old: dict[str, Camera]) -> str | None:
    def sort_key(name: str):
        cam = old[name]
        return (cam.frame_index is None, cam.frame_index if cam.frame_index is not None else 0, name)

    if not overlap_names:
        return None
    return sorted(overlap_names, key=sort_key)[0]
