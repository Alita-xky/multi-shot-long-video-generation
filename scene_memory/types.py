"""Scene memory records. Poses stored here come from reconstruction or an explicit placement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scene_memory.geometry import as_R, as_t, camera_center


def _tolist(R):
    if R is None:
        return None
    return np.asarray(R, dtype=np.float64).reshape(3, 3).tolist()


def _tlist(t):
    if t is None:
        return None
    return np.asarray(t, dtype=np.float64).reshape(3).tolist()


@dataclass
class Camera:
    image_name: str
    registered: bool
    width: int
    height: int
    image_path: str | None = None
    R_w2c: np.ndarray | None = None
    t_w2c: np.ndarray | None = None
    model: str = "SIMPLE_PINHOLE"
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    k: float = 0.0
    intrinsics_refined: bool = False
    principal_point_refined: bool = False
    frame_index: int | None = None
    group_id: str | None = None
    reconstructor: str | None = None

    def center(self) -> np.ndarray:
        if not self.registered or self.R_w2c is None or self.t_w2c is None:
            raise RuntimeError(f"{self.image_name} is not registered")
        return camera_center(self.R_w2c, self.t_w2c)

    def to_json(self) -> dict:
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "registered": bool(self.registered),
            "width": int(self.width),
            "height": int(self.height),
            "R_w2c": _tolist(self.R_w2c),
            "t_w2c": _tlist(self.t_w2c),
            "model": self.model,
            "fx": None if self.fx is None else float(self.fx),
            "fy": None if self.fy is None else float(self.fy),
            "cx": None if self.cx is None else float(self.cx),
            "cy": None if self.cy is None else float(self.cy),
            "k": float(self.k),
            "intrinsics_refined": bool(self.intrinsics_refined),
            "principal_point_refined": bool(self.principal_point_refined),
            "frame_index": self.frame_index,
            "group_id": self.group_id,
            "reconstructor": self.reconstructor,
        }

    @classmethod
    def from_json(cls, blob: dict) -> "Camera":
        R = None if blob["R_w2c"] is None else as_R(blob["R_w2c"])
        t = None if blob["t_w2c"] is None else as_t(blob["t_w2c"])
        return cls(
            image_name=blob["image_name"],
            image_path=blob.get("image_path"),
            registered=bool(blob["registered"]),
            width=int(blob["width"]),
            height=int(blob["height"]),
            R_w2c=R,
            t_w2c=t,
            model=blob.get("model", "SIMPLE_PINHOLE"),
            fx=blob.get("fx"),
            fy=blob.get("fy"),
            cx=blob.get("cx"),
            cy=blob.get("cy"),
            k=float(blob.get("k") or 0.0),
            intrinsics_refined=bool(blob.get("intrinsics_refined", False)),
            principal_point_refined=bool(blob.get("principal_point_refined", False)),
            frame_index=blob.get("frame_index"),
            group_id=blob.get("group_id"),
            reconstructor=blob.get("reconstructor"),
        )


@dataclass
class Reconstruction:
    """One overlap group's reconstruction, in that reconstruction's own frame."""

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
    pair_inliers: dict[tuple[str, str], int] = field(default_factory=dict)
    pose_prior_count: int = 0
    input_image_count: int = 0
    focal_prior_px: float | None = None

    def __post_init__(self) -> None:
        self.xyz = np.zeros((0, 3), np.float32) if self.xyz is None else np.asarray(self.xyz, np.float32).reshape(-1, 3)
        self.rgb = np.zeros((0, 3), np.float32) if self.rgb is None else np.asarray(self.rgb, np.float32).reshape(-1, 3)
        feat = np.zeros((0, 4), np.float32) if self.features is None else np.asarray(self.features, np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(-1, 1)
        if len(self.xyz) and len(feat) != len(self.xyz):
            raise ValueError("features and xyz length differ")
        self.features = feat
        self.src = np.zeros((0,), np.int32) if self.src is None else np.asarray(self.src, np.int32).reshape(-1)
        if self.src_name is None:
            self.src_name = np.array([], dtype="U256")
        else:
            self.src_name = np.asarray(self.src_name).astype("U256").reshape(-1)
        self.reproj = np.zeros((0,), np.float32) if self.reproj is None else np.asarray(self.reproj, np.float32).reshape(-1)
        n = len(self.xyz)
        for name, arr in (("rgb", self.rgb), ("src", self.src), ("src_name", self.src_name), ("reproj", self.reproj)):
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != points {n}")


@dataclass
class UpdateResult:
    accepted: bool
    reason: str
    added_points: int = 0
    skipped_frames: list[str] = field(default_factory=list)
    closure: str = "not_requested"
    alignment_rmse: float | None = None
    return_center_distance: float | None = None
    verified_inliers: int | None = None
    group_id: str | None = None

    def to_json(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "added_points": int(self.added_points),
            "skipped_frames": list(self.skipped_frames),
            "closure": self.closure,
            "alignment_rmse": self.alignment_rmse,
            "return_center_distance": self.return_center_distance,
            "verified_inliers": self.verified_inliers,
            "group_id": self.group_id,
        }
