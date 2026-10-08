"""CPU COLMAP backend.

Matches the sparse-gate settings: SIMPLE_RADIAL, one shared camera, CPU SIFT,
exhaustive matching, principal point not refined. ``camera_params`` is not passed.
``pose_prior_mapper`` is not used. Nominal waypoint poses are not read.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from scene_memory.geometry import qvec_to_R
from scene_memory.types import Camera, Reconstruction


DEFAULT_COLMAP = "colmap"
PAIR_STRIDE = 2147483647
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def frame_index_from_name(name: str) -> int | None:
    stem = Path(name).stem
    tail = stem.split("_")[-1]
    if tail.isdigit():
        return int(tail)
    return None


class ColmapBackend:
    name = "colmap"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("COLMAP_BIN", DEFAULT_COLMAP)

    def feature_extractor_argv(self, database: Path, image_path: Path) -> list[str]:
        argv = [
            self.binary,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_path),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
            "--FeatureExtraction.use_gpu",
            "0",
        ]
        _reject_priors(argv)
        return argv

    def matcher_argv(self, database: Path) -> list[str]:
        argv = [
            self.binary,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            "0",
        ]
        _reject_priors(argv)
        return argv

    def mapper_argv(self, database: Path, image_path: Path, output_path: Path) -> list[str]:
        argv = [
            self.binary,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_path),
            "--output_path",
            str(output_path),
            "--Mapper.ba_refine_principal_point",
            "0",
        ]
        _reject_priors(argv)
        return argv

    def reconstruct(self, image_paths: list[Path], work_dir: Path | None) -> Reconstruction:
        if work_dir is None:
            raise ValueError("ColmapBackend requires a work directory")
        work = Path(work_dir)
        if work.exists():
            shutil.rmtree(work)
        image_dir = work / "images"
        image_dir.mkdir(parents=True)
        staged = _stage_images(image_paths, image_dir)
        database = work / "database.db"
        sparse = work / "sparse"
        sparse.mkdir()
        log_dir = work / "logs"
        log_dir.mkdir()
        self._run(self.feature_extractor_argv(database, image_dir), log_dir / "feature_extractor.log")
        self._run(self.matcher_argv(database), log_dir / "exhaustive_matcher.log")
        self._run(self.mapper_argv(database, image_dir, sparse), log_dir / "mapper.log")
        model = _largest_model(sparse)
        if model is None:
            return _empty_reconstruction(staged, database)
        txt = work / "sparse_txt"
        txt.mkdir()
        convert = [
            self.binary,
            "model_converter",
            "--input_path",
            str(model),
            "--output_path",
            str(txt),
            "--output_type",
            "TXT",
        ]
        _reject_priors(convert)
        self._run(convert, log_dir / "model_converter.log")
        return reconstruction_from_text(txt, image_dir=image_dir, database=database, image_paths=staged)

    def _run(self, argv: list[str], log_path: Path) -> None:
        _reject_priors(argv)
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"COLMAP failed ({argv[1]}, code {proc.returncode}):\n{tail}")


def reconstruction_from_text(
    model_dir: Path,
    *,
    image_dir: Path | None = None,
    database: Path | None = None,
    image_paths: list[Path] | None = None,
) -> Reconstruction:
    model_dir = Path(model_dir)
    cam = _parse_camera(model_dir / "cameras.txt")
    poses = _parse_images(model_dir / "images.txt")
    xyz, rgb, err, src_ids = _parse_points(model_dir / "points3D.txt")
    id_to_name = {item["image_id"]: item["name"] for item in poses}
    known = {item["name"] for item in poses}
    width = int(cam["width"])
    height = int(cam["height"])
    focal_prior = 1.2 * max(width, height)
    cameras = []
    for item in poses:
        cameras.append(
            Camera(
                image_name=item["name"],
                image_path=_path_for(item["name"], image_dir, image_paths),
                registered=True,
                width=width,
                height=height,
                R_w2c=item["R"],
                t_w2c=item["t"],
                model=cam["model"],
                fx=cam["fx"],
                fy=cam["fy"],
                cx=cam["cx"],
                cy=cam["cy"],
                k=cam["k"],
                intrinsics_refined=True,
                principal_point_refined=False,
                frame_index=frame_index_from_name(item["name"]),
                reconstructor="colmap",
            )
        )
    input_names = _input_names(image_dir, image_paths)
    for name in input_names:
        if name in known:
            continue
        cameras.append(
            Camera(
                image_name=name,
                image_path=_path_for(name, image_dir, image_paths),
                registered=False,
                width=width,
                height=height,
                model=cam["model"],
                fx=cam["fx"],
                fy=cam["fy"],
                cx=cam["cx"],
                cy=cam["cy"],
                k=cam["k"],
                intrinsics_refined=True,
                principal_point_refined=False,
                frame_index=frame_index_from_name(name),
                reconstructor="colmap",
            )
        )
    src_name = np.array([id_to_name.get(int(i), "") for i in src_ids], dtype="U256")
    src = np.array(
        [(-1 if frame_index_from_name(n) is None else int(frame_index_from_name(n))) for n in src_name.tolist()],
        dtype=np.int32,
    )
    rgb_f = (rgb.astype(np.float32) / 255.0) if len(rgb) else np.zeros((0, 3), np.float32)
    frame = src.astype(np.float32).reshape(-1, 1) if len(src) else np.zeros((0, 1), np.float32)
    if len(rgb_f):
        features = np.concatenate([rgb_f, frame], axis=1).astype(np.float32)
    else:
        features = np.zeros((0, 4), np.float32)
    mean = float(err.mean()) if len(err) else None
    median = float(np.median(err)) if len(err) else None
    pairs = {}
    priors = 0
    if database is not None and Path(database).is_file():
        pairs = read_pair_inliers(Path(database))
        priors = pose_prior_count(Path(database))
        if priors:
            raise RuntimeError(f"COLMAP database {database} contains {priors} pose priors")
    return Reconstruction(
        reconstructor="colmap",
        cameras=cameras,
        xyz=xyz.astype(np.float32),
        rgb=rgb_f,
        features=features,
        src=src,
        src_name=src_name,
        reproj=err.astype(np.float32),
        mean_reproj_error=mean,
        median_reproj_error=median,
        pair_inliers=pairs,
        pose_prior_count=priors,
        input_image_count=len(input_names) if input_names else len(cameras),
        focal_prior_px=focal_prior,
    )


def read_pair_inliers(database: Path) -> dict[tuple[str, str], int]:
    con = sqlite3.connect(str(database))
    try:
        names = {int(i): n for i, n in con.execute("SELECT image_id, name FROM images")}
        pairs = {}
        for pair_id, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
            id1, id2 = decode_pair_id(int(pair_id))
            a = names.get(id1)
            b = names.get(id2)
            if not a or not b:
                continue
            key = (a, b) if a <= b else (b, a)
            pairs[key] = int(rows)
        return pairs
    finally:
        con.close()


def pose_prior_count(database: Path) -> int:
    con = sqlite3.connect(str(database))
    try:
        return int(con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0])
    finally:
        con.close()


def decode_pair_id(pair_id: int) -> tuple[int, int]:
    return pair_id // PAIR_STRIDE, pair_id % PAIR_STRIDE


def _parse_camera(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = list(map(float, parts[4:]))
        if model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy, k = params[1], params[2], params[3]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy, k = params[1], params[2], 0.0
        elif model == "PINHOLE":
            fx, fy, cx, cy, k = params[0], params[1], params[2], params[3], 0.0
        elif model == "RADIAL":
            fx, fy, cx, cy, k = params[0], params[1], params[2], params[3], params[4]
        else:
            raise RuntimeError(f"unsupported COLMAP camera model {model}")
        return {"model": model, "width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "k": k}
    raise RuntimeError(f"no camera in {path}")


def _parse_images(path: Path) -> list[dict]:
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or not Path(parts[-1]).suffix.lower() in IMAGE_SUFFIXES:
            continue
        R = qvec_to_R(list(map(float, parts[1:5])))
        t = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        poses.append({"image_id": int(parts[0]), "name": parts[9], "R": R, "t": t})
    return poses


def _parse_points(path: Path):
    if not path.is_file():
        return (
            np.zeros((0, 3), np.float64),
            np.zeros((0, 3), np.float64),
            np.zeros((0,), np.float64),
            np.zeros((0,), np.int32),
        )
    xyz, rgb, err, src = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyz.append(list(map(float, parts[1:4])))
        rgb.append(list(map(float, parts[4:7])))
        err.append(float(parts[7]))
        src.append(int(parts[8]) if len(parts) > 8 else -1)
    if not xyz:
        return (
            np.zeros((0, 3), np.float64),
            np.zeros((0, 3), np.float64),
            np.zeros((0,), np.float64),
            np.zeros((0,), np.int32),
        )
    return (
        np.asarray(xyz, np.float64),
        np.asarray(rgb, np.float64),
        np.asarray(err, np.float64),
        np.asarray(src, np.int32),
    )


def _largest_model(sparse: Path) -> Path | None:
    dirs = [p for p in sparse.iterdir() if p.is_dir()]
    if not dirs:
        return None
    def score(path: Path) -> int:
        images = path / "images.bin"
        if images.is_file():
            return images.stat().st_size
        txt = path / "images.txt"
        return txt.stat().st_size if txt.is_file() else 0
    return max(dirs, key=score)


def _stage_images(paths: list[Path], image_dir: Path) -> list[Path]:
    staged = []
    seen = set()
    for raw in paths:
        src = Path(raw).resolve()
        if src.name in seen:
            raise ValueError(f"duplicate image name {src.name}")
        seen.add(src.name)
        dest = image_dir / src.name
        dest.symlink_to(src)
        staged.append(dest)
    return staged


def _input_names(image_dir: Path | None, image_paths: list[Path] | None) -> list[str]:
    if image_paths:
        return [Path(p).name for p in image_paths]
    if image_dir is None or not image_dir.is_dir():
        return []
    return sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _path_for(name: str, image_dir: Path | None, image_paths: list[Path] | None) -> str | None:
    if image_paths:
        for path in image_paths:
            if Path(path).name == name:
                return str(path)
    if image_dir is not None and (image_dir / name).exists():
        return str(image_dir / name)
    return None


def _empty_reconstruction(staged: list[Path], database: Path) -> Reconstruction:
    priors = pose_prior_count(database) if database.is_file() else 0
    if priors:
        raise RuntimeError(f"COLMAP database {database} contains {priors} pose priors")
    cameras = [
        Camera(
            image_name=path.name,
            image_path=str(path),
            registered=False,
            width=0,
            height=0,
            reconstructor="colmap",
            frame_index=frame_index_from_name(path.name),
        )
        for path in staged
    ]
    return Reconstruction(
        reconstructor="colmap",
        cameras=cameras,
        xyz=np.zeros((0, 3), np.float32),
        rgb=np.zeros((0, 3), np.float32),
        features=np.zeros((0, 4), np.float32),
        src=np.zeros((0,), np.int32),
        src_name=np.array([], dtype="U256"),
        reproj=np.zeros((0,), np.float32),
        mean_reproj_error=None,
        median_reproj_error=None,
        pair_inliers=read_pair_inliers(database) if database.is_file() else {},
        pose_prior_count=0,
        input_image_count=len(staged),
        focal_prior_px=None,
    )


def _reject_priors(argv: list[str]) -> None:
    for token in argv:
        lowered = token.lower()
        if "camera_params" in lowered or "pose_prior" in lowered:
            raise RuntimeError(f"refusing COLMAP argument that injects a calibration or pose prior: {token}")
