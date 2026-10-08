"""SceneMemory data structure, render, and synthetic geometry. No COLMAP process."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from support import POSE, camera, reconstruction

from scene_memory.backends.colmap_backend import (
    ColmapBackend,
    _reject_priors,
    decode_pair_id,
    reconstruction_from_text,
)
from scene_memory.backends.unavailable import VGGT
from scene_memory.geometry import project, qvec_to_R, sim3_points, umeyama, unproject
from scene_memory.memory import SceneMemory


PREVIOUS = Path(__file__).resolve().parents[1] / "evidence" / "colmap_sparse_gate"


class FixedBackend:
    def __init__(self, recon):
        self.recon = recon
        self.calls = []
        self.name = "test"

    def reconstruct(self, paths, work_dir):
        self.calls.append([Path(p).name for p in paths])
        return self.recon


class ColorBackend:
    def __init__(self):
        self.calls = []
        self.name = "test"

    def reconstruct(self, paths, work_dir):
        names = [Path(p).name for p in paths]
        self.calls.append(names)
        red = names[0].startswith("bed")
        color = [1.0, 0.0, 0.0] if red else [0.0, 1.0, 0.0]
        return reconstruction([], [[0.0, 0.0, 5.0]], [color], [names[0]], input_image_count=1)


class GeometryTests(unittest.TestCase):
    def test_project_unproject_roundtrip(self):
        X = np.array([[0.2, -0.3, 4.0], [1.0, 0.5, 6.0], [-0.4, 0.2, 3.5]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)
        for k in (0.0, -0.015):
            u, v, z = project(X, R, t, 50, 50, 16, 12, k)
            back = unproject(u, v, z, R, t, 50, 50, 16, 12, k)
            np.testing.assert_allclose(back, X, atol=1e-8 if k == 0 else 1e-6)

    def test_identity_quaternion(self):
        np.testing.assert_allclose(qvec_to_R([1, 0, 0, 0]), np.eye(3), atol=1e-12)

    def test_umeyama_recovers_sim3(self):
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3], [0.2, 0.1, 1.0], [0.4, 0.6, 0.2]])
        angle = np.radians(20.0)
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        dst = sim3_points(src, 1.7, R, [0.3, -0.2, 0.5])
        s, Rest, t = umeyama(src, dst)
        np.testing.assert_allclose(sim3_points(src, s, Rest, t), dst, atol=1e-8)


class MemoryTests(unittest.TestCase):
    def test_empty_query_is_an_empty_mask(self):
        view = SceneMemory().query(POSE)
        self.assertEqual(view.valid_pixels, 0)
        self.assertFalse(view.mask.any())
        self.assertEqual(int(view.rgb.max()), 0)
        self.assertEqual(view.hit_image_names, [])

    def test_query_refuses_a_guessed_fov(self):
        memory = SceneMemory()
        with self.assertRaises(ValueError):
            memory.query({"R": np.eye(3), "t": np.zeros(3), "width": 32, "height": 24})

    def test_render_keeps_holes_and_nearer_point(self):
        recon = reconstruction(
            [],
            [[0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.4, 0.0, 5.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ["near.png", "far.png", "side.png"],
        )
        memory = SceneMemory()
        memory._add_reconstruction("room", recon, np.eye(4))
        view = memory.query(POSE)
        self.assertEqual(view.rgb[12, 16].tolist(), [255, 0, 0])
        self.assertAlmostEqual(float(view.depth[12, 16]), 5.0, places=5)
        self.assertEqual(view.rgb[12, 20].tolist(), [0, 0, 255])
        self.assertFalse(view.mask[12, 18])
        self.assertEqual(view.rgb[12, 18].tolist(), [0, 0, 0])
        back = unproject([16], [12], [view.depth[12, 16]], POSE["R"], POSE["t"], 50, 50, 16, 12, 0)
        np.testing.assert_allclose(back, [[0, 0, 5]], atol=1e-6)
        self.assertIn("near.png", view.hit_image_names)
        self.assertNotIn("far.png", view.hit_image_names)

    def test_disjoint_groups_are_not_registered_together(self):
        backend = ColorBackend()
        memory = SceneMemory()
        memory.build({"bedroom": ["bed.png"], "kitchen": ["kit.png"]}, backend)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0], ["bed.png"])
        self.assertEqual(backend.calls[1], ["kit.png"])
        self.assertEqual(memory.query(POSE).valid_pixels, 0)
        memory.set_placement("bedroom", np.eye(4))
        view = memory.query(POSE)
        self.assertEqual(view.rgb[12, 16].tolist(), [255, 0, 0])
        self.assertNotIn("kit.png", view.hit_image_names)
        placed = np.eye(4)
        placed[0, 3] = 0.2
        memory.set_placement("bedroom", placed)
        shifted = memory.query(POSE)
        self.assertFalse(shifted.mask[12, 16])
        self.assertTrue(shifted.mask[12, 18])

    def test_save_load_roundtrip(self):
        recon = reconstruction(
            [camera("a.png", [0, 0, 0], 0)],
            [[0.0, 0.0, 5.0]],
            [[0.2, 0.4, 0.6]],
            ["a.png"],
            pair_inliers={("a.png", "b.png"): 4},
        )
        memory = SceneMemory()
        memory._add_reconstruction("room", recon, np.eye(4))
        memory.update_log.append({"accepted": False, "reason": "example"})
        with tempfile.TemporaryDirectory() as tmp:
            memory.save(tmp)
            loaded = SceneMemory.load(tmp)
        view = loaded.query(POSE)
        self.assertEqual(view.rgb[12, 16].tolist(), [51, 102, 153])
        self.assertEqual(loaded.update_log[0]["reason"], "example")
        self.assertEqual(loaded.groups["room"].pair_inliers[("a.png", "b.png")], 4)
        meta = Path(tmp)
        self.assertFalse(meta.exists())

    def test_pose_priors_are_rejected_at_build(self):
        recon = reconstruction([], [[0, 0, 5]], [[1, 0, 0]], ["a.png"], pose_prior_count=2, input_image_count=1)
        with self.assertRaises(RuntimeError):
            SceneMemory().build(["a.png"], FixedBackend(recon))


class BackendContractTests(unittest.TestCase):
    def test_colmap_commands_carry_no_prior(self):
        backend = ColmapBackend(binary="colmap")
        commands = [
            backend.feature_extractor_argv(Path("db.db"), Path("images")),
            backend.matcher_argv(Path("db.db")),
            backend.mapper_argv(Path("db.db"), Path("images"), Path("sparse")),
        ]
        for argv in commands:
            text = " ".join(argv)
            self.assertNotIn("camera_params", text)
            self.assertNotIn("pose_prior", text)
            self.assertNotIn("pose_prior_mapper", text)
        extract = commands[0]
        self.assertEqual(extract[extract.index("--FeatureExtraction.use_gpu") + 1], "0")
        self.assertIn("SIMPLE_RADIAL", extract)
        self.assertEqual(extract[extract.index("--ImageReader.single_camera") + 1], "1")
        matcher = commands[1]
        self.assertEqual(matcher[matcher.index("--FeatureMatching.use_gpu") + 1], "0")
        mapper = commands[2]
        self.assertEqual(mapper[mapper.index("--Mapper.ba_refine_principal_point") + 1], "0")
        self.assertEqual(mapper[1], "mapper")
        with self.assertRaises(RuntimeError):
            _reject_priors(["colmap", "feature_extractor", "--ImageReader.camera_params", "1000,480,272"])

    def test_feedforward_backends_are_not_runnable(self):
        with self.assertRaises(RuntimeError) as ctx:
            VGGT.reconstruct([], None)
        self.assertIn("ColmapBackend", str(ctx.exception))

    def test_pair_id_matches_the_previous_database(self):
        self.assertEqual(decode_pair_id(2147483650), (1, 3))

    @unittest.skipUnless((PREVIOUS / "sparse" / "0_txt" / "cameras.txt").is_file(), "previous sparse model is absent")
    def test_parse_previous_sparse_model(self):
        recon = reconstruction_from_text(PREVIOUS / "sparse" / "0_txt", database=PREVIOUS / "workspace" / "database.db")
        names = {cam.image_name for cam in recon.cameras if cam.registered}
        self.assertIn("frame_000000.png", names)
        self.assertNotIn("frame_000728.png", names)
        self.assertEqual(len(names), 15)
        self.assertEqual(len(recon.xyz), 831)
        focal = next(cam.fx for cam in recon.cameras if cam.registered)
        self.assertGreater(focal, 640)
        self.assertLess(focal, 660)
        self.assertLess(recon.mean_reproj_error, 1.0)
        self.assertEqual(recon.pose_prior_count, 0)
        self.assertFalse(any(cam.principal_point_refined for cam in recon.cameras))
        self.assertTrue(all(cam.intrinsics_refined for cam in recon.cameras if cam.registered))
        self.assertEqual(recon.pair_inliers[("frame_000000.png", "frame_000024.png")], 312)
        self.assertEqual(recon.features.shape, (831, 4))


class PackageBoundaryTests(unittest.TestCase):
    def test_sources_do_not_import_the_failed_pseudo_geometry_or_the_generator(self):
        root = Path(__file__).resolve().parents[1] / "scene_memory"
        for path in root.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip().lower()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                self.assertNotIn("spatial_memory", stripped, path.name)
                self.assertNotIn("depth_anything", stripped, path.name)
                self.assertNotIn("torch", stripped, path.name)
                self.assertNotIn("minimax", stripped, path.name)


if __name__ == "__main__":
    unittest.main()
