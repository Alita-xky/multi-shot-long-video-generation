"""Chunk update accepts only reconstructed overlap and refuses nominal poses."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from support import camera, reconstruction

from scene_memory.memory import ChunkEvidence, SceneMemory


CENTERS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
NAMES = ["c0.png", "c1.png", "c2.png", "c3.png"]


def _memory():
    cams = [camera(name, center, idx, path=f"/data/{name}") for idx, (name, center) in enumerate(zip(NAMES, CENTERS))]
    recon = reconstruction(
        cams,
        [[0.0, 0.0, 5.0]],
        [[1.0, 0.0, 0.0]],
        ["c0.png"],
        input_image_count=4,
    )
    memory = SceneMemory()
    memory._add_reconstruction("default", recon, np.eye(4))
    return memory


def _joint(pair_inliers, *, shift_first=0.0, include_new=True, extra_unregistered=False):
    cams = [camera(name, center, idx, path=f"/data/{name}") for idx, (name, center) in enumerate(zip(NAMES, CENTERS))]
    if shift_first:
        cams[0] = camera(NAMES[0], [shift_first, 0.0, 0.0], 0, path=f"/data/{NAMES[0]}")
    new_names = ["cnew.png"]
    if include_new:
        cams.append(camera("cnew.png", [2.0, 0.0, 0.0], 10, path="/data/cnew.png"))
    if extra_unregistered:
        new_names.append("cout.png")
    recon = reconstruction(
        cams,
        [[2.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0]],
        ["cnew.png"],
        pair_inliers=pair_inliers,
        input_image_count=len(cams) + (1 if extra_unregistered else 0),
    )
    return recon, new_names


def _evidence(recon, new_names, *, return_frame=None, anchor_frame=None, source="reconstruction"):
    return ChunkEvidence(
        source=source,
        reconstructor="test",
        reconstruction=recon,
        overlap_names=list(NAMES),
        new_names=list(new_names),
        group_id="default",
        return_frame=return_frame,
        anchor_frame=anchor_frame,
    )


class UpdateTests(unittest.TestCase):
    def test_nominal_pose_is_rejected_and_memory_stays(self):
        memory = _memory()
        before = np.array(memory.groups["default"].xyz, copy=True)
        recon, names = _joint({("c0.png", "cnew.png"): 40})
        result = memory.update(
            ["/data/cnew.png"],
            {"source": "nominal", "R": np.eye(3), "t": np.zeros(3)},
            evidence=_evidence(recon, names),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "nominal_pose_rejected")
        np.testing.assert_array_equal(memory.groups["default"].xyz, before)
        self.assertEqual(memory.update_log[-1]["reason"], "nominal_pose_rejected")

    def test_pose_alone_is_not_geometry(self):
        memory = _memory()
        before = np.array(memory.groups["default"].xyz, copy=True)
        result = memory.update(["/data/cnew.png"], {"R": np.eye(3), "t": np.array([3.0, 0.0, 0.0])})
        self.assertEqual(result.reason, "missing_reconstruction_evidence")
        np.testing.assert_array_equal(memory.groups["default"].xyz, before)

    def test_verified_overlap_appends_only_the_new_points(self):
        memory = _memory()
        recon, names = _joint({("c0.png", "cnew.png"): 40}, extra_unregistered=True)
        garbage = {"R": np.eye(3) * 5.0, "t": np.array([9.0, 9.0, 9.0])}
        result = memory.update(["/data/cnew.png", "/data/cout.png"], garbage, evidence=_evidence(recon, names))
        self.assertTrue(result.accepted)
        self.assertEqual(result.added_points, 1)
        self.assertIn("cout.png", result.skipped_frames)
        xyz = memory.groups["default"].xyz
        self.assertEqual(len(xyz), 2)
        np.testing.assert_allclose(xyz[0], [0, 0, 5], atol=1e-5)
        np.testing.assert_allclose(xyz[1], [2, 0, 1], atol=1e-5)
        self.assertIsNotNone(memory.groups["default"].camera("cnew.png"))

    def test_alignment_failure_keeps_the_old_cloud(self):
        memory = _memory()
        before = np.array(memory.groups["default"].xyz, copy=True)
        recon, names = _joint({("c0.png", "cnew.png"): 40}, shift_first=5.0)
        result = memory.update(["/data/cnew.png"], evidence=_evidence(recon, names))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "alignment_failed")
        np.testing.assert_array_equal(memory.groups["default"].xyz, before)
        self.assertIsNone(memory.groups["default"].camera("cnew.png"))

    def test_return_frame_with_no_matches_rejects_closure(self):
        memory = _memory()
        before = np.array(memory.groups["default"].xyz, copy=True)
        recon, names = _joint({("c3.png", "cnew.png"): 100, ("c0.png", "cnew.png"): 0})
        result = memory.update(
            ["/data/cnew.png"],
            evidence=_evidence(recon, names, return_frame="cnew.png", anchor_frame="c0.png"),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "closure_rejected_no_verified_matches")
        self.assertEqual(result.closure, "rejected")
        np.testing.assert_array_equal(memory.groups["default"].xyz, before)

    def test_return_frame_with_matches_is_supported_not_invented(self):
        memory = _memory()
        recon, names = _joint({("c0.png", "cnew.png"): 40})
        result = memory.update(
            ["/data/cnew.png"],
            evidence=_evidence(recon, names, return_frame="cnew.png", anchor_frame="c0.png"),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.closure, "supported")
        self.assertEqual(result.verified_inliers, 40)
        self.assertGreater(result.return_center_distance, 1.0)

    def test_backend_is_not_called_without_enough_overlap(self):
        cams = [camera(NAMES[i], CENTERS[i], i, path=f"/data/{NAMES[i]}") for i in range(2)]
        recon = reconstruction(cams, [[0, 0, 5]], [[1, 0, 0]], ["c0.png"], input_image_count=2)
        memory = SceneMemory()
        memory._add_reconstruction("default", recon, np.eye(4))

        class Bomb:
            name = "test"
            calls = 0

            def reconstruct(self, paths, work_dir):
                self.calls += 1
                raise AssertionError("backend should not run")

        bomb = Bomb()
        result = memory.update(["/data/cnew.png"], backend=bomb, work_dir=Path("."))
        self.assertEqual(result.reason, "insufficient_overlap")
        self.assertEqual(bomb.calls, 0)

    def test_waypoint_pose_does_not_call_the_backend(self):
        memory = _memory()

        class Bomb:
            name = "test"

            def reconstruct(self, paths, work_dir):
                raise AssertionError("nominal pose must not reach reconstruction")

        result = memory.update(
            ["/data/cnew.png"],
            {"source": "waypoint", "R": np.eye(3), "t": np.zeros(3)},
            backend=Bomb(),
            work_dir=Path("."),
        )
        self.assertEqual(result.reason, "nominal_pose_rejected")
        self.assertEqual(len(memory.groups["default"].xyz), 1)

    def test_backend_merges_a_reconstructed_chunk(self):
        memory = _memory()
        recon, _names = _joint({("c0.png", "cnew.png"): 40})

        class Once:
            name = "test"

            def reconstruct(self, paths, work_dir):
                got = {Path(p).name for p in paths}
                if got != set(NAMES) | {"cnew.png"}:
                    raise AssertionError(got)
                return recon

        result = memory.update(["/data/cnew.png"], backend=Once(), work_dir=Path("."))
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "merged")
        self.assertEqual(len(memory.groups["default"].xyz), 2)
        np.testing.assert_allclose(memory.groups["default"].xyz[1], [2, 0, 1], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
