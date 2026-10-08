"""Conditioning stays an image reference. Evaluation does not invent closure or objects."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from support import POSE, camera, reconstruction

from scene_memory.conditioning import ablation, generation_condition, write_memory_view
from scene_memory.evaluate import object_persistence, return_report, spatial_report
from scene_memory.memory import SceneMemory


def _two_view(inliers):
    cam_a = camera("a.png", [0.0, 0.0, 0.0], 0)
    cam_b = camera("b.png", [0.5, 0.0, 0.0], 1)
    pairs = {}
    if inliers:
        pairs[("a.png", "b.png")] = inliers
    recon = reconstruction(
        [cam_a, cam_b],
        [[0.0, 0.0, 5.0]],
        [[1.0, 0.0, 0.0]],
        ["a.png"],
        reproj=[0.2],
        pair_inliers=pairs,
        input_image_count=2,
    )
    memory = SceneMemory()
    memory._add_reconstruction("room", recon, np.eye(4))
    return memory


class ConditioningTests(unittest.TestCase):
    def test_memory_view_is_one_extra_reference_and_holes_stay_black(self):
        recon = reconstruction(
            [],
            [[0.0, 0.0, 5.0]],
            [[1.0, 0.0, 0.0]],
            ["bed.png"],
            input_image_count=1,
        )
        memory = SceneMemory()
        memory._add_reconstruction("room", recon, np.eye(4))
        with tempfile.TemporaryDirectory() as tmp:
            written = write_memory_view(memory, POSE, tmp)
            self.assertFalse(written["changes_weights"])
            pair = ablation("The camera returns to the bedroom.", Path("ref.png"), written["rgb"], written["mask"])
        base, prop = pair["baseline"], pair["proposed"]
        self.assertEqual(base["references"], [("image", "ref.png")])
        self.assertEqual(prop["references"][0], base["references"][0])
        self.assertEqual(len(prop["references"]), 2)
        self.assertTrue(prop["uses_memory_view"])
        self.assertFalse(prop["changes_weights"])
        self.assertFalse(base["changes_weights"])
        self.assertNotIn("memory_mask", " ".join(prop["argv"]))
        self.assertIn("<Picture 2>", prop["prompt"])
        self.assertNotIn("<Picture 2>", base["prompt"])
        self.assertNotIn("--model-path", prop["argv"])

    def test_reference_cap(self):
        extras = [("image", f"{i}.png") for i in range(9)]
        with self.assertRaises(ValueError):
            generation_condition("prompt", "ref.png", extra_references=extras)


class EvalTests(unittest.TestCase):
    def test_overlap_without_matches_is_not_closure(self):
        silent = return_report(_two_view(0), "a.png", "b.png")
        self.assertGreater(silent["geometric_overlap"], 0.9)
        self.assertEqual(silent["verified_inliers"], 0)
        self.assertFalse(silent["closure_supported"])
        self.assertEqual(silent["reason"], "no_verified_matches")
        linked = return_report(_two_view(12), "a.png", "b.png")
        self.assertTrue(linked["closure_supported"])
        self.assertEqual(linked["verified_inliers"], 12)
        missing = return_report(_two_view(12), "a.png", "frame_000728.png")
        self.assertFalse(missing["registered_b"])
        self.assertFalse(missing["closure_supported"])
        self.assertEqual(missing["reason"], "not_registered")

    def test_object_spread_and_unobserved_clicks(self):
        memory = _two_view(12)
        report = object_persistence(
            memory,
            {
                "table": [
                    {"camera": "a.png", "u": 16, "v": 12},
                    {"camera": "b.png", "u": 11, "v": 12},
                ],
                "missing": [{"camera": "a.png", "u": 0, "v": 0}],
            },
        )
        by_name = {item["name"]: item for item in report["objects"]}
        self.assertLess(by_name["table"]["position_spread"], 1e-5)
        self.assertEqual(by_name["missing"]["unobserved"], 1)
        self.assertIsNone(by_name["missing"]["position_spread"])
        self.assertEqual(object_persistence(memory, None)["status"], "not_provided")

    def test_spatial_report_counts_registration_and_a_straight_path(self):
        cams = [camera(f"{i}.png", [float(i), 0.0, 0.0], i) for i in range(3)]
        recon = reconstruction(cams, [[0, 0, 5]], [[1, 0, 0]], ["0.png"], reproj=[0.4], input_image_count=4)
        recon.cameras.append(
            camera("skip.png", [9, 0, 0], 9)
        )
        recon.cameras[-1].registered = False
        recon.cameras[-1].R_w2c = None
        memory = SceneMemory()
        memory._add_reconstruction("room", recon, np.eye(4))
        report = spatial_report(memory)
        self.assertEqual(report["registered"], 3)
        self.assertEqual(report["input_images"], 4)
        self.assertAlmostEqual(report["registration_ratio"], 0.75)
        self.assertAlmostEqual(report["mean_reproj_error"], 0.4)
        self.assertAlmostEqual(report["trajectory"]["step_mean"], 1.0, places=6)
        self.assertAlmostEqual(report["trajectory"]["turn_mean_deg"], 0.0, places=5)
        self.assertFalse(report["nominal_poses_used_in_mapping"])
        self.assertGreater(report["source_view_fill_median"], 0.0)
        self.assertLess(report["source_view_fill_median"], 0.2)


if __name__ == "__main__":
    unittest.main()
