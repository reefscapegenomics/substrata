"""Tests for correspondence-based alignment (``substrata align --cameras``).

``Transform.from_point_correspondences`` is tested by loading ``geom.py``
directly (numpy only). ``Cameras.match_by_name`` needs ``cameras.py``, which
imports open3d/cv2, so those tests are skipped outside the conda environment.
"""

# Standard Library
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

# Third-Party
import numpy as np

_GEOM = Path(__file__).resolve().parents[1] / "src" / "substrata" / "geom.py"
_spec = importlib.util.spec_from_file_location("_geom_for_align_tests", _GEOM)
geom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geom)

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import cameras
except Exception:  # noqa: BLE001 - any import failure -> skip those tests.
    cameras = None


def _similarity(scale=2.5, seed=0):
    """Random proper similarity transform."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.eye(4)
    t[:3, :3] = scale * q
    t[:3, 3] = rng.normal(size=3) * 10
    return t


def _apply(t, pts):
    return pts @ t[:3, :3].T + t[:3, 3]


class TestFromPointCorrespondences(unittest.TestCase):
    def setUp(self):
        self.pts = np.random.default_rng(1).normal(size=(6, 3))

    def test_recovers_similarity(self):
        t = _similarity()
        est = geom.Transform.from_point_correspondences(self.pts, _apply(t, self.pts))
        np.testing.assert_allclose(est.mat, t, atol=1e-9)

    def test_recovers_rigid_without_scaling(self):
        t = _similarity(scale=1.0)
        est = geom.Transform.from_point_correspondences(
            self.pts, _apply(t, self.pts), with_scaling=False
        )
        np.testing.assert_allclose(est.mat, t, atol=1e-9)

    def test_no_scaling_keeps_unit_scale(self):
        t = _similarity(scale=3.0)
        est = geom.Transform.from_point_correspondences(
            self.pts, _apply(t, self.pts), with_scaling=False
        )
        self.assertAlmostEqual(np.linalg.det(est.mat[:3, :3]), 1.0, places=9)

    def test_minimal_three_points_with_noise(self):
        t = _similarity(scale=0.3, seed=4)
        rng = np.random.default_rng(5)
        pts = rng.normal(size=(20, 3))
        dst = _apply(t, pts) + rng.normal(scale=1e-3, size=pts.shape)
        est = geom.Transform.from_point_correspondences(pts, dst)
        np.testing.assert_allclose(est.mat, t, atol=5e-3)
        est3 = geom.Transform.from_point_correspondences(pts[:3], _apply(t, pts[:3]))
        np.testing.assert_allclose(est3.mat, t, atol=1e-8)

    def test_mirrored_input_gives_proper_rotation(self):
        mirrored = self.pts * np.array([1.0, 1.0, -1.0])
        est = geom.Transform.from_point_correspondences(self.pts, mirrored)
        self.assertGreater(np.linalg.det(est.mat[:3, :3]), 0)

    def test_degenerate_inputs_raise(self):
        with self.assertRaises(ValueError):
            geom.Transform.from_point_correspondences(self.pts[:2], self.pts[:2])
        line = np.outer(np.arange(5.0), [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            geom.Transform.from_point_correspondences(line, line)
        with self.assertRaises(ValueError):
            geom.Transform.from_point_correspondences(self.pts, self.pts[:4])

    def test_compose_with_target_world_transform(self):
        # raw source -> raw target, then target world_transform on top
        t_raw = _similarity(seed=2)
        w_t = _similarity(scale=0.2, seed=3)
        est = geom.Transform.from_point_correspondences(
            self.pts, _apply(t_raw, self.pts)
        )
        t_final = w_t @ est.mat
        np.testing.assert_allclose(
            _apply(t_final, self.pts), _apply(w_t, _apply(t_raw, self.pts)), atol=1e-8
        )


class TestRejectingOutliers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.t = _similarity(scale=0.7, seed=8)
        self.pts = rng.uniform(-10, 10, size=(100, 3))
        self.dst = _apply(self.t, self.pts) + rng.normal(scale=0.01, size=(100, 3))
        # Two blocks of cameras displaced as a whole, as with a misplaced run
        self.dst[:15] += [3.0, 0.0, 0.0]
        self.dst[15:25] += [0.0, -2.5, 1.0]

    def test_auto_threshold_drops_displaced_blocks(self):
        est, inliers = geom.fit_correspondences_rejecting_outliers(
            self.pts, self.dst
        )
        np.testing.assert_array_equal(inliers[:25], False)
        self.assertTrue(inliers[25:].all())
        np.testing.assert_allclose(est.mat, self.t, atol=0.02)

    def test_fixed_threshold(self):
        _, inliers = geom.fit_correspondences_rejecting_outliers(
            self.pts, self.dst, max_residual=0.5
        )
        self.assertEqual(int(inliers.sum()), 75)

    def test_clean_data_keeps_nearly_all(self):
        clean = _apply(self.t, self.pts)
        est, inliers = geom.fit_correspondences_rejecting_outliers(self.pts, clean)
        self.assertGreaterEqual(int(inliers.sum()), 3)
        np.testing.assert_allclose(est.mat, self.t, atol=1e-8)


def _write_meta(dirpath, name, cams):
    """Write a minimal cameras meta JSON; cams maps id -> (path, center|None)."""
    data = {"cameras": {}}
    for cid, (path, center) in cams.items():
        data["cameras"][cid] = {
            "path": path,
            "center": center,
            "transform": np.eye(4).ravel().tolist() if center is not None else None,
        }
    fp = os.path.join(dirpath, name)
    with open(fp, "w") as f:
        json.dump(data, f)
    return fp


@unittest.skipUnless(cameras is not None, "requires the open3d/substrata environment")
class TestMatchByName(unittest.TestCase):
    def test_matches_across_groups_case_and_extension(self):
        with tempfile.TemporaryDirectory() as d:
            src = cameras.Cameras(
                _write_meta(
                    d,
                    "src.json",
                    {
                        "0": ("/f/plot.focal/FC_001.jpg", [0, 0, 0]),
                        "1": ("/f/plot.auv/AUV_0001.JPG", [1, 0, 0]),
                        "2": ("/f/plot.auv/AUV_0002.jpg", [0, 1, 0]),
                        "3": ("/f/plot.auv/AUV_0003.jpg", None),  # no pose
                    },
                )
            )
            tgt = cameras.Cameras(
                _write_meta(
                    d,
                    "tgt.json",
                    {
                        "a": ("/a/dive1/auv_0001.png", [5, 0, 0]),
                        "b": ("/a/dive2/AUV_0002.jpg", [5, 1, 0]),
                        "c": ("/a/dive2/AUV_0003.jpg", [5, 2, 0]),
                    },
                )
            )
            matches = src.match_by_name(tgt)
        self.assertEqual([m[0] for m in matches], ["auv_0001", "auv_0002"])
        self.assertEqual(matches[0][1].cam_id, "1")
        self.assertEqual(matches[0][2].cam_id, "a")

    def test_duplicate_names_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            src = cameras.Cameras(
                _write_meta(
                    d,
                    "src.json",
                    {
                        "0": ("/f/g1/IMG_1.jpg", [0, 0, 0]),
                        "1": ("/f/g2/IMG_1.jpg", [1, 0, 0]),
                        "2": ("/f/g2/IMG_2.jpg", [2, 0, 0]),
                    },
                )
            )
            tgt = cameras.Cameras(
                _write_meta(
                    d,
                    "tgt.json",
                    {
                        "a": ("/a/IMG_1.jpg", [0, 0, 0]),
                        "b": ("/a/IMG_2.jpg", [1, 0, 0]),
                    },
                )
            )
            with self.assertLogs(cameras.logger, level="WARNING"):
                matches = src.match_by_name(tgt)
        self.assertEqual([m[0] for m in matches], ["img_2"])


if __name__ == "__main__":
    unittest.main()
