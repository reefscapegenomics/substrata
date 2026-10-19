"""Tests for ``Annotations.subset_by_depth_range``.

``annotations.py`` imports the heavy conda deps at module load, so these tests
are skipped when they are not installed.
"""

# Standard Library
import importlib
import unittest

# Third-party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    a = importlib.import_module("substrata.annotations")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    a = None


def _make(up_vector, depth_offset):
    # z runs 0..5 in orig coords; x stays fixed so only z drives depth.
    coords = {f"p{i}": np.array([1.0, 2.0, float(i)]) for i in range(6)}
    anns = a.Annotations(coords=coords)
    anns.up_vector = np.array(up_vector)
    anns.depth_offset = depth_offset
    return anns


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestSubsetByDepthRange(unittest.TestCase):
    def test_positive_depths_ignore_z(self):
        # depth = 13 - 2 * z -> 13, 11, 9, 7, 5, 3 (z itself is 0..5)
        anns = _make([0.0, 0.0, -2.0], 13.0)
        subset = anns.subset_by_depth_range(8, 12)
        self.assertEqual(sorted(subset.data), ["p1", "p2"])

    def test_negative_heights_and_swapped_bounds(self):
        # depth_in_m = -13 + 2 * z -> -13, -11, -9, ... (height, not depth)
        anns = _make([0.0, 0.0, 2.0], -13.0)
        subset = anns.subset_by_depth_range(12, 8)
        self.assertEqual(sorted(subset.data), ["p1", "p2"])

    def test_bounds_inclusive_and_metadata_kept(self):
        anns = _make([0.0, 0.0, -2.0], 13.0)
        anns.world_transform = np.diag([2.0, 2.0, 2.0, 1.0])
        subset = anns.subset_by_depth_range(9, 11)
        self.assertEqual(sorted(subset.data), ["p1", "p2"])
        self.assertTrue(np.allclose(subset.world_transform, anns.world_transform))
        self.assertIsNot(subset.data, anns.data)

    def test_falls_back_to_transformed_z_with_warning(self):
        coords = {f"p{i}": np.array([0.0, 0.0, float(i)]) for i in range(8)}
        anns = a.Annotations(coords=coords)
        t = np.eye(4)
        t[2, 2] = -2.0  # z = 0, -2, -4, ..., -14 after the transform
        anns.transform_coords(t)
        with self.assertLogs(a.logger, level="WARNING") as cm:
            subset = anns.subset_by_depth_range(8, 12)
        self.assertEqual(sorted(subset.data), ["p4", "p5", "p6"])
        self.assertIn("transformed coords", cm.output[0])

    def test_untransformed_coords_raise(self):
        # z here equals orig z, so it says nothing about depth.
        coords = {f"p{i}": np.array([0.0, 0.0, -2.0 * i]) for i in range(8)}
        anns = a.Annotations(coords=coords)
        with self.assertRaises(ValueError):
            anns.subset_by_depth_range(8, 12)

    def test_untransformed_annotations_left_out(self):
        anns = a.Annotations(coords={"p0": np.zeros(3), "p1": np.zeros(3)})
        anns["p1"].coords = np.array([0.0, 0.0, -10.0])
        with self.assertLogs(a.logger, level="WARNING") as cm:
            subset = anns.subset_by_depth_range(0, 12)
        self.assertEqual(list(subset.data), ["p1"])
        self.assertIn("1 annotation(s)", cm.output[1])

if __name__ == "__main__":
    unittest.main()
