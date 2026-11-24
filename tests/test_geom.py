"""Known-answer tests for :mod:`substrata.geom` (Transform / Vector primitives).

``geom.py`` only needs numpy, so it is loaded straight from ``src`` by path
(like ``test_align_correspondences``) without importing the package.
"""

# Standard Library
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src" / "substrata" / "geom.py"
_spec = importlib.util.spec_from_file_location("_substrata_geom_under_test", _SRC)
geom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geom)
Transform, Vector = geom.Transform, geom.Vector


class TestTransformComposition(unittest.TestCase):
    def setUp(self):
        self.T = Transform.from_translation((1.0, 0.0, 0.0))
        self.S = Transform.from_scale(2.0)
        self.p = np.array([[1.0, 0.0, 0.0]])

    def test_matmul_composes_like_numpy(self):
        np.testing.assert_allclose((self.T @ self.S).mat, self.T.mat @ self.S.mat)
        # (T @ S) applies S first, then T: (1,0,0) -> (2,0,0) -> (3,0,0)
        TS, ST = self.T @ self.S, self.S @ self.T
        np.testing.assert_allclose(TS.apply_to_points(self.p), [[3, 0, 0]])
        np.testing.assert_allclose(ST.apply_to_points(self.p), [[4, 0, 0]])

    def test_matmul_with_arrays(self):
        TS = self.T.mat @ self.S.mat
        np.testing.assert_allclose((self.T @ self.S.mat).mat, TS)
        np.testing.assert_allclose(self.T.mat @ self.S, TS)
        np.testing.assert_allclose((self.T.mat.tolist() @ self.S).mat, TS)

    def test_from_depth_offset_adds_dz(self):
        out = Transform.from_depth_offset(5.0).apply_to_points(np.zeros((1, 3)))
        self.assertEqual(out[0, 2], 5.0)


class TestNoInputMutation(unittest.TestCase):
    def test_from_axis_angle_keeps_axis(self):
        ax = np.array([0.0, 0.0, 2.0])
        R = Transform.from_axis_angle(ax, 0.3)
        np.testing.assert_array_equal(ax, [0.0, 0.0, 2.0])
        Rz = Transform.from_euler(0, 0, 0.3).mat
        np.testing.assert_allclose(R.mat, Rz, atol=1e-12)

    def test_from_axis_angle_rejects_zero_axis(self):
        with self.assertRaises(ValueError):
            Transform.from_axis_angle([0.0, 0.0, 0.0], 0.3)

    def test_align_x_to_vector_keeps_vector(self):
        v = Vector([1.0, 1.0, 5.0])
        Transform.align_x_to_vector(v)
        np.testing.assert_array_equal(v.vec, [1.0, 1.0, 5.0, 1.0])
        a = np.array([1.0, 1.0, 5.0])
        Transform.align_x_to_vector(a)
        np.testing.assert_array_equal(a, [1.0, 1.0, 5.0])


class TestVectorDot(unittest.TestCase):
    def test_dot_uses_xyz_only(self):
        self.assertEqual(Vector([1, 0, 0]).dot([0, 1, 0]), 0.0)
        self.assertEqual(Vector([1, 2, 3]) @ [1, 1, 1], 6.0)
        self.assertEqual([1, 1, 1] @ Vector([1, 2, 3, 0]), 6.0)
        self.assertEqual(Vector([1, 2, 3, 7]).dot(Vector([1, 1, 1, 9])), 6.0)


class TestUpVector(unittest.TestCase):
    def test_from_up_vector_maps_to_z(self):
        for u in ([0.3, 0.2, 0.9], [0, 0, -1], [1, 0, 0], [0, 0, 2]):
            R = Transform.from_up_vector(u).mat[:3, :3]
            uu = np.asarray(u, float) / np.linalg.norm(u)
            np.testing.assert_allclose(R @ uu, [0, 0, 1], atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0)


if __name__ == "__main__":
    unittest.main()
