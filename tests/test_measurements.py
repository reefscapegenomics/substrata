"""Tests for the pure helpers of :mod:`substrata.measurements`.

``measurements.py`` imports open3d/cv2/sklearn at module load, so the whole
module can only be imported when those heavy deps are installed (the conda
``substrata`` env). The tests below are skipped otherwise; they exercise only
the dependency-free helpers ``_resolve_outer_radius`` and
``_benthic_fraction_from_results`` (no point cloud / camera pipeline needed).
"""

# Standard Library
import math
import unittest

# Third-Party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import measurements as m
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    m = None


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestResolveOuterRadius(unittest.TestCase):
    def test_annulus_width(self):
        self.assertAlmostEqual(
            m._resolve_outer_radius(0.1, None, 0.4, default_outer=0.5), 0.5
        )

    def test_radius_outer(self):
        self.assertAlmostEqual(
            m._resolve_outer_radius(0.1, 0.8, None, default_outer=0.5), 0.8
        )

    def test_default_fallback(self):
        self.assertAlmostEqual(
            m._resolve_outer_radius(0.1, None, None, default_outer=0.5), 0.5
        )

    def test_both_raises(self):
        with self.assertRaises(ValueError):
            m._resolve_outer_radius(0.1, 0.8, 0.4, default_outer=0.5)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestBenthicFractionFromResults(unittest.TestCase):
    def _res(self, label, probs=None):
        return {"label": label, "confidence": None, "probs": probs, "pred_idx": 0}

    def test_unweighted_count_fraction(self):
        results = {
            "a": self._res("MAF"),
            "b": self._res("MAF"),
            "c": self._res("CB"),
            "d": None,  # unmatched/unclassified -> ignored
        }
        frac, breakdown = m._benthic_fraction_from_results(results, "MAF", False)
        self.assertAlmostEqual(frac, 2 / 3)
        self.assertEqual(breakdown["n_classified"], 3)
        self.assertEqual(breakdown["n_target"], 2)
        self.assertEqual(breakdown["class_counts"], {"MAF": 2, "CB": 1})

    def test_weighted_mean_probability(self):
        results = {
            "a": self._res("MAF", {"MAF": 0.9, "CB": 0.1}),
            "b": self._res("CB", {"MAF": 0.3, "CB": 0.7}),
        }
        frac, _ = m._benthic_fraction_from_results(results, "MAF", True)
        self.assertAlmostEqual(frac, (0.9 + 0.3) / 2)

    def test_weighted_fallback_to_indicator_without_probs(self):
        # No probs map -> hard 0/1 indicator contributes.
        results = {"a": self._res("MAF", None), "b": self._res("CB", None)}
        frac, _ = m._benthic_fraction_from_results(results, "MAF", True)
        self.assertAlmostEqual(frac, 0.5)

    def test_empty_returns_nan(self):
        results = {"a": None, "b": None}
        frac, breakdown = m._benthic_fraction_from_results(results, "MAF", False)
        self.assertTrue(math.isnan(frac))
        self.assertEqual(breakdown["n_classified"], 0)
        self.assertEqual(breakdown["n_target"], 0)
        self.assertEqual(breakdown["class_counts"], {})


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestColonyBaseZ(unittest.TestCase):
    def test_p10_of_inner_footprint(self):
        # 10 inner points (z = 0..9) at the centre + far points that must be
        # excluded (high z, outside radius_inner).
        inner = [[0.0, 0.0, float(z)] for z in range(10)]
        far = [[10.0, 10.0, 99.0], [-10.0, -10.0, 99.0]]
        pts = np.array(inner + far, dtype=float)
        z = m._colony_base_z(pts, np.array([0.0, 0.0]), radius_inner=0.5,
                             percentile=10)
        self.assertAlmostEqual(z, np.percentile(range(10), 10))

    def test_no_inner_points_returns_nan(self):
        pts = np.array([[10.0, 10.0, 5.0]], dtype=float)
        z = m._colony_base_z(pts, np.array([0.0, 0.0]), radius_inner=0.5,
                             percentile=10)
        self.assertTrue(math.isnan(z))

    def test_zero_radius_returns_nan(self):
        pts = np.array([[0.0, 0.0, 5.0]], dtype=float)
        z = m._colony_base_z(pts, np.array([0.0, 0.0]), radius_inner=0.0,
                             percentile=10)
        self.assertTrue(math.isnan(z))


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestHeightWeight(unittest.TestCase):
    def test_ramp(self):
        zc, d = 1.0, 0.2
        self.assertEqual(m._height_weight(1.2, zc, d), 1.0)  # above base
        self.assertEqual(m._height_weight(1.0, zc, d), 1.0)  # at base
        self.assertAlmostEqual(m._height_weight(0.9, zc, d), 0.5)  # half down
        self.assertAlmostEqual(m._height_weight(0.8, zc, d), 0.0)  # at falloff
        self.assertEqual(m._height_weight(0.5, zc, d), 0.0)  # well below

    def test_zero_falloff_is_strict_step(self):
        self.assertEqual(m._height_weight(1.0, 1.0, 0.0), 1.0)
        self.assertEqual(m._height_weight(0.99, 1.0, 0.0), 0.0)

    def test_nan_base(self):
        self.assertTrue(math.isnan(m._height_weight(1.0, float("nan"), 0.2)))


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestBenthicInteraction(unittest.TestCase):
    def _res(self, label, probs=None):
        return {"label": label, "confidence": None, "probs": probs, "pred_idx": 0}

    def test_height_weighted_cover(self):
        # z_colony=1.0, falloff=0.2. Four sand samples at decreasing height plus
        # one non-target sample. Weights: 1, 1, 0.5, 0 -> sum 2.5 over 5 samples.
        results = {
            "a": self._res("SAND"), "b": self._res("SAND"),
            "c": self._res("SAND"), "d": self._res("SAND"),
            "e": self._res("CB"),
        }
        sample_z = {"a": 1.2, "b": 1.0, "c": 0.9, "d": 0.8, "e": 1.5}
        cover, bd = m._benthic_interaction_from_results(
            results, sample_z, "SAND", z_colony=1.0, falloff_depth=0.2,
            weight_by_probability=False,
        )
        self.assertAlmostEqual(cover, 2.5 / 5)
        self.assertAlmostEqual(bd["interaction_weight_sum"], 2.5)
        self.assertEqual(bd["n_classified"], 5)
        self.assertAlmostEqual(bd["z_colony"], 1.0)

    def test_strict_cutoff_falloff_zero(self):
        results = {
            "a": self._res("SAND"), "b": self._res("SAND"),
            "c": self._res("SAND"), "d": self._res("SAND"),
            "e": self._res("CB"),
        }
        sample_z = {"a": 1.2, "b": 1.0, "c": 0.9, "d": 0.8, "e": 1.5}
        cover, bd = m._benthic_interaction_from_results(
            results, sample_z, "SAND", z_colony=1.0, falloff_depth=0.0,
            weight_by_probability=False,
        )
        # Only a (1.2) and b (1.0) are at/above base -> 2 of 5.
        self.assertAlmostEqual(cover, 2 / 5)
        self.assertAlmostEqual(bd["interaction_weight_sum"], 2.0)

    def test_weight_by_probability_multiplies_height(self):
        # t = P(SAND) for every classified sample, times the height weight.
        results = {
            "a": self._res("SAND", {"SAND": 1.0, "CB": 0.0}),
            "b": self._res("CB", {"SAND": 0.5, "CB": 0.5}),
        }
        sample_z = {"a": 1.2, "b": 0.9}  # w = 1.0, 0.5
        cover, bd = m._benthic_interaction_from_results(
            results, sample_z, "SAND", z_colony=1.0, falloff_depth=0.2,
            weight_by_probability=True,
        )
        self.assertAlmostEqual(bd["interaction_weight_sum"], 1.0 * 1.0 + 0.5 * 0.5)
        self.assertAlmostEqual(cover, 1.25 / 2)

    def test_nan_base_returns_nan(self):
        results = {"a": self._res("SAND")}
        cover, bd = m._benthic_interaction_from_results(
            results, {"a": 1.0}, "SAND", z_colony=float("nan"),
            falloff_depth=0.2, weight_by_probability=False,
        )
        self.assertTrue(math.isnan(cover))
        self.assertEqual(bd["n_classified"], 1)

    def test_empty_returns_nan(self):
        cover, bd = m._benthic_interaction_from_results(
            {"a": None}, {}, "SAND", z_colony=1.0, falloff_depth=0.2,
            weight_by_probability=False,
        )
        self.assertTrue(math.isnan(cover))
        self.assertEqual(bd["n_classified"], 0)


if __name__ == "__main__":
    unittest.main()
