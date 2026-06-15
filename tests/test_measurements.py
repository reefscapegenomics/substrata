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


if __name__ == "__main__":
    unittest.main()
