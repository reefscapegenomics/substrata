# Standard Library
import importlib.util
import unittest
from pathlib import Path

# Third-Party Libraries
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_CC_PATH = _ROOT / "src/substrata/color_calibration.py"
_SETTINGS_PATH = _ROOT / "src/substrata/settings.py"


def _load_color_calibration():
    """Load color_calibration module without importing the full ``substrata`` package.

    We mock the ``substrata`` and ``substrata.settings`` entries in ``sys.modules``
    so that the top-level ``from substrata import settings`` inside the module
    resolves without pulling in the rest of the package.
    """
    import sys
    import types

    st = _load_settings_module()

    fake_substrata = types.ModuleType("substrata")
    fake_substrata.settings = st
    saved = {
        "substrata": sys.modules.get("substrata"),
        "substrata.settings": sys.modules.get("substrata.settings"),
    }
    sys.modules["substrata"] = fake_substrata
    sys.modules["substrata.settings"] = st
    try:
        spec = importlib.util.spec_from_file_location("color_calibration", _CC_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load color_calibration")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


def _load_settings_module():
    spec = importlib.util.spec_from_file_location("substrata_settings_standalone", _SETTINGS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load settings")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBilinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cc = _load_color_calibration()

    def test_corners(self) -> None:
        g = self.cc
        tl = np.array([0.0, 0.0, 0.0])
        tr = np.array([2.0, 0.0, 0.0])
        bl = np.array([0.0, 3.0, 0.0])
        br = np.array([2.0, 3.0, 0.0])
        self.assertTrue(np.allclose(g.bilinear_point_3d(0, 0, tl, tr, bl, br), tl))
        self.assertTrue(np.allclose(g.bilinear_point_3d(1, 0, tl, tr, bl, br), tr))
        self.assertTrue(np.allclose(g.bilinear_point_3d(0, 1, tl, tr, bl, br), bl))
        self.assertTrue(np.allclose(g.bilinear_point_3d(1, 1, tl, tr, bl, br), br))

    def test_center(self) -> None:
        g = self.cc
        tl = np.array([0.0, 0.0, 0.0])
        tr = np.array([2.0, 0.0, 0.0])
        bl = np.array([0.0, 2.0, 0.0])
        br = np.array([2.0, 2.0, 0.0])
        mid = g.bilinear_point_3d(0.5, 0.5, tl, tr, bl, br)
        self.assertTrue(np.allclose(mid, np.array([1.0, 1.0, 0.0])))


class TestChartRemap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cc = _load_color_calibration()

    def test_identity_bounds(self) -> None:
        """Marker quad == chart bounds => output == input."""
        g = self.cc
        u, v = g.chart_uv_to_marker_quad_uv(0.25, 0.75, 0.0, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(u, 0.25)
        self.assertAlmostEqual(v, 0.75)

    def test_center_stays_center(self) -> None:
        """Chart center maps to marker-quad center for any symmetric bounds."""
        g = self.cc
        u, v = g.chart_uv_to_marker_quad_uv(0.5, 0.5, 0.1, 0.1, 0.9, 0.9)
        self.assertAlmostEqual(u, 0.5)
        self.assertAlmostEqual(v, 0.5)

    def test_offset_bounds(self) -> None:
        """TL target is outside the chart (negative u_min)."""
        g = self.cc
        u, v = g.chart_uv_to_marker_quad_uv(0.0, 0.0, -0.277, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(u, 0.277 / 1.277, places=4)
        self.assertAlmostEqual(v, 0.0)


class TestSettingsPatchTable(unittest.TestCase):
    def test_twenty_four_patches(self) -> None:
        st = _load_settings_module()
        self.assertEqual(len(st.COLORCHECKER_CLASSIC_PATCHES), 24)


class TestAffineColorCorrection(unittest.TestCase):
    """Verify the affine least-squares solve in compute_color_correction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cc = _load_color_calibration()
        cls.st = _load_settings_module()

    def _make_calibrations(self, measured_rgb_255: np.ndarray) -> object:
        """Create a minimal ColorCalibrations with pre-filled measured data."""
        cc_cls = self.cc.ColorCalibrations
        dummy_labels = ["a", "b", "c", "d"]
        obj = cc_cls.__new__(cc_cls)
        obj.data = []
        obj.patch_definitions = self.st.COLORCHECKER_CLASSIC_PATCHES
        obj.num_cards = 1
        obj.median_rgb_255_per_patch = measured_rgb_255.copy()
        obj.outlier_mask = None
        obj._last_marker_u_min = None
        obj._last_marker_u_max = None
        obj._last_marker_v_min = None
        obj._last_marker_v_max = None
        return obj

    def test_identity_correction(self) -> None:
        """When measured == reference, the correction should be (near) identity."""
        ref = np.array(
            [[r, g, b] for (_, _, _, r, g, b) in self.st.COLORCHECKER_CLASSIC_PATCHES],
            dtype=float,
        )
        obj = self._make_calibrations(ref)
        corr = obj.compute_color_correction()

        np.testing.assert_allclose(corr["matrix"], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(corr["offset"], np.zeros(3), atol=1e-8)

    def test_known_affine_roundtrip(self) -> None:
        """Apply a known affine transform, then verify the solver recovers it."""
        rng = np.random.RandomState(42)
        M_true = np.eye(3) + rng.randn(3, 3) * 0.1
        b_true = rng.randn(3) * 5.0

        ref = np.array(
            [[r, g, b] for (_, _, _, r, g, b) in self.st.COLORCHECKER_CLASSIC_PATCHES],
            dtype=float,
        )
        measured = (ref - b_true) @ np.linalg.inv(M_true).T

        obj = self._make_calibrations(measured)
        corr = obj.compute_color_correction()

        corrected = measured @ corr["matrix"].T + corr["offset"]
        np.testing.assert_allclose(corrected, ref, atol=1e-6)

    def test_correction_dict_keys(self) -> None:
        """The returned dict has the expected keys and shapes."""
        ref = np.array(
            [[r, g, b] for (_, _, _, r, g, b) in self.st.COLORCHECKER_CLASSIC_PATCHES],
            dtype=float,
        )
        obj = self._make_calibrations(ref)
        corr = obj.compute_color_correction()

        self.assertIn("matrix", corr)
        self.assertIn("offset", corr)
        self.assertEqual(corr["matrix"].shape, (3, 3))
        self.assertEqual(corr["offset"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
