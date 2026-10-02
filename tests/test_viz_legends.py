"""Tests for the compact 3-D plot colour-bar helpers in ``visualizations``.

``visualizations.py`` imports the heavy conda deps at module load, so these
tests are skipped when they are not installed.
"""

# Standard Library
import importlib
import unittest

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    v = importlib.import_module("substrata.visualizations")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    v = None


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestCompactColorbar(unittest.TestCase):
    def test_symmetric_ticks(self):
        self.assertEqual(v._symmetric_ticks(0.14), (-0.1, 0, 0.1))
        self.assertEqual(v._symmetric_ticks(0.036), (-0.03, 0, 0.03))
        self.assertEqual(v._symmetric_ticks(0.3), (-0.3, 0, 0.3))
        self.assertEqual(v._symmetric_ticks(0), (0,))

    def test_array_ticks_carry_the_unit(self):
        cb = v._compact_colorbar("°", tickvals=(0, 45, 90))
        self.assertEqual(cb["ticktext"], ["0°", "45°", "90°"])
        self.assertEqual(cb["title"], {"text": ""})
        self.assertEqual((cb["xanchor"], cb["yanchor"]), ("left", "bottom"))




@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestGapFractionAxes(unittest.TestCase):
    def test_white_background_and_orientation(self):
        import numpy as np

        res = 40
        yy, xx = np.mgrid[:res, :res]
        mask = (xx - res // 2) ** 2 + (yy - res // 2) ** 2 <= (res // 2) ** 2
        image = np.zeros((res, res, 3), dtype=np.uint8)
        image[mask] = 80
        image[34:37, 18:22] = (255, 0, 0)  # rows = +X: far along +X, Y ~ 0
        image[18:22, 34:37] = (0, 255, 0)  # columns = +Y: far along +Y, X ~ 0
        out = v.render_gap_fraction_axes(image, mask, width=300, height=300)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue((out[2, 2] == 255).all())  # white figure corner
        red = np.argwhere((out[:, :, 0] > 200) & (out[:, :, 1] < 60))
        green = np.argwhere((out[:, :, 1] > 200) & (out[:, :, 0] < 60))
        gray = np.argwhere((out == 80).all(axis=2))
        centre = gray.mean(axis=0)  # (row, col) of the imaging circle
        self.assertGreater(red[:, 1].mean(), centre[1] + 20)  # +X to the right
        self.assertLess(green[:, 0].mean(), centre[0] - 20)  # +Y up


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestTpiShowTitle(unittest.TestCase):
    def test_hiding_the_title_removes_the_text(self):
        import os
        import tempfile
        import types

        import numpy as np

        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(-1, 1, (500, 2)), np.zeros(500)])
        pcd = types.SimpleNamespace(points=pts)
        vals = rng.uniform(-0.5, 0.5, 500)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tpi.png")
            fig = v.visualize_tpi(pcd, vals, vals, output_filename=out)
            titles = [ax.get_title() for ax in fig.axes[:2]]  # colorbars follow
            self.assertTrue(all("TPI" in t for t in titles))
            fig = v.visualize_tpi(pcd, vals, vals, output_filename=out,
                                  show_title=False)
            self.assertEqual([ax.get_title() for ax in fig.axes], [""] * 4)


if __name__ == "__main__":
    unittest.main()
