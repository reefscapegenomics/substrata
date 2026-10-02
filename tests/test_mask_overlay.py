"""Tests for the SA text / radius-line options of ``get_crop_img_from_masks``.

``visualizations.py`` imports the heavy conda deps at module load, so these
tests are skipped when they are not installed.
"""

# Standard Library
import importlib
import os
import tempfile
import types
import unittest

import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    v = importlib.import_module("substrata.visualizations")
    import cv2
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    v = None


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestRadiusLineEndpoint(unittest.TestCase):
    def test_length_is_radius_and_points_to_closest_distance(self):
        contour = np.array([[10, 0], [0, 30], [-50, 0]])
        end = v._radius_line_endpoint((0, 0), contour, 25.0)
        np.testing.assert_allclose(np.linalg.norm(end), 25.0)
        np.testing.assert_allclose(end, [0, 25.0])  # towards (0, 30)

    def test_no_usable_points(self):
        self.assertIsNone(v._radius_line_endpoint((0, 0), np.empty((0, 2)), 5))
        self.assertIsNone(v._radius_line_endpoint((0, 0), [[0, 0]], 5))
        self.assertIsNone(v._radius_line_endpoint((0, 0), [[1, 1]], 0))


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestCropImgFromMasksOverlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "img.png")
        cv2.imwrite(self.path, np.full((200, 200, 3), 100, dtype=np.uint8))
        vals = np.zeros((200, 200), dtype=np.uint8)
        vals[50:150, 50:150] = 1
        self.mask = types.SimpleNamespace(
            vals=vals, area_in_px=int(vals.sum()), area_in_cm2=12.5
        )
        self.match = types.SimpleNamespace(
            filepath=self.path, masks=[self.mask], mask=self.mask, x=100, y=100
        )

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, **kwargs):
        return v.get_crop_img_from_masks(self.match, 100, 100, **kwargs)

    def test_show_text_false_only_removes_text(self):
        with_text = self.render()
        without = self.render(show_text=False)
        diff = np.any(with_text != without, axis=2)
        self.assertTrue(diff.any())
        ys, _ = np.nonzero(diff)
        self.assertLess(ys.max(), 40)  # text sits in the top-left corner

    def test_unknown_area_does_not_raise(self):
        self.mask.area_in_cm2 = None
        self.render()
        self.render(show_radius=True)

    def test_radius_line_is_drawn_in_annotation_color(self):
        base = self.render(show_text=False, annotation_radius=2)
        line = self.render(show_text=False, show_radius=True, annotation_radius=2)
        green = (line[:, :, 1] > 200) & (line[:, :, 0] < 80) & (line[:, :, 2] < 80)
        added = green & np.any(base != line, axis=2)
        self.assertTrue(added.any())
        # Square 100x100 px -> r = sqrt(10000 / pi) = 56.4 px full-res; the
        # crop maps the 100 px mask box onto 100 output px, so ~56 px long.
        ys, xs = np.nonzero(added)
        length = np.max(np.hypot(xs - 50, ys - 50))
        self.assertAlmostEqual(length, 56.4, delta=3)

    def test_show_point_false_hides_the_dot(self):
        def dot_pixels(img):
            green = (img[:, :, 1] > 200) & (img[:, :, 0] < 80) & (img[:, :, 2] < 80)
            return green[40:61, 40:61].sum()  # around the point (crop centre)

        self.assertGreater(dot_pixels(self.render(show_text=False)), 50)
        self.assertEqual(
            dot_pixels(self.render(show_text=False, show_point=False)), 0
        )
        # The radius line is still drawn from the (hidden) point
        line = self.render(show_text=False, show_point=False, show_radius=True)
        self.assertGreater(dot_pixels(line), 0)


if __name__ == "__main__":
    unittest.main()
