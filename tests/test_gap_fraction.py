"""Tests for ``measurements.calc_gap_fraction`` (fill colour is display only).

``measurements.py`` imports the heavy conda deps at module load, so these tests
are skipped when they are not installed.
"""

# Standard Library
import importlib
import types
import unittest

import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    m = importlib.import_module("substrata.measurements")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    m = None


def _scene(color):
    """A wall of points on one side of the hemisphere, all in *color*."""
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [rng.uniform(0.5, 1.0, 4000), rng.uniform(-1, 1, 4000),
         rng.uniform(0.05, 1.0, 4000)]
    )
    colors = np.tile(np.asarray(color, dtype=float) / 255.0, (len(pts), 1))
    return types.SimpleNamespace(points=pts, colors=colors)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestGapFractionFillColor(unittest.TestCase):
    def test_benthic_pixels_in_fill_colour_are_not_counted_as_sky(self):
        ann = types.SimpleNamespace(coords=np.zeros(3))
        fill = (55, 131, 187)
        _, reference, _ = m.calc_gap_fraction(ann, _scene((200, 120, 40)),
                                              fill_color=fill)
        _, same_colour, _ = m.calc_gap_fraction(ann, _scene(fill), fill_color=fill)
        self.assertLess(reference, 0.95)
        self.assertAlmostEqual(same_colour, reference)

    def test_default_fill_and_ring_colours(self):
        ann = types.SimpleNamespace(coords=np.zeros(3))
        _, _, img = m.calc_gap_fraction(ann, _scene((200, 120, 40)))
        self.assertEqual(tuple(img[100, 100]), m.settings.GAP_FRACTION_FILL_COLOR)
        _, _, rings = m.calc_gap_fraction(ann, _scene((200, 120, 40)),
                                          show_rings=True)
        # 22.5 deg ring: 25 px from the centre (anti-aliased, so near-white)
        window = rings[98:103, 122:129].reshape(-1, 3)
        self.assertTrue((window.min(axis=1) > 200).any())


if __name__ == "__main__":
    unittest.main()
