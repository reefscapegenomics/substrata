"""Tests for the pure helpers of :mod:`substrata.visualizations`.

``visualizations.py`` imports open3d/cv2/ffmpeg/fpdf at module load, so the
whole module can only be imported when those heavy deps are installed (the
conda ``substrata`` env). The tests below are skipped otherwise; they exercise
the dependency-free helper ``_resolve_scene_axes`` and ``visualize_rgb_stats``
(matplotlib only - no plotly figure or point cloud pipeline needed).
"""

# Standard Library
import unittest

# Third-Party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import visualizations as v
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    v = None


def _pts(x_span, y_span, z_span, origin=(0.0, 0.0, 0.0)):
    """Two points spanning the given extents from *origin*."""
    ox, oy, oz = origin
    return np.array([
        [ox, oy, oz],
        [ox + x_span, oy + y_span, oz + z_span],
    ])


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestResolveSceneAxes(unittest.TestCase):
    def test_default_is_an_equal_sided_cube(self):
        # Unequal data extents: every axis still gets the largest span, so the
        # aspect ratio is 1:1:1 (the pre-existing aspectmode="cube" look).
        ranges, aspect, max_range = v._resolve_scene_axes(
            _pts(4.0, 2.0, 1.0)
        )
        self.assertAlmostEqual(max_range, 4.0)
        for axis in "xyz":
            lo, hi = ranges[axis]
            self.assertAlmostEqual(hi - lo, 4.0)
            self.assertAlmostEqual(aspect[axis], 1.0)

    def test_default_centres_each_axis_on_the_data(self):
        ranges, _, _ = v._resolve_scene_axes(
            _pts(4.0, 2.0, 1.0),
        )
        # Data midpoints are 2.0, 1.0, 0.5 -> each range is mid +/- 2.0.
        self.assertAlmostEqual(0.5 * sum(ranges["x"]), 2.0)
        self.assertAlmostEqual(0.5 * sum(ranges["y"]), 1.0)
        self.assertAlmostEqual(0.5 * sum(ranges["z"]), 0.5)

    def test_explicit_limits_are_used_verbatim(self):
        ranges, _, _ = v._resolve_scene_axes(
            _pts(4.0, 4.0, 1.0),
            xlim=(0.0, 10.0), ylim=(-5.0, 5.0), zlim=(-2.0, 0.0),
        )
        self.assertEqual(ranges["x"], [0.0, 10.0])
        self.assertEqual(ranges["y"], [-5.0, 5.0])
        self.assertEqual(ranges["z"], [-2.0, 0.0])

    def test_unequal_limits_preserve_true_scale(self):
        # A 10x10x2 m window must render as a slab, not a stretched cube:
        # the aspect ratio tracks the spans so a metre is a metre everywhere.
        _, aspect, max_range = v._resolve_scene_axes(
            _pts(4.0, 4.0, 1.0),
            xlim=(0.0, 10.0), ylim=(0.0, 10.0), zlim=(-2.0, 0.0),
        )
        self.assertAlmostEqual(max_range, 10.0)
        self.assertAlmostEqual(aspect["x"], 1.0)
        self.assertAlmostEqual(aspect["y"], 1.0)
        self.assertAlmostEqual(aspect["z"], 0.2)

    def test_partial_limits_leave_other_axes_on_defaults(self):
        ranges, aspect, _ = v._resolve_scene_axes(
            _pts(4.0, 2.0, 1.0), zlim=(-1.0, 1.0),
        )
        self.assertEqual(ranges["z"], [-1.0, 1.0])
        # x and y keep the default 4.0 span.
        self.assertAlmostEqual(ranges["x"][1] - ranges["x"][0], 4.0)
        self.assertAlmostEqual(ranges["y"][1] - ranges["y"][0], 4.0)
        self.assertAlmostEqual(aspect["z"], 0.5)

    def test_max_range_follows_the_requested_window(self):
        # max_range sizes the arrows/sticks, so widening the window scales
        # them with it (keeping panels consistent at identical ranges).
        _, _, max_range = v._resolve_scene_axes(
            _pts(4.0, 4.0, 1.0), xlim=(-6.0, 10.0),
        )
        self.assertAlmostEqual(max_range, 16.0)

    def test_degenerate_data_does_not_divide_by_zero(self):
        ranges, aspect, max_range = v._resolve_scene_axes(
            np.array([[1.0, 2.0, 3.0]])
        )
        self.assertGreater(max_range, 0.0)
        for axis in "xyz":
            self.assertAlmostEqual(aspect[axis], 1.0)
        self.assertAlmostEqual(0.5 * sum(ranges["x"]), 1.0)

    def test_empty_points_with_explicit_limits(self):
        ranges, aspect, max_range = v._resolve_scene_axes(
            np.empty((0, 3)),
            xlim=(0.0, 4.0), ylim=(0.0, 4.0), zlim=(0.0, 2.0),
        )
        self.assertEqual(ranges["x"], [0.0, 4.0])
        self.assertAlmostEqual(aspect["z"], 0.5)
        self.assertAlmostEqual(max_range, 4.0)

    def test_inverted_or_flat_limits_raise(self):
        pts = _pts(4.0, 4.0, 1.0)
        for bad in [(4.0, 4.0), (5.0, 1.0)]:
            with self.assertRaises(ValueError):
                v._resolve_scene_axes(pts, xlim=bad)

    def test_wrong_length_limits_raise(self):
        pts = _pts(4.0, 4.0, 1.0)
        with self.assertRaises(ValueError):
            v._resolve_scene_axes(pts, ylim=(1.0,))
        with self.assertRaises(ValueError):
            v._resolve_scene_axes(pts, zlim=(1.0, 2.0, 3.0))

    def test_error_names_the_offending_axis(self):
        pts = _pts(4.0, 4.0, 1.0)
        with self.assertRaises(ValueError) as cm:
            v._resolve_scene_axes(pts, zlim=(2.0, 1.0))
        self.assertIn("zlim", str(cm.exception))


class _ColorPC:
    """Minimal stand-in: visualize_rgb_stats only needs ``.colors``."""

    def __init__(self, colors):
        self.colors = np.asarray(colors, dtype=float)


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestVisualizeRgbStats(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.colors = np.clip(
            rng.normal([0.5, 0.4, 0.25], [0.1, 0.09, 0.07], (2000, 3)), 0, 1
        )

    def test_returns_rgb_image_array(self):
        img = v.visualize_rgb_stats(_ColorPC(self.colors))
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)

    def test_0_255_input_matches_0_1_input(self):
        # 0-255 colours are normalised, so the figure must be identical.
        a = v.visualize_rgb_stats(_ColorPC(self.colors))
        b = v.visualize_rgb_stats(_ColorPC(self.colors * 255.0))
        self.assertTrue(np.array_equal(a, b))

    def test_medians_match_get_rgb_stats(self):
        # The title quotes get_rgb_stats' medians; check the source agrees.
        from substrata import measurements
        pc = _ColorPC(self.colors)
        r, g, b, lum = measurements.get_rgb_stats(pc)
        for got, want in zip((r, g, b), np.median(self.colors, axis=0)):
            self.assertAlmostEqual(got, want, places=10)
        self.assertAlmostEqual(lum, 0.2126*r + 0.7152*g + 0.0722*b, places=10)

    def test_no_colours_raises(self):
        with self.assertRaises(ValueError):
            v.visualize_rgb_stats(_ColorPC(np.empty((0, 3))))

    def test_figure_returned_when_saved(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "rgb.png")
            fig = v.visualize_rgb_stats(_ColorPC(self.colors), output_filename=out)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(hasattr(fig, "savefig"))
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class _PointsPC:
    """Minimal stand-in: visualize_tpi only needs ``.points``."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)


@unittest.skipUnless(v is not None, "requires the open3d/substrata environment")
class TestVisualizeTpiInputs(unittest.TestCase):
    def setUp(self):
        self.pc = _PointsPC(np.random.default_rng(0).normal(size=(50, 3)))

    def test_scalar_tpi_raises_clear_error(self):
        # Annotation.measurements stores scalar focal-point TPI values.
        with self.assertRaises(ValueError) as cm:
            v.visualize_tpi(self.pc, tpi_abs=-0.18, tpi_plane=0.05)
        self.assertIn("tpi_image", str(cm.exception))

    def test_per_point_arrays_render(self):
        vals = np.asarray(self.pc.points)[:, 2]
        img = v.visualize_tpi(self.pc, vals, vals)
        self.assertEqual(img.ndim, 3)
