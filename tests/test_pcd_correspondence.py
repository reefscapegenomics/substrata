"""Tests for the point-cloud correspondence QC plots (``align --pdf``).

``visualizations.py`` imports open3d/cv2 at module load, so these tests are
skipped outside the conda ``substrata`` environment.
"""

# Standard Library
import unittest

# Third-Party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from substrata import visualizations
    from substrata.pointclouds import SimplePointCloud
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    visualizations = None


def _terrain(n=40_000, extent=10.0, seed=0):
    """Gridded rough surface (z = f(x, y)) with random colours."""
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n))
    g = np.linspace(0, extent, side)
    x, y = np.meshgrid(g, g)
    z = 0.5 * np.sin(x) + 0.4 * np.cos(1.3 * y)
    pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
    return SimplePointCloud(pts, rng.uniform(size=pts.shape))


def _patch(pcd, lo=3.0, hi=6.0, dz=0.0):
    """Sub-area of ``pcd`` shifted vertically by ``dz``."""
    pts = np.asarray(pcd.points)
    keep = np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), axis=1)
    out = pts[keep] + [0.0, 0.0, dz]
    return SimplePointCloud(out, np.asarray(pcd.colors)[keep])


@unittest.skipUnless(
    visualizations is not None, "requires the open3d/substrata environment"
)
class TestPcdCorrespondence(unittest.TestCase):
    def setUp(self):
        self.tgt = _terrain()

    def tearDown(self):
        plt.close("all")

    def test_identical_patch_has_zero_distances(self):
        fig, stats = visualizations.plot_pcd_correspondence(
            _patch(self.tgt), self.tgt, return_stats=True
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertAlmostEqual(stats["c2c_median"], 0.0, places=9)
        self.assertAlmostEqual(stats["dz_median"], 0.0, delta=1e-3)
        self.assertGreater(stats["dz_coverage"], 0.99)

    def test_vertical_offset_shows_in_dz(self):
        _, stats = visualizations.plot_pcd_correspondence(
            _patch(self.tgt, dz=0.04), self.tgt, return_stats=True
        )
        self.assertAlmostEqual(stats["dz_median"], 0.04, delta=0.01)
        self.assertGreater(stats["c2c_median"], 0.01)

    def test_no_overlap_raises(self):
        far = _patch(self.tgt)
        far.points = np.asarray(far.points) + [100.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            visualizations.plot_pcd_correspondence(far, self.tgt)

    def test_context_figure(self):
        fig = visualizations.plot_pcd_context(
            _patch(self.tgt), self.tgt, labels=("plot", "auv"), title="t"
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len([a for a in fig.axes if a.get_title()]), 4)

    def test_spatial_axes_have_equal_scale(self):
        # Any panel plotting one coordinate against another (labels like
        # "x (m)" vs "z (m)") must use the same scale on both axes
        rng = np.random.default_rng(2)
        cams = rng.uniform(0, 10, size=(10, 3))
        figs = [
            visualizations.plot_pcd_context(_patch(self.tgt), self.tgt),
            visualizations.plot_pcd_correspondence(_patch(self.tgt), self.tgt),
            visualizations.plot_camera_correspondence(cams + 0.01, cams),
        ]
        n_spatial = 0
        for fig in figs:
            for ax in fig.axes:
                xl, yl = ax.get_xlabel(), ax.get_ylabel()
                if xl[:1] in "xyz" and xl[1:3] == " (" and yl[:1] in "xyz":
                    n_spatial += 1
                    self.assertEqual(ax.get_aspect(), 1.0, f"{xl} vs {yl}")
        self.assertEqual(n_spatial, 9)

    def test_camera_correspondence_with_rejected(self):
        rng = np.random.default_rng(1)
        tgt = rng.uniform(0, 10, size=(30, 3))
        src = tgt + rng.normal(scale=0.01, size=tgt.shape)
        src[:3] += 3.0
        inliers = np.ones(30, bool)
        inliers[:3] = False
        fig = visualizations.plot_camera_correspondence(src, tgt, inliers=inliers)
        self.assertIsInstance(fig, plt.Figure)


if __name__ == "__main__":
    unittest.main()
