"""Safety net for the separation of measurements and figures.

Two properties are checked for every metric that can draw a QC figure:

- the metric is the same number with the figure on and off
  (``generate_image=True`` / ``False``, display options of the gap fraction), so
  drawing never changes a reported value;
- the figure shows the value it is given (the title or text of the figure
  carries the number the measurement function returned), so every figure shows
  exactly the reported number.

Plotly figures are returned as figure objects (the PNG export is forced to
fail, as without Kaleido); matplotlib figures are captured when they are
rasterised or saved. ``measurements.py`` and ``visualizations.py`` import
open3d/cv2 at module load, so the module is skipped without those deps.
"""

# Standard Library
import contextlib
import math
import types
import unittest
from unittest import mock

import numpy as np

try:  # Heavy deps (open3d, cv2, ...) are only present in the conda env.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import plotly.graph_objects as go

    from substrata import measurements as m
    from substrata import pointclouds
    from substrata import visualizations as viz
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    m = None


def _cloud(n=4000, seed=0, tilt=0.2):
    """A rough tilted surface with upward normals and random colours."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = tilt * xy[:, 0] + 0.05 * np.sin(5 * xy[:, 0]) * np.cos(4 * xy[:, 1])
    z = z + rng.normal(0, 0.01, n)
    pts = np.c_[xy, z]
    normals = np.c_[-tilt + rng.normal(0, 0.2, n), rng.normal(0, 0.2, n), np.ones(n)]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return pointclouds.SimplePointCloud(pts, rng.uniform(size=(n, 3)), normals)


@contextlib.contextmanager
def _plotly_returns_figure():
    """Make the plotly PNG export fail, so the figure functions return the figure."""
    def _fail(*_a, **_k):
        raise RuntimeError("no kaleido in this test")

    with mock.patch.object(go.Figure, "to_image", _fail):
        yield


@contextlib.contextmanager
def _captured_mpl_figures():
    """Collect the matplotlib figures that are saved / rasterised in the block."""
    figs = []
    original = matplotlib.figure.Figure.savefig

    def _savefig(self, *args, **kwargs):
        figs.append(self)
        return original(self, *args, **kwargs)

    with mock.patch.object(matplotlib.figure.Figure, "savefig", _savefig):
        yield figs


def _mpl_text(fig):
    """All text of a matplotlib figure (titles, labels, text artists)."""
    out = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_title("left"), ax.get_title("right")]
        out += [t.get_text() for t in ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            out += [t.get_text() for t in legend.get_texts()]
    return "\n".join(out)


def _plotly_text(fig):
    title = fig.layout.title.text or ""
    annotations = [a.text or "" for a in (fig.layout.annotations or ())]
    traces = []
    for tr in fig.data:
        text = getattr(tr, "text", None)
        if text is not None:
            traces += [str(t) for t in np.atleast_1d(text)]
    return "\n".join([title] + annotations + traces)


def _same(testcase, a, b):
    """Assert two metric tuples are identical (NaN equal to NaN)."""
    testcase.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        if isinstance(x, float) and math.isnan(x):
            testcase.assertTrue(isinstance(y, float) and math.isnan(y))
        elif isinstance(x, (list, tuple, np.ndarray)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        else:
            testcase.assertEqual(x, y)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestMetricsIndependentOfFigures(unittest.TestCase):
    """The reported numbers are the same with and without the QC figure."""

    def setUp(self):
        self.pcd = _cloud()

    def test_plane_angles(self):
        off = m.get_plane_angles(self.pcd, generate_image=False)
        with _plotly_returns_figure():
            on = m.get_plane_angles(self.pcd, generate_image=True)
        _same(self, off[:5], on[:5])
        self.assertIsNone(off[5])
        self.assertIsNotNone(on[5])

    def test_roughness(self):
        off = m.calc_roughness(self.pcd, generate_image=False)
        with _plotly_returns_figure():
            on = m.calc_roughness(self.pcd, generate_image=True)
        _same(self, off[:2], on[:2])
        self.assertIsNone(off[2])

    def test_vector_dispersion(self):
        off = m.get_vector_dispersion(self.pcd, generate_image=False)
        with _plotly_returns_figure():
            on = m.get_vector_dispersion(self.pcd, generate_image=True)
        self.assertEqual(off[0], on[0])
        self.assertIsNone(off[1])
        self.assertIsNotNone(on[1])

    def test_tpi_and_tri(self):
        center = np.array([0.0, 0.0, 0.1])
        for raster in (None, 0.05):
            kw = dict(center=center, radius_inner=0.2, radius_outer=0.8,
                      raster_cell_size=raster)
            off = m.calc_tpi_and_tri(self.pcd, generate_image=False, **kw)
            on = m.calc_tpi_and_tri(self.pcd, generate_image=True, **kw)
            _same(self, off[:6], on[:6])
            self.assertIsNone(off[6])
            self.assertEqual(on[6].ndim, 3)

    def test_gap_fraction_display_options(self):
        rng = np.random.default_rng(1)
        pts = np.c_[rng.uniform(-1, 1, (20000, 2)), rng.uniform(0.2, 1.0, 20000)]
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > 0.5]
        scene = types.SimpleNamespace(points=pts, colors=rng.uniform(size=pts.shape))
        ann = types.SimpleNamespace(coords=np.zeros(3), id="t")
        base = m.calc_gap_fraction(ann, scene, color_output=False)[:2]
        options = (
            dict(color_output=True),
            dict(show_rings=True),
            dict(show_axes=True, show_rings=True),
            dict(seed_points=[(0.5, 0.5), (0.25, 0.5)]),
            dict(fill_color=(80, 80, 80)),
        )
        for opts in options:
            with self.subTest(**{k: str(v) for k, v in opts.items()}):
                self.assertEqual(m.calc_gap_fraction(ann, scene, **opts)[:2], base)

    def test_gap_fraction_matches_pure_helpers(self):
        rng = np.random.default_rng(2)
        pts = np.c_[rng.uniform(-1, 1, (20000, 2)), rng.uniform(0.2, 1.0, 20000)]
        scene = types.SimpleNamespace(points=pts, colors=rng.uniform(size=pts.shape))
        ann = types.SimpleNamespace(coords=np.zeros(3), id="t")
        raw, fill, _ = m.calc_gap_fraction(ann, scene, resolution=100)
        cover, disk = m.hemispherical_cover(pts, np.zeros(3), 100)
        raw2, fill2, _ = m.gap_fractions(cover, disk, [(50, 50), (25, 50)])
        self.assertEqual((raw, fill), (raw2, fill2))


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestFiguresShowGivenValues(unittest.TestCase):
    """Each figure prints the number it is given (not a recomputed one)."""

    def setUp(self):
        self.pcd = _cloud()

    def test_roughness_title(self):
        with _plotly_returns_figure():
            fig = viz.visualize_roughness(self.pcd, ra=0.012345, rq=0.054321)
        text = _plotly_text(fig)
        self.assertIn("0.012345", text)
        self.assertIn("0.054321", text)

    def test_roughness_default_is_the_measurement(self):
        ra, rq, _ = m.calc_roughness(self.pcd, generate_image=False)
        with _plotly_returns_figure():
            fig = viz.visualize_roughness(self.pcd)
        text = _plotly_text(fig)
        self.assertIn(f"{ra:.6f}", text)
        self.assertIn(f"{rq:.6f}", text)

    def test_roughness_measurement_figure(self):
        with _plotly_returns_figure():
            ra, rq, fig = m.calc_roughness(self.pcd, generate_image=True)
        text = _plotly_text(fig)
        self.assertIn(f"{ra:.6f}", text)
        self.assertIn(f"{rq:.6f}", text)

    def test_vector_dispersion_title(self):
        with _plotly_returns_figure():
            fig = viz.visualize_vector_dispersion(self.pcd, dispersion=0.4321)
        self.assertIn("0.4321", _plotly_text(fig))

    def test_vector_dispersion_measurement_figure(self):
        with _plotly_returns_figure():
            disp, fig = m.get_vector_dispersion(self.pcd, generate_image=True)
        self.assertIn(f"{disp:.4f}", _plotly_text(fig))

    def test_elevation_measurement_figure(self):
        with _plotly_returns_figure():
            *_, elev, _plane, _az, fig = m.get_plane_angles(self.pcd)
        self.assertIn(f"{elev:.1f}°", _plotly_text(fig))

    def test_tpi_titles(self):
        center = np.array([0.0, 0.0, 0.1])
        with _captured_mpl_figures() as figs:
            res = m.calc_tpi_and_tri(
                self.pcd, center=center, radius_inner=0.2, radius_outer=0.8
            )
        text = _mpl_text(figs[-1])
        tpi_abs, _, tpi_plane, _, tri_abs, tri_plane, _ = res
        for value in (tpi_abs, tpi_plane, tri_abs, tri_plane):
            self.assertIn(f"{value:.4f} m", text)

    def test_tpi_figure_values(self):
        pts = np.asarray(self.pcd.points)
        vals = np.zeros(len(pts))
        with _captured_mpl_figures() as figs:
            viz.visualize_tpi(self.pcd, vals, vals, mean_tpi_abs=0.1234,
                              mean_tpi_plane=-0.4321, mean_tri_abs=0.2468,
                              mean_tri_plane=0.1357)
        text = _mpl_text(figs[-1])
        for s in ("0.1234 m", "-0.4321 m", "0.2468 m", "0.1357 m"):
            self.assertIn(s, text)

    def test_elevation_given_angle(self):
        with _plotly_returns_figure():
            fig = viz.visualize_elevation_angle(
                self.pcd, [0.1, 0.0, 1.0, 0.0], elev_deg=12.3
            )
        self.assertIn("12.3°", _plotly_text(fig))

    def test_depth_regression_errors(self):
        rng = np.random.default_rng(3)
        obs = -rng.uniform(5, 15, 40)
        pred = obs + rng.normal(0, 0.2, 40)
        errors = {"rmse": 0.1234, "mae": 0.0567, "r2": 0.9876}
        fig = viz.plot_depth_regression(obs, pred, errors=errors)
        text = _mpl_text(fig)
        for s in ("0.988", "0.123m", "0.057m"):
            self.assertIn(s, text)
        res = m.fit_depth_regression(rng.uniform(0, 5, (40, 3)), obs)
        text = _mpl_text(viz.plot_depth_regression(res.depths, res.depths_pred))
        self.assertIn(f"R² = {res.r2:.3f}, RMSE = {res.rmse:.3f}m", text)
        self.assertIn(f"MAE = {res.mae:.3f}m", text)
        matplotlib.pyplot.close("all")

    def test_regression_errors_match_fit(self):
        rng = np.random.default_rng(4)
        pts = rng.uniform(0, 5, (30, 3))
        res = m.fit_depth_regression(pts, -(2 + pts[:, 2] + rng.normal(0, .1, 30)))
        err = m.regression_errors(res.depths, res.depths_pred)
        self.assertEqual((err["mse"], err["rmse"], err["mae"], err["r2"]),
                         (res.mse, res.rmse, res.mae, res.r2))

    def test_camera_correspondence_residuals(self):
        rng = np.random.default_rng(1)
        tgt = rng.uniform(0, 10, size=(30, 3))
        src = tgt + rng.normal(scale=0.01, size=tgt.shape)
        src[:3] += 3.0
        inliers = np.ones(30, bool)
        inliers[:3] = False
        res = m.camera_residuals(src, tgt, inliers)
        r = np.linalg.norm(src - tgt, axis=1)
        self.assertAlmostEqual(res["rms"], float(np.sqrt(np.mean(r[3:] ** 2))))
        self.assertEqual((res["n_inliers"], res["n_rejected"]), (27, 3))
        fig = viz.plot_camera_correspondence(src, tgt, inliers=inliers, residuals=res)
        self.assertIn(f"inlier RMS {res['rms']:.4f}", _mpl_text(fig))
        matplotlib.pyplot.close("all")

    def test_pcd_correspondence_stats_are_the_measurement(self):
        rng = np.random.default_rng(5)
        tgt = pointclouds.SimplePointCloud(
            np.c_[rng.uniform(0, 4, (20000, 2)), rng.normal(0, 0.01, 20000)])
        src = pointclouds.SimplePointCloud(
            np.asarray(tgt.points)[:5000] + [0, 0, 0.03])
        corr = m.pcd_correspondence(src, tgt)
        fig, stats = viz.plot_pcd_correspondence(src, tgt, return_stats=True)
        self.assertEqual(stats, corr.stats)
        self.assertIn(f"median {corr.stats['dz_median']:+.4f}", _mpl_text(fig))
        matplotlib.pyplot.close("all")

    def test_benthic_panel_shows_result(self):
        anns = {
            f"a{k}": types.SimpleNamespace(
                id=f"a{k}", coords=np.array([k * 0.1, 0.0, 0.0]), image_match=object())
            for k in range(4)
        }
        intercepts = types.SimpleNamespace(data=anns)
        results = {f"a{k}": {"label": "sand" if k < 3 else "coral"} for k in range(4)}
        with _captured_mpl_figures() as figs:
            viz.visualize_benthic_fraction(intercepts, results, "sand",
                                           center=np.zeros(3), radius_outer=1.0)
        self.assertIn("fraction=0.750", _mpl_text(figs[-1]))
        result = {"fraction": 0.123, "n_target": 7, "n_classified": 9}
        with _captured_mpl_figures() as figs:
            viz.visualize_benthic_fraction(intercepts, results, "sand",
                                           center=np.zeros(3), radius_outer=1.0,
                                           result=result)
        text = _mpl_text(figs[-1])
        self.assertIn("fraction=0.123", text)
        self.assertIn("n_target=7", text)

    def test_color_summary_shows_fit_residuals(self):
        from substrata import color_calibration as cc

        rng = np.random.default_rng(6)
        ref = rng.uniform(20, 230, (24, 3))
        meas = ref * 0.8 + 10 + rng.normal(0, 3, (24, 3))
        corr = {"matrix": np.eye(3) * 1.25, "offset": np.full(3, -12.5)}
        res = cc.correction_residuals(meas, ref, corr)
        fig = viz.plot_color_correction_summary(
            meas, ref, corr, [str(i) for i in range(24)], residuals=res)
        text = _mpl_text(fig)
        for ch in range(3):
            self.assertIn(
                f"RMSE {res['rmse_before_per_channel'][ch]:.1f} -> "
                f"{res['rmse_per_channel'][ch]:.1f}", text)
        matplotlib.pyplot.close("all")


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestPureHelpers(unittest.TestCase):
    """The helpers the metrics and the figures share."""

    def test_annulus_plane_and_tpi_point_values(self):
        rng = np.random.default_rng(7)
        pts = np.c_[rng.uniform(-1, 1, (500, 2)), np.zeros(500)]
        pts[:, 2] = 0.3 * pts[:, 0] + 2.0
        normal, d = m.annulus_plane(pts)
        self.assertGreater(normal[2], 0)
        np.testing.assert_allclose(pts @ normal + d, 0, atol=1e-12)
        focal = np.array([0.0, 0.0, 2.5])
        z_rel, plane_rel = m.tpi_point_values(pts, focal, (normal, d))
        np.testing.assert_allclose(z_rel, pts[:, 2] - 2.5)
        np.testing.assert_allclose(plane_rel, -(normal @ focal + d), atol=1e-12)
        self.assertIsNone(m.annulus_plane(pts[:2]))
        self.assertTrue(np.isnan(m.tpi_point_values(pts, focal, None)[1]).all())

    def test_plane_orientation_matches_get_plane_angles(self):
        pcd = _cloud()
        theta, psi, elev, plane, az, _ = m.get_plane_angles(pcd, generate_image=False)
        self.assertEqual(m.plane_orientation(plane[:3]), (theta, psi, elev, az))
        self.assertAlmostEqual(elev, m.get_elevation_angle(plane[:3]))

    def test_distances_to_plane(self):
        d = m.distances_to_plane(np.array([[0, 0, 1.0], [0, 0, -2.0]]), (0, 0, 2, 0))
        np.testing.assert_allclose(d, [1.0, -2.0])

    def test_dispersion_helpers(self):
        normals = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, -1.0]])
        self.assertAlmostEqual(m.vector_dispersion_from_normals(normals), 1.0)
        dev, mean = m.normal_deviation_deg(normals)
        np.testing.assert_allclose(dev, [0, 0, 180])
        np.testing.assert_allclose(mean, [0, 0, 1])


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestImportDirection(unittest.TestCase):
    """Compute modules import no figure library (nor visualizations) at load."""

    def _loaded(self, module, watch):
        import os
        import subprocess
        import sys

        src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
        code = (
            f"import sys; sys.path.insert(0, {src!r}); import {module}; "
            f"print([m for m in {watch!r} if m in sys.modules])"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip().splitlines()[-1]

    def test_measurements_is_light(self):
        watch = ("matplotlib.pyplot", "plotly", "fpdf", "substrata.visualizations",
                 "open3d", "sklearn")
        self.assertEqual(self._loaded("substrata.measurements", watch), "[]")

    def test_compute_modules_load_no_figures(self):
        watch = ("matplotlib.pyplot", "fpdf", "substrata.visualizations")
        for module in ("cameras", "annotations", "pointclouds", "classification",
                       "color_calibration", "imagecrops"):
            with self.subTest(module=module):
                self.assertEqual(self._loaded(f"substrata.{module}", watch), "[]")


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestAliases(unittest.TestCase):
    """Moved functions stay reachable under their old names."""

    def test_crop_helpers(self):
        from substrata import classification, imagecrops

        self.assertIs(viz.get_crop_img, imagecrops.get_crop_img)
        self.assertIs(viz.encode_to_png_buffer, imagecrops.encode_to_png_buffer)
        self.assertIs(classification.classifier_crop_box,
                      imagecrops.classifier_crop_box)

    def test_benthic_fraction_helper(self):
        self.assertIs(m._benthic_fraction_from_results,
                      m.benthic_fraction_from_results)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestPackageLayout(unittest.TestCase):
    """``substrata.visualizations`` is a package behind one front module."""

    def test_front_exports_functions_not_submodules(self):
        import types

        for name in viz.__all__:
            obj = getattr(viz, name)
            with self.subTest(name=name):
                self.assertFalse(isinstance(obj, types.ModuleType)
                                 and obj.__name__.startswith("substrata."))

    def test_flat_namespace_keeps_the_real_modules(self):
        import substrata

        substrata._load_all()
        for name in ("segmentation", "measurements", "annotations"):
            with self.subTest(name=name):
                self.assertEqual(getattr(substrata, name).__name__,
                                 f"substrata.{name}")
        self.assertIs(substrata.tpi_figure, viz.tpi_figure)

    def test_private_helpers_reexported(self):
        for name in ("_resolve_scene_axes", "_symmetric_ticks", "_compact_colorbar",
                     "_radius_line_endpoint", "_classifier_example_pages"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(viz, name)))


if __name__ == "__main__":
    unittest.main()
