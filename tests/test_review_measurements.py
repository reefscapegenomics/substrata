"""Known-answer tests for :mod:`substrata.measurements` (review fixes).

Each test builds synthetic data whose answer is known analytically: gap
fractions of an opaque disk, plane fits without normals, PCA on degenerate
clouds, the depth-regression sign convention, TPI on a partial annulus, the
vector dispersion of ideal normals and the box search.

``measurements.py`` imports open3d/cv2/sklearn at module load, so the module is
skipped when those heavy deps are not installed.
"""

# Standard Library
import inspect
import types
import unittest

import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import geom
    from substrata import measurements as m
    from substrata import pointclouds
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    m = None

LOGGER = "substrata.logging"


def _grid(n=201, half=1.0):
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x)
    return X.ravel(), Y.ravel()


def _disk_canopy(n=200_000, height=1.0, seed=0):
    """An opaque horizontal disk of radius ``height`` at ``height``: zenith 0-45 deg."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n)) * height
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t), np.full(n, height)]


def _scene(points, rgb=(0.5, 0.5, 0.5)):
    return types.SimpleNamespace(
        points=points, colors=np.tile(np.asarray(rgb, float), (len(points), 1))
    )


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestGapFraction(unittest.TestCase):
    ann = types.SimpleNamespace(coords=np.zeros(3), id="t")

    def test_opaque_disk_to_45_degrees_fill_is_three_quarters(self):
        # Equidistant projection: sky between 45 and 90 deg zenith covers
        # 1 - (45/90)^2 = 0.75 of the image disk. The first seed lies on the
        # canopy and must not flood it.
        seeds = [(0.5, 0.5), (0.05, 0.5)]
        for color_output in (True, False):
            with self.assertLogs(LOGGER, level="WARNING"):
                raw, fill, _ = m.calc_gap_fraction(
                    self.ann, _scene(_disk_canopy()), resolution=200,
                    seed_points=seeds, color_output=color_output,
                )
            self.assertAlmostEqual(fill, 0.75, delta=0.015)
            self.assertAlmostEqual(raw, 0.75, delta=0.015)

    def test_grey_points_same_as_white_mode(self):
        # Benthos painted in the old display "sky" grey (80, 80, 80) must not be
        # flooded as sky.
        rng = np.random.default_rng(0)
        wall = np.column_stack([rng.uniform(0.5, 1.0, 4000),
                                rng.uniform(-1, 1, 4000),
                                rng.uniform(0.05, 1.0, 4000)])
        grey = _scene(wall, rgb=(80 / 255, 80 / 255, 80 / 255))
        raw_c, fill_c, _ = m.calc_gap_fraction(self.ann, grey, color_output=True)
        raw_w, fill_w, _ = m.calc_gap_fraction(self.ann, grey, color_output=False)
        self.assertEqual(raw_c, raw_w)
        self.assertEqual(fill_c, fill_w)
        self.assertLess(fill_c, 0.95)

    def test_gap_fractions_helper_connected_components(self):
        res = 101
        disk = m.hemispherical_disk(res)
        yy, xx = np.mgrid[0:res, 0:res] + 0.5
        r = np.hypot(xx - res / 2, yy - res / 2)
        cover = disk & (r > 20) & (r < 25)  # a closed ring
        inner = disk & (r <= 20)
        raw, fill, sky = m.gap_fractions(cover, disk, [(50, 50)])
        self.assertAlmostEqual(raw, (disk & ~cover).sum() / disk.sum())
        self.assertAlmostEqual(fill, inner.sum() / disk.sum())
        self.assertTrue(np.array_equal(sky, inner))
        # A seed on cover is skipped (warning); the other seed still counts.
        on_ring = (50 + 22, 50)
        with self.assertLogs(LOGGER, level="WARNING"):
            _, fill2, _ = m.gap_fractions(cover, disk, [on_ring, (50, 50)])
        self.assertAlmostEqual(fill2, fill)
        # 4-connectivity: a diagonal gap in a one-pixel wall does not leak.
        c = np.zeros((5, 5), bool)
        c[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]] = True
        _, f, s = m.gap_fractions(c, np.ones((5, 5), bool), [(0, 0)])
        self.assertEqual(int(s.sum()), 10)

    def test_hemispherical_cover_zenith_and_horizon(self):
        res = 200
        pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, -0.5]])  # zenith; below horizon
        cover, disk = m.hemispherical_cover(pts, np.zeros(3), res)
        self.assertEqual(int(cover.sum()), 1)
        self.assertTrue(cover[res // 2, res // 2])
        self.assertEqual(cover.shape, disk.shape)
        self.assertAlmostEqual(disk.sum() / (np.pi * (res / 2) ** 2), 1.0, delta=0.01)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestPlaneFitsWithoutNormals(unittest.TestCase):
    def setUp(self):
        X, Y = _grid(51)
        self.pts = np.c_[X, Y, 0.5 * X]  # elevation atan(0.5) = 26.565 deg

    def test_best_fit_plane_without_normals_points_up(self):
        for pcd in (types.SimpleNamespace(points=self.pts),
                    types.SimpleNamespace(points=self.pts[::-1]),
                    pointclouds.SimplePointCloud(self.pts)):
            a, b, c, d, _ = m.get_best_fit_plane_pca(pcd)
            self.assertGreater(c, 0)

    def test_plane_angles_without_normals(self):
        out = m.get_plane_angles(pointclouds.SimplePointCloud(self.pts),
                                 generate_image=False)
        theta, psi, elev, plane, az, img = out
        self.assertAlmostEqual(elev, np.degrees(np.arctan(0.5)), places=6)
        self.assertAlmostEqual(az, 180.0, places=6)
        # theta, psi: n = R_y(theta) R_x(psi) z_hat
        n = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
        t, p = np.radians(theta), np.radians(psi)
        ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0],
                       [-np.sin(t), 0, np.cos(t)]])
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)],
                       [0, np.sin(p), np.cos(p)]])
        np.testing.assert_allclose(ry @ rx @ [0, 0, 1], n, atol=1e-9)

    def test_plane_follows_normals_when_given(self):
        down = np.tile([0, 0, -1.0], (len(self.pts), 1))
        pcd = pointclouds.SimplePointCloud(self.pts, None, down)
        elev = m.get_plane_angles(pcd, generate_image=False)[2]
        self.assertAlmostEqual(elev, 180 - np.degrees(np.arctan(0.5)), places=6)

    def test_roughness_without_normals(self):
        x = np.linspace(-1, 1, 2001)[:-1]
        X2, Y2 = np.meshgrid(x, np.linspace(-1, 1, 41))
        A = 0.1
        P = np.c_[X2.ravel(), Y2.ravel(), (A * np.cos(2 * np.pi * X2 / 0.5)).ravel()]
        ra, rq, _ = m.calc_roughness(pointclouds.SimplePointCloud(P),
                                     generate_image=False)
        self.assertAlmostEqual(ra, 2 * A / np.pi, delta=1e-3)
        self.assertAlmostEqual(rq, A / np.sqrt(2), delta=1e-3)
        self.assertAlmostEqual(m.get_dev_rugosity(pointclouds.SimplePointCloud(P)),
                               ra, places=9)

    def test_ransac_on_simple_point_cloud(self):
        pcd = pointclouds.SimplePointCloud(self.pts)
        a, b, c, d, inliers = m.get_best_fit_plane_ransac(pcd)
        self.assertGreater(c, 0)
        self.assertEqual(len(inliers), len(self.pts))


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestConductPCA(unittest.TestCase):
    def test_real_sorted_orthonormal(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            n = int(rng.integers(2, 6))
            P = rng.normal(size=(n, 3)) * rng.choice([1, 1e-8], size=3)
            w, v = m.conduct_pca(types.SimpleNamespace(points=P))
            self.assertFalse(np.iscomplexobj(w) or np.iscomplexobj(v))
            self.assertTrue(np.all(np.diff(w) <= 1e-12))
            np.testing.assert_allclose(v.T @ v, np.eye(3), atol=1e-9)

    def test_columns_are_axes(self):
        rng = np.random.default_rng(1)
        P = rng.normal(size=(5000, 3)) * [3.0, 1.0, 0.1]
        w, v = m.conduct_pca(types.SimpleNamespace(points=P))
        self.assertAlmostEqual(abs(v[0, 0]), 1.0, places=2)
        self.assertAlmostEqual(abs(v[2, 2]), 1.0, places=2)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestDepthRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.P = rng.uniform(-5, 5, (200, 3))
        self.up = np.array([0.2, -0.1, 1.0]) / np.linalg.norm([0.2, -0.1, 1.0])
        self.elev = -20 + 0.8 * self.P @ self.up  # negative below the surface

    def test_negative_depths(self):
        r = m.fit_depth_regression(self.P, self.elev)
        np.testing.assert_allclose(r.up_vector, 0.8 * self.up, atol=1e-9)
        self.assertAlmostEqual(r.depth_offset, -20)
        self.assertFalse(r.depths_negated)

    def test_positive_depths_are_negated_with_warning(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            r = m.fit_depth_regression(self.P, -self.elev)
        np.testing.assert_allclose(r.up_vector, 0.8 * self.up, atol=1e-9)
        self.assertAlmostEqual(r.depth_offset, -20)
        self.assertTrue(r.depths_negated)
        np.testing.assert_allclose(r.depths, self.elev)
        np.testing.assert_allclose(r.depth_offset + self.P @ r.up_vector,
                                   r.depths_pred, atol=1e-9)


class _FakeContainer:
    def __init__(self, items, up_vector, depth_offset):
        self.data = {it.id: it for it in items}
        self.up_vector = up_vector
        self.depth_offset = depth_offset
        for it in items:
            it.parent = self

    def _empty_like(self):
        c = _FakeContainer([], self.up_vector, self.depth_offset)
        return c


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestDepthResidualAnalyzer(unittest.TestCase):
    def _items(self):
        return [types.SimpleNamespace(id=f"a{i}", coords=np.array([i, 0, -10.0 - i]),
                                      orig_coords=np.array([i, 0, -10.0 - i]),
                                      depth_sensor_m=-10.0 - i) for i in range(3)]

    def test_items_keep_their_parent(self):
        items = self._items()
        cont = _FakeContainer(items, np.array([0, 0, 1.0]), 0.0)
        an = m.DepthResidualAnalyzer(cont)
        d, est, filt = an.get_depths_and_estimated_depths()
        np.testing.assert_allclose(est, d)
        d2, z, filt2 = an.get_depths_and_z_coords()
        self.assertTrue(all(it.parent is cont for it in items))
        self.assertEqual(len(filt.data), 3)
        self.assertEqual(len(filt2.data), 3)

    def test_vector_up_vector_and_positive_depths(self):
        items = self._items()
        for it in items:
            it.depth_sensor_m = -it.depth_sensor_m  # positive below the surface
        cont = _FakeContainer(items, geom.Vector([0, 0, 1.0]), 0.0)
        d, est, _ = m.DepthResidualAnalyzer(cont).get_depths_and_estimated_depths()
        np.testing.assert_allclose(est, d)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestTPIPartialAnnulus(unittest.TestCase):
    def setUp(self):
        X, Y = _grid(201, 1.0)
        self.P = np.c_[X, Y, 0.5 * X]

    def test_annulus_coverage(self):
        xy = self.P[:, :2]
        full = m.annulus_coverage(xy, np.zeros(2), 0.1, 0.5)
        edge = m.annulus_coverage(xy, np.array([1.0, 0.0]), 0.1, 0.5)
        self.assertGreater(full, 0.99)
        self.assertAlmostEqual(edge, 0.5, delta=0.06)

    def test_partial_annulus_is_nan(self):
        c = np.array([0.9, 0.0, 0.45])
        with self.assertLogs(LOGGER, level="WARNING"):
            r = m.calc_tpi_and_tri(pointclouds.SimplePointCloud(self.P), center=c,
                                   radius_inner=0.1, radius_outer=0.5,
                                   generate_image=False)
        self.assertTrue(all(np.isnan(v) for v in r[:6]))
        # opt out: the old (biased) value
        r2 = m.calc_tpi_and_tri(pointclouds.SimplePointCloud(self.P), center=c,
                                radius_inner=0.1, radius_outer=0.5,
                                generate_image=False, min_annulus_coverage=None)
        self.assertGreater(abs(r2[0]), 0.05)

    def test_full_annulus_on_slope(self):
        r = m.calc_tpi_and_tri(pointclouds.SimplePointCloud(self.P),
                               center=np.zeros(3), radius_inner=0.1,
                               radius_outer=0.5, generate_image=False)
        self.assertAlmostEqual(r[0], 0.0, delta=1e-3)
        self.assertAlmostEqual(r[2], 0.0, delta=1e-9)
        self.assertAlmostEqual(r[5], 0.0, delta=1e-9)

    def test_area_weighted_by_default(self):
        # the downslope half of the annulus photographed 16x denser: weighting per
        # point pulls the annulus mean down (TPI > 0); 1 cm cells (the default) do not
        X, Y = (v + 0.0013 for v in _grid(201, 1.0))  # (no point exactly on a cell boundary)
        Xd, Yd = (v + 0.0013 for v in _grid(801, 1.0))
        keep = Xd < 0
        P = np.vstack([np.c_[X, Y, 0.5 * X], np.c_[Xd[keep], Yd[keep], 0.5 * Xd[keep]]])
        kw = dict(center=np.zeros(3), radius_inner=0.1, radius_outer=0.5, generate_image=False)
        self.assertEqual(m.TPI_RASTER_CELL_SIZE, 0.01)
        area = m.calc_tpi_and_tri(pointclouds.SimplePointCloud(P), **kw)
        point = m.calc_tpi_and_tri(pointclouds.SimplePointCloud(P), raster_cell_size=None, **kw)
        self.assertAlmostEqual(area[0], 0.0, delta=0.005)
        self.assertGreater(point[0], 0.05)
        with self.assertLogs(LOGGER, level="WARNING"):  # 5 mm cells on a 1 cm cloud
            m.calc_tpi_and_tri(pointclouds.SimplePointCloud(np.c_[X, Y, 0.5 * X]),
                               raster_cell_size=0.005, **kw)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestVectorDispersion(unittest.TestCase):
    def test_open3d_point_cloud(self):
        import open3d as o3d

        op = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(100, 3)))
        op.normals = o3d.utility.Vector3dVector(np.tile([0, 0, 1.0], (100, 1)))
        disp, img = m.get_vector_dispersion(op, generate_image=False)
        self.assertAlmostEqual(disp, 0.0, places=9)

    def test_known_value_and_legacy_keyword(self):
        # two opposite normals: R = 0 -> (N - R) / (N - 1) = 2
        n = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        pcd = pointclouds.SimplePointCloud(np.zeros((2, 3)), None, n)
        disp, _ = m.get_vector_dispersion(pcd, generate_image=False)
        self.assertAlmostEqual(disp, 2.0)
        self.assertAlmostEqual(
            m.get_vector_dispersion(geom=pcd, generate_image=False)[0], 2.0)

    def test_unsupported_type_message(self):
        with self.assertRaisesRegex(TypeError, "ndarray"):
            m.get_vector_dispersion(np.zeros((3, 3)), generate_image=False)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestBoxSearch(unittest.TestCase):
    def test_no_figure_by_default(self):
        sig = inspect.signature(m.find_optimal_box_position)
        self.assertIs(sig.parameters["vis"].default, False)

    def test_box_finds_dense_patch(self):
        rng = np.random.default_rng(0)
        P = np.vstack([rng.uniform(0, 10, (20000, 3)),
                       np.c_[rng.uniform(6, 8, (5000, 2)), np.zeros(5000)]])
        (x0, y0), (x1, y1) = m.find_optimal_box_position(
            pointclouds.SimplePointCloud(P), 2.0, 2.0, step_size=0.1)
        self.assertAlmostEqual(x1 - x0, 2.0)
        self.assertAlmostEqual(x0, 6.0, delta=0.15)
        self.assertAlmostEqual(y0, 6.0, delta=0.15)

    def test_length_not_a_multiple_of_step_warns(self):
        P = np.random.default_rng(0).uniform(0, 10, (2000, 3))
        with self.assertLogs(LOGGER, level="WARNING"):
            m.find_optimal_box_position(pointclouds.SimplePointCloud(P), 2.05, 1.0,
                                        step_size=0.1)

    def test_box_larger_than_cloud_starts_at_its_minimum(self):
        P = np.random.default_rng(0).uniform(0, 1, (200, 3))
        cloud = pointclouds.SimplePointCloud(P)
        for length, width in ((5.0, 5.0), (5.0, 0.5)):
            with self.assertLogs(LOGGER, level="WARNING"):
                (x0, y0), (x1, y1) = m.find_optimal_box_position(cloud, length, width)
            self.assertAlmostEqual(x0, P[:, 0].min())
            self.assertAlmostEqual(x1 - x0, length)
            self.assertAlmostEqual(y1 - y0, width)
        with self.assertRaises(ValueError):
            m.find_optimal_box_position(cloud, 0.01, 0.5)


@unittest.skipUnless(m is not None, "requires the open3d/substrata environment")
class TestPublicNamespace(unittest.TestCase):
    def test_all_lists_public_api_only(self):
        self.assertNotIn("plt", m.__all__)
        self.assertNotIn("Axes3D", m.__all__)
        self.assertNotIn("slerp", m.__all__)
        for name in ("conduct_pca", "calc_gap_fraction", "gap_fractions",
                     "hemispherical_cover", "calc_tpi_and_tri", "depth_band_sample",
                     "DepthResidualAnalyzer", "fit_depth_regression",
                     "find_optimal_box_position", "slope_normal_from_transform"):
            self.assertIn(name, m.__all__)
        for name in m.__all__:
            self.assertTrue(hasattr(m, name), name)


if __name__ == "__main__":
    unittest.main()
