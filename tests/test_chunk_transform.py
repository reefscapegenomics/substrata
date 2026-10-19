"""Tests for orienting a project from the Metashape chunk transform.

Loads ``geom.py`` and ``initializer.py`` directly (no open3d needed).
"""

# Standard Library
import json
import os
import sys
import tempfile
import types
import unittest

# Third-Party
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_initializer_ply_paths import _load_initializer  # noqa: E402

ProjectInitializer = _load_initializer()
import substrata.geom as geom  # noqa: E402  (registered by the loader)

# Reference location near Samoa (lat, lon)
LAT, LON = -13.4807, -172.5577


def _chunk_transform(scale=0.29, lat=LAT, lon=LON, h=-15.0, seed=0):
    """Random chunk-to-ECEF similarity transform centred on (lat, lon, h)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.eye(4)
    t[:3, :3] = scale * q
    t[:3, 3] = geom.geodetic_to_ecef(lat, lon, h)
    return t


class TestGeodesy(unittest.TestCase):
    def test_geodetic_round_trip(self):
        p = geom.geodetic_to_ecef(LAT, LON, -13.25)
        lat, lon, h = geom.ecef_to_geodetic(p)
        np.testing.assert_allclose([lat, lon, h], [LAT, LON, -13.25], atol=1e-6)

    def test_enu_axes(self):
        enu = geom.ecef_to_enu_transform(LAT, LON, 0.0)
        origin = geom.geodetic_to_ecef(LAT, LON, 0.0)
        up = geom.geodetic_to_ecef(LAT, LON, 10.0)
        north = geom.geodetic_to_ecef(LAT + 1e-4, LON, 0.0)
        np.testing.assert_allclose(geom.transform_coords(origin, enu), 0, atol=1e-6)
        np.testing.assert_allclose(
            geom.transform_coords(up, enu), [0, 0, 10], atol=1e-6
        )
        n = geom.transform_coords(north, enu)
        self.assertGreater(n[1], 10.0)
        self.assertAlmostEqual(n[0], 0.0, places=6)


class TestAlignYUpslope(unittest.TestCase):
    def test_diagonal_slope_maps_to_y(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(-20, 20, size=(1000, 2))
        z = 0.2 * xy[:, 0] + 0.2 * xy[:, 1]  # rises to the north-east
        pts = np.c_[xy, z]
        t = geom.Transform.align_y_upslope(pts, max_points=500)
        grad = t.mat[:3, :3] @ np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        np.testing.assert_allclose(grad, [0, 1, 0], atol=1e-9)

    @staticmethod
    def _strip(axis_deg, length=300.0, width=50.0, n=4000):
        """Rectangle footprint rotated to axis_deg, terrain rising to +Y."""
        rng = np.random.default_rng(5)
        u = rng.uniform(-length / 2, length / 2, n)
        v = rng.uniform(-width / 2, width / 2, n)
        t = np.radians(axis_deg)
        x = u * np.cos(t) - v * np.sin(t)
        y = u * np.sin(t) + v * np.cos(t)
        return np.c_[x, y, 0.5 * y]  # fall line exactly along +Y

    def _long_axis_deg(self, pts):
        xy = pts[:, :2] - pts[:, :2].mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov(xy.T))
        v = vecs[:, -1]
        return np.degrees(np.arctan2(v[1], v[0])) % 180

    def test_downslope_strip_snaps_to_its_own_axis(self):
        pts = self._strip(87.5)  # survey lines 2.5 deg off the fall line
        out = geom.Transform.align_y_upslope(pts).apply_to_points(pts)
        self.assertAlmostEqual(self._long_axis_deg(out), 90.0, places=1)
        self.assertGreater(np.corrcoef(out[:, 1], out[:, 2])[0, 1], 0.9)

    def test_contour_strip_long_axis_along_x(self):
        pts = self._strip(10.0)  # elongated across the slope
        out = geom.Transform.align_y_upslope(pts).apply_to_points(pts)
        ang = self._long_axis_deg(out)
        self.assertAlmostEqual(min(ang, 180 - ang), 0.0, places=1)
        # Still upslope along +Y: the fall line is 10 deg off +Y
        design = np.c_[out[:, :2], np.ones(len(out))]
        (gx, gy, _), *_ = np.linalg.lstsq(design, out[:, 2], rcond=None)
        self.assertGreater(gy, 0)
        self.assertAlmostEqual(np.degrees(np.arctan2(abs(gx), gy)), 10.0, delta=0.5)

    def test_no_snap_uses_exact_fall_line(self):
        pts = self._strip(87.5)
        t = geom.Transform.align_y_upslope(pts, snap_to_plot_axes=False)
        np.testing.assert_allclose(t.mat[:3, :3], np.eye(3), atol=1e-9)

    def test_flat_cloud_is_identity(self):
        pts = np.c_[np.random.default_rng(4).uniform(size=(50, 2)), np.zeros(50)]
        np.testing.assert_allclose(
            geom.Transform.align_y_upslope(pts).mat, np.eye(4)
        )


class TestChunkTransformToEnu(unittest.TestCase):
    def test_scale_and_heights(self):
        chunk = _chunk_transform(scale=0.29)
        transform, scale, (lat, lon) = geom.chunk_transform_to_enu(
            chunk, np.zeros(3)
        )
        self.assertAlmostEqual(scale, 0.29, places=12)
        self.assertAlmostEqual(lat, LAT, places=6)
        self.assertAlmostEqual(lon, LON, places=6)
        # Uniform scale preserved, rotation proper
        rot = transform[:3, :3] / scale
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        # Z equals the ellipsoidal height of mapped chunk points
        for pt in [np.zeros(3), np.array([30.0, -12.0, 5.0])]:
            _, _, h = geom.ecef_to_geodetic(geom.transform_coords(pt, chunk))
            z = geom.transform_coords(pt, transform)[2]
            self.assertAlmostEqual(z, h, places=3)


class _FakePcd:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)
        self.world_transform = np.eye(4)

    @property
    def points(self):
        return self._points

    @property
    def world_transform_is_identity(self):
        return np.allclose(self.world_transform, np.eye(4))

    def apply_transform(self, tm):
        tm = np.asarray(tm, dtype=float)
        hom = np.c_[self._points, np.ones(len(self._points))]
        self._points = (tm @ hom.T).T[:, :3]
        self.world_transform = tm @ self.world_transform


class TestPropagateDepthModel(unittest.TestCase):
    def _init(self):
        init = ProjectInitializer(path=tempfile.mkdtemp())
        for name in ("pcd", "cams", "markers", "annotations"):
            setattr(init, name, types.SimpleNamespace())
        return init

    def test_yaml_values_reach_all_objects(self):
        init = self._init()
        init.up_vector = geom.Vector([0.0, 0.1, -0.9])  # as init_with_yaml stores it
        init.depth_offset = 4.5
        init.depth_per_unit = 0.9
        init.propagate_depth_model()
        for name in ("pcd", "cams", "markers", "annotations"):
            obj = getattr(init, name)
            self.assertIsInstance(obj.up_vector, np.ndarray)
            np.testing.assert_allclose(obj.up_vector, [0.0, 0.1, -0.9])
            self.assertEqual(obj.depth_offset, 4.5)
        self.assertEqual(init.pcd.depth_scale_factor, 0.9)
        self.assertEqual(init.cams.depth_per_unit, 0.9)

    def test_unset_model_leaves_objects_alone(self):
        init = self._init()
        init.annotations = None  # missing objects are skipped
        init.propagate_depth_model()
        self.assertFalse(hasattr(init.cams, "up_vector"))


class TestOrientFromChunkTransform(unittest.TestCase):
    def _init(self, chunk, points=None):
        tmp = tempfile.mkdtemp()
        meta = os.path.join(tmp, "x.meta.json")
        with open(meta, "w") as f:
            json.dump({"chunk_transform": np.asarray(chunk).tolist()}, f)
        init = ProjectInitializer(path=tmp)
        init.cams_meta_json_filepath = meta
        rng = np.random.default_rng(1)
        if points is None:
            points = rng.uniform(-50, 50, size=(200, 3))
        init.pcd = _FakePcd(points)
        return init

    def test_orient_sets_scale_and_depth_model(self):
        init = self._init(_chunk_transform(scale=0.29))
        orig = init.pcd.points.copy()
        world = init.orient_from_chunk_transform()
        self.assertAlmostEqual(init.scale_factor, 0.29, places=12)
        self.assertAlmostEqual(init.depth_per_unit, 0.29, places=12)
        self.assertGreaterEqual(init.pcd.points[:, :2].min(), -1e-9)
        # Depth model reproduces z exactly
        z_pred = orig @ init.up_vector + init.depth_offset
        np.testing.assert_allclose(z_pred, init.pcd.points[:, 2], atol=1e-6)
        np.testing.assert_allclose(world, init.world_transform)
        # The depth model also lands on the point cloud
        np.testing.assert_allclose(init.pcd.up_vector, init.up_vector)
        self.assertEqual(init.pcd.depth_offset, init.depth_offset)
        self.assertEqual(init.pcd.depth_scale_factor, init.depth_per_unit)

    def _north_sloping(self, scale, long_axis="east"):
        """Chunk transform + cloud (chunk coords) that deepens to the north."""
        chunk = _chunk_transform(scale=scale)
        enu, _, _ = geom.chunk_transform_to_enu(chunk, np.zeros(3))
        rng = np.random.default_rng(2)
        e = rng.uniform(-40, 40, 500)
        n = rng.uniform(-10, 10, 500)
        if long_axis == "north":  # elongated downslope (e.g. AUV transects)
            e, n = n, e
        local = np.c_[e, n, -15.0 - 0.3 * n]  # deeper to the north
        to_chunk = np.linalg.inv(enu)
        pts = (to_chunk @ np.c_[local, np.ones(len(local))].T).T[:, :3]
        return chunk, pts

    def test_default_rotates_y_upslope(self):
        chunk, pts = self._north_sloping(0.29)
        init = self._init(chunk, pts)
        init.orient_from_chunk_transform()
        out = init.pcd.points
        self.assertGreater(np.corrcoef(out[:, 1], out[:, 2])[0, 1], 0.5)
        # Long (east-west) axis stays along X; depth and scale unchanged
        self.assertGreater(np.ptp(out[:, 0]), np.ptp(out[:, 1]))
        design = np.c_[out[:, :2], np.ones(len(out))]
        (gx, gy, _), *_ = np.linalg.lstsq(design, out[:, 2], rcond=None)
        # ~level along X (footprint axes of 500 random points are ~exact)
        self.assertAlmostEqual(gx, 0.0, places=2)
        self.assertAlmostEqual(gy, 0.3, places=4)  # rises along +Y
        np.testing.assert_allclose(init.world_transform[2, :3], init.up_vector)
        self.assertAlmostEqual(init.scale_factor, 0.29, places=12)
        z_pred = pts @ init.up_vector + init.depth_offset
        np.testing.assert_allclose(z_pred, out[:, 2], atol=1e-6)

    def test_downslope_elongated_plot_is_upright(self):
        chunk, pts = self._north_sloping(0.29, long_axis="north")
        init = self._init(chunk, pts)
        init.orient_from_chunk_transform()
        out = init.pcd.points
        # Shallow end at +Y even though the long axis runs downslope
        self.assertGreater(np.corrcoef(out[:, 1], out[:, 2])[0, 1], 0.9)
        self.assertGreater(np.ptp(out[:, 1]), np.ptp(out[:, 0]))

    def test_north_up_keeps_georeferenced_axes(self):
        chunk, pts = self._north_sloping(0.29)
        init = self._init(chunk, pts)
        init.orient_from_chunk_transform(north_up=True)
        out = init.pcd.points
        self.assertLess(np.corrcoef(out[:, 1], out[:, 2])[0, 1], -0.5)
        enu, _, _ = geom.chunk_transform_to_enu(chunk, pts.mean(axis=0))
        np.testing.assert_allclose(
            init.world_transform[:3, :3], enu[:3, :3], atol=1e-6
        )

    def test_missing_chunk_transform_raises(self):
        init = self._init(np.eye(4))
        with open(init.cams_meta_json_filepath, "w") as f:
            json.dump({"chunk_transform": None}, f)
        with self.assertRaises(ValueError):
            init.orient_from_chunk_transform()

    def test_local_crs_applied_as_is(self):
        chunk = np.diag([2.0, 2.0, 2.0, 1.0])
        init = self._init(chunk)
        init.orient_from_chunk_transform(north_up=True, shift_to_positive_xy=False)
        self.assertAlmostEqual(init.scale_factor, 2.0)
        np.testing.assert_allclose(init.world_transform, chunk)


if __name__ == "__main__":
    unittest.main()
