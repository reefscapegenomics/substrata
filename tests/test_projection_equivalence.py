"""One projection core for Camera.get_pixel_coords, Camera.project_points and pixel_hits.

The three functions share one projection core (cameras.py) and the stacked camera
arrays are cached on :class:`Cameras`. These tests compare them with frozen copies
of the three implementations as they were before (``_ref_*`` below, kept verbatim but for the relevance, in
consistent units since November 2026 (``cameras._relevance``), and the lens:
since 0.2 the pixels follow Metashape's frame camera model, here in ``_ref_pixels``, written out on its
own; tests/test_lens_metashape.py checks the model against Metashape itself):
same pixels, depths, relevances, in-view flags and types, on distorted and
undistorted cameras of several sensors, points behind the cameras, at depth 0, on
the image borders and on exact half pixels, in both frames. A timing of the old and
new code is printed (not asserted).
"""

# Standard Library
import importlib
import time
import unittest

# Third-Party Libraries
import numpy as np

try:  # Heavy deps (cv2, …) are only present in the conda envs.
    cm = importlib.import_module("substrata.cameras")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    cm = None


# --- The implementations before the projection core (verbatim copies) ---------------


def _ref_pixels(x, y, s):
    """Metashape's frame camera model, written out: normalised (x, y) -> pixels (u, v)."""
    k4, p3, p4, b2 = (getattr(s, a, 0.0) for a in ("k4", "p3", "p4", "b2"))
    r2 = x * x + y * y  # (Horner, in the core's order: the same floats to the last bit)
    radial = 1 + r2 * (s.k1 + r2 * (s.k2 + r2 * (s.k3 + r2 * k4)))
    tang = 1 + r2 * (p3 + r2 * p4)
    xd = x * radial + (s.p1 * (r2 + 2 * x * x) + 2 * s.p2 * x * y) * tang
    yd = y * radial + (s.p2 * (r2 + 2 * y * y) + 2 * s.p1 * x * y) * tang
    return s.fx * xd + b2 * yd + s.cx, s.fy * yd + s.cy


def _ref_get_pixel_coords(self, coords, use_orig_coords=False, required_to_be_in_view=True):
    if use_orig_coords:
        cam_coords = self.orig_coords
        cam_transform = self.orig_camera_transform
    else:
        cam_coords = self.coords
        cam_transform = self.camera_transform
    cam_transform = np.array(cam_transform, dtype=float).reshape((4, 4))
    proj_point = np.dot(np.linalg.inv(cam_transform), np.append(coords, 1))
    x_norm = proj_point[0] / proj_point[2]
    y_norm = proj_point[1] / proj_point[2]
    r2 = x_norm**2 + y_norm**2
    x_img, y_img = _ref_pixels(x_norm, y_norm, self.sensor)
    in_view = (
        proj_point[2] > 0
        and r2 * self.sensor.fx**2 < 1.01 * self.sensor.width**2
        and 0 <= x_img < self.sensor.width
        and 0 <= y_img < self.sensor.height
    )
    if in_view or not required_to_be_in_view:
        dist_sq = np.sum((cam_coords - coords) ** 2)
        rm = dist_sq / (self.sensor.fx * self.sensor.fy) * (
            10 + np.abs(x_img - 0.5 * self.sensor.width) + np.abs(y_img - 0.5 * self.sensor.height))
        if in_view:  # (rounded, at most the last pixel: 0 <= x < width)
            return (min(int(round(x_img)), int(self.sensor.width) - 1),
                    min(int(round(y_img)), int(self.sensor.height) - 1), float(proj_point[2]), rm)
        return (int(round(x_img)), int(round(y_img)), float(proj_point[2]), rm)
    return None, None, None, None


def _ref_project_points(self, coords, use_orig_coords=False):
    pts = np.asarray(coords, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    if use_orig_coords:
        cam_coords = self.orig_coords
        cam_transform = self.orig_camera_transform
    else:
        cam_coords = self.coords
        cam_transform = self.camera_transform
    cam_transform = np.array(cam_transform, dtype=float).reshape((4, 4))
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = homog @ np.linalg.inv(cam_transform).T
    depth = proj[:, 2]
    x_norm = proj[:, 0] / depth
    y_norm = proj[:, 1] / depth
    s = self.sensor
    r2 = x_norm**2 + y_norm**2
    x_img, y_img = _ref_pixels(x_norm, y_norm, s)
    in_view = ((depth > 0) & (r2 * s.fx**2 < 1.01 * s.width**2)
               & (x_img >= 0) & (x_img < s.width) & (y_img >= 0) & (y_img < s.height))
    cam_center = np.asarray(cam_coords, dtype=float).ravel()[:3]
    dist_sq = np.sum((cam_center - pts) ** 2, axis=1)
    relevance = dist_sq / (s.fx * s.fy) * (
        10 + np.abs(x_img - 0.5 * s.width) + np.abs(y_img - 0.5 * s.height))
    x_px = np.where(in_view, np.minimum(np.rint(x_img), s.width - 1), np.rint(x_img))
    y_px = np.where(in_view, np.minimum(np.rint(y_img), s.height - 1), np.rint(y_img))
    if single:
        return (float(x_px[0]), float(y_px[0]), float(depth[0]),
                float(relevance[0]), bool(in_view[0]))
    return x_px, y_px, depth, relevance, in_view


def _ref_stack(cams, cache=None):
    n = len(cams)  # (the old cache: the last list of cameras; the timing uses it)
    key = (n, tuple(id(c) for c in cams[:: max(1, n // 64)]), id(cams[-1]))
    if cache is not None and key in cache:
        return cache[key]
    try:
        S = [c.sensor for c in cams]
        par = {a: np.array([float(getattr(s, a)) for s in S]) for a in
               ("k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4", "b2", "fx", "fy", "cx", "cy", "width", "height")}
        inv = np.linalg.inv(np.stack([np.asarray(c.orig_camera_transform, float).reshape(4, 4) for c in cams]))
        centres = np.stack([np.asarray(c.orig_coords, float).reshape(3) for c in cams])
    except (AttributeError, TypeError, ValueError):
        return None
    got = dict(cams=list(cams), inv=inv, centres=centres, **par)
    if cache is not None:
        cache.clear()
        cache[key] = got
    return got


def _ref_pixel_hits(cams, coords, use_orig_coords=True, enabled_only=True, cache=None):
    cams = cams if isinstance(cams, list) else list(cams)
    if not cams:
        return []
    st = _ref_stack(cams, cache) if use_orig_coords else None
    if st is None:
        out = []
        for cam in cams:
            if enabled_only and getattr(cam, "enabled", True) is False:
                continue
            x, y, depth, rel = _ref_get_pixel_coords(cam, coords, use_orig_coords=use_orig_coords)
            if x is not None:
                out.append((cam, x, y, depth, rel))
        return out
    p = np.append(np.asarray(coords, float).ravel()[:3], 1.0)
    proj = st["inv"] @ p
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = proj[:, 0] / proj[:, 2], proj[:, 1] / proj[:, 2]
        r2 = xn ** 2 + yn ** 2
        from types import SimpleNamespace
        xi, yi = _ref_pixels(xn, yn, SimpleNamespace(**{k: v for k, v in st.items() if isinstance(v, np.ndarray)}))
        ok = ((proj[:, 2] > 0) & (r2 * st["fx"] ** 2 < 1.01 * st["width"] ** 2)
              & (xi >= 0) & (xi < st["width"]) & (yi >= 0) & (yi < st["height"]))
    idx = np.flatnonzero(ok)
    if enabled_only:
        idx = np.array([i for i in idx.tolist() if getattr(st["cams"][i], "enabled", True) is not False], int)
    if not len(idx):
        return []
    dist_sq = ((st["centres"][idx] - p[:3]) ** 2).sum(1)
    rm = (dist_sq / (st["fx"][idx] * st["fy"][idx])
          * (10 + np.abs(xi[idx] - 0.5 * st["width"][idx]) + np.abs(yi[idx] - 0.5 * st["height"][idx])))
    return [(st["cams"][i], min(int(round(xi[i])), int(st["width"][i]) - 1),
             min(int(round(yi[i])), int(st["height"][i]) - 1), float(proj[i, 2]), r)
            for i, r in zip(idx.tolist(), rm.tolist())]


# --- Synthetic cameras and points ----------------------------------------------------

_DISTORTED = {"k1": -0.1, "k2": 0.02, "k3": -0.003, "p1": 1e-4, "p2": -2e-4}
_NONE = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}


def _sensors():
    """Two distorted sensors (one with fx != fy) and one ideal pinhole."""
    return [
        cm.Sensor(0, "auv", {"width": 4000, "height": 3000},
                  {"f": 3500.0, "cx": 3.0, "cy": -2.0, **_DISTORTED}),
        cm.Sensor(1, "macro", {"width": 2848, "height": 2816},
                  {"f": 3100.0, "cx": -11.5, "cy": 7.25, "b1": 4.0, "b2": -1.5, "k4": 4e-4,
                   "p3": 0.1, "p4": -0.02, **_DISTORTED}),
        cm.Sensor(2, "pinhole", {"width": 4000, "height": 3000},
                  {"f": 1024.0, "cx": 0.0, "cy": 0.0, **_NONE}),
    ]


def _world():
    """A non-trivial world transform (rotation about z, scale 1, shift)."""
    a = 0.4
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = [100.0, -50.0, 3.0]
    return T


def _cameras(n=120, seed=0, parent=True):
    """Cameras looking down (z towards -z) over a 20 m square; with ``parent``
    they live in a :class:`Cameras` with a world transform applied, as loaded."""
    rng = np.random.default_rng(seed)
    sensors = _sensors()
    cams_obj = cm.Cameras() if parent else None
    cams = []
    for i in range(n):
        a, b = rng.normal(0, 0.3, 2)
        Rx = np.array([[1, 0, 0], [0, np.cos(np.pi + a), -np.sin(np.pi + a)],
                       [0, np.sin(np.pi + a), np.cos(np.pi + a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        T = np.eye(4)
        T[:3, :3] = Ry @ Rx
        T[:3, 3] = [*rng.uniform(0, 20, 2), rng.uniform(1, 4)]
        transform = T.tolist() if i % 3 == 0 else T  # (as loaded from JSON: nested lists)
        c = cm.Camera(parent=cams_obj, cam_id=str(i), camera_transform=transform,
                      coords=T[:3, 3].copy(), path=f"/photos/{i}.jpg")
        c.sensor = sensors[i % 3]
        c.enabled = i % 17 != 0
        if parent:
            cams_obj.data[c.cam_id] = c
        cams.append(c)
    if parent:
        cams_obj.apply_transform(_world())
    return cams


def _points(seed=1, n=60):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-2, 22, (n, 2)), rng.uniform(-0.5, 0.5, n)]


def _edge_case_camera():
    """A pinhole camera at the origin looking along +z (identity pose)."""
    c = cm.Camera(cam_id="edge", camera_transform=np.eye(4), coords=np.zeros(3),
                  path="/photos/edge.jpg")
    c.sensor = _sensors()[2]  # 4000 x 3000, f 1024, centre (2000, 1500)
    return c


def _edge_case_points():
    """Points on the borders, just past them, on exact half pixels, behind the
    camera, at depth 0, beyond the field-of-view radius."""
    f = 1024.0
    return np.array([
        [2000 / f, 0.0, 1.0],          # x = 4000 exactly (right border: out of view, 0 <= x < width)
        [-2000 / f, 0.0, 1.0],         # x = 0 exactly (left border: in view)
        [0.0, 1500 / f, 1.0],          # y = 3000 exactly (bottom border: out of view)
        [0.0, -1500 / f, 1.0],         # y = 0 exactly (top border: in view)
        [2000 / f, 1500 / f, 1.0],     # the far corner (out of view)
        [2001 / f, 0.0, 1.0],          # one pixel past the right border
        [0.0, -1501 / f, 1.0],         # one pixel above the top
        [0.5 / f, 0.5 / f, 1.0],       # 2000.5, 1500.5 (half pixels: rounded to even)
        [1.5 / f, -2.5 / f, 1.0],      # 2001.5, 1497.5
        [0.0, 0.0, -1.0],              # behind the camera (projects to the centre)
        [1.0, 0.5, -3.0],              # behind, off-centre
        [0.3, 0.2, 0.0],               # depth 0 (in the camera plane)
        [0.0, 0.0, 1e-12],             # just in front
        [5.0, 0.0, 1.0],               # far outside
    ])


def _same(a, b):
    """Exactly equal values and types (NaN equal to NaN)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, (float, np.floating)) and np.isnan(a):
        return bool(np.isnan(b))
    return a == b


def _same_hits(old, new, world=False):
    """Same cameras, pixels, depths and relevances. In the world frame the old code went
    one camera at a time (relevance an np.float64); now stacked, as the orig frame
    always was (a float): the same value, a float either way."""
    def same_rel(u, v):
        return _same(float(u), v) if world and type(u) is np.float64 else _same(u, v)
    return len(old) == len(new) and all(
        len(a) == len(b) and a[0] is b[0] and all(_same(u, v) for u, v in zip(a[1:4], b[1:4]))
        and same_rel(a[4], b[4]) for a, b in zip(old, new))


@unittest.skipIf(cm is None, "substrata.cameras not importable")
class TestProjectionEquivalence(unittest.TestCase):
    def setUp(self):
        self._err = np.seterr(all="ignore")  # (depth-0 points divide by zero, as before)

    def tearDown(self):
        np.seterr(**self._err)

    def test_get_pixel_coords(self):
        cams = _cameras(60) + [_edge_case_camera()]
        pts = np.vstack([_points(), _edge_case_points()])
        n_in = 0
        for c in cams:
            for p in pts:
                for orig in (True, False):
                    for req in (True, False):
                        try:
                            old = _ref_get_pixel_coords(c, p, orig, req)
                        except ValueError:  # (depth 0, not required in view: int(nan) raises, before and after)
                            with self.assertRaises(ValueError):
                                c.get_pixel_coords(p, use_orig_coords=orig, required_to_be_in_view=req)
                            continue
                        new = c.get_pixel_coords(p, use_orig_coords=orig, required_to_be_in_view=req)
                        self.assertEqual(len(old), len(new))
                        for u, v in zip(old, new):
                            self.assertTrue(_same(u, v), (c.cam_id, p, orig, req, old, new))
                        n_in += req and old[0] is not None
            # a list point (not an array) as some callers pass
            p = list(map(float, pts[0]))
            self.assertTrue(all(_same(u, v) for u, v in zip(
                _ref_get_pixel_coords(c, p, True), c.get_pixel_coords(p, use_orig_coords=True))))
        self.assertGreater(n_in, 200)  # (a real comparison: many points are seen)

    def test_project_points(self):
        cams = _cameras(60) + [_edge_case_camera()]
        pts = np.vstack([_points(), _edge_case_points()])
        for c in cams:
            for orig in (True, False):
                old = _ref_project_points(c, pts, orig)
                new = c.project_points(pts, use_orig_coords=orig)
                for u, v in zip(old, new):
                    self.assertTrue(_same(u, v), (c.cam_id, orig))
                for p in pts[[0, 5, -3, -1]]:  # single points: Python scalars
                    old1 = _ref_project_points(c, p, orig)
                    new1 = c.project_points(p, use_orig_coords=orig)
                    self.assertTrue(all(_same(u, v) for u, v in zip(old1, new1)), (old1, new1))
        # the edge cases, explicitly
        x, y, d, _r, iv = _edge_case_camera().project_points(_edge_case_points())
        self.assertEqual(iv.tolist()[:9], [False, True, False, True, False, False, False, True, True])
        self.assertEqual((x[7], y[7], x[8], y[8]), (2000.0, 1500.0, 2002.0, 1498.0))
        self.assertFalse(iv[9] or iv[10] or iv[11])

    def test_pixel_hits(self):
        for parent in (True, False):
            cm.invalidate_projection_cache()
            cams = _cameras(150, parent=parent) + [_edge_case_camera()]
            pts = np.vstack([_points(n=40), _edge_case_points()])
            seen = 0
            for p in pts:
                for orig in (True, False):
                    for enabled_only in (True, False):
                        old = _ref_pixel_hits(cams, p, orig, enabled_only)
                        new = cm.pixel_hits(cams, p, use_orig_coords=orig, enabled_only=enabled_only)
                        self.assertTrue(_same_hits(old, new, not orig), (parent, p, orig, enabled_only))
                        seen += len(old)
            self.assertGreater(seen, 100)
            # a generator of cameras, and the same cameras through Cameras
            p = pts[3]
            self.assertTrue(_same_hits(_ref_pixel_hits(iter(cams), p), cm.pixel_hits(iter(cams), p)))
        cams = _cameras(150)
        parent = cams[0].parent
        self.assertTrue(_same_hits(_ref_pixel_hits(parent, pts[3]), cm.pixel_hits(parent, pts[3])))

    def test_stack_cached_on_cameras_and_rebuilt_on_change(self):
        cams = _cameras(90)
        parent = cams[0].parent
        p = _points()[4]
        st = parent.projection_arrays()
        self.assertIs(st, parent.projection_arrays())  # (cached)
        self.assertEqual(st.inv.shape, (90, 4, 4))
        cm.pixel_hits(cams, p)
        hit = next(h[0] for h in cm.pixel_hits(cams, p))
        # moving a camera (its orig pose, in place) restacks
        T = np.array(hit.camera_transform, float).reshape(4, 4)
        T[:3, 3] += [50.0, 0, 0]
        hit.camera_transform, hit.coords = T, T[:3, 3].copy()
        hit.reverse_transform_coords(_world())
        self.assertIsNot(st, parent.projection_arrays())
        self.assertNotIn(hit, [h[0] for h in cm.pixel_hits(cams, p)])
        self.assertTrue(_same_hits(_ref_pixel_hits(cams, p), cm.pixel_hits(cams, p)))
        # a world transform restacks the world frame
        parent.apply_transform(_world())
        for orig in (True, False):
            self.assertTrue(_same_hits(_ref_pixel_hits(cams, p, orig), cm.pixel_hits(cams, p, orig), not orig))
        # replacing a camera in the container restacks
        st = parent.projection_arrays()
        parent["0"] = cams[0]
        self.assertIsNot(st, parent.projection_arrays())

    def test_cameras_that_cannot_project_see_nothing(self):
        """A camera without a calibrated sensor (no <calibration> in cams.xml) or with a placeholder pose (null
        in meta.json) sees nothing; the others still go through the stack (sa3_60m's crop job stopped on one)."""
        cams = _cameras(60)
        parent = cams[0].parent
        p = _points()[4]
        hits = [h[0] for h in cm.pixel_hits(cams, p)]
        self.assertGreaterEqual(len(hits), 2)
        no_sensor, placeholder = hits[0], hits[1]
        no_sensor.sensor = None
        placeholder.missing_pose_from_meta = True  # (its pose: the identity, at the origin, as loaded)
        cm.invalidate_projection_cache()
        for c in (no_sensor, placeholder):
            self.assertFalse(cm.can_project(c))
            self.assertEqual(c.get_pixel_coords(p, use_orig_coords=True), (None, None, None, None))
            x, y, d, r, ok = c.project_points(np.array([p, p]), use_orig_coords=True)
            self.assertTrue(np.isnan(x).all() and np.isnan(d).all() and not ok.any())
            self.assertFalse(c.project_points(p, use_orig_coords=True)[4])
        rest = [c for c in cams if c not in (no_sensor, placeholder)]
        for orig in (True, False):
            got = cm.pixel_hits(cams, p, orig)
            self.assertTrue(_same_hits(_ref_pixel_hits(rest, p, orig), got, not orig))
            self.assertEqual(len(parent.projection_arrays(orig)), len(rest))  # (stacked, not one at a time)
        self.assertEqual(cm.pixel_hits([no_sensor, placeholder], p), [])

    def test_timing(self):
        """Old vs new timings (printed, not asserted)."""
        cams = _cameras(3000)
        pts = _points(n=50)
        cm.pixel_hits(cams, pts[0])  # (stack once, old and new)
        cache = {}
        _ref_pixel_hits(cams, pts[0], cache=cache)

        def clock(fn, reps):
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t) / reps * 1e6

        c = cams[1]
        rows = [
            ("get_pixel_coords, 1 point", clock(lambda: _ref_get_pixel_coords(c, pts[0], True), 2000),
             clock(lambda: c.get_pixel_coords(pts[0], use_orig_coords=True), 2000)),
            ("project_points, 10k points", clock(lambda: _ref_project_points(c, np.tile(pts, (200, 1)), True), 50),
             clock(lambda: c.project_points(np.tile(pts, (200, 1)), use_orig_coords=True), 50)),
            ("pixel_hits, 3000 cams (orig)", clock(lambda: _ref_pixel_hits(cams, pts[0], cache=cache), 20),
             clock(lambda: cm.pixel_hits(cams, pts[0]), 20)),
            ("pixel_hits, 3000 cams (world)", clock(lambda: _ref_pixel_hits(cams, pts[0], False), 3),
             clock(lambda: cm.pixel_hits(cams, pts[0], False), 20)),
        ]
        print("\nprojection timings (us per call, old -> new):")
        for name, old, new in rows:
            print(f"  {name:32s} {old:10.1f} -> {new:10.1f}")


if __name__ == "__main__":
    unittest.main()
