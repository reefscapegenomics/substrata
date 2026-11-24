"""Regression tests for the point-cloud / stratified / ortho review fixes.

``pointclouds.py`` is loaded from ``src/substrata`` with its heavy siblings
(annotations, visualizations) stubbed, as in ``test_pointclouds.py``; it imports
the real ``stratified.py``. ``ortho.py`` is loaded directly (numpy/Pillow only).
Synthetic PLYs are small and written to a temporary directory that is removed.
"""

# Standard Library
import importlib
import importlib.util
import logging
import os
import shutil
import signal
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path

# Third-Party Libraries
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_modules():
    """Load pointclouds (stubbed siblings), stratified and ortho from src."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    before = set(sys.modules)
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    def _load(mod_name, rel):
        spec = importlib.util.spec_from_file_location(mod_name, _SUB / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    stubs = {}
    for name in ("substrata.annotations", "substrata.visualizations"):
        if name not in sys.modules:
            stubs[name] = types.ModuleType(name)
    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")
        log_mod.logger = logging.getLogger("substrata-test")
        stubs["substrata.logging"] = log_mod
    sys.modules.update(stubs)
    if "substrata.settings" not in sys.modules:
        _load("substrata.settings", "settings.py")
    if "substrata.geom" not in sys.modules:
        _load("substrata.geom", "geom.py")
    # Fresh copies bound to the stubs (and to the src files under test)
    pc = _load("substrata._review_pointclouds", "pointclouds.py")
    st = importlib.import_module("substrata.stratified")
    ortho = _load("substrata._review_ortho", "ortho.py")
    for name in set(sys.modules) - before:
        if name in stubs:
            sys.modules.pop(name, None)
    return pc, st, ortho


pcmod, st, ortho = _load_modules()
import open3d as o3d  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_ply(path, xyz, extra=(), comments=(), fmt="binary_little_endian",
               pre_elements=b"", post_elements=b""):
    """Write a binary PLY with float32 xyz plus `extra` (name, ply type, values)."""
    endian = "<" if fmt == "binary_little_endian" else ">"
    types_np = {"uchar": "u1", "ushort": "u2", "float": "f4", "double": "f8"}
    fields = [("x", endian + "f4"), ("y", endian + "f4"), ("z", endian + "f4")]
    fields += [(name, endian + types_np[t]) for name, t, _ in extra]
    rec = np.zeros(len(xyz), np.dtype(fields))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, _, values in extra:
        rec[name] = values
    head = b"ply\n" + f"format {fmt} 1.0\n".encode()
    for c in comments:
        head += b"comment " + (c if isinstance(c, bytes) else c.encode()) + b"\n"
    head += pre_elements
    head += f"element vertex {len(xyz)}\n".encode()
    head += b"property float x\nproperty float y\nproperty float z\n"
    for name, t, _ in extra:
        head += f"property {t} {name}\n".encode()
    head += post_elements + b"end_header\n"
    with open(path, "wb") as f:
        f.write(head)
        f.write(rec.tobytes())
    return rec


def _rgb_extra(rgb, ptype):
    return [(c, ptype, rgb[:, i]) for i, c in enumerate(("red", "green", "blue"))]


def _make_pc(points, colors=None, normals=None):
    pc = pcmod.PointCloud()
    pc.o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    if colors is not None:
        pc.o3d_pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    if normals is not None:
        pc.o3d_pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, float))
    return pc


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)


# ---------------------------------------------------------------------------
# P1 / P11: colours decoded as stratified.ply_rgb for every PLY colour type
# ---------------------------------------------------------------------------


class TestColourDecoding(_TmpDir):
    def _cases(self):
        rng = np.random.default_rng(3)
        rgb8 = rng.integers(0, 256, (50, 3))
        return {
            "uchar": rgb8,
            "ushort": rgb8 * 257,  # 16-bit colours
            "float": rgb8 / 255.0,  # colours in [0, 1]
        }

    def test_stream_sample_colours_all_types(self):
        xyz = np.random.default_rng(0).uniform(0, 1, (50, 3))
        for ptype, values in self._cases().items():
            with self.subTest(ptype=ptype):
                p = self.path(f"c_{ptype}.ply")
                rec = _write_ply(p, xyz, _rgb_extra(values, ptype))
                pts, cols, _ = pcmod._stream_sample_ply_to_arrays(
                    p, 50, progress=False)
                self.assertEqual(len(pts), 50)
                expect = st.ply_rgb(rec) / 255.0
                order = np.lexsort(pts.T)
                order_e = np.lexsort(np.asarray(xyz, np.float32).astype(float).T)
                np.testing.assert_allclose(cols[order], expect[order_e], atol=1e-6)

    def test_neighbourhood_colours_all_types(self):
        xyz = np.random.default_rng(0).uniform(0, 1, (50, 3))
        for ptype, values in self._cases().items():
            with self.subTest(ptype=ptype):
                p = self.path(f"n_{ptype}.ply")
                rec = _write_ply(p, xyz, _rgb_extra(values, ptype))
                (nb,) = pcmod.stream_extract_neighborhoods_ply(
                    p, [np.zeros(3)], 10.0, progress=False)
                np.testing.assert_allclose(
                    nb.colors, st.ply_rgb(rec) / 255.0, atol=1e-6)


# ---------------------------------------------------------------------------
# P2 / P3 / P4: one header parser
# ---------------------------------------------------------------------------


class TestHeaderParsing(_TmpDir):
    def test_non_empty_element_before_vertex_rejected(self):
        p = self.path("pre.ply")
        _write_ply(p, np.zeros((3, 3)),
                   pre_elements=b"element camera 1\nproperty float k\n")
        with open(p, "rb") as f, self.assertRaises(ValueError):
            pcmod._parse_ply_header(f)

    def test_non_empty_element_after_vertex_rejected(self):
        p = self.path("post.ply")
        _write_ply(p, np.zeros((3, 3)),
                   post_elements=b"element face 2\n"
                                 b"property list uchar int vertex_indices\n")
        with open(p, "rb") as f, self.assertRaises(ValueError):
            pcmod._parse_ply_header(f)

    def test_empty_other_elements_allowed(self):
        p = self.path("empty_face.ply")
        _write_ply(p, np.zeros((3, 3)),
                   post_elements=b"element face 0\n"
                                 b"property list uchar int vertex_indices\n")
        with open(p, "rb") as f:
            fmt, endian, n, props, rec_size, lines = pcmod._parse_ply_header(f)
            self.assertEqual(f.tell(), os.path.getsize(p) - 3 * 12)
        self.assertEqual((fmt, endian, n, rec_size),
                         ("binary_little_endian", "<", 3, 12))

    def test_latin1_comment(self):
        p = self.path("latin1.ply")
        _write_ply(p, np.zeros((4, 3)), comments=[b"site Cura\xe7ao"])
        h = st.read_ply_header(p)
        self.assertEqual(h.n_vertices, 4)
        with open(p, "rb") as f:
            self.assertEqual(pcmod._parse_ply_header(f)[2], 4)

    def test_ply_head_binary_and_ascii(self):
        p = self.path("head.ply")
        xyz = np.arange(12, dtype=float).reshape(4, 3)
        _write_ply(p, xyz)
        recs = pcmod.ply_head(p, n=2, print_output=False)
        self.assertEqual(len(recs), 2)
        self.assertAlmostEqual(recs[1]["z"], 5.0)
        a = self.path("ascii.ply")
        with open(a, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
                    "property float y\nproperty float z\nend_header\n1 2 3\n")
        with self.assertRaisesRegex(ValueError, "binary"):
            pcmod.ply_head(a, print_output=False)


# ---------------------------------------------------------------------------
# P5 / P6: seeded, exact-size decimation
# ---------------------------------------------------------------------------


class TestSeededDecimation(_TmpDir):
    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(1)
        self.xyz = rng.uniform(0, 1, (2000, 3))
        self.src = self.path("src.ply")
        self.rec = _write_ply(self.src, self.xyz,
                              _rgb_extra(rng.integers(0, 256, (2000, 3)), "uchar"),
                              comments=["made by test"])

    def _read(self, p):
        h = st.read_ply_header(p)
        return h, np.asarray(st.open_ply_records(h))

    def test_decimate_is_reproducible_and_exact(self):
        a, b, c = self.path("a.ply"), self.path("b.ply"), self.path("c.ply")
        kw = dict(progress=False, chunk_bytes=1500)  # many chunks
        pcmod.decimate_ply_file(self.src, a, 300, **kw)
        pcmod.decimate_ply_file(self.src, b, 300, **kw)
        pcmod.decimate_ply_file(self.src, c, 300, seed=7, **kw)
        with open(a, "rb") as fa, open(b, "rb") as fb, open(c, "rb") as fc:
            ba, bb, bc = fa.read(), fb.read(), fc.read()
        self.assertEqual(ba, bb)
        self.assertNotEqual(ba, bc)
        h, rec = self._read(a)
        self.assertEqual(h.n_vertices, 300)
        self.assertEqual(len(rec), 300)
        # a subset of the source records, in file order
        v = f"V{h.record_size}"
        src_v = self.rec.view(v)
        pos = np.array([np.flatnonzero(src_v == r)[0] for r in rec.view(v)])
        self.assertTrue((np.diff(pos) > 0).all())
        text = b"".join(h.lines).decode("latin-1")
        self.assertIn("seed 0", text)
        self.assertIn("comment made by test", text)
        self.assertNotIn("reservoir", text)
        self.assertEqual(len(o3d.io.read_point_cloud(a).points), 300)

    def test_decimate_is_uniform(self):
        # inclusion frequency over many seeds is flat along the file
        hits = np.zeros(2000)
        for seed in range(40):
            out = self.path("u.ply")
            pcmod.decimate_ply_file(self.src, out, 500, progress=False,
                                    chunk_bytes=3000, seed=seed)
            _, rec = self._read(out)
            v = f"V{rec.dtype.itemsize}"
            mask = np.isin(self.rec.view(v), rec.view(v))
            hits += mask
        blocks = hits.reshape(10, 200).mean(1) / 40
        np.testing.assert_allclose(blocks, 0.25, atol=0.03)

    def test_stream_sample_is_reproducible_and_exact(self):
        a = pcmod._stream_sample_ply_to_arrays(self.src, 333, progress=False,
                                               chunk_bytes=1500)
        b = pcmod._stream_sample_ply_to_arrays(self.src, 333, progress=False,
                                               chunk_bytes=1500)
        c = pcmod._stream_sample_ply_to_arrays(self.src, 333, progress=False,
                                               chunk_bytes=1500, seed=5)
        self.assertEqual(len(a[0]), 333)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(a[0], c[0]))


# ---------------------------------------------------------------------------
# P8: subsample_pointcloud_by_radius
# ---------------------------------------------------------------------------


class TestSubsampleByRadius(unittest.TestCase):
    def setUp(self):
        g = np.mgrid[0:11, 0:11].reshape(2, -1).T * 0.1
        self.pts = np.column_stack([g, np.zeros(len(g))])

    def test_keeps_nearest_point_off_cloud(self):
        pc = _make_pc(self.pts)
        q = np.array([0.5, 0.5, 0.01])  # not a cloud point
        out = pc.subsample_pointcloud_by_radius(q, 0.05)
        self.assertEqual(len(out.points), 1)
        np.testing.assert_allclose(out.points[0], [0.5, 0.5, 0.0])
        out = pc.subsample_pointcloud_by_radius(q, 0.05, cylinder=True)
        self.assertEqual(len(out.points), 1)

    def test_query_point_itself_is_excluded(self):
        pc = _make_pc(self.pts)
        out = pc.subsample_pointcloud_by_radius(np.array([0.5, 0.5, 0.0]), 0.11)
        self.assertEqual(len(out.points), 4)  # the 4 grid neighbours
        out = pc.subsample_pointcloud_by_radius(
            np.array([0.5, 0.5, 0.0]), 0.11, exclude_query_point=False)
        self.assertEqual(len(out.points), 5)

    def test_without_colours_or_normals(self):
        pc = _make_pc(self.pts)
        out = pc.subsample_pointcloud_by_radius(np.array([0.5, 0.5, 0.0]), 0.25)
        self.assertGreater(len(out.points), 4)
        self.assertIsNone(getattr(out, "colors", None))
        self.assertIsNone(getattr(out, "normals", None))

    def test_with_colours_and_normals_aligned(self):
        cols = np.column_stack([self.pts[:, 0], self.pts[:, 1], self.pts[:, 0]])
        nrm = np.tile([0.0, 0.0, 1.0], (len(self.pts), 1))
        pc = _make_pc(self.pts, cols, nrm)
        out = pc.subsample_pointcloud_by_radius(np.array([0.33, 0.47, 0.0]), 0.2)
        np.testing.assert_allclose(out.colors[:, :2], out.points[:, :2])
        self.assertEqual(out.normals.shape, out.points.shape)

    def test_empty_neighbourhood_does_not_raise(self):
        pc = _make_pc(self.pts)
        out = pc.subsample_pointcloud_by_radius(np.array([50.0, 50.0, 0.0]), 0.1)
        self.assertEqual(len(out.points), 0)


# ---------------------------------------------------------------------------
# P9: normals under transforms
# ---------------------------------------------------------------------------


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class TestNormalsUnderTransforms(unittest.TestCase):
    def setUp(self):
        # plane z = x: normal (-1, 0, 1)/sqrt(2)
        g = np.random.default_rng(0).uniform(0, 1, (20, 2))
        self.pts = np.column_stack([g[:, 0], g[:, 1], g[:, 0]])
        self.n = np.tile(np.array([-1.0, 0.0, 1.0]) / np.sqrt(2), (20, 1))

    def _expected(self, A):
        n = self.n @ np.linalg.inv(A)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def _check(self, T):
        A = T[:3, :3]
        pc = _make_pc(self.pts, normals=self.n)
        pc.apply_transform(T)
        np.testing.assert_allclose(pc.normals, self._expected(A), atol=1e-9)
        sp = pcmod.SimplePointCloud(self.pts, normals=self.n)
        sp.transform(T)
        np.testing.assert_allclose(sp.normals, self._expected(A), atol=1e-9)
        # still perpendicular to the transformed surface
        d = pc.points[1] - pc.points[0]
        self.assertAlmostEqual(float(pc.normals[0] @ d), 0.0, places=9)

    def test_uniform_scale_and_rotation(self):
        T = np.eye(4)
        T[:3, :3] = 2.5 * _rot_z(30)
        T[:3, 3] = [1, 2, 3]
        self._check(T)

    def test_non_uniform_scale(self):
        T = np.diag([2.0, 1.0, 1.0, 1.0])
        self._check(T)


# ---------------------------------------------------------------------------
# P10: neighbourhood extractor frames
# ---------------------------------------------------------------------------


class TestNeighbourhoodFrames(_TmpDir):
    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(2)
        self.xyz = rng.uniform(-1, 1, (3000, 3))
        self.p = self.path("nb.ply")
        _write_ply(self.p, self.xyz)
        self.xyz = self.xyz.astype(np.float32).astype(float)
        # world = 2 x (rotation about x by 90 deg, so raw y is world z) + shift
        R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.T = np.eye(4)
        self.T[:3, :3] = 2.0 * R
        self.T[:3, 3] = [5, 6, 7]
        self.c = np.array([0.1, -0.2, 0.3])

    def test_cylinder_radius_is_raw_units_about_world_vertical(self):
        r = 0.3
        (nb,) = pcmod.stream_extract_neighborhoods_ply(
            self.p, [self.c], r, progress=False,
            world_transform=self.T, cylinder=True)
        w = (np.c_[self.xyz, np.ones(len(self.xyz))] @ self.T.T)[:, :3]
        wc = self.T[:3, :3] @ self.c + self.T[:3, 3]
        # world XY distance <= r x scale (2) == raw distance across raw y <= r
        expect = np.hypot(*(w[:, :2] - wc[:2]).T) <= 2 * r
        self.assertGreater(expect.sum(), 10)
        self.assertEqual(len(nb.points), int(expect.sum()))
        np.testing.assert_allclose(
            np.sort(nb.points[:, 0]), np.sort(self.xyz[expect, 0]))

    def test_sphere_same_with_or_without_transform(self):
        a = pcmod.stream_extract_neighborhoods_ply(
            self.p, [self.c], 0.4, progress=False)[0]
        b = pcmod.stream_extract_neighborhoods_ply(
            self.p, [self.c], 0.4, progress=False, world_transform=self.T)[0]
        self.assertGreater(len(a.points), 10)
        np.testing.assert_array_equal(a.points, b.points)
        d = np.linalg.norm(self.xyz - self.c, axis=1)
        self.assertEqual(len(a.points), int((d <= 0.4).sum()))


# ---------------------------------------------------------------------------
# P12 / P13: intercepts
# ---------------------------------------------------------------------------


class _Intercept:
    def __init__(self, coords, radius, **kw):
        self.coords = np.asarray(coords, float)
        self.radius = radius
        self.kw = kw


class _Timeout:
    def __enter__(self):
        def _raise(*_):
            raise TimeoutError("infinite loop")
        self.old = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(5)

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)


class TestIntercepts(unittest.TestCase):
    def setUp(self):
        self._ann = pcmod.annotations
        pcmod.annotations = types.SimpleNamespace(InterceptAnnotation=_Intercept)

    def tearDown(self):
        pcmod.annotations = self._ann

    def test_z_intercept_always_return_empty_cloud(self):
        pc = _make_pc(np.zeros((0, 3)))
        with _Timeout():
            self.assertIsNone(pc.get_z_intercept(np.array([0.0, 0.0]), 0.1,
                                                 always_return=True))

    def test_point_within_radius_between_samples_is_found(self):
        r = 0.1
        # 0.95 r from the ray, half-way between the old sample points (step r)
        p = np.array([0.95 * r, 0.0, -2.5 * r])
        pc = _make_pc(np.array([p, [5.0, 5.0, 5.0]]))
        hit = pc.get_intercept(np.zeros(3), r, vector=np.array([0.0, 0.0, -1.0]),
                               max_dist_from_origin=1.0)
        self.assertIsNotNone(hit)
        np.testing.assert_allclose(hit.coords, p)

    def test_point_beyond_radius_is_not_found(self):
        r = 0.1
        pc = _make_pc(np.array([[1.05 * r, 0.0, -2.0 * r]]))
        hit = pc.get_intercept(np.zeros(3), r, vector=np.array([0.0, 0.0, -1.0]),
                               max_dist_from_origin=1.0)
        self.assertIsNone(hit)

    def test_hit_is_on_first_surface_near_the_ray(self):
        # dense plane at z = 0, ray from above at an angle: the hit is at most
        # r behind the first contact and the point nearest the ray there
        r = 0.05
        g = np.mgrid[0:101, 0:101].reshape(2, -1).T * 0.01
        pts = np.column_stack([g, np.zeros(len(g))])
        pc = _make_pc(pts)
        for o, direction in (([0.2, 0.3, 1.0], [0.3, 0.1, -1.0]),
                             ([0.2, 0.3, 1.0], [0.6, 0.4, -1.0]),
                             ([0.1, 0.1, 0.3], [1.0, 0.5, -0.4]),  # grazing
                             ([0.53, 0.47, 1.0], [0.0, 0.0, -1.0])):
            o = np.asarray(o)
            u = np.asarray(direction) / np.linalg.norm(direction)
            hit = pc.get_intercept(o, r, vector=u, max_dist_from_origin=5.0)
            q = pts - o
            t = q @ u
            perp = np.linalg.norm(q - np.outer(t, u), axis=1)
            cand = (perp <= r) & (t >= 0)
            t_first = t[cand].min()
            t_hit = float((hit.coords - o) @ u)
            self.assertGreaterEqual(t_hit, t_first - 1e-12)
            self.assertLessEqual(t_hit, t_first + r + 1e-12)
            window = cand & (t <= t_first + r)
            perp_hit = np.linalg.norm((hit.coords - o) - t_hit * u)
            self.assertAlmostEqual(perp_hit, perp[window].min(), places=12)
        # straight down onto a grid node: the node itself
        hit = pc.get_intercept(np.array([0.5, 0.5, 1.0]), r,
                               vector=np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(hit.coords, [0.5, 0.5, 0.0])


# ---------------------------------------------------------------------------
# P16, S2, S4
# ---------------------------------------------------------------------------


class TestMemoryAndOccluder(unittest.TestCase):
    def test_single_available_memory_function(self):
        orig = st.available_memory
        try:
            st.available_memory = lambda: 123 * 1024**2
            pcmod.available_memory = st.available_memory
            self.assertEqual(pcmod._available_memory_bytes(), 123 * 1024**2)
            st.available_memory = lambda: 0  # unknown -> conservative fallback
            pcmod.available_memory = st.available_memory
            self.assertEqual(pcmod._available_memory_bytes(),
                             pcmod._FALLBACK_AVAILABLE_BYTES)
        finally:
            st.available_memory = orig
            pcmod.available_memory = orig

    def test_line_of_sight_margin_floor_in_metres(self):
        oc = st.Occluder.__new__(st.Occluder)
        oc.spacing = np.full(4, 0.0001)
        self.assertAlmostEqual(oc.line_of_sight_margin(), 0.01)
        # 0.01 m at 2 m per model unit is 0.005 model units
        self.assertAlmostEqual(oc.line_of_sight_margin(scale_factor=2.0), 0.005)
        oc.spacing = np.full(4, 0.01)
        self.assertAlmostEqual(oc.line_of_sight_margin(scale_factor=2.0), 0.04)


class TestPrefixForBudget(_TmpDir):
    def test_round_end_prefix(self):
        rng = np.random.default_rng(4)
        n = 20_000
        xyz = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 10, n),
                               rng.normal(0, 0.1, n)])
        p = self.path("s.ply")
        _write_ply(p, xyz)
        out = st.stratify_ply(p, self.path("strat.ply"), cluster_size=2000,
                              progress=False)
        lp = st.StratifiedPly(out)
        ends = lp.index.round_ends
        self.assertGreater(len(ends), 2)
        self.assertEqual(lp.prefix_for_budget(int(ends[1]) + 5), int(ends[1]))
        self.assertEqual(lp.prefix_for_budget(int(ends[1])), int(ends[1]))
        self.assertEqual(lp.prefix_for_budget(1), int(ends[0]))
        self.assertEqual(lp.prefix_for_budget(10 * n), n)
        with self.assertLogs(level="DEBUG") as cm:
            lp.read_prefix(int(ends[1]) + 5)
        self.assertTrue(any("round" in m for m in cm.output))


# ---------------------------------------------------------------------------
# ortho O1, O2, O4, O5, O6
# ---------------------------------------------------------------------------


class _PC:
    def __init__(self, points, colors=None):
        self.points = np.asarray(points, dtype=float)
        self.colors = (np.ones((len(self.points), 3)) if colors is None
                       else np.asarray(colors, dtype=float))


class TestOrtho(unittest.TestCase):
    def test_narrow_cloud_covers_full_extent(self):
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 1, 20000), rng.uniform(0, 10, 20000),
                               np.zeros(20000)])
        ex, ey = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        om = ortho.OrthoMap(_PC(pts, np.zeros((20000, 3))))
        self.assertGreaterEqual(om.width * om.resolution, ex - 1e-9)
        self.assertGreaterEqual(om.height * om.resolution, ey - 1e-9)
        filled_rows = (om.image.min(axis=(1, 2)) < 255).mean()
        self.assertGreater(filled_rows, 0.95)
        # requested size is a bound in both dimensions
        om = ortho.OrthoMap(_PC(pts), pixel_width=100, pixel_height=100)
        self.assertLessEqual(om.width, 100)
        self.assertLessEqual(om.height, 100)
        self.assertGreaterEqual(om.height * om.resolution, ey - 1e-9)

    def test_project_matches_rasteriser(self):
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(0, 3, 400), rng.uniform(0, 2, 400),
                               np.zeros(400)])
        om = ortho.OrthoMap(_PC(pts), pixel_width=30)
        px = om.project(pts)
        ix = np.clip(np.floor(px[:, 0]).astype(int), 0, om.width - 1)
        iy = np.clip(np.floor(px[:, 1]).astype(int), 0, om.height - 1)
        ix_r = np.clip(((pts[:, 0] - om.origin[0]) / om.resolution).astype(int),
                       0, om.width - 1)
        iy_r = np.clip(om.height - 1 - ((pts[:, 1] - om.origin[1])
                                        / om.resolution).astype(int),
                       0, om.height - 1)
        np.testing.assert_array_equal(ix, ix_r)
        np.testing.assert_array_equal(iy, iy_r)
        # the min corner maps to the bottom-left image corner
        np.testing.assert_allclose(om.project([om.origin[0], om.origin[1], 0]),
                                   [0.0, om.height])

    def test_value_at_tilted_up_vector(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(0, 4, (3000, 3))
        up = np.array([0.0, 1.0, 1.0])
        og = ortho.OrthoGrid(pcd=_PC(pts), value_by="count", cell_size=1.0,
                             up_vector=up)
        p = pts[17]
        proj = og.rotation @ p
        a = og.value_at(proj[0], proj[1])
        b = og.value_at(p[0], p[1], p[2])
        self.assertIsNotNone(a)
        self.assertEqual((a["ix"], a["iy"]), (b["ix"], b["iy"]))
        self.assertGreater(a["n_points"], 0)

    def test_display_rotation_keyword(self):
        pts = np.column_stack([np.linspace(0, 4, 50), np.linspace(0, 2, 50),
                               np.zeros(50)])
        a = ortho.OrthoMap(_PC(pts), pixel_width=40, display_rotation=90).show()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b = ortho.OrthoMap(_PC(pts), pixel_width=40, rotation=90).show()
        self.assertEqual(a.size, b.size)
        self.assertGreater(a.size[1], a.size[0])

    def test_units_label(self):
        pts = np.random.default_rng(3).uniform(0, 2, (500, 3))
        pc = _PC(pts)
        self.assertIn("units/px", repr(ortho.OrthoMap(pc)))
        og = ortho.OrthoGrid(pcd=pc, value_by="z", cell_size=0.5)
        self.assertEqual(og._value_label(), "z (units)")
        pc.scale_factor = 1.7
        self.assertIn("m/px", repr(ortho.OrthoMap(pc)))
        og = ortho.OrthoGrid(pcd=pc, value_by="density", cell_size=0.5)
        self.assertEqual(og._value_label(), "points / m²")
        om = ortho.OrthoMap(_PC(pts), units="m")
        self.assertIn("m/px", repr(om))


if __name__ == "__main__":
    unittest.main()
