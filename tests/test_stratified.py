"""Tests for stratified PLYs (``substrata.stratified``).

The module itself only needs numpy and tqdm, but importing it goes through the
``substrata`` package (open3d, cv2, …), so the tests are skipped without the
conda environment.
"""

# Standard Library
import importlib
import os
import shutil
import tempfile
import unittest

# Third-party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    st = importlib.import_module("substrata.stratified")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    st = None

N = 300_000


def _write_ply(path, n=N, seed=1):
    """A wavy 20 x 10 m surface with colours and normals (float32 / uchar)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 20, n)
    y = rng.uniform(0, 10, n)
    z = 0.5 * np.sin(x) * np.cos(y) + rng.normal(0, 0.01, n)
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    rec = np.zeros(n, dtype)
    rec["x"], rec["y"], rec["z"] = x, y, z
    rec["nz"] = 1
    rec["red"] = (x * 12).astype(np.uint8)
    rec["green"] = (y * 25).astype(np.uint8)
    rec["blue"] = 128
    props = "".join(
        f"property {'uchar' if dtype[k].kind == 'u' else 'float'} {k}\n"
        for k in dtype.names
    )
    head = (
        "ply\nformat binary_little_endian 1.0\ncomment made by test\n"
        f"element vertex {n}\n{props}end_header\n"
    )
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(rec.tobytes())
    return rec


@unittest.skipUnless(st is not None, "requires the open3d/substrata environment")
class TestStratifiedPly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.src = os.path.join(cls.dir, "plot.ply")
        cls.orig = _write_ply(cls.src)
        with open(cls.src, "rb") as f:
            cls.orig_bytes = f.read()
        cls.dst_dir = os.path.join(cls.dir, "fast")
        os.makedirs(cls.dst_dir)
        cls.copy = st.stratify_ply(
            cls.src, os.path.join(cls.dst_dir, "plot.ply"), cluster_size=10_000,
            progress=False,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_copy_leaves_source_alone(self):
        self.assertFalse(st.is_stratified(self.src))
        with open(self.src, "rb") as f:
            self.assertEqual(f.read(), self.orig_bytes)
        self.assertTrue(st.is_stratified(self.copy))
        self.assertTrue(os.path.isfile(os.path.join(self.dst_dir, "plot.strat.npz")))

    def test_same_records_and_properties(self):
        lp = st.StratifiedPly(self.copy)
        self.assertEqual(lp.header.props, st.read_ply_header(self.src).props)
        self.assertEqual(lp.n, N)
        v = f"V{lp.header.record_size}"
        a = np.sort(np.asarray(lp.records).view(v))
        b = np.sort(self.orig.view(v))
        self.assertTrue((a == b).all())

    def test_open3d_reads(self):
        o3d = importlib.import_module("open3d")
        pc = o3d.io.read_point_cloud(self.copy)
        self.assertEqual(len(pc.points), N)
        self.assertTrue(pc.has_colors() and pc.has_normals())

    def test_prefix_is_uniform(self):
        lp = st.StratifiedPly(self.copy)
        k = int(lp.index.round_ends[lp.index.rounds - 3])  # ~1/4 of the cloud
        pre = st.ply_xyz(lp.read_prefix(k))
        full = st.ply_xyz(self.orig)

        def hist(p):
            h, *_ = np.histogram2d(p[:, 0], p[:, 1], bins=8, range=[[0, 20], [0, 10]])
            return h / h.sum()

        self.assertLess(np.abs(hist(pre) - hist(full)).max(), 0.004)

    def test_query_box_matches_bruteforce(self):
        lp = st.StratifiedPly(self.copy)
        lo, hi = np.array([3.0, 4.0, -1.0]), np.array([7.5, 6.0, 0.2])
        xyz = st.ply_xyz(self.orig)
        expect = self.orig[((xyz >= lo) & (xyz <= hi)).all(1)]
        got = lp.query_box(lo, hi)
        self.assertLess(len(lp.clusters_in_box(lo, hi)), lp.index.n_clusters)
        v = f"V{lp.header.record_size}"
        self.assertEqual(len(got), len(expect))
        self.assertTrue((np.sort(got.view(v)) == np.sort(expect.view(v))).all())

    def test_summary(self):
        text = st.StratifiedPly(self.copy).summary()
        self.assertIn(f"{N:,} x 27 B", text)
        self.assertIn("prefix ends:", text)

    def test_clusters_cover_everything(self):
        lp = st.StratifiedPly(self.copy)
        total = sum(len(lp.read_cluster(c)) for c in range(lp.index.n_clusters))
        self.assertEqual(total, lp.n)

    def test_in_place(self):
        path = os.path.join(self.dir, "inplace.ply")
        shutil.copy(self.src, path)
        self.assertEqual(st.stratify_ply(path, cluster_size=10_000, progress=False),
                         os.path.abspath(path))
        self.assertTrue(st.is_stratified(path))
        with open(path, "rb") as a, open(self.copy, "rb") as b:
            self.assertEqual(a.read(), b.read())  # same seed: same file
        self.assertEqual(
            sorted(os.listdir(self.dir)),
            sorted(["fast", "inplace.ply", "inplace.strat.npz", "plot.ply"]),
        )

    def test_keep_original(self):
        path = os.path.join(self.dir, "kept.ply")
        shutil.copy(self.src, path)
        seen = []
        st.stratify_ply(path, cluster_size=10_000, progress=False, keep_original=True,
                        on_progress=lambda f, stage: seen.append(f))
        backup = os.path.join(self.dir, "backup_kept.ply")
        self.assertEqual(st.original_backup_path(path), backup)
        with open(backup, "rb") as f:
            self.assertEqual(f.read(), self.orig_bytes)
        self.assertTrue(st.is_stratified(path))
        self.assertEqual(seen[-1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(seen, seen[1:])))
        for f in (path, backup, os.path.join(self.dir, "kept.strat.npz")):
            os.remove(f)

    def test_space(self):
        size = os.path.getsize(self.src)
        sp = st.stratify_space(self.src, mode="stream")  # + the temp file
        self.assertEqual((sp["mode"], sp["need"]), ("stream", 2 * size))
        sp = st.stratify_space(self.src, mode="memory")  # only the new PLY
        self.assertEqual((sp["mode"], sp["need"]), ("memory", size))
        self.assertGreater(sp["free"], 0)
        self.assertGreater(sp["mem_need"], N * (27 + 8))
        self.assertGreater(sp["mem_free"], 0)

    def test_modes(self):
        self.assertEqual(st.stratify_mode(self.src), "memory")  # small: fits in RAM
        self.assertEqual(st.stratify_mode(self.src, "stream"), "stream")
        with self.assertRaises(ValueError):
            st.stratify_mode(self.src, "fast")

    def test_modes_write_the_same_file(self):
        out = os.path.join(self.dir, "modes")
        os.makedirs(out)
        with open(self.copy, "rb") as f:
            expect = f.read()
        runs = {
            "memory": dict(mode="memory", chunk=7_000),
            "stream": dict(mode="stream"),
            "stream_small": dict(mode="stream", chunk=7_000, buffer_bytes=1 << 16),
        }
        for name, kw in runs.items():
            os.makedirs(os.path.join(out, name))
            path = st.stratify_ply(self.src, os.path.join(out, name, "plot.ply"),
                                   cluster_size=10_000, progress=False, **kw)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), expect, name)
            self.assertEqual(sorted(os.listdir(os.path.dirname(path))),
                             ["plot.ply", "plot.strat.npz"], name)
        shutil.rmtree(out)

    def test_work_dir(self):
        out = os.path.join(self.dir, "wd")
        work = os.path.join(self.dir, "wd_work")
        os.makedirs(out)
        os.makedirs(work)
        path = os.path.join(out, "plot.ply")
        for mode in ("memory", "stream"):
            shutil.copy(self.src, path)
            seen = []
            st.stratify_ply(path, mode=mode, work_dir=work, cluster_size=10_000,
                            progress=False, on_progress=lambda f, stage: seen.append(f))
            with open(path, "rb") as a, open(self.copy, "rb") as b:
                self.assertEqual(a.read(), b.read(), mode)
            self.assertEqual(os.listdir(work), [], mode)  # nothing left behind
            self.assertEqual(sorted(os.listdir(out)), ["plot.ply", "plot.strat.npz"])
            self.assertEqual(seen[-1], 1.0)
            self.assertTrue(all(a <= b for a, b in zip(seen, seen[1:])))
        shutil.rmtree(out)
        shutil.rmtree(work)

    def test_not_stratified_when_index_missing_or_stale(self):
        path = os.path.join(self.dir, "stale.ply")
        shutil.copy(self.copy, path)
        self.assertFalse(st.is_stratified(path))  # no index
        with self.assertRaises(ValueError):
            st.StratifiedPly(path)
        shutil.copy(os.path.join(self.dst_dir, "plot.strat.npz"),
                    os.path.join(self.dir, "stale.strat.npz"))
        self.assertTrue(st.is_stratified(path))
        with open(path, "ab") as f:
            f.write(b"\0" * 27)  # one extra record: index no longer matches
        self.assertFalse(st.is_stratified(path))
        with self.assertRaises(ValueError):
            st.StratifiedPly(path)
        self.assertFalse(st.is_stratified(None))


if __name__ == "__main__":
    unittest.main()
