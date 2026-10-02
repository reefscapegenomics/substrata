"""Tests for point-cloud segmentation helpers in :mod:`substrata.segmentation`.

The pure-numpy segmentation surface (``Segmentation``, ``sample_query_points``,
label propagation, recolouring, npz round-trip) is loaded directly from
``src/substrata`` via ``importlib`` against a bare ``substrata`` package shell,
so the heavy SAM2/OpenCV/torch code in the same module never runs. ``cv2`` and
``joblib`` are only referenced by the SAM2/SIFT functions (not by the helpers
tested here); they are stubbed when absent so the module file imports. Only
numpy, scipy and matplotlib (Agg) are actually exercised.
"""

# Standard Library
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Third-Party
import matplotlib
import numpy as np

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_segmentation():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    # Stub heavy deps only used by the SAM2/SIFT code (not by the helpers here),
    # so the module file imports in a bare environment.
    if "cv2" not in sys.modules:
        try:
            import cv2  # noqa: F401
        except Exception:  # noqa: BLE001
            sys.modules["cv2"] = types.ModuleType("cv2")
    if "joblib" not in sys.modules:
        try:
            import joblib  # noqa: F401
        except Exception:  # noqa: BLE001
            jl = types.ModuleType("joblib")
            jl.Parallel = lambda *a, **k: (lambda it: list(it))
            jl.delayed = lambda f: f
            sys.modules["joblib"] = jl

    if "substrata.segmentation" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "substrata.segmentation", _SUB / "segmentation.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["substrata.segmentation"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["substrata.segmentation"]


seg = _load_segmentation()


class _FakePcd:
    """Minimal stand-in exposing ``points`` (xy_grid sampling only)."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)


def _make_seg(labels=("A", "B"), coords=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
              colors=None):
    ql = np.array(list(labels), dtype=object)
    qc = np.asarray(coords, dtype=np.float32)
    conf = np.ones(len(ql))
    return seg.Segmentation.from_query_labels(qc, ql, conf, label_colors=colors)


class TestSampleQueryPoints(unittest.TestCase):
    def test_xy_grid_highest_one_per_cell(self):
        # Two XY cells (x≈0 and x≈1 at 0.5 m spacing), each with two z values.
        pts = np.array([
            [0.0, 0.0, 0.0], [0.1, 0.1, 5.0],   # cell (0,0): highest z=5
            [1.0, 0.0, 2.0], [1.1, 0.1, 1.0],   # cell (2,0): highest z=2
        ])
        out = seg.sample_query_points(_FakePcd(pts), 0.5, method="xy_grid")
        self.assertEqual(len(out), 2)
        zs = sorted(out[:, 2].tolist())
        self.assertEqual(zs, [2.0, 5.0])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            seg.sample_query_points(_FakePcd([[0, 0, 0]]), 0.5, method="bogus")


class TestPropagate(unittest.TestCase):
    def test_nearest_label(self):
        s = _make_seg()
        pts = np.array([[1.0, 0, 0], [9.0, 0, 0], [100.0, 0, 0]])
        codes = s.propagate(pts)
        # codebook == ["A", "B"] (sorted); nearest -> A, B, B
        self.assertEqual(codes.tolist(), [0, 1, 1])

    def test_max_radius_leaves_far_points_unlabeled(self):
        s = _make_seg()
        pts = np.array([[1.0, 0, 0], [100.0, 0, 0]])
        codes = s.propagate(pts, max_radius=5.0)
        self.assertEqual(codes.tolist(), [0, -1])


class TestRecolor(unittest.TestCase):
    def test_category_tint_and_unlabeled(self):
        s = _make_seg(colors={"A": (255, 0, 0), "B": (0, 255, 0)})
        orig = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        codes = np.array([0, 1, -1])
        out = s.recolor(orig, codes, unlabeled="gray")
        self.assertEqual(out.shape, (3, 3))
        # A -> reddish, B -> greenish
        self.assertGreater(out[0, 0], out[0, 1])
        self.assertGreater(out[1, 1], out[1, 0])
        # unlabeled -> gray (all channels equal)
        self.assertAlmostEqual(out[2, 0], out[2, 1])
        self.assertAlmostEqual(out[2, 1], out[2, 2])

    def test_unlabeled_keep_uses_original(self):
        s = _make_seg(colors={"A": (255, 0, 0), "B": (0, 255, 0)})
        orig = np.array([[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]])
        codes = np.array([-1, -1])
        out = s.recolor(orig, codes, unlabeled="keep")
        np.testing.assert_allclose(out, orig)

    def test_accepts_0_255_input(self):
        s = _make_seg(colors={"A": (255, 0, 0), "B": (0, 255, 0)})
        out = s.recolor(np.array([[255.0, 255.0, 255.0]]), np.array([0]))
        self.assertLessEqual(out.max(), 1.0)


class TestColorsAndSummary(unittest.TestCase):
    def test_manual_colors_override_auto(self):
        s = _make_seg(colors={"A": (255, 0, 0)})
        self.assertEqual(s.label_colors["A"], (255, 0, 0))
        # B not overridden -> still present (auto tab20)
        self.assertIn("B", s.label_colors)
        self.assertEqual(len(s.label_colors["B"]), 3)

    def test_from_query_labels_drops_unclassified(self):
        qc = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        labels = np.array(["A", "", "B"], dtype=object)
        conf = np.array([0.9, np.nan, 0.8])
        s = seg.Segmentation.from_query_labels(qc, labels, conf)
        self.assertEqual(s.n_queries, 2)
        self.assertEqual(s.codebook, ["A", "B"])

    def test_from_query_labels_all_empty_raises(self):
        qc = np.zeros((2, 3), dtype=np.float32)
        labels = np.array(["", ""], dtype=object)
        with self.assertRaises(ValueError):
            seg.Segmentation.from_query_labels(qc, labels, np.zeros(2))

    def test_summary_counts(self):
        qc = np.zeros((3, 3), dtype=np.float32)
        labels = np.array(["A", "A", "B"], dtype=object)
        s = seg.Segmentation.from_query_labels(qc, labels, np.ones(3))
        self.assertEqual(s.summary(), {"A": 2, "B": 1})


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_npz_roundtrip(self):
        s = _make_seg(colors={"A": (255, 0, 0), "B": (0, 255, 0)})
        path = os.path.join(self.tmp.name, "x_seg.npz")
        s.save(path)
        self.assertTrue(os.path.isfile(path))
        s2 = seg.Segmentation.load(path)
        self.assertEqual(s2.codebook, s.codebook)
        self.assertEqual(s2.label_colors, s.label_colors)
        np.testing.assert_array_equal(s2.query_coords, s.query_coords)
        # propagation identical after reload
        pts = np.array([[1.0, 0, 0], [9.0, 0, 0]])
        np.testing.assert_array_equal(s.propagate(pts), s2.propagate(pts))


class _FakeDL:
    def __init__(self, crops):
        self.n = len(crops)


class _FakeDLS:
    vocab = ["A", "B"]

    def test_dl(self, crops, bs=64):
        return _FakeDL(crops)


class _FakeLearn:
    """Counts ``get_preds`` calls so we can assert crops are flushed in batches."""

    def __init__(self):
        self.dls = _FakeDLS()
        self.calls = 0

    def get_preds(self, dl=None):
        self.calls += 1
        probs = np.tile([0.7, 0.3], (dl.n, 1))  # always predicts class "A"
        return probs, None


class _FakeCam:
    def __init__(self, filepath):
        self.filepath = filepath


class TestClassifyCropsFlush(unittest.TestCase):
    """`_classify_crops` must classify in bounded batches (memory-safe)."""

    def setUp(self):
        from PIL import Image

        self.tmp = tempfile.TemporaryDirectory()
        self.fp = os.path.join(self.tmp.name, "photo.png")
        Image.fromarray(
            np.zeros((300, 300, 3), dtype=np.uint8)
        ).save(self.fp)

    def tearDown(self):
        self.tmp.cleanup()

    def test_flush_runs_multiple_batches(self):
        n = 10
        qc = np.zeros((n, 3), dtype=float)
        best_cam = np.zeros(n, dtype=int)          # all -> cam 0
        best_x = np.full(n, 150.0)                  # centre of the 300px photo
        best_y = np.full(n, 150.0)
        learn = _FakeLearn()
        labels, conf = seg._classify_crops(
            qc, best_cam, best_x, best_y, [_FakeCam(self.fp)], learn,
            crop_size=100, batch_size=2, flush_batch=4, verbose=False,
        )
        # 10 crops with flush_batch=4 -> flush at 4, 8, then remainder 2 == 3 passes
        self.assertEqual(learn.calls, 3)
        self.assertTrue(all(l == "A" for l in labels))
        self.assertEqual(len(labels), n)

    def test_single_pass_when_under_flush_batch(self):
        n = 3
        qc = np.zeros((n, 3), dtype=float)
        best_cam = np.zeros(n, dtype=int)
        best_x = np.full(n, 150.0)
        best_y = np.full(n, 150.0)
        learn = _FakeLearn()
        seg._classify_crops(
            qc, best_cam, best_x, best_y, [_FakeCam(self.fp)], learn,
            crop_size=100, batch_size=64, flush_batch=4096, verbose=False,
        )
        self.assertEqual(learn.calls, 1)


if __name__ == "__main__":
    unittest.main()
