"""Tests for the point-feature classifier (substrata.pointfeatures).

The modules are loaded directly (no ``substrata`` package import), with a tiny
fake backbone instead of DINOv2, so nothing is downloaded and no GPU is needed.
Tests that need torch are skipped without it.
"""

# Standard Library
import csv
import importlib.util
import logging
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

# Third-Party Libraries
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for this suite
    torch = None


def _load(name):
    """Load ``substrata.<name>`` without importing the ``substrata`` package."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
    for dep in ("settings",):
        key = f"substrata.{dep}"
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, _SUB / f"{dep}.py")
            m = importlib.util.module_from_spec(spec)
            sys.modules[key] = m
            spec.loader.exec_module(m)
        setattr(sys.modules["substrata"], dep, sys.modules[key])
    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")
        log_mod.logger = logging.getLogger("substrata-test")

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        log_mod.tqdm_joblib = tqdm_joblib
        sys.modules["substrata.logging"] = log_mod
    key = f"substrata.{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _SUB / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    setattr(sys.modules["substrata"], name, sys.modules[key])
    return sys.modules[key]


pf = _load("pointfeatures")
cl = _load("classification")


if torch is not None:

    class FakeBackbone(torch.nn.Module):
        """DINOv2-like: one token per 14 px patch = the patch's mean colour."""

        patch_size = 14
        embed_dim = 3

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward_features(self, x):
            self.calls += 1
            p = torch.nn.functional.avg_pool2d(x, self.patch_size)
            b, c = p.shape[:2]
            tokens = p.flatten(2).transpose(1, 2).reshape(b, -1, c)
            return {"x_norm_patchtokens": tokens}


def _make_classifier(**kw):
    args = dict(
        backbone="dinov2_vitb14_reg", classes=["A", "B", "C"],
        weights=np.eye(3, dtype=np.float32), bias=np.zeros(3, np.float32),
        source_mean=np.zeros(3, np.float32), source_std=np.ones(3, np.float32),
    )
    args.update(kw)
    return pf.PointFeatureClassifier(**args)


class TestWindows(unittest.TestCase):
    def test_window_box_shifts_inside_image(self):
        self.assertEqual(pf.window_box(50, 50, 20, 100, 100), (40, 40, 60, 60))
        # Near an edge the box shifts inwards (point off-centre).
        self.assertEqual(pf.window_box(5, 98, 20, 100, 100), (0, 80, 20, 100))
        # Larger than the image: anchored at 0, extends beyond.
        self.assertEqual(pf.window_box(10, 10, 200, 100, 100), (0, 0, 200, 200))

    def test_read_window_pads_outside_black(self):
        im = Image.new("RGB", (50, 40), (255, 0, 0))
        out = np.asarray(pf.read_window(im, (-10, -10, 30, 30), 40))
        self.assertTrue((out[0, 0] == 0).all())
        self.assertEqual(tuple(out[30, 30]), (255, 0, 0))

    def _gradient(self, path, w=400, h=300):
        x = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
        arr = np.stack([x, np.zeros_like(x), np.zeros_like(x)], -1).astype(np.uint8)
        Image.fromarray(arr).save(path)
        return x

    def test_image_windows_point_position(self):
        with tempfile.TemporaryDirectory() as d:
            for ext in ("png", "jpg"):
                path = os.path.join(d, f"g.{ext}")
                ramp = self._gradient(path)
                pts = np.array([[200.0, 150.0], [5.0, 5.0]])
                wins, xy = pf.image_windows(path, pts, np.array([0.25, 1.0]), 64)
                self.assertEqual(wins.shape, (2, 64, 64, 3))
                np.testing.assert_allclose(xy[0], [32, 32], atol=0.6)
                np.testing.assert_allclose(xy[1], [5, 5], atol=0.6)
                # The window pixel at the point shows the photo at the point.
                r = wins[0][int(xy[0][1]), int(xy[0][0]), 0]
                self.assertAlmostEqual(float(r), float(ramp[150, 200]), delta=6)

    def test_image_windows_unreadable(self):
        self.assertIsNone(pf.image_windows("/no/such.png", np.zeros((1, 2)),
                                           np.ones(1), 28))


class TestGroundResolution(unittest.TestCase):
    def test_world_scale_and_gsd(self):
        w = np.diag([0.2, 0.2, 0.2, 1.0])
        self.assertAlmostEqual(pf.world_scale(w), 0.2)
        self.assertIsNone(pf.world_scale(None))
        # 6.25 units x 0.2 = 1.25 m at f = 3125 px -> 0.4 mm/px
        self.assertAlmostEqual(pf.ground_resolution_mm(6.25, 3125, 0.2), 0.4)
        self.assertIsNone(pf.ground_resolution_mm(None, 3125, 0.2))
        self.assertIsNone(pf.ground_resolution_mm(1.0, 0, 0.2))

    def test_image_match_gsd(self):
        parent = types.SimpleNamespace(world_transform=np.diag([0.5, 0.5, 0.5, 1]))

        class Cam:
            sensor = types.SimpleNamespace(fx=1000.0)

            def __init__(self):
                self.parent = parent

            def project_points(self, coords, use_orig_coords=False):
                return np.array([0.0]), np.array([0.0]), np.array([4.0]), None, None

        m = types.SimpleNamespace(cam=Cam(), depth=2.0)
        self.assertAlmostEqual(pf.image_match_gsd_mm(m), 1.0)  # 2*0.5/1000 m
        ann = types.SimpleNamespace(orig_coords=[1, 2, 3])
        m2 = types.SimpleNamespace(cam=Cam(), depth=None, annotation=ann)
        self.assertAlmostEqual(pf.image_match_gsd_mm(m2), 2.0)
        self.assertIsNone(pf.image_match_gsd_mm(types.SimpleNamespace(cam=None)))

    def test_project_yaml_lookup_ignores_sidecars(self):
        with tempfile.TemporaryDirectory() as d:
            proj = os.path.join(d, "plot_x")
            photos = os.path.join(proj, "plot_x.photos")
            os.makedirs(photos)
            open(os.path.join(proj, "a.csv.yaml"), "w").close()
            self.assertIsNone(pf._find_project_yaml(photos, levels=1))
            open(os.path.join(proj, "plot_x.yaml"), "w").close()
            self.assertEqual(pf._find_project_yaml(photos),
                             os.path.join(proj, "plot_x.yaml"))


class TestClassifier(unittest.TestCase):
    def test_scales_for(self):
        clf = _make_classifier(target_mm_px=1.2, scale_factor=0.5)
        np.testing.assert_allclose(clf.scales_for([1.2, 2.4, None], 3), [1, 2, 0.5])
        np.testing.assert_allclose(clf.scales_for(None, 2), [0.5, 0.5])
        fixed = _make_classifier(scale_factor=0.25)
        np.testing.assert_allclose(fixed.scales_for([1.2], 1), [0.25])

    def test_predict_and_calibration(self):
        clf = _make_classifier()
        p = clf.predict_proba_features(np.array([[5.0, 0, 0], [0, 0, 5.0]]))
        self.assertEqual(p.argmax(1).tolist(), [0, 2])
        np.testing.assert_allclose(p.sum(1), 1, rtol=1e-6)
        # A target calibration shifts the standardisation.
        cal = clf.with_calibration(np.array([5.0, 0, -5.0]), np.ones(3), {"n": 1})
        self.assertTrue(cal.calibrated and not clf.calibrated)
        # Standardised with the target mean, [5, 0, 0] becomes [0, 0, 5].
        self.assertEqual(cal.predict_proba_features(np.array([[5.0, 0, 0]])).argmax(),
                         2)

    def test_results_format(self):
        clf = _make_classifier()
        res = clf.results(np.array([[0.1, 0.7, 0.2], [np.nan] * 3]),
                          np.array([True, False]))
        self.assertEqual(res[0]["label"], "B")
        self.assertAlmostEqual(res[0]["confidence"], 0.7)
        self.assertEqual(set(res[0]["probs"]), {"A", "B", "C"})
        self.assertIsNone(res[1])

    @unittest.skipUnless(torch, "torch not installed")
    def test_save_load_roundtrip_and_dispatch(self):
        clf = _make_classifier(target_mm_px=1.2, scale_factor=0.9,
                               label_map={"x": "A"}, meta={"probe_C": 0.01})
        clf = clf.with_calibration(np.ones(3), 2 * np.ones(3), {"n": 4})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            clf.save(path)
            back = pf.PointFeatureClassifier.load(path)
            self.assertEqual(back.classes, clf.classes)
            self.assertEqual(back.target_mm_px, 1.2)
            self.assertEqual(back.label_map, {"x": "A"})
            self.assertTrue(back.calibrated)
            x = np.random.default_rng(0).normal(size=(5, 3))
            np.testing.assert_allclose(back.predict_proba_features(x),
                                       clf.predict_proba_features(x), rtol=1e-5)
            # classification.get_image_classifier loads .pt as a point classifier
            loaded = cl.get_image_classifier(path, device="cpu")
            self.assertTrue(pf.is_point_classifier(loaded))
            self.assertIsNone(cl.classifier_crop_size(loaded))
            # Other dicts are refused.
            torch.save({"x": 1}, os.path.join(d, "other.pt"))
            with self.assertRaises(ValueError):
                pf.PointFeatureClassifier.load(os.path.join(d, "other.pt"))

    def test_arch_and_path_detection(self):
        self.assertTrue(pf.is_point_arch("dinov2_vitb14_reg"))
        self.assertFalse(pf.is_point_arch("resnet34"))
        self.assertTrue(pf.is_point_classifier_path("a/b.pt"))
        self.assertFalse(pf.is_point_classifier_path("a/b.pkl"))


class TestProbe(unittest.TestCase):
    def _blobs(self, n_classes, n=60, dim=5, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.normal(scale=5, size=(n_classes, dim))
        x = np.vstack([c + rng.normal(size=(n, dim)) for c in centres])
        y = np.repeat([f"c{i}" for i in range(n_classes)], n)
        return x, y

    def test_multiclass_probe_separates(self):
        x, y = self._blobs(3)
        x = x * 10 + 3  # folded standardisation must cope with any scale
        classes, w, b, c = pf.fit_point_probe(x, y)
        self.assertEqual(classes, ["c0", "c1", "c2"])
        self.assertEqual(w.shape, (3, 5))
        pred = np.array(classes)[(x @ w.T + b).argmax(1)]
        self.assertGreater((pred == y).mean(), 0.97)
        self.assertIn(c, [float(v) for v in pf.settings.POINT_PROBE_CS])

    def test_binary_probe_has_two_rows(self):
        x, y = self._blobs(2)
        classes, w, b, _c = pf.fit_point_probe(x, y)
        self.assertEqual(w.shape, (2, 5))
        pred = np.array(classes)[(x @ w.T + b).argmax(1)]
        self.assertGreater((pred == y).mean(), 0.97)


@unittest.skipUnless(torch, "torch not installed")
class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        # Left half red, right half blue.
        arr = np.zeros((280, 560, 3), np.uint8)
        arr[:, :280, 0] = 255
        arr[:, 280:, 2] = 255
        self.photo = os.path.join(self.d, "p.png")
        Image.fromarray(arr).save(self.photo)
        self.model = FakeBackbone()

    def tearDown(self):
        self.tmp.cleanup()

    def test_sample_feature_grid_reads_cell_centres(self):
        h, w = 4, 6
        grid = torch.arange(w, dtype=torch.float32).repeat(1, 1, h, 1)  # (1,1,h,w)
        size = 14 * w
        xy = np.array([[(3 + 0.5) * 14, 7.0]])
        self.assertAlmostEqual(float(pf.sample_feature_grid(grid, xy, size)[0, 0]), 3.0,
                               places=5)

    def test_extract_features_by_colour(self):
        feats, ok = pf.extract_point_features(
            [self.photo, self.photo, "/missing.png"], [100, 460, 5], [140, 140, 5],
            [0.5, 0.5, 0.5], self.model, window=56, batch_size=2, n_jobs=2,
            progress=False,
        )
        self.assertEqual(ok.tolist(), [True, True, False])
        # Red point: red channel high, blue low (ImageNet-normalised).
        self.assertGreater(feats[0, 0], feats[0, 2])
        self.assertGreater(feats[1, 2], feats[1, 0])
        self.assertTrue((feats[2] == 0).all())

    def test_cached_features_incremental(self):
        cache = os.path.join(self.d, "cache", "c.npz")
        loads = []

        def loader():
            loads.append(1)
            return self.model

        args = (["a", "b"], [self.photo] * 2, [100, 460], [140, 140])
        f1, ok1 = pf.cached_features(*args, [0.5, 0.5], cache, loader, 56)
        f2, ok2 = pf.cached_features(*args, [0.5, 0.5], cache, loader, 56)
        self.assertEqual(len(loads), 1)  # second call fully cached
        np.testing.assert_allclose(f1, f2, atol=1e-2)  # float16 cache
        self.assertTrue(ok2.all())
        pf.cached_features(*args, [0.25, 0.5], cache, loader, 56)
        self.assertEqual(len(loads), 2)  # changed scale -> re-extract
        z = np.load(cache)
        self.assertEqual(len(z["keys"]), 2)  # stale key dropped

    def test_classify_points_and_matches(self):
        clf = _make_classifier(classes=["red", "green", "blue"],
                               target_mm_px=1.0, scale_factor=0.5, window=56)
        clf.backbone_model = lambda: self.model
        probs, ok = clf.classify_points([self.photo] * 2, [100, 460], [140, 140],
                                        gsd_mm=[0.5, None], progress=False)
        self.assertTrue(ok.all())
        self.assertEqual([clf.classes[k] for k in probs.argmax(1)], ["red", "blue"])

        # Through classification's ImageMatch path, results are attached.
        m = types.SimpleNamespace(filepath=self.photo, x=460.0, y=140.0, cam=None)
        missing = types.SimpleNamespace(filepath="/missing.png", x=1.0, y=1.0)
        res = cl.classify_image_matches_batch([m, missing], clf, n_jobs=1)
        self.assertEqual(res[0]["label"], "blue")
        self.assertIs(m.classification, res[0])
        self.assertIsNone(res[1])
        self.assertEqual(cl.classify_image_match(m, clf)["label"], "blue")


class TestTrainingPlan(unittest.TestCase):
    def _photos(self, d, width, n=2):
        paths = []
        for i in range(n):
            p = os.path.join(d, f"w{width}_{i}.png")
            Image.new("RGB", (width, 100)).save(p)
            paths.append(p)
        return paths

    def _rows(self, paths, **extra):
        return [dict(id=str(i), cam_filepath=p, _x=10.0, _y=10.0, **extra)
                for i, p in enumerate(paths)]

    def test_auto_scale_from_image_width(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertAlmostEqual(pf.auto_scale_factor(self._photos(d, 9600)),
                                   448 / 1600)
            self.assertEqual(pf.auto_scale_factor(self._photos(d, 1200)), 1.0)

    def test_plan_fixed_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self._rows(self._photos(d, 9600))
            plan = pf.plan_training_scales(rows, d, feature_scale=0.3)
            self.assertIsNone(plan.target_mm_px)
            np.testing.assert_allclose(plan.scales, 0.3)
            # No project cameras -> automatic fixed scale
            plan = pf.plan_training_scales(rows, d)
            self.assertIsNone(plan.target_mm_px)
            self.assertAlmostEqual(plan.scale_factor, 448 / 1600)
            with self.assertRaises(ValueError):
                pf.plan_training_scales(rows, d, feature_mm_px=1.0)

    def test_plan_physical_from_gsd_column(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._photos(d, 9600)
            rows = [dict(r, cam_gsd_mm=g) for r, g in
                    zip(self._rows(paths), ["0.2", "0.4"])]
            plan = pf.plan_training_scales(rows, d, feature_mm_px=0.8)
            self.assertEqual(plan.target_mm_px, 0.8)
            np.testing.assert_allclose(plan.scales, [0.25, 0.5])
            auto = pf.plan_training_scales(rows, d)
            # Automatic target: median ground resolution at the automatic scale.
            self.assertEqual(auto.target_mm_px, float(f"{0.3 / (448 / 1600):.3g}"))

    def test_read_training_points_label_map_and_split(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "label", "cam_filepath", "cam_x", "cam_y",
                            "split_key"])
                w.writerow(["1", "CST", "/p/a.jpg", "10", "20", "imgA"])
                w.writerow(["2", "CST", "/p/a.jpg", "11", "21", "imgA"])
                w.writerow(["3", "EXCL", "/p/a.jpg", "1", "2", ""])
                w.writerow(["4", "SUSC", "/p/b.jpg", "", "2", ""])
            rows = pf.read_training_points(path, {"CST": "CORAL", "SUSC": "S"})
            self.assertEqual([r["id"] for r in rows], ["1", "2"])
            self.assertEqual(rows[0]["_cls"], "CORAL")
            self.assertEqual(rows[0]["_split"], rows[1]["_split"])  # same key
            self.assertEqual(rows[0]["_split"], cl.split_for_id("imgA", (80, 10, 10)))
            # Without a label map every labelled row with camera fields is kept.
            every = pf.read_training_points(path, None)
            self.assertEqual([r["id"] for r in every], ["1", "2", "3"])
            self.assertIsNone(every[2]["_cls"])


if __name__ == "__main__":
    unittest.main()
