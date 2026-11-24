"""Regression tests for the classification / segmentation / curation review fixes.

Each test names the review finding it covers (H = high, M = medium, L = low).
Modules are loaded from ``src/substrata`` without the package (see
test_pointfeatures); nothing is downloaded and no GPU is used. Tests that need
torch or scikit-learn are skipped without them.
"""

# Standard Library
import csv
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

# Third-Party Libraries
import numpy as np
from PIL import Image

import test_pointfeatures
from test_pointfeatures import _load, _make_classifier

pf = _load("pointfeatures")
cl = _load("classification")
cur = _load("curation")
settings = pf.settings

try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover
    sklearn = None
torch = test_pointfeatures.torch


def _photo(d, name, width, height=100, color=(120, 120, 120)):
    path = os.path.join(d, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)
    return path


def _write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


class TestSplitByPhoto(unittest.TestCase):
    """H1: points of one photo never straddle train / validation / test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        rows = []
        for plot in ("plotA", "plotB"):
            for i in range(30):  # 30 photos x 6 points per plot
                for k in range(6):
                    rows.append({
                        "id": str(len(rows)), "label": "CST",
                        "cam_filepath": f"/photos/{plot}/IMG_{i:03d}.jpg",
                        "cam_x": str(10 + k), "cam_y": "20",
                    })
        self.csvs = []
        for plot in ("plotA", "plotB"):
            path = os.path.join(self.d, f"{plot}_ann.csv")
            _write_csv(path, [r for r in rows if f"/{plot}/" in r["cam_filepath"]])
            self.csvs.append(path)

    def tearDown(self):
        self.tmp.cleanup()

    def _collate(self, split_by):
        out = os.path.join(self.d, f"train_{split_by}.csv")
        with mock.patch.object(cl, "resolve_cam_dirs", lambda dirs, *a, **k: {}):
            cl.collate_training_annotations(
                self.csvs, "*_ann.csv", {"CST"}, out, self.d, prompt=False,
                split_by=split_by)
        return pf.read_training_points(out, None)

    def _splits_per(self, rows, key):
        per = {}
        for r in rows:
            per.setdefault(key(r), set()).add(r["_split"])
        return per

    def test_no_photo_spans_two_splits(self):
        rows = self._collate("photo")
        per_photo = self._splits_per(rows, lambda r: r["cam_filepath"])
        self.assertEqual(len(per_photo), 60)
        self.assertTrue(all(len(s) == 1 for s in per_photo.values()))
        # the photos are spread over the splits (60 photos, 80/10/10)
        self.assertGreater(len({next(iter(s)) for s in per_photo.values()}), 1)
        # crops use the same split (plan_crops reads the same split_key)
        out = os.path.join(self.d, "train_photo.csv")
        plan = cl.plan_crops(out, os.path.join(self.d, "crops"))
        crop_split = {}
        for path, info in plan.items():
            crop_split.setdefault(info["cam_filepath"], set()).add(
                os.path.relpath(path, os.path.join(self.d, "crops")).split(os.sep)[0])
        self.assertTrue(all(len(s) == 1 for s in crop_split.values()))

    def test_point_split_reproduces_the_old_behaviour(self):
        rows = self._collate("point")
        self.assertTrue(all(r["split_key"] == "" for r in rows))
        self.assertTrue(all(r["_split"] == cl.split_for_id(r["id"], settings.TRAIN_SPLIT)
                            for r in rows))
        per_photo = self._splits_per(rows, lambda r: r["cam_filepath"])
        self.assertTrue(any(len(s) > 1 for s in per_photo.values()))

    def test_plot_split(self):
        rows = self._collate("plot")
        per_plot = self._splits_per(rows, lambda r: r["cam_filepath"].split("/")[2])
        self.assertTrue(all(len(s) == 1 for s in per_plot.values()))

    def test_row_split_key(self):
        row = {"cam_filepath": "/p/IMG_1.JPG", "_model_name": "plotA", "split_key": ""}
        self.assertEqual(cl.row_split_key(row, "photo"), "IMG_1")
        self.assertEqual(cl.row_split_key(row, "plot"), "plotA")
        self.assertEqual(cl.row_split_key(row, "point"), "")
        # a CoralNet / pool row's own key (its image) is kept
        self.assertEqual(cl.row_split_key(dict(row, split_key="img9"), "photo"), "img9")
        with self.assertRaises(ValueError):
            cl.collate_training_annotations([], "*", set(), os.devnull, self.d,
                                            split_by="pixel")


class TestCalibrationScale(unittest.TestCase):
    """H2: calibrate resamples a photo exactly as prediction does."""

    def test_calibrate_and_predict_agree_for_another_width(self):
        clf = _make_classifier(scale_factor=0.5, meta={"ref_width": 1000})
        seen = {}

        def fake_extract(paths, xs, ys, scales, model, window, **kw):
            seen.setdefault("calls", []).append(dict(zip(paths, map(float, scales))))
            n = len(paths)
            if kw.get("on_batch"):
                kw["on_batch"](np.arange(n), np.zeros((n, 3), np.float32))
            return np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32), \
                np.ones(n, bool)

        with tempfile.TemporaryDirectory() as d:
            other = _photo(d, "auv/F1.jpg", 2000)  # another camera: twice as wide
            same = _photo(d, "gopro/G1.jpg", 1000)
            clf.backbone_model = lambda: object()
            with mock.patch.object(pf, "extract_point_features", fake_extract):
                cal = pf.calibrate_point_classifier(clf, [other, same], n_images=2,
                                                    per_image=3)
                clf.classify_points([other, same], [10, 10], [10, 10])
        cal_scales, pred_scales = seen["calls"]
        self.assertAlmostEqual(cal_scales[other], 0.25)  # 0.5 x 1000 / 2000
        self.assertAlmostEqual(cal_scales[same], 0.5)
        self.assertEqual(cal_scales, pred_scales)
        info = cal.meta["calibration"]
        self.assertEqual(info["scales"], {other: 0.25, same: 0.5})
        self.assertIn("width", info["scale_rule"])

    def test_calibrate_with_ground_resolution(self):
        clf = _make_classifier(target_mm_px=0.8, scale_factor=0.5,
                               meta={"ref_width": 1000})
        captured = {}

        def fake_extract(paths, xs, ys, scales, model, window, **kw):
            captured["scales"] = np.asarray(scales, float)
            n = len(paths)
            return np.random.default_rng(1).normal(size=(n, 3)), np.ones(n, bool)

        with tempfile.TemporaryDirectory() as d:
            p = _photo(d, "F1.jpg", 2000)
            clf.backbone_model = lambda: object()
            with mock.patch.object(pf, "extract_point_features", fake_extract):
                cal = pf.calibrate_point_classifier(clf, [p], gsd_mm=0.4, n_images=1,
                                                    per_image=2)
        np.testing.assert_allclose(captured["scales"], 0.5)  # 0.4 / 0.8
        self.assertEqual(cal.meta["calibration"]["scale"], 0.5)


class TestTrainingScales(unittest.TestCase):
    """M1: evaluation and curation re-read the points at the training scales."""

    def test_domain_points_width_scaled_as_in_training(self):
        with tempfile.TemporaryDirectory() as d:
            main = [_photo(d, f"m{i}.jpg", 1000) for i in range(3)]
            auv = _photo(d, "auv.jpg", 2000)
            rows = [dict(id=str(i), cam_filepath=p, _x=5.0, _y=5.0, domain="")
                    for i, p in enumerate(main)]
            rows.append(dict(id="9", cam_filepath=auv, _x=5.0, _y=5.0, domain="auv"))
            plan = pf.plan_training_scales(rows, d, feature_scale=0.4)
            clf = _make_classifier(scale_factor=plan.scale_factor,
                                   meta={"ref_width": plan.ref_width})
            got = pf.training_scales(clf, rows, d)
            np.testing.assert_allclose(got, plan.scales)
            self.assertAlmostEqual(got[-1], 0.2)
            # the old way (no photo widths) read the AUV point at 0.4
            self.assertAlmostEqual(clf.scales_for(None, len(rows))[-1], 0.4)

    def test_evaluate_uses_training_scales(self):
        with tempfile.TemporaryDirectory() as d:
            main = _photo(d, "m.jpg", 1000)
            auv = _photo(d, "auv.jpg", 2000)
            ann = _write_csv(os.path.join(d, "train.csv"), [
                {"id": str(i), "label": "A", "cam_filepath": p, "cam_x": "5",
                 "cam_y": "5", "split_key": "", "domain": dom}
                for i, (p, dom) in enumerate([(main, ""), (auv, "auv")] * 20)])
            clf = _make_classifier(scale_factor=0.5, meta={"ref_width": 1000},
                                   label_map={"A": "A"})
            seen = {}

            def fake_cached(keys, paths, xs, ys, scales, *a, **k):
                seen["scales"] = np.asarray(scales, float)
                return np.zeros((len(keys), 3), np.float32), np.ones(len(keys), bool)

            with mock.patch.object(pf, "cached_features", fake_cached), \
                    mock.patch.object(pf, "evaluate_predictions",
                                      lambda y, p: {"n": len(y)}):
                for split in ("train", "validation", "test"):
                    try:
                        pf.evaluate_point_classifier(clf, ann, d, split)
                        break
                    except ValueError:
                        continue
            rows = pf.read_training_points(ann, None)
        expect = [0.25 if r["domain"] else 0.5 for r in rows]
        np.testing.assert_allclose(seen["scales"], expect)


class TestMetresPerUnit(unittest.TestCase):
    """M5: an identity world transform falls back to the scale_factor."""

    def test_metres_per_unit(self):
        self.assertEqual(pf.metres_per_unit(np.eye(4), 0.02), 0.02)
        self.assertEqual(pf.metres_per_unit(None, 0.02), 0.02)
        self.assertEqual(pf.metres_per_unit(np.eye(4), None), 1.0)
        self.assertIsNone(pf.metres_per_unit(None, None))
        w = np.diag([0.5, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(pf.metres_per_unit(w, 0.02), 0.5)  # W wins
        self.assertEqual(pf.metres_per_unit(np.eye(4), 0), 1.0)  # (invalid factor)

    def test_image_match_gsd_uses_the_scale_factor(self):
        cams = types.SimpleNamespace(world_transform=np.eye(4), scale_factor=0.01)
        cam = types.SimpleNamespace(sensor=types.SimpleNamespace(fx=1000.0),
                                    parent=cams)
        match = types.SimpleNamespace(cam=cam, depth=200.0)  # 200 units = 2 m
        self.assertAlmostEqual(pf.image_match_gsd_mm(match), 2.0)


class TestDenseViewScales(unittest.TestCase):
    """M6: segment resamples photos by width like predict and masks."""

    def test_photo_widths_used(self):
        seg = _load_segmentation()
        with tempfile.TemporaryDirectory() as d:
            same = _photo(d, "a.jpg", 1000)
            wide = _photo(d, "b.jpg", 2000)
            cams = [types.SimpleNamespace(filepath=p, sensor=types.SimpleNamespace(
                fx=1000.0)) for p in (same, wide)]
            clf = _make_classifier(scale_factor=0.5, meta={"ref_width": 1000})
            seen = {}

            def fake_iter(clf_, paths, scales, **kw):
                seen.update(zip(paths, scales))
                return iter([(i, None) for i in range(len(paths))])

            vc = np.array([[0, 1]])
            v = np.full((1, 2), 10.0)
            with mock.patch.object(pf, "iter_dense_maps", fake_iter):
                seg._classify_dense_views(vc, v, v, v, np.array([[1.0, 2.0]]), cams,
                                          clf, depth_scale=1.0, verbose=False)
        self.assertAlmostEqual(seen[same], 0.5)
        self.assertAlmostEqual(seen[wide], 0.25)
        # the same rule as masks / predict
        self.assertAlmostEqual(seen[wide], pf.photo_scale(clf, None, 2000))


@unittest.skipIf(sklearn is None, "needs scikit-learn")
class TestProbeFoldsByPhoto(unittest.TestCase):
    """M7: the probe's C is chosen with folds grouped by photo."""

    def test_folds_never_split_a_photo(self):
        from sklearn import model_selection

        rng = np.random.default_rng(0)
        y = np.array(["A", "B"] * 60)
        z = rng.normal(size=(120, 4)).astype(np.float32)
        z[y == "A", 0] += 2
        groups = np.repeat(np.arange(24), 5)
        made = []
        real = model_selection.StratifiedGroupKFold

        class Recording(real):
            def split(self, X, yy=None, g=None):
                for tr, te in super().split(X, yy, g):
                    made.append((tr, te))
                    yield tr, te

        with mock.patch.object(model_selection, "StratifiedGroupKFold", Recording):
            classes, w, b, c = pf.fit_point_probe(z, y, cs=(0.01, 1.0), cv=4,
                                                  groups=groups)
        self.assertEqual(len(made), 4)
        for tr, te in made:
            self.assertFalse(set(groups[tr]) & set(groups[te]))
        self.assertEqual(classes, ["A", "B"])


class TestLowFindings(unittest.TestCase):
    def test_aggregation_missing_relevance_does_not_take_all_weight(self):
        # L2: a crop without a relevance used to get weight 1 against ~1e-12
        ims = [types.SimpleNamespace(relevance=None), types.SimpleNamespace(relevance=1.0),
               types.SimpleNamespace(relevance=1.0)]
        res = [{"label": "A", "probs": {"A": 1.0, "B": 0.0}},
               {"label": "B", "probs": {"A": 0.0, "B": 1.0}},
               {"label": "B", "probs": {"A": 0.0, "B": 1.0}}]
        out = cl.aggregate_predictions(ims, res, "weighted")
        self.assertEqual(out["label"], "B")
        self.assertAlmostEqual(out["confidence"], 2 / 3)
        zero = [types.SimpleNamespace(relevance=0.0), types.SimpleNamespace(relevance=2.0)]
        out = cl.aggregate_predictions(zero, res[:2], "weighted")
        self.assertAlmostEqual(out["probs"]["A"], 0.5)

    def test_bolding_rule_is_at_least_min_count(self):
        # L3: parents and tips both bold at >= --min-count
        self.assertTrue(cl._is_bold(True, False, 5, 9, 5, False))
        self.assertTrue(cl._is_bold(False, False, 5, 5, 5, False))
        self.assertFalse(cl._is_bold(True, False, 4, 9, 5, False))
        self.assertFalse(cl._is_bold(True, False, 5, 9, 5, True))

    def test_nan_confidence_is_no_value(self):
        # L4: 'nan' behaves like an empty / non-numeric label_conf
        for raw in ("nan", "NaN", "inf", "abc", ""):
            self.assertTrue(cl.confidence_passes(raw, 0.5))
            self.assertFalse(cl.confidence_passes(raw, 0.5, strict=True))
        self.assertTrue(cl.confidence_passes("0.7", 0.5))
        self.assertFalse(cl.confidence_passes("0.3", 0.5))

    def test_crop_box(self):
        # L12: the training crops' box (int(x - w / 2)), clipped
        self.assertEqual(cl.classifier_crop_box(100.7, 50.2, 224, 224), (-11, -61, 213, 163))
        self.assertEqual(cl.classifier_crop_box(100.7, 50.2, 224, 224, 300, 120), (0, 0, 213, 120))
        self.assertEqual(cl.classifier_crop_box(100.5, 100.5, 5, 5), (98, 98, 103, 103))

    def test_same_named_photos_do_not_share_calibration_patches(self):
        # L1: patch seeds and cache keys by file name collided for same-named photos
        with tempfile.TemporaryDirectory() as d:
            a = _photo(d, "x/IMG_1.jpg", 300)
            b = _photo(d, "y/IMG_1.jpg", 300)
            c = _photo(d, "IMG_2.jpg", 300)
            ids = pf.photo_ids([a, b, c])
            self.assertEqual(ids, {a: a, b: b, c: "IMG_2.jpg"})
            patches = pf.random_patches([a, b], n_images=2, per_image=3)
            pa = [(x, y) for p, x, y in patches if p == a]
            pb = [(x, y) for p, x, y in patches if p == b]
            self.assertEqual(len(pa), 3)
            self.assertNotEqual(pa, pb)

    def test_curation_bad_line_count_per_file(self):
        # L10: the warning counted the lines of earlier files again
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            with open(p1, "w") as f:
                f.write("{bad\n")
            with open(p2, "w") as f:
                f.write('{"key": "I.jpg|1|1", "action": "accept"}\n')
            with self.assertLogs(level="WARNING") as logs:
                store = cur.CurationStore(p2, extra=[p1])
            self.assertEqual(store.n_bad_lines, 1)
            self.assertEqual(len(logs.output), 1)

    def test_source_labels_edits_every_file_listing_the_point(self):
        # L11: a point in two annotation files was edited in the first only
        with tempfile.TemporaryDirectory() as d:
            row = {"id": "1", "cam_filepath": "/p/I1.jpg", "cam_x": "5", "cam_y": "6",
                   "label": "A", "label_conf": ""}
            f1 = _write_csv(os.path.join(d, "a_points.csv"), [row])
            f2 = _write_csv(os.path.join(d, "b_points.csv"), [row])
            cur.write_training_sources(d, [f1, f2], "annotations")
            src = cur.SourceLabels(d)
            key = cur.row_point_key(row)
            sa = types.ModuleType("substrata.annotations")

            def read_rows(path):
                with open(path, newline="") as fh:
                    rows = list(csv.reader(fh))
                return rows, "\n", [c.lower() for c in rows[0]]

            def write_rows(path, rows, eol):
                with open(path, "w", newline="") as fh:
                    csv.writer(fh, lineterminator=eol).writerows(rows)

            sa.read_csv_rows, sa.write_csv_rows = read_rows, write_rows
            sa.backup_annotation_csv = lambda path: None
            with mock.patch.dict(sys.modules, {"substrata.annotations": sa}), \
                    mock.patch.object(sys.modules["substrata"], "annotations", sa,
                                      create=True):
                self.assertEqual([p for p, _k in src.locate_all(key)], [f1, f2])
                res = src.set(key, "B")
                self.assertEqual(res["files"], [f1, f2])
                for f in (f1, f2):
                    with open(f, newline="") as fh:
                        self.assertEqual(next(csv.DictReader(fh))["label"], "B")

    def test_uint16_contrast_input(self):
        # L15: a dark 16-bit image is scaled by its dtype, not by "max > 1"
        ia = _load("imageadjust")
        a = np.full((2, 2, 3), 300, np.uint16)
        np.testing.assert_allclose(ia._to_float(a), 300 / 65535, rtol=1e-6)
        np.testing.assert_allclose(ia._to_float(np.full((1, 1, 3), 255, np.uint8)), 1.0)
        np.testing.assert_allclose(ia._to_float(np.full((1, 1, 3), 0.5)), 0.5)


@unittest.skipIf(torch is None, "needs torch")
class TestFeatureCacheConsistency(unittest.TestCase):
    """L14: a first (extracting) run and a cached run return the same features."""

    def test_first_and_cached_runs_equal(self):
        model = test_pointfeatures.FakeBackbone()
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            arr = (rng.uniform(size=(100, 100, 3)) * 255).astype(np.uint8)
            p = os.path.join(d, "n.png")
            Image.fromarray(arr).save(p)
            cache = os.path.join(d, "c.npz")
            args = (["k1", "k2"], [p, p], [30.0, 60.0], [30.0, 60.0], [1.0, 1.0], cache,
                    lambda: model, 28)
            first, _ok = pf.cached_features(*args)
            again, _ok = pf.cached_features(*args)
        np.testing.assert_array_equal(first, again)


class TestColourOutliers(unittest.TestCase):
    """M9: flagged card values are left out of the per-patch median."""

    def setUp(self):
        import test_color_calibration as tcc

        self.cc = tcc._load_color_calibration()

    def _calibrations(self, values):
        """``values[card][patch]`` (0-255 grey levels) measured on each card."""
        defs = [(f"p{j}", 0, 0, 10 * j, 10 * j, 10 * j) for j in range(len(values[0]))]

        class Card:
            def __init__(self, vals):
                self.vals, self.patch_results, self.name = vals, [], None

            def has_all_corners(self):
                return True

            def sample_patches(self, pcd, patch_definitions, **kw):
                self.patch_results = [{"median_rgb_0_1": np.full(3, v / 255.0)}
                                      for v in self.vals]

        obj = self.cc.ColorCalibrations([], patch_definitions=defs)
        obj.data = [Card(v) for v in values]
        obj._excluded_card_indices = frozenset()
        obj.sample_point_cloud(object())
        return obj

    def test_outlier_card_dropped_from_median(self):
        n = 6
        base = [10.0 * j for j in range(n)]
        cards = [list(base), [v + 2 for v in base], [v + 1 for v in base]]
        cards[2][3] = 200.0  # one card's patch 3 is far off
        obj = self._calibrations(cards)
        self.assertTrue(obj.outlier_mask[2, 3])
        self.assertEqual(int(obj.outlier_mask.sum()), 1)
        # median of the two unflagged cards (30, 32), not of all three (32)
        np.testing.assert_allclose(obj.median_rgb_255_per_patch[3], 31.0)
        self.assertTrue(obj.patch_fit_mask.all())
        obj.compute_color_correction()
        self.assertEqual(obj.fit_stats["n_patches_fit"], n)
        self.assertGreaterEqual(obj.fit_stats["rmse"], 0.0)
        self.assertEqual(obj.fit_stats["space"], "sRGB 0-255")


class TestFireFishDepth(unittest.TestCase):
    """M4: --target-depth is a depth below the surface, whatever its sign."""

    def test_infer_target_depth(self):
        cli = _cli()
        self.assertEqual(cli._infer_target_depth("cur_sna_20m_20200303", None), -20.0)
        self.assertEqual(cli._infer_target_depth("x", 20), -20.0)
        self.assertEqual(cli._infer_target_depth("x", -20.5), -20.5)
        self.assertIsNone(cli._infer_target_depth("plot", None))

    def test_empty_depth_band_is_a_clear_error(self):
        ff = _load_firefish()
        if ff is None:
            self.skipTest("firefish needs pandas")
        import pandas as pd

        fish = ff.FireFish.__new__(ff.FireFish)
        fish.data = pd.DataFrame({"unixtime": [0, 1, 2], "depth": [-5.0, -5.5, -6.0]})
        with self.assertRaises(ValueError) as e:
            fish.filter_by_depth_range(-24, -16)
        self.assertIn("No FireFish records", str(e.exception))
        self.assertEqual(len(fish.filter_by_depth_range(-7, -5.2)), 2)

    def test_offset_search_starts_at_the_earliest_times(self):
        ff = _load_firefish()
        if ff is None:
            self.skipTest("firefish needs pandas")
        import pandas as pd

        class Cam:
            coords = (0.0, 0.0, 0.0)

            def __init__(self, dt):
                self.datetime = dt

            def has_coords_datetime(self):
                return True

        class Cams(list):
            def get_time_delta_between_first_and_last_photo(self):
                return 10

        # cameras and records out of time order: the earliest ones count
        cams = Cams([Cam("2020:01:01 00:00:30"), Cam("2020:01:01 00:00:20")])
        t0 = ff.get_unix_time("2020:01:01 00:00:00")
        fish = ff.FireFish.__new__(ff.FireFish)
        fish.data = pd.DataFrame({"unixtime": [t0 + 5.0, t0 + 0.0, t0 + 1.0],
                                  "depth": [-20.0] * 3, "altitude": [1.0] * 3})
        tried = []
        fish.get_camera_coords_and_depths_from_firefish = (
            lambda cams, offset, sensor_offset_units=None: tried.append(offset) or ([], []))
        ff.cameras.sensor_offset_units = lambda *a: None
        with mock.patch.object(ff, "Parallel", lambda **k: list), \
                mock.patch.object(ff, "delayed", lambda f: f), \
                mock.patch.object(ff, "tqdm", lambda it, **k: it):
            fish.determine_camera_time_offset_based_on_lowest_depth_regression_error(
                cams, depth_and_outlier_threshold=5)
        start = ff.get_time_diff_s(t0, "2020:01:01 00:00:20")
        self.assertEqual(start, -20)
        self.assertEqual((min(tried), max(tried)), (start - 10, 3 + start - 1))


class TestCommandLine(unittest.TestCase):
    """CLI: hyphen spellings with hidden underscore aliases; colors frames; masks."""

    def setUp(self):
        self.cli = _cli()
        if self.cli is None:
            self.skipTest("substrata.cli not importable here")

    def test_hyphen_options_and_hidden_aliases(self):
        p = self.cli.build_parser()
        cases = [
            (["decimate", "--color-calibrate"], "color_calibrate", True),
            (["decimate", "--color_calibrate"], "color_calibrate", True),
            (["colors", "--output-pdf", "x.pdf"], "output_pdf", "x.pdf"),
            (["colors", "--output_pdf", "x.pdf"], "output_pdf", "x.pdf"),
            (["scalebars", "--save_yaml"], "save_yaml", True),
            (["firefish", "--cams-group", "g"], "cams_group", "g"),
            (["firefish", "--target-depth", "20.5"], "target_depth", 20.5),
            (["cams2video", "--output-mp4", "v.mp4"], "output_mp4", "v.mp4"),
            (["transform", "in.csv", "--no-header"], "no_header", True),
            (["transform", "in.csv", "--ignore_header"], "ignore_header", True),
            (["train", "--tips-only"], "tips_only", True),
            (["train", "--min_conf", "0.5"], "min_conf", 0.5),
            (["train", "--min-conf-strict", "0.5"], "min_conf_strict", 0.5),
            (["train"], "split_by", "photo"),
            (["train", "--split-by", "point"], "split_by", "point"),
            (["orient", "--use_chunk_transform", "--north-up"], "north_up", True),
            (["curate", "--all-splits"], "all_splits", True),
        ]
        for argv, dest, value in cases:
            self.assertEqual(getattr(p.parse_args(argv), dest), value, argv)
        sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
        for name, shown, hidden in (("train", "--min-conf", "--min_conf"),
                                    ("colors", "--save-yaml", "--save_yaml"),
                                    ("orient", "--north-up", "--north_up")):
            text = sub.choices[name].format_help()
            self.assertIn(shown, text)
            self.assertNotIn(hidden, text)

    def test_colors_puts_cloud_and_markers_in_the_world_frame(self):
        # M2 / M3: the sampling radii are metres, so both go to the world frame
        world = np.diag([0.01, 0.01, 0.01, 1.0])
        calls = []

        class Init:
            def __init__(self, path=None, local=False):
                self.world_transform = world
                self.markers_filepath = "m.csv"
                self.ply_filepath = "c.ply"
                self.markers = None
                self.pcd = None
                self.path, self.id, self.yaml_path = path, "plot", None

            def initialize(self, **kw):
                self.pcd = Thing("init.pcd")
                self.markers = Thing("init.markers")

        class Thing:
            def __init__(self, name, *a, **k):
                self.name = name

            def apply_transform(self, m):
                calls.append((self.name, "apply_transform", np.asarray(m)))

            def transform_coords(self, m):
                calls.append((self.name, "transform_coords", np.asarray(m)))

        got = {}

        class CC:
            def __init__(self, **kw):
                got.update(kw)
                self.color_correction = None
                self.fit_stats = None

            def save_pdf(self, filepath=None):
                pass

        init_mod = types.SimpleNamespace(ProjectInitializer=Init)
        cc_mod = types.SimpleNamespace(ColorCalibrations=CC)
        with mock.patch.dict(sys.modules, {"substrata.initializer": init_mod,
                                           "substrata.color_calibration": cc_mod}), \
                mock.patch.object(self.cli, "PointCloud",
                                  lambda path, max_points=None: Thing("cloud")), \
                mock.patch.object(self.cli, "Annotations",
                                  lambda path, **k: Thing("markers")), \
                tempfile.TemporaryDirectory() as d, mock.patch("os.getcwd", lambda: d):
            base = dict(input=None, markers=None, output_pdf=os.path.join(d, "o.pdf"),
                        save_yaml=False, exclude_indices=None, exclude_names=None,
                        local=False)
            # -n: streamed cloud + markers CSV, both from the model frame
            self.cli.handle_colors(types.SimpleNamespace(points=1000, **base))
            self.assertEqual([(n, m) for n, m, _w in calls],
                             [("cloud", "apply_transform"), ("markers", "apply_transform")])
            self.assertTrue(all(np.allclose(w, world) for *_x, w in calls))
            self.assertEqual((got["pcd"].name, got["target_data"].name),
                             ("cloud", "markers"))
            # no -n but --markers: the project cloud is already in the world frame
            calls.clear()
            self.cli.handle_colors(types.SimpleNamespace(
                points=None, **dict(base, markers="other.csv")))
            self.assertEqual([(n, m) for n, m, _w in calls],
                             [("markers", "apply_transform")])
            self.assertEqual(got["pcd"].name, "init.pcd")

    def test_label_agreement_maps_labels(self):
        # L5: existing labels are mapped to the classifier's classes first
        anns = [types.SimpleNamespace(label="CST", pred_label="CORAL"),
                types.SimpleNamespace(label="SAND", pred_label="CORAL"),
                types.SimpleNamespace(label="UNK", pred_label="CORAL")]
        out = []
        with mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a)))):
            self.cli._print_label_agreement(anns, {"CORAL", "SAND"},
                                            {"CST": "CORAL", "SAND": "SAND"})
        text = "\n".join(out)
        self.assertIn("50.0%", text)
        self.assertIn("1 labelled point(s) have labels the classifier's label map", text)

    def test_seed_is_drawn_and_recorded(self):
        # L13: sampling without --seed still records the seed used
        args = types.SimpleNamespace(seed=None)
        with mock.patch("builtins.print"):
            seed = self.cli._resolve_seed(args)
        self.assertEqual(args.seed, seed)
        self.assertEqual(self.cli._resolve_seed(types.SimpleNamespace(seed=7)), 7)

    def test_label_color_specs(self):
        self.assertEqual(self.cli._parse_label_color_specs(["A=ff0000", "B C=#00ff00"]),
                         {"A": (255, 0, 0), "B C": (0, 255, 0)})
        with self.assertRaises(SystemExit):
            self.cli._parse_label_color_specs(["A"])


@unittest.skipIf(torch is None, "needs torch")
class TestMasksReuse(unittest.TestCase):
    """M10: masks are only reused when made by the same classifier file."""

    def test_retrained_classifier_redoes_masks(self):
        cli = _cli()
        if cli is None:
            self.skipTest("substrata.cli not importable here")
        with tempfile.TemporaryDirectory() as d:
            photo = _photo(d, "p.jpg", 56, 56)
            clf_path = os.path.join(d, "m.pt")
            _make_classifier(meta={"ref_width": 56}).save(clf_path)
            out = os.path.join(d, "masks")
            done = []

            def fake_iter(clf, paths, scales, **kw):
                done.extend(paths)
                for i, _p in enumerate(paths):
                    yield i, pf.DenseMap(np.full((4, 4, 3), 1 / 3, np.float32), 1.0, 14,
                                         56, 56, list(clf.classes))

            args = types.SimpleNamespace(classifier=clf_path, images=[photo], gsd_mm=None,
                                         output=out, color=None, overwrite=False,
                                         jobs=1, probs=False)
            from substrata import pointfeatures as real_pf
            with mock.patch.object(real_pf, "iter_dense_maps", fake_iter), \
                    mock.patch("builtins.print"):
                cli.handle_masks(args)
                cli.handle_masks(args)  # same classifier: reused
                self.assertEqual(len(done), 1)
                # retrained: same classes, other weights
                _make_classifier(meta={"ref_width": 56},
                                 bias=np.ones(3, np.float32)).save(clf_path)
                cli.handle_masks(args)
                self.assertEqual(len(done), 2)
            with open(os.path.join(out, settings.MASKS_INDEX_FILE)) as f:
                index = json.load(f)
            self.assertTrue(index["classifier_fingerprint"].startswith("m"))


def _cli():
    """The real ``substrata.cli`` (needs the installed package), else None."""
    try:
        import importlib

        return importlib.import_module("substrata.cli")
    except Exception:  # noqa: BLE001 - outside the substrata environment
        return None


def _load_segmentation():
    import test_segmentation

    return test_segmentation.seg


def _load_firefish():
    """firefish.py with its heavy siblings stubbed (only pandas is needed)."""
    try:
        import pandas  # noqa: F401
    except ImportError:  # pragma: no cover
        return None
    import importlib.util

    key = "substrata.firefish_under_test"
    if key in sys.modules:
        return sys.modules[key]
    stubs = {f"substrata.{m}": types.ModuleType(f"substrata.{m}")
             for m in ("visualizations", "measurements", "cameras")}
    pkg = sys.modules["substrata"]
    saved = {m: getattr(pkg, m.split(".")[1], None) for m in stubs}
    with mock.patch.dict(sys.modules, stubs):
        for m, mod in stubs.items():
            setattr(pkg, m.split(".")[1], mod)
        try:
            spec = importlib.util.spec_from_file_location(
                key, os.path.join(os.path.dirname(pf.__file__), "firefish.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            for m, old in saved.items():
                if old is None:
                    delattr(pkg, m.split(".")[1])
                else:
                    setattr(pkg, m.split(".")[1], old)
    sys.modules[key] = mod
    return mod


class TestSourceLabelsLock(unittest.TestCase):
    """SourceLabels.set holds the CSV's fileio.path_lock from read to write,
    as every other annotation CSV editor does."""

    @staticmethod
    def _csv_module():
        """Plain CSV helpers in place of annotations' (whose module other test
        modules may have left stubbed in sys.modules)."""
        sa = types.ModuleType("substrata.annotations")

        def read(path):
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            return rows, "\r\n", [h.strip().lower() for h in rows[0]]

        def write(path, rows, eol="\r\n"):
            with open(path, "w", newline="") as f:
                csv.writer(f, lineterminator=eol).writerows(rows)

        sa.read_csv_rows, sa.write_csv_rows = read, write
        sa.backup_annotation_csv = lambda path: None
        return sa

    def test_waits_for_the_file_lock(self):
        import threading
        import time

        sa = self._csv_module()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(sys.modules, {"substrata.annotations": sa}), \
                mock.patch.object(sys.modules["substrata"], "annotations", sa, create=True):
            path = _write_csv(os.path.join(d, "pts.csv"), [
                {"id": "1", "cam_filepath": "/p/I1.jpg", "cam_x": "5", "cam_y": "6",
                 "label": "A", "orig_x": "nan", "world_x": "nan"}])
            cur.write_training_sources(d, [path], "annotations")
            src = cur.SourceLabels(d)
            key = cur.point_key("/p/I1.jpg", 5, 6)
            held, release = threading.Event(), threading.Event()

            def holder():
                with cur.path_lock(path):
                    held.set()
                    release.wait(5)

            h = threading.Thread(target=holder)
            h.start()
            held.wait(5)
            done = []

            def edit():
                try:
                    done.append(src.set(key, "B"))
                except Exception as e:  # noqa: BLE001 - reported below
                    done.append(e)

            t = threading.Thread(target=edit)
            t.start()
            time.sleep(0.2)
            blocked = list(done)
            release.set()
            t.join(5)
            h.join(5)
            self.assertEqual(blocked, [])  # waited for the other editor's lock
            self.assertEqual(len(done), 1)
            if isinstance(done[0], Exception):
                raise done[0]
            self.assertTrue(done[0]["changed"])
            self.assertEqual(src.get(key)["label"], "B")

if __name__ == "__main__":
    unittest.main()
