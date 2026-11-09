"""Tests for training-data curation (substrata.curation) and its hooks in
training (label counts, collation, per-domain standardisation, stable scales).

Modules are loaded without the ``substrata`` package (see test_pointfeatures);
scoring and pools run end to end with a tiny fake backbone (mean colour per
14 px patch) on synthetic photos, so no download or GPU is needed.
"""

# Standard Library
import csv
import io
import json
import os
import tempfile
import unittest
from collections import Counter

# Third-Party Libraries
import numpy as np
from PIL import Image

from test_pointfeatures import (
    FakeBackbone,
    _load,
    _make_classifier,
    torch,
)  # noqa: F401

pf = _load("pointfeatures")
cl = _load("classification")
cur = _load("curation")
settings = pf.settings

try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover
    sklearn = None


def _row(photo, x, y, label, **kw):
    r = {
        "id": kw.pop("id", "p"),
        "cam_filepath": photo,
        "cam_x": str(x),
        "cam_y": str(y),
        "label": label,
        "label_conf": kw.pop("label_conf", ""),
    }
    r.update(kw)
    return r


class TestKeysAndStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.path = os.path.join(self.d, settings.CURATION_FILE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_point_key_uses_file_name_and_rounded_pixel(self):
        self.assertEqual(
            cur.point_key("/a/b/IMG_1.jpg", "10.4", 20.6), "IMG_1.jpg|10|21"
        )
        # moved photo folders keep the key
        self.assertEqual(cur.point_key("/other/IMG_1.jpg", 10, 21), "IMG_1.jpg|10|21")
        self.assertEqual(
            cur.row_point_key(_row("/x/IMG_1.jpg", "10.2", "21.1", "A")),
            "IMG_1.jpg|10|21",
        )
        self.assertIsNone(cur.row_point_key({"label": "A"}))

    def test_latest_decision_wins_and_clear_withdraws(self):
        s = cur.CurationStore(self.path)
        k = cur.point_key("IMG.jpg", 1, 2)
        s.append([{"key": k, "action": "exclude", "reason": "mixed"}])
        s.append([{"key": k, "action": "relabel", "label": "B"}])
        self.assertEqual(s.get(k)["action"], "relabel")
        s2 = cur.CurationStore(self.path)  # reread from disk
        self.assertEqual(s2.get(k)["label"], "B")
        s2.append([{"key": k, "action": "clear"}])
        self.assertIsNone(cur.CurationStore(self.path).get(k))
        self.assertEqual(len(cur.CurationStore(self.path)), 0)

    def test_validation(self):
        s = cur.CurationStore(self.path)
        with self.assertRaises(ValueError):
            s.append([{"key": "k", "action": "dance"}])
        with self.assertRaises(ValueError):
            s.append([{"key": "k", "action": "relabel"}])
        with self.assertRaises(ValueError):
            s.append([{"action": "accept"}])
        with self.assertRaises(ValueError):
            cur.CurationStore(None).append([{"key": "k", "action": "accept"}])

    def test_apply(self):
        s = cur.CurationStore(self.path)
        a = _row("/p/A.jpg", 5, 5, "X")
        b = _row("/p/B.jpg", 5, 5, "X")
        c = _row("/p/C.jpg", 5, 5, "X")
        s.append(
            [
                {"key": cur.row_point_key(a), "action": "exclude", "reason": "unclear"},
                {"key": cur.row_point_key(b), "action": "relabel", "label": "Y"},
                {"key": cur.row_point_key(c), "action": "accept", "orig_label": "X"},
            ]
        )
        self.assertEqual(s.apply(a), (None, "exclude"))
        row, act = s.apply(b)
        self.assertEqual((row["label"], act), ("Y", "relabel"))
        self.assertEqual(b["label"], "X")  # the input row is not changed
        self.assertEqual(s.apply(c)[1], "accept")
        # an accept no longer holds once the point's label changed
        self.assertEqual(s.apply(dict(c, label="Z")), (dict(c, label="Z"), None))

    def test_extra_stores_are_read_only_and_overridden(self):
        other = os.path.join(self.d, "other.jsonl")
        k = cur.point_key("IMG.jpg", 1, 2)
        cur.CurationStore(other).append([{"key": k, "action": "exclude"}])
        s = cur.CurationStore(self.path, [other])
        self.assertEqual(s.get(k)["action"], "exclude")
        s.append([{"key": k, "action": "accept"}])
        self.assertEqual(
            cur.CurationStore(self.path, [other]).get(k)["action"], "accept"
        )
        with open(other) as f:
            self.assertEqual(len(f.readlines()), 1)
        with self.assertRaises(FileNotFoundError):
            cur.CurationStore(self.path, [os.path.join(self.d, "missing.jsonl")])

    def test_bad_lines_are_skipped(self):
        with open(self.path, "w") as f:
            f.write('{"key": "a|1|1", "action": "accept"}\nnot json\n{"no": "key"}\n')
        s = cur.CurationStore(self.path)
        self.assertEqual(len(s), 1)
        self.assertEqual(s.n_bad_lines, 2)

    def test_import_and_export(self):
        s = cur.CurationStore(self.path)
        src = os.path.join(self.d, "dec.csv")
        with open(src, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["id", "Action", "label", "reason", "cam_filepath", "cam_x", "cam_y"]
            )
            w.writerow(["p1", "exclude", "", "photo", "", "", ""])  # by id
            w.writerow(
                ["", "relabel", "Y", "", "/q/I.jpg", "3", "4"]
            )  # by photo + pixel
            w.writerow(["p9", "accept", "", "", "", "", ""])  # unknown id
            w.writerow(["p1", "relabel", "", "", "", "", ""])  # no label
        lookup = {"p1": {"key": "I.jpg|1|1", "label": "A", "id": "p1"}}
        n, problems = cur.import_decisions(s, src, lookup)
        self.assertEqual(n, 2)
        self.assertEqual(len(problems), 2)
        self.assertEqual(s.get("I.jpg|1|1")["orig_label"], "A")
        self.assertEqual(s.get("I.jpg|3|4")["label"], "Y")
        out = os.path.join(self.d, "out.csv")
        self.assertEqual(s.export_csv(out), 2)
        with open(out) as f:
            self.assertEqual(
                {r["action"] for r in csv.DictReader(f)}, {"exclude", "relabel"}
            )


class TestTrainingHooks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.csv = os.path.join(self.d, "m1_slope_intercepts.csv")
        rows = [
            _row(
                os.path.join(self.d, f"I{i}.jpg"),
                10,
                10,
                lab,
                id=str(i),
                label_conf=conf,
            )
            for i, (lab, conf) in enumerate(
                [("A", ""), ("A", ""), ("B", "0.2"), ("B", ""), ("C", "")]
            )
        ]
        with open(self.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.rows = rows
        self.store = cur.CurationStore(os.path.join(self.d, settings.CURATION_FILE))
        self.store.append(
            [
                {"key": cur.row_point_key(rows[0]), "action": "exclude"},
                {"key": cur.row_point_key(rows[4]), "action": "relabel", "label": "A"},
                {
                    "key": cur.row_point_key(rows[2]),
                    "action": "accept",
                    "orig_label": "B",
                },
            ]
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_count_labels(self):
        counts, _ = cl.count_labels([self.csv], None)
        self.assertEqual(counts, Counter({"A": 2, "B": 2, "C": 1}))
        counts, _ = cl.count_labels([self.csv], None, self.store)
        self.assertEqual(counts, Counter({"A": 2, "B": 2}))

    def test_collate(self):
        out = os.path.join(self.d, "training_annotations.csv")
        stats = Counter()
        n, _drop, n_conf = cl.collate_training_annotations(
            [self.csv],
            "*_slope_intercepts.csv",
            {"A", "B"},
            out,
            self.d,
            prompt=False,
            min_conf=0.5,
            curation=self.store,
            curation_stats=stats,
        )
        with open(out) as f:
            got = {r["id"]: r["label"] for r in csv.DictReader(f)}
        # 0 excluded; 4 relabelled C -> A; 2 accepted, so kept despite a low label_conf
        self.assertEqual(got, {"m1_1": "A", "m1_2": "B", "m1_3": "B", "m1_4": "A"})
        self.assertEqual((n, n_conf), (4, 0))
        self.assertEqual(stats, Counter({"exclude": 1, "relabel": 1, "accept": 1}))

    def test_collate_keeps_gsd_and_domain(self):
        pool = os.path.join(self.d, "pool.csv")
        with open(pool, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "cam_filepath",
                    "cam_x",
                    "cam_y",
                    "label",
                    "cam_gsd_mm",
                    "domain",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "id": "pool_0",
                    "cam_filepath": os.path.join(self.d, "F.png"),
                    "cam_x": "5.0",
                    "cam_y": "6.0",
                    "label": "",
                    "cam_gsd_mm": "1.2",
                    "domain": "auv",
                }
            )
            w.writerow(
                {
                    "id": "pool_1",
                    "cam_filepath": os.path.join(self.d, "F.png"),
                    "cam_x": "9.0",
                    "cam_y": "6.0",
                    "label": "",
                    "cam_gsd_mm": "1.2",
                    "domain": "auv",
                }
            )
        self.store.append([{"key": "F.png|5|6", "action": "relabel", "label": "B"}])
        out = os.path.join(self.d, "training_annotations.csv")
        cl.collate_training_annotations(
            [pool], "*.csv", {"B"}, out, self.d, prompt=False, curation=self.store
        )
        with open(out) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)  # the unlabelled pool point stays out
        self.assertEqual(
            (rows[0]["label"], rows[0]["cam_gsd_mm"], rows[0]["domain"]),
            ("B", "1.2", "auv"),
        )


class TestScalesAndDomains(unittest.TestCase):
    def test_domain_rows_do_not_move_the_target(self):
        rows = [
            dict(_row(f"/p/{i}.jpg", 1, 1, "A"), cam_gsd_mm="0.2", id=str(i))
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as d:
            for r in rows:
                r["_x"], r["_y"] = 1.0, 1.0
            base = pf.plan_training_scales(rows, d, feature_scale=None, window=28)
            more = rows + [
                dict(
                    _row(f"/a/{i}.png", 1, 1, "B"),
                    cam_gsd_mm="1.5",
                    domain="auv",
                    id=f"a{i}",
                    _x=1.0,
                    _y=1.0,
                )
                for i in range(30)
            ]
            plan = pf.plan_training_scales(more, d, feature_scale=None, window=28)
        self.assertEqual(plan.target_mm_px, base.target_mm_px)
        self.assertAlmostEqual(plan.scales[-1], 1.5 / plan.target_mm_px, places=5)

    def test_random_patches_stable_when_photos_change(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(12):
                p = os.path.join(d, f"P{i}.jpg")
                Image.new("RGB", (60, 40)).save(p)
                paths.append(p)
            a = pf.random_patches(paths, n_images=6, per_image=2)
            b = pf.random_patches(paths[1:], n_images=6, per_image=2)
        common = {p for p, _x, _y in a} & {p for p, _x, _y in b}
        self.assertGreaterEqual(len(common), 5)
        for p in common:
            self.assertEqual([t for t in a if t[0] == p], [t for t in b if t[0] == p])

    @unittest.skipIf(torch is None, "needs torch")
    def test_domains_round_trip_and_standardise(self):
        dom = {"auv": (np.full(3, 10, np.float32), np.full(3, 2, np.float32))}
        clf = _make_classifier(domains=dom)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            clf.save(p)
            back = pf.PointFeatureClassifier.load(p)
        np.testing.assert_allclose(back.domains["auv"][0], 10)
        f = np.full((2, 3), 12, np.float32)
        z = back.standardise_rows(f, ["", "auv"])
        np.testing.assert_allclose(z[0], 12)
        np.testing.assert_allclose(z[1], 1)
        z3 = back.standardise_rows(np.full((2, 4, 3), 12, np.float32), ["auv", ""])
        np.testing.assert_allclose(z3[0], 1)
        p1 = back.predict_proba_standardised(z)
        np.testing.assert_allclose(p1.sum(1), 1, rtol=1e-5)


class TestHelpers(unittest.TestCase):
    def test_ring_offsets(self):
        r = cur.ring_offsets(10, 8)
        self.assertEqual(r.shape, (8, 2))
        np.testing.assert_allclose(np.hypot(r[:, 0], r[:, 1]), 10, rtol=1e-6)

    def test_knn_skips_same_group(self):
        z = cur._l2(np.array([[1, 0], [1, 0.01], [0, 1], [0.1, 1]], np.float32))
        g = np.array([0, 0, 1, 1])
        sims, idx = cur._knn(z, z, 1, g, g)
        self.assertEqual(idx[:, 0].tolist(), [3, 3, 1, 1])
        sims, idx = cur._knn(z, z, 2)
        self.assertEqual(idx[0, 0], 0)  # without groups, itself first

    def test_grid_points(self):
        rng = np.random.default_rng(0)
        pts = cur._grid_points(400, 200, 12, 50, 30, rng)
        self.assertEqual(len(pts), 12)
        for x, y in pts:
            self.assertTrue(50 <= x <= 350 and 30 <= y <= 170)
            self.assertEqual(round(x, 1), x)

    @unittest.skipIf(sklearn is None, "needs scikit-learn")
    def test_cv_probabilities(self):
        rng = np.random.default_rng(0)
        y = np.array(["A"] * 30 + ["B"] * 30)
        z = rng.normal(size=(60, 4)).astype(np.float32)
        z[:30, 0] += 4
        groups = np.repeat(np.arange(20), 3)
        p = cur._cv_probabilities(z, y, groups, ["A", "B", "C"], 1.0, 5)
        np.testing.assert_allclose(p.sum(1), 1, rtol=1e-5)
        self.assertTrue((p[:, 2] == 0).all())
        self.assertGreater((p[:, :2].argmax(1) == (y == "B")).mean(), 0.9)


def _photos(d, n=6, size=(280, 140)):
    """Photos half red (left) and half blue (right)."""
    out = []
    for i in range(n):
        im = Image.new("RGB", size, (200, 30, 30))
        im.paste((30, 30, 200), (size[0] // 2, 0, size[0], size[1]))
        p = os.path.join(d, f"IMG_{i}.jpg")
        im.save(p, quality=95)
        out.append(p)
    return out


@unittest.skipIf(torch is None or sklearn is None, "needs torch and scikit-learn")
class TestScoringAndPools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.photos = _photos(self.d)
        # red = "R", blue = "B" (raw labels Rr / Bb map to them); a few mislabelled
        self.clf = _make_classifier(
            classes=["B", "R"],
            weights=np.array([[-1, 0, 1], [1, 0, -1]], np.float32) * 3,
            bias=np.zeros(2, np.float32),
            source_mean=np.full(3, 0.45, np.float32),
            source_std=np.full(3, 0.3, np.float32),
            window=28,
            scale_factor=0.5,
            label_map={"Rr": "R", "Bb": "B"},
            meta={"probe_C": 1.0},
        )
        self.clf.backbone_model = lambda: FakeBackbone()
        rows = []
        k = 0
        for p in self.photos:
            for x in (35, 70, 210, 245):
                for y in (35, 105):
                    lab = "Rr" if x < 140 else "Bb"
                    rows.append(
                        {
                            "id": f"pt_{k}",
                            "label": lab,
                            "cam_filepath": p,
                            "cam_x": str(x),
                            "cam_y": str(y),
                        }
                    )
                    k += 1
        rows[0]["label"] = "Bb"  # wrong
        rows[12]["label"] = "Rr"  # wrong (blue)
        self.ann = os.path.join(self.d, settings.TRAIN_ANNOTATIONS_FILE)
        with open(self.ann, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=settings.TRAIN_ANN_COLUMNS, extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_score_training_points(self):
        summary = cur.score_training_points(self.d, self.clf, progress=False, folds=3)
        self.assertEqual(summary["n_points"], 48)
        rows = cur.read_table(summary["table"])
        self.assertEqual(list(rows[0]), cur.POINT_COLUMNS)
        p = {r["id"]: float(r["p_label"]) for r in rows}
        # the two mislabelled points are the most suspect
        worst = sorted(p, key=p.get)[:2]
        self.assertEqual(set(worst), {"pt_0", "pt_12"})
        tiles = cur.TileStore(summary["tiles"])
        for r in rows:
            data, x, y = tiles.get(r["tile"])
            self.assertEqual(Image.open(io.BytesIO(data)).size, (224, 224))
            self.assertEqual((x, y), (float(r["tile_x"]), float(r["tile_y"])))
            self.assertEqual(r["key"], cur.point_key(r["photo"], r["x"], r["y"]))
            self.assertTrue(0 <= float(r["ring_agree"]) <= 1)
        with open(
            os.path.join(cur.curation_dir(self.d), settings.CURATION_META_FILE)
        ) as f:
            meta = json.load(f)
        self.assertEqual(meta["n_points"], 48)
        # second run: everything cached (float16), same scores
        again = cur.read_table(
            cur.score_training_points(self.d, self.clf, progress=False, folds=3)[
                "table"
            ]
        )
        np.testing.assert_allclose(
            [float(r["p_label"]) for r in again],
            [float(r["p_label"]) for r in rows],
            atol=2e-3,
        )

    def test_mine_pool(self):
        meta = cur.mine_pool(
            self.d,
            self.clf,
            self.photos,
            "auv test",
            gsd_mm=None,
            n_images=4,
            per_image=6,
            n_clusters=3,
            progress=False,
        )
        self.assertEqual(meta["name"], "auv_test")
        self.assertEqual(meta["domain"], "auv_test")
        rows = cur.read_table(meta["csv"])
        self.assertEqual(len(rows), 24)
        for r in rows:
            self.assertEqual(
                r["key"], cur.point_key(r["cam_filepath"], r["cam_x"], r["cam_y"])
            )
            self.assertEqual(r["label"], "")
            self.assertEqual(r["domain"], "auv_test")
        self.assertEqual(sum(c["n"] for c in meta["clusters"]), 24)
        # points well inside the red and the blue halves never share a cluster
        red = {r["cluster"] for r in rows if float(r["cam_x"]) < 100}
        blue = {r["cluster"] for r in rows if float(r["cam_x"]) > 180}
        self.assertTrue(red and blue and not red & blue)
        self.assertIn("auv_test", cur.domain_statistics(self.d))
        # re-run with other clusters: same points, no new sampling
        again = cur.mine_pool(
            self.d,
            self.clf,
            self.photos,
            "auv test",
            n_images=4,
            per_image=6,
            n_clusters=2,
            progress=False,
        )
        self.assertEqual(
            [r["key"] for r in cur.read_table(again["csv"])], [r["key"] for r in rows]
        )
        self.assertEqual(again["n_clusters"], 2)
        # thumbnails: all stored; a deleted store is re-rendered on the next run
        names = [r["tile"] for r in rows]
        self.assertEqual(len(cur.tile_store(self.d).positions(names)), 24)
        os.remove(os.path.join(cur.curation_dir(self.d), settings.CURATION_TILES_FILE))
        cur.mine_pool(self.d, self.clf, self.photos, "auv test", n_images=4,
                      per_image=6, n_clusters=2, progress=False)
        self.assertEqual(len(cur.tile_store(self.d).positions(names)), 24)


if __name__ == "__main__":
    unittest.main()


class TestBackbones(unittest.TestCase):
    def test_short_hands(self):
        self.assertEqual(pf.resolve_arch("dinov3"), "dinov3_vitb16")
        self.assertEqual(pf.resolve_arch("DINOv3-L"), "dinov3_vitl16")
        self.assertEqual(pf.resolve_arch("dinov2"), "dinov2_vitb14_reg")
        self.assertEqual(pf.resolve_arch("dinov3_vits16plus"), "dinov3_vits16plus")
        self.assertEqual(pf.resolve_arch("resnet34"), "resnet34")
        for a in ("dinov3", "dinov3-h", "dinov2-l", "dinov2_vitb14_reg", "dinov3_vitb16"):
            self.assertTrue(pf.is_point_arch(a), a)
        self.assertFalse(pf.is_point_arch("resnet34"))
        self.assertEqual(pf.backbone_patch_size("dinov3_vitb16"), 16)
        self.assertEqual(pf.backbone_patch_size("dinov2_vitb14_reg"), 14)

    def test_dinov3_weights_lookup(self):
        env = settings.POINT_DINOV3_WEIGHTS_ENV
        old = os.environ.pop(env, None)
        try:
            with tempfile.TemporaryDirectory() as d:
                os.environ[env] = d
                hub = torch.hub.get_dir() if torch is not None else None
                if hub is not None:
                    torch.hub.set_dir(os.path.join(d, "hub"))
                try:
                    with self.assertRaises(FileNotFoundError) as e:
                        pf.dinov3_weights("dinov3_vitb16")
                    self.assertIn(settings.POINT_DINOV3_ACCESS_URL, str(e.exception))
                    f = os.path.join(d, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
                    open(f, "wb").close()
                    self.assertEqual(pf.dinov3_weights("dinov3_vitb16"), f)  # folder
                    os.environ[env] = f
                    self.assertEqual(pf.dinov3_weights("dinov3_vitb16"), f)  # file
                    with self.assertRaises(ValueError):  # another model's file
                        pf.dinov3_weights("dinov3_vitl16")
                    os.environ[env] = "https://example.org/x/dinov3_vitb16_p.pth?sig=1"
                    self.assertTrue(pf.dinov3_weights("dinov3_vitb16").startswith("https"))
                finally:
                    if hub is not None:
                        torch.hub.set_dir(hub)
        finally:
            os.environ.pop(env, None)
            if old is not None:
                os.environ[env] = old

    @unittest.skipIf(torch is None, "needs torch")
    def test_load_dinov3_from_its_backbone_module(self):
        import sys

        env = settings.POINT_DINOV3_WEIGHTS_ENV
        old = os.environ.pop(env, None)
        with tempfile.TemporaryDirectory() as d:
            # a stand-in for the facebookresearch/dinov3 clone: only its
            # backbone module is imported (the hub entry file is never read)
            pkg = os.path.join(d, "repo", "dinov3", "hub")
            os.makedirs(pkg)
            for p in (os.path.dirname(pkg), pkg):
                open(os.path.join(p, "__init__.py"), "w").close()
            with open(os.path.join(d, "repo", "hubconf.py"), "w") as f:
                f.write("raise ImportError('the hub entry file is not used')\n")
            with open(os.path.join(pkg, "backbones.py"), "w") as f:
                f.write(
                    "import torch\n"
                    "class M(torch.nn.Module):\n"
                    "    patch_size = 16\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    "        self.w = torch.nn.Parameter(torch.zeros(3))\n"
                    "    def forward_features(self, x):\n"
                    "        p = torch.nn.functional.avg_pool2d(x, 16)\n"
                    "        t = p.flatten(2).transpose(1, 2) + self.w\n"
                    "        return {'x_norm_patchtokens': t}\n"
                    "def dinov3_vitb16(pretrained=True, **kw):\n"
                    "    assert not pretrained\n"
                    "    return M()\n"
                )
            wfile = os.path.join(d, "dinov3_vitb16_pretrain_lvd1689m-0.pth")
            torch.save({"w": torch.tensor([1.0, 2.0, 3.0])}, wfile)
            os.environ[env] = wfile
            try:
                m = pf.load_backbone("dinov3", device="cpu",
                                     hub_repo=os.path.join(d, "repo"))
                self.assertEqual(m.w.tolist(), [1.0, 2.0, 3.0])  # weights loaded
                grid = pf.forward_feature_grid(m, np.zeros((2, 32, 32, 3), np.uint8))
                self.assertEqual(tuple(grid.shape), (2, 3, 2, 2))
            finally:
                os.environ.pop(env, None)
                if old is not None:
                    os.environ[env] = old
                for k in [k for k in sys.modules if k.split(".")[0] == "dinov3"]:
                    del sys.modules[k]
                sys.path[:] = [p for p in sys.path if not p.startswith(d)]
                pf._BACKBONES.clear()

    def test_window_must_fit_the_patch_size(self):
        with tempfile.TemporaryDirectory() as d:
            ann = os.path.join(d, "a.csv")
            open(ann, "w").close()
            with self.assertRaises(ValueError):
                pf.train_point_classifier(ann, {}, d, os.path.join(d, "m.pt"),
                                          backbone="dinov3", window=448 - 14)


class TestWidthScale(unittest.TestCase):
    def test_photo_scale(self):
        clf = _make_classifier(scale_factor=0.25, meta={"ref_width": 8000})
        self.assertAlmostEqual(pf.photo_scale(clf, None, 2000), 1.0)  # capped at 1
        self.assertAlmostEqual(pf.photo_scale(clf, None, 4000), 0.5)
        self.assertAlmostEqual(pf.photo_scale(clf, None, 8000), 0.25)
        self.assertAlmostEqual(pf.photo_scale(clf, None), 0.25)  # no width
        mm = _make_classifier(scale_factor=0.25, target_mm_px=0.8, meta={"ref_width": 8000})
        self.assertAlmostEqual(pf.photo_scale(mm, 0.4, 4000), 0.5)  # gsd first

    def test_training_scales_of_pool_rows_follow_the_width(self):
        with tempfile.TemporaryDirectory() as d:
            big = _photos(d, n=2, size=(560, 140))
            small = []
            for i in range(2):
                p = os.path.join(d, f"S_{i}.jpg")
                Image.new("RGB", (280, 140)).save(p)
                small.append(p)
            rows = [dict(_row(p, 10, 10, "A"), id=p) for p in big]
            rows += [dict(_row(p, 10, 10, "B"), id=p, domain="cam2") for p in small]
            for r in rows:
                r["_x"], r["_y"] = 10.0, 10.0
            plan = pf.plan_training_scales(rows, d, feature_scale=0.25, window=28)
            self.assertEqual(plan.ref_width, 560)
            np.testing.assert_allclose(plan.scales, [0.25, 0.25, 0.5, 0.5])
            # a pool row with a ground resolution keeps it (mm/px classifiers)
            for r in rows[:2]:
                r["cam_gsd_mm"] = "0.2"
            rows[2]["cam_gsd_mm"] = "0.8"
            plan = pf.plan_training_scales(rows, d, feature_mm_px=0.4, window=28)
            np.testing.assert_allclose(plan.scales[:3], [0.5, 0.5, 2.0])
            self.assertAlmostEqual(plan.scales[3], plan.scale_factor * 560 / 280)


@unittest.skipIf(torch is None or sklearn is None, "needs torch and scikit-learn")
class TestImageOnlyPools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.clf = _make_classifier(
            classes=["B", "R"],
            weights=np.array([[-1, 0, 1], [1, 0, -1]], np.float32) * 3,
            bias=np.zeros(2, np.float32),
            source_mean=np.full(3, 0.45, np.float32),
            source_std=np.full(3, 0.3, np.float32),
            window=28, scale_factor=0.25, meta={"ref_width": 560},
        )
        self.clf.backbone_model = lambda: FakeBackbone()
        self.photos = _photos(os.path.join(self.d), n=4, size=(280, 140))

    def tearDown(self):
        self.tmp.cleanup()

    def test_points_csv_pool_keeps_given_labels(self):
        src = os.path.join(self.d, "predicted_points.csv")
        with open(src, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "cam_filepath", "cam_x", "cam_y", "label", "pred_label"])
            for i, p in enumerate(self.photos):
                w.writerow([f"q{i}a", os.path.basename(p), 40, 70, "Rr", "R"])  # relative
                w.writerow([f"q{i}b", p, 240, 70, "", "B"])
            w.writerow(["gone", os.path.join(self.d, "missing.jpg"), 1, 1, "X", ""])
        pts, info = cur.read_pool_points(src)
        self.assertEqual((info["n_rows"], info["n_used"], info["n_skipped"]), (9, 8, 1))
        meta = cur.mine_pool(self.d, self.clf, sorted({q["path"] for q in pts}), "given",
                             n_clusters=2, progress=False, points=pts, points_info=info)
        rows = cur.read_table(meta["csv"])
        self.assertEqual([r["id"] for r in rows][:2], ["q0a", "q0b"])
        self.assertEqual(sum(1 for r in rows if r["given_label"] == "Rr"), 4)
        self.assertTrue(all(r["label"] == "" for r in rows))  # only reviewed points train
        # 280 px photos, training photos 560 px at 0.25: the window keeps its share
        self.assertTrue(all(abs(float(r["scale"]) - 0.5) < 1e-6 for r in rows))
        self.assertEqual(meta["sampling"]["scale"], "width")
        self.assertEqual(meta["n_given_labels"], 4)
        self.assertEqual({r["pred_label"] for r in rows if float(r["cam_x"]) < 100}, {"R"})

    def test_coralnet_export_pool(self):
        sub = os.path.join(self.d, "imgs")
        os.makedirs(sub)
        for p in self.photos:
            os.rename(p, os.path.join(sub, os.path.basename(p)))
        export = os.path.join(self.d, "annotations.csv")
        with open(export, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Name", "Row", "Column", "Label code"])
            for p in self.photos:
                w.writerow([os.path.basename(p), 70, 40, "Rr"])
                w.writerow([os.path.basename(p), 70, 240, "Bb"])
        pts, info = cur.read_pool_points(export, sub)
        self.assertTrue(info["coralnet"])
        self.assertEqual(len(pts), 8)
        self.assertEqual(sorted({q["label"] for q in pts}), ["Bb", "Rr"])
        self.assertTrue(all(q["path"].startswith(sub) for q in pts))

    def test_sampled_pool_uses_the_width_rule(self):
        meta = cur.mine_pool(self.d, self.clf, self.photos, "plain", n_images=2,
                             per_image=4, n_clusters=2, progress=False)
        rows = cur.read_table(meta["csv"])
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(abs(float(r["scale"]) - 0.5) < 1e-6 for r in rows))
        self.assertEqual(meta["sampling"]["ref_width"], 560)


class TestScalePrecedence(unittest.TestCase):
    def test_width_tolerance_keeps_the_training_camera_at_its_scale(self):
        self.assertEqual(pf.width_scale(0.25, 8000, 7000), 0.25)  # within 25%
        self.assertAlmostEqual(pf.width_scale(0.25, 8000, 2816), 0.25 * 8000 / 2816)

    def test_scales_for_photos(self):
        with tempfile.TemporaryDirectory() as d:
            big = _photos(d, n=1, size=(560, 140))[0]
            small = os.path.join(d, "small.jpg")
            Image.new("RGB", (280, 140)).save(small)
            fixed = _make_classifier(scale_factor=0.25, meta={"ref_width": 560})
            np.testing.assert_allclose(
                fixed.scales_for(None, 2, [big, small]), [0.25, 0.5])
            mm = _make_classifier(scale_factor=0.25, target_mm_px=0.8,
                                  meta={"ref_width": 560})
            # ground resolution first, the width for points without one
            np.testing.assert_allclose(
                mm.scales_for([0.4, None], 2, [small, small]), [0.5, 0.5])
            np.testing.assert_allclose(
                mm.scales_for([0.2, None], 2, [small, big]), [0.25, 0.25])

    @unittest.skipIf(torch is None, "needs torch")
    def test_classify_points_with_a_domain(self):
        with tempfile.TemporaryDirectory() as d:
            photo = _photos(d, n=1, size=(280, 140))[0]
            clf = _make_classifier(
                classes=["B", "R"], weights=np.array([[-1, 0, 1], [1, 0, -1]], np.float32),
                bias=np.zeros(2, np.float32), source_mean=np.full(3, 0.45, np.float32),
                source_std=np.full(3, 0.3, np.float32), window=28, scale_factor=0.5,
                domains={"cam2": (np.array([9.0, 0.0, 0.0], np.float32),
                                  np.ones(3, np.float32))})
            clf.backbone_model = lambda: FakeBackbone()
            p_main, _ = clf.classify_points([photo], [40], [70], progress=False)
            p_dom, _ = clf.classify_points([photo], [40], [70], progress=False, domain="cam2")
            self.assertEqual(clf.classes[int(p_main[0].argmax())], "R")  # red
            self.assertEqual(clf.classes[int(p_dom[0].argmax())], "B")  # shifted stats
            with self.assertRaises(ValueError):
                clf.classify_points([photo], [40], [70], progress=False, domain="nope")
