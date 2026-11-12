"""Tests for choosing views and training data.

Covers ``include_cam_in_training`` (counting and collating training rows),
``set_training_views`` (writing training views into an annotation CSV),
``assign_camera_types``, ``Annotation.get_image_matches(occluder=)``,
``Annotation.select_views``, the CoralNet import (``include_column``) and the
stratified-subsample ``Occluder``.
"""

# Standard Library
import csv
import importlib
import os
import shutil
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace

# Third-Party Libraries
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    an = importlib.import_module("substrata.annotations")
    cam_mod = importlib.import_module("substrata.cameras")
    cl = importlib.import_module("substrata.classification")
    st = importlib.import_module("substrata.stratified")
    settings = importlib.import_module("substrata.settings")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    an = None

INC = "include_cam_in_training"


def _write(path, rows, eol="\r\n"):
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator=eol).writerows(rows)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestIncludeInTraining(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)

    def test_values(self):
        for v in (None, "", "True", "true", "1", "yes", "anything"):
            row = {} if v is None else {INC: v}
            self.assertTrue(cl.included_in_training(row), v)
        for v in ("False", "false", "0", "no", " FALSE "):
            self.assertFalse(cl.included_in_training({INC: v}), v)

    def test_count_and_collate_leave_out_excluded_rows(self):
        photo = os.path.join(self.d, "p.photos", "IMG_1.jpg")
        os.makedirs(os.path.dirname(photo))
        open(photo, "w").close()
        path = os.path.join(self.d, "p_slope_intercepts.csv")
        _write(path, [
            ["id", "label", "cam_filepath", "cam_x", "cam_y", INC],
            ["1", "A", photo, "1", "2", "True"],
            ["2", "A", photo, "3", "4", "False"],
            ["3", "B", photo, "5", "6", ""],
        ])
        counts, _unknown = cl.count_labels([path], None)
        self.assertEqual(counts, Counter({"A": 1, "B": 1}))
        out = os.path.join(self.d, "train.csv")
        skipped = Counter()
        n, _drop, _conf = cl.collate_training_annotations(
            [path], "*_slope_intercepts.csv", {"A", "B"}, out, self.d,
            prompt=False, skipped=skipped)
        self.assertEqual(n, 2)
        self.assertEqual(skipped["not_in_training"], 1)
        self.assertEqual({r["id"] for r in _read(out)}, {"p_1", "p_3"})


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestSetTrainingViews(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.path = os.path.join(self.d, "s.csv")
        _write(self.path, [
            ["id", "label", "cam_filepath", "cam_x", "cam_y"],
            ["a", "X", "", "", ""],
            ["b", "Y", "/photos/annotated_on.jpg", "10", "20"],
            ["c", "Z", "", "", ""],
        ])

    def test_fills_empty_views_and_adds_the_column(self):
        n = an.set_training_views(self.path, {
            "a": ("/photos/1.jpg", 100.4, 200.6), "b": ("/photos/2.jpg", 1, 2)})
        self.assertEqual(n, 1)  # b keeps the photo it was annotated on
        rows = {r["id"]: r for r in _read(self.path)}
        self.assertEqual(
            (rows["a"]["cam_filepath"], rows["a"]["cam_x"], rows["a"]["cam_y"]),
            ("/photos/1.jpg", "100.40", "200.60"))
        self.assertEqual(rows["b"]["cam_filepath"], "/photos/annotated_on.jpg")
        self.assertEqual(rows["c"]["cam_filepath"], "")
        self.assertEqual({r[INC] for r in rows.values()}, {"True"})  # every row
        with open(self.path, "rb") as f:
            self.assertIn(b"\r\n", f.read())  # line ending kept

    def test_overwrite_and_reset_to_included(self):
        an.set_training_views(self.path, {})  # adds the column only
        rows = _read(self.path)
        rows[1][INC] = "False"
        _write(self.path, [list(rows[0])] + [list(r.values()) for r in rows])
        n = an.set_training_views(self.path, {"b": ("/photos/2.jpg", 1, 2)},
                                  overwrite=True)
        self.assertEqual(n, 1)
        b = {r["id"]: r for r in _read(self.path)}["b"]
        self.assertEqual((b["cam_filepath"], b[INC]), ("/photos/2.jpg", "True"))

    def test_set_included_in_training(self):
        self.assertEqual(an.set_included_in_training(self.path, ["a", "c"], False), 2)
        rows = {r["id"]: r[INC] for r in _read(self.path)}
        self.assertEqual(rows, {"a": "False", "b": "True", "c": "False"})  # added: True
        an.set_included_in_training(self.path, ["a"], True)
        self.assertEqual(_read(self.path)[0][INC], "True")
        with self.assertRaises(KeyError):
            an.set_included_in_training(self.path, ["zz"], False)

    def test_nothing_to_do_leaves_the_file(self):
        an.set_training_views(self.path, {})

        def data():
            with open(self.path, "rb") as f:
                return f.read()

        before = data()
        self.assertEqual(an.set_training_views(self.path, {"zz": ("/x.jpg", 1, 1)}), 0)
        self.assertEqual(data(), before)


def _cam(w, h, fx, name="c"):
    return SimpleNamespace(sensor=SimpleNamespace(width=w, height=h, fx=fx),
                           filepath=f"/p/{name}.jpg", filename=f"{name}.jpg")


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestCameraTypes(unittest.TestCase):
    def test_types_by_sensor(self):
        cams = [
            _cam(9568, 6376, 6460), _cam(9568, 6376, 6470),  # two bodies
            _cam(6376, 9568, 6500),  # portrait, focal within 3%
            _cam(3840, 2160, 1754), _cam(8726, 5840, 23610),
            _cam(9568, 6376, 7000),  # same resolution, other lens
            SimpleNamespace(sensor=None, filepath="/p/x.jpg"),
        ]
        types = cam_mod.assign_camera_types(cams)
        self.assertEqual(types[0], "9568x6376 f6460")  # most cameras first
        self.assertEqual(len(types), 4)
        self.assertEqual({c.camera_type for c in cams[:3]}, {"9568x6376 f6460"})
        self.assertEqual(cams[5].camera_type, "9568x6376 f7000")
        self.assertIsNone(cams[6].camera_type)


class _FakeCam:
    """In view at a fixed pixel with a fixed relevance; its centre sets the
    blocker count of the fake occluder below."""

    def __init__(self, name, relevance, blockers, camera_type="main"):
        self.filename = f"{name}.jpg"
        self.filepath = f"/p/{name}.jpg"
        self.enabled = True
        self.camera_type = camera_type
        self._rel = relevance
        self.orig_camera_transform = np.eye(4)
        self.orig_camera_transform[0, 3] = blockers

    def get_pixel_coords(self, coords, use_orig_coords=True):
        return 10.0, 20.0, 1.0, self._rel


class _FakeOccluder:
    def normal_at(self, p):
        return None

    def line_of_sight_margin(self):
        return 0.01

    def blockers(self, eye, p, margin, normal=None):
        return int(eye[0])  # the camera's x position


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestOccluderMatches(unittest.TestCase):
    def test_flags_and_order(self):
        cams = [_FakeCam("a", 0.1, 9), _FakeCam("b", 0.2, 0), _FakeCam("c", 0.3, 2),
                _FakeCam("d", 0.4, 3), _FakeCam("e", 0.5, 50)]
        ann = an.Annotation([0.0, 0.0, 0.0])
        ms = ann.get_image_matches(cams, occluder=_FakeOccluder(), blocked_min=3,
                                   occlusion_check=4)
        self.assertEqual([m.cam.filename[0] for m in ms], ["b", "c", "e", "a", "d"])
        by = {m.cam.filename[0]: m for m in ms}
        self.assertEqual([by[k].blockers for k in "abcd"], [9, 0, 2, 3])
        self.assertIsNone(by["e"].blockers)  # beyond occlusion_check: untested
        self.assertIsNone(by["e"].potentially_obstructed)
        self.assertTrue(by["a"].potentially_obstructed)
        self.assertTrue(by["d"].potentially_obstructed)

    def test_without_occluder_unchanged(self):
        cams = [_FakeCam("a", 0.3, 9), _FakeCam("b", 0.1, 0)]
        ms = an.Annotation([0.0, 0.0, 0.0]).get_image_matches(cams)
        self.assertEqual([m.cam.filename[0] for m in ms], ["b", "a"])
        self.assertFalse(hasattr(ms[0], "blockers"))


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestSelectViews(unittest.TestCase):
    """Through the real get_image_matches (fake cameras and occluder)."""

    def setUp(self):
        self.ann = an.Annotation([0.0, 0.0, 0.0])

    def names(self, views):
        return [v.cam.filename[:-4] for v in views]

    def test_main_type_first_then_others(self):
        cams = [_FakeCam("m1", .1, 0), _FakeCam("c1", .15, 0, "close"),
                _FakeCam("m2", .2, 0), _FakeCam("g1", .25, 0, "gopro"),
                _FakeCam("c2", .3, 0, "close"), _FakeCam("m3", .4, 0),
                _FakeCam("mb", .05, 9)]  # most relevant, but blocked
        views, training = self.ann.select_views(
            cams, occluder=_FakeOccluder(), main_type="main", max_views=3,
            per_other_type=1)
        self.assertEqual(self.names(views), ["m1", "m2", "m3", "c1", "g1"])
        self.assertEqual(training.cam.filename, "m1.jpg")
        self.assertIs(self.ann.training_view, training)

    def test_blocked_views_fill_up_and_no_training_view(self):
        cams = [_FakeCam("m1", .1, 9), _FakeCam("c1", .2, 0, "close"),
                _FakeCam("m2", .3, 9)]
        views, training = self.ann.select_views(
            cams, occluder=_FakeOccluder(), main_type="main", max_views=10)
        self.assertEqual(self.names(views), ["m1", "m2", "c1"])
        self.assertIsNone(training)  # every main view blocked: needs a look

    def test_untested_takes_the_most_relevant_main_view(self):
        cams = [_FakeCam("c1", .05, 0, "close"), _FakeCam("m1", .1, 9),
                _FakeCam("m2", .2, 0)]
        views, training = self.ann.select_views(cams, main_type="main")
        self.assertEqual(training.cam.filename, "m1.jpg")  # no test: relevance
        self.assertEqual(self.names(views), ["m1", "m2", "c1"])

    def test_tests_only_until_enough(self):
        calls = []

        class Counting(_FakeOccluder):
            def blockers(self, eye, p, margin, normal=None):
                calls.append(int(eye[0]))
                return super().blockers(eye, p, margin, normal)

        cams = [_FakeCam("b1", .1, 9), _FakeCam("m1", .2, 0), _FakeCam("c1", .25, 0, "close"),
                _FakeCam("m2", .3, 0), _FakeCam("m3", .4, 0), _FakeCam("m4", .5, 0)]
        views, training = self.ann.select_views(cams, occluder=Counting(), main_type="main",
                                                max_views=2, per_other_type=0)
        self.assertEqual(self.names(views), ["m1", "m2"])  # b1 blocked: not among the 2
        self.assertEqual(training.cam.filename, "m1.jpg")
        self.assertEqual(len(calls), 3)  # b1, m1, m2: then enough (c1: no other types wanted)

    def test_edge_view_only_as_training_fallback(self):
        class Edge(_FakeCam):  # its crop would run past the photo edge
            def __init__(self, *a, edge=False, **k):
                super().__init__(*a, **k)
                self.edge = edge

        cams = [Edge("e1", .1, 0, edge=True), Edge("m1", .2, 9), Edge("m2", .3, 9)]
        fits = lambda m: not m.cam.edge  # noqa: E731
        for c in cams:
            c.sensor = SimpleNamespace(width=100, height=100)
        an.cameras.ImageMatch.is_clear_from_image_edge, orig = (
            lambda self, size, image_size=None: fits(self), an.cameras.ImageMatch.is_clear_from_image_edge)
        try:
            views, training = self.ann.select_views(cams, occluder=_FakeOccluder(), main_type="main",
                                                    max_views=5, clear_of_edge=(800, 800))
        finally:
            an.cameras.ImageMatch.is_clear_from_image_edge = orig
        self.assertEqual(self.names(views), ["m1", "m2"])  # cropped: clear of the edge (here obstructed)
        self.assertEqual(training.cam.filename, "e1.jpg")  # the only unobstructed view: at the edge

    def test_all_cameras_alike(self):
        cams = [_FakeCam("c1", .05, 0, "close"), _FakeCam("m1", .1, 0)]
        views, training = self.ann.select_views(cams, occluder=_FakeOccluder())
        self.assertEqual((self.names(views), training.cam.filename),
                         (["c1", "m1"], "c1.jpg"))


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestViewCropNames(unittest.TestCase):
    def test_names_and_lookup(self):
        fp = "/data/p/site.photos/IMG_7.JPG"
        self.assertEqual(cam_mod.view_crop_name(fp, 10.4, 20.6), "IMG_7_10_21.jpg")
        self.assertEqual(cam_mod.view_crop_name(r"D:\x\s.auv\G1.jpg", 1, 2), "G1_1_2.jpg")
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        self.assertIsNone(cam_mod.find_view_crop(root, "a1", fp, 10, 21))
        legacy = os.path.join(root, "a1", "IMG_7.JPG")  # oldest: no photo folder, no pixel
        older = os.path.join(root, "a1", "site.photos", "IMG_7.JPG")
        new = os.path.join(root, "a1", "site.photos", "IMG_7_10_21.jpg")
        for path in (legacy, older, new):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
            self.assertEqual(cam_mod.find_view_crop(root, "a1", fp, 10.2, 20.8), path)
        self.assertEqual(cam_mod.find_view_crop(root, "a1", fp, 99, 99), older)  # moved


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestStaleCrops(unittest.TestCase):
    def test_stale_and_training_views(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        ann = an.Annotation([0.0, 0.0, 0.0])
        ann.id = "a1"
        cam = SimpleNamespace(filepath="/p/site.photos/IMG_1.jpg",
                              get_pixel_coords=lambda c, use_orig_coords=True: (100.0, 50.0, 1.0, 0.1))
        d = os.path.join(root, "a1", "site.photos")
        os.makedirs(d)
        names = ["IMG_1_100_50.jpg", "IMG_1_300_50.jpg", "IMG_2_1_1.jpg", "IMG_1.jpg"]
        for n in names:
            open(os.path.join(d, n), "w").close()
        stale = ann.stale_crops([cam], root)
        # moved in IMG_1; IMG_2 is not a known camera, IMG_1.jpg has no pixel: both kept
        self.assertEqual([os.path.basename(p) for p in stale], ["IMG_1_300_50.jpg"])
        ann.training_view = SimpleNamespace(cam=cam, x=100.0, y=50.0)
        other = an.Annotation([1.0, 1.0, 1.0])
        other.id, other.training_view = "a2", None
        untouched = an.Annotation([2.0, 2.0, 2.0])
        self.assertEqual(an.training_views([ann, other, untouched]),
                         ({"a1": ("/p/site.photos/IMG_1.jpg", 100.0, 50.0)}, ["a2"]))


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestPredictionModel(unittest.TestCase):
    def test_pred_agree(self):
        ann = an.Annotation([0.0, 0.0, 0.0])
        res = [{"label": "A", "confidence": .9}, {"label": "B", "confidence": .6},
               {"label": "A", "confidence": .8}, None]
        cl.aggregate_predictions = cl.aggregate_predictions  # (the real one)
        ann._apply_predictions([SimpleNamespace(relevance=1.0, potentially_obstructed=False)] * 4, res, None,
                               False, "first", "m@1")
        self.assertEqual((ann.pred_label, ann.pred_agree), ("A", 0.67))  # 2 of the 3 views predicted

    def test_fingerprint_and_pred_model_column(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        clf = os.path.join(d, "point_classifier.pt")
        with open(clf, "wb") as f:
            f.write(b"model v1")
        fp = cl.classifier_fingerprint(clf)
        self.assertTrue(fp.startswith("point_classifier@") and len(fp) == len("point_classifier@") + 8)
        copy = os.path.join(d, "copy", "point_classifier.pt")
        os.makedirs(os.path.dirname(copy))
        shutil.copy(clf, copy)
        self.assertEqual(cl.classifier_fingerprint(copy), fp)  # moved / copied: the same
        with open(clf, "wb") as f:
            f.write(b"model v2, retrained")
        self.assertNotEqual(cl.classifier_fingerprint(clf), fp)
        # the CSV keeps pred_model next to the other prediction columns
        csv_path = os.path.join(d, "p_ann.csv")
        _write(csv_path, [["id", "orig_x", "orig_y", "orig_z", "label"], ["a", "1", "2", "3", "SP"]], "\n")
        anns = an.Annotations(csv_path)
        a = anns.data["a"]
        a.pred_label, a.pred_conf, a.pred_n, a.pred_model = "SP", 0.9, 3, fp
        anns.save(csv_path)
        row = _read(csv_path)[0]
        self.assertEqual((row["pred_label"], row["pred_model"]), ("SP", fp))
        self.assertEqual(an.Annotations(csv_path).data["a"].pred_model, fp)


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestSetPredictions(unittest.TestCase):
    def test_other_columns_kept(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = os.path.join(d, "p_slope_intercepts.csv")
        _write(path, [["id", "label", "search_radius", "manual_id", INC],
                      ["a", "SP", "0.005", "m_1", "False"], ["b", "", "0.005", "", "True"]])
        top = an.format_pred_top({"SP": 0.912345, "CSE": 0.05, "MA": 0.03})
        n = an.set_predictions(path, {"a": ("SP", 0.912345, 3, "clf@1234abcd", 0.667, top), "b": None})
        self.assertEqual(n, 2)
        rows = _read(path)
        self.assertEqual(list(rows[0]), ["id", "label", "search_radius", "manual_id", INC,
                                         "pred_label", "pred_conf", "pred_n", "pred_model", "pred_agree",
                                         "pred_top"])
        self.assertEqual([rows[0][c] for c in ("search_radius", "manual_id", INC, "pred_conf", "pred_model",
                                                "pred_agree")],
                         ["0.005", "m_1", "False", "0.9123", "clf@1234abcd", "0.67"])
        self.assertEqual(rows[0]["pred_top"], "SP:0.912;CSE:0.050;MA:0.030")
        self.assertEqual(an.parse_pred_top(rows[0]["pred_top"]), {"SP": 0.912, "CSE": 0.05, "MA": 0.03})
        self.assertEqual(rows[1]["pred_label"], "")
        with open(path, "rb") as f:
            self.assertIn(b"\r\n", f.read())


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestCoralnetImport(unittest.TestCase):
    def test_include_column(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        os.makedirs(os.path.join(d, "imgs"))
        open(os.path.join(d, "imgs", "f1.png"), "w").close()
        export = os.path.join(d, "export.csv")
        _write(export, [["Name", "Row", "Column", "Label code"],
                        ["f1.png", "5", "7", "SP"], ["gone.png", "1", "1", "SP"]])
        out = os.path.join(d, "export_points.csv")
        info = cl.convert_coralnet_csv(export, cl.index_image_dir(d), out,
                                       include_column=True)
        self.assertEqual((info["n_written"], info["n_missing"]), (1, 1))
        row = _read(out)[0]
        self.assertEqual((row["label"], row["cam_x"], row["cam_y"], row[INC]),
                         ("SP", "7", "5", "True"))


def _write_planes(path, n=400_000, seed=2):
    """Two horizontal 10 x 10 planes, at z = 0 and z = 1 (dense enough per
    cluster for the stratified index's spacing estimate)."""
    rng = np.random.default_rng(seed)
    rec = np.zeros(n, [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    rec["x"], rec["y"] = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
    rec["z"] = (np.arange(n) % 2).astype(np.float32)
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex "
            f"{n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(rec.tobytes())


@unittest.skipUnless(an is not None, "requires the substrata environment")
class TestOccluder(unittest.TestCase):
    def test_line_of_sight(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        ply = os.path.join(d, "plot.ply")
        _write_planes(ply)
        self.assertIsNone(st.ply_occluder(ply))  # not stratified yet
        st.stratify_ply(ply, cluster_size=200_000, progress=False)
        oc = st.ply_occluder(ply)
        p = np.array([5.0, 5.0, 0.0])  # on the lower plane
        margin = oc.line_of_sight_margin()
        self.assertEqual(oc.blockers(p + [0, 0, -5], p, margin), 0)  # from below
        self.assertGreaterEqual(  # from above: through the upper plane
            oc.blockers(p + [0.5, 0, 5], p, margin), settings.OCCLUSION_BLOCKED_MIN)


if __name__ == "__main__":
    unittest.main()
