"""Tests for annotation crops, thumbnails and predictions.

Covers ``ImageMatch.crop_path/find_crop/save_crop``, the crop EXIF metadata,
``Camera`` thumbnails, ``Annotations.crops_root``, the ``pred_*`` CSV columns,
``Annotation.save_crops/predict`` and classification from stored crops.
"""

# Standard Library
import importlib.util
import json
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
import piexif
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


_MODULES = (
    "settings", "logging", "geom", "pathrepair", "visualizations",
    "measurements", "pointclouds", "cameras", "classification", "annotations",
)
_KEYS = ["substrata"] + [f"substrata.{name}" for name in _MODULES]


def _load_isolated():
    """Load cameras, classification and annotations from this checkout.

    The modules are loaded into a private ``substrata`` package (open3d-based
    pointclouds, measurements and visualizations are stubbed; the code under
    test does not use them), then ``sys.modules`` is restored so other test
    modules - and any installed ``substrata`` - are unaffected. ``setUpModule``
    swaps these modules back in while this file's tests run, because some
    methods import siblings lazily.
    """
    saved = {key: sys.modules.get(key) for key in _KEYS}
    try:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
        for key in _KEYS[1:]:
            sys.modules.pop(key, None)

        def install(name, mod):
            sys.modules[f"substrata.{name}"] = mod
            setattr(pkg, name, mod)
            return mod

        def load(name):
            spec = importlib.util.spec_from_file_location(
                f"substrata.{name}", _SUB / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            install(name, mod)
            spec.loader.exec_module(mod)
            return mod

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        load("settings")
        load("geom")
        log_mod = install("logging", types.ModuleType("substrata.logging"))
        log_mod.logger = logging.getLogger("substrata-test")
        log_mod.tqdm_joblib = tqdm_joblib
        for name in ("visualizations", "measurements"):
            install(name, types.ModuleType(f"substrata.{name}"))
        pcd_mod = install("pointclouds", types.ModuleType("substrata.pointclouds"))
        pcd_mod.SimplePointCloud = pcd_mod.PointCloud = object
        mods = (load("cameras"), load("classification"), load("annotations"))
        mine = {key: sys.modules[key] for key in _KEYS if key in sys.modules}
    finally:
        for key, mod in saved.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod
    return mods, mine


(cam_mod, cl, an), _MINE = _load_isolated()
_SAVED = {}


def setUpModule():
    _SAVED.update({key: sys.modules.get(key) for key in _MINE})
    sys.modules.update(_MINE)


def tearDownModule():
    for key, mod in _SAVED.items():
        if mod is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = mod


PHOTO_W, PHOTO_H = 400, 300


def _write_photo(path, with_exif=True):
    """Write a photo whose pixel values encode their (x, y) position."""
    xs = np.arange(PHOTO_W, dtype=np.uint16)
    ys = np.arange(PHOTO_H, dtype=np.uint16)
    arr = np.zeros((PHOTO_H, PHOTO_W, 3), dtype=np.uint8)
    arr[:, :, 0] = (xs[None, :] % 256).astype(np.uint8)
    arr[:, :, 1] = (ys[:, None] % 256).astype(np.uint8)
    arr[:, :, 2] = 128
    img = Image.fromarray(arr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if with_exif:
        exif = piexif.dump({
            "0th": {
                piexif.ImageIFD.Make: b"TestCam",
                piexif.ImageIFD.XResolution: (72, 1),
            },
            "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None,
        })
        img.save(path, format="PNG" if path.endswith(".png") else "JPEG",
                 quality=100, exif=exif)
    else:
        img.save(path, format="JPEG", quality=100)


def _make_match(photo_path, ann_id="ann_0001", x=200.0, y=150.0, relevance=1.0,
                parent=None, obstructed=False):
    cam = cam_mod.Camera(cam_id=1, path=photo_path)
    cam.sensor = types.SimpleNamespace(width=PHOTO_W, height=PHOTO_H)
    ann = an.Annotation([0.0, 0.0, 0.0], id=ann_id, parent=parent)
    im = cam_mod.ImageMatch(cam, x, y, depth=1.0, relevance=relevance,
                            annotation=ann)
    im.potentially_obstructed = obstructed
    return im


class _TmpDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.photo = os.path.join(self.tmp, "model", "site.auv", "IMG_0001.jpg")
        _write_photo(self.photo)
        self.root = os.path.join(self.tmp, "proj_ann.cropped")

    def tearDown(self):
        self._tmp.cleanup()


class TestCropPaths(_TmpDirTest):
    def test_new_layout_uses_photo_folder_name(self):
        im = _make_match(self.photo)
        self.assertEqual(
            im.crop_path(self.root),
            os.path.join(self.root, "ann_0001", "site.auv", "IMG_0001.jpg"),
        )

    def test_windows_style_photo_path(self):
        cam = cam_mod.Camera(cam_id=1, path=r"D:\photos\site.closeup\IMG_9.jpg")
        self.assertEqual(cam_mod._photo_folder_name(cam.filepath), "site.closeup")

    def test_find_crop_falls_back_to_legacy_layout(self):
        im = _make_match(self.photo)
        self.assertIsNone(im.find_crop(self.root))
        legacy = os.path.join(self.root, "ann_0001", "IMG_0001.jpg")
        os.makedirs(os.path.dirname(legacy))
        Image.new("RGB", (10, 10)).save(legacy)
        self.assertEqual(im.find_crop(self.root), legacy)

    def test_root_defaults_to_annotations_crops_root(self):
        anns = an.Annotations()
        anns.filepath = os.path.join(self.tmp, "proj_ann.csv")
        im = _make_match(self.photo, parent=anns)
        self.assertTrue(im.crop_path().startswith(self.root + os.sep))

    def test_missing_root_raises(self):
        im = _make_match(self.photo)
        with self.assertRaises(ValueError):
            im.crop_path()


class TestSaveCrop(_TmpDirTest):
    def test_generated_then_skipped(self):
        im = _make_match(self.photo, obstructed=True)
        self.assertEqual(im.save_crop(self.root, (100, 80)), "generated")
        path = im.find_crop(self.root)
        with Image.open(path) as crop:
            self.assertEqual(crop.size, (100, 80))
        meta = cam_mod.read_crop_metadata(path)
        self.assertTrue(meta["potentially_obstructed"])
        self.assertEqual(meta["v"], 1)
        self.assertEqual(piexif.load(path)["0th"][piexif.ImageIFD.Make], b"TestCam")
        self.assertEqual(im.save_crop(self.root, (100, 80)), "skipped")

    def test_rejected_near_edge(self):
        im = _make_match(self.photo, x=30.0, y=150.0)
        self.assertEqual(im.save_crop(self.root, (100, 100)), "rejected")
        self.assertIsNone(im.find_crop(self.root))

    def test_failed_when_photo_missing(self):
        im = _make_match(os.path.join(self.tmp, "missing", "IMG_1.jpg"))
        self.assertEqual(im.save_crop(self.root, 50), "failed")

    def test_photo_without_exif(self):
        photo = os.path.join(self.tmp, "model", "site.photos", "IMG_0002.jpg")
        _write_photo(photo, with_exif=False)
        im = _make_match(photo)
        self.assertEqual(im.save_crop(self.root, 50), "generated")
        meta = cam_mod.read_crop_metadata(im.find_crop(self.root))
        self.assertEqual(meta["pred_label"], "")

    def test_overwrite_replaces_legacy_crop(self):
        im = _make_match(self.photo)
        legacy = os.path.join(self.root, "ann_0001", "IMG_0001.jpg")
        os.makedirs(os.path.dirname(legacy))
        Image.new("RGB", (10, 10)).save(legacy)
        self.assertEqual(im.save_crop(self.root, 50), "skipped")
        self.assertEqual(im.save_crop(self.root, 50, overwrite=True), "generated")
        self.assertFalse(os.path.exists(legacy))
        self.assertEqual(im.find_crop(self.root), im.crop_path(self.root))


class TestCropMetadata(_TmpDirTest):
    def test_reads_legacy_reefcat_metadata(self):
        path = os.path.join(self.tmp, "legacy.jpg")
        comment = b"ASCII\x00\x00\x00" + json.dumps(
            {"potentially_obstructed": False, "pred_label": "C", "pred_conf": 0.7}
        ).encode("ascii")
        exif = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: comment}})
        Image.new("RGB", (10, 10)).save(path, exif=exif)
        meta = cam_mod.read_crop_metadata(path)
        self.assertEqual(meta["pred_label"], "C")
        self.assertNotIn("v", meta)

    def test_no_metadata(self):
        path = os.path.join(self.tmp, "plain.jpg")
        Image.new("RGB", (10, 10)).save(path)
        self.assertIsNone(cam_mod.read_crop_metadata(path))
        self.assertIsNone(cam_mod.read_crop_metadata(path + ".missing"))

    def test_update_keeps_pixels_and_other_keys(self):
        im = _make_match(self.photo, obstructed=True)
        im.save_crop(self.root, 60)
        path = im.find_crop(self.root)
        with Image.open(path) as before:
            pixels = np.asarray(before).copy()
        cam_mod._update_crop_metadata(path, {"pred_label": "S", "pred_conf": 0.9})
        meta = cam_mod.read_crop_metadata(path)
        self.assertEqual(meta["pred_label"], "S")
        self.assertTrue(meta["potentially_obstructed"])
        with Image.open(path) as after:
            np.testing.assert_array_equal(np.asarray(after), pixels)


class TestThumbnails(_TmpDirTest):
    def test_make_and_find(self):
        cam = cam_mod.Camera(cam_id=1, path=self.photo)
        root = os.path.join(self.tmp, "model", "site.thumbnails")
        self.assertEqual(cam.make_thumbnail(root, (100, 100)), "generated")
        path = cam.find_thumbnail(root)
        self.assertEqual(path, os.path.join(root, "site.auv", "IMG_0001.jpg"))
        with Image.open(path) as thumb:
            self.assertEqual(thumb.size, (100, 75))  # aspect ratio kept
        self.assertEqual(cam.make_thumbnail(root), "skipped")

    def test_legacy_flat_thumbnail_found(self):
        cam = cam_mod.Camera(cam_id=1, path=self.photo)
        root = os.path.join(self.tmp, "thumbs")
        os.makedirs(root)
        Image.new("RGB", (10, 10)).save(os.path.join(root, "IMG_0001.jpg"))
        self.assertEqual(
            cam.find_thumbnail(root), os.path.join(root, "IMG_0001.jpg")
        )

    def test_cameras_make_thumbnails(self):
        cams = cam_mod.Cameras()
        cams.data = {
            1: cam_mod.Camera(parent=cams, cam_id=1, path=self.photo),
            2: cam_mod.Camera(parent=cams, cam_id=2, path=self.photo + ".x"),
        }
        counts = cams.make_thumbnails(os.path.join(self.tmp, "t"), n_jobs=1)
        self.assertEqual(counts, {"generated": 1, "skipped": 0, "failed": 1})


class TestStoredCropPrediction(_TmpDirTest):
    def test_centre_of_stored_crop_equals_photo_crop(self):
        for x, y in [(200.0, 150.0), (201.6, 149.3), (123.5, 98.9)]:
            im = _make_match(self.photo, ann_id=f"a{x}", x=x, y=y)
            im.save_crop(self.root, (120, 90))
            from_crop = cl._crop_from_stored_crop(im, (24, 24), self.root)
            with Image.open(self.photo) as photo:
                left, top = int(x - 12), int(y - 12)
                expected = photo.crop((left, top, left + 24, top + 24))
                expected = np.asarray(expected.convert("RGB"))
            # Lossy JPEG on both sides, so compare position-encoding channels
            diff = np.abs(np.asarray(from_crop, dtype=int) - expected.astype(int))
            self.assertLess(diff[:, :, :2].mean(), 3.0)

    def test_falls_back_when_stored_crop_too_small(self):
        im = _make_match(self.photo)
        im.save_crop(self.root, (20, 20))
        self.assertIsNone(cl._crop_from_stored_crop(im, (24, 24), self.root))
        self.assertIsNone(cl._crop_from_stored_crop(
            _make_match(self.photo, ann_id="none"), 24, self.root
        ))


class _FakeProbs:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class _FakeLearner:
    """Mimics the fastai learner calls used by classify_image_matches_batch."""

    def __init__(self, rows):
        self._rows = rows
        self.dls = types.SimpleNamespace(
            vocab=["C", "S"], test_dl=lambda items, bs=64: list(items)
        )

    def get_preds(self, dl):
        return _FakeProbs(self._rows[: len(dl)]), None


class TestAnnotationCropsAndPredict(_TmpDirTest):
    def _annotations(self):
        anns = an.Annotations()
        anns.filepath = os.path.join(self.tmp, "proj_ann.csv")
        ann = an.Annotation([0.0, 0.0, 0.0], id="ann_0001")
        anns.append(ann)
        photo2 = os.path.join(self.tmp, "model", "site.photos", "IMG_0002.jpg")
        _write_photo(photo2)
        matches = []
        for path, rel in [(self.photo, 1.0), (photo2, 2.0)]:
            cam = cam_mod.Camera(cam_id=path, path=path)
            im = cam_mod.ImageMatch(cam, 200.0, 150.0, 1.0, rel, annotation=ann)
            matches.append(im)
        return anns, ann, matches

    def test_save_crops_and_predict(self):
        anns, ann, matches = self._annotations()
        counts = ann.save_crops(None, crop_size=100, image_matches=matches)
        self.assertEqual(counts["generated"], 2)
        learner = _FakeLearner([[0.9, 0.1], [0.2, 0.8]])
        consensus = ann.predict(
            None, learner, crop_size=24, image_matches=matches, n_jobs=1
        )
        # weights 1 and 0.5: C = (0.9 + 0.1) / 1.5, S = (0.1 + 0.4) / 1.5
        self.assertEqual(consensus["label"], "C")
        self.assertAlmostEqual(consensus["confidence"], 1.0 / 1.5)
        self.assertEqual((ann.pred_label, ann.pred_n), ("C", 2))
        meta = matches[1].crop_metadata()
        self.assertEqual(meta["pred_label"], "S")
        self.assertAlmostEqual(meta["pred_conf"], 0.8)

    def test_remove_crops(self):
        anns, ann, matches = self._annotations()
        ann.save_crops(None, crop_size=50, image_matches=matches)
        ann.remove_crops()
        self.assertFalse(os.path.isdir(ann.crops_dir()))


class TestAnnotationsFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.csv = os.path.join(self._tmp.name, "proj_ann.csv")
        with open(self.csv, "w") as f:
            f.write("id,orig_x,orig_y,orig_z,label,label_conf\n")
            f.write("ann_0001,1,2,3,C,1\n")
            f.write("ann_0002,4,5,6,,\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_crops_root_default_and_explicit(self):
        anns = an.Annotations(self.csv, orig_coords_only=True)
        self.assertEqual(
            anns.crops_root, os.path.join(self._tmp.name, "proj_ann.cropped")
        )
        anns.crops_root = "/elsewhere"
        self.assertEqual(anns.crops_root, "/elsewhere")

    def test_no_pred_columns_until_predicted(self):
        anns = an.Annotations(self.csv, orig_coords_only=True)
        out = os.path.join(self._tmp.name, "out.csv")
        anns.save(out, orig_coords_only=True)
        with open(out) as f:
            self.assertNotIn("pred_label", f.readline())
        # Saving elsewhere does not move the crops of loaded annotations
        self.assertEqual(anns.filepath, self.csv)

    def test_pred_columns_round_trip(self):
        anns = an.Annotations(self.csv, orig_coords_only=True)
        anns["ann_0001"].pred_label = "S"
        anns["ann_0001"].pred_conf = 0.75
        anns["ann_0001"].pred_n = 3
        anns.save(self.csv, orig_coords_only=True)
        reloaded = an.Annotations(self.csv, orig_coords_only=True)
        first, second = reloaded["ann_0001"], reloaded["ann_0002"]
        self.assertEqual((first.pred_label, first.pred_conf, first.pred_n),
                         ("S", 0.75, 3))
        self.assertIsNone(second.pred_label)
        self.assertEqual(first.other_cols, [])


if __name__ == "__main__":
    unittest.main()
