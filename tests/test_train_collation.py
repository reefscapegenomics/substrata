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

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_classification():
    """Load ``classification.py`` without importing the ``substrata`` package.

    fastai/torch are imported lazily inside the training functions, so only
    ``settings`` and a stub ``logging`` module are needed here.
    """
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    if "substrata.settings" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "substrata.settings", _SUB / "settings.py"
        )
        sm = importlib.util.module_from_spec(spec)
        sys.modules["substrata.settings"] = sm
        spec.loader.exec_module(sm)
    sys.modules["substrata"].settings = sys.modules["substrata.settings"]

    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")
        log_mod.logger = logging.getLogger("substrata-test")

        # Provide tqdm_joblib too so this stub is a superset of what other
        # test modules' loaders expect (cameras.py imports it); otherwise,
        # whichever module is imported first during discovery installs an
        # incomplete substrata.logging and breaks the others.
        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        log_mod.tqdm_joblib = tqdm_joblib
        sys.modules["substrata.logging"] = log_mod

    spec = importlib.util.spec_from_file_location(
        "substrata.classification", _SUB / "classification.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["substrata.classification"] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load_classification()


# A small synthetic CATAMI tree:
#   Corals [C]
#     ├── Branching [CB]            (leaf -> tip, bold)
#     └── Massive [CM]              (parent with children)
#           ├── Massive sub a [CMA] (leaf -> tip, bold)
#           └── Massive sub b [CMB] (leaf -> tip, bold)
#   Algae [A]                       (root leaf -> NOT bold, bare root)
_CLASSES_ROWS = [
    # SPECIES_CODE, CATAMI_PARENT_ID, CPC_CODES, L1, L2, L3
    ("1000", "", "C", "Corals", "", ""),
    ("1100", "1000", "CB", "Corals", "Branching", ""),
    ("1200", "1000", "CM", "Corals", "Massive", ""),
    ("1210", "1200", "CMA", "Corals", "Massive", "Sub a"),
    ("1220", "1200", "CMB", "Corals", "Massive", "Sub b"),
    ("2000", "", "A", "Algae", "", ""),
]


def _write_classes(path):
    header = [
        "SPECIES_CODE", "CATAMI_PARENT_ID", "CPC_CODES",
        "CATAMI_LEVEL_1", "CATAMI_LEVEL_2", "CATAMI_LEVEL_3",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in _CLASSES_ROWS:
            w.writerow(row)


_ANN_HEADER = [
    "id", "orig_x", "orig_y", "orig_z", "label", "label_conf",
    "world_x", "world_y", "world_z", "cam_filepath", "cam_x", "cam_y", "depth",
]


def _write_ann(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_ANN_HEADER)
        for r in rows:
            w.writerow(r)


def _ann_row(ann_id, label, cam_filepath="", cam_x="", cam_y=""):
    return [
        ann_id, "0", "0", "0", label, "",
        "1", "2", "3", cam_filepath, cam_x, cam_y, "",
    ]


class TestLabelTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.classes = os.path.join(self.d, "classes.csv")
        _write_classes(self.classes)

    def tearDown(self):
        self.tmp.cleanup()

    def _ann_file(self, name, labels):
        path = os.path.join(self.d, name)
        _write_ann(path, [_ann_row(f"x{i}", lab) for i, lab in enumerate(labels)])
        return path

    def test_training_labels_are_tips(self):
        # CB, CMA, CMB are tips; CM is a parent (self count 0 -> not heavy);
        # A is a bare root leaf -> excluded.
        f = self._ann_file("m_ann.csv", ["CB", "CMA", "CMB", "A"])
        _lines, labels, _c, _u = cl.build_label_tree(self.classes, [f])
        self.assertEqual(labels, {"CB", "CMA", "CMB"})

    def test_heavy_parent_bolded_when_not_tips_only(self):
        # Give CM many direct hits so it becomes a heavy parent (bold),
        # plus children so it still has visible kids.
        labels = ["CM"] * 5 + ["CMA", "CMB"]
        f = self._ann_file("m_ann.csv", labels)
        _l, got, _c, _u = cl.build_label_tree(
            self.classes, [f], min_count=1, tips_only=False
        )
        self.assertIn("CM", got)
        # With tips_only, the heavy parent is NOT bolded.
        _l, got2, _c, _u = cl.build_label_tree(
            self.classes, [f], min_count=1, tips_only=True
        )
        self.assertNotIn("CM", got2)

    def test_min_count_gates_bolding_not_visibility(self):
        # One CB (count 1) and many CMA (count 5); min_count=2.
        # CB is below threshold so it is NOT a training label, but it must
        # still appear in the rendered tree (just unbolded).
        labels = ["CB"] + ["CMA"] * 5
        f = self._ann_file("m_ann.csv", labels)
        lines, got, _c, _u = cl.build_label_tree(self.classes, [f], min_count=2)
        self.assertIn("CMA", got)
        self.assertNotIn("CB", got)
        # CB row is still shown in the tree...
        cb_lines = [ln for ln in lines if "[CB]" in ln]
        self.assertEqual(len(cb_lines), 1)
        # ...and it is not bolded (no ANSI bold styling around it).
        self.assertNotIn(cl.settings.TRAIN_BOLD, cb_lines[0])
        # CMA, being above threshold, is bolded.
        cma_lines = [ln for ln in lines if "[CMA]" in ln]
        self.assertIn(cl.settings.TRAIN_BOLD, cma_lines[0])

    def test_unknown_labels_collected(self):
        f = self._ann_file("m_ann.csv", ["CB", "ZZZ", "ZZZ"])
        _l, _labels, counts, unknown = cl.build_label_tree(self.classes, [f])
        self.assertEqual(unknown.get("ZZZ"), 2)
        self.assertEqual(counts.get("CB"), 1)

    def test_include_labels_override_min_count(self):
        # CM has 5 direct hits (would be a heavy parent by min_count), and CMA
        # is a tip. include_labels should bold exactly {CB, CMA}, ignoring the
        # count rules, and leave CM unbolded.
        labels = ["CM"] * 5 + ["CMA", "CB"]
        f = self._ann_file("m_ann.csv", labels)
        lines, got, _c, _u = cl.build_label_tree(
            self.classes, [f], min_count=1, include_labels={"CB", "CMA"}
        )
        self.assertEqual(got, {"CB", "CMA"})
        # CM appears but is not bolded.
        cm_lines = [ln for ln in lines if "[CM]" in ln and "[CMA]" not in ln]
        self.assertEqual(len(cm_lines), 1)
        self.assertNotIn(cl.settings.TRAIN_BOLD, cm_lines[0])
        # CB is bolded.
        cb_lines = [ln for ln in lines if "[CB]" in ln]
        self.assertIn(cl.settings.TRAIN_BOLD, cb_lines[0])

    def test_include_labels_missing_detectable(self):
        # A requested label absent from the tree is not returned, so the caller
        # (CLI) can detect the difference and error out.
        f = self._ann_file("m_ann.csv", ["CB", "CMA"])
        _l, got, _c, _u = cl.build_label_tree(
            self.classes, [f], include_labels={"CB", "NOPE"}
        )
        self.assertEqual(got, {"CB"})
        self.assertEqual({"CB", "NOPE"} - got, {"NOPE"})


class TestModelName(unittest.TestCase):
    def test_strip_pattern_suffix(self):
        name = cl.model_name_from_filename(
            "ton_ko1_05m_20241005_slope_intercepts.csv", "*_slope_intercepts.csv"
        )
        self.assertEqual(name, "ton_ko1_05m_20241005")

    def test_path_basename(self):
        name = cl.model_name_from_filename(
            "/a/b/ton_ko2_60m_20241001_slope_intercepts.csv",
            "*_slope_intercepts.csv",
        )
        self.assertEqual(name, "ton_ko2_60m_20241001")

    def test_no_wildcard(self):
        name = cl.model_name_from_filename("foo.csv", "foo.csv")
        self.assertEqual(name, "foo")


class TestConventionDir(unittest.TestCase):
    def test_nested_convention(self):
        got = cl.convention_dir_for_model(
            "/base", "ton_ko1_05m_20241005", "ton_ko1_05m_20241005.photos"
        )
        self.assertEqual(
            got,
            os.path.join(
                "/base", "ton_ko1", "ton_ko1_05m", "ton_ko1_05m_20241005",
                "ton_ko1_05m_20241005.photos",
            ),
        )


class TestResolveCamDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_dir_kept(self):
        real = os.path.join(self.d, "real.photos")
        os.makedirs(real)
        mapping = cl.resolve_cam_dirs({real}, self.d, "m", prompt=False)
        self.assertEqual(mapping[real], real)

    def test_missing_dir_remapped_to_convention(self):
        model = "ton_ko1_05m_20241005"
        last = f"{model}.photos"
        conv = cl.convention_dir_for_model(self.d, model, last)
        os.makedirs(conv)
        missing = f"/mnt/sdd/whatever/{last}"
        mapping = cl.resolve_cam_dirs({missing}, self.d, model, prompt=False)
        self.assertEqual(mapping[missing], conv)
        # Final folder preserved.
        self.assertEqual(os.path.basename(mapping[missing]), last)

    def test_unresolvable_raises_without_prompt(self):
        with self.assertRaises(FileNotFoundError):
            cl.resolve_cam_dirs(
                {"/nope/x.photos"}, self.d, "ton_ko1_05m_20241005", prompt=False
            )


class TestCollate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        # A real image directory so paths resolve without prompting.
        self.model = "ton_ko1_05m_20241005"
        self.last = f"{self.model}.photos"
        self.imgdir = cl.convention_dir_for_model(self.d, self.model, self.last)
        os.makedirs(self.imgdir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_filter_prefix_and_remap(self):
        missing_dir = f"/mnt/sdd/x/{self.last}"
        csv_path = os.path.join(
            self.d, f"{self.model}_slope_intercepts.csv"
        )
        rows = [
            # integer-only id, training label, has cam fields -> kept+prefixed
            _ann_row("0", "CB", f"{missing_dir}/IMG_1.jpg", "100", "200"),
            # non-training label -> dropped
            _ann_row("1", "A", f"{missing_dir}/IMG_2.jpg", "10", "20"),
            # training label but no cam fields -> dropped (missing cam)
            _ann_row("2", "CMA", "", "", ""),
            # already-prefixed id -> kept as-is
            _ann_row("ton_ko1_05m_20241005_9", "CMB",
                     f"{missing_dir}/IMG_3.jpg", "5", "6"),
        ]
        _write_ann(csv_path, rows)
        out = os.path.join(self.d, "training_annotations.csv")
        n_written, n_dropped = cl.collate_training_annotations(
            [csv_path], "*_slope_intercepts.csv",
            {"CB", "CMA", "CMB"}, out, self.d, prompt=False,
        )
        self.assertEqual(n_written, 2)
        self.assertEqual(n_dropped, 1)  # CMA row had no cam fields
        with open(out, newline="") as f:
            written = list(csv.DictReader(f))
        ids = {r["id"] for r in written}
        self.assertIn("ton_ko1_05m_20241005_0", ids)  # int id prefixed
        self.assertIn("ton_ko1_05m_20241005_9", ids)  # already prefixed kept
        # cam_filepath remapped from /mnt/sdd/... to the convention dir.
        for r in written:
            self.assertTrue(r["cam_filepath"].startswith(self.imgdir))


class TestSplitAndCrops(unittest.TestCase):
    def test_split_deterministic_and_balanced(self):
        ids = [f"ann_{i}" for i in range(3000)]
        first = [cl.split_for_id(i, (80, 10, 10)) for i in ids]
        second = [cl.split_for_id(i, (80, 10, 10)) for i in ids]
        self.assertEqual(first, second)  # deterministic
        counts = {s: first.count(s) for s in (0, 1, 2)}
        # Roughly 80/10/10 over 3000 ids.
        self.assertGreater(counts[0], 2200)
        self.assertLess(counts[0], 2600)
        self.assertGreater(counts[1], 200)
        self.assertGreater(counts[2], 200)

    def test_crop_filename_encoding(self):
        fn = cl.crop_filename(
            "ton_ko1_05m_20241005_0",
            "/x/ton_ko1_05m/PGRAM01_DSC02972.jpg", 1234.6, 567.2,
        )
        self.assertEqual(
            fn, "ton_ko1_05m_20241005_0_PGRAM01_DSC02972_1235_567.jpg"
        )

    def test_plan_and_stale_detection(self):
        tmp = tempfile.TemporaryDirectory()
        d = tmp.name
        try:
            ann = os.path.join(d, "training_annotations.csv")
            rows = [
                _ann_row("a0", "CB", "/imgs/IMG_1.jpg", "100", "200"),
                _ann_row("a1", "CMA", "/imgs/IMG_2.jpg", "50", "60"),
            ]
            _write_ann(ann, rows)
            expected = cl.plan_crops(ann, d)
            self.assertEqual(len(expected), 2)
            # Each expected path sits under one of the crop split folders.
            for p in expected:
                rel = os.path.relpath(p, d)
                self.assertIn(rel.split(os.sep)[0], cl.settings.TRAIN_CROP_DIRS)
            # Simulate an existing crop that no longer matches -> stale.
            existing = set(expected) | {
                os.path.join(d, "training_crops", "CB", "old_stale_1_2.jpg")
            }
            stale = existing - set(expected)
            self.assertEqual(len(stale), 1)
        finally:
            tmp.cleanup()


class TestImageSafeguards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        from PIL import Image
        self._Image = Image

    def tearDown(self):
        self.tmp.cleanup()

    def _valid_jpg(self, name):
        p = os.path.join(self.d, name)
        self._Image.new("RGB", (32, 32), (10, 20, 30)).save(p, "JPEG")
        return p

    def _empty_file(self, name):
        p = os.path.join(self.d, name)
        open(p, "wb").close()  # zero bytes
        return p

    def test_is_unreadable_image(self):
        good = self._valid_jpg("ok.jpg")
        empty = self._empty_file("empty.jpg")
        garbage = os.path.join(self.d, "garbage.jpg")
        with open(garbage, "w") as f:
            f.write("not an image")
        self.assertFalse(cl._is_unreadable_image(good))
        self.assertTrue(cl._is_unreadable_image(empty))
        self.assertTrue(cl._is_unreadable_image(garbage))
        self.assertTrue(cl._is_unreadable_image(os.path.join(self.d, "missing.jpg")))

    def test_filter_readable_images(self):
        good = self._valid_jpg("a.jpg")
        empty = self._empty_file("b.jpg")
        readable, bad = cl.filter_readable_images([good, empty])
        self.assertEqual(readable, [good])
        self.assertEqual(bad, [empty])

    def test_existing_crops_skips_zero_byte(self):
        train = os.path.join(self.d, cl.settings.TRAIN_CROP_DIRS[0], "CB")
        os.makedirs(train)
        good = os.path.join(train, "good_1_2.jpg")
        self._Image.new("RGB", (16, 16), (0, 0, 0)).save(good, "JPEG")
        empty = os.path.join(train, "empty_3_4.jpg")
        open(empty, "wb").close()
        found = cl.existing_crops(self.d)
        self.assertIn(good, found)
        self.assertNotIn(empty, found)  # zero-byte treated as not present


if __name__ == "__main__":
    unittest.main()
