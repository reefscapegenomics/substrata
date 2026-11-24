"""Review fixes in annotations.py: CSV round trips, duplicate ids, subsets.

The annotation CSVs are users' data: a load-save round trip must keep every row
and cell (quoted cells with commas, a BOM, columns the class does not model).
Modules are loaded from this checkout into a private ``substrata`` package
(open3d-based modules stubbed), as in ``test_crops.py``.
"""

# Standard Library
import importlib.util
import logging
import os
import stat
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

# Third-Party Libraries
import numpy as np

_SUB = Path(__file__).resolve().parents[1] / "src" / "substrata"
_MODULES = (
    "settings", "logging", "geom", "fileio", "frames", "pathrepair", "sidecars", "imagecrops",
    "visualizations", "measurements", "pointclouds", "cameras", "annotations",
)
_KEYS = ["substrata"] + [f"substrata.{name}" for name in _MODULES]


def _load_isolated():
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
                f"substrata.{name}", _SUB / f"{name}.py")
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
        mods = (load("cameras"), load("annotations"), load("fileio"))
        mine = {key: sys.modules[key] for key in _KEYS if key in sys.modules}
    finally:
        for key, mod in saved.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod
    return mods, mine


(cam_mod, an, fio), _MINE = _load_isolated()
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


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.csv = os.path.join(self.tmp, "anns.csv")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text, bom=False, path=None):
        with open(path or self.csv, "wb") as fh:
            fh.write((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))

    def rows(self, path=None):
        return an._read_csv_table(path or self.csv)


class TestCsvRoundTrip(_Tmp):
    """A1-A3: one csv-module reader; save writes back unmodelled columns."""

    def test_comma_in_label_and_path(self):
        anns = an.Annotations(coords={"p1": [1.0, 2.0, 3.0]})
        anns["p1"].label = "Porites, massive"
        anns["p1"].cam_filepath = "/data/a,b/IMG_1.jpg"
        anns.save(self.csv)
        back = an.Annotations(self.csv)
        self.assertEqual(back["p1"].label, "Porites, massive")
        self.assertEqual(back["p1"].cam_filepath, "/data/a,b/IMG_1.jpg")
        np.testing.assert_allclose(back["p1"].orig_coords, [1, 2, 3])

    def test_fully_quoted_csv(self):
        self.write('"id","x","y","z","label"\r\n"p1","1.5","2","3","ACRO"\r\n')
        back = an.Annotations(self.csv)
        self.assertEqual(back["p1"].label, "ACRO")
        np.testing.assert_allclose(back["p1"].orig_coords, [1.5, 2, 3])

    def test_bom_header(self):
        self.write("id,x,y,z,label\np1,1,2,3,ACRO\n", bom=True)
        back = an.Annotations(self.csv)
        self.assertEqual(list(back.data), ["p1"])

    def test_extra_columns_written_back(self):
        self.write(
            "id,x,y,z,label,include_cam_in_training,manual_id,notes\n"
            'p1,1,2,3,ACRO,False,m_7,"a, b"\n'
            "p2,4,5,6,PORI,,,\n"
        )
        anns = an.Annotations(self.csv, orig_coords_only=True)
        self.assertEqual(
            anns["p1"].other_cols,
            {"include_cam_in_training": "False", "manual_id": "m_7", "notes": "a, b"},
        )
        anns["p2"].label = "SAND"
        anns.save(self.csv, orig_coords_only=True)
        rows = self.rows()
        head = rows[0]
        for col in ("include_cam_in_training", "manual_id", "notes"):
            self.assertIn(col, head)
        p1 = dict(zip(head, rows[1]))
        self.assertEqual(
            (p1["include_cam_in_training"], p1["manual_id"], p1["notes"]),
            ("False", "m_7", "a, b"),
        )
        self.assertEqual(dict(zip(head, rows[2]))["label"], "SAND")

    def test_meta_data_overrides_other_column(self):
        self.write("id,x,y,z,label,include_cam_in_training\np1,1,2,3,A,True\n")
        anns = an.Annotations(self.csv, orig_coords_only=True)
        anns["p1"].meta_data["include_cam_in_training"] = "False"
        anns.save(self.csv, orig_coords_only=True)
        head, row = self.rows()
        self.assertEqual(head.count("include_cam_in_training"), 1)
        self.assertEqual(dict(zip(head, row))["include_cam_in_training"], "False")

    def test_save_keeps_bom_and_mode(self):
        self.write("id,x,y,z,label\np1,1,2,3,A\n", bom=True)
        os.chmod(self.csv, 0o664)
        an.Annotations(self.csv).save(self.csv)
        with open(self.csv, "rb") as fh:
            self.assertEqual(fh.read(3), b"\xef\xbb\xbf")
        self.assertEqual(stat.S_IMODE(os.stat(self.csv).st_mode), 0o664)

    def test_edit_helpers_read_quoted_newlines(self):
        self.write('id,x,y,z,label,notes\np1,1,2,3,A,"two\nlines"\np2,1,2,3,B,\n')
        an.update_annotation_csv_row(self.csv, "p2", {"label": "C"})
        rows = self.rows()
        self.assertEqual(rows[1][5], "two\nlines")
        self.assertEqual(rows[2][4], "C")

    def test_measurements_csv_with_bom(self):
        anns = an.Annotations(coords={"p1": [0.0, 0.0, 0.0]})
        self.write("id,rugosity\np1,1.5\n", bom=True)
        anns.load_measurements_from_csv(self.csv)
        self.assertEqual(anns["p1"].measurements["rugosity"], 1.5)


class TestDuplicateIds(_Tmp):
    """A4: postfix rows merge at the end of the id, in any order; exact
    duplicates warn and survive a save."""

    def test_postfix_rows_before_base(self):
        self.write("id,x,y,z,label\nsb1_left,0,0,0,\nsb1,1,1,1,T\nsb1_right,2,2,2,\n")
        anns = an.Annotations(self.csv)
        self.assertEqual(list(anns.data), ["sb1"])
        self.assertEqual(set(anns["sb1"].extra_coords), {"sb1_left", "sb1_right"})

    def test_postfix_only_at_end(self):
        self.write("id,x,y,z,label\npt,0,0,0,\npt_left_2,1,1,1,\n")
        anns = an.Annotations(self.csv)
        self.assertEqual(set(anns.data), {"pt", "pt_left_2"})

    def test_exact_duplicate_warns_and_is_kept(self):
        self.write("id,x,y,z,label\np1,0,0,0,A\np1,9,9,9,B\n")
        with self.assertLogs("substrata-test", level="WARNING"):
            anns = an.Annotations(self.csv)
        self.assertEqual(anns["p1"].label, "A")
        self.assertEqual(anns["p1"].extra_coords, {})
        anns.save(self.csv)
        rows = self.rows()
        self.assertEqual([r[0] for r in rows[1:]], ["p1", "p1"])
        self.assertEqual(rows[2][4], "B")

    def test_postfix_rows_written_back(self):
        self.write("id,x,y,z,label\nsb1,1,1,1,T\nsb1_left,2,2,2,\n")
        an.Annotations(self.csv).save(self.csv)
        self.assertEqual([r[0] for r in self.rows()[1:]], ["sb1", "sb1_left"])


class TestContainers(_Tmp):
    """A14 nested iteration; A28 subsets keep metadata; A13; A27; A5."""

    def make(self):
        anns = an.Annotations(coords={"a": [0.0, 0, 0], "b": [1.0, 0, 0]})
        for a, label in zip(anns, ("X", "Y")):
            a.label = label
        return anns

    def test_nested_iteration(self):
        anns = self.make()
        self.assertEqual(len([(a.id, b.id) for a in anns for b in anns]), 4)

    def test_subsets_keep_metadata(self):
        anns = self.make()
        anns.world_transform = np.diag([2.0, 2, 2, 1])
        anns.filepath = self.csv
        anns.up_vector, anns.depth_offset = np.array([0, 0, 1.0]), 3.0
        for sub in (anns.subset(1), anns.subset_by_prefix("a"), anns.subset_by_label("X")):
            self.assertEqual(list(sub.data), ["a"])
            np.testing.assert_allclose(sub.world_transform, anns.world_transform)
            self.assertEqual(sub.crops_root, anns.crops_root)
            self.assertEqual(sub.depth_offset, 3.0)
            self.assertIs(sub["a"].parent, anns)
        sub.world_transform[0, 0] = 5
        self.assertEqual(anns.world_transform[0, 0], 2.0)

    def test_get_original_coords_sets_world_transform(self):
        anns = self.make()
        T = np.eye(4)
        T[:3, 3] = [10, 0, 0]
        anns.reverse_transform(T)
        np.testing.assert_allclose(anns.world_transform, T)
        np.testing.assert_allclose(anns["b"].orig_coords, [-9, 0, 0])

    def test_default_label(self):
        a = an.Annotation([0, 0, 0], id="p")
        self.assertIsNone(a.label)
        self.assertIsNone(a.label_conf)

    def test_reduce_point_cloud_centres_on_coords(self):
        a = an.Annotation([0.0, 0, 0], id="p")
        a.coords = np.array([100.0, 0, 0])
        a.radius = 1.0
        a.simple_pcd = types.SimpleNamespace(
            points=np.array([[100.0, 0, 0], [100.5, 0, 0], [0, 0, 0]]),
            colors=None, normals=None, labels=None)
        a.reduce_point_cloud_to_radius()
        self.assertEqual(len(a.simple_pcd.points), 2)


class TestCsvEdits(_Tmp):
    """A29 mixed-case column names; A30 concurrent edits; backups."""

    def setUp(self):
        super().setUp()
        self.write("id,x,y,z,Label\np1,0,0,0,A\np2,0,0,0,B\n")

    def test_mixed_case_column(self):
        an.update_annotation_csv_rows(self.csv, {"p1": {"LABEL": "Z"}})
        rows = self.rows()
        self.assertEqual(rows[0], ["id", "x", "y", "z", "Label"])
        self.assertEqual(rows[1][4], "Z")

    def test_append_mixed_case_keys(self):
        an.append_annotation_csv_row(self.csv, {"ID": "p3", "Label": "C"})
        self.assertEqual(self.rows()[3], ["p3", "", "", "", "C"])

    def test_concurrent_edits_keep_every_change(self):
        n = 40
        rows = "".join(f"q{i},0,0,0,\n" for i in range(n))
        self.write("id,x,y,z,label,flag\n" + rows.replace(",\n", ",,\n"))
        threads = [
            threading.Thread(target=an.update_annotation_csv_row,
                             args=(self.csv, f"q{i}", {"flag": str(i)}))
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = {r[0]: r[5] for r in self.rows()[1:]}
        self.assertEqual(got, {f"q{i}": str(i) for i in range(n)})
        self.assertEqual([f for f in os.listdir(self.tmp) if f != "anns.csv"], [])

    def test_backups_in_one_second_do_not_overwrite(self):
        first = an.backup_annotation_csv(self.csv)
        second = an.backup_annotation_csv(self.csv)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first) and os.path.exists(second))


class TestStaleCrops(_Tmp):
    """A7: crops are matched to photos by (folder, stem)."""

    def test_same_stem_other_folder_not_stale(self):
        class Cam:
            def __init__(self, fp, px):
                self.filepath, self.px = fp, px

            def get_pixel_coords(self, *_a, **_k):
                return self.px

        cams = [Cam("/p/a.auv/IMG_1.jpg", (10.0, 10.0)),
                Cam("/p/b.auv/IMG_1.jpg", (50.0, 50.0))]
        anns = an.Annotations(coords={"p": [0.0, 0, 0]})
        anns.crops_root = self.tmp
        for folder, x in (("a.auv", 10), ("b.auv", 50)):
            os.makedirs(os.path.join(self.tmp, "p", folder))
            open(os.path.join(self.tmp, "p", folder, f"IMG_1_{x}_{x}.jpg"), "w").close()
        self.assertEqual(anns["p"].stale_crops(cams), [])


class TestFileio(_Tmp):
    """A9/C1: atomic writes keep the destination's mode (umask for new files)."""

    def test_existing_mode_kept(self):
        self.write("x")
        os.chmod(self.csv, 0o640)
        fio.atomic_write_text(self.csv, "y")
        self.assertEqual(stat.S_IMODE(os.stat(self.csv).st_mode), 0o640)

    def test_new_file_gets_umask_mode(self):
        old = os.umask(0o027)
        try:
            fio.atomic_write_text(self.csv, "y")
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(os.stat(self.csv).st_mode), 0o640)

    def test_failure_leaves_file_and_no_temp(self):
        self.write("keep")
        with self.assertRaises(RuntimeError):
            with fio.atomic_write(self.csv) as fh:
                fh.write("partial")
                raise RuntimeError
        with open(self.csv) as fh:
            self.assertEqual(fh.read(), "keep")
        self.assertEqual(os.listdir(self.tmp), ["anns.csv"])


class _MarginCam:
    filename = filepath = "c.jpg"
    enabled = True
    camera_type = None
    orig_camera_transform = np.eye(4)

    def get_pixel_coords(self, coords, use_orig_coords=True):
        return 10.0, 20.0, 1.0, 0.5


class _RecordingOccluder:
    def __init__(self):
        self.scale_factors = []

    def normal_at(self, p):
        return None

    def line_of_sight_margin(self, scale_factor=None):
        self.scale_factors.append(scale_factor)
        return 0.01

    def blockers(self, eye, p, margin, normal=None):
        return 0


class TestLineOfSightMarginScale(unittest.TestCase):
    """The occlusion margin's metre floor gets the project's metres per unit."""

    def margin_scale(self, parent):
        occ = _RecordingOccluder()
        an.Annotation([0.0, 0.0, 0.0], parent=parent).get_image_matches(
            [_MarginCam()], occluder=occ)
        return occ.scale_factors[0]

    def test_scale_from_world_transform(self):
        parent = types.SimpleNamespace(world_transform=np.diag([2.0, 2.0, 2.0, 1.0]))
        self.assertAlmostEqual(self.margin_scale(parent), 2.0)

    def test_metres_per_unit_attribute_wins(self):
        parent = types.SimpleNamespace(world_transform=np.eye(4), metres_per_unit=0.5)
        self.assertAlmostEqual(self.margin_scale(parent), 0.5)

    def test_scale_factor_of_unoriented_project(self):
        parent = types.SimpleNamespace(world_transform=np.eye(4), scale_factor=0.25)
        self.assertAlmostEqual(self.margin_scale(parent), 0.25)

    def test_no_parent(self):
        self.assertIsNone(self.margin_scale(None))


if __name__ == "__main__":
    unittest.main()
