"""Tests for the in-place annotation CSV edits and ``depth_in_m``.

``annotations.py`` imports the heavy conda deps at module load, so these tests
are skipped when they are not installed.
"""

# Standard Library
import importlib
import os
import tempfile
import unittest

# Third-party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    a = importlib.import_module("substrata.annotations")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    a = None

CSV = (
    "id,orig_x,orig_y,orig_z,label,label_conf,world_x,world_y,world_z,"
    "cam_filepath,cam_x,cam_y,depth,Extra\r\n"
    "pt_1,1.25,2.5,-3.0,PORI,4,10.1,20.2,-5.3,,,,,keep me\r\n"
    "pt_2,0.1,0.2,0.3,,,1,2,3,,,,,\r\n"
)


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestAnnotationCsvEdit(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "anns.csv")
        with open(self.path, "w", newline="") as fh:
            fh.write(CSV)

    def tearDown(self):
        self.dir.cleanup()

    def read(self):
        with open(self.path, newline="") as fh:
            return fh.read()

    def test_columns(self):
        cols = a.annotation_csv_columns(self.path)
        self.assertEqual(cols[0], "id")
        self.assertEqual(cols[-1], "extra")

    def test_update_changes_only_those_cells(self):
        a.update_annotation_csv_row(
            self.path, "pt_2", {"label": "ACRO", "label_conf": 3}
        )
        lines = self.read().split("\r\n")
        self.assertEqual(lines[1], CSV.split("\r\n")[1])
        self.assertEqual(lines[2], "pt_2,0.1,0.2,0.3,ACRO,3,1,2,3,,,,,")
        self.assertTrue(self.read().endswith("\r\n"))

    def test_update_adds_missing_column(self):
        a.update_annotation_csv_row(self.path, "pt_1", {"pred_label": "PORI"})
        lines = self.read().split("\r\n")
        self.assertTrue(lines[0].endswith(",pred_label"))
        self.assertTrue(lines[1].endswith(",keep me,PORI"))
        self.assertTrue(lines[2].endswith(","))

    def test_replace_column_values_only_touches_those_cells(self):
        with open(self.path, "w", newline="") as fh:
            fh.write(CSV.replace(",,,,,keep me", ",/old/IMG_1.JPG,,,,keep me"))
        n = a.replace_annotation_csv_values(
            self.path, "cam_filepath", {"/old/IMG_1.JPG": "/new/IMG_1.JPG"}
        )
        self.assertEqual(n, 1)
        expected = CSV.replace(",,,,,keep me", ",/new/IMG_1.JPG,,,,keep me")
        self.assertEqual(self.read(), expected)  # CRLF and other cells kept
        self.assertEqual(
            a.replace_annotation_csv_values(self.path, "nope", {"a": "b"}), 0
        )

    def test_update_unknown_id(self):
        with self.assertRaises(KeyError):
            a.update_annotation_csv_row(self.path, "nope", {"label": "X"})
        self.assertEqual(self.read(), CSV)

    def test_append_then_delete_is_identical(self):
        a.append_annotation_csv_row(
            self.path,
            {"id": "pt_3", "orig_x": 1.0, "orig_y": 2.0, "orig_z": 3.0, "bogus": 1},
        )
        lines = self.read().split("\r\n")
        self.assertEqual(lines[3], "pt_3,1.0,2.0,3.0,,,,,,,,,,")
        a.delete_annotation_csv_row(self.path, "pt_3")
        self.assertEqual(self.read(), CSV)

    def test_delete_unknown_id(self):
        with self.assertRaises(KeyError):
            a.delete_annotation_csv_row(self.path, "nope")

    def test_backup(self):
        out = a.backup_annotation_csv(self.path)
        self.assertEqual(os.path.dirname(out), os.path.join(self.dir.name, "backup"))
        with open(out, newline="") as fh:
            self.assertEqual(fh.read(), CSV)

    def test_new_id_after_append(self):
        anns = a.Annotations(self.path, orig_coords_only=True)
        self.assertEqual(anns.get_new_id(), "pt_3")
        self.assertEqual(anns.get_new_id(last_highest_id=7), "pt_8")


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestDepthInM(unittest.TestCase):
    def test_matches_annotation_property(self):
        anns = a.Annotations(coords={"p": np.array([1.0, 2.0, 3.0])})
        anns.up_vector = np.array([0.0, 0.5, -2.0])
        anns.depth_offset = 13.0
        expected = 13.0 + 1.0 - 6.0
        self.assertAlmostEqual(anns["p"].depth_in_m, expected)
        self.assertAlmostEqual(
            a.depth_in_m([1.0, 2.0, 3.0], anns.up_vector, 13.0), expected
        )

    def test_vectorised(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        d = a.depth_in_m(pts, [0.0, 0.0, 1.0], 5.0)
        np.testing.assert_allclose(d, [5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
