"""substrata.lensfix (``substrata reproject-views``): training views written with the lens model before 0.2 moved
to the corrected model's pixels; those it does not see the point in cleared; everything else in the CSV kept."""

# Standard Library
import csv
import importlib
import json
import os
import tempfile
import unittest

# Third-Party Libraries
import numpy as np

try:  # Heavy deps (cv2, …) are only present in the conda envs.
    cm = importlib.import_module("substrata.cameras")
    lensfix = importlib.import_module("substrata.lensfix")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    cm = lensfix = None

DATA = os.path.join(os.path.dirname(__file__), "data", "metashape_lens.json")
TERMS = ("f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")


def _pgram02():
    """sa3_20m's PGRAM02 calibration (large P2, B2, and a lens that folds back at r 0.878, inside the image)."""
    with open(DATA) as fh:
        cal = next(e["calibration"] for e in json.load(fh)["sensors"] if "sa3_20m PGRAM02" in e["calibration"]["name"])
    return cm.Sensor(4, "PGRAM02", {"width": cal["width"], "height": cal["height"]},
                     {k: float(cal.get(k, 0.0)) for k in TERMS})


class _Cams:
    def __init__(self, cams):
        self.data = {c.cam_id: c for c in cams}


def _camera(name, sensor, shift=(0.0, 0.0, 0.0)):
    """A camera looking along +z from `shift` (its pose in the original frame)."""
    T = np.eye(4)
    T[:3, 3] = shift
    c = cm.Camera(cam_id=name, camera_transform=T, coords=T[:3, 3].copy(), path=f"/photos/p.photos/{name}.jpg")
    c.orig_camera_transform, c.orig_coords, c.sensor = T, T[:3, 3].copy(), sensor
    return c


@unittest.skipIf(cm is None, "substrata not importable")
class TestReprojectViews(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.s = _pgram02()
        self.cams = _Cams([_camera("IMG_1", self.s), _camera("IMG_2", self.s, (1.0, 0.0, 0.0))])

    def tearDown(self):
        self.tmp.cleanup()

    def _old(self, xyz, shift=(0.0, 0.0, 0.0)):
        p = np.asarray(xyz, float) - np.asarray(shift)
        x, y = lensfix.legacy_pixels(p[0], p[1], p[2], self.s)
        return float(round(x)), float(round(y))

    def _new(self, xyz, shift=(0.0, 0.0, 0.0)):
        p = np.asarray(xyz, float) - np.asarray(shift)
        x, y, ok = cm._distort(p[0], p[1], p[2], self.s)
        return float(round(float(x))), float(round(float(y))), bool(ok)

    def _write(self, rows):
        path = os.path.join(self.dir, "p_ann.csv")
        head = ["id", "orig_x", "orig_y", "orig_z", "label", "cam_filepath", "cam_x", "cam_y",
                "include_cam_in_training", "pred_label"]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\r\n")  # (the line ending is kept)
            w.writerow(head)
            for r in rows:
                w.writerow([r.get(h, "") for h in head])
        with open(os.path.join(self.dir, "p_ann.features.npz"), "wb") as fh:
            fh.write(b"features at the old pixels")
        return path

    def test_moves_clears_and_keeps(self):
        moved, ghost = (0.5 * 2, 0.3 * 2, 2.0), (0.0, 1.206 * 3, 3.0)  # (ghost: beyond the fold, inside the image)
        nx, ny, ok = self._new(moved)
        self.assertTrue(ok and self._old(moved) != (nx, ny))
        gx, gy, gok = self._new(ghost)
        self.assertFalse(gok)
        ox, oy = self._old(ghost)
        self.assertTrue(0 < ox < self.s.width and 0 < oy < self.s.height)  # (the old model's view: in the photo)
        cur, other = (0.1, -0.2, 1.5), (1.3, 0.1, 2.0)
        rows = [
            dict(id="moved", orig_x=moved[0], orig_y=moved[1], orig_z=moved[2], label="CCA",
                 cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x=f"{self._old(moved)[0]:.2f}",
                 cam_y=f"{self._old(moved)[1]:.2f}", include_cam_in_training="True", pred_label="CCA"),
            dict(id="current", orig_x=cur[0], orig_y=cur[1], orig_z=cur[2],
                 cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x=f"{self._new(cur)[0]:.2f}",
                 cam_y=f"{self._new(cur)[1]:.2f}", include_cam_in_training="True"),
            dict(id="other_cam", orig_x=other[0], orig_y=other[1], orig_z=other[2],
                 cam_filepath="C:\\photos\\p.photos\\IMG_2.jpg",
                 cam_x=f"{self._old(other, (1, 0, 0))[0]:.2f}", cam_y=f"{self._old(other, (1, 0, 0))[1]:.2f}",
                 include_cam_in_training="True"),
            dict(id="elsewhere", orig_x=moved[0], orig_y=moved[1], orig_z=moved[2],
                 cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x=f"{self._old(moved)[0] + 30:.2f}",
                 cam_y=f"{self._old(moved)[1]:.2f}", include_cam_in_training="True"),
            dict(id="ghost", orig_x=ghost[0], orig_y=ghost[1], orig_z=ghost[2], label="SAND",
                 cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x=f"{ox:.2f}", cam_y=f"{oy:.2f}",
                 include_cam_in_training="True"),
            dict(id="ghost_left_out", orig_x=ghost[0], orig_y=ghost[1], orig_z=ghost[2],
                 cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x=f"{ox:.2f}", cam_y=f"{oy:.2f}",
                 include_cam_in_training="False"),
            dict(id="other_plot", orig_x=moved[0], orig_y=moved[1], orig_z=moved[2],
                 cam_filepath="/elsewhere/IMG_9.jpg", cam_x="10.00", cam_y="10.00"),
            dict(id="image_only", cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x="5", cam_y="6"),
            dict(id="no_view", orig_x=moved[0], orig_y=moved[1], orig_z=moved[2]),
        ]
        path = self._write(rows)
        before = open(path, newline="").read()

        dry = lensfix.reproject_views(path, self.cams, dry_run=True)
        self.assertEqual(open(path, newline="").read(), before)  # (a dry run writes nothing)
        self.assertEqual(dry["counts"], dict(rows=9, with_view=7, moved=2, current=1, elsewhere=1, not_in_plot=1,
                                             out_of_view=1, out_of_view_kept=1))
        self.assertEqual(sorted(dry["moves"]), ["moved", "other_cam"])
        self.assertEqual(dry["clear"], ["ghost"])
        self.assertIsNone(dry["backup"])

        res = lensfix.reproject_views(path, self.cams)
        self.assertEqual(res["counts"], dry["counts"])
        self.assertTrue(os.path.isfile(res["backup"]) and open(res["backup"], newline="").read() == before)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "p_ann.features.npz")))
        self.assertTrue(os.path.isfile(res["features"]))
        self.assertEqual(open(res["cleared_list"]).read().split(), ["ghost"])
        text = open(path, newline="").read()
        self.assertIn("\r\n", text)
        got = {r["id"]: r for r in csv.DictReader(text.splitlines())}
        old = {r["id"]: r for r in csv.DictReader(before.splitlines())}
        self.assertEqual((got["moved"]["cam_x"], got["moved"]["cam_y"]), (f"{nx:.2f}", f"{ny:.2f}"))
        nx2, ny2, _ = self._new(other, (1, 0, 0))
        self.assertEqual((got["other_cam"]["cam_x"], got["other_cam"]["cam_y"]), (f"{nx2:.2f}", f"{ny2:.2f}"))
        self.assertEqual([got["ghost"][c] for c in ("cam_filepath", "cam_x", "cam_y")], ["", "", ""])
        self.assertEqual(got["ghost"]["label"], "SAND")
        changed = {("moved", "cam_x"), ("moved", "cam_y"), ("other_cam", "cam_x"), ("other_cam", "cam_y"),
                   ("ghost", "cam_filepath"), ("ghost", "cam_x"), ("ghost", "cam_y")}
        for i, r in old.items():  # (nothing else changed: labels, training flags, predictions, the other rows)
            for c, v in r.items():
                if (i, c) not in changed:
                    self.assertEqual(got[i][c], v, (i, c))

        again = lensfix.reproject_views(path, self.cams)  # (again: nothing to do)
        self.assertEqual((again["counts"]["moved"], again["counts"]["out_of_view"], again["counts"]["current"]),
                         (0, 0, 3))
        self.assertIsNone(again["backup"])

    def test_keep_out_of_view_and_sets_without_points(self):
        ghost = (0.0, 1.206 * 3, 3.0)
        ox, oy = self._old(ghost)
        path = self._write([dict(id="ghost", orig_x=ghost[0], orig_y=ghost[1], orig_z=ghost[2],
                                 cam_filepath="/photos/p.photos/IMG_1.jpg", cam_x=f"{ox:.2f}", cam_y=f"{oy:.2f}")])
        before = open(path, newline="").read()
        res = lensfix.reproject_views(path, self.cams, keep_out_of_view=True)
        self.assertEqual((res["counts"]["out_of_view"], res["counts"]["out_of_view_kept"]), (0, 1))
        self.assertEqual(open(path, newline="").read(), before)
        flat = os.path.join(self.dir, "flat.csv")  # (an image set: no 3-D columns)
        with open(flat, "w") as fh:
            fh.write("id,cam_filepath,cam_x,cam_y,label\na,/x/IMG_1.jpg,1,2,C\n")
        self.assertEqual(lensfix.reproject_views(flat, self.cams)["counts"]["with_view"], 0)


if __name__ == "__main__":
    unittest.main()
