"""substrata.grouppredict (``substrata predict --camera-group``): a set's points classified from one camera
group's photos, disabled cameras included, into a separate CSV; the set's own CSV untouched."""

# Standard Library
import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# Third-Party Libraries
import numpy as np

try:  # Heavy deps (cv2, …) are only present in the conda envs.
    cm = importlib.import_module("substrata.cameras")
    an = importlib.import_module("substrata.annotations")
    gp = importlib.import_module("substrata.grouppredict")
    seg = importlib.import_module("substrata.segmentation")
    cl = importlib.import_module("substrata.classification")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    cm = None


def _sensor(sid, w, h, f):
    params = {k: 0.0 for k in ("cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")}
    return cm.Sensor(sid, f"s{sid}", {"width": w, "height": h}, dict(params, f=float(f)))


def _cameras(specs):
    """A Cameras container: (name, folder, sensor, x shift, enabled) per camera, each looking along +z."""
    cams = cm.Cameras.__new__(cm.Cameras)
    cams.sensors, cams.data = {}, {}
    cams.world_transform = np.eye(4)
    for name, folder, sensor, dx, enabled in specs:
        T = np.eye(4)
        T[0, 3] = dx
        c = cm.Camera(cam_id=name, camera_transform=T, coords=T[:3, 3].copy(),
                      path=f"/plot/{folder}/{name}.jpg")
        c.orig_camera_transform, c.orig_coords, c.sensor = T, T[:3, 3].copy(), sensor
        c.enabled, c.parent = enabled, cams
        cams.data[name] = c
    return cams


@unittest.skipIf(cm is None, "substrata not importable")
class TestCameraGroups(unittest.TestCase):
    def setUp(self):
        div, auv = _sensor(0, 2000, 1500, 1500), _sensor(1, 1000, 1000, 800)
        self.cams = _cameras([
            ("DIV_1", "p.photos", div, 0.0, True), ("DIV_2", "p.photos", div, 0.2, True),
            ("PR_1", "p.auv", auv, 0.0, False), ("PR_2", "p.auv", auv, 0.3, False),
            ("PR_3", "p.auv", auv, 9.0, False),  # (not seeing the points)
        ])

    def test_resolve(self):
        name, kind, m = gp.resolve_camera_group(self.cams, "1000x1000 f800")
        self.assertEqual((name, kind, len(m)), ("1000x1000 f800", "type", 3))
        self.assertEqual(len(gp.resolve_camera_group(self.cams, "auv")[2]), 3)  # (the folder of auv_PR)
        self.assertEqual(gp.resolve_camera_group(self.cams, "group:auv_PR")[1], "group")
        self.assertEqual(len(gp.resolve_camera_group(self.cams, "enabled")[2]), 2)
        self.assertEqual(len(gp.resolve_camera_group(self.cams, "all")[2]), 5)
        self.assertEqual(gp.resolve_camera_group(self.cams, "1000x1000")[:2], ("1000x1000", "system"))
        self.assertEqual(len(gp.resolve_camera_group(self.cams, "system:2000x1500")[2]), 2)
        with self.assertRaises(ValueError) as e:
            gp.resolve_camera_group(self.cams, "gopro")
        self.assertIn("auv", str(e.exception))
        groups = gp.camera_groups(self.cams)
        self.assertEqual(groups["2000x1500 f1500"], dict(kind="type", n=2, enabled=2))
        self.assertEqual(groups["auv"], dict(kind="group", n=3, enabled=0))

    def test_match_includes_disabled_and_counts_available(self):
        qc = np.array([[0.1, 0.0, 5.0], [0.2, 0.1, 4.0]])
        auv = gp.resolve_camera_group(self.cams, "auv")[2]
        vc, *_r, available = seg._match_points_to_cameras_multi(
            qc, auv, views=3, verbose=False, include_disabled=True, return_available=True)
        self.assertEqual(available.tolist(), [2, 2])
        self.assertEqual(((vc >= 0).sum(1)).tolist(), [2, 2])
        vc_default, *_ = seg._match_points_to_cameras_multi(qc, auv, views=3, verbose=False)
        self.assertTrue((vc_default == -1).all())  # (default: disabled cameras are not used)


@unittest.skipIf(cm is None, "substrata not importable")
class TestPredictFromGroup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        div, auv = _sensor(0, 2000, 1500, 1500), _sensor(1, 1000, 1000, 800)
        self.cams = _cameras([
            ("DIV_1", "p.photos", div, 0.0, True),
            ("PR_1", "p.auv", auv, 0.0, False), ("PR_2", "p.auv", auv, 0.3, False),
        ])
        self.csv = os.path.join(self.tmp.name, "p.annos", "p_slope_intercepts.csv")
        os.makedirs(os.path.dirname(self.csv))
        with open(self.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "orig_x", "orig_y", "orig_z", "label", "cam_filepath", "cam_x", "cam_y",
                        "pred_label"])
            w.writerow(["a", 0.1, 0.0, 5.0, "CCA", "/plot/p.photos/DIV_1.jpg", 1030, 750, "CCA"])
            w.writerow(["b", 0.2, 0.1, 4.0, "SAND", "/plot/p.photos/DIV_1.jpg", 1075, 788, ""])
            w.writerow(["far", 50.0, 0.0, 5.0, "TURF", "", "", "", ""])  # (no camera sees it)
        self.before = open(self.csv, newline="").read()
        self.anns = an.Annotations(self.csv)
        self.auv = gp.resolve_camera_group(self.cams, "auv")[2]

    def tearDown(self):
        self.tmp.cleanup()

    def test_segment_mode(self):
        seen = {}

        def dense(vc, vx, vy, vd, vr, cam_list, clf, depth_scale, **kw):
            seen["cams"] = [cam_list[c].filename for c in vc[vc >= 0]]
            n = len(vc)
            ok = (vc >= 0).any(1)
            probs = np.where(ok[:, None], np.array([[0.7, 0.3]]), 0.0)
            return dict(labels=np.where(ok, "CCA", "").astype(object), conf=np.where(ok, 0.7, np.nan),
                        n_views=(vc >= 0).sum(1), agreement=np.where(ok, 1.0, np.nan),
                        cam=np.where(ok, vc[:, 0], -1), x=np.where(ok, vx[:, 0], np.nan),
                        y=np.where(ok, vy[:, 0], np.nan), probs=probs)

        clf = mock.Mock(is_point_classifier=True, classes=["CCA", "SAND"])
        # (other test modules stub substrata.annotations in sys.modules: the real one here)
        with mock.patch.object(seg, "_classify_dense_views", dense), \
                mock.patch.dict(sys.modules, {"substrata.annotations": an}):
            res = gp.predict_from_group(self.anns, self.auv, clf, mode="segment", verbose=False)
        self.assertTrue(seen["cams"] and all(c.startswith("PR_") for c in seen["cams"]))
        self.assertEqual((res["a"]["label"], res["a"]["n"], res["a"]["available"]), ("CCA", 2, 2))
        self.assertEqual(res["a"]["top"], "CCA:0.700;SAND:0.300")
        self.assertEqual((res["far"]["label"], res["far"]["n"], res["far"]["available"]), (None, 0, 0))

        out = gp.output_path(self.csv, "auv", "segment")
        self.assertEqual(out, os.path.join(self.tmp.name, "p.annos", "predictions",
                                           "p_slope_intercepts__auv__segment.csv"))
        n = gp.write_results(out, self.anns, res, group="auv", mode="segment", model="m1",
                             meta=dict(cameras=2))
        self.assertEqual(n, dict(points=3, with_view_available=2, predicted=2, no_view=1))
        rows = list(csv.DictReader(open(out, newline="")))
        self.assertEqual([r["id"] for r in rows], ["a", "b", "far"])
        self.assertEqual(tuple(rows[0][k] for k in ("label", "pred_label", "pred_n", "views_available",
                                                     "camera_group", "view_mode", "pred_model")),
                         ("CCA", "CCA", "2", "2", "auv", "segment", "m1"))
        self.assertTrue(rows[0]["view_filepath"].endswith("/p.auv/PR_1.jpg"))
        self.assertEqual((rows[2]["pred_label"], rows[2]["views_available"]), ("", "0"))
        info = json.load(open(out + ".json"))
        self.assertEqual((info["camera_group"], info["view_mode"], info["cameras"]), ("auv", "segment", 2))
        self.assertEqual(open(self.csv, newline="").read(), self.before)  # (the set's CSV: untouched)

    def test_predict_mode(self):
        got = {}

        def batch(matches, classifier, crop_size=None, **kw):
            got["files"] = [m.cam.filename for m in matches]
            return [dict(label="SAND", confidence=0.6, probs={"SAND": 0.6, "CCA": 0.4}) for _ in matches]

        with mock.patch.object(cl, "classify_image_matches_batch", batch):
            res = gp.predict_from_group(self.anns, self.auv, object(), mode="predict", views=10,
                                        clear_of_edge=(100, 100), verbose=False)
        self.assertTrue(got["files"] and all(f.startswith("PR_") for f in got["files"]))
        self.assertEqual((res["a"]["label"], res["a"]["n"], res["a"]["available"]), ("SAND", 2, 2))
        self.assertTrue(res["a"]["view"][0].endswith("PR_1.jpg") or res["a"]["view"][0].endswith("PR_2.jpg"))
        self.assertEqual((res["far"]["label"], res["far"]["available"]), (None, 0))
        self.assertEqual(open(self.csv, newline="").read(), self.before)


@unittest.skipIf(cm is None, "substrata not importable")
class TestCommandLine(unittest.TestCase):
    def test_predict_parser_has_the_camera_group_options(self):
        """The parser builds (an option clashing with another of predict's stops every substrata command)."""
        import subprocess
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        r = subprocess.run([sys.executable, "-m", "substrata.cli", "predict", "--help"], capture_output=True,
                           text=True, cwd=tempfile.gettempdir(), env=dict(os.environ, PYTHONPATH=src))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        for opt in ("--camera-group", "--view-mode", "--views", "--output-csv", "--tag"):
            self.assertIn(opt, r.stdout)


if __name__ == "__main__":
    unittest.main()
