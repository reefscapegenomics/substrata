"""substrata.imagesync (camsync --match-images): poses for unaligned photos (close-ups) from matches with aligned
frames, and the Metashape import script's helpers.

The pose parts need the real package (open3d, cv2), whose heavy imports other test modules replace with stubs in
``sys.modules``: they run in a fresh interpreter, skipped outside the conda env. The learned matchers (kornia) are
not exercised here (their weights are downloaded): the real-data check is in the changelog.
"""

# Standard Library
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_HEAVY = importlib.util.find_spec("open3d") is not None and importlib.util.find_spec("cv2") is not None


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _SRC / "substrata" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPure(unittest.TestCase):
    """The parts without heavy dependencies (numpy only)."""

    @classmethod
    def setUpClass(cls):
        cls.isy = _load("imagesync_under_test", "imagesync.py")

    def test_times_and_window(self):
        isy = self.isy

        class C:
            def __init__(self, t):
                self.datetime = t

        self.assertEqual(isy.parse_time("2025:09:28 09:45:35"), dt.datetime(2025, 9, 28, 9, 45, 35))
        self.assertIsNone(isy.parse_time(None))
        self.assertIsNone(isy.parse_time("28/09/2025"))
        cams = [C("2025:09:28 09:00:00"), C("2025:09:28 09:00:00"), C(None), C("2025:09:28 09:00:05")]
        got = isy.by_time(cams)
        self.assertEqual(list(got), [dt.datetime(2025, 9, 28, 9, 0, 0), dt.datetime(2025, 9, 28, 9, 0, 5)])
        self.assertIs(got[dt.datetime(2025, 9, 28, 9, 0, 0)], cams[0])  # (the first at a second)
        t = lambda s: dt.datetime(2025, 9, 28, 9, 0, s)  # noqa: E731
        self.assertEqual(isy.overlap_window([t(10), t(20)], [t(0), t(50)]), (-20, 40))

    def test_peak(self):
        import numpy as np
        isy = self.isy
        ks = np.arange(-5, 6)
        curve = np.array([1, 2, 1, 2, 1, 9, 2, 1, 0, 0, 2], float)
        p = isy.peak(ks, curve)
        self.assertEqual(p["offset"], 0)
        self.assertAlmostEqual(p["prominence"], 9 / 2)  # (the median of the scoring offsets: 2)
        self.assertEqual(isy.peak(ks, np.zeros(11))["prominence"], 0.0)

    def test_candidate_offsets_and_confirmed(self):
        import numpy as np
        isy = self.isy
        ks = np.arange(-50, 51)
        curve = np.exp(-((ks - 10) / 3.0) ** 2) + 0.5 * np.exp(-((ks + 30) / 3.0) ** 2)
        # highest first, each at least 20 s from those before (the shoulder round 10 s is not a candidate)
        self.assertEqual(isy.candidate_offsets(ks, curve, n=2), [10, -30])
        self.assertEqual(isy.candidate_offsets(ks, np.zeros(len(ks))), [])
        # 3 close-ups and a quarter of those sampled; all of a sample smaller than 3
        self.assertTrue(isy.confirmed(3, 12))
        self.assertFalse(isy.confirmed(2, 12))
        self.assertFalse(isy.confirmed(4, 20))
        self.assertTrue(isy.confirmed(5, 20))
        self.assertTrue(isy.confirmed(2, 2))
        self.assertFalse(isy.confirmed(1, 2))
        self.assertFalse(isy.confirmed(0, 0))

    def test_crop_box(self):
        isy = self.isy
        self.assertEqual(isy.crop_box((1890, 1396), 400, 3840, 2160), (1090, 796, 1600, 1200))
        self.assertEqual(isy.crop_box((100, 100), 400, 3840, 2160), (0, 0, 1600, 1200))  # (inside the frame)
        self.assertEqual(isy.crop_box((1900, 1000), 1000, 3840, 2160), (0, 0, 3840, 2160))  # (at most the frame)

    def test_turns(self):
        import numpy as np
        isy = self.isy
        img = np.zeros((40, 60))
        img[7, 11] = 1  # (x 11, y 7)
        for k in range(4):
            r = np.rot90(img, k)
            y, x = np.argwhere(r == 1)[0]
            self.assertEqual(isy.unrotate([[x, y]], k, 60, 40).tolist(), [[11.0, 7.0]], k)
        # np.rot90(k=1) maps (x, y) -> (y, w - 1 - x): a homography angle of -90 deg (image coordinates)
        self.assertEqual(isy.quarter_turns(-90.0), 1)
        self.assertEqual(isy.quarter_turns(-170.0), 2)
        self.assertEqual(isy.quarter_turns(178.0), 2)
        self.assertEqual(isy.quarter_turns(-28.0), 0)
        self.assertEqual(isy.quarter_turns(None), 0)
        self.assertEqual(isy.circular_median_deg([-170, 178, -165, -178, 10]), -178)

    def test_frame_tiles(self):
        import numpy as np
        isy = self.isy
        tiles = isy.frame_tiles((1890, 1396), 350, 3840, 2160)
        self.assertEqual(tiles[0], isy.crop_box((1890, 1396), 350, 3840, 2160))  # (round the aim first)
        self.assertEqual(len(set(tiles)), len(tiles))
        self.assertTrue(all(0 <= x and 0 <= y and x + w <= 3840 and y + h <= 2160 for x, y, w, h in tiles))
        # a close-up (350 px wide) anywhere in the frame lies whole in one crop
        for cx, cy in np.random.default_rng(0).uniform([175, 175], [3840 - 175, 2160 - 175], (200, 2)):
            self.assertTrue(any(x <= cx - 175 and cx + 175 <= x + w and y <= cy - 175 and cy + 175 <= y + h
                                for x, y, w, h in tiles), (cx, cy))
        d = [np.hypot(x + w / 2 - 1890, y + h / 2 - 1396) for x, y, w, h in tiles[1:]]
        self.assertEqual(d, sorted(d))  # (nearest first)

    def test_assess(self):
        import numpy as np
        isy = self.isy
        aim = np.zeros(3)
        p = lambda r, e=30.0, a=aim: dict(range=r, error=e, aim=a)  # noqa: E731
        poses = {"a": p(0.30), "b": p(0.31), "c": p(0.29), "d": p(0.80), "e": p(0.30, e=90.0), "f": None,
                 "g": dict(range=None, error=20.0, aim=None), "h": p(0.30)}
        poses["i"] = dict(p(0.30), R=np.diag([1.0, -1.0, -1.0]))  # (looking down -z)
        second = {"a": p(0.30, a=aim + [0.005, 0, 0]), "b": None, "c": p(0.29, a=aim + [0.05, 0, 0]),
                  "i": p(0.33, a=aim + [0.004, 0, -0.04])}  # (i: 4 cm deeper along the ray, 4 mm across)
        st = isy.assess(poses, second, width={k: 8726 for k in poses})
        self.assertEqual(st["a"], ("posed", "checked"))
        self.assertEqual(st["b"], ("posed", "unchecked"))
        self.assertEqual(st["c"][0], "rejected")
        self.assertIn("5.0 cm apart", st["c"][1])
        self.assertEqual(st["d"][0], "rejected")
        self.assertIn("range 0.80", st["d"][1])
        self.assertEqual(st["e"][0], "rejected")
        self.assertIn("reprojection", st["e"][1])
        self.assertEqual(st["f"], ("no pose", ""))
        self.assertEqual(st["g"], ("rejected", "its centre ray meets no cloud"))
        self.assertEqual(st["h"], ("posed", "unchecked"))  # (not in the second opinion)
        self.assertEqual(st["i"], ("posed", "checked"))
        self.assertAlmostEqual(isy.aim_apart(poses["i"], second["i"]), 0.004)
        self.assertEqual(isy.assess({"a": p(0.3)})["a"], ("posed", ""))  # (no second matcher)


class TestMetashapeImportHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mip = _load("import_camera_poses_under_test", "metashape_scripts/import_camera_poses.py")

    def test_planned_poses(self):
        T = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        payload = {"cameras": {"12": {"label": "B", "status": "posed", "transform": T},
                               "3": {"label": "A", "status": "posed", "transform": T},
                               "4": {"label": "C", "status": "rejected", "note": "range"},
                               "5": {"label": "D", "status": "posed"},
                               "6": {"label": "E", "status": "estimated", "transform": T}}}
        got = self.mip.planned_poses(payload)
        self.assertEqual([(k, l) for k, l, _t in got], [("12", "B"), ("3", "A"), ("6", "E")])  # (with a transform)
        self.assertEqual(got[0][2][1][3], 2.0)
        got = self.mip.planned_poses(payload, include_marked=False)
        self.assertEqual([(k, l) for k, l, _t in got], [("12", "B"), ("3", "A")])  # (posed only)
        with self.assertRaises(ValueError):
            self.mip.planned_poses({"cameras": {"1": {"status": "posed", "transform": [[1, 0], [0, 1]]}}})

    def test_default_paths(self):
        psx, poses = self.mip.default_paths("/data/sam_sa3_40m_20250928")
        self.assertEqual(psx, "/data/sam_sa3_40m_20250928/sam_sa3_40m_20250928.psx")
        self.assertEqual(poses, "/data/sam_sa3_40m_20250928/sam_sa3_40m_20250928_camsync_poses.json")
        self.assertEqual(self.mip.default_paths("/d", "x.psx", "p.json"), ("x.psx", "p.json"))


_SETUP = textwrap.dedent(
    """
    import json, os, tempfile
    import numpy as np
    import open3d as o3d
    import xml.etree.ElementTree as ET
    from substrata import cameras as cm, imagesync as isy
    from substrata.pointclouds import PointCloud

    def sensor(sid, w, h, f, **kw):
        params = {k: 0.0 for k in ("cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")}
        params.update(kw)
        return cm.Sensor(sid, f"s{sid}", {"width": w, "height": h}, dict(params, f=float(f)))

    def rot(ax, deg):
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler(ax, deg, degrees=True).as_matrix()

    # A world transform with a scale (as a plot's: Metashape's units -> metres)
    W = np.eye(4)
    W[:3, :3] = 0.2 * rot("z", 30)
    W[:3, 3] = [5.0, -2.0, -40.0]

    def camera(cams, name, R_cw, C, sens):
        # (pose in the world frame: camera -> world rotation R_cw, centre C; stored like a loaded camera's)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = 0.2 * R_cw, C
        c = cm.Camera(cams, name, T, np.asarray(C, float), path=f"/plot/p.x/{name}.jpg")
        c.reverse_transform_coords(W)
        c.sensor, c.enabled = sens, True
        cams.data[name] = c
        return c

    cams = cm.Cameras.__new__(cm.Cameras)
    cams.sensors, cams.data, cams.world_transform = {}, {}, W
    down = np.diag([1.0, -1.0, -1.0])  # (looking down -z)

    # The reef: a plane z = -40 (world), 1.5 mm grid, 0.6 x 0.6 m
    g = np.mgrid[-200:200, -200:200].reshape(2, -1).T * 0.0015
    pc = PointCloud()
    pc.o3d_pcd.points = o3d.utility.Vector3dVector(np.c_[g, np.full(len(g), -40.0)])
    """
)

_POSE = _SETUP + textwrap.dedent(
    """
    gopro = camera(cams, "G", down, [0.0, 0.0, -39.45], sensor(2, 3840, 2160, 1800))
    macro_s = sensor(1, 8726, 5840, 23610, k1=-0.13)
    R_true = down @ rot("x", 8) @ rot("y", -5)  # (camera -> world)
    C_true = np.array([0.03, -0.02, -39.70])
    macro = camera(cams, "M", np.eye(3), [0, 0, 0], macro_s)  # (pose unknown)

    # matches: points on the reef seen by both (the close-up's pixels through its lens model)
    rng = np.random.default_rng(1)
    ray = R_true @ np.array([0, 0, 1.0]); t = (-40.0 - C_true[2]) / ray[2]; aim_true = C_true + t * ray
    P = aim_true + np.c_[rng.uniform(-0.03, 0.03, 60), rng.uniform(-0.02, 0.02, 60), np.zeros(60)]  # (in its ~11 x 7 cm view)
    cam_pts = (P - C_true) @ R_true  # (world -> camera: R^T (P - C))
    ax, ay, ok = cm._distort(cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2], macro_s)
    assert ok.all()
    A = np.c_[ax, ay]
    B = np.array([gopro.get_pixel_coords(p, required_to_be_in_view=False)[:2] for p in P], float)
    A[:6] += 900  # (a few wrong matches)

    p = isy.solve_pose(macro, gopro, A, B, pc)
    assert p is not None
    # A flat patch through a narrow lens: a tilt and a sideways shift look alike, so the centre is loose (here
    # ~9 cm) while where it aims, its range and what it sees on the surface are exact
    assert np.linalg.norm(p["C"] - C_true) < 0.15, (p["C"], C_true)
    assert np.linalg.norm(p["aim"] - aim_true) < 0.003, (p["aim"], aim_true)
    assert abs(p["range"] - np.linalg.norm(aim_true - C_true)) < 0.01
    assert p["inliers"] >= 30 and p["error"] < 60, (p["inliers"], p["error"])

    # set_pose: the camera now sees the reef where the true camera does, in both frames
    isy.set_pose(macro, p["R"], p["C"], W)
    X = aim_true + [0.01, 0.005, 0.0]  # (a point on the reef)
    want = cm._distort(*((X - C_true) @ R_true), macro_s)[:2]
    x, y, *_ = macro.get_pixel_coords(X)
    assert abs(x - want[0]) < 60 and abs(y - want[1]) < 60, ((x, y), want)  # (< 1 mm at 0.3 m)
    Xo = (np.linalg.inv(W) @ np.r_[X, 1.0])[:3]
    assert macro.get_pixel_coords(Xo, use_orig_coords=True)[:2] == (x, y)
    Ro = macro.orig_camera_transform[:3, :3]
    assert np.allclose(Ro.T @ Ro, np.eye(3), atol=1e-9), "the original-frame pose is rigid (Metashape's)"
    assert cm.can_project(macro, use_orig_coords=False)

    # pose_from_points: too few points or a degenerate set give None
    assert isy.pose_from_points(P[:4], A[6:10], macro_s) is None
    print("OK")
    """
)

_SAVE = _SETUP + textwrap.dedent(
    """
    # Cameras.save writes a cleared pose as null, a set pose as the original-frame pose
    d = tempfile.mkdtemp()
    meta = os.path.join(d, "p.meta.json")
    pose = np.eye(4).tolist()
    json.dump({"cameras": {"M": {"path": "/plot/p.x/M.jpg", "center": [1, 2, 3], "transform": pose, "enabled": True},
                           "N": {"path": "/plot/p.x/N.jpg", "center": [1, 2, 3], "transform": pose, "enabled": True},
                           "G": {"path": "/plot/p.x/G.jpg", "center": None, "transform": None}}}, open(meta, "w"))
    s = sensor(1, 1000, 800, 1000)
    m = camera(cams, "M", down, [0.0, 0.0, -39.7], s)
    n = camera(cams, "N", down, [0.1, 0.0, -39.7], s)
    gcam = camera(cams, "G", down, [0.0, 0.0, -39.7], s)
    gcam.missing_pose_from_meta = True  # (loaded with a null pose: stays null)
    isy.set_pose(n, np.diag([1.0, -1.0, -1.0]).T, [0.2, 0.1, -39.6], W)
    isy.clear_pose(m)
    assert m.enabled is False and not cm.can_project(m)
    cams.save(meta)
    out = json.load(open(meta))["cameras"]
    assert out["M"]["center"] is None and out["M"]["transform"] is None and out["M"]["enabled"] is False, out["M"]
    assert np.allclose(out["N"]["center"], (np.linalg.inv(W) @ np.r_[0.2, 0.1, -39.6, 1.0])[:3])
    assert out["G"]["center"] is None
    # pose_status (camsync's marks) is written, and taken away with None
    n.pose_status, m.pose_status = "estimated", None
    cams.save(meta)
    out = json.load(open(meta))["cameras"]
    assert out["N"]["pose_status"] == "estimated" and "pose_status" not in out["M"] and "pose_status" not in out["G"]
    # ... and set_pose takes the flag away again
    isy.set_pose(m, np.diag([1.0, -1.0, -1.0]).T, [0.0, 0.0, -39.6], W)
    assert not getattr(m, "pose_cleared", False) and cm.can_project(m, use_orig_coords=False)

    # A sensor without a calibration: its nominal lens from cams.xml
    el = ET.fromstring('<sensor id="1" label="macro"><resolution width="8726" height="5840"/>'
                       '<property name="pixel_width" value="0.0042355260128913441"/>'
                       '<property name="focal_length" value="100"/></sensor>')
    ns = cm._nominal_sensor(el, 1, "macro", {"width": 8726, "height": 5840})
    assert abs(ns.f - 23610.0) < 1 and ns.cx == 4363 and ns.k1 == 0 and ns.calibration_class == "nominal"
    assert cm._nominal_sensor(ET.fromstring('<sensor id="2"><resolution width="3840" height="2160"/></sensor>'),
                              2, "x", {"width": 3840, "height": 2160}) is None
    assert sensor(0, 10, 10, 10).calibration_class == "adjusted"
    print("OK")
    """
)


_RIG = _SETUP + textwrap.dedent(
    """
    import datetime as dt
    from substrata.framepose import world_to_cam

    # A rig: the close-up 9 cm to the side and 15 cm in front of the GoPro, tilted 6 deg; frames every second
    R_rel, t_rel = rot("x", 6), np.array([0.0, 0.09, 0.15])
    gs, ms = sensor(2, 3840, 2160, 1800), sensor(1, 8726, 5840, 23610)
    t0 = dt.datetime(2025, 9, 28, 9, 0, 0)
    rng = np.random.default_rng(3)
    targets, matches, poses, status, src = [], {}, {}, {}, {}
    for i in range(9):
        R_cw = down @ rot("z", rng.uniform(-40, 40)) @ rot("x", rng.uniform(-10, 10))
        f = camera(cams, f"G{i}", R_cw, [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), -39.45], gs)
        src[t0 + dt.timedelta(seconds=i - 80)] = f  # (offset -80 s)
        c = camera(cams, f"M{i}", np.eye(3), [0, 0, 0], ms)
        c.datetime = (t0 + dt.timedelta(seconds=i)).strftime("%Y:%m:%d %H:%M:%S")
        Rf = world_to_cam(f.camera_transform)
        assert np.allclose(Rf, R_cw.T)
        R = (rot("y", rng.normal(0, 2)) @ R_rel) @ Rf  # (each matched pose a little off: the tilt ambiguity)
        C = np.asarray(f.coords) + Rf.T @ (t_rel + rng.normal(0, 0.01, 3))
        targets.append(c)
        cid = str(c.cam_id)
        if i < 7:
            matches[cid], poses[cid], status[cid] = dict(source=f, n=500), dict(R=R, C=C), ("posed", "checked")
        elif i == 7:
            matches[cid], poses[cid], status[cid] = dict(source=f, n=10), None, ("no pose", "")
            truth = (R_rel @ Rf, np.asarray(f.coords) + Rf.T @ t_rel)
        else:
            matches[cid], poses[cid], status[cid] = dict(source=f, n=300), dict(R=R, C=C), \
                ("rejected", "the matchers aim 4.4 cm apart")
    rig = isy.rig_offset(targets, matches, poses, status)
    assert rig["n"] == 7 and np.allclose(rig["t"], t_rel, atol=0.02), rig
    out = isy.estimate_poses(targets, src, -80, matches, poses, status, pc)
    assert out["n"] == 7 and out["spread_m"] < 0.03, out
    c7, c8 = str(targets[7].cam_id), str(targets[8].cam_id)
    assert status[c7][0] == "estimated" and "G7" in status[c7][1], status[c7]
    assert status[c8] == ("unverified", "the matchers aim 4.4 cm apart")  # (its matched pose kept)
    p = poses[c7]
    assert np.linalg.norm(p["C"] - truth[1]) < 0.02 and np.degrees(np.arccos((np.trace(p["R"] @ truth[0].T) - 1) / 2)) < 2
    assert p["aim"] is not None and p["inliers"] == 0 and p["frame"] is src[t0 + dt.timedelta(seconds=7 - 80)]
    # too few posed close-ups: no rig, nothing estimated (the unverified still marked)
    st = {str(c.cam_id): ("no pose", "") for c in targets}
    st[c8] = ("rejected", "the matchers aim 3 cm apart")
    assert isy.estimate_poses(targets, src, -80, matches, dict(poses), st, pc) is None
    assert st[c7] == ("no pose", "") and st[c8][0] == "unverified"
    print("OK")
    """
)


@unittest.skipUnless(_HEAVY, "requires the open3d/substrata environment")
class TestPoses(unittest.TestCase):
    def _run(self, script):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(_SRC), timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("OK", r.stdout)

    def test_solve_and_set_pose(self):
        self._run(_POSE)

    def test_save_clear_and_nominal_sensor(self):
        self._run(_SAVE)

    def test_rig_estimate(self):
        self._run(_RIG)

    def test_parsers(self):
        for cmd, opts in ((["camsync", "--help"], ["--match-images", "--offset-window", "--matcher", "--no-check",
                                                   "--frames", "--dry-run"]),
                          (["metashape-import-poses", "--help"], ["--psx", "--poses", "--dry-run"])):
            r = subprocess.run([sys.executable, "-m", "substrata.cli", *cmd], capture_output=True, text=True,
                               cwd=tempfile.gettempdir(), env=dict(os.environ, PYTHONPATH=str(_SRC)), timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for o in opts:
                self.assertIn(o, r.stdout)


if __name__ == "__main__":
    unittest.main()
