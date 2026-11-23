"""PointCloud.camera_targets and ``substrata cams2annos``: the points that cameras aim at (the ray through a
photo's centre meeting the cloud), each with its camera as the view.

Needs the real ``substrata`` package (open3d, cv2), whose heavy imports other test modules replace with stubs in
``sys.modules``. The checks therefore run in a fresh interpreter; skipped outside the conda env.
"""

# Standard Library
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_SETUP = textwrap.dedent(
    """
    import numpy as np
    import open3d as o3d
    from substrata import cameras as cm
    from substrata.pointclouds import PointCloud

    # A flat reef (z = 0, a 2 mm grid over 40 x 40 cm) with a hole of radius 6 mm around (0.10, 0.10)
    g = np.mgrid[-100:100, -100:100].reshape(2, -1).T * 0.002
    g = g[np.hypot(g[:, 0] - 0.10, g[:, 1] - 0.10) > 0.006]
    pc = PointCloud()
    pc.o3d_pcd.points = o3d.utility.Vector3dVector(np.c_[g, np.zeros(len(g))])
    t = np.eye(4)  # the world transform: a rotation about z and a shift
    t[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t[:3, 3] = [5.0, 1.0, -30.0]
    pc.apply_transform(t)

    params = {k: 0.0 for k in ("cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")}
    sensor = cm.Sensor(0, "macro", {"width": 1000, "height": 800}, dict(params, f=1000.0))
    down = np.diag([1.0, -1.0, -1.0])  # (looking along -z: the camera's +z)
    up = np.eye(3)  # (looking away from the reef)

    def camera(cams, name, R, xyz, sensor=sensor):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, xyz
        c = cm.Camera(cams, name, t @ T, (t @ T)[:3, 3].copy(), path=f"/plot/p.closeup/{name}.jpg")
        c.orig_camera_transform, c.orig_coords, c.sensor, c.enabled = T, T[:3, 3].copy(), sensor, True
        cams.data[name] = c
        return c

    cams = cm.Cameras.__new__(cm.Cameras)
    cams.sensors, cams.data, cams.world_transform = {0: sensor}, {}, t
    camera(cams, "C0", down, [0.05, -0.03, 0.5])  # aims at (0.05, -0.03, 0)
    camera(cams, "C1", up, [0.0, 0.0, 0.5])  # aims at the water
    camera(cams, "C2", down, [0.10, 0.10, 0.4])  # aims into the hole
    camera(cams, "C3", down, [0.0, 0.0, 0.5], sensor=None)  # no calibration
    """
)

_LIBRARY = _SETUP + textwrap.dedent(
    """
    anns = pc.camera_targets(cams, id_prefix="p_closeup")
    assert sorted(anns.data) == ["p_closeup_0", "p_closeup_2"], sorted(anns.data)
    assert [m[0] for m in anns.missed] == ["C1.jpg", "C3.jpg"], anns.missed
    assert "cannot project" in anns.missed[1][1]

    a = anns["p_closeup_0"]
    assert np.allclose(a.orig_coords, [0.05, -0.03, 0.0], atol=0.0015), a.orig_coords
    assert np.allclose(t @ np.r_[a.orig_coords, 1.0], np.r_[a.coords, 1.0])  # (the cloud's frame)
    assert a.search_radius == 0.003 and not a.is_extrapolated
    m = a.image_match
    assert m is a.image_matches[0] and m.cam.filename == "C0.jpg" and m.annotation is a
    assert abs(m.x - 500) <= 3 and abs(m.y - 400) <= 3, (m.x, m.y)
    assert np.isclose(a.meta_data["target_distance"], 0.5, atol=0.002)
    assert a.meta_data["target_offset_px"] <= 4
    assert (m.x, m.y) == cams["C0"].get_pixel_coords(a.orig_coords, use_orig_coords=True)[:2]

    # Into the hole: missed at 3 mm, met 3x wider, at the hole's edge (~6 mm from the ray)
    b = anns["p_closeup_2"]
    assert b.search_radius == 0.009, b.search_radius
    assert 0.005 < np.hypot(b.orig_coords[0] - 0.10, b.orig_coords[1] - 0.10) < 0.0085, b.orig_coords
    assert np.isclose(b.orig_coords[2], 0.0)

    # Another pixel than the centre
    one = pc.camera_targets([cams["C0"]], pixel=(700, 400))
    a2 = next(iter(one))
    assert np.allclose(a2.orig_coords[:2], [0.05 + 0.1, -0.03], atol=0.0015), a2.orig_coords  # (200 px at f 1000, 0.5 m)
    assert abs(a2.image_match.x - 700) <= 3

    # The view is saved as the point's cam_filepath / cam_x / cam_y
    import csv, os, tempfile
    path = os.path.join(tempfile.mkdtemp(), "t.csv")
    anns.save(path)
    rows = {r["id"]: r for r in csv.DictReader(open(path))}
    r = rows["p_closeup_0"]
    assert r["cam_filepath"].endswith("p.closeup/C0.jpg") and (int(r["cam_x"]), int(r["cam_y"])) == (m.x, m.y)
    assert float(r["target_distance"]) == a.meta_data["target_distance"]
    assert r["is_extrapolated"] == "False"

    # A camera whose pose camsync estimated (no image match): its point is marked extrapolated
    cams["C0"].pose_status = "estimated"
    est = next(iter(pc.camera_targets([cams["C0"]])))
    assert est.is_extrapolated
    cams["C0"].pose_status = "unverified"  # (a match the second matcher disputes: a match still)
    assert not next(iter(pc.camera_targets([cams["C0"]]))).is_extrapolated
    print("OK")
    """
)

_CLI = _SETUP + textwrap.dedent(
    """
    import argparse, csv, os, tempfile, yaml
    from unittest import mock
    from substrata import cli

    class Init:
        def __init__(self, path=None, local=False):
            self.path = path
        def initialize(self):
            self.pcd, self.cams, self.id, self.world_transform = pc, cams, "p", t
            self.annos_root, self.classes_filepath = None, None

    def args(**kw):
        d = dict(group="closeup", pixel=None, search_radius=0.003, max_search_radius=0.03, max_distance=5.0,
                 tag=None, overwrite=False, input=None, local=False)
        d.update(kw)
        return argparse.Namespace(**d)

    os.chdir(tempfile.mkdtemp())
    with mock.patch("substrata.initializer.ProjectInitializer", Init):
        cli.handle_cams2annos(args())
        out = os.path.join("p.annos", "p_closeup_targets.csv")
        rows = list(csv.DictReader(open(out)))
        assert [r["id"] for r in rows] == ["p_closeup_0", "p_closeup_2"], rows
        sc = yaml.safe_load(open(out + ".yaml"))
        assert (sc["type"], sc["purpose"], sc["plot_id"]) == ("camera_targets", "training_only", "p"), sc
        ct = sc["camera_targets"]
        assert (ct["camera_group"], ct["cameras"], ct["points"], ct["pixel"]) == ("closeup", 4, 2, "centre"), ct
        assert len(ct["missed"]) == 2 and ct["missed"][0].startswith("C1.jpg")

        # A set with labels is not overwritten
        rows[0]["label"] = "CS_ACR"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        try:
            cli.handle_cams2annos(args())
            raise AssertionError("overwrote a labelled set")
        except SystemExit as e:
            assert "labelled" in str(e), e
        cli.handle_cams2annos(args(tag="again"))
        assert os.path.exists(os.path.join("p.annos", "p_closeup_again_targets.csv"))

        try:
            cli.handle_cams2annos(args(group="gopro"))
            raise AssertionError("unknown group accepted")
        except SystemExit as e:
            assert "closeup" in str(e), e  # (the groups listed)
    print("OK")
    """
)


@unittest.skipUnless(
    importlib.util.find_spec("open3d") is not None and importlib.util.find_spec("cv2") is not None,
    "requires the open3d/substrata environment",
)
class TestCameraTargets(unittest.TestCase):
    def _run(self, script, args=()):
        result = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True,
                                cwd=str(_SRC), timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("OK", result.stdout)
        return result

    def test_camera_targets(self):
        self._run(_LIBRARY)

    def test_cams2annos_command(self):
        self._run(_CLI)

    def test_parser(self):
        result = subprocess.run([sys.executable, "-m", "substrata.cli", "cams2annos", "--help"],
                                capture_output=True, text=True, cwd=str(_SRC), timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        for opt in ("--group", "--pixel", "--search-radius", "--max-search-radius", "--max-distance", "--tag"):
            self.assertIn(opt, result.stdout)


if __name__ == "__main__":
    unittest.main()
