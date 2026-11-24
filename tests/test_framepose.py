"""substrata.framepose: poses for video frames Metashape left unaligned, from matches with posed photos.

As tests/test_imagesync.py: the parts need the real package (open3d, cv2, scipy), so they run in a fresh
interpreter, skipped outside the conda env. The matcher (kornia's DISK + LightGlue) is not exercised (its weights are
downloaded): everything after the matching is, on a synthetic reef. The real-data check is in the changelog.
"""

# Standard Library
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_HEAVY = all(importlib.util.find_spec(m) is not None for m in ("open3d", "cv2", "scipy"))

_SETUP = textwrap.dedent(
    """
    import datetime as dt
    import numpy as np
    import open3d as o3d
    from substrata import cameras as cm, imagesync as isy, framepose as fp
    from substrata.pointclouds import PointCloud

    def sensor(sid, w, h, f, **kw):
        params = {k: 0.0 for k in ("cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")}
        params.update(kw)
        return cm.Sensor(sid, f"s{sid}", {"width": w, "height": h}, dict(params, f=float(f)))

    def rot(ax, deg):
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler(ax, deg, degrees=True).as_matrix()

    S = 0.2  # (the world transform's scale, as a plot's)
    W = np.eye(4)
    W[:3, :3] = S * rot("z", 30)
    W[:3, 3] = [5.0, -2.0, -40.0]

    def camera(cams, name, R_cw, C, sens, when=None):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = S * R_cw, C
        c = cm.Camera(cams, name, T, np.asarray(C, float), path=f"/plot/p.x/{name}.jpg")
        c.reverse_transform_coords(W)
        c.sensor, c.enabled = sens, True
        if when is not None:
            c.datetime = when.strftime("%Y:%m:%d %H:%M:%S")
        cams.data[name] = c
        return c

    cams = cm.Cameras.__new__(cm.Cameras)
    cams.sensors, cams.data, cams.world_transform = {}, {}, W
    down = np.diag([1.0, -1.0, -1.0])  # (looking down -z)

    # The reef: z = -40 (world) with 5 cm bumps, 4 mm grid, 2 x 2 m
    g = np.mgrid[-250:250, -250:250].reshape(2, -1).T * 0.004
    z = -40.0 + 0.05 * np.sin(g[:, 0] * 9) * np.cos(g[:, 1] * 7)
    pc = PointCloud()
    pc.o3d_pcd.points = o3d.utility.Vector3dVector(np.c_[g, z])
    pc.world_transform = W
    gopro_s = sensor(2, 3840, 2160, 1750, k1=-0.23, k2=0.07, k3=-0.012)
    survey_s = sensor(3, 9568, 6376, 6460, k1=-0.05, k2=0.07)
    """
)

_GEOMETRY = _SETUP + textwrap.dedent(
    """
    # world_to_cam / pose_transform: a current-frame transform carries the scale, R and C come back
    R_cw = down @ rot("x", 10) @ rot("z", 25)
    T = fp.pose_transform(R_cw.T, [0.1, 0.2, -39.4], S)
    assert np.allclose(fp.world_to_cam(T), R_cw.T) and np.allclose(T[:3, 3], [0.1, 0.2, -39.4])
    assert abs(fp.rotation_deg(rot("z", 7), np.eye(3)) - 7) < 1e-9

    # roll_turns: a photo turned 90 / 180 deg about its view (looking the same way)
    a = fp.pose_transform(down.T, [0, 0, -39], S)
    for deg, k in ((0, 0), (90, 1), (180, 2), (-90, 3), (170, 2)):
        b = fp.pose_transform((down @ rot("z", deg)).T, [0.2, 0, -38.5], S)
        assert fp.roll_turns(a, b) == k, (deg, fp.roll_turns(a, b), k)

    # DepthLookup: a distorted wide lens's pixels lifted onto the bumpy reef, on their own rays
    cam = camera(cams, "G0", down @ rot("x", 15), [0.05, -0.1, -39.45], gopro_s)
    dl = fp.DepthLookup(gopro_s, cam.camera_transform, pc.points)
    px = np.array([(x, y) for x in np.linspace(600, 3200, 9) for y in np.linspace(300, 1800, 7)])
    P = dl.lift(px)
    ok = np.isfinite(P).all(1)
    assert ok.mean() > 0.9, ok.mean()
    zr = -40.0 + 0.05 * np.sin(P[ok, 0] * 9) * np.cos(P[ok, 1] * 7)
    assert np.max(np.abs(P[ok, 2] - zr)) < 0.006, np.max(np.abs(P[ok, 2] - zr))  # (on the reef: < 6 mm)
    back = np.array([cam.get_pixel_coords(p, required_to_be_in_view=False)[:2] for p in P[ok]], float)
    assert np.max(np.abs(back - px[ok])) <= 1.0, np.max(np.abs(back - px[ok]))  # (on the pixel's ray)

    # local_cloud: the points round centres, thinned
    loc = fp.local_cloud(pc, np.array([[0.0, 0.0, -40.0]]), radius=0.3, voxel=0.008)
    assert len(loc) and np.linalg.norm(loc[:, :2], axis=1).max() <= 0.3 + 1e-9 and len(loc) < 0.3 * len(pc.points)

    # neighbours / anchors / plausible
    t0 = dt.datetime(2025, 9, 28, 9, 0, 0)
    f1 = camera(cams, "F1", down, [0.0, 0.0, -39.4], gopro_s, t0)
    f2 = camera(cams, "F2", down, [0.6, 0.0, -39.4], gopro_s, t0 + dt.timedelta(seconds=10))
    f3 = camera(cams, "F3", down, [3.0, 0.0, -39.4], gopro_s, t0 + dt.timedelta(seconds=500))
    posed = isy.by_time([f1, f2, f3])
    nb = fp.neighbours(t0 + dt.timedelta(seconds=4), posed)
    assert [(d, c.cam_id) for d, c in nb] == [(-4.0, "F1"), (6.0, "F2")], nb
    assert fp.neighbours(t0 + dt.timedelta(seconds=4), posed, exclude=[f1])[0][1] is f2
    assert fp.neighbours(t0 + dt.timedelta(seconds=300), posed, max_gap=100) == []
    an = fp.anchors(nb, pc, step=0.2)
    assert len(an) >= 4 and abs(an[0][0]) < 0.01 and abs(an[-1][0] - 0.6) < 0.01, an  # (their aims and between)
    assert fp.plausible(dict(ok=True, C=np.array([0.3, 0, -39.4])), nb)[0]
    assert not fp.plausible(dict(ok=True, C=np.array([4.0, 0, -39.4])), nb)[0]

    # gap_frames: the unaligned seconds from a frame out to the posed ones (a side that reaches none, a missing
    # second first or past max_gap, adds none)
    sec = lambda k: t0 + dt.timedelta(seconds=k)  # noqa: E731
    by_t = {sec(k): k for k in list(range(0, 20)) + list(range(21, 30))}
    run = fp.gap_frames(sec(8), by_t, {sec(5), sec(12)})
    assert run == [sec(k) for k in (6, 7, 8, 9, 10, 11)], run
    assert fp.gap_frames(sec(18), by_t, {sec(5)}) == [sec(k) for k in range(6, 19)]  # (20 is missing)
    assert fp.gap_frames(sec(8), by_t, set(), max_gap=3) == [sec(8)]
    assert fp.gap_frames(sec(8), by_t, {sec(5)}, max_gap=2) == [sec(8)]

    # posed_copy: the copy has the pose, the camera itself none
    blank = camera(cams, "B", np.eye(3), [0, 0, 0], gopro_s)
    blank.missing_pose_from_meta = True
    c2 = fp.posed_copy(blank, down.T, [0.1, 0.0, -39.4], W)
    assert cm.can_project(c2, use_orig_coords=False) and not cm.can_project(blank, use_orig_coords=False)
    assert np.allclose(c2.coords, [0.1, 0.0, -39.4]) and c2.camera_transform is not blank.camera_transform

    # photo_sensors: an AUV's photos from 3.5 m show the reef in far less detail than survey photos from 1 m
    auv_s = sensor(5, 5472, 3648, 3700)
    survey = [camera(cams, f"S{i}", down.T, [0.1 * i, 0.0, -39.0], survey_s) for i in range(4)]
    auv = [camera(cams, f"A{i}", down.T, [0.1 * i, 0.1, -36.5], auv_s) for i in range(4)]
    keep, detail = fp.photo_sensors(survey + auv, pc)
    assert {id(c) for c in keep} == {id(c) for c in survey}, detail
    assert 700 < detail["3"] < 900 and detail["5"] < 0.4 * detail["3"], detail
    keep, _d = fp.photo_sensors(auv, pc)  # (the only photos: kept)
    assert len(keep) == 4
    # place() with nothing to index: no pose, no error
    poser = fp.FramePoser.__new__(fp.FramePoser)
    poser._image = lambda *a: np.zeros((8, 8))
    r = fp.FramePoser.place(poser, survey[0], survey_s, fp.LazyIndex([]))
    assert not r["ok"] and "0 tried" in r["reason"], r
    print("OK")
    """
)

_POSE = _SETUP + textwrap.dedent(
    """
    # A frame posed from its matches with two survey photos (and a third matched by mistake)
    R_true = down @ rot("x", 12) @ rot("y", -6) @ rot("z", 40)
    C_true = np.array([0.08, -0.05, -39.42])
    frame_T = fp.pose_transform(R_true.T, C_true, S)
    s1 = camera(cams, "S1", down @ rot("z", 90), [0.1, 0.1, -38.6], survey_s)
    s2 = camera(cams, "S2", down @ rot("x", 8), [-0.2, -0.1, -38.7], survey_s)
    s3 = camera(cams, "S3", down, [0.4, 0.3, -38.6], survey_s)
    rng = np.random.default_rng(3)
    local = fp.local_cloud(pc, np.array([[0.1, 0.0, -40.0]]), radius=1.2)
    # (the frame's pixels on the reef, seen by each photo: its pixels there)
    a_all = np.c_[rng.uniform(300, 3500, 600), rng.uniform(200, 1950, 600)]
    P_all = fp.DepthLookup(gopro_s, frame_T, local).lift(a_all)
    per = []
    for cam, wrong in ((s1, False), (s2, False), (s3, True)):
        x, y, _d, _r, ok = cam.project_points(np.nan_to_num(P_all))
        ok &= np.isfinite(P_all).all(1)
        a, b = a_all[ok], np.c_[x, y][ok]
        if wrong:
            a = a[rng.permutation(len(a))]  # (matches that mean nothing)
        else:
            a[:30] += rng.uniform(-300, 300, (30, 2))  # (some wrong matches)
        got = fp.source_pose(gopro_s, cam, a, b, local)
        if wrong:
            assert got is None or got["inliers"] < 40, got and got["inliers"]
        else:
            assert got is not None and got["inliers"] > 100, got and got["inliers"]
        if got:
            per.append(got)
    # a frame posed earlier in a chain (more inliers, a pose 20 cm off): not the reference while photos agree
    from types import SimpleNamespace
    bad = dict(per[0], cam=SimpleNamespace(filename="CH.jpg", framepose_chained=True), inliers=5000,
               C=per[0]["C"] + [0.2, 0.0, 0.0], obj=per[0]["obj"] + [0.2, 0.0, 0.0])
    r = fp.pool(gopro_s, per + [bad], scale=S, pcd=pc)
    assert r["ok"] and r["anchored"], r
    assert set(r["pooled"]) == {"S1.jpg", "S2.jpg"}, r["pooled"]
    assert np.linalg.norm(r["C"] - C_true) < 0.01, (r["C"], C_true)
    assert fp.rotation_deg(r["R"], R_true.T) < 0.3
    assert r["aim"] is not None and 0.3 < r["range"] < 0.8, r["range"]
    # the pose as imagesync.set_pose writes it: the frame sees the reef as the true one does
    f = camera(cams, "F", np.eye(3), [0, 0, 0], gopro_s)
    isy.set_pose(f, r["R"], r["C"], W)
    X = r["aim"]
    want = cm.camera_points_to_pixels(*((X - C_true) @ R_true.T.T), gopro_s)[:2]
    x, y, *_ = f.get_pixel_coords(X)
    assert abs(x - want[0]) < 15 and abs(y - want[1]) < 15, ((x, y), want)
    print("OK")
    """
)


@unittest.skipUnless(_HEAVY, "needs open3d, cv2 and scipy (the substrata conda env)")
class TestFramePose(unittest.TestCase):
    def _run(self, script):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(_SRC), timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("OK", r.stdout)

    def test_geometry(self):
        self._run(_GEOMETRY)

    def test_pose_from_sources(self):
        self._run(_POSE)


if __name__ == "__main__":
    unittest.main()
