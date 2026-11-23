"""substrata.videopose: the lens model (pixel <-> ray, where it is valid), a calibration and poses recovered from
synthetic correspondences (a known SuperView-like lens, cameras 1.5 m above a bumpy reef, noise and outliers), the
stored pose, a GoPro's metadata. The module is loaded from its file (numpy / scipy / cv2 only)."""

# Standard Library
import importlib.util
import os
import struct
import tempfile
import unittest
from pathlib import Path

# Third-Party Libraries
import numpy as np

_FILE = Path(__file__).resolve().parents[1] / "src" / "substrata" / "videopose.py"
_spec = importlib.util.spec_from_file_location("substrata._videopose_under_test", _FILE)
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)

LENS = np.array([980.0, 0.94, 958.0, 528.0, -0.3, 0.12, -0.14, -0.08])  # (as fitted on SAV03)
SIZE = (1920, 1080)


def _look_at(C, aim, roll=0.0):
    """world -> camera rotation for a camera at C looking at aim (image up roughly world +y, turned by roll)."""
    z = (aim - C) / np.linalg.norm(aim - C)
    up = np.array([np.sin(roll), np.cos(roll), 0.0])
    x = np.cross(-up, z)  # (x right with y down)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _frames(n=16, seed=0, noise=0.7, outliers=0.2):
    rng = np.random.default_rng(seed)
    out, truth = [], []
    for i in range(n):
        g = rng.uniform(-1, 1, (4000, 2)) * 2.2 + rng.uniform(-5, 5, 2)
        X = np.c_[g, 0.15 * np.sin(3 * g[:, 0]) * np.cos(2 * g[:, 1])]  # a bumpy reef
        aim = np.r_[g.mean(0), 0.0]
        C = aim + np.r_[rng.uniform(-0.4, 0.4, 2), rng.uniform(1.3, 1.9)]
        R = _look_at(C, aim, roll=rng.uniform(-np.pi, np.pi))
        uv = vp.project(LENS, (X - C) @ R.T)
        ok = np.isfinite(uv).all(1) & (uv[:, 0] > 0) & (uv[:, 0] < SIZE[0]) & (uv[:, 1] > 0) & (uv[:, 1] < SIZE[1])
        uv, X = uv[ok][:600], X[ok][:600]
        uv = uv + rng.normal(0, noise, uv.shape)
        bad = rng.random(len(uv)) < outliers
        uv[bad] = rng.uniform([0, 0], SIZE, (bad.sum(), 2))
        out.append({"uv": uv, "xyz": X, "size": np.array(SIZE)})
        truth.append((R, C))
    return out, truth


class TestLens(unittest.TestCase):
    def test_round_trip(self):
        uv = np.stack(np.meshgrid(np.linspace(50, 1870, 12), np.linspace(40, 1040, 8)), -1).reshape(-1, 2)
        back = vp.project(LENS, vp.rays(LENS, uv))
        self.assertLess(np.nanmax(np.abs(back - uv)), 1e-3)

    def test_beyond_the_lens_is_nan(self):
        a = vp.max_angle(LENS, SIZE)
        self.assertTrue(30 < a < 80)
        far = np.array([[np.sin(np.radians(a + 25)), 0, np.cos(np.radians(a + 25))], [0, 0, -1.0]])
        self.assertTrue(np.isnan(vp.project(LENS, far)).all())

    def test_fov(self):
        f = vp.fov(LENS, SIZE)
        self.assertTrue(f["across"] > f["down"] > 0 and f["diagonal"] > f["across"])
        self.assertAlmostEqual(f["middle_v"], np.degrees(2 * np.arctan(540 / (980 * 0.94))), places=6)


class TestCalibration(unittest.TestCase):
    def test_recovers_lens_and_poses(self):
        frames, truth = _frames()
        cal = vp.calibrate(frames, SIZE)
        self.assertIsNotNone(cal)
        p = np.array(cal["params"])
        self.assertLess(abs(p[0] - LENS[0]) / LENS[0], 0.03)
        self.assertLess(cal["px"], 1.5)
        # the lens agrees with the truth over the frame's middle (what is used)
        uv = np.stack(np.meshgrid(np.linspace(300, 1620, 9), np.linspace(150, 930, 7)), -1).reshape(-1, 2)
        ang = np.degrees(np.arccos(np.clip((vp.rays(p, uv) * vp.rays(LENS, uv)).sum(1), -1, 1)))
        self.assertLess(np.max(ang), 0.5)
        for fr, (R, C) in zip(frames[:5], truth[:5]):
            fp = vp.frame_pose(p, fr["uv"], fr["xyz"])
            self.assertIsNotNone(fp)
            self.assertLess(np.linalg.norm(fp["C"] - C), 0.05)
            rec = vp.pose_record(fp)
            _x, _y, z = vp.camera_axes(rec["q"])
            self.assertLess(np.degrees(np.arccos(np.clip(z @ R[2], -1, 1))), 1.0)  # (forward: R's third row)

    def test_too_few_frames(self):
        frames, _ = _frames(n=4)
        self.assertIsNone(vp.calibrate(frames, SIZE))

    def test_stop_pose(self):
        self.assertIsNone(vp.stop_pose([None]))
        self.assertEqual(vp.stop_pose([{"n": 3}, {"n": 9}, None])["n"], 9)


class TestMetadata(unittest.TestCase):
    def test_gopro_lens(self):
        def klv(key, t, data):
            sz = {"c": 1, "f": 4}[t]
            return key + t.encode() + bytes([sz]) + struct.pack(">H", len(data) // sz) + data
        blob = b"\0" * 100 + klv(b"DVNM", "c", b"HERO13 Black") + klv(b"VFOV", "c", b"S") + klv(b"ZFOV", "f", struct.pack(">f", 146.8))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "GX01.MP4")
            with open(path, "wb") as f:
                f.write(blob)
            lens = vp.video_lens(path)
        self.assertEqual(lens, {"model": "HERO13 Black", "lens": "S", "zfov": 146.8})
        self.assertEqual(vp.lens_key({"width": 1920, "height": 1080}, lens), "HERO13-Black_S_1920x1080")
        self.assertEqual(vp.lens_key({"width": 640, "height": 480}, {}), "camera_lens_640x480")


if __name__ == "__main__":
    unittest.main()
