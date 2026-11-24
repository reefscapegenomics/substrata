"""The lens model against Metashape itself.

``data/metashape_lens.json`` holds Agisoft Metashape 2.2's own ``Calibration.project`` (camera-frame point ->
pixel) and ``Calibration.unproject`` (pixel -> normalised direction) for real Samoa calibrations (the AUV camera,
the two diver cameras of sa3_20m and sa4_60m, a video camera) and synthetic ones that set every term (K1-K4, P1-P4,
B1, B2), made with Metashape's Python API (see its ``source``). substrata's projection (``cameras.camera_points_to_pixels``, as
``get_pixel_coords``, ``project_points`` and ``pixel_hits`` use it) must give the same pixels, its inverse
(``pixels_to_normalized``, ``Camera.pixel_to_ray``) the same directions, and the cams.xml reader every term.

Until 0.2 substrata put P1 and P2 the other way round (OpenCV's order) and left out K4 and B2: up to ~20-100 px
near the image edges on the Samoa sensors (546 px on sa3_20m's PGRAM02).
"""

# Standard Library
import importlib
import json
import os
import tempfile
import unittest

# Third-Party Libraries
import numpy as np

try:  # Heavy deps (cv2, …) are only present in the conda envs.
    cm = importlib.import_module("substrata.cameras")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    cm = None

DATA = os.path.join(os.path.dirname(__file__), "data", "metashape_lens.json")
TERMS = ("f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")


def _fixture():
    with open(DATA) as fh:
        return json.load(fh)["sensors"]


def _sensor(cal):
    params = {k: float(cal.get(k, 0.0)) for k in TERMS}
    return cm.Sensor(0, cal["name"], {"width": cal["width"], "height": cal["height"]}, params)


def _cams_xml(cal) -> str:
    """A minimal Metashape cams.xml with this calibration as sensor 0 (only the terms it has, as Metashape
    writes them) and one camera at the origin, looking along +z."""
    terms = "".join(f"<{k}>{cal[k]!r}</{k}>" for k in TERMS if k in cal)
    return (
        '<?xml version="1.0" encoding="UTF-8"?><document version="2.0.0"><chunk><sensors>'
        f'<sensor id="0" label="s" type="frame"><resolution width="{cal["width"]}" height="{cal["height"]}"/>'
        f'<calibration type="frame" class="adjusted"><resolution width="{cal["width"]}" '
        f'height="{cal["height"]}"/>{terms}</calibration></sensor></sensors>'
        '<cameras><camera id="0" sensor_id="0" label="IMG_0"><transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
        "</transform></camera></cameras></chunk></document>"
    )


@unittest.skipIf(cm is None, "substrata.cameras not importable")
class TestLensMetashape(unittest.TestCase):
    def test_projection_matches_metashape(self):
        for entry in _fixture():
            cal, ref = entry["calibration"], np.array(entry["project"])
            with self.subTest(sensor=cal["name"]):
                s = _sensor(cal)
                u, v, _ = cm.camera_points_to_pixels(ref[:, 0], ref[:, 1], ref[:, 2], s)
                err = np.hypot(u - ref[:, 3], v - ref[:, 4])
                self.assertLess(err.max(), 1e-6, f"{cal['name']}: up to {err.max():.3g} px from Metashape")

    def test_inverse_matches_metashape(self):
        """pixels_to_normalized: the exact inverse of the projection (back through it: the pixel again, to 1e-6 px), the
        same direction as Metashape's unproject wherever that is exact itself (Metashape's unproject is an
        approximation: on sa3_20m's PGRAM02 its direction projects up to ~9 px from the pixel), and NaN only
        where there is no direction (the far corners of the strongly distorted video frames, where Metashape's
        fails too)."""
        for entry in _fixture():
            cal, ref = entry["calibration"], np.array(entry["unproject"])
            with self.subTest(sensor=cal["name"]):
                s = _sensor(cal)
                x, y = cm.pixels_to_normalized(ref[:, 0], ref[:, 1], s)
                ms_u, ms_v, _ = cm.camera_points_to_pixels(ref[:, 2], ref[:, 3], np.ones(len(ref)), s)
                ms_exact = np.hypot(ms_u - ref[:, 0], ms_v - ref[:, 1]) < 1e-6
                found = np.isfinite(x)
                self.assertTrue((found | ~ms_exact).all(), "no direction where Metashape has one")
                self.assertGreater(found.mean(), 0.9)
                u, v, _ = cm.camera_points_to_pixels(x[found], y[found], np.ones(found.sum()), s)  # (back: the pixel again)
                self.assertLess(np.hypot(u - ref[found, 0], v - ref[found, 1]).max(), 1e-6)
                both = found & ms_exact
                self.assertGreater(both.sum(), 0)
                err_px = np.hypot(x[both] - ref[both, 2], y[both] - ref[both, 3]) * s.f
                self.assertLess(err_px.max(), 1e-5, f"{cal['name']}: {err_px.max():.3g} px")

    def test_fold_and_portrait(self):
        """In view: within the radius before the lens folds back (sa3_20m's PGRAM02 folds at r 0.878, inside
        its image's width; a point beyond would land in the image again), and the longer image side bounds a
        portrait photo (the video frames' top and bottom rows are in view)."""
        for entry in _fixture():
            cal, ref = entry["calibration"], np.array(entry["project"])
            with self.subTest(sensor=cal["name"]):
                s = _sensor(cal)
                inside = ((ref[:, 3] > 0) & (ref[:, 3] < cal["width"]) & (ref[:, 4] > 0)
                          & (ref[:, 4] < cal["height"]))
                _u, _v, ok = cm.camera_points_to_pixels(ref[:, 0], ref[:, 1], ref[:, 2], s)
                self.assertTrue(ok[inside].all())
        pgram02 = _sensor(next(e["calibration"] for e in _fixture() if "sa3_20m PGRAM02" in e["calibration"]["name"]))
        self.assertAlmostEqual(np.sqrt(pgram02.r2_max), 0.878, places=3)
        # beyond the fold: a point ~50 degrees off the axis lands at (4784, 5236), inside the image; the
        # earlier bound (1.005 image widths: r < 1.338) took it as in view
        u, v, ok = cm.camera_points_to_pixels(np.array([0.0]), np.array([1.206]), np.ones(1), pgram02)
        self.assertTrue(0 < u[0] < pgram02.width and 0 < v[0] < pgram02.height)
        self.assertLess(1.206 ** 2 * pgram02.fx ** 2, 1.01 * pgram02.width ** 2)
        self.assertFalse(ok[0])

    def test_cams_xml_reader_and_camera_paths(self):
        """Every term read from cams.xml; get_pixel_coords, project_points, pixel_hits and pixel_to_ray through
        a camera at the origin give Metashape's pixels and directions."""
        for entry in _fixture():
            cal, ref = entry["calibration"], np.array(entry["project"])
            with self.subTest(sensor=cal["name"]):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "p.cams.xml")
                    with open(path, "w") as fh:
                        fh.write(_cams_xml(cal))
                    cams = cm.Cameras.__new__(cm.Cameras)
                    cams.sensors, cams.data = {}, {}
                    cams.get_cam_sensor_parameters_from_file(path)
                s = cams.sensors[0]
                for k in TERMS[3:]:
                    self.assertEqual(getattr(s, k), float(cal.get(k, 0.0)), k)
                cam = cm.Camera(cam_id="0", camera_transform=np.eye(4), coords=np.zeros(3), path="/p/IMG_0.jpg")
                cam.orig_camera_transform, cam.orig_coords = np.eye(4), np.zeros(3)
                cam.sensor = s
                inside = ((ref[:, 3] > 0) & (ref[:, 3] < cal["width"]) & (ref[:, 4] > 0)
                          & (ref[:, 4] < cal["height"]))
                pts = ref[inside, :3]
                px, py, depth, _rel, ok = cam.project_points(pts)
                self.assertTrue(ok.all())
                np.testing.assert_allclose(px, np.rint(ref[inside, 3]))
                np.testing.assert_allclose(py, np.rint(ref[inside, 4]))
                for p, (u, v) in zip(pts[:20], ref[inside, 3:5][:20]):
                    x, y, *_ = cam.get_pixel_coords(p)
                    self.assertEqual((x, y), (int(round(u)), int(round(v))))
                    hit = cm.pixel_hits([cam], p, use_orig_coords=True, enabled_only=False)
                    self.assertEqual(hit[0][1:3], (int(round(u)), int(round(v))))
                    d = np.asarray(cam.pixel_to_ray(u, v), float)
                    np.testing.assert_allclose(d, p / np.linalg.norm(p), atol=1e-9)

    def test_tangential_terms_metashape_order(self):
        """P1 goes with x, P2 with y (OpenCV's p2 and p1): a P1-only lens moves points along x only on both
        axes (x' gains P1 (r2 + 2 x^2), y' gains 2 P1 x y = 0 there); OpenCV's order would move the point on
        the y axis along y."""
        s = _sensor(dict(name="p1", width=2000, height=2000, f=1000.0, p1=1e-3))
        u, v, _ = cm.camera_points_to_pixels(np.array([0.5, 0.0]), np.array([0.0, 0.5]), np.ones(2), s)
        self.assertAlmostEqual(u[0], 1500 + 1000 * 1e-3 * 0.75)  # (r2 + 2 x^2 = 0.75 at x = 0.5)
        self.assertAlmostEqual(v[0], 1000.0)
        self.assertAlmostEqual(u[1], 1000 + 1000 * 1e-3 * 0.25)  # (r2 = 0.25 at y = 0.5)
        self.assertAlmostEqual(v[1], 1500.0)


if __name__ == "__main__":
    unittest.main()
