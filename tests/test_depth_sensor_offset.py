"""Tests for the depth-sensor lever-arm correction (sensor offset in camera frame).

``cameras.py`` imports open3d/cv2 at module load, so these tests are skipped
outside the conda ``substrata`` environment.
"""

# Standard Library
import unittest

# Third-Party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import cameras, geom
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    cameras = geom = None


def _camera_transform(coords, tilt_deg=0.0, scale=1.0):
    """Camera-to-world transform looking down (-z), tilted about the world x-axis.

    Camera frame: x image right, y image down, z view direction.
    """
    t = np.radians(tilt_deg)
    view = np.array([0.0, np.sin(t), -np.cos(t)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(view, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, down, view]) * scale
    transform[:3, 3] = coords
    return transform


@unittest.skipUnless(geom is not None, "requires the open3d/substrata environment")
class TestOffsetInCameraFrame(unittest.TestCase):
    def test_behind_down_looking_camera_is_above(self):
        c = np.array([1.0, 2.0, 3.0])
        p = geom.offset_in_camera_frame(c, _camera_transform(c), [0, 0, -0.15])
        np.testing.assert_allclose(p, [1.0, 2.0, 3.15])

    def test_tilted_camera_vertical_component(self):
        c = np.zeros(3)
        p = geom.offset_in_camera_frame(c, _camera_transform(c, 28.0), [0, 0, -0.15])
        self.assertAlmostEqual(p[2], 0.15 * np.cos(np.radians(28.0)))

    def test_scaled_transform_is_normalised(self):
        c = np.zeros(3)
        p = geom.offset_in_camera_frame(
            c, _camera_transform(c, scale=0.06), [0, 0, -0.15]
        )
        self.assertAlmostEqual(p[2], 0.15)


@unittest.skipUnless(cameras is not None, "requires the open3d/substrata environment")
class TestLeverArmRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.offset_m = np.array([0.0, 0.0, -0.15])  # sensor 15 cm behind camera
        self.scale = 0.05  # metres per coordinate unit
        self.depth_offset = -40.0
        self.cams = cameras.Cameras()
        self.cams.data = {}
        for i in range(200):
            xyz = np.r_[rng.uniform(0, 30, 2), rng.uniform(-1, 1)] / self.scale
            tilt = rng.uniform(10, 40)
            cam = cameras.Camera(
                self.cams, str(i), _camera_transform(xyz, tilt), xyz, f"c{i}.jpg"
            )
            sensor = cam.sensor_coords(self.offset_m / self.scale)
            # depth field: z is up, one coordinate unit = `scale` metres
            cam.depth_sensor_m = self.depth_offset + sensor[2] * self.scale
            self.cams.data[str(i)] = cam

    def test_sensor_offset_recovers_depth_field(self):
        up, depth_offset, per_unit, _, rmse, *_ = (
            self.cams.get_up_vector_from_camera_depths(
                sensor_offset=self.offset_m, scale_factor=self.scale
            )
        )
        np.testing.assert_allclose(up / np.linalg.norm(up), [0, 0, 1], atol=1e-9)
        self.assertAlmostEqual(depth_offset, self.depth_offset, places=6)
        self.assertAlmostEqual(per_unit, self.scale, places=9)
        self.assertLess(rmse, 1e-9)

    def test_without_offset_is_biased(self):
        _, depth_offset, _, _, rmse, *_ = (
            self.cams.get_up_vector_from_camera_depths()
        )
        # Camera centres sit below the sensor, so the fitted field is too shallow
        self.assertGreater(depth_offset - self.depth_offset, 0.1)
        self.assertGreater(rmse, 1e-4)

    def test_zero_offset_matches_default(self):
        default = self.cams.get_up_vector_from_camera_depths()
        zero = self.cams.get_up_vector_from_camera_depths(
            sensor_offset=(0, 0, 0), scale_factor=self.scale
        )
        np.testing.assert_allclose(default[0], zero[0])
        self.assertEqual(default[1], zero[1])


@unittest.skipUnless(cameras is not None, "requires the open3d/substrata environment")
class TestWaterLevel(unittest.TestCase):
    def _ff(self):
        import pandas as pd
        from substrata.firefish import FireFish

        ff = FireFish.__new__(FireFish)
        ff.data = pd.DataFrame({"unixtime": [0, 1], "depth": [-40.0, -39.5]})
        return ff

    def test_low_tide_makes_depths_deeper_relative_to_msl(self):
        ff = self._ff()
        ff.apply_water_level(-0.32)
        np.testing.assert_allclose(ff.data["depth"], [-40.32, -39.82])

    def test_zero_is_noop(self):
        ff = self._ff()
        ff.apply_water_level(0.0)
        np.testing.assert_allclose(ff.data["depth"], [-40.0, -39.5])


if __name__ == "__main__":
    unittest.main()
