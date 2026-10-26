"""Tests for Cameras.from_image_dir / get_random_image_points (photos-only points).

``cameras.py`` imports open3d/cv2 at module load, so these tests are skipped
outside the conda ``substrata`` environment.
"""

# Standard Library
import os
import sys
import tempfile
import unittest
from unittest import mock

# Third-Party
import numpy as np
from PIL import Image

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import Annotations, cameras

    # Other test modules stub ``substrata.annotations`` in sys.modules; keep the
    # real one so cameras.py resolves it during these tests.
    _ANNOTATIONS_MODULE = sys.modules["substrata.annotations"]
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    Annotations = cameras = _ANNOTATIONS_MODULE = None

IMG_W, IMG_H = 400, 300


@unittest.skipUnless(cameras is not None, "requires the open3d/substrata environment")
class TestImagePoints(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(
            sys.modules, {"substrata.annotations": _ANNOTATIONS_MODULE}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        for name in ("b.jpg", "a.PNG"):
            Image.new("RGB", (IMG_W, IMG_H)).save(os.path.join(self.dir, name))
        with open(os.path.join(self.dir, "notes.txt"), "w") as f:
            f.write("not an image")

    def tearDown(self):
        self._tmp.cleanup()

    def test_from_image_dir(self):
        cams = cameras.Cameras.from_image_dir(self.dir)
        self.assertEqual(list(cams.data), ["a", "b"])
        for cam in cams.data.values():
            self.assertIsNone(cam.camera_transform)
            self.assertTrue(cam.missing_pose_from_meta)
            self.assertTrue(os.path.isabs(cam.filepath))

    def test_random_points_within_margin(self):
        cams = cameras.Cameras.from_image_dir(self.dir)
        pts = cams.get_random_image_points(50, margin=40, seed=0)
        self.assertEqual(len(pts), 100)
        for ann in pts:
            self.assertTrue(40 <= ann.image_match.x < IMG_W - 40)
            self.assertTrue(40 <= ann.image_match.y < IMG_H - 40)
            self.assertTrue(np.isnan(ann.coords).all())
        self.assertIn("a_0", pts)

    def test_stratified_one_point_per_cell(self):
        cam = cameras.Cameras.from_image_dir(self.dir).data["a"]
        pts = cam.get_random_image_points(12, stratified=True, seed=1)
        # 400x300 with 12 points -> 4 columns x 3 rows of 100x100 cells
        cells = {
            (int(a.image_match.x // 100), int(a.image_match.y // 100)) for a in pts
        }
        self.assertEqual(len(cells), 12)

    def test_seed_reproducible(self):
        cams = cameras.Cameras.from_image_dir(self.dir)
        xy1 = [(a.cam_x, a.cam_y) for a in cams.get_random_image_points(5, seed=3)]
        xy2 = [(a.cam_x, a.cam_y) for a in cams.get_random_image_points(5, seed=3)]
        self.assertEqual(xy1, xy2)

    def test_margin_too_large(self):
        cam = cameras.Cameras.from_image_dir(self.dir).data["a"]
        with self.assertRaises(ValueError):
            cam.get_random_image_points(5, margin=200)

    def test_save_roundtrip(self):
        cams = cameras.Cameras.from_image_dir(self.dir)
        pts = cams.get_random_image_points(3, seed=0)
        for ann in pts:
            ann.label = "coral"
        out = os.path.join(self.dir, "points.csv")
        pts.save(out, orig_coords_only=True)
        loaded = Annotations(out)
        self.assertEqual(len(loaded), 6)
        for ann_id, ann in pts.items():
            other = loaded[ann_id]
            self.assertEqual(other.cam_filepath, ann.cam_filepath)
            self.assertAlmostEqual(other.cam_x, ann.cam_x)
            self.assertAlmostEqual(other.cam_y, ann.cam_y)

    def test_from_annotation_images(self):
        cams = cameras.Cameras.from_image_dir(self.dir)
        pts = cams.get_random_image_points(3, seed=0)
        out = os.path.join(self.dir, "points.csv")
        pts.save(out, orig_coords_only=True)
        loaded = Annotations(out)
        reloaded_cams = cameras.Cameras.from_annotation_images(loaded)
        self.assertEqual(len(reloaded_cams.data), 2)  # one per image
        for ann_id, ann in pts.items():
            match = loaded[ann_id].image_match
            self.assertEqual(match.filepath, ann.cam_filepath)
            self.assertAlmostEqual(match.x, ann.cam_x)
            self.assertAlmostEqual(match.y, ann.cam_y)


if __name__ == "__main__":
    unittest.main()
