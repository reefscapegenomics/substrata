"""Tests for Camera.group / Cameras.group_names (cached filename prefixes).

``cameras.py`` imports open3d/cv2 at module load, so these tests are skipped
outside the conda ``substrata`` environment.
"""

# Standard Library
import unittest

# Third-Party
import numpy as np

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    from substrata import cameras
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    cameras = None


def _cams(paths):
    cams = cameras.Cameras()
    for i, path in enumerate(paths):
        cams.data[str(i)] = cameras.Camera(
            cams, str(i), np.eye(4).tolist(), [0.0, 0.0, 0.0], path
        )
    return cams


@unittest.skipUnless(cameras is not None, "requires the open3d/substrata environment")
class TestCameraGroups(unittest.TestCase):
    def test_single_prefix_uses_folder(self):
        cams = _cams(["/a/run1.auv/PR_001.png", "/a/run2/PR_002.png"])
        self.assertEqual(cams.group_names, ["auv", "run2"])

    def test_multiple_prefixes_split_groups(self):
        cams = _cams(["/a/run1/PR_001.png", "/a/run1/FC_001.png"])
        self.assertEqual(cams.group_names, ["run1_FC", "run1_PR"])

    def test_cache_follows_changes(self):
        cams = _cams(["/a/run1/PR_001.png", "/a/run1/PR_002.png"])
        self.assertEqual(cams.group_names, ["run1"])
        cams.data["1"].set_filepath("/a/run1/FC_002.png")
        self.assertEqual(cams.group_names, ["run1_FC", "run1_PR"])
        del cams.data["1"]
        self.assertEqual(cams.group_names, ["run1"])


if __name__ == "__main__":
    unittest.main()
