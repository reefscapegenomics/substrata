# Standard Library
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

# Third-Party Libraries
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_cameras_module():
    """Load ``substrata.cameras`` without importing package ``__init__``."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    def _load(mod_name: str, rel: str) -> types.ModuleType:
        path = _SUB / rel
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {mod_name} from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    if "substrata.settings" not in sys.modules:
        _load("substrata.settings", "settings.py")
    if "substrata.geom" not in sys.modules:
        _load("substrata.geom", "geom.py")

    if "substrata.visualizations" not in sys.modules:
        sys.modules["substrata.visualizations"] = types.ModuleType(
            "substrata.visualizations"
        )
    if "substrata.measurements" not in sys.modules:
        sys.modules["substrata.measurements"] = types.ModuleType(
            "substrata.measurements"
        )

    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        log_mod.tqdm_joblib = tqdm_joblib
        sys.modules["substrata.logging"] = log_mod

    if "substrata.cameras" not in sys.modules:
        _load("substrata.cameras", "cameras.py")

    return sys.modules["substrata.cameras"]


class TestSpatialTimeOffset(unittest.TestCase):
    def test_median_k_from_nearest_neighbors(self) -> None:
        cm = _load_cameras_module()
        Cameras = cm.Cameras
        Camera = cm.Camera
        spatial_nearest_time_offset_report = cm.spatial_nearest_time_offset_report

        pose = Cameras()
        tgt = Cameras()
        eye = np.eye(4).tolist()
        cp = Camera(pose, "p0", eye, [0.0, 0.0, 0.0], "/tmp/a.jpg")
        cp.datetime = "2024:06:01 12:00:00"
        pose.data["p0"] = cp

        ct = Camera(tgt, "t0", eye, [0.05, 0.0, 0.0], "/tmp/b.jpg")
        ct.datetime = "2024:06:01 12:00:30"
        tgt.data["t0"] = ct

        rep = spatial_nearest_time_offset_report(
            tgt,
            pose,
            spatial_max_dist_m=0.5,
            min_pairs=1,
            scale_factor=1.0,
        )
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["n_inliers"], 1)
        # k = t_pose - t_target = -30 seconds for this clock setup
        self.assertAlmostEqual(rep["median_k_sec"], -30.0, places=3)


class TestXyzOffset(unittest.TestCase):
    def test_median_xyz_identity_rotation(self) -> None:
        cm = _load_cameras_module()
        Cameras = cm.Cameras
        Camera = cm.Camera
        xyz_offset_datetime_matches_report = cm.xyz_offset_datetime_matches_report

        pose = Cameras()
        tgt = Cameras()
        eye = np.eye(4).tolist()
        dt = "2024:06:01 12:00:00"
        cp = Camera(pose, "p0", eye, [0.0, 0.1, 0.0], "/tmp/a.jpg")
        cp.datetime = dt
        pose.data["p0"] = cp

        ct = Camera(tgt, "t0", eye, [0.0, 0.0, 0.0], "/tmp/b.jpg")
        ct.datetime = dt
        tgt.data["t0"] = ct

        rep = xyz_offset_datetime_matches_report(
            tgt, pose, scale_factor=1.0
        )
        self.assertTrue(rep["ok"])
        self.assertEqual(len(rep["unmatched_ids"]), 0)
        self.assertAlmostEqual(rep["median_xyz"][0], 0.0, places=6)
        self.assertAlmostEqual(rep["median_xyz"][1], 0.1, places=6)
        self.assertAlmostEqual(rep["median_xyz"][2], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
