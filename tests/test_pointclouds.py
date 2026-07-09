# Standard Library
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

# Third-Party Libraries
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_pointclouds():
    """Load ``pointclouds.py`` without importing the ``substrata`` package.

    open3d/matplotlib are real imports here; the heavy sibling modules
    (annotations, visualizations) are stubbed. ``visualizations.show`` is a
    recording stub so the ``preview=True`` path can be asserted without opening
    a browser.
    """
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    def _load(mod_name: str, rel: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(mod_name, _SUB / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    if "substrata.settings" not in sys.modules:
        _load("substrata.settings", "settings.py")
    if "substrata.geom" not in sys.modules:
        _load("substrata.geom", "geom.py")

    if "substrata.annotations" not in sys.modules:
        sys.modules["substrata.annotations"] = types.ModuleType(
            "substrata.annotations"
        )

    if "substrata.visualizations" not in sys.modules:
        viz = types.ModuleType("substrata.visualizations")
        viz.plot_2d_calls = []

        def _plot_2d(pcd, highlight_coords=None, **kwargs):
            viz.plot_2d_calls.append(
                {"pcd": pcd, "highlight_coords": highlight_coords}
            )
            return "FIG"

        viz.plot_2d = _plot_2d
        sys.modules["substrata.visualizations"] = viz

    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")
        log_mod.logger = logging.getLogger("substrata-test")
        sys.modules["substrata.logging"] = log_mod

    if "substrata.pointclouds" not in sys.modules:
        _load("substrata.pointclouds", "pointclouds.py")
    return sys.modules["substrata.pointclouds"]


pcmod = _load_pointclouds()
import open3d as o3d  # noqa: E402  (real open3d, as in the sibling tests)


def _make_pc(points, colors=None):
    """Build a PointCloud from raw arrays via the codebase's o3d idiom."""
    pc = pcmod.PointCloud()
    pc.o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    if colors is not None:
        pc.o3d_pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    return pc


def _dense_with_strays():
    """A spread-out dense XY patch (occupies many grid cells) + far strays.

    The dense patch is 5000 points uniformly filling a 10 x 10 m footprint, so
    most occupied cells are dense and the median-occupied threshold reflects the
    body; the 5 far strays each land in their own single-point cell.
    """
    rng = np.random.RandomState(0)
    n_blob = 5000
    blob = np.empty((n_blob, 3))
    blob[:, :2] = rng.uniform(0.0, 10.0, size=(n_blob, 2))
    blob[:, 2] = rng.uniform(0.0, 0.5, size=n_blob)
    strays = np.array(
        [[100, 100, 0], [-90, 20, 0], [40, -80, 0], [110, -110, 0], [-70, 90, 0]],
        dtype=float,
    )
    pts = np.vstack([blob, strays])
    colors = rng.rand(len(pts), 3)
    return pts, colors, n_blob, len(strays)


class TestRemoveStrayXYPoints(unittest.TestCase):
    def test_removes_strays_and_preserves_colors(self):
        pts, colors, n_blob, n_stray = _dense_with_strays()
        pc = _make_pc(pts, colors)

        pc.remove_stray_xy_points(density_frac=0.1)

        remaining = pc.points
        # The far strays must be gone; kept points stay in the dense patch.
        self.assertLess(len(remaining), len(pts))
        self.assertTrue(
            np.all(np.abs(remaining[:, :2]) < 20.0), "strays not removed"
        )
        # Colors are carried over and stay aligned with points.
        self.assertEqual(len(pc.colors), len(remaining))
        # The dense body is preserved (nearly) intact.
        self.assertGreaterEqual(len(remaining), int(n_blob * 0.98))

    def test_preview_does_not_mutate(self):
        pts, colors, _, _ = _dense_with_strays()
        pc = _make_pc(pts, colors)
        viz = sys.modules["substrata.visualizations"]
        viz.plot_2d_calls.clear()

        ret = pc.remove_stray_xy_points(density_frac=0.1, preview=True)

        self.assertEqual(ret, "FIG")
        self.assertEqual(len(pc.points), len(pts), "preview must not drop points")
        self.assertEqual(len(viz.plot_2d_calls), 1)
        hl = viz.plot_2d_calls[0]["highlight_coords"]
        self.assertIsNotNone(hl)
        # Highlighted (removed) points are the far strays.
        self.assertTrue(np.all(np.abs(np.asarray(hl)[:, :2]) > 20.0))

    def test_density_frac_zero_is_noop(self):
        pts, colors, _, _ = _dense_with_strays()
        pc = _make_pc(pts, colors)
        pc.remove_stray_xy_points(density_frac=0.0)
        self.assertEqual(len(pc.points), len(pts))

    def test_invalid_density_frac_raises(self):
        pc = _make_pc(np.random.RandomState(1).rand(50, 3))
        for bad in (-0.1, 1.0, 2.0):
            with self.assertRaises(ValueError):
                pc.remove_stray_xy_points(density_frac=bad)

    def test_empty_cloud_is_noop(self):
        pc = pcmod.PointCloud()
        pc.remove_stray_xy_points()  # should not raise
        self.assertEqual(len(pc.points), 0)


if __name__ == "__main__":
    unittest.main()
