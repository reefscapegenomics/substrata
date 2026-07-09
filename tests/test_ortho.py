"""Tests for :mod:`substrata.ortho` — ``OrthoMap`` and ``OrthoGrid``.

``ortho.py`` has no open3d/cv2 dependency, so it is loaded directly from
``src/substrata`` via ``importlib`` against a bare ``substrata`` package shell
(avoiding ``substrata/__init__.py``'s heavy star-imports). Only numpy, Pillow
and matplotlib (Agg) are needed.
"""

# Standard Library
import importlib.util
import sys
import types
import unittest
from pathlib import Path

# Third-Party
import matplotlib
import numpy as np

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_ortho():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
    if "substrata.ortho" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "substrata.ortho", _SUB / "ortho.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["substrata.ortho"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["substrata.ortho"]


ortho = _load_ortho()


class _PC:
    def __init__(self, points, colors=None):
        self.points = np.asarray(points, dtype=float)
        self.colors = (
            np.ones((len(self.points), 3), dtype=float)
            if colors is None else np.asarray(colors, dtype=float)
        )


class _Ann:
    def __init__(self, coords, label=None):
        self.coords = np.asarray(coords, dtype=float)
        self.label = label


class _Container:
    def __init__(self, items):
        self.data = {i: it for i, it in enumerate(items)}


def _ramp_pc(n=40, span=4.0):
    """A dense square cloud on [0, span]^2 with z == x (a planar ramp)."""
    g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
    pts = np.column_stack([g[:, 0], g[:, 1], g[:, 0]])
    return _PC(pts)


class TestOrthoGridDEM(unittest.TestCase):
    def setUp(self):
        self.pc = _ramp_pc()

    def test_lattice_dimensions(self):
        og = ortho.OrthoGrid(pcd=self.pc, value_by="z", cell_size=1.0)
        self.assertEqual((og.nx, og.ny), (4, 4))
        self.assertEqual(og.values.shape, (4, 4))

    def test_mean_z(self):
        og = ortho.OrthoGrid(pcd=self.pc, value_by="z", agg="mean", cell_size=1.0)
        # Cell x in [0,1): z==x mean ~ 0.45 (points at 0,0.1,...,0.9).
        self.assertAlmostEqual(og.value_at(0.5, 0.5)["value"], 0.45, places=2)
        self.assertAlmostEqual(og.value_at(3.5, 0.5)["value"], 3.45, places=2)

    def test_max_and_min_z(self):
        omax = ortho.OrthoGrid(pcd=self.pc, value_by="z", agg="max", cell_size=1.0)
        omin = ortho.OrthoGrid(pcd=self.pc, value_by="z", agg="min", cell_size=1.0)
        self.assertAlmostEqual(omax.value_at(0.5, 0.5)["value"], 0.9, places=2)
        self.assertAlmostEqual(omin.value_at(0.5, 0.5)["value"], 0.0, places=2)

    def test_median_z(self):
        om = ortho.OrthoGrid(pcd=self.pc, value_by="z", agg="median", cell_size=1.0)
        self.assertAlmostEqual(om.value_at(0.5, 0.5)["value"], 0.45, places=2)

    def test_count_and_density(self):
        oc = ortho.OrthoGrid(pcd=self.pc, value_by="count", cell_size=1.0)
        self.assertEqual(oc.value_at(0.5, 0.5)["value"], 100)
        od = ortho.OrthoGrid(pcd=self.pc, value_by="density", cell_size=1.0)
        # 100 points per 1 m^2 cell.
        self.assertAlmostEqual(od.value_at(0.5, 0.5)["value"], 100.0, places=6)

    def test_value_at_outside_is_none(self):
        og = ortho.OrthoGrid(pcd=self.pc, value_by="z", cell_size=1.0)
        self.assertIsNone(og.value_at(100.0, 100.0))

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(pcd=self.pc, value_by="nope")
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(pcd=self.pc, bbox=([0, 0], [1, 1]),
                            intercepts=_Container([_Ann([0, 0, 0])]))


class TestOrthoGridReportBbox(unittest.TestCase):
    def test_bbox_restricts_report_not_cells(self):
        pc = _ramp_pc()
        og = ortho.OrthoGrid(pcd=pc, value_by="z", cell_size=1.0,
                             bbox=([0, 0], [2, 2]))
        # All 16 cells present; only the 4 within [0,2]^2 reported.
        self.assertEqual(int(og.present.sum()), 16)
        self.assertEqual(int(og.report_mask.sum()), 4)
        self.assertTrue(og.value_at(0.5, 0.5)["in_report"])
        self.assertFalse(og.value_at(3.5, 3.5)["in_report"])


class TestOrthoGridLabel(unittest.TestCase):
    def setUp(self):
        self.anns = _Container([
            _Ann([0.5, 0.5, 0.0], "coral"),
            _Ann([0.6, 0.6, 0.0], "coral"),   # majority coral in cell (0,0)
            _Ann([0.5, 0.5, 0.0], "algae"),
            _Ann([3.5, 3.5, 0.0], "algae"),
        ])

    def test_majority_label(self):
        og = ortho.OrthoGrid(annotations=self.anns, value_by="label",
                             cell_size=1.0)
        self.assertEqual(og.value_at(0.5, 0.5)["label"], "coral")
        self.assertEqual(og.value_at(3.5, 3.5)["label"], "algae")

    def test_tie_breaks_alphabetically(self):
        anns = _Container([_Ann([0.5, 0.5, 0.0], "zeta"),
                           _Ann([0.6, 0.6, 0.0], "alpha")])
        og = ortho.OrthoGrid(annotations=anns, value_by="label", cell_size=1.0)
        self.assertEqual(og.value_at(0.5, 0.5)["label"], "alpha")

    def test_no_data_cell_label_none(self):
        anns = _Container([_Ann([0.5, 0.5, 0.0], None)])
        og = ortho.OrthoGrid(annotations=anns, value_by="label", cell_size=1.0)
        rec = og.value_at(0.5, 0.5)
        self.assertIsNone(rec["label"])
        self.assertEqual(rec["n_points"], 1)


class TestOrthoGridIntercepts(unittest.TestCase):
    def test_fit_recovers_lattice(self):
        # One intercept per 1 m cell centre over a 4x3 grid.
        ic = _Container([
            _Ann([i + 0.5, j + 0.5, 0.0]) for i in range(4) for j in range(3)
        ])
        og = ortho.OrthoGrid(annotations=ic, value_by="label",
                             cell_size=1.0, intercepts=ic)
        self.assertEqual(og.info.get("nx"), 4)
        self.assertEqual(og.info.get("ny"), 3)
        self.assertEqual(og.report_bbox, ((0.0, 0.0), (4.0, 3.0)))


class TestOrthoGridShow(unittest.TestCase):
    def test_continuous_show_returns_figure(self):
        og = ortho.OrthoGrid(pcd=_ramp_pc(), value_by="z", cell_size=1.0)
        fig = og.show(title="DEM")
        self.assertEqual(type(fig).__name__, "Figure")

    def test_label_show_returns_figure(self):
        anns = _Container([_Ann([0.5, 0.5, 0.0], "coral"),
                           _Ann([3.5, 3.5, 0.0], "algae")])
        og = ortho.OrthoGrid(annotations=anns, value_by="label", cell_size=1.0)
        fig = og.show(title="labels")
        self.assertEqual(type(fig).__name__, "Figure")


class TestOrthoMapBasics(unittest.TestCase):
    def test_show_returns_pil(self):
        om = ortho.OrthoMap(_ramp_pc(), pixel_width=100)
        img = om.show()
        self.assertEqual(img.size, (om.width, om.height))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
