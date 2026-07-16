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
    def __init__(self, coords, label=None, group=None):
        self.coords = np.asarray(coords, dtype=float)
        self.label = label
        self.group = group


class _Container:
    def __init__(self, items):
        self.data = {i: it for i, it in enumerate(items)}


def _ramp_pc(n=40, span=4.0):
    """A dense square cloud on [0, span]^2 with z == x (a planar ramp)."""
    g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
    pts = np.column_stack([g[:, 0], g[:, 1], g[:, 0]])
    return _PC(pts)


def _flat_pc(z, span=4.0, n=20, x0=0.0, y0=0.0):
    """A flat square cloud at constant height *z*, offset by (x0, y0)."""
    g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
    pts = np.column_stack([
        g[:, 0] + x0, g[:, 1] + y0, np.full(len(g), float(z)),
    ])
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

    def test_show_pcd_widens_view_to_full_cloud(self):
        # Annotations occupy only the [0,1] corner; the cloud spans [0,4].
        pc = _ramp_pc()
        anns = _Container([_Ann([0.3, 0.3, 0.0], "coral"),
                           _Ann([0.6, 0.6, 0.0], "algae")])
        og = ortho.OrthoGrid(annotations=anns, pcd=pc, value_by="label",
                             cell_size=0.5)
        # Grid lattice covers only the annotation corner.
        self.assertLess(og.extent[1], 1.5)
        # show_pcd=True widens the axes to the full cloud extent (~4).
        ax = og.show(show_pcd=True).axes[0]
        self.assertGreater(ax.get_xlim()[1], 3.5)
        # show_pcd=False clips to the grid extent.
        ax2 = og.show(show_pcd=False).axes[0]
        self.assertLess(ax2.get_xlim()[1], 1.5)


def _count_color(img, rgb):
    arr = np.asarray(img)
    return int(np.all(arr == np.array(rgb, dtype=arr.dtype), axis=-1).sum())


class TestOrthoMapBasics(unittest.TestCase):
    def test_show_returns_pil(self):
        om = ortho.OrthoMap(_ramp_pc(), pixel_width=100)
        img = om.show()
        self.assertEqual(img.size, (om.width, om.height))

    def test_default_highlight_is_red(self):
        om = ortho.OrthoMap(_ramp_pc(), pixel_width=120)
        img = om.show(highlights=np.array([[2.0, 2.0, 2.0]]), point_size=8)
        self.assertGreater(_count_color(img, (255, 0, 0)), 0)


class TestOrthoMapStyling(unittest.TestCase):
    def setUp(self):
        self.pc = _ramp_pc()
        self.anns = _Container([
            _Ann([0.5, 0.5, 0.5], label="coral", group="A"),
            _Ann([3.5, 3.5, 3.5], label="algae", group="B"),
            _Ann([2.0, 2.0, 2.0], label="coral", group="A"),
        ])

    def test_color_by_label_differs_from_default(self):
        om = ortho.OrthoMap(self.pc, pixel_width=150)
        default = np.asarray(om.show(highlights=self.anns, point_size=8))
        labelled = np.asarray(
            om.show(highlights=self.anns, color_by="label", point_size=8)
        )
        self.assertFalse(np.array_equal(default, labelled))

    def test_fill_by_group_changes_markers(self):
        om = ortho.OrthoMap(self.pc, pixel_width=150)
        filled = np.asarray(om.show(highlights=self.anns, point_size=8))
        grouped = np.asarray(
            om.show(highlights=self.anns, fill_by_group=True, point_size=8)
        )
        self.assertFalse(np.array_equal(filled, grouped))

    def test_explicit_label_colors(self):
        om = ortho.OrthoMap(self.pc, pixel_width=150)
        img = np.asarray(om.show(
            highlights=self.anns, color_by="label",
            label_colors={"coral": (0, 255, 0), "algae": (0, 0, 255)},
            point_size=8,
        ))
        self.assertGreater(_count_color(img, (0, 255, 0)), 0)
        self.assertGreater(_count_color(img, (0, 0, 255)), 0)

    def test_grayscale_background(self):
        om = ortho.OrthoMap(_PC(_ramp_pc().points, np.random.RandomState(0)
                                .rand(len(_ramp_pc().points), 3)),
                            pixel_width=120)
        img = np.asarray(om.show(grayscale=True))
        # A grayscale image has r == g == b everywhere.
        self.assertTrue(np.all((img[..., 0] == img[..., 1])
                               & (img[..., 1] == img[..., 2])))

    def test_crop_narrows_output(self):
        om = ortho.OrthoMap(self.pc, pixel_width=200)
        img = om.show(highlights=_Ann([2.0, 2.0, 2.0]),
                      crop=([2.0, 2.0, 2.0], 1.0))
        self.assertLess(img.size[0], om.width)
        self.assertLess(img.size[1], om.height)


class TestExtractHighlights(unittest.TestCase):
    def test_single_annotation(self):
        coords, labels, groups = ortho.OrthoMap._extract_highlights(
            _Ann([1.0, 2.0, 3.0], label="x", group="g")
        )
        self.assertEqual(coords.shape, (1, 3))
        self.assertEqual(labels, ["x"])
        self.assertEqual(groups, ["g"])

    def test_container_metadata(self):
        cont = _Container([_Ann([0, 0, 0], "a", "g1"),
                           _Ann([1, 1, 1], "b", "g2")])
        coords, labels, groups = ortho.OrthoMap._extract_highlights(cont)
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(groups, ["g1", "g2"])

    def test_plain_array(self):
        coords, labels, groups = ortho.OrthoMap._extract_highlights(
            np.array([[0, 0, 0], [1, 1, 1]])
        )
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(labels, [None, None])


class TestOrthoMapGroup(unittest.TestCase):
    def test_raw_composite_arrange_none(self):
        # Legacy behaviour: composite clouds at their true world coords.
        pc1 = _ramp_pc()
        pc2 = _PC(pc1.points + np.array([4.0, 0.0, 0.0]))
        single = ortho.OrthoMap(pc1, pixel_width=150)
        grp = ortho.OrthoMapGroup([pc1, pc2], pixel_width=300, arrange=None)
        self.assertGreaterEqual(grp.width, single.width)
        self.assertEqual(grp.image.shape, (grp.height, grp.width, 3))
        self.assertTrue(all(off == (0.0, 0.0) for off in grp._offsets))

    def test_stack_orders_by_depth_auto(self):
        # Input deep-first; auto-order must still put the shallow plot on top
        # (larger world-y => higher in the image => larger dy).
        deep = _flat_pc(z=-20.0)
        shallow = _flat_pc(z=-5.0)
        grp = ortho.OrthoMapGroup([deep, shallow], pixel_width=200)
        dy_deep = grp._offsets[0][1]
        dy_shallow = grp._offsets[1][1]
        self.assertGreater(dy_shallow, dy_deep)

    def test_manual_order_respected(self):
        deep = _flat_pc(z=-20.0)
        shallow = _flat_pc(z=-5.0)
        # Force the deep plot (index 0) to the top despite being deeper.
        grp = ortho.OrthoMapGroup(
            [deep, shallow], pixel_width=200, order=[0, 1],
        )
        self.assertGreater(grp._offsets[0][1], grp._offsets[1][1])

    def test_vertical_spacing_increases_height(self):
        clouds = [_flat_pc(z=-5.0), _flat_pc(z=-20.0)]
        tight = ortho.OrthoMapGroup(clouds, pixel_width=200,
                                    vertical_spacing=0.5)
        loose = ortho.OrthoMapGroup(clouds, pixel_width=200,
                                    vertical_spacing=5.0)
        self.assertGreater(loose.height, tight.height)

    def test_per_plot_annotations_land_in_bounds(self):
        deep = _flat_pc(z=-20.0)
        shallow = _flat_pc(z=-5.0)
        grp = ortho.OrthoMapGroup([deep, shallow], pixel_width=200)
        # Each set is in its own plot's frame (same xy, different depth).
        anns_deep = _Container([_Ann([2.0, 2.0, -20.0], "coral")])
        anns_shallow = _Container([_Ann([2.0, 2.0, -5.0], "sponge")])
        merged = grp._build_highlights([anns_deep, anns_shallow])
        coords = np.vstack([it.coords for it in merged.data.values()])
        pixels = grp.project(coords)
        self.assertTrue(np.all(pixels[:, 0] >= 0))
        self.assertTrue(np.all(pixels[:, 0] < grp.width))
        self.assertTrue(np.all(pixels[:, 1] >= 0))
        self.assertTrue(np.all(pixels[:, 1] < grp.height))
        # Deep annotation (index 0) sits lower => larger pixel-y.
        self.assertGreater(pixels[0, 1], pixels[1, 1])

    def test_autosplit_single_combined(self):
        # Distinct xy frames so a combined set can be split by position.
        shallow = _flat_pc(z=-5.0, x0=0.0, y0=0.0)
        deep = _flat_pc(z=-20.0, x0=100.0, y0=100.0)
        grp = ortho.OrthoMapGroup([shallow, deep], pixel_width=200)
        combined = _Container([
            _Ann([2.0, 2.0, -5.0], "coral"),        # -> shallow
            _Ann([102.0, 102.0, -20.0], "sponge"),  # -> deep
        ])
        merged = grp._build_highlights(combined)
        coords = np.vstack([it.coords for it in merged.data.values()])
        pixels = grp.project(coords)
        self.assertTrue(np.all(pixels[:, 0] >= 0))
        self.assertTrue(np.all(pixels[:, 0] < grp.width))
        self.assertTrue(np.all(pixels[:, 1] >= 0))
        self.assertTrue(np.all(pixels[:, 1] < grp.height))
        # Shallow annotation (first) sits higher => smaller pixel-y.
        self.assertLess(pixels[0, 1], pixels[1, 1])

    def test_show_with_labels(self):
        grp = ortho.OrthoMapGroup(
            [_flat_pc(z=-5.0), _flat_pc(z=-20.0)],
            pixel_width=200, names=["shallow", "deep"],
        )
        img = grp.show(show_labels=True, label_color=(0, 0, 0))
        # Labels live in an added left gutter: same height, wider than the raster.
        self.assertEqual(img.height, grp.height)
        self.assertGreater(img.width, grp.width)

    def test_show_per_plot_annotations(self):
        grp = ortho.OrthoMapGroup(
            [_flat_pc(z=-5.0), _flat_pc(z=-20.0)], pixel_width=200,
        )
        anns = [
            _Container([_Ann([2.0, 2.0, -5.0], "coral")]),
            _Container([_Ann([2.0, 2.0, -20.0], "sponge")]),
        ]
        img = grp.show(anns, color_by="label", point_size=6)
        self.assertEqual(img.size, (grp.width, grp.height))

    def test_per_plot_annotations_with_none_entry(self):
        # A plot with no annotations (None) must be skipped, not collapse the
        # per-plot list into a coordinate array.
        grp = ortho.OrthoMapGroup(
            [_flat_pc(z=-5.0), _flat_pc(z=-20.0)], pixel_width=200,
        )
        anns = [_Container([_Ann([2.0, 2.0, -5.0], "coral")]), None]
        merged = grp._build_highlights(anns)
        self.assertEqual(len(merged.data), 1)
        img = grp.show(anns, color_by="label")
        self.assertEqual(img.size, (grp.width, grp.height))

    def test_centroid_alignment_differs_from_bbox_center(self):
        # An L-shaped (asymmetric) plot: centroid != bbox midpoint, so the two
        # alignment modes place it at different x offsets.
        pts = np.array([[x / 5.0, y / 5.0, 0.0]
                        for x in range(20) for y in range(20)
                        if x < 6 or y < 6], dtype=float)
        plots = [_PC(pts), _flat_pc(z=-10.0)]
        g_centroid = ortho.OrthoMapGroup(plots, pixel_width=200,
                                         align="centroid")
        g_center = ortho.OrthoMapGroup(plots, pixel_width=200, align="center")
        self.assertNotAlmostEqual(
            g_centroid._offsets[0][0], g_center._offsets[0][0],
        )

    def test_slope_orientation_reduces_foreshortening(self):
        # A steeply tilted planar cloud is heavily foreshortened top-down; the
        # slope view sees it face-on, so its view-plane extent is larger.
        n, span = 25, 4.0
        g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
        u, v = g[:, 0], g[:, 1]
        # Plane tilted ~60 deg: z grows fast with u (steep down-slope).
        pts = np.column_stack([u, v, 1.8 * u])
        pc = _PC(pts)
        top = ortho.OrthoMapGroup([pc], pixel_width=200, orient="topdown")
        slope = ortho.OrthoMapGroup([pc], pixel_width=200, orient="slope")

        def max_extent(bb):
            return max(bb[1] - bb[0], bb[3] - bb[2])

        self.assertGreater(
            max_extent(slope._plot_bboxes[0]),
            max_extent(top._plot_bboxes[0]) * 1.2,
        )

    def test_slope_view_is_face_on(self):
        # After the slope rotation the plane should be flat (view-z ~ 0).
        n, span = 20, 4.0
        g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
        u, v = g[:, 0], g[:, 1]
        pts = np.column_stack([u, v, 1.3 * u + 0.4 * v])
        grp = ortho.OrthoMapGroup([_PC(pts)], pixel_width=200, orient="slope")
        rotated = (grp._rotations[0] @ pts.T).T
        self.assertLess(np.ptp(rotated[:, 2]), 1e-6)

    def test_slope_tilt_has_no_z_spin(self):
        # A near-flat plot must not be spun in-plane: the slope rotation is
        # (near) identity, i.e. no z-axis rotation of the horizontal frame.
        n, span = 20, 4.0
        g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
        u, v = g[:, 0], g[:, 1]
        pts = np.column_stack([u, v, 0.001 * u])  # barely sloped
        grp = ortho.OrthoMapGroup([_PC(pts)], pixel_width=200, orient="slope")
        np.testing.assert_allclose(grp._rotations[0], np.eye(3), atol=1e-2)

    def test_slope_flat_plot_falls_back(self):
        # A perfectly flat plot -> slope rotation == top-down (identity, z-up).
        grp = ortho.OrthoMapGroup([_flat_pc(z=-5.0)], orient="slope",
                                  pixel_width=100)
        np.testing.assert_allclose(grp._rotations[0], np.eye(3), atol=1e-9)

    def test_marker_size_scales_with_composite(self):
        # Default marker size grows with the composite so dots stay visible.
        plots = [_flat_pc(z=-5.0), _flat_pc(z=-20.0)]
        anns = [_Container([_Ann([2.0, 2.0, -5.0], "a")]),
                _Container([_Ann([2.0, 2.0, -20.0], "b")])]

        def nonwhite(img):
            return int((np.asarray(img) != 255).any(-1).sum())

        big = ortho.OrthoMapGroup(plots, pixel_height=4000)
        small = ortho.OrthoMapGroup(plots, pixel_height=600)
        # Same annotations, but the larger composite draws larger markers.
        self.assertGreater(
            nonwhite(big.show(anns, show_labels=False)),
            nonwhite(small.show(anns, show_labels=False)),
        )

    def test_load_font_returns_scalable(self):
        font = ortho.OrthoMapGroup._load_font(120)
        # A scalable font exposes the requested size; the fixed fallback would
        # not carry 120. Either way it must be a usable font object.
        self.assertTrue(hasattr(font, "getbbox") or hasattr(font, "getsize"))

    def test_label_colours_consistent_across_plots(self):
        # A label shared by different plots must map to one colour composite-wide
        # (colours are assigned over the union of all plots' labels).
        grp = ortho.OrthoMapGroup([_flat_pc(z=-5.0), _flat_pc(z=-20.0)],
                                  pixel_width=200, names=["a", "b"])
        anns = [
            _Container([_Ann([1.0, 1.0, -5.0], "coral"),
                        _Ann([3.0, 3.0, -5.0], "sponge")]),
            _Container([_Ann([1.0, 1.0, -20.0], "sponge"),
                        _Ann([3.0, 3.0, -20.0], "algae")]),
        ]
        merged = grp._build_highlights(anns)
        labels = [it.label for it in merged.data.values()]
        coords = np.vstack([it.coords for it in merged.data.values()])
        fills, _ = grp._resolve_marker_style(
            coords, labels, [None] * len(labels),
            "label", None, False, (255, 0, 0), (0, 0, 0),
        )
        by_label = {}
        for lbl, col in zip(labels, fills):
            by_label.setdefault(lbl, set()).add(tuple(col))
        # "sponge" is in both plots -> exactly one colour.
        self.assertEqual(len(by_label["sponge"]), 1)

    def test_vertical_labels_narrower_than_horizontal(self):
        grp = ortho.OrthoMapGroup([_flat_pc(z=-5.0), _flat_pc(z=-20.0)],
                                  pixel_height=1200,
                                  names=["cur_sna_05m", "cur_sna_20m"])
        base = grp.width
        vert = grp.show(show_labels=True, label_rotation=90)
        horiz = grp.show(show_labels=True, label_rotation=0)
        self.assertLess(vert.width - base, horiz.width - base)

    def test_legend_adds_panel_with_marker_colours(self):
        grp = ortho.OrthoMapGroup([_flat_pc(z=-5.0), _flat_pc(z=-20.0)],
                                  pixel_height=1200, names=["a", "b"])
        anns = [
            _Container([_Ann([1.0, 1.0, -5.0], "coral")]),
            _Container([_Ann([3.0, 3.0, -20.0], "sponge")]),
        ]
        plain = grp.show(anns, show_labels=False)
        withleg = grp.show(anns, show_labels=False, legend=True)
        # Legend is appended below -> taller image.
        self.assertGreater(withleg.height, plain.height)
        # The legend swatches use the same LUT as the markers.
        lut = ortho._label_color_lut(["coral", "sponge"])
        panel = np.asarray(withleg)[plain.height:]
        panel_colours = {tuple(c) for c in panel.reshape(-1, 3)}
        for col in lut.values():
            self.assertIn(tuple(col), panel_colours)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ortho.OrthoMapGroup([])


class _CountingGrid(ortho.OrthoGrid):
    """OrthoGrid whose per-cell measurement is just the member-point count.

    Overriding ``_measure_cell`` exercises the real ``_reduce_custom`` binning /
    window-widening / present-masking loop without importing ``measurements``
    or building a real ``SimplePointCloud``.
    """

    def _measure_cell(self, idx, center_world):
        self.last_center = center_world
        return float(0 if idx is None else len(idx))


def _sparse_pc():
    """Two point clusters in cells (0,0) and (2,2) of a 1 m grid."""
    pts = np.array([
        [0.5, 0.5, 0.0], [0.55, 0.6, 0.1],
        [2.5, 2.5, 0.0], [2.4, 2.55, 0.2], [2.6, 2.45, -0.1],
    ])
    return _PC(pts)


class TestOrthoGridCustom(unittest.TestCase):
    def _dummy(self, metric=None):
        # A callable whose __name__ maps to a default metric ("Ra").
        def calc_roughness(*a, **k):  # noqa: D401 - stub
            return None
        return calc_roughness

    def test_dense_values_equal_member_counts(self):
        pc = _ramp_pc()  # 4x4 cells, 100 points each
        og = _CountingGrid(
            pcd=pc, value_by="custom", measurement=self._dummy(),
            cell_size=1.0,
        )
        # metric defaults to "Ra" for a calc_roughness-named callable.
        self.assertEqual(og.metric, "Ra")
        self.assertEqual(og.value_by, "custom")
        self.assertEqual(og.values.shape, (4, 4))
        # Every cell present; value == its own member count (100).
        self.assertTrue(np.array_equal(np.isfinite(og.values), og.present))
        self.assertEqual(og.value_at(0.5, 0.5)["value"], 100.0)

    def test_sparse_present_and_nan(self):
        og = _CountingGrid(
            pcd=_sparse_pc(), value_by="custom", measurement=self._dummy(),
            metric="Ra", cell_size=1.0, min_points=1,
        )
        # Finite exactly on present cells.
        self.assertTrue(np.array_equal(np.isfinite(og.values), og.present))
        self.assertEqual(og.value_at(0.5, 0.5)["value"], 2.0)
        self.assertEqual(og.value_at(2.5, 2.5)["value"], 3.0)
        # An empty interior cell is NaN.
        rec = og.value_at(1.5, 1.5)
        self.assertTrue(rec is None or np.isnan(og.values[rec["iy"], rec["ix"]]))

    def test_default_min_points(self):
        og = _CountingGrid(
            pcd=_ramp_pc(), value_by="custom", measurement=self._dummy(),
            cell_size=1.0,
        )
        self.assertEqual(og.min_points, 10)

    def test_min_points_gates_sparse_cells(self):
        # Sparse cells: (0,0) has 2 points, (2,2) has 3.
        og = _CountingGrid(
            pcd=_sparse_pc(), value_by="custom", measurement=self._dummy(),
            metric="Ra", cell_size=1.0, min_points=3,
        )
        # Below-threshold cell is left NaN even though it holds points.
        self.assertTrue(og.present[0, 0])
        self.assertTrue(np.isnan(og.values[0, 0]))
        # At/above threshold is measured.
        self.assertEqual(og.value_at(2.5, 2.5)["value"], 3.0)
        # min_points < 1 is rejected.
        with self.assertRaises(ValueError):
            _CountingGrid(pcd=_sparse_pc(), value_by="custom",
                          measurement=self._dummy(), metric="Ra", min_points=0)

    def test_neighborhood_scale_widens_window(self):
        pc = _ramp_pc()  # dense, so neighbours exist
        og1 = _CountingGrid(
            pcd=pc, value_by="custom", measurement=self._dummy(),
            metric="Ra", cell_size=1.0, neighborhood_scale=1.0,
        )
        og2 = _CountingGrid(
            pcd=pc, value_by="custom", measurement=self._dummy(),
            metric="Ra", cell_size=1.0, neighborhood_scale=2.0,
        )
        # Interior cell (1,1) borrows neighbouring points when widened.
        v1 = og1.value_at(1.5, 1.5)["value"]
        v2 = og2.value_at(1.5, 1.5)["value"]
        self.assertGreater(v2, v1)

    def test_value_label_is_metric(self):
        og = _CountingGrid(
            pcd=_ramp_pc(), value_by="custom", measurement=self._dummy(),
            metric="Rq", cell_size=1.0,
        )
        self.assertEqual(og._value_label(), "Rq")
        og2 = _CountingGrid(
            pcd=_ramp_pc(), value_by="custom", measurement=self._dummy(),
            metric="Ra", cell_size=1.0, value_label="roughness Ra (m)",
        )
        self.assertEqual(og2._value_label(), "roughness Ra (m)")

    def test_invalid_custom_args(self):
        pc = _ramp_pc()
        # No measurement.
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(pcd=pc, value_by="custom")
        # Non-callable measurement.
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(pcd=pc, value_by="custom", measurement="calc_roughness")
        # neighborhood_scale < 1.
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(pcd=pc, value_by="custom", measurement=self._dummy(),
                            neighborhood_scale=0.5)
        # No pcd.
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(value_by="custom", measurement=self._dummy())
        # Unknown measurement name and no metric -> cannot infer.
        with self.assertRaises(ValueError):
            ortho.OrthoGrid(pcd=pc, value_by="custom",
                            measurement=(lambda *a, **k: None))


class TestScaleLimits(unittest.TestCase):
    def test_robust_clips_outlier(self):
        # Many small values (~0.01-0.05) with a single large outlier, like a
        # degenerate roughness cell on an otherwise smooth map.
        data = np.concatenate([np.linspace(0.01, 0.05, 200), [6.0]])
        vmin_r, vmax_r = ortho.OrthoGrid._scale_limits(data, True, (2.0, 98.0))
        vmin_e, vmax_e = ortho.OrthoGrid._scale_limits(data, False, (2.0, 98.0))
        self.assertLess(vmax_r, 1.0)     # robust ignores the 6.0 outlier
        self.assertEqual(vmax_e, 6.0)    # exact keeps it

    def test_equal_values_nudged(self):
        data = np.array([2.0, 2.0, 2.0])
        vmin, vmax = ortho.OrthoGrid._scale_limits(data, False, (2.0, 98.0))
        self.assertLess(vmin, vmax)

    def test_empty_defaults(self):
        self.assertEqual(
            ortho.OrthoGrid._scale_limits(np.array([]), True, (2.0, 98.0)),
            (0.0, 1.0),
        )


class TestOrthoGridShowRobust(unittest.TestCase):
    def test_robust_sets_histogram_xlim(self):
        og = ortho.OrthoGrid(pcd=_ramp_pc(), value_by="z", cell_size=1.0)
        rep = og.values[og.report_mask & ~np.isnan(og.values)]
        vmin, vmax = ortho.OrthoGrid._scale_limits(rep, True, (2.0, 98.0))
        fig = og.show(show_pcd=False, robust=True)
        ax_hist = next(a for a in fig.axes if a.get_ylabel() == "Cell count")
        xlo, xhi = ax_hist.get_xlim()
        self.assertAlmostEqual(xlo, vmin, places=6)
        self.assertAlmostEqual(xhi, vmax, places=6)


class TestOrthoGridHighlights(unittest.TestCase):
    def _grid(self, **kw):
        return ortho.OrthoGrid(pcd=_ramp_pc(), value_by="z", cell_size=1.0, **kw)

    def _anns(self):
        return _Container([_Ann([0.5, 0.5, 0.2], "a"),
                           _Ann([2.5, 2.5, 0.6], "b")])

    def test_scatter_added(self):
        fig = self._grid().show(show_pcd=False, highlights=self._anns())
        self.assertGreaterEqual(len(fig.axes[0].collections), 1)

    def test_auto_colours_by_label(self):
        # No color_by given: labelled points auto-colour by label (2 labels).
        fig = self._grid().show(show_pcd=False, highlights=self._anns())
        fc = fig.axes[0].collections[-1].get_facecolors()
        uniq = {tuple(np.round(c, 3)) for c in fc}
        self.assertGreaterEqual(len(uniq), 2)

    def test_unlabelled_points_single_colour(self):
        # No labels -> auto falls back to a single point_color.
        pts = np.array([[0.5, 0.5, 0.0], [2.5, 2.5, 0.0]])
        fig = self._grid().show(show_pcd=False, highlights=pts)
        fc = fig.axes[0].collections[-1].get_facecolors()
        uniq = {tuple(np.round(c, 3)) for c in fc}
        self.assertEqual(len(uniq), 1)

    def test_color_by_none_forces_single_colour(self):
        # Explicit color_by=None ignores labels -> single colour.
        fig = self._grid().show(
            show_pcd=False, highlights=self._anns(), color_by=None
        )
        fc = fig.axes[0].collections[-1].get_facecolors()
        uniq = {tuple(np.round(c, 3)) for c in fc}
        self.assertEqual(len(uniq), 1)

    def test_color_by_label_distinct(self):
        fig = self._grid().show(
            show_pcd=False, highlights=self._anns(), color_by="label"
        )
        fc = fig.axes[0].collections[-1].get_facecolors()
        uniq = {tuple(np.round(c, 3)) for c in fc}
        self.assertGreaterEqual(len(uniq), 2)

    def test_point_size_metres_uses_patches(self):
        import matplotlib.patches as mpatches
        fig = self._grid().show(
            show_pcd=False, highlights=self._anns(), point_size_metres=0.5,
        )
        circles = [p for p in fig.axes[0].patches
                   if isinstance(p, mpatches.Circle)]
        self.assertEqual(len(circles), 2)
        fig2 = self._grid().show(
            show_pcd=False, highlights=self._anns(),
            point_size_metres=0.5, point_shape="square",
        )
        rects = [p for p in fig2.axes[0].patches
                 if isinstance(p, mpatches.Rectangle)]
        self.assertGreaterEqual(len(rects), 2)

    def test_out_of_bounds_warns(self):
        anns = _Container([_Ann([100.0, 100.0, 0.0], "x")])
        with self.assertLogs("substrata.ortho", level="WARNING") as cm:
            self._grid().show(show_pcd=False, highlights=anns)
        self.assertTrue(
            any("outside the grid extent" in m for m in cm.output)
        )

    def test_overlay_in_label_mode(self):
        anns = _Container([_Ann([0.5, 0.5, 0.0], "coral")])
        og = ortho.OrthoGrid(annotations=anns, value_by="label", cell_size=1.0)
        fig = og.show(show_pcd=False, highlights=anns)
        self.assertGreaterEqual(len(fig.axes[0].collections), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
