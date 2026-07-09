"""Tests for :mod:`substrata.animations` (animated OrthoGrid fill).

Loaded in isolation against a bare ``substrata`` shell (real ``settings`` +
``ortho``, stubbed ``logging``) so the suite runs without open3d. Only numpy,
matplotlib (Agg) and Pillow are needed.
"""

# Standard Library
import importlib.util
import os
import shutil
import sys
import tempfile
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


def _load():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    def _load_mod(name, rel):
        spec = importlib.util.spec_from_file_location(name, _SUB / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    if "substrata.logging" not in sys.modules:
        # Load the real (lightweight) logging module rather than a stub, so the
        # shared sys.modules stays compatible with sibling tests that need
        # substrata.logging.tqdm_joblib (e.g. test_cameras_save).
        _load_mod("substrata.logging", "logging.py")
    if "substrata.settings" not in sys.modules:
        _load_mod("substrata.settings", "settings.py")
    if "substrata.ortho" not in sys.modules:
        _load_mod("substrata.ortho", "ortho.py")
    if "substrata.animations" not in sys.modules:
        _load_mod("substrata.animations", "animations.py")
    return sys.modules["substrata.ortho"], sys.modules["substrata.animations"]


ortho, anim = _load()


class _PC:
    def __init__(self, points, colors=None):
        self.points = np.asarray(points, dtype=float)
        self.colors = (np.ones((len(self.points), 3)) if colors is None
                       else np.asarray(colors, dtype=float))


class _Ann:
    def __init__(self, coords, label=None):
        self.coords = np.asarray(coords, dtype=float)
        self.label = label


class _Container:
    def __init__(self, items):
        self.data = {i: it for i, it in enumerate(items)}


def _ramp_pc(n=40, span=4.0):
    g = np.mgrid[0:n, 0:n].reshape(2, -1).T / n * span
    return _PC(np.column_stack([g[:, 0], g[:, 1], g[:, 0]]))


def _label_grid():
    anns = _Container([
        _Ann([0.3, 0.3, 0.0], "coral"), _Ann([1.5, 1.5, 0.0], "algae"),
        _Ann([2.5, 0.5, 0.0], "coral"), _Ann([3.5, 3.5, 0.0], "sand"),
    ])
    return ortho.OrthoGrid(annotations=anns, pcd=_ramp_pc(),
                           value_by="label", cell_size=0.5)


class TestRevealOrder(unittest.TestCase):
    def test_column_major_top_first(self):
        rt = anim._reveal_t(3, 2)   # nx=3, ny=2
        # Top-left cell (top row j=1, col 0) reveals first.
        self.assertEqual(rt[1, 0], rt.min())
        # Bottom-right cell reveals last.
        self.assertEqual(rt[0, 2], rt.max())

    def test_columns_are_sequential(self):
        rt = anim._reveal_t(4, 5)
        for c in range(3):
            self.assertLess(rt[:, c].max(), rt[:, c + 1].min())

    def test_within_column_top_to_bottom(self):
        rt = anim._reveal_t(2, 4)
        # Within a column, reveal_t decreases with row index (top first).
        col = rt[:, 0]
        self.assertTrue(np.all(np.diff(col) < 0))

    def test_alternative_orders_are_valid_fractions(self):
        # rows / random / spiral each assign every cell a distinct fraction in
        # (0, 1] with max == 1 (so the final frame completes).
        for fn in (anim._reveal_t_rows, anim._reveal_t_random,
                   anim._reveal_t_spiral):
            rt = fn(5, 3)
            self.assertEqual(rt.shape, (3, 5))
            self.assertEqual(len(np.unique(rt)), 15, fn.__name__)
            self.assertGreater(rt.min(), 0.0)
            self.assertAlmostEqual(rt.max(), 1.0)

    def test_rows_are_top_to_bottom(self):
        rt = anim._reveal_t_rows(5, 3)
        # Whole top row reveals before the whole bottom row.
        self.assertLess(rt[2, :].max(), rt[0, :].min())

    def test_random_is_deterministic(self):
        np.testing.assert_array_equal(anim._reveal_t_random(5, 3),
                                      anim._reveal_t_random(5, 3))

    def test_spiral_expands_from_centre(self):
        rt = anim._reveal_t_spiral(5, 3)
        self.assertLess(rt[1, 2], rt[0, 0])   # centre before a corner

    def test_categories_dominant_first_equal_slices(self):
        grid = _label_grid()   # coral x2 (dominant), algae x1, sand x1
        rt, k = anim._reveal_t_categories(grid, grid.present, grid.report_mask)
        self.assertEqual(k, 3)
        labels = grid.cell_labels

        def mean_t(lbl):
            vals = [rt[j, i] for j in range(grid.ny) for i in range(grid.nx)
                    if grid.present[j, i] and labels[j, i] == lbl]
            return float(np.mean(vals))
        # Dominant class reveals earliest.
        self.assertLess(mean_t("coral"), mean_t("algae"))
        self.assertLess(mean_t("coral"), mean_t("sand"))
        # Each category occupies an equal 1/k slice: the dominant class fits in
        # the first third, and the last cell of all reaches 1.0.
        coral = [rt[j, i] for j in range(grid.ny) for i in range(grid.nx)
                 if grid.present[j, i] and labels[j, i] == "coral"]
        self.assertLessEqual(max(coral), 1.0 / k + 1e-9)
        self.assertAlmostEqual(rt[grid.present].max(), 1.0)


class TestWriterSelection(unittest.TestCase):
    def test_gif_and_bad(self):
        from matplotlib.animation import PillowWriter
        self.assertIsInstance(anim._writer_for("x.gif", 10), PillowWriter)
        # Case-insensitive extension handling.
        self.assertIsInstance(anim._writer_for("CLIP.GIF", 10), PillowWriter)
        with self.assertRaises(ValueError):
            anim._writer_for("x.png", 10)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_ffmpeg_branch(self):
        from matplotlib.animation import FFMpegWriter
        self.assertIsInstance(anim._writer_for("x.mp4", 10), FFMpegWriter)
        self.assertIsInstance(anim._writer_for("x.MOV", 10), FFMpegWriter)


class TestAnimateOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="anim_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_label_gif_written(self):
        out = os.path.join(self.tmp, "label.gif")
        ret = anim.animate_ortho_grid(_label_grid(), out, duration=0.4, fps=5)
        self.assertEqual(ret, out)
        self.assertTrue(os.path.getsize(out) > 0)

    def test_continuous_gif_written(self):
        grid = ortho.OrthoGrid(pcd=_ramp_pc(), value_by="z", cell_size=0.5)
        out = os.path.join(self.tmp, "dem.gif")
        anim.animate_ortho_grid(grid, out, duration=0.4, fps=5)
        self.assertTrue(os.path.getsize(out) > 0)

    def test_count_mode_gif(self):
        grid = ortho.OrthoGrid(pcd=_ramp_pc(), value_by="count", cell_size=0.5)
        out = os.path.join(self.tmp, "count.gif")
        anim.animate_ortho_grid(grid, out, duration=0.4, fps=5, show_pcd=False)
        self.assertTrue(os.path.getsize(out) > 0)

    def test_bad_extension_raises(self):
        with self.assertRaises(ValueError):
            anim.animate_ortho_grid(_label_grid(),
                                    os.path.join(self.tmp, "x.png"))

    def test_bad_sweep_raises(self):
        with self.assertRaises(ValueError):
            anim.animate_ortho_grid(_label_grid(),
                                    os.path.join(self.tmp, "x.gif"),
                                    sweep="zigzag")

    def test_all_sweeps_write_gifs(self):
        for mode in ("columns", "rows", "scan", "random", "spiral",
                     "categories"):
            out = os.path.join(self.tmp, f"{mode}.gif")
            anim.animate_ortho_grid(_label_grid(), out, sweep=mode,
                                    duration=0.4, fps=5)
            self.assertTrue(os.path.getsize(out) > 0, mode)

    def test_categories_requires_label_grid(self):
        grid = ortho.OrthoGrid(pcd=_ramp_pc(), value_by="z", cell_size=0.5)
        with self.assertRaises(ValueError):
            anim.animate_ortho_grid(grid, os.path.join(self.tmp, "x.gif"),
                                    sweep="categories")

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires ffmpeg")
    def test_mp4_written(self):
        out = os.path.join(self.tmp, "label.mp4")
        anim.animate_ortho_grid(_label_grid(), out, duration=0.4, fps=5)
        self.assertTrue(os.path.getsize(out) > 0)

    def test_creates_missing_output_directory(self):
        out = os.path.join(self.tmp, "nested", "sub", "label.gif")
        anim.animate_ortho_grid(_label_grid(), out, duration=0.2, fps=5)
        self.assertTrue(os.path.exists(out))


class TestFillProgression(unittest.TestCase):
    """The core promise: cells (and bars) fill in progressively over frames."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="anim_prog_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _colored(frame_rgb):
        # Saturated (non-grey, non-white) pixels ~ filled colour cells/bars.
        a = np.asarray(frame_rgb).astype(int)
        return int(((a.max(axis=2) - a.min(axis=2)) > 30).sum())

    def _frames(self, out):
        from PIL import Image, ImageSequence
        with Image.open(out) as im:
            n = getattr(im, "n_frames", 1)
            frames = [np.asarray(f.convert("RGB"))
                      for f in ImageSequence.Iterator(im)]
        return n, frames

    def test_fill_grows_across_frames(self):
        # A dense grid so per-frame reveals differ (Pillow drops identical
        # consecutive frames, so exact frame counts are not asserted).
        out = os.path.join(self.tmp, "grow.gif")
        anim.animate_ortho_grid(_label_grid(), out, duration=1.0, fps=8)
        _, frames = self._frames(out)
        c0 = self._colored(frames[0])                    # checkerboard (empty)
        cmid = self._colored(frames[len(frames) // 2])
        clast = self._colored(frames[-1])                # full colour grid
        self.assertLess(c0, clast)                       # it fills in
        self.assertLessEqual(c0, cmid)                   # monotonic growth
        self.assertLessEqual(cmid, clast)

    def test_gif_loop_flag(self):
        from PIL import Image
        once = os.path.join(self.tmp, "once.gif")
        loopy = os.path.join(self.tmp, "loop.gif")
        anim.animate_ortho_grid(_label_grid(), once, duration=0.3, fps=5)
        anim.animate_ortho_grid(_label_grid(), loopy, duration=0.3, fps=5,
                                loop=True)
        # No loop block -> plays once and holds the last frame.
        self.assertIsNone(Image.open(once).info.get("loop"))
        # loop=0 -> infinite.
        self.assertEqual(Image.open(loopy).info.get("loop"), 0)


class TestColourConsistency(unittest.TestCase):
    def test_resolve_label_colors_covers_present_labels(self):
        grid = _label_grid()
        lc, present = grid._resolve_label_colors()
        self.assertEqual(present, ["algae", "coral", "sand"])
        for lbl in present:
            self.assertIn(lbl, lc)
        # Stable across calls.
        lc2, _ = grid._resolve_label_colors()
        self.assertEqual(lc[present[0]], lc2[present[0]])

    def test_colors_are_tab20_and_painted_per_cell(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        grid = _label_grid()
        lc, present = grid._resolve_label_colors()
        cmap = plt.get_cmap("tab20")
        # Colour VALUES match the tab20 assignment show() uses.
        for i, lbl in enumerate(present):
            self.assertEqual(tuple(lc[lbl]), tuple(cmap(i % 20)))
        # The animation setup paints each present cell with its label colour.
        fig = plt.figure()
        axl, axr = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        target_rgb = anim._setup_label(
            grid, axl, axr, None, mpatches, plt,
            grid.present, grid.report_mask, anim._reveal_t(grid.nx, grid.ny),
        )[0]
        j, i = np.argwhere(grid.present)[0]
        lbl = grid.cell_labels[j, i]
        np.testing.assert_allclose(target_rgb[j, i], np.asarray(lc[lbl])[:3])
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
