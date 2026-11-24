"""imageadjust (display contrast) and annotationpages (``substrata images --layout pages``): contrast methods,
close-up detection, scale bars, zoom windows and the page / PDF rendering.

Loads the two modules directly (with ``substrata.settings``), without the package's heavy imports.
"""

# Standard Library
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Third-Party Libraries
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src/substrata"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # (dataclasses look the module up while it loads)
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    st = _load("substrata.settings", _SRC / "settings.py")
    saved = {k: sys.modules.get(k) for k in ("substrata", "substrata.settings", "substrata.imageadjust",
                                             "substrata.annotationpages")}
    fake = types.ModuleType("substrata")
    fake.settings = st
    sys.modules["substrata"], sys.modules["substrata.settings"] = fake, st
    try:
        ia = _load("substrata.imageadjust", _SRC / "imageadjust.py")
        sys.modules["substrata.imageadjust"] = ia
        fake.imageadjust = ia
        ap = _load("substrata.annotationpages", _SRC / "annotationpages.py")
        return st, ia, ap
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


ST, IA, AP = _load_modules()

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


class TestAutoContrast(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # a dark, blue-green photo: values within 20..100 (blue/green higher than red)
        self.img = np.clip(rng.normal([40, 70, 80], 10, (60, 80, 3)), 0, 255).astype(np.uint8)

    def test_methods_registered(self):
        for m in ("none", "stretch", "stretch-rgb", "clahe"):
            self.assertIn(m, IA.contrast_methods())

    def test_none_returns_input(self):
        self.assertIs(IA.auto_contrast(self.img, "none"), self.img)
        self.assertIs(IA.auto_contrast(self.img, None), self.img)

    def test_stretch_widens_range_and_keeps_channel_order(self):
        out = IA.auto_contrast(self.img, "stretch")
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, self.img.shape)
        lum = out.astype(float) @ [0.299, 0.587, 0.114]
        self.assertGreater(np.percentile(lum, 99) - np.percentile(lum, 1),
                           np.ptp(self.img.astype(float) @ [0.299, 0.587, 0.114]))
        m = out.reshape(-1, 3).mean(0)
        self.assertLess(m[0], m[1])  # one stretch for all channels: still less red than green

    def test_stretch_rgb_balances_channels(self):
        m = IA.auto_contrast(self.img, "stretch-rgb").reshape(-1, 3).mean(0)
        self.assertLess(np.ptp(m), 25)  # per-channel stretch removes the cast

    def test_reference_gives_crops_the_same_adjustment(self):
        crop = self.img[10:30, 10:40]
        a = IA.auto_contrast(crop, "stretch", reference=self.img)
        b = IA.auto_contrast(self.img, "stretch")[10:30, 10:40]
        np.testing.assert_array_equal(a, b)

    def test_register_new_method(self):
        IA.register_contrast_method("invert")(lambda img, ref, **kw: 1 - img)
        out = IA.auto_contrast(np.zeros((2, 2, 3), np.uint8), "invert")
        self.assertTrue((out == 255).all())

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            IA.auto_contrast(self.img, "nope")

    @unittest.skipIf(Image is None, "Pillow missing")
    def test_pil_in_pil_out(self):
        out = IA.auto_contrast(Image.fromarray(self.img), "stretch")
        self.assertTrue(hasattr(out, "convert"))
        self.assertEqual(out.size, (80, 60))

    def test_clahe(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV missing")
        self.assertEqual(IA.auto_contrast(self.img, "clahe").shape, self.img.shape)


class TestPageHelpers(unittest.TestCase):
    def test_closeup_types_for_camera_targets(self):
        self.assertEqual(AP.closeup_camera_types(["macro", "macro"], "camera_targets", None), {"macro"})

    def test_closeup_types_other_than_main(self):
        self.assertEqual(AP.closeup_camera_types(["macro", "survey"], None, "survey"), {"macro"})
        self.assertEqual(AP.closeup_camera_types(["survey"], None, "survey"), set())  # training views
        self.assertEqual(AP.closeup_camera_types([], None, "survey"), set())

    def test_nice_scale_length(self):
        self.assertEqual(AP.nice_scale_length(0.07), 0.05)
        self.assertEqual(AP.nice_scale_length(0.013), 0.01)
        self.assertEqual(AP.nice_scale_length(0.0001), 0.001)
        self.assertEqual([AP.scale_label(v) for v in (0.002, 0.05, 2.0)], ["2 mm", "5 cm", "2 m"])

    def test_zoom_window_centred_and_inside(self):
        self.assertEqual(AP.zoom_window(500, 400, 300, 1.5, (1000, 800)), (350, 300, 650, 500))
        x0, y0, x1, y1 = AP.zoom_window(10, 790, 300, 1.5, (1000, 800))  # near a corner: moved inside
        self.assertEqual((x0, y1), (0, 800))
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 1.5, places=2)
        x0, y0, x1, y1 = AP.zoom_window(500, 400, 5000, 1.5, (1000, 800))  # wider than the photo: shrunk
        self.assertLessEqual(x1 - x0, 1000)
        self.assertLessEqual(y1 - y0, 800)


@unittest.skipIf(Image is None, "Pillow missing")
class TestPageRendering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        self.ov = os.path.join(self.tmp.name, "overview.jpg")
        self.cu = os.path.join(self.tmp.name, "closeup.jpg")
        Image.fromarray(rng.integers(0, 120, (1600, 2400, 3), dtype=np.uint8)).save(self.ov)
        Image.fromarray(rng.integers(0, 120, (2000, 3000, 3), dtype=np.uint8)).save(self.cu)

    def tearDown(self):
        self.tmp.cleanup()

    def test_page_size_and_pdf(self):
        ov = AP.PageView(self.ov, 1200, 800, 0.0005, "Overview")
        cu = AP.PageView(self.cu, 1500, 1000, 0.00004, "Close-up")
        page = AP.annotation_page("plot · 1 · CS_ACR", "annotation 1", ov, cu, dpi=100)
        self.assertEqual(page.size, (round(297 / 25.4 * 100), round(210 / 25.4 * 100)))
        out = os.path.join(self.tmp.name, "pages.pdf")
        n = AP.save_pages_pdf((p for p in [page, page]), out)
        self.assertEqual(n, 2)
        with open(out, "rb") as f:
            self.assertTrue(f.read(5).startswith(b"%PDF"))

    def test_bottom_panel_is_100_percent(self):
        # a 10 x 10 px red square 100 px right of the point: in the bottom panel it is still 10 x 10 px
        img = np.full((2000, 3000, 3), 60, np.uint8)
        img[1000:1010, 1600:1610] = (255, 0, 0)
        path = os.path.join(self.tmp.name, "square.png")
        Image.fromarray(img).save(path)
        cu = AP.PageView(path, 1500, 1000, None, "Close-up")
        page = np.asarray(AP.annotation_page("t", "s", None, cu, dpi=100, contrast="none")).astype(int)
        lower = page[int(page.shape[0] * 0.6):]  # below the top panels (whose copy is downscaled)
        red = (lower[..., 0] > 200) & (lower[..., 1] < 50) & (lower[..., 2] < 50)
        rows, cols = np.nonzero(red)
        self.assertEqual(red.sum(), 100)
        self.assertEqual((np.ptp(rows) + 1, np.ptp(cols) + 1), (10, 10))

    def test_overview_only_page(self):
        ov = AP.PageView(self.ov, 100, 100, None, "Overview")
        page = AP.annotation_page("t", "s", ov, None, dpi=100)
        self.assertEqual(page.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
