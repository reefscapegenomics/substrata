"""Tests for storing the colour correction in the YAML instead of the PLY.

``substrata colors`` writes ``color_correction`` to the project YAML and
loading applies it; ``decimate -c`` bakes it into a PLY whose header is marked
so it is never applied twice. ``pointclouds.py`` imports the heavy conda deps
at module load, so these tests are skipped when they are not installed.
"""

# Standard Library
import importlib
import os
import struct
import tempfile
import unittest

# Third-party
import numpy as np
import yaml as yaml_lib

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    pc = importlib.import_module("substrata.pointclouds")
    init_mod = importlib.import_module("substrata.initializer")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    pc = None

# Halve red, keep green, add 10 to blue (0-255 space)
CC = {
    "matrix": np.diag([0.5, 1.0, 1.0]),
    "offset": np.array([0.0, 0.0, 10.0]),
}
RGB = [(200, 100, 50), (100, 50, 0), (0, 0, 0)]


def _write_ply(path, rgb=RGB):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rgb)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        for i, (r, g, b) in enumerate(rgb):
            f.write(struct.pack("<fffBBB", i, 0.0, 0.0, r, g, b))


def _read_rgb(path):
    with open(path, "rb") as f:
        data = f.read()
    body = data[data.index(b"end_header\n") + len(b"end_header\n"):]
    return [struct.unpack_from("<fffBBB", body, i * 15)[3:] for i in range(3)]


@unittest.skipUnless(pc is not None, "requires the open3d/substrata environment")
class TestColorCorrectedPly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src.ply")
        _write_ply(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_decimate_marks_corrected_output(self):
        out = os.path.join(self.tmp.name, "cc.ply")
        self.assertFalse(pc.ply_is_color_corrected(self.src))
        pc.decimate_ply_file(self.src, out, 10, show_progress=False,
                             color_correction=CC)
        self.assertTrue(pc.ply_is_color_corrected(out))
        self.assertEqual(_read_rgb(out)[0], (100, 100, 60))

    def test_marker_kept_and_double_correction_refused(self):
        out = os.path.join(self.tmp.name, "cc.ply")
        pc.decimate_ply_file(self.src, out, 10, show_progress=False,
                             color_correction=CC)
        again = os.path.join(self.tmp.name, "cc_dec.ply")
        pc.decimate_ply_file(out, again, 2, show_progress=False)
        self.assertTrue(pc.ply_is_color_corrected(again))
        with self.assertRaises(ValueError):
            pc.decimate_ply_file(out, again, 10, show_progress=False,
                                 color_correction=CC)

    def test_plain_decimate_is_not_marked(self):
        out = os.path.join(self.tmp.name, "dec.ply")
        pc.decimate_ply_file(self.src, out, 2, show_progress=False)
        self.assertFalse(pc.ply_is_color_corrected(out))


@unittest.skipUnless(pc is not None, "requires the open3d/substrata environment")
class TestYamlColorCorrection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proj = os.path.join(self.tmp.name, "cur_sna")
        os.makedirs(self.proj)
        self.yaml = os.path.join(self.proj, "cur_sna.yaml")
        with open(self.yaml, "w") as f:
            f.write("# keep me\nid: cur_sna\npath: " + self.proj
                    + "\nply: cur_sna.ply\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _colors(self, ply_name):
        init_mod.set_project_yaml_keys(
            self.yaml, {"ply": ply_name, "color_correction": CC}
        )
        p = init_mod.ProjectInitializer(yaml=self.yaml)
        p.initialize()
        return np.round(np.asarray(p.pcd.colors) * 255).astype(int)

    def test_set_keys_writes_nested_correction_in_place(self):
        init_mod.set_project_yaml_keys(self.yaml, {"color_correction": CC})
        with open(self.yaml) as f:
            text = f.read()
        self.assertTrue(text.startswith("# keep me\n"))
        self.assertIn("ply: cur_sna.ply", text)
        cfg = yaml_lib.safe_load(text)
        np.testing.assert_allclose(cfg["color_correction"]["matrix"], CC["matrix"])
        p = init_mod.ProjectInitializer(yaml=self.yaml)
        np.testing.assert_allclose(p.color_correction["offset"], CC["offset"])

    def test_initialize_applies_once(self):
        _write_ply(os.path.join(self.proj, "cur_sna.ply"))
        pc.decimate_ply_file(
            os.path.join(self.proj, "cur_sna.ply"),
            os.path.join(self.proj, "cur_sna_cc.ply"),
            10, show_progress=False, color_correction=CC,
        )
        # Plain PLY: the YAML correction is applied on load
        self.assertEqual(tuple(self._colors("cur_sna.ply")[0]), (100, 100, 60))
        # Marked PLY: already corrected, not applied again
        self.assertEqual(tuple(self._colors("cur_sna_cc.ply")[0]), (100, 100, 60))


if __name__ == "__main__":
    unittest.main()
