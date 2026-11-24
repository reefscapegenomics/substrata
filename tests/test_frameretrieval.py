"""substrata.frameretrieval: a lens calibration borrowed from another plot's cams.xml (and turned 180 deg), and the
top-k search over global descriptors.

The parts that build a :class:`substrata.cameras.Sensor` need the real package (open3d, cv2), whose heavy imports
other test modules replace with stubs in ``sys.modules``: they run in a fresh interpreter, skipped outside the
conda env. DINOv2 itself (:class:`GlobalIndex`) is not exercised here (its weights live in the local torch hub
cache): the real-data check is in the changelog.
"""

# Standard Library
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_HEAVY = importlib.util.find_spec("open3d") is not None and importlib.util.find_spec("cv2") is not None

_CAMS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document version="2.0.0">
  <chunk>
    <sensors next_id="3">
      <sensor id="1" label="macro" type="frame">
        <resolution width="8726" height="5840"/>
      </sensor>
      <sensor id="2" label="unknown" type="frame">
        <resolution width="3840" height="2160"/>
        <calibration type="frame" class="adjusted">
          <resolution width="3840" height="2160"/>
          <f>1754.27</f>
          <cx>1.70</cx>
          <cy>9.09</cy>
          <k1>-0.2283</k1>
          <k2>0.0733</k2>
          <k3>-0.0121</k3>
          <p1>-0.000596</p1>
          <p2>0.000999</p2>
        </calibration>
      </sensor>
    </sensors>
  </chunk>
</document>
"""


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _SRC / "substrata" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPure(unittest.TestCase):
    """Parsing and the search (numpy and the standard library only)."""

    @classmethod
    def setUpClass(cls):
        cls.fr = _load("frameretrieval_under_test", "frameretrieval.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.xml = os.path.join(cls.tmp.name, "a.cams.xml")
        with open(cls.xml, "w") as f:
            f.write(_CAMS_XML)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_calibration_params(self):
        p, res, label = self.fr.calibration_params(self.xml, 2)
        self.assertEqual(res, {"width": 3840, "height": 2160})
        self.assertEqual(label, "unknown")
        self.assertAlmostEqual(p["f"], 1754.27)
        self.assertAlmostEqual(p["cy"], 9.09)
        self.assertEqual(p["k4"], 0.0)  # (missing terms are 0)
        self.assertEqual(p["b1"], 0.0)
        with self.assertRaises(ValueError):  # (no adjusted calibration: none of its photos aligned)
            self.fr.calibration_params(self.xml, 1)
        with self.assertRaises(ValueError):
            self.fr.calibration_params(self.xml, 7)

    def test_rotated_params(self):
        p, _res, _l = self.fr.calibration_params(self.xml, 2)
        r = self.fr.rotated_params(p)
        for k in ("cx", "cy", "p1", "p2"):
            self.assertAlmostEqual(r[k], -p[k])
        for k in ("f", "k1", "k2", "k3", "b1", "b2"):
            self.assertAlmostEqual(r[k], p[k])
        self.assertEqual(self.fr.rotated_params(r), p)

    def test_top_k(self):
        import numpy as np

        rng = np.random.default_rng(0)
        common = rng.normal(size=16) * 10  # (a large part all descriptors share: taken off)
        index = common + rng.normal(size=(50, 16))
        query = index[[7, 21]] + 0.05 * rng.normal(size=(2, 16))
        idx, sim = self.fr.top_k(query, index, k=3)
        self.assertEqual(idx.shape, (2, 3))
        self.assertEqual(list(idx[:, 0]), [7, 21])
        self.assertTrue(np.all(np.diff(sim, axis=1) <= 0))
        # several versions of a query (e.g. turned): the best counts
        turned = np.stack([common + rng.normal(size=16), query[0]])[None]
        idx2, _s = self.fr.top_k(turned, index, k=1)
        self.assertEqual(idx2[0, 0], 7)
        self.assertEqual(self.fr.top_k(query, index, k=500)[0].shape, (2, 50))


@unittest.skipUnless(_HEAVY, "needs the conda env (open3d, cv2)")
class TestSensors(unittest.TestCase):
    """A borrowed Sensor in a fresh interpreter (the real substrata.cameras)."""

    def _run(self, code):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "a.cams.xml")
            with open(xml, "w") as f:
                f.write(_CAMS_XML)
            src = f"import sys; sys.path.insert(0, {str(_SRC)!r}); XML = {xml!r}\n" + textwrap.dedent(code)
            r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_borrowed_and_turned_sensor(self):
        self._run("""
            import numpy as np
            from substrata import frameretrieval as fr
            from substrata.cameras import camera_points_to_pixels
            s = fr.sensor_from_cams_xml(XML, 2)
            assert s.calibration_class == "borrowed" and s.sensor_id == 2, (s.calibration_class, s.sensor_id)
            assert abs(s.cx - (1920 + 1.70)) < 1e-9 and abs(s.cy - (1080 + 9.09)) < 1e-9
            r = fr.sensor_from_cams_xml(XML, 2, rotate180=True, as_sensor_id=5)
            assert r.sensor_id == 5
            # a camera turned 180 deg about its axis sees point (x, y, z) where the other sees (-x, -y, z), at the
            # image point turned round: (W - u, H - v)
            rng = np.random.default_rng(1)
            P = np.c_[rng.uniform(-0.8, 0.8, (200, 2)) * 0.5, np.full(200, 0.5)]
            u, v, ok = camera_points_to_pixels(P[:, 0], P[:, 1], P[:, 2], s)
            u2, v2, ok2 = camera_points_to_pixels(-P[:, 0], -P[:, 1], P[:, 2], r)
            assert np.allclose(u2, 3840 - u, atol=1e-6) and np.allclose(v2, 2160 - v, atol=1e-6)
            assert np.array_equal(ok, ok2)
            assert np.allclose(fr.rotated_180(s).cx, r.cx) and np.allclose(fr.rotated_180(s).p1, r.p1)
            same = fr.calibration_difference(s, s)
            assert same["px_median"] < 1e-6 and same["deg_max"] < 1e-4, same  # (arccos near 1)
            d = fr.calibration_difference(s, r)
            assert d["px_median"] > 5, d  # (the upright lens and the turned one differ)
        """)

    def test_give_sensor(self):
        self._run("""
            from types import SimpleNamespace
            from substrata import frameretrieval as fr
            s = fr.sensor_from_cams_xml(XML, 2)
            cams = SimpleNamespace(sensors={}, data={"a": SimpleNamespace(sensor_id=2, sensor=None),
                                                     "b": SimpleNamespace(sensor_id=1, sensor=None)})
            assert fr.give_sensor(cams, s) == 1
            assert cams.data["a"].sensor is s and cams.data["b"].sensor is None and cams.sensors[2] is s
        """)


if __name__ == "__main__":
    unittest.main()
