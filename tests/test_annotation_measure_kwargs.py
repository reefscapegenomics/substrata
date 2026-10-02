"""Tests that ``Annotation.measure`` forwards keyword arguments.

``annotations.py`` imports the heavy conda deps at module load, so these tests
are skipped when they are not installed.
"""

# Standard Library
import importlib
import types
import unittest

try:  # Heavy deps (open3d, cv2, …) are only present in the conda env.
    # The package star-imports ``from __future__ import annotations``, which
    # shadows the submodule attribute, so load it by its module path.
    a = importlib.import_module("substrata.annotations")
except Exception:  # noqa: BLE001 - any import failure -> skip the module.
    a = None


@unittest.skipUnless(a is not None, "requires the open3d/substrata environment")
class TestMeasureGapFractionKwargs(unittest.TestCase):
    def test_kwargs_reach_calc_gap_fraction(self):
        received = {}

        def calc_gap_fraction(annotation, pcd, **kwargs):
            received.update(kwargs)
            return 0.1, 0.2, "image"

        ann = types.SimpleNamespace(id="ann1", measurements={})
        a.Annotation.measure(
            ann, calc_gap_fraction, "pcd", generate_image=True, show_rings=True
        )
        self.assertEqual(received, {"show_rings": True})
        self.assertEqual(ann.measurements["gapF_image"], "image")


if __name__ == "__main__":
    unittest.main()
