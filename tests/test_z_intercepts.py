"""Intercepts from a transformed cloud carry its world_transform.

Needs the real ``substrata`` package (pointclouds + annotations), whose heavy
imports other test modules replace with stubs in ``sys.modules``. The check
therefore runs in a fresh interpreter; skipped outside the conda env.
"""

# Standard Library
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import open3d as o3d
    from substrata.pointclouds import PointCloud

    g = np.mgrid[0:20, 0:20].reshape(2, -1).T / 2.0
    pc = PointCloud()
    pc.o3d_pcd.points = o3d.utility.Vector3dVector(np.c_[g, np.zeros(len(g))])
    t = np.eye(4)
    t[:3, :3] = 2.0 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t[:3, 3] = [5.0, 1.0, -30.0]
    pc.apply_transform(t)

    ints = pc.get_z_intercepts(np.array([[2.0, 3.0], [-4.0, 8.0]]), 0.6)
    assert np.allclose(ints.world_transform, pc.world_transform)
    for a in ints.data.values():
        back = t @ np.r_[a.orig_coords, 1.0]
        assert np.allclose(back[:3], a.coords), (a.orig_coords, a.coords)
        assert np.isclose(a.coords[2], -30.0)

    # The depth regression travels with the intercepts
    pc.up_vector = t[2, :3]
    pc.depth_offset = float(t[2, 3])
    pc.depth_scale_factor = 2.0
    ints = pc.get_z_intercepts(np.array([[2.0, 3.0]]), 0.6)
    assert np.allclose(ints.up_vector, pc.up_vector)
    assert ints.depth_offset == pc.depth_offset and ints.depth_per_unit == 2.0
    assert all(np.isclose(a.depth_in_m, -30.0) for a in ints.data.values())
    assert len(ints.subset_by_depth_range(29, 31)) == 1
    print("OK")
    """
)


@unittest.skipUnless(
    importlib.util.find_spec("open3d") is not None,
    "requires the open3d/substrata environment",
)
class TestZInterceptsTransform(unittest.TestCase):
    def test_intercepts_match_pcd_frame(self):
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            capture_output=True,
            text=True,
            cwd=str(_SRC),
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
