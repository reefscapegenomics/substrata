"""Tests for the parent-plot link (``parent_id`` / ``parent_transform``)."""

# Standard Library
import os
import tempfile
import unittest

# Third-party
import numpy as np
import yaml as yaml_lib

from tests.test_initializer_ply_paths import _load_initializer

ProjectInitializer = _load_initializer()
init_mod = __import__("sys").modules["substrata._real_initializer_for_tests"]


def _similarity(scale, yaw_deg, t):
    a = np.radians(yaw_deg)
    m = np.eye(4)
    m[:3, :3] = scale * np.array(
        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    )
    m[:3, 3] = t
    return m


class TestParentLink(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def _project(self, rel, pid, extra=""):
        d = os.path.join(self.root, rel)
        os.makedirs(d, exist_ok=True)
        y = os.path.join(d, f"{pid}.yaml")
        with open(y, "w") as f:
            f.write(f"# keep me\nid: {pid}\npath: {d}\nply: {pid}.ply\n{extra}")
        return y

    def test_round_trip(self):
        y = self._project("child", "child")
        T = _similarity(1.01, 5, [1, 2, 3])
        init_mod.set_project_yaml_keys(
            y, {"parent_id": "par", "parent_transform": T}
        )
        init = ProjectInitializer(yaml=y)
        self.assertEqual(init.parent_id, "par")
        np.testing.assert_allclose(init.parent_transform, T)
        with open(y) as f:
            text = f.read()
        self.assertTrue(text.startswith("# keep me\n"))
        self.assertIn("ply: child.ply", text)  # relative path kept

        # Setting again replaces the old block instead of duplicating it
        init_mod.set_project_yaml_keys(y, {"parent_transform": np.eye(4)})
        cfg = yaml_lib.safe_load(open(y))
        np.testing.assert_allclose(cfg["parent_transform"], np.eye(4))
        with open(y) as f:
            self.assertEqual(f.read().count("parent_transform"), 1)

        # save_config_to_yaml writes both keys too
        out = os.path.join(self.root, "saved.yaml")
        init.save_config_to_yaml(out)
        cfg = yaml_lib.safe_load(open(out))
        self.assertEqual(cfg["parent_id"], "par")

    def test_world_composition_and_scale_ratio(self):
        child = ProjectInitializer(yaml=self._project("child", "child"))
        parent = ProjectInitializer(yaml=self._project("par", "par"))
        child.world_transform = _similarity(0.2, 30, [5, 0, -40])
        parent.world_transform = _similarity(0.3, -10, [60, 150, -70])
        # Both plots agree on scale except a 0.5 % offset of the parent's
        # scalebars; T maps raw child -> raw parent.
        ratio = 1.005
        Mw = _similarity(ratio, 2, [30, 190, 0])
        T = np.linalg.inv(parent.world_transform) @ Mw @ child.world_transform
        child.parent_transform = T
        np.testing.assert_allclose(
            child.to_parent_world_transform(parent), Mw, atol=1e-9
        )
        self.assertAlmostEqual(child.parent_scale_ratio(parent), ratio)

    def test_find_parent_project(self):
        y = self._project(os.path.join("site", "focal", "child"), "child",
                          "parent_id: auv\n")
        auv = os.path.join(self.root, "site", "auv_plots", "auv")
        os.makedirs(auv)
        init = ProjectInitializer(yaml=y)
        self.assertEqual(init.find_parent_project(), auv)
        other = os.path.join(self.root, "elsewhere", "auv")
        os.makedirs(other)
        self.assertEqual(
            init.find_parent_project([os.path.join(self.root, "elsewhere")]), other
        )
        init.parent_id = "nope"
        self.assertIsNone(init.find_parent_project())


if __name__ == "__main__":
    unittest.main()
