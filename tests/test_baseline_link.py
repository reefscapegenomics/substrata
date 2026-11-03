"""Tests for the baseline-timepoint link (``baseline_id`` / ``baseline_transform``).

A later timepoint inherits its baseline's frame: ``world_transform`` is a cache
of ``baseline.world_transform @ baseline_transform`` that ``substrata
baseline`` checks and refreshes.
"""

# Standard Library
import argparse
import contextlib
import importlib
import io
import os
import tempfile
import unittest

# Third-party
import numpy as np
import yaml as yaml_lib

from tests.test_initializer_ply_paths import _load_initializer
from tests.test_parent_link import _similarity

ProjectInitializer = _load_initializer()
init_mod = __import__("sys").modules["substrata._real_initializer_for_tests"]

try:  # The CLI imports the heavy conda deps at module load
    cli = importlib.import_module("substrata.cli")
except Exception:  # noqa: BLE001 - any import failure -> skip those tests.
    cli = None

W_B = _similarity(0.2, 30, [5, 0, -40])
T_12 = _similarity(1.02, 3, [0.5, -0.2, 0.1])  # tp2 orig -> tp1 orig
T_23 = _similarity(0.97, -2, [0.1, 0.3, 0.0])  # tp3 orig -> tp2 orig


class _Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def _project(self, pid, keys=None):
        d = os.path.join(self.root, "site", "plot", pid)
        os.makedirs(d, exist_ok=True)
        y = os.path.join(d, f"{pid}.yaml")
        with open(y, "w") as f:
            f.write(f"# keep me\nid: {pid}\npath: {d}\nply: {pid}.ply\n")
        if keys:
            init_mod.set_project_yaml_keys(y, keys)
        return y

    def _baseline_keys(self):
        return {
            "world_transform": W_B,
            "scale_factor": 0.2,
            "up_vector": [0.01, 0.02, 0.19],
            "depth_offset": -40.0,
            "depth_per_unit": 0.19,
        }

    def _cfg(self, y):
        with open(y) as f:
            return yaml_lib.safe_load(f)


class TestBaselineLink(_Base):
    def test_round_trip_and_derivation(self):
        b = ProjectInitializer(yaml=self._project("tp1", self._baseline_keys()))
        y = self._project("tp2", {"baseline_id": "tp1", "baseline_transform": T_12})
        tp = ProjectInitializer(yaml=y)
        self.assertEqual(tp.baseline_id, "tp1")
        np.testing.assert_allclose(tp.baseline_transform, T_12)

        keys = tp.baseline_world_keys(b)
        np.testing.assert_allclose(keys["world_transform"], W_B @ T_12)
        self.assertAlmostEqual(keys["scale_factor"], 0.2 * 1.02)
        self.assertAlmostEqual(keys["depth_per_unit"], 0.19 * 1.02)
        # The depth model gives the same depth for the same physical point
        x = np.array([1.0, 2.0, 3.0])
        x_b = (T_12 @ np.append(x, 1))[:3]
        self.assertAlmostEqual(
            np.dot(keys["up_vector"], x) + keys["depth_offset"],
            np.dot([0.01, 0.02, 0.19], x_b) - 40.0,
        )

    def test_save_config_round_trip(self):
        y = self._project("tp2", {"baseline_id": "tp1", "baseline_transform": T_12})
        out = os.path.join(self.root, "saved.yaml")
        ProjectInitializer(yaml=y).save_config_to_yaml(out)
        with open(out) as f:
            self.assertIn("baseline_transform: [[", f.read())  # flow style
        again = ProjectInitializer(yaml=out)
        self.assertEqual(again.baseline_id, "tp1")
        np.testing.assert_array_equal(again.baseline_transform, T_12)

        # Without the keys nothing is written
        plain = os.path.join(self.root, "plain.yaml")
        ProjectInitializer(yaml=self._project("tp1")).save_config_to_yaml(plain)
        self.assertNotIn("baseline_id", self._cfg(plain))
        self.assertNotIn("baseline_transform", self._cfg(plain))

    def test_link_transform_shapes(self):
        y = self._project("tp2", {"baseline_transform": T_12[:3]})
        np.testing.assert_array_equal(
            ProjectInitializer(yaml=y).baseline_transform, T_12
        )
        init_mod.set_project_yaml_keys(y, {"baseline_transform": [[1, 2], [3, 4]]})
        with self.assertRaises(ValueError):
            ProjectInitializer(yaml=y)

    def test_staleness_and_used_frame(self):
        b = ProjectInitializer(yaml=self._project("tp1", self._baseline_keys()))
        tp = ProjectInitializer(
            yaml=self._project(
                "tp2",
                {"baseline_id": "tp1", "baseline_transform": T_12,
                 "world_transform": W_B @ T_12},
            )
        )
        self.assertFalse(tp.world_transform_is_stale(b))
        np.testing.assert_allclose(
            tp.baseline_world_transform_used(), W_B, atol=1e-12
        )
        # Re-orient the baseline: the link still holds, the cache is stale
        b.world_transform = _similarity(1, 1, [0, 0, 1]) @ W_B
        self.assertTrue(tp.world_transform_is_stale(b))
        np.testing.assert_allclose(
            tp.baseline_world_transform_used(), W_B, atol=1e-12
        )

    def test_set_keys_replaces_block_style_lists(self):
        # PyYAML's default dump writes lists as column-0 "- " items
        y = self._project("tp1")
        with open(y, "a") as f:
            yaml_lib.dump(
                {"world_transform": np.eye(4).tolist(), "up_vector": [0, 0, 1]}, f
            )
        init_mod.set_project_yaml_keys(y, {"world_transform": W_B})
        cfg = self._cfg(y)
        np.testing.assert_allclose(cfg["world_transform"], W_B)
        self.assertEqual(cfg["up_vector"], [0, 0, 1])

    def test_find_baseline_project(self):
        self._project("tp1")
        tp = ProjectInitializer(yaml=self._project("tp2", {"baseline_id": "tp1"}))
        self.assertEqual(
            tp.find_baseline_project(), os.path.join(self.root, "site", "plot", "tp1")
        )

    def test_effective_parent_link(self):
        P = _similarity(0.5, 10, [100, 0, 0])  # tp1 orig -> auv orig
        self._project("tp1", {"parent_id": "auv", "parent_transform": P})
        self._project("tp2", {"baseline_id": "tp1", "baseline_transform": T_12})
        y3 = self._project("tp3", {"baseline_id": "tp2", "baseline_transform": T_23})
        pid, T = ProjectInitializer(yaml=y3).effective_parent_link()
        self.assertEqual(pid, "auv")
        np.testing.assert_allclose(T, P @ T_12 @ T_23, atol=1e-12)

        # Its own parent link wins
        Q = _similarity(1, 0, [1, 1, 1])
        init_mod.set_project_yaml_keys(y3, {"parent_id": "other", "parent_transform": Q})
        pid, T = ProjectInitializer(yaml=y3).effective_parent_link()
        self.assertEqual(pid, "other")
        np.testing.assert_allclose(T, Q)

        # No parent anywhere, a missing baseline, a cycle
        y4 = self._project("tp4", {"baseline_id": "tp5", "baseline_transform": T_12})
        self.assertEqual(ProjectInitializer(yaml=y4).effective_parent_link(), (None, None))
        self._project("tp5", {"baseline_id": "tp4", "baseline_transform": T_23})
        self.assertEqual(ProjectInitializer(yaml=y4).effective_parent_link(), (None, None))
        self.assertEqual(
            ProjectInitializer(yaml=self._project("tp6")).effective_parent_link(),
            (None, None),
        )


@unittest.skipUnless(cli is not None, "requires the open3d/substrata environment")
class TestBaselineCommand(_Base):
    def _run(self, *projects, update=False, link=None):
        args = argparse.Namespace(projects=list(projects), update=update, link=link)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.handle_baseline(args)
        return buf.getvalue()

    def _chain(self):
        y1 = self._project("tp1", self._baseline_keys())
        y2 = self._project(
            "tp2", {"baseline_id": "tp1", "baseline_transform": T_12,
                    "world_transform": W_B @ T_12},
        )
        y3 = self._project(
            "tp3", {"baseline_id": "tp2", "baseline_transform": T_23,
                    "world_transform": W_B @ T_12 @ T_23},
        )
        return y1, y2, y3

    def test_chain_refreshed_after_reorienting_baseline(self):
        y1, y2, y3 = self._chain()
        site = os.path.join(self.root, "site")
        self.assertIn("2 fresh", self._run(site))

        W_new = _similarity(1, 4, [1, 2, 0]) @ W_B
        init_mod.set_project_yaml_keys(y1, {"world_transform": W_new})
        out = self._run(site)
        self.assertIn("2 stale", out)  # tp3 is stale through tp2
        np.testing.assert_allclose(  # report only: nothing written
            self._cfg(y3)["world_transform"], W_B @ T_12 @ T_23
        )

        self._run(site, update=True)
        np.testing.assert_allclose(self._cfg(y2)["world_transform"], W_new @ T_12)
        np.testing.assert_allclose(
            self._cfg(y3)["world_transform"], W_new @ T_12 @ T_23, atol=1e-12
        )
        with open(y3) as f:
            text = f.read()
        self.assertTrue(text.startswith("# keep me\n"))
        self.assertEqual(text.count("world_transform"), 1)
        self.assertAlmostEqual(
            self._cfg(y3)["scale_factor"], 0.2 * 1.02 * 0.97, places=12
        )
        self.assertIn("2 fresh", self._run(site))

    def test_cycle_rejected(self):
        self._project("tp1", {"baseline_id": "tp2", "baseline_transform": T_12})
        self._project("tp2", {"baseline_id": "tp1", "baseline_transform": T_23})
        with self.assertRaises(SystemExit):
            self._run(os.path.join(self.root, "site"))

    def test_link_backfills_existing_cumulative_yaml(self):
        y1 = self._project("tp1", self._baseline_keys())
        y2 = self._project("tp2", {"world_transform": W_B @ T_12})
        plain = self._project("other", {"world_transform": W_B})
        # --link only takes project YAMLs/dirs, not a folder of plots
        with self.assertRaises(SystemExit):
            self._run(os.path.join(self.root, "site"), link=y1)
        self._run(y2, link=y1)
        cfg = self._cfg(y2)
        self.assertEqual(cfg["baseline_id"], "tp1")
        np.testing.assert_allclose(cfg["baseline_transform"], T_12, atol=1e-12)
        np.testing.assert_allclose(cfg["world_transform"], W_B @ T_12, atol=1e-12)
        self.assertNotIn("baseline_id", self._cfg(plain))  # untouched


if __name__ == "__main__":
    unittest.main()
