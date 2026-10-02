# Standard Library
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml as yaml_lib

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_initializer():
    """Load :mod:`substrata.initializer` without importing the package."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    def _load(mod_name: str, rel: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(mod_name, _SUB / rel)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {mod_name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    if "substrata.settings" not in sys.modules:
        _load("substrata.settings", "settings.py")
    if "substrata.geom" not in sys.modules:
        _load("substrata.geom", "geom.py")
    if "substrata.annotations" not in sys.modules:
        ann = types.ModuleType("substrata.annotations")
        ann.Annotations = object
        ann.Scalebars = object
        sys.modules["substrata.annotations"] = ann
    sys.modules["substrata"].settings = sys.modules["substrata.settings"]
    sys.modules["substrata"].geom = sys.modules["substrata.geom"]
    sys.modules["substrata"].annotations = sys.modules["substrata.annotations"]

    # Load the real module under a private name: other test files register a
    # stubbed "substrata.initializer" in sys.modules, and reusing that stub
    # would silently test `object` instead of ProjectInitializer.
    mod = _load("substrata._real_initializer_for_tests", "initializer.py")
    return mod.ProjectInitializer


class TestYamlDerivedPlyPaths(unittest.TestCase):
    """``init_with_yaml`` must populate ply_dec_path / ply_full_path.

    Those attributes were previously set only by ``init_with_path``, so in any
    folder containing a ``<dirname>.yaml`` the commands that resolve their input
    through ``ply_full_path`` (decimate, ply-repair, head, scalebars,
    ``segment --full-ply``) failed with "No input PLY found" despite a valid
    ``ply:`` key.
    """

    def _project(self, tmp, ply_names):
        proj = os.path.join(tmp, "cur_sna")
        os.makedirs(proj, exist_ok=True)
        for name in ply_names:
            with open(os.path.join(proj, name), "wb") as f:
                f.write(b"ply\n")
        return proj

    def _write_yaml(self, proj, config):
        path = os.path.join(proj, "cur_sna.yaml")
        with open(path, "w") as f:
            yaml_lib.dump(config, f)
        return path

    def test_full_ply_populated_from_convention(self) -> None:
        ProjectInitializer = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp, ["cur_sna.ply"])
            self._write_yaml(
                proj, {"path": proj, "id": "cur_sna", "ply": "cur_sna.ply"}
            )
            init = ProjectInitializer(path=proj)

            self.assertEqual(init.ply_full_path, os.path.join(proj, "cur_sna.ply"))
            self.assertIsNone(init.ply_dec_path)

    def test_both_ply_paths_populated(self) -> None:
        ProjectInitializer = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp, ["cur_sna.ply", "cur_sna_dec50M.ply"])
            self._write_yaml(
                proj, {"path": proj, "id": "cur_sna", "ply": "cur_sna_dec50M.ply"}
            )
            init = ProjectInitializer(path=proj)

            self.assertEqual(
                init.ply_dec_path, os.path.join(proj, "cur_sna_dec50M.ply")
            )
            self.assertEqual(init.ply_full_path, os.path.join(proj, "cur_sna.ply"))
            # ply_filepath itself is whatever the YAML asked for.
            self.assertEqual(
                init.ply_filepath, os.path.join(proj, "cur_sna_dec50M.ply")
            )

    def test_non_conventional_ply_counts_as_full(self) -> None:
        ProjectInitializer = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp, ["custom_cloud.ply"])
            self._write_yaml(
                proj, {"path": proj, "id": "cur_sna", "ply": "custom_cloud.ply"}
            )
            init = ProjectInitializer(path=proj)

            self.assertEqual(
                init.ply_full_path, os.path.join(proj, "custom_cloud.ply")
            )

    def test_missing_ply_leaves_derived_paths_none(self) -> None:
        ProjectInitializer = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp, [])
            self._write_yaml(
                proj, {"path": proj, "id": "cur_sna", "ply": "gone.ply"}
            )
            init = ProjectInitializer(path=proj)

            self.assertIsNone(init.ply_dec_path)
            self.assertIsNone(init.ply_full_path)


if __name__ == "__main__":
    unittest.main()
