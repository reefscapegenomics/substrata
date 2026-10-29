"""Tests for annotation sidecar YAMLs (``<csv>.yaml``) and their use by the
initializer.

Loads ``sidecars.py`` and ``initializer.py`` directly (stubbing the heavy
``annotations`` module), so no open3d/opencv is needed.
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
import yaml as yaml_lib

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load(mod_name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, _SUB / rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {mod_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    """Return (sidecars, ProjectInitializer) without importing the package."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
    for name in ("settings", "geom"):
        if f"substrata.{name}" not in sys.modules:
            _load(f"substrata.{name}", f"{name}.py")
    if "substrata.annotations" not in sys.modules:
        ann = types.ModuleType("substrata.annotations")
        ann.Annotations = object
        ann.Scalebars = object
        sys.modules["substrata.annotations"] = ann
    sidecars = _load("substrata.sidecars", "sidecars.py")
    for name in ("settings", "geom", "annotations", "sidecars"):
        setattr(sys.modules["substrata"], name, sys.modules[f"substrata.{name}"])
    init = _load("substrata._real_initializer_for_sidecar_tests", "initializer.py")
    return sidecars, init.ProjectInitializer


class TestSidecarReadWrite(unittest.TestCase):
    def setUp(self):
        self.sidecars, _ = _load_modules()
        self.tmp = tempfile.TemporaryDirectory()
        self.proj = os.path.join(self.tmp.name, "cur_sna")
        os.makedirs(self.proj)
        self.csv = os.path.join(self.proj, "cur_sna_ann.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_name_keeps_csv_extension(self):
        csv = os.path.join(self.proj, "cur_sna.csv")
        self.assertEqual(
            self.sidecars.annotation_sidecar_path(csv),
            os.path.join(self.proj, "cur_sna.csv.yaml"),
        )

    def test_missing_sidecar_is_empty(self):
        self.assertEqual(self.sidecars.read_annotation_sidecar(self.csv), {})

    def test_round_trip_with_relative_paths(self):
        classes = os.path.join(self.tmp.name, "classes.csv")
        crops = "/abs/crops"
        wt = np.diag([2.0, 2.0, 2.0, 1.0])
        path = self.sidecars.write_annotation_sidecar(
            self.csv,
            {
                "type": "annotations",
                "classes": classes,
                "cropped_path": crops,
                "world_transform": wt,
                "last_highest_id": np.int64(12),
                "classifier": None,
            },
        )
        with open(path) as f:
            raw = yaml_lib.safe_load(f)
        # Stored relative to the sidecar's folder
        self.assertEqual(raw["classes"], os.path.join("..", "classes.csv"))
        self.assertNotIn("classifier", raw)
        self.assertEqual(raw["last_highest_id"], 12)

        cfg = self.sidecars.read_annotation_sidecar(self.csv)
        self.assertEqual(cfg["classes"], classes)
        self.assertEqual(cfg["cropped_path"], crops)
        np.testing.assert_allclose(cfg["world_transform"], wt)

    def test_update_keeps_other_keys(self):
        self.sidecars.write_annotation_sidecar(self.csv, {"type": "annotations"})
        self.sidecars.write_annotation_sidecar(self.csv, {"last_highest_id": 3})
        cfg = self.sidecars.read_annotation_sidecar(self.csv)
        self.assertEqual(cfg["type"], "annotations")
        self.assertEqual(cfg["last_highest_id"], 3)
        self.sidecars.write_annotation_sidecar(self.csv, {"x": 1}, update=False)
        self.assertEqual(self.sidecars.read_annotation_sidecar(self.csv), {"x": 1})


class TestInitializerSidecar(unittest.TestCase):
    def setUp(self):
        self.sidecars, self.ProjectInitializer = _load_modules()
        self.tmp = tempfile.TemporaryDirectory()
        self.proj = os.path.join(self.tmp.name, "cur_sna")
        os.makedirs(self.proj)
        # Annotations named like the project: its sidecar must not be the
        # project YAML
        self.csv = os.path.join(self.proj, "cur_sna.csv")
        with open(self.csv, "w") as f:
            f.write("id,orig_x,orig_y,orig_z,label\n")
        self.project_yaml = os.path.join(self.proj, "cur_sna.yaml")
        with open(self.project_yaml, "w") as f:
            yaml_lib.dump(
                {
                    "path": self.proj,
                    "id": "cur_sna",
                    "annotations": "cur_sna.csv",
                    "classes": "classes.csv",
                    "cropped_path": "old_crops",
                    "annotations_last_highest_id": 50,
                },
                f,
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_bare_csv_keeps_project_keys(self):
        init = self.ProjectInitializer(path=self.proj)
        self.assertEqual(init.cropped_path, os.path.join(self.proj, "old_crops"))
        self.assertEqual(init.classes_filepath, os.path.join(self.proj, "classes.csv"))
        self.assertEqual(init.annotations_last_highest_id, 50)

    def test_sidecar_overrides_project_keys(self):
        self.sidecars.write_annotation_sidecar(
            self.csv,
            {
                "classes": os.path.join(self.tmp.name, "shared_classes.csv"),
                "cropped_path": os.path.join(self.proj, "my_crops"),
                "crop_size": 1600,
                "last_highest_id": 80,
            },
        )
        self.assertNotEqual(
            self.sidecars.annotation_sidecar_path(self.csv), self.project_yaml
        )
        init = self.ProjectInitializer(path=self.proj)
        self.assertEqual(init.cropped_path, os.path.join(self.proj, "my_crops"))
        self.assertEqual(
            init.classes_filepath, os.path.join(self.tmp.name, "shared_classes.csv")
        )
        self.assertEqual(init.crop_size, 1600)
        self.assertEqual(init.annotations_last_highest_id, 80)

        # Saving the project YAML leaves the sidecar's keys in the sidecar
        out = os.path.join(self.tmp.name, "saved.yaml")
        init.save_config_to_yaml(out)
        with open(out) as f:
            saved = yaml_lib.safe_load(f)
        for key in ("classes", "cropped_path", "crop_size"):
            self.assertNotIn(key, saved)

    def test_last_highest_id_never_decreases(self):
        self.sidecars.write_annotation_sidecar(self.csv, {"last_highest_id": 10})
        init = self.ProjectInitializer(path=self.proj)
        self.assertEqual(init.annotations_last_highest_id, 50)

    def test_relative_cropped_path_resolves_against_sidecar(self):
        with open(self.sidecars.annotation_sidecar_path(self.csv), "w") as f:
            yaml_lib.dump({"cropped_path": "crops/"}, f)
        init = self.ProjectInitializer(path=self.proj)
        self.assertEqual(init.cropped_path, os.path.join(self.proj, "crops"))


if __name__ == "__main__":
    unittest.main()
