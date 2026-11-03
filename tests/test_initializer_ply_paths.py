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


class TestAnnosConvention(unittest.TestCase):
    """Annotation sets live in <id>.annos/; there is no default set."""

    def test_annos_root_and_no_default_set(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            annos = os.path.join(proj, "cur_sna.annos")
            os.makedirs(annos)
            point = "id,orig_x,orig_y,orig_z,label\np1,1,2,3,X\n"
            # The old folder-convention default is no longer picked up
            for path in (
                os.path.join(proj, "cur_sna_annotations.csv"),
                os.path.join(annos, "b_ann.csv"),
                os.path.join(annos, "a_intercepts.csv"),
            ):
                with open(path, "w") as f:
                    f.write(point)
            with open(os.path.join(annos, "a_intercepts.csv.yaml"), "w") as f:
                f.write("type: intercepts\n")
            init = PI(path=proj)
            self.assertIsNone(init.annotations_filepath)
            self.assertEqual(init.annos_root, annos)
            self.assertEqual(
                [os.path.basename(p) for p in init.annotation_csvs()],
                ["a_intercepts.csv", "b_ann.csv"],
            )

    def test_yaml_annotations_key_names_the_default_set(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            os.makedirs(os.path.join(proj, "cur_sna.annos"))
            y = os.path.join(proj, "cur_sna.yaml")
            with open(y, "w") as f:
                yaml_lib.dump(
                    {"id": "cur_sna", "path": proj,
                     "annotations": "cur_sna.annos/cur_sna_ann.csv"}, f
                )
            init = PI(yaml=y)
            self.assertEqual(
                init.annotations_filepath,
                os.path.join(proj, "cur_sna.annos", "cur_sna_ann.csv"),
            )
            self.assertEqual(init.annotation_csvs(), [])  # folder empty

    def test_legacy_cropped_folder_not_picked_up(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            os.makedirs(os.path.join(proj, "cur_sna.cropped"))
            os.makedirs(os.path.join(proj, "cur_sna.thumbnails"))
            init = PI(path=proj)
            # Crops belong to a set (<csv stem>.cropped), not the project
            self.assertIsNone(init.cropped_path)
            self.assertEqual(
                init.thumbnail_path, os.path.join(proj, "cur_sna.thumbnails")
            )

    def _touch(self, *parts):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_classes_default_prefers_annos(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            legacy = self._touch(proj, "classes.csv")
            self.assertEqual(PI(path=proj).classes_filepath, legacy)
            annos = self._touch(proj, "cur_sna.annos", "classes.csv")
            init = PI(path=proj)
            self.assertEqual(init.classes_filepath, annos)
            self.assertTrue(init.classes_from_convention)

    def test_classes_next_to_csv_and_explicit_key_wins(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            self._touch(proj, "cur_sna.annos", "classes.csv")
            beside = self._touch(proj, "other", "classes.csv")
            self._touch(proj, "other", "x_ann.csv")
            y = os.path.join(proj, "cur_sna.yaml")
            config = {"id": "cur_sna", "path": proj,
                      "annotations": "other/x_ann.csv"}
            with open(y, "w") as f:
                yaml_lib.dump(config, f)
            init = PI(yaml=y)
            self.assertEqual(init.classes_filepath, beside)
            # A conventional default is not written back to the project YAML
            out = os.path.join(tmp, "saved.yaml")
            init.save_config_to_yaml(out)
            with open(out) as f:
                self.assertNotIn("classes", yaml_lib.safe_load(f))

            shared = self._touch(tmp, "shared.csv")
            config["classes"] = shared
            with open(y, "w") as f:
                yaml_lib.dump(config, f)
            init = PI(yaml=y)
            self.assertEqual(init.classes_filepath, shared)
            self.assertFalse(init.classes_from_convention)

    def test_use_annotations_file_applies_its_own_sidecar(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            annos = os.path.join(proj, "cur_sna.annos")
            self._touch(annos, "classes.csv")
            self._touch(annos, "a_ann.csv")
            self._touch(annos, "cur_sna_slope_intercepts.csv")
            with open(os.path.join(annos, "a_ann.csv.yaml"), "w") as f:
                yaml_lib.dump({"cropped_path": "/elsewhere/crops",
                               "last_highest_id": 620, "crop_size": 400}, f)
            with open(os.path.join(
                    annos, "cur_sna_slope_intercepts.csv.yaml"), "w") as f:
                yaml_lib.dump({"type": "intercepts", "classes": "classes.csv"}, f)
            y = os.path.join(proj, "cur_sna.yaml")
            with open(y, "w") as f:
                yaml_lib.dump({"id": "cur_sna", "path": proj,
                               "annotations": "cur_sna.annos/a_ann.csv"}, f)
            init = PI(yaml=y)
            self.assertEqual(init.cropped_path, "/elsewhere/crops")
            init.use_annotations_file(
                os.path.join(annos, "cur_sna_slope_intercepts.csv"))
            # The previous set's crops folder and ids do not carry over
            self.assertIsNone(init.cropped_path)
            self.assertIsNone(init.annotations_last_highest_id)
            self.assertEqual(init.crop_size, 400)
            self.assertEqual(init.annotation_sidecar["type"], "intercepts")
            self.assertEqual(
                init.classes_filepath, os.path.join(annos, "classes.csv"))

    def test_reset_paths_keeps_relative_layout(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            site = os.path.join(tmp, "site")
            proj = os.path.join(site, "cur_sna")
            shared = self._touch(site, "shared_classes.csv")
            self._touch(proj, "cur_sna.annos", "a_ann.csv")
            y = os.path.join(proj, "cur_sna.yaml")
            old = "/Volumes/other_machine/site/cur_sna"
            with open(y, "w") as f:
                yaml_lib.dump({
                    "id": "cur_sna", "path": old,
                    "ply": "cur_sna.ply",
                    "annotations": "cur_sna.annos/a_ann.csv",
                    "photos_path": "cur_sna.photos/",
                    "classes": "../shared_classes.csv",
                    "classifier": "/Volumes/models/caribbean.pkl",
                }, f)
            init = PI(yaml=y)
            init.reset_paths_to_local(proj)
            self.assertEqual(init.path, proj)
            self.assertEqual(
                init.annotations_filepath,
                os.path.join(proj, "cur_sna.annos", "a_ann.csv"),
            )
            self.assertEqual(init.ply_filepath, os.path.join(proj, "cur_sna.ply"))
            self.assertEqual(init.photos_path, os.path.join(proj, "cur_sna.photos"))
            # Outside the base: relative place kept when it exists there...
            self.assertEqual(init.classes_filepath, shared)
            # ...otherwise looked for in the new base by name
            self.assertEqual(
                init.classifier_filepath, os.path.join(proj, "caribbean.pkl")
            )

    def test_local_flag_applies_the_moved_sidecar(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            proj = os.path.join(tmp, "cur_sna")
            annos = os.path.join(proj, "cur_sna.annos")
            self._touch(annos, "a_ann.csv")
            with open(os.path.join(annos, "a_ann.csv.yaml"), "w") as f:
                yaml_lib.dump({"last_highest_id": 7}, f)
            y = os.path.join(proj, "cur_sna.yaml")
            with open(y, "w") as f:
                yaml_lib.dump({"id": "cur_sna", "path": "/gone/cur_sna",
                               "annotations": "cur_sna.annos/a_ann.csv"}, f)
            cwd = os.getcwd()
            os.chdir(proj)
            try:
                init = PI(yaml=y, local=True)
            finally:
                os.chdir(cwd)
            self.assertEqual(
                os.path.realpath(init.annotations_filepath),
                os.path.realpath(os.path.join(annos, "a_ann.csv")),
            )
            self.assertEqual(init.annotations_last_highest_id, 7)

    def test_no_path_means_no_annos_root(self) -> None:
        PI = _load_initializer()
        with tempfile.TemporaryDirectory() as tmp:
            init = PI(path=tmp)
        init.path = None
        self.assertIsNone(init.annos_root)
        self.assertEqual(init.annotation_csvs(), [])


if __name__ == "__main__":
    unittest.main()
