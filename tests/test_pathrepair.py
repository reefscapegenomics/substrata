# Standard Library
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load(mod_name: str, rel: str) -> types.ModuleType:
    """Load a module directly from ``src/substrata`` without running __init__."""
    path = _SUB / rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_package() -> None:
    """Register a synthetic ``substrata`` package so __init__.py never runs."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg
    if "substrata.settings" not in sys.modules:
        _load("substrata.settings", "settings.py")
    sys.modules["substrata"].settings = sys.modules["substrata.settings"]
    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")
        log_mod.logger = logging.getLogger("substrata")

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        log_mod.tqdm_joblib = tqdm_joblib
        sys.modules["substrata.logging"] = log_mod


def _load_pathrepair() -> types.ModuleType:
    """Load :mod:`substrata.pathrepair` with no heavy dependencies."""
    _fake_package()
    if "substrata.pathrepair" not in sys.modules:
        _load("substrata.pathrepair", "pathrepair.py")
    return sys.modules["substrata.pathrepair"]


def _load_cameras_class():
    """Load :class:`Cameras` without importing the ``substrata`` package."""
    _load_pathrepair()
    if "substrata.geom" not in sys.modules:
        _load("substrata.geom", "geom.py")
    for name in ("substrata.visualizations", "substrata.measurements"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if "substrata.cameras" not in sys.modules:
        _load("substrata.cameras", "cameras.py")
    return sys.modules["substrata.cameras"].Cameras


def _touch(path: str, content: bytes = b"x") -> str:
    """Create a file (and its parents) with the given content."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


class TestPathSplitting(unittest.TestCase):
    """Separator-agnostic path splitting for Metashape's Windows paths."""

    def test_path_basename_handles_windows_paths(self) -> None:
        cp = _load_pathrepair()
        self.assertEqual(
            cp.path_basename(r"D:\photos\dive1\IMG_0001.JPG"), "IMG_0001.JPG"
        )
        self.assertEqual(
            cp.path_basename("/mnt/a/dive1/IMG_0001.JPG"), "IMG_0001.JPG"
        )
        self.assertEqual(cp.path_basename("IMG_0001.JPG"), "IMG_0001.JPG")
        self.assertEqual(cp.path_basename(""), "")

    def test_path_dirname_normalises_separators(self) -> None:
        cp = _load_pathrepair()
        self.assertEqual(cp.path_dirname("D:\\photos\\dive1\\"), "D:/photos")
        self.assertEqual(
            cp.path_dirname(r"D:\photos\dive1\IMG_0001.JPG"), "D:/photos/dive1"
        )
        self.assertEqual(
            cp.path_dirname("/mnt/a/IMG_0001.JPG"), "/mnt/a"
        )
        self.assertEqual(cp.path_dirname("IMG_0001.JPG"), "")


class TestBuildImageIndex(unittest.TestCase):
    """Indexing images by basename, with pruning and case folding."""

    def test_index_case_folding_and_pruning(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "a", "IMG_0001.JPG"))
            _touch(os.path.join(tmp, "b", "img_0001.jpg"))
            _touch(os.path.join(tmp, ".hidden", "IMG_0002.JPG"))
            _touch(os.path.join(tmp, "training_crops", "IMG_0003.JPG"))
            _touch(os.path.join(tmp, "a", "notes.txt"))

            index = cp.build_image_index(tmp, quiet=True)

            # Case-differing duplicates collide into one entry.
            self.assertIn("img_0001.jpg", index)
            self.assertEqual(len(index["img_0001.jpg"]), 2)
            self.assertEqual(index["img_0001.jpg"], sorted(index["img_0001.jpg"]))
            # Dot-directories and crop folders are pruned.
            self.assertNotIn("img_0002.jpg", index)
            self.assertNotIn("img_0003.jpg", index)
            # Non-image extensions are skipped.
            self.assertNotIn("notes.txt", index)


class TestResolveImagePaths(unittest.TestCase):
    """The auto resolver, its folder memo, and its abort contract."""

    def test_memo_applies_to_whole_folder(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            for i in (1, 2, 3):
                _touch(os.path.join(tmp, "new", "dive1", f"IMG_000{i}.JPG"))
            # Decoy in a same-named folder defeats the final-directory rule,
            # forcing a duplicate decision on the first camera only.
            _touch(os.path.join(tmp, "other", "dive1", "IMG_0001.JPG"))

            stored = {
                f"cam_000{i}": rf"D:\old\dive1\IMG_000{i}.JPG" for i in (1, 2, 3)
            }
            index = cp.build_image_index(tmp, quiet=True)

            calls = []

            def on_duplicate(cam_id, old_path, candidates):
                calls.append(cam_id)
                pick = os.path.join(tmp, "new", "dive1", cp.path_basename(old_path))
                return ("pick", pick)

            plan = cp.resolve_image_paths(
                stored, index, on_duplicate=on_duplicate, on_missing=None
            )

            # Only the first camera was ambiguous; the memo silenced the rest.
            self.assertEqual(calls, ["cam_0001"])
            self.assertEqual(len(plan.changes), 3)
            self.assertEqual(
                plan.dir_map, {"D:/old/dive1": os.path.join(tmp, "new", "dive1")}
            )
            self.assertEqual(plan.n_duplicate_prompts, 1)
            self.assertEqual(plan.collisions(), {})

    def test_already_valid_path_is_left_alone(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            good = _touch(os.path.join(tmp, "here", "IMG_0001.JPG"))
            _touch(os.path.join(tmp, "elsewhere", "IMG_0001.JPG"))
            index = cp.build_image_index(tmp, quiet=True)

            plan = cp.resolve_image_paths({"cam_0001": good}, index)

            self.assertEqual(plan.changes, {})
            self.assertEqual(plan.unchanged, {"cam_0001": good})

    def test_abort_raises_and_leaves_input_untouched(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "a", "IMG_0001.JPG"))
            _touch(os.path.join(tmp, "b", "IMG_0001.JPG"))
            index = cp.build_image_index(tmp, quiet=True)

            stored = {"cam_0001": r"D:\old\dive1\IMG_0001.JPG"}
            snapshot = dict(stored)

            with self.assertRaises(cp.PathAborted):
                cp.resolve_image_paths(
                    stored, index, on_duplicate=lambda *a: ("abort", None)
                )
            # The resolver must not mutate what it was handed.
            self.assertEqual(stored, snapshot)

    def test_missing_callback_required_when_non_interactive(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            index = cp.build_image_index(tmp, quiet=True)
            with self.assertRaises(RuntimeError):
                cp.resolve_image_paths({"cam_0001": "/gone/IMG_0001.JPG"}, index)


class TestPlanFindReplace(unittest.TestCase):
    """Literal find/replace planning and its existence gate."""

    def test_missing_is_reported_for_nonexistent_results(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            real = _touch(os.path.join(tmp, "new", "IMG_0001.JPG"))
            stored = {
                "cam_0001": os.path.join(tmp, "old", "IMG_0001.JPG"),
                "cam_0002": os.path.join(tmp, "old", "IMG_0002.JPG"),
            }

            plan = cp.plan_find_replace(stored, os.path.join(tmp, "old"),
                                        os.path.join(tmp, "new"))

            self.assertEqual(plan.changes["cam_0001"], real)
            self.assertIn("cam_0002", plan.missing)
            self.assertNotIn("cam_0001", plan.missing)

    def test_paths_without_find_are_unchanged(self) -> None:
        cp = _load_pathrepair()
        stored = {"cam_0001": "/other/IMG_0001.JPG"}
        plan = cp.plan_find_replace(stored, "/nope", "/yep")
        self.assertEqual(plan.changes, {})
        self.assertEqual(plan.unchanged, stored)

class TestCollisions(unittest.TestCase):
    """Two cameras must never be pointed at the same image file."""

    def test_collision_detected_when_two_cameras_resolve_to_one_file(self) -> None:
        cp = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            # One surviving copy, but two cameras that both want that basename.
            only_copy = _touch(os.path.join(tmp, "new", "IMG_0001.JPG"))
            stored = {
                "cam_0001": r"D:\old\dive1\IMG_0001.JPG",
                "cam_0002": r"D:\old\dive2\IMG_0001.JPG",
            }
            index = cp.build_image_index(tmp, quiet=True)

            plan = cp.resolve_image_paths(stored, index)

            self.assertEqual(plan.changes["cam_0001"], only_copy)
            self.assertEqual(plan.changes["cam_0002"], only_copy)
            self.assertEqual(
                plan.collisions(), {only_copy: ["cam_0001", "cam_0002"]}
            )


class TestSavePaths(unittest.TestCase):
    """Persisting image paths without disturbing poses."""

    def _write_meta(self, path: str) -> dict:
        meta = {
            "cameras": {
                "cam_a": {
                    "path": "/old/IMG_A.JPG",
                    "center": [1.0, 2.0, 3.0],
                    "transform": [[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 1, 0], [0, 0, 0, 1]],
                    "enabled": True,
                    "label": "IMG_A",
                },
                "cam_b": {
                    "path": "/old/IMG_B.JPG",
                    "center": None,
                    "transform": None,
                    "enabled": False,
                },
            }
        }
        with open(path, "w") as f:
            json.dump(meta, f)
        return meta

    def test_writes_path_and_preserves_other_keys(self) -> None:
        Cameras = _load_cameras_class()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "test.meta.json")
            before = self._write_meta(meta_path)

            cams = Cameras(cams_meta_filepath=meta_path, cams_xml_filepath=None)
            cams.data["cam_a"].set_filepath("/new/IMG_A.JPG")
            n = cams.save_paths(only_cam_ids={"cam_a"})

            self.assertEqual(n, 1)
            with open(meta_path) as f:
                out = json.load(f)
            ca = out["cameras"]["cam_a"]
            self.assertEqual(ca["path"], "/new/IMG_A.JPG")
            # Every other key survives untouched.
            for key in ("center", "transform", "enabled", "label"):
                self.assertEqual(ca[key], before["cameras"]["cam_a"][key])
            # The unselected camera is not rewritten.
            self.assertEqual(out["cameras"]["cam_b"]["path"], "/old/IMG_B.JPG")
            # The cached basename was refreshed alongside the path.
            self.assertEqual(cams.data["cam_a"].filename, "IMG_A.JPG")

    def test_writes_path_for_pose_less_camera(self) -> None:
        """A camera with a null pose still gets its path written.

        This is the case ``Cameras.save`` deliberately skips, and the reason
        ``save_paths`` exists as a separate method.
        """
        Cameras = _load_cameras_class()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "test.meta.json")
            self._write_meta(meta_path)

            cams = Cameras(cams_meta_filepath=meta_path, cams_xml_filepath=None)
            self.assertTrue(
                getattr(cams.data["cam_b"], "missing_pose_from_meta", False)
            )
            cams.data["cam_b"].set_filepath("/new/IMG_B.JPG")
            n = cams.save_paths()

            self.assertEqual(n, 1)
            with open(meta_path) as f:
                out = json.load(f)
            self.assertEqual(out["cameras"]["cam_b"]["path"], "/new/IMG_B.JPG")
            self.assertIsNone(out["cameras"]["cam_b"]["center"])

    def test_missing_image_paths_reports_absent_files(self) -> None:
        Cameras = _load_cameras_class()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "test.meta.json")
            self._write_meta(meta_path)
            real = _touch(os.path.join(tmp, "IMG_A.JPG"))

            cams = Cameras(cams_meta_filepath=meta_path, cams_xml_filepath=None)
            cams.data["cam_a"].set_filepath(real)

            missing = cams.missing_image_paths(use_orig=True)
            self.assertNotIn("cam_a", missing)
            self.assertIn("cam_b", missing)



class TestPatchYamlPaths(unittest.TestCase):
    """The surgical YAML writer must not disturb anything it is not asked to."""

    SAMPLE = (
        "# Project config for cur_sna\n"
        "path: /mnt/old/cur_sna\n"
        "id: cur_sna\n"
        "ply: /mnt/old/cur_sna/cur_sna.ply  # the decimated cloud\n"
        "my_custom_key: keep me\n"
        "color_correction:\n"
        "  matrix: [[1, 0], [0, 1]]\n"
        "  ply: not-a-top-level-key\n"
    )

    def test_preserves_comments_and_unknown_keys(self) -> None:
        pr = _load_pathrepair()
        out = pr.patch_yaml_paths(self.SAMPLE, {"ply": "cur_sna.ply"})

        self.assertIn("# Project config for cur_sna\n", out)
        self.assertIn("my_custom_key: keep me\n", out)
        self.assertIn("  matrix: [[1, 0], [0, 1]]\n", out)
        self.assertIn("ply: cur_sna.ply", out)
        self.assertNotIn("/mnt/old/cur_sna/cur_sna.ply", out)
        # Key order is unchanged.
        self.assertLess(out.index("path:"), out.index("id:"))
        self.assertLess(out.index("id:"), out.index("ply:"))

    def test_ignores_nested_keys(self) -> None:
        pr = _load_pathrepair()
        out = pr.patch_yaml_paths(self.SAMPLE, {"ply": "cur_sna.ply"})
        # The indented `ply:` under color_correction must survive untouched.
        self.assertIn("  ply: not-a-top-level-key\n", out)

    def test_preserves_inline_comment_on_a_patched_line(self) -> None:
        pr = _load_pathrepair()
        out = pr.patch_yaml_paths(self.SAMPLE, {"ply": "cur_sna.ply"})
        self.assertIn("ply: cur_sna.ply  # the decimated cloud", out)

    def test_hash_inside_a_quoted_value_is_not_a_comment(self) -> None:
        pr = _load_pathrepair()
        text = 'note: "has # inside"\n'
        out = pr.patch_yaml_paths(text, {"note": "x"})
        self.assertEqual(out, "note: x\n")

    def test_no_updates_returns_input_unchanged(self) -> None:
        pr = _load_pathrepair()
        self.assertEqual(pr.patch_yaml_paths(self.SAMPLE, {}), self.SAMPLE)

    def test_quotes_only_when_needed(self) -> None:
        pr = _load_pathrepair()
        self.assertEqual(pr.yaml_quote("plain.ply"), "plain.ply")
        self.assertEqual(pr.yaml_quote(" leading"), '" leading"')
        self.assertEqual(pr.yaml_quote("a: b"), '"a: b"')


class TestRelativise(unittest.TestCase):
    """Repaired paths inside the project folder become bare filenames."""

    def test_inside_project_becomes_bare_name(self) -> None:
        pr = _load_pathrepair()
        self.assertEqual(pr.relativise("/proj/a.ply", "/proj"), "a.ply")

    def test_outside_project_stays_absolute(self) -> None:
        pr = _load_pathrepair()
        self.assertEqual(
            pr.relativise("/elsewhere/a.ply", "/proj"), "/elsewhere/a.ply"
        )

    def test_nested_inside_project_stays_absolute(self) -> None:
        pr = _load_pathrepair()
        # Only files directly in the project dir round-trip through
        # __add_path_if_needed, so a nested one must keep its full path.
        self.assertEqual(
            pr.relativise("/proj/sub/a.ply", "/proj"), "/proj/sub/a.ply"
        )


class TestPlanYamlRepair(unittest.TestCase):
    """Phase A planning: stale path: key, conventions, and no initialize()."""

    def _project(self, tmp, files=("cur_sna.ply", "cur_sna.cams.xml",
                                   "cur_sna.meta.json")):
        proj = os.path.join(tmp, "cur_sna")
        os.makedirs(proj, exist_ok=True)
        for name in files:
            _touch(os.path.join(proj, name))
        return proj

    def test_fixes_stale_path_key_then_conventions(self) -> None:
        pr = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            yaml_path = os.path.join(proj, "cur_sna.yaml")
            config = {
                "path": "/mnt/gone/cur_sna",
                "id": "cur_sna",
                "ply": "/mnt/gone/cur_sna/cur_sna.ply",
                "cams_xml": "/mnt/gone/cur_sna/cur_sna.cams.xml",
                "cams_meta_json": "/mnt/gone/cur_sna/cur_sna.meta.json",
            }
            plan = pr.plan_yaml_repair(yaml_path, config)

            changes = plan.changes
            self.assertEqual(changes["path"], proj)
            # Everything else resolves by convention, as a bare filename.
            self.assertEqual(changes["ply"], "cur_sna.ply")
            self.assertEqual(changes["cams_xml"], "cur_sna.cams.xml")
            self.assertEqual(changes["cams_meta_json"], "cur_sna.meta.json")
            self.assertEqual(plan.unresolved, [])
            # The resolved map gives phase B somewhere to read the cameras from.
            self.assertEqual(
                plan.resolved["cams_meta_json"],
                os.path.join(proj, "cur_sna.meta.json"),
            )

    def test_already_valid_paths_are_left_alone(self) -> None:
        pr = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            yaml_path = os.path.join(proj, "cur_sna.yaml")
            config = {"path": proj, "id": "cur_sna", "ply": "cur_sna.ply"}
            plan = pr.plan_yaml_repair(yaml_path, config)
            self.assertEqual(plan.changes, {})

    def test_legacy_pcd_key_is_repaired_in_place(self) -> None:
        pr = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            yaml_path = os.path.join(proj, "cur_sna.yaml")
            config = {
                "path": "/mnt/gone/cur_sna",
                "id": "cur_sna",
                "pcd": "/mnt/gone/cur_sna/cur_sna.ply",
            }
            plan = pr.plan_yaml_repair(yaml_path, config)
            # Repaired under its own key; never silently migrated to `ply`.
            self.assertEqual(plan.changes["pcd"], "cur_sna.ply")
            self.assertNotIn("ply", plan.changes)

    def test_plan_is_computed_without_loading_the_ply(self) -> None:
        """A dead PLY path must still produce a plan.

        Guards the constraint that phase A never constructs a
        ProjectInitializer, whose initialize() would try to open the file.
        """
        pr = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp, files=("cur_sna.meta.json",))
            yaml_path = os.path.join(proj, "cur_sna.yaml")
            config = {
                "path": "/mnt/gone/cur_sna",
                "id": "cur_sna",
                "ply": "/mnt/gone/cur_sna/nowhere.ply",
                "cams_meta_json": "/mnt/gone/cur_sna/cur_sna.meta.json",
            }
            plan = pr.plan_yaml_repair(yaml_path, config)

            self.assertEqual(plan.changes["cams_meta_json"], "cur_sna.meta.json")
            # The PLY cannot be found; it is reported, not fatal.
            unresolved = {e.key for e in plan.unresolved}
            self.assertIn("ply", unresolved)

    def test_missing_directory_key_is_reported(self) -> None:
        pr = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            yaml_path = os.path.join(proj, "cur_sna.yaml")
            config = {
                "path": proj,
                "id": "cur_sna",
                "photos_path": "/mnt/gone/cur_sna/cur_sna.photos",
            }
            plan = pr.plan_yaml_repair(yaml_path, config)
            self.assertIn("photos_path", {e.key for e in plan.unresolved})

    def test_directory_key_resolves_by_convention(self) -> None:
        pr = _load_pathrepair()
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            os.makedirs(os.path.join(proj, "cur_sna.photos"))
            yaml_path = os.path.join(proj, "cur_sna.yaml")
            config = {
                "path": proj,
                "id": "cur_sna",
                "photos_path": "/mnt/gone/cur_sna/cur_sna.photos",
            }
            plan = pr.plan_yaml_repair(yaml_path, config)
            self.assertEqual(plan.changes["photos_path"], "cur_sna.photos")


if __name__ == "__main__":
    unittest.main()
