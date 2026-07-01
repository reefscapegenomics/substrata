# Standard Library
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_module(mod_name: str, rel_path: Path) -> types.ModuleType:
    """Load a module from a file path without importing its parent package."""
    spec = importlib.util.spec_from_file_location(mod_name, rel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {mod_name} from {rel_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_export_project() -> types.ModuleType:
    """Load the Metashape exporter script (its ``import Metashape`` is deferred)."""
    return _load_module(
        "substrata_export_project",
        _SUB / "metashape_scripts" / "export_project.py",
    )


def _load_cli() -> types.ModuleType:
    """Load ``substrata.cli`` with heavy submodules stubbed out."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    # Real settings (pure constants); stub the open3d/torch-heavy modules.
    if "substrata.settings" not in sys.modules:
        _load_module("substrata.settings", _SUB / "settings.py")

    def _stub(name: str, attrs: dict) -> None:
        if name in sys.modules:
            return
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    _stub(
        "substrata.pointclouds",
        {
            "PointCloud": object,
            "decimate_ply_file": lambda *a, **k: None,
            "ply_head": lambda *a, **k: None,
            "repair_ply_for_open3d": lambda *a, **k: None,
        },
    )
    _stub("substrata.initializer", {"ProjectInitializer": object})
    _stub("substrata.annotations", {"Annotations": object, "Scalebars": object})

    return _load_module("substrata.cli", _SUB / "cli.py")


class TestExportProjectHelpers(unittest.TestCase):
    """Pure (Metashape-free) helpers in the bundled exporter script."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_export_project()

    def test_default_project_id_strips_psx(self):
        self.assertEqual(
            self.mod.default_project_id("/data/cur_sna_20m_20200303.psx"),
            "cur_sna_20m_20200303",
        )
        # Case-insensitive extension; trailing separators tolerated.
        self.assertEqual(self.mod.default_project_id("foo.PSX"), "foo")
        self.assertEqual(self.mod.default_project_id("/a/b/foo/"), "foo")

    def test_project_layout_paths(self):
        layout = self.mod.project_layout("/out", "proj")
        self.assertEqual(layout["folder"], os.path.join("/out", "proj"))
        self.assertEqual(
            layout["cams_xml"], os.path.join("/out", "proj", "proj.cams.xml")
        )
        self.assertEqual(
            layout["meta_json"], os.path.join("/out", "proj", "proj.meta.json")
        )
        self.assertEqual(
            layout["markers"], os.path.join("/out", "proj", "proj_markers.csv")
        )
        self.assertEqual(
            layout["ply"], os.path.join("/out", "proj", "proj.ply")
        )

    def test_build_meta_dict_structure(self):
        cams = {
            "7": {
                "path": "/img/0007.jpg",
                "center": [1.0, 2.0, 3.0],
                "transform": [float(i) for i in range(16)],
                "enabled": True,
            }
        }
        meta = self.mod.build_meta_dict(cams, crs_authority=None, chunk_transform=None)
        self.assertEqual(set(meta.keys()), {"crs", "chunk_transform", "cameras"})
        self.assertIn("7", meta["cameras"])
        self.assertEqual(len(meta["cameras"]["7"]["transform"]), 16)
        # Keys must be strings so they match the .cams.xml <camera id> join.
        self.assertTrue(all(isinstance(k, str) for k in meta["cameras"]))
        # Round-trips through JSON.
        self.assertEqual(json.loads(json.dumps(meta)), meta)

    def test_format_marker_row(self):
        self.assertEqual(
            self.mod.format_marker_row(3, 1.5, -2.0, 0.25, "target_1"),
            "3,1.5,-2.0,0.25,target_1",
        )
        self.assertEqual(self.mod.MARKERS_HEADER, "id,x,y,z,label")

    def test_export_markers_to_csv(self):
        def _marker(key, x, y, z, label):
            return types.SimpleNamespace(
                key=key,
                label=label,
                position=types.SimpleNamespace(x=x, y=y, z=z),
            )

        markers = [
            _marker(1, 0.1, 0.2, 0.3, "a"),
            _marker(2, 1.0, 2.0, 3.0, "b"),
            types.SimpleNamespace(key=9, label="skip", position=None),  # skipped
        ]
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "m.csv")
            self.mod.export_markers_to_csv(markers, out)
            with open(out) as f:
                lines = [ln.rstrip("\n") for ln in f]
        self.assertEqual(lines[0], "id,x,y,z,label")
        self.assertEqual(lines[1], "1,0.1,0.2,0.3,a")
        self.assertEqual(lines[2], "2,1.0,2.0,3.0,b")
        self.assertEqual(len(lines), 3)  # marker with no position omitted


class TestCliMetashapeHelpers(unittest.TestCase):
    """Executable resolution + id defaulting in the CLI wrapper."""

    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli()

    def _make_exe(self, d, name):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(path, 0o755)
        return path

    def test_default_metashape_id(self):
        self.assertEqual(
            self.cli._default_metashape_id("/x/cur_sna_20m.psx"), "cur_sna_20m"
        )

    def test_explicit_flag_wins(self):
        with tempfile.TemporaryDirectory() as d:
            exe = self._make_exe(d, "metashape.sh")
            self.assertEqual(self.cli._find_metashape_executable(exe), exe)

    def test_explicit_flag_not_executable_raises(self):
        with tempfile.TemporaryDirectory() as d:
            bogus = os.path.join(d, "nope.sh")
            with open(bogus, "w") as f:
                f.write("x")  # exists but not chmod +x
            with self.assertRaises(SystemExit):
                self.cli._find_metashape_executable(bogus)

    def test_env_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            exe = self._make_exe(d, "metashape.sh")
            old = os.environ.get("METASHAPE_EXE")
            os.environ["METASHAPE_EXE"] = exe
            try:
                self.assertEqual(self.cli._find_metashape_executable(None), exe)
            finally:
                if old is None:
                    os.environ.pop("METASHAPE_EXE", None)
                else:
                    os.environ["METASHAPE_EXE"] = old

    def test_not_found_raises(self):
        old = os.environ.get("METASHAPE_EXE")
        os.environ.pop("METASHAPE_EXE", None)
        # Patch isfile->False so a real local Metashape install can't be found.
        try:
            with mock.patch.object(self.cli.os.path, "isfile", return_value=False):
                with self.assertRaises(SystemExit):
                    self.cli._find_metashape_executable(None)
        finally:
            if old is not None:
                os.environ["METASHAPE_EXE"] = old


if __name__ == "__main__":
    unittest.main()
