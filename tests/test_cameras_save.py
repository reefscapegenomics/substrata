# Standard Library
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

# Third-Party Libraries
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SUB = _SRC / "substrata"


def _load_cameras_class():
    """Load :class:`Cameras` without importing ``substrata`` package ``__init__``."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    def _load(mod_name: str, rel: str) -> types.ModuleType:
        path = _SUB / rel
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {mod_name} from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    if "substrata.settings" not in sys.modules:
        _load("substrata.settings", "settings.py")
    if "substrata.geom" not in sys.modules:
        _load("substrata.geom", "geom.py")

    if "substrata.visualizations" not in sys.modules:
        sys.modules["substrata.visualizations"] = types.ModuleType(
            "substrata.visualizations"
        )
    if "substrata.measurements" not in sys.modules:
        sys.modules["substrata.measurements"] = types.ModuleType(
            "substrata.measurements"
        )

    if "substrata.logging" not in sys.modules:
        log_mod = types.ModuleType("substrata.logging")

        @contextmanager
        def tqdm_joblib(_tqdm_object):
            yield

        log_mod.tqdm_joblib = tqdm_joblib
        sys.modules["substrata.logging"] = log_mod

    if "substrata.cameras" not in sys.modules:
        _load("substrata.cameras", "cameras.py")

    return sys.modules["substrata.cameras"].Cameras


def _load_parse_xyz_from_cli() -> object:
    """Load :func:`substrata.cli._parse_xyz_csv` without importing package ``__init__``."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    pkg = sys.modules.get("substrata")
    if pkg is None:
        pkg = types.ModuleType("substrata")
        pkg.__path__ = [str(_SUB)]
        sys.modules["substrata"] = pkg

    pc = types.ModuleType("substrata.pointclouds")
    pc.PointCloud = object
    pc.decimate_ply_file = lambda *a, **k: None
    pc.ply_head = lambda *a, **k: None
    sys.modules["substrata.pointclouds"] = pc

    ini = types.ModuleType("substrata.initializer")
    ini.ProjectInitializer = object
    sys.modules["substrata.initializer"] = ini

    ann = types.ModuleType("substrata.annotations")
    ann.Annotations = object
    ann.Scalebars = object
    sys.modules["substrata.annotations"] = ann

    if "substrata.settings" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "substrata.settings", _SUB / "settings.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("settings")
        sm = importlib.util.module_from_spec(spec)
        sys.modules["substrata.settings"] = sm
        spec.loader.exec_module(sm)
    sys.modules["substrata"].settings = sys.modules["substrata.settings"]

    spec = importlib.util.spec_from_file_location("substrata.cli", _SUB / "cli.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cli")
    cli_mod = importlib.util.module_from_spec(spec)
    sys.modules["substrata.cli"] = cli_mod
    spec.loader.exec_module(cli_mod)
    return cli_mod._parse_xyz_csv


class TestCamerasSave(unittest.TestCase):
    """Tests for :meth:`substrata.cameras.Cameras.save`."""

    def test_save_merges_center_transform(self) -> None:
        Cameras = _load_cameras_class()

        meta = {
            "cameras": {
                "cam_a": {
                    "center": [1.0, 2.0, 3.0],
                    "transform": np.eye(4).tolist(),
                    "path": "/tmp/placeholder_a.jpg",
                },
                "cam_b": {
                    "center": [0.0, 0.0, 0.0],
                    "transform": np.eye(4).tolist(),
                    "path": "/tmp/placeholder_b.jpg",
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.meta.json")
            with open(path, "w") as f:
                json.dump(meta, f)

            cams = Cameras(cams_meta_filepath=path, cams_xml_filepath=None)
            cams.data["cam_a"].orig_coords = np.array([10.0, 20.0, 30.0])
            cams.data["cam_a"].orig_camera_transform = np.diag([2.0, 2.0, 2.0, 1.0])
            cams.save()

            with open(path) as f:
                out = json.load(f)
            self.assertIn("cameras", out)
            ca = out["cameras"]["cam_a"]
            self.assertEqual(ca["center"], [10.0, 20.0, 30.0])
            self.assertEqual(len(ca["transform"]), 4)
            self.assertEqual(ca["transform"][0][0], 2.0)
            cb = out["cameras"]["cam_b"]
            self.assertEqual(cb["center"], [0.0, 0.0, 0.0])


class TestParseXyzCsv(unittest.TestCase):
    def test_parse(self) -> None:
        parse = _load_parse_xyz_from_cli()
        self.assertEqual(parse("0, 0.12, 0"), [0.0, 0.12, 0.0])
