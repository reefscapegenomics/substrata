"""Substrata package for point cloud processing and analysis.

This package provides tools for working with point clouds, including
loading, processing, and analyzing 3D point cloud data.
"""

import warnings

# Suppress noisy tqdm warning emitted by tqdm.autonotebook when ipywidgets is missing.
# This often surfaces via third-party packages (e.g., tqdm_joblib) on import.
try:
    from tqdm import TqdmWarning  # type: ignore
except Exception:  # pragma: no cover - robust fallback if tqdm is unavailable
    TqdmWarning = Warning  # type: ignore[assignment]

# Under Sphinx autodoc_mock_imports, the import above succeeds but yields a
# mock object (not a class), so guard before using it as a warning category.
if not (isinstance(TqdmWarning, type) and issubclass(TqdmWarning, Warning)):
    TqdmWarning = Warning  # type: ignore[assignment]

warnings.filterwarnings(
    "ignore",
    message=".*IProgress not found.*",
    category=TqdmWarning,
)

import importlib as _importlib
from typing import List

from .logging import logger

# The modules whose public names make up the package namespace, as `from .<module> import *` each, in
# this order (a later one wins). They are imported on the first use of a package attribute (PEP 562),
# not on `import substrata`: `import substrata.stratified` (e.g. substrata-explore's encoder processes)
# then loads just that module, not open3d, matplotlib, pandas, scipy and cv2 (seconds on a cold start).
_STAR_MODULES = (
    "annotations", "sidecars", "color_calibration", "pathrepair", "cameras", "pointclouds",
    "stratified", "initializer", "measurements", "visualizations", "geom", "settings",
    "classification", "ortho", "animations", "segmentation",
)
_state = {"loaded": False, "loading": False}


def _load_all():
    """Import the modules of _STAR_MODULES and bind their public names here, as the star imports did."""
    if _state["loaded"] or _state["loading"]:
        return
    _state["loading"] = True
    try:
        g = globals()
        for m in _STAR_MODULES:
            mod = _importlib.import_module(f".{m}", __name__)
            names = getattr(mod, "__all__", None)
            if names is None:
                names = [n for n in vars(mod) if not n.startswith("_")]
            for n in names:
                g[n] = getattr(mod, n)
        g["__all__"] = ["logger"] + [n for n in list(g) if not n.startswith("_")]
        _state["loaded"] = True
    finally:
        _state["loading"] = False


def __getattr__(name):
    # while loading, a missing name is a submodule not imported yet (`from . import x`): Python then
    # imports it itself
    if (name.startswith("__") and name != "__all__") or _state["loading"] or _state["loaded"]:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_all()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    _load_all()
    return sorted(globals())
