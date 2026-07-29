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

from typing import List

from .logging import logger

from .annotations import *
from .color_calibration import *
from .cameras import *
from .pointclouds import *
from .initializer import *

from .measurements import *

# from .examples import *
from .visualizations import *
from .geom import *

from .settings import *

from .classification import *

from .ortho import *

from .animations import *

from .segmentation import *

# from .firefish import *
# from .utils import *
# from .initializer import *


__all__: List[str] = ["logger"] + [name for name in dir() if not name.startswith("_")]
