# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`substrata` is a Python package for 3-D geometry and point-cloud handling, built for coral-reef photogrammetry workflows (point clouds, camera poses, annotations, terrain measurements). Source lives in `src/substrata/`, with an installed CLI entry point `substrata` (see `src/substrata/cli.py:main`).

## Environment and commands

Heavy dependencies (open3d, opencv, pytorch) come from conda, not pyproject.toml (which only declares numpy):

```bash
conda env create -f environment.yml
conda activate substrata
python -m pip install -e .
```

Tests use stdlib `unittest` (no pytest dependency):

```bash
python -m unittest discover -s tests -v          # all tests
python -m unittest tests.test_color_calibration  # single module
python -m unittest tests.test_cameras_save.TestCamerasSave.test_name  # single test
```

Note: test files deliberately load modules directly from `src/substrata/` via `importlib` stubs instead of importing the `substrata` package, so they run without open3d/opencv installed. Follow that pattern when adding tests for modules with heavy imports.

Lint: `flake8` configured in `setup.cfg` (max-line-length 88, Google docstring convention).

Docs: Sphinx + myst-nb in `docs/` (`pip install -r docs/requirements.txt`, then `make -C docs html`). Tutorial notebooks in `docs/notebooks/` are executed/rendered into the docs — they double as integration examples. `sandbox/` holds scratch notebooks and is not part of the package or docs.

## Code style (from AGENTS.md)

- PEP8, Google-style docstrings
- Type hints in function signatures; use `from __future__ import annotations` to avoid circular imports
- AGENTS.md says max line length 100, but flake8 in setup.cfg enforces 88 — stay within 88 to pass lint

## Architecture

**`ProjectInitializer` (initializer.py) is the central factory.** A "project" is a directory named like `cur_sna_20m_20200303` (site_location_depth_date). The initializer either reads a `<dirname>.yaml` project file or falls back to filename conventions within the directory (`<id>.ply`, `<id>.cams.xml`, `<id>_markers.csv`, `<id>_ann.csv`, etc.), then lazily spawns the main domain objects: `PointCloud` (pointclouds.py), `Cameras` (cameras.py), and `Annotations`/`Scalebars` (annotations.py). It also carries project-level state: `world_transform` (4×4), `scale_factor`, `depth_offset`, color correction. Both the CLI and the notebooks start from this class, so changes to project-file keys or naming conventions ripple everywhere. The YAML schema is documented in the `init_with_yaml` docstring; the legacy `pcd:` key is still accepted as an alias for `ply:`.

**Module layout:**
- `pointclouds.py` — `PointCloud` wrapper around open3d, plus streaming PLY tooling (`decimate_ply_file`, `repair_ply_for_open3d`, `stream_extract_neighborhoods_ply`) that parses PLY binaries directly so huge files never need to fit in memory; memory-safe vertex caps are computed from available RAM
- `cameras.py` — `Cameras`/`Camera` (Metashape `.cams.xml` poses + sensors), image matching, video frame extraction, camera/depth-log time-sync
- `annotations.py` — `Annotations` (CSV-backed point labels), `Scalebars`
- `geom.py` — `Transform` (4×4) and `Vector` primitives used across all modules
- `measurements.py` — terrain/ecology metrics (TPI/TRI, rugosity, fractal dimension, surface area, gap fraction, depth regression)
- `visualizations.py` — open3d/matplotlib rendering, composite-view and QC PDF reports
- `settings.py` — module of constants (default radii/thresholds, RGL scalebar target definitions, column orders); tune defaults here rather than hardcoding
- `cli.py` — argparse subcommands (`decimate`, `head`, `scalebars`, `views`, `orient`, `colors`, `camsync`, `transform`, …); most default to auto-detecting the project from the CWD via `ProjectInitializer`. CLI usage is documented in `docs/command_line_usage.md` — update it when adding/changing subcommands

**Import style:** `__init__.py` star-imports every module, so all public names share one flat `substrata.*` namespace — avoid name collisions across modules. Use the shared `logger` from `substrata.logging`.
