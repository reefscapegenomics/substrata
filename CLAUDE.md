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

Docs: Sphinx + myst-nb in `docs/` (`pip install -r docs/requirements.txt`, then `make -C docs html`). Tutorial notebooks in `docs/notebooks/` are rendered into the docs from their saved outputs (`nb_execution_mode = "off"`) — they double as integration examples. Keep their outputs small: no interactive Plotly figures, and downscale very large embedded images. `sandbox/` holds local-only scratch notebooks and is gitignored (not tracked).

## Code style

- PEP8, Google-style docstrings
- Type hints in function signatures; use `from __future__ import annotations` to avoid circular imports
- Max line length 88 (enforced by flake8 in setup.cfg)

## Architecture

**`ProjectInitializer` (initializer.py) is the central factory.** A "project" is a directory named like `cur_sna_20m_20200303` (site_location_depth_date). The initializer either reads a `<dirname>.yaml` project file or falls back to filename conventions within the directory (`<id>.ply`, `<id>.cams.xml`, `<id>_markers.csv`, etc.; annotation sets live in `<id>.annos/` with their `<csv>.yaml` sidecars, `<csv stem>.cropped/` crops and the default `classes.csv`, and there is no default set beyond the YAML `annotations:` key; switch sets with `use_annotations_file`), then lazily spawns the main domain objects: `PointCloud` (pointclouds.py), `Cameras` (cameras.py), and `Annotations`/`Scalebars` (annotations.py). It also carries project-level state: `world_transform` (4×4), `scale_factor`, `depth_offset`, color correction, and optional links to other plots (`parent_id`/`parent_transform`: placement in a larger plot, own frame kept; `baseline_id`/`baseline_transform`: a later timepoint whose `world_transform` is a cached `baseline.world_transform @ baseline_transform`). Both the CLI and the notebooks start from this class, so changes to project-file keys or naming conventions ripple everywhere. The YAML schema is documented in the `init_with_yaml` docstring; the legacy `pcd:` key is still accepted as an alias for `ply:`.

**Module layout:**
- `pointclouds.py` — `PointCloud` wrapper around open3d, plus streaming PLY tooling (`decimate_ply_file`, `repair_ply_for_open3d`, `stream_extract_neighborhoods_ply`) that parses PLY binaries directly so huge files never need to fit in memory; memory-safe vertex caps are computed from available RAM
- `cameras.py` — `Cameras`/`Camera` (Metashape `.cams.xml` poses + sensors), image matching, video frame extraction, camera/depth-log time-sync; `assign_camera_types` (camera types by sensor resolution + focal length, sets `cam.camera_type`)
- `annotations.py` — `Annotations` (CSV-backed point labels), `Scalebars`. View choice: `get_image_matches(occluder=)` tests the most relevant matches on the stratified subsample (nothing dropped, obstructed last; `pcd=` keeps the legacy reprojection test, still used by `images`, benthic fraction and videos); `Annotation.select_views` picks the crops (main camera type first, a few of each other type) and the training view (most relevant unobstructed view of the main type, None = needs a look); `Annotations.main_camera_type`; `set_training_views` writes training views into a CSV (empty `cam_filepath` only; `training_views` collects them), `set_included_in_training` the include flag; `Annotation.stale_crops` finds crops whose pixel no longer matches the point's projection (`save_crops(prune=True)` removes them). Review crops are named `<photo stem>_<x>_<y>.jpg` (`cameras.view_crop_name`, `find_view_crop` also finds older names). Used by `substrata crops` and substrata-explore's grid
- `stratified.py` — stratified PLY (progressive / regional reads) and `Occluder`, the line-of-sight occlusion test on its subsample (shared with substrata-explore; `settings.OCCLUSION_BLOCKED_MIN`)
- `classification.py` — `substrata train` pipeline (label tree/map, collation, crops; rows with `include_cam_in_training` = False are left out, `included_in_training`; CoralNet exports via `convert_coralnet_csv`, also `substrata import-coralnet`); `classifier_fingerprint` (file stem + content hash) is stored with predictions as `pred_model` (CSV and crop EXIF) to tell stale predictions after retraining and the fastai crop classifier; `get_image_classifier` also loads point classifiers, and every `classify_*` function accepts either kind
- `pointfeatures.py` — point-feature classifier (default `train --arch dinov2`; `dinov3` short-hands in `settings.POINT_ARCH_ALIASES`, DINOv3 loaded from its repo's backbone module with locally downloaded gated weights): frozen DINOv2/DINOv3 features read at the annotated pixel + linear probe (`PointFeatureClassifier`, `.pt` file); per-point resampling to a physical resolution (ground resolution = camera depth × world scale / fx), feature caches, per-camera calibration (`substrata calibrate`), and dense per-photo probability maps (overlapping tiles blended per feature cell; `DenseMap`) used by `segment` (multi-view fusion in `segmentation._classify_dense_views`) and `substrata masks`
- `curation.py` — training-data curation (`substrata curate`): append-only decision store `curation.jsonl` keyed by photo file name + rounded pixel (`CurationStore`; applied by `train` in `classification.count_labels` / `collate_training_annotations`), per-point review scores (`score_training_points`: cross-validated label probability, neighbourhood ring, kNN label consistency, novelty; `curate/points.csv` + tiles) and pools of unlabelled points mined from photos or given as a points CSV / CoralNet export (`mine_pool`, `read_pool_points`: clustered, with their own camera-domain statistics used by `train`; without a ground resolution each photo is scaled so the window keeps its share of the photo width, `pointfeatures.photo_scale` / `width_scale`, as `plan_training_scales` scales pool rows; given labels are kept as `given_label`, used only once accepted). Decisions can carry the reviewer's `label_conf` (0-1; `--min_conf` applies to it). `train` records the annotation files it read in `training_sources.json`; `SourceLabels` edits a point's label in them (image-only points only, found by curation key, backup per file per session) for the review grid's "also edit the source annotations". The review grid is substrata-explore's `curate` mode (`substrata curate --gui`)
- `geom.py` — `Transform` (4×4) and `Vector` primitives used across all modules
- `measurements.py` — terrain/ecology metrics (TPI/TRI, rugosity, fractal dimension, surface area, gap fraction, depth regression)
- `visualizations.py` — open3d/matplotlib rendering, composite-view and QC PDF reports
- `settings.py` — module of constants (default radii/thresholds, RGL scalebar target definitions, column orders); tune defaults here rather than hardcoding
- `cli.py` — argparse subcommands (`decimate`, `head`, `scalebars`, `views`, `orient`, `colors`, `camsync`, `transform`, …); most default to auto-detecting the project from the CWD via `ProjectInitializer`. CLI usage is documented in `docs/command_line_usage.md` — update it when adding/changing subcommands

**Import style:** `__init__.py` star-imports every module, so all public names share one flat `substrata.*` namespace — avoid name collisions across modules. Use the shared `logger` from `substrata.logging`.
