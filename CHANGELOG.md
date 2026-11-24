# Changelog

The version is `substrata.__version__` (`src/substrata/__init__.py`; pyproject.toml reads it). substrata-explore
checks it at start-up.

## 0.2.0.dev2 (unreleased)

### 24–25 November 2026: package review

A review of the whole package (calculations checked on synthetic data with known answers; every fix below has a test
that failed before it, mostly in `tests/test_review_*.py`).

**Calculations fixed**
- **Gap fraction** (`calc_gap_fraction`): the "fill" gap fraction flooded the canopy as sky when a seed fell on a covered
  pixel (an opaque 0–45° canopy gave 0.25 instead of 0.75), and with `color_output=True` (the default) benthic points of
  a colour close to the grey sky counted as sky. Both fractions now come from a binary raster (`hemispherical_cover`,
  `gap_fractions`), independent of the display options. The docstring now says what is measured (sky / disk area in an
  equidistant projection; it depends on point density).
- **Neighbourhood of an annotation** (`Annotation.reduce_point_cloud_to_radius`): filtered the world-frame cloud around
  the original-frame coordinates, so with a world transform it kept no points.
- **TPI / TRI**: return NaN (with a warning) when less than 75 % of the annulus holds points
  (`annulus_coverage`, `settings.TPI_MIN_ANNULUS_COVERAGE`; `min_annulus_coverage=None` turns it off). A one-sided
  annulus near the edge of the cloud gave a slope-biased TPI (0.084 m on a smooth plane). `OrthoGrid` per-cell TPI: the
  annulus now runs from half the cell side to half the window side (it was empty); it needs `neighborhood_scale >= 2`.
  On real colonies (10 close-up targets of sa3_40m, cylinder neighbourhoods as documented) TPI/TRI, Ra/Rq, plane
  angles and dispersion are unchanged by the review (≤ 1e-4); all 166 close-up targets of sa3_40m and sa4_20m have
  ≥ 83 % annulus coverage. A spherical neighbourhood on a steep slope now gives NaN (the sphere clips the annulus).
- **Plane fits without normals** (roughness, plane angles, RANSAC): crashed, or gave an arbitrary sign (153° instead of
  27°); the normal now faces +Z. `conduct_PCA` uses `eigh` (complex results on near-degenerate clouds).
- **Depth regression**: positive depths gave an up vector pointing down (the flip meant to catch this never ran); they are
  now negated with a warning (`normalize_depth_sign`).
- **Vector dispersion** accepts open3d clouds; its figure shows the directional deviation (0–180°) like the metric.
- **`geom`**: `Transform @ Transform` composed in the reverse order; `Vector.dot` included w; two constructors changed
  their inputs.
- **Point clouds**: colours of 16-bit or float PLYs were wrong in capped/streamed loads and neighbourhood extraction;
  `subsample_pointcloud_by_radius` dropped the true nearest neighbour; normals were scaled by scaled transforms
  (now inverse-transpose, renormalised); `stream_extract_neighborhoods_ply` used the wrong radius in cylinder mode;
  `get_intercept` could miss points and return a hit up to one step behind the surface (now an exact capsule search);
  `get_z_intercept` hung on an empty cloud.
- **Reproducibility**: `decimate` (and `metashape-export`'s decimation) is seeded (`--seed`, default 0, recorded in
  the header) and keeps exactly the requested number of points; the figures' decimation is seeded too. New `dec*` files
  differ from existing ones, so results computed on them (orientation, intercepts) can shift slightly.
- **Orthos**: a narrow, tall plot was cut off (pixel size from x only); markers were half a pixel off.
- **Training**: the train / validation / test split was drawn per point, so points of one photo fell in several splits
  and the reported metrics were optimistic. It is now per photo (`train --split-by photo|plot|point`; `point` = the old
  behaviour), recorded in the summary and the model. The probe's C is cross-validated grouped by photo.
- **Resampling scales**: `calibrate`, `train --validate/--test`, `curate` scores and `segment`'s dense views now resample
  photos of other cameras exactly as `predict` does (they used another scale). Recalibrate calibrated models used on
  wider cameras. Ground resolution falls back to `scale_factor` when there is no world transform.
- **`colors`**: the cloud and markers are in the same (world) frame with metre radii (`-n` and `--markers` mixed
  frames); outlier cards are dropped from the per-patch medians (the old rule almost never fired); the fit RMSE is
  stored.
- `curate` reviews the train split only (`--all-splits`), so the test set is not edited using the model's errors.
- `firefish --target-depth` accepts 20 or -20 (the documented example crashed); `masks` reuses masks only from the same
  classifier file; same-named photos no longer share feature caches or crops (`stale_crops` deleted valid crops of
  same-named photos in other folders).

**Data safety**
- **Annotation CSVs** keep every column, row, quoted cell and BOM through load and save: a comma in a label or path broke
  reloading, a BOM header did not load, and `save()` dropped columns it did not know (e.g. `include_cam_in_training`).
  Duplicate ids warn and are kept. Concurrent edits are locked (`fileio.path_lock`).
- One atomic writer (`substrata.fileio`) that keeps the file's permissions (the old ones left 0600 files), now used for
  annotation CSVs, meta.json / cams.xml, crops, camsync / curate / train / masks outputs, the stratified index and PLY
  outputs; `save_config_to_yaml` patches the project YAML in place (comments, unknown keys, relative paths and the
  colour fit's quality keys kept).
- `segment`'s recoloured PLY keeps 16-bit colours.

**Changed**
- Pixels: a point is in view for 0 ≤ u < width; photos are read in their stored grid (EXIF orientation ignored, as the
  projection does); matched keypoints are mapped to full-size pixels with the half-pixel offset (camsync / video poses
  shift by about half a reduced pixel). EXIF `DateTimeOriginal` is read from the Exif IFD, with sub-seconds.
- CLI options are spelled with hyphens (`--output-pdf`, `--min-conf`, `--use-chunk-transform`, …); the underscore
  spellings still work.
- `Camera.get_image_matches` returns `[]` instead of None; camera subsets no longer re-parent the original's cameras;
  nested loops over `Cameras` / `Annotations` work.
- One definition of the world transform's scale (`frames.uniform_scale`, `frames.metres_per_unit`), set on the
  project's cameras, markers and annotations (`ProjectInitializer.propagate_metres_per_unit`); the occlusion test's
  minimum margin is in metres.
- New helpers: `StratifiedPly.prefix_for_budget`, `OrthoMap(display_rotation=, units=)`.
- FireFish's offset search starts from the earliest camera and record times (it used the first in file order).
- View ranking (`cameras._relevance`): the squared distance to the camera over `fx * fy` (the area a pixel covers
  there) times the centre penalty. It added the depth (a length) to the squared distance, so the ranking depended on
  the chunk's units. On sa3_40m, 97 % of 400 points keep their best view (99.5 % within their camera type). Training
  views already stored in annotation CSVs are kept; views chosen anew (crops, predict, substrata-explore) can differ.
- **TPI / TRI are weighted by area** (`calc_tpi_and_tri(raster_cell_size=0.01)`, `settings.TPI_RASTER_CELL_SIZE`): the
  annulus points of each 1 cm XY cell are replaced by their mean, so densely photographed parts no longer weigh more.
  On 83 close-up targets of sa3_40m the points per cell varied 5x within an annulus; TPI_abs changed by 0.02 m
  (median, on values of about 0.19 m; up to 0.32 m), and 5 mm, 1 cm and 2 cm cells agreed within 2 mm.
  `raster_cell_size=None` gives the earlier per-point values. A warning is logged when fewer than half of the cells
  hold points (cells finer than the cloud).

**Measurements separate from figures**
- Every QC figure shows exactly the reported value: figures take the measurement's result (`elev_deg=`, `plane=`,
  `errors=`, `residuals=`, `result=`) and otherwise call the measurement function, never a copy of it. Roughness and
  dispersion figures no longer refit on a decimated cloud; depth-regression, camera-residual, benthic-fraction and
  colour-summary numbers come from the measurement.
- Metric functions only compute; their display images are built by figure functions in `visualizations`
  (`tpi_figure`, `benthic_fraction_figure`, `render_gap_fraction`). Signatures and return values unchanged.
- New pure helpers: `distances_to_plane`, `plane_orientation`, `annulus_plane`, `tpi_point_values`,
  `vector_dispersion_from_normals`, `normal_deviation_deg`, `regression_errors`, `pcd_correspondence` (the align QC
  statistics, previously computed only inside a plotting function), `camera_residuals`, `benthic_fraction_from_results`,
  `color_calibration.correction_residuals`, `OrthoGrid.label_counts`.
- Figure and report code that lived in compute modules (depth residuals, scalebars, colour cards, camsync and
  training reports, align QC, curation sheets) is in `visualizations`; the old methods delegate to it.
- `substrata.imagecrops`: the crop arithmetic used by training and prediction (it lived in `visualizations`).
- `substrata.visualizations` is a package of themed submodules (`clouds`, `colorcards`, `photos`, `depth`,
  `terrain`, `gapfraction`, `benthic`, `sampling`, `reports`, `runreports`, `alignment`) behind the same front
  module; `from substrata import visualizations` and every `visualizations.<name>` work as before. Each submodule
  names the measurement functions its numbers come from.
- Compute modules import figure libraries only when drawing: `import substrata.measurements` takes 0.07 s instead of
  1.2 s (substrata-explore imports it).
- Fixed: `intercepts --slope` drew `slope_pca.png` from a refit on the already aligned cloud (about 0°), not the stored
  `slope_elevation_deg`; the classifier stats PDF crashed with NumPy ≥ 2.3.

**Consistent names** (old names stay as deprecated aliases; old keywords still work and raise a `DeprecationWarning`
once, `substrata.compat.renamed_kwargs`)
- One name per concept: `output_path` / `output_dir` for what a function writes, `progress` for progress bars, unit
  suffixes `_m`, `_mm`, `_px`, `_deg`, `_s` (e.g. `point_size_m`, `offset_s`), `transform` for a 4×4 argument.
- Lower case: `conduct_pca`, `conduct_xy_pca`, `get_best_fit_plane_pca`, `get_radius_from_2d_surface_area`,
  `principal_axis_xy`, `calc_scale_factor`, `get_time_diff_s`.
- `Cameras` / `Annotations`: `.apply_transform` / `.reverse_transform` (were `transform_coords` /
  `get_original_coords`; single cameras and annotations keep `transform_coords`).
- Helpers used across modules (or by substrata-explore) have public names (e.g. `photo_folder_name`,
  `lens_distortion`, `pixels_to_normalized`, `PLY_TYPES`); the underscore names remain.

**Removed** (unused: checked in substrata, substrata-explore, the tests, docs and notebooks)
- About 5,000 lines (the package is 600 lines smaller despite the new helpers; the tests grew from 13,400 to 17,300
  lines): the SIFT / heat-map experiments in `segmentation`, 16 unused figure functions, 30 functions in `measurements` (old grid,
  transect and intercept code, `measurements.slerp` (duplicate of `geom.slerp`), `get_fractal_dimension`, …), the
  `Video` / `Frame` classes and other unused `Camera(s)` / `Annotations` methods, unused `geom` constructors and
  `PointCloud` helpers. `measurements` has an `__all__`.

### 24 November 2026

- **`substrata images --layout pages`** (`annotationpages.py`): one A4 page per annotation - the best overview photo
  zoomed on the point (most relevant unobstructed view of the main camera type, as `crops` chooses it; `--zoom`,
  at least 0.5 m or 2.5x the close-up's frame), the close-up's full frame (box = the crop below) and the close-up at
  100% (1 photo pixel = 1 dot at 300 dpi); point marked, scale bars from camera distance, header with plot,
  annotation, label, depth and `--info` columns. `--layout auto` (default) makes pages when the stored views are
  close-ups (a `cams2annos` camera-targets set, or stored photos of a non-main camera type), else the grid as
  before; `--closeup GROUP` names the close-up cameras for any set (points without one get overview-only pages).
  A stored close-up is shown also when its camera is disabled (a pose camsync estimated), with a note to check it.
  The overview prefers views with the point clear of the photo edge (`settings.PAGES_CLEAR_OF_EDGE_PX`).
  `--labels` / `--ids` select annotations in both layouts; `--occlusion` / `--camera-type` as for `crops`.
- **Display contrast** (`imageadjust.py`, `substrata images --contrast`): `auto_contrast(img, method,
  reference=)` with `stretch` (default: one luminance stretch, colours kept), `stretch-rgb`, `clahe`, `none`;
  statistics from a reference image so crops of one photo match its full frame; new methods via
  `@register_contrast_method`.

### 20 November 2026

- **`substrata track-video` camera poses** (`videopose.py`; `--no-poses` to leave out): each stop's camera position
  and orientation (`cam_x .. cam_qw`, `pose_px` in `stops.csv`) and per camera setting a lens calibration
  (`videotrack.json`: `lenses`), from the stop frames' matches with the survey photos (3D from the photos' depth
  maps): one lens (fisheye + SuperView stretch) fitted with all poses, then each frame's pose. SAV03: 3 px
  reprojection error, the two frames of a stop 4 cm / 1.3 deg apart.
- **`substrata track-video`**: a camera's videos over a plot are one *sequence* (its stops numbered in order over
  the videos, `seq_stop`); stops with no frame placed get an estimated position (`placed` = `estimated`,
  interpolated in time between the sequence's placed frames); each stop's heading and footprint (`heading_deg`,
  `width_m`; cached tracks get them on the next run); maps draw each sequence as a smooth line through its numbered
  stops. The stops set of a plot is now added to (new stops, updated stop columns, backed up) instead of left alone;
  points are never moved, relabelled or re-added once removed. Fix: a stop placed on one plot only by frames kept
  while still could lose to its unplaced copy on another plot.
- **`track-video` stops sets mark estimated positions with `is_extrapolated`** (True / False), as intercepts and
  camera targets do (was a `placed` column, removed from existing sets when run again).
- **`track-video` `diver` column** (`stops.csv`, the stops sets): the diver each camera filmed, from `labels.json` in
  `--out` (named in substrata-explore's video player).

### 19 November 2026

- **`substrata track-video`** (`videotrack.py`): videos filmed over plots (a GRAV camera filming coral sampling)
  placed on them: GoPro keyframes (one a second), camera motion and stops (still >= 10 s), which plots each video is
  on (12 probe frames), LoFTR matches with the plots' survey photos (turned north-up; a seed grid, then tracking
  round the last position), the frame centre followed to the cloud. Outputs: `stops.csv`, `track.csv`,
  `videos.csv`, a map per plot, `README.txt`, `videotrack.json` (substrata-explore `--videos`), H.264 proxies
  (`--proxy`) and the stops as each plot's point set `<id>.annos/<id>_video_stops.csv` (never overwritten).

### 18 November 2026

- **`substrata camsync --match-images`** (`imagesync.py`): close-ups posed from image matches with the GoPro frames
  instead of copying the pose of the frame at the same EXIF time. Steps:
  - the time offset from the images: SIFT on every close-up / frame pair (whole frame at 1600 px), each close-up
    voting with its scores / its best; the highest peaks are confirmed with LoFTR on up to 12 close-ups
    (`--time-offset` skips the search, `--offset-window` narrows it);
  - each close-up matched with kornia's LoFTR (or DISK + LightGlue, `--matcher`) to the frames within `--frames` s
    (round where close-ups usually lie in a frame, then the whole frame, turned as step 1 found them: sa4's GoPro
    was upside down; then frames up to `--wide-frames` s off), then PnP on the frames' rays to the cloud;
  - checks (inliers, error, range, the second matcher's aim across the view; `--no-check`);
  - close-ups with no aligned frame near their time are reported as `no frames`, with the unaligned frames;
  - `<id>_camsync_images.pdf`; after a confirmation, the poses in the meta JSON (backed up first; cameras without a
    good pose get a null pose, disabled) and `<id>_camsync_poses.json`.

  Dry runs (19 November): sa3_40m offset -81 s, 82 of 83 close-ups posed (they had been synced at -121 s by hand;
  those poses were 1-5 m off); sa4_40m -3 s, 82 of 90; sa4_60m -2 s, 26 of 46 (13 without an aligned frame near
  their time: Metashape aligned 331 of its 478 frames). Camera to reef about 0.30 m. kornia is an optional
  dependency (the `match` extra).
- **`camsync --match-images --pose-frames`** (`framepose.py`, `frameretrieval.py`): the GoPro frames Metashape left
  unaligned at the close-ups' times are posed in substrata (LoFTR with the survey photos and aligned frames, pixels
  followed to the cloud, PnP pooled over photos that agree; gaps chained from their edges 1 s at a time, only on
  frames that photos with their own pose agree on; a frame with no posed frame within 10 s placed by DINOv2
  retrieval of the survey photos). Only used in the run, not saved. `--frame-photos` picks the sensors matched with
  (default: those showing the reef in at least 60% of the best sensor's detail, so not an AUV's photos from 3 m);
  `--frame-calibration OTHER.cams.xml` borrows the frames' lens where none was calibrated, and
  `--frame-calibration-orientation auto` (default) tells from the frames whether it was mounted turned 180 deg
  there. sa3_40m hold-out: frames 2-3 mm from Metashape's poses (median). sa4_60m dry run: 101 of 129
  unaligned frames posed, close-ups 45 of 46 posed (26 without; offset -3 s with all frames voting); the 26 posed
  either way aim within 1.3 mm (median, 2.4 cm max). sa3_60m, where Metashape aligned none of its 701 frames
  (lens from sa3_40m): offset -82 s (sa3_40m, same day: -81), 390 of 452 frames posed (11 placed by retrieval, the
  rest chained), close-ups 74 of 83 posed (66 confirmed by the second matcher), ~67 min. Orientation, all frames
  treated as unaligned, another plot's lens: turned chosen on sa4_60m, sa4_40m and sa3_40m (17 of 17 frames fit
  turned better); a lens the same way up kept as given (sa4_60m with sa4_40m's: 2 of 5, not clear).
- **Close-ups without a good match still get a pose, marked** (`camsync --match-images`; `--no-estimate`: none):
  `estimated` (no match: the frame's pose at its time + offset and the rig, the close-up's median pose relative to
  its frame over the posed close-ups; sa3_40m / sa4_40m leave-one-out: aims 4-6 cm from its matched aim, median,
  against 21-23 cm for the frame's pose copied) or `unverified` (the matchers disagree; LoFTR's pose kept). Saved
  disabled, with `pose_status` in the meta JSON (`matched` for the posed), read back on load; `cams2annos` marks
  the points of estimated poses `is_extrapolated` (as `intercepts` does); `metashape-import-poses` sets them disabled (`--posed-only`: not at all). The check PDF
  also shows posed close-ups at an unusual range.
- **`Cameras.save`**: a camera whose pose changes or is cleared loses its `center_crs` (the CRS centre exported
  from Metashape, which belonged to the old pose).
- **`substrata metashape-import-poses`** (`metashape_scripts/import_camera_poses.py`): the poses file into the .psx
  (cameras by key, label checked). The GoPro frames' transforms in the three plots' .psx equal those in cams.xml, so
  the poses land in the same frame.
- A sensor without a calibration in cams.xml gets its nominal lens (`focal_length` / `pixel_width`, no distortion;
  `Sensor.calibration_class` "nominal") instead of none. Its cameras still see nothing until they have a pose.
- `Cameras.save` writes a null pose for a camera whose pose was taken away (`imagesync.clear_pose`).
- Removed `segmentation.estimate_macro_camera_pose_from_gopro` (and its helper `_perform_detailed_sift_matching`):
  never called, and it did not work. It returned OpenCV's world-to-camera translation as the camera's position, and
  reported success whatever the result. Its outlier threshold (2 px at the macro photo's full resolution, for
  features found at 800 px) let almost no correct match through. On five sa3_40m close-ups every pose was wrong
  (reprojection errors of 570-73,000 px).
- **`substrata cams2annos`** (alias `cams-to-annos`) and **`PointCloud.camera_targets(cams)`**: the points a camera
  group's photos aim at (default the close-ups in `<id>.closeup/`): where the ray through each photo's centre (or
  `--pixel`) first meets the cloud, refined to the surface point nearest the ray, as an annotation set
  `<id>.annos/<id>_<group>[_<tag>]_targets.csv` with each photo as its point's view (`image_match`; `cam_filepath`,
  `cam_x`, `cam_y`). A ray through a gap is tried 3x wider (3 mm up to 3 cm). Sidecar `type: camera_targets`,
  `purpose: training_only`. sa3_40m: 78 of its 83 close-ups (10 s once the cloud is loaded).

### 17 November 2026

- **Unmarked image sets**: `imagesets.is_image_set(folder)`; a folder with `<id>.photos/` (or `<id>.annos/` with a
  `*_points.csv`) and neither `<id>.yaml` nor `<id>.ply` is an image set with the defaults (no camera, gsd_mm or
  classes), and `read_image_set` reads it (`yaml` None) instead of raising. substrata-explore lists these.

### 16 November 2026 (afternoon)

- **`substrata predict --camera-group GROUP`** (`grouppredict.py`) predicts a set's points from one camera group's
  photos only, including cameras disabled in the project (e.g. AUV photos aligned into a diver plot). A group is a
  camera type, a camera system (a resolution), a photo folder / file-prefix group, `enabled` or `all`.
  - `--view-mode segment` (default) chooses and combines views exactly as `substrata segment` does; `predict` does
    as `substrata predict` does.
  - Results go to `<set folder>/predictions/<set>__<group>__<mode>.csv` (+ `.json` with the run's settings and
    counts), with each point's views available and used. The set's CSV is not touched.
- `segmentation.classify_query_points`: segment's matching and classification of query points, now shared with
  the above (segment's results are unchanged). `_match_points_to_cameras_multi` can include disabled cameras and
  count each point's cameras in view; `_classify_dense_views` also returns the fused probabilities.
- `Annotations.predict` / `_collect_image_matches`: `enabled_cameras_only` (default True, as before).
- Version 0.2.0.dev2 (substrata-explore's camera choice for predictions needs it).
- **Cameras that cannot project are skipped** instead of stopping `crops`, `predict` and `segment`: an enabled
  camera whose sensor has no calibration in cams.xml (`sensor` None) or that was loaded with a placeholder pose
  (null in meta.json) sees nothing (`cameras.can_project`). `pixel_hits` leaves them out of its stack (it fell back
  to one camera at a time and failed in the lens model: sa3_60m's 83 close-ups and 701 video frames);
  `get_pixel_coords` returns None, `project_points` NaN / not in view. A placeholder pose with a sensor
  would have projected from the origin.

## 0.2.0.dev1

### 16 November 2026

- **Lens model: Metashape's frame camera model exactly.** P1 now goes with x and P2 with y (they were the other way
  round: OpenCV's order), and K4, P3, P4 and the skew B2 are read from cams.xml and used. Every projection
  (`get_pixel_coords`, `project_points`, `pixel_hits`) and its inverse (`pixel_to_ray`, now Newton's method,
  exact to ~1e-12 px) follow it; the macro-camera PnP (`segmentation`) undoes the distortion itself instead of
  passing OpenCV coefficients in OpenCV's order. Checked against Metashape 2.2's own `Calibration.project` for every
  Samoa sensor (to 4e-12 px; `tests/test_lens_metashape.py`, with the reference values in `tests/data/`) and on real
  photos (the tie points of sa3_20m: residuals now equal Metashape's, e.g. 100 → 25 px RMS on PGRAM02, 5.5 → 1.4 px
  on the AUV camera). Pixels move by a few px on most sensors, up to ~20–100 px near the edges, 546 px on sa3_20m's
  PGRAM02.
- **In view only before the lens folds back**: the in-view bound is the radius where the radial distortion's
  derivative turns 0 (`Sensor.r2_max`), else 1.005 × the longer image side (was the width: the top and bottom of
  portrait video frames were out of view). Beyond the fold, points far outside the field landed in the image: on
  sa3_20m's PGRAM02, 879 of its 1,695 training views were points ~50° off the axis.
- **`substrata reproject-views`** (`lensfix.py`): moves the training views stored in a plot's annotation CSVs to the
  corrected pixels, clears those the corrected model does not see the point in (listed to look at again), backs up
  first; `--dry-run`. See the command-line docs.
- Version 0.2.0.dev1 (substrata-explore requires it: its views and crops are the corrected model's).

## 0.2.0.dev0

### 13–14 November 2026

- **One projection core** in `cameras.py`: `get_pixel_coords`, `project_points` and `pixel_hits` share the same
  distortion and relevance code (identical results, tested against the previous implementations in
  `tests/test_projection_equivalence.py`). The cameras' arrays are stacked once per `Cameras` (cached, rebuilt when
  cameras or transforms change); world-frame `pixel_hits` is ~35x faster (15 ms → 0.4 ms for 3,000 cameras).
- (Found then, fixed in 0.2.0.dev1: the lens model's P1 / P2 order, K4 and B2.)
- **`substrata train` with a kept label map** (`--label-map … --keep-map`) no longer stops with "No bolded training
  labels were derived" when the classes file is flat (every class at the root, as in a simple classes.csv): the
  map sets the training classes.
- **`fit_point_probe`**: the regularisation search by hand (stratified folds, macro-F1), as `LogisticRegressionCV`
  did; scikit-learn 1.9 raised for two-class problems.
- Tests: the point-cloud tests remove their stub modules after loading; the curation tests skip without torch.

### 12–13 November 2026

- Refactoring for substrata-explore, several bug fixes, the slope orient function (see the git history).
