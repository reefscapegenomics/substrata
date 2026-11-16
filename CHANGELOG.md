# Changelog

The version is `substrata.__version__` (`src/substrata/__init__.py`; pyproject.toml reads it). substrata-explore
checks it at start-up.

## 0.2.0.dev2 (unreleased)

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
