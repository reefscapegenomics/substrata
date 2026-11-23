# Changelog

The version is `substrata.__version__` (`src/substrata/__init__.py`; pyproject.toml reads it). substrata-explore
checks it at start-up.

## 0.2.0.dev2 (unreleased)

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
