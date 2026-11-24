# Command-line usage of `substrata`

This tutorial walks through the available command-line tools of the `substrata` Python package.

Note: All commands look for a YAML project file or default filenames in the current working directory, unless separately specified. Most commands use the `ProjectInitializer` to auto-detect project files based on the current directory name.

## Decimation of PLY files

Decimate a PLY file to reduce the number of points. With no arguments, uses initializer on CWD, output to `<id>_dec50M.ply`, target = 50,000,000 points.

With `-c` (`--color-calibrate`), the YAML's `color_correction` (from `substrata colors`) is applied to the RGB while writing. The output's header is then marked, so loading it does not apply the correction again (see [Color calibration](#color-calibration)).

Usage: `substrata decimate [--input PLY] [--output PLY] [--points N] [--seed N] [-c]`

```bash
# Using default behavior (auto-detects from CWD)
substrata decimate

# With explicit arguments
substrata decimate --input cur_sna_20m_20200303.ply --output cur_sna_20m_20200303_dec50M.ply --points 50000000

# Using short flags
substrata decimate --ply input.ply -n 10000000

# Bake the YAML color_correction into the decimated PLY
substrata decimate -c
```

## Interactive viewer (explore)

`substrata explore` opens the interactive viewer (point cloud, photos, annotations, classification) for the project in the current folder, in a browser. It runs the separate `substrata-explore` package (same as the `substrata-explore` command) and takes its arguments, e.g. `substrata explore --edit`; see `substrata explore -h`.

## Stratifying PLY files

Reorder a PLY so it can be read progressively and by region. The file stays the project's ordinary `<id>.ply` (same header plus a `comment substrata stratified v1` line, byte-identical points, only reordered) with a small index `<id>.strat.npz` next to it; a PLY counts as stratified when that index exists and matches it (`ProjectInitializer.ply_stratified`). The points are split into spatially compact clusters, each shuffled and written coarse-to-fine, so:

- the first N points of the file are a uniform random subsample when N is a round boundary (a decimated cloud is just a prefix read, `StratifiedPly(path).read_prefix(n)` with n from `index.round_ends`);
- a region can be read at any density without scanning the whole file (`StratifiedPly(path).query_box(lo, hi)`);
- other PLY readers (Open3D, Metashape, CloudCompare) still read it as before.

The file and index layout, for implementing readers in other software, is in {doc}`stratified_ply`.

With no arguments, stratifies `<id>.ply` of the project in the CWD in place (it is written next to it first and checksummed, then swapped in). It prints the mode it will use and the RAM and disk space that needs.

Two modes, same result:

- `memory`: reads the whole PLY into RAM, orders it there and writes it back in one sequential pass. Needs RAM for the points plus 8 B per point (~100 GB for 2.7 B points of 28 B), and disk space for one copy of the PLY. Fastest by far on hard disks (about 20 min for 75 GB on a ~190 MB/s disk, where `stream` takes hours).
- `stream`: three passes through a cluster-sorted temp file, little RAM. Needs disk space for two copies of the PLY (`--tmp-dir` or `--work-dir` can hold one of them).
- `auto` (default): `memory` if it fits in the RAM available (within any systemd / container / job memory limit), else `stream`.

Usage: `substrata stratify [--input PLY] [--out-dir DIR] [--mode auto|memory|stream] [--work-dir DIR] [--tmp-dir DIR] [--cluster-size N] [--keep-original] [--force]`

`--work-dir` (or `$SUBSTRATA_WORK_DIR`) writes the new PLY, and the temp file, on another disk first (e.g. an SSD when the PLY is on a hard disk) and then copies it next to the PLY. `--keep-original` keeps the unstratified PLY as `backup_<name>.ply` next to it (a rename, so it costs no copy, but the disk space stays used).

```bash
# In place, auto-detected from CWD
substrata stratify

# Low on RAM: stream, with the temp file on a fast local disk
substrata stratify --mode stream --work-dir /fast/tmp

# A stratified copy on a fast local disk (same file name)
substrata stratify --out-dir /fast/plots
```

## Metashape project export

Initialize a substrata project directly from an Agisoft Metashape project (`.psx`). This writes the point cloud (`<id>.ply`), a decimated copy (`<id>_dec50M.ply`), camera calibration/poses (`<id>.cams.xml`, `<id>.meta.json`), markers (`<id>_markers.csv`), and a starter project file (`<id>.yaml`) — everything in raw chunk-local coordinates. Scale and orientation are left for `substrata orient` (or `substrata firefish`) to compute afterward.

Like the other commands, it follows substrata's folder convention: with no arguments it treats the current directory (named `<id>`) as the project folder, exports from `<id>.psx` inside it, and writes all the files alongside it. The common workflow is simply to `cd` into the project folder and run it with no arguments.

The export itself must run inside Metashape's own bundled Python (`substrata` and its dependencies cannot be installed there), so this command shells out to the Metashape executable and runs a packaged export script under it. Metashape must be installed locally. The executable is resolved from `--metashape`, then the `METASHAPE_EXE` environment variable, then common install locations. The point cloud is decimated to `<id>_dec50M.ply` afterward (open3d runs in the substrata environment, not in Metashape's).

By default, outputs that already exist are skipped with a warning and the remaining files are still generated, so a re-run fills in only what is missing (for example, running `--metadata-only` first and then the full command later). Pass `--overwrite` to regenerate everything.

Usage: `substrata metashape-export [--psx PSX] [-o OUTPUT_DIR] [--id ID] [--chunk CHUNK] [--metashape PATH] [--metadata-only] [--overwrite]`

```bash
# Common usage: run inside the project folder (uses <foldername>.psx)
cd cur_sna_20m_20200303
substrata metashape-export

# Cameras + markers only, skipping the (slow) point cloud and decimation
substrata metashape-export --metadata-only

# Explicit project file and Metashape launcher
substrata metashape-export --psx project.psx --metashape ~/tools/metashape-pro/metashape.sh

# Write into a different output directory
substrata metashape-export --psx /data/raw/survey.psx -o /data/projects/cur_sna_20m_20200303

# Choose a specific chunk (by label or 0-based index)
substrata metashape-export --chunk "dense_chunk"

# Re-run and overwrite everything (otherwise existing files are skipped)
substrata metashape-export --overwrite

# Typical next step (in the same folder)
substrata orient     # compute scale + world_transform
```

## PLY repair (Open3D-compatible rewrite)

Rewrite a PLY in a strict Open3D-compatible form (float32 xyz, optional `uchar` RGB, optional float32 normals). Extra vertex properties and non-finite rows are dropped. By default the input is renamed to `<input>_old.ply` and the repaired file is written in its place.

Usage: `substrata ply-repair [--input PLY] [--output PLY] [--local]`

```bash
# Repair in place (original kept as <input>_old.ply)
substrata ply-repair --input pointcloud.ply

# Write the repaired copy to an explicit path (input left untouched)
substrata ply-repair --input pointcloud.ply --output pointcloud_fixed.ply
```

## PLY file preview (head)

Show the first N vertex rows from a PLY file. For a stratified PLY (see `substrata stratify`) it also prints the index summary: point count and properties, bounding box, clusters, rounds and the prefix ends (how many leading points form each uniform subsample).

For a project with a `baseline_id` it also says whether its `world_transform` still follows the baseline (see `substrata baseline`).

Usage: `substrata head [--input PLY] [-n N]`

```bash
# Show first 5 rows (default)
substrata head

# Show first 10 rows
substrata head --input pointcloud.ply -n 10
```

## Visual assessment of scalebars

Generate a scalebar PDF from a point cloud and marker annotations. Optionally save the computed scale factor to YAML. By default the project's point cloud is used: `<id>_dec50M.ply` when present, otherwise `<id>.ply` (or the `ply:` entry of the project YAML). The point cloud is only used for the figures; the scale factor comes from the marker coordinates.

Usage: `substrata scalebars [--input PLY] [--markers CSV] [--output-pdf PDF] [--points N] [--save-yaml]`

```bash
# Using default behavior (auto-detects from CWD)
substrata scalebars

# With explicit arguments
substrata scalebars --input cur_sna_20m_20200303_dec50M.ply --markers cur_sna_20m_20200303_markers.csv --output-pdf ~/scalebar_check.pdf

# Save scale factor to YAML
substrata scalebars --save-yaml

# Limit points loaded (stream decimation)
substrata scalebars --points 10000000
```

## Composite views

Save composite views PDF for a point cloud showing multiple perspectives.

Usage: `substrata views [--input PLY] [--output-pdf PDF]`

```bash
# Using default behavior
substrata views

# With explicit output path
substrata views --input pointcloud.ply --output-pdf views.pdf
```

## Orientation and scaling

Calculate and apply scale and orientation transforms, then save to YAML. Also generates composite views and camera depth residuals PDFs.

Usage: `substrata orient [--input PLY] [--use-chunk-transform [--north-up]] [--markers CSV] [--transform] [--manual] [--slope]`

```bash
# Using default behavior (auto-detects from CWD)
substrata orient

# With explicit PLY path
substrata orient --input pointcloud.ply

# Use the Metashape chunk transform instead of fitting scale/orientation
substrata orient --use-chunk-transform

# Same, but keep the map north-up (+Y north) instead of +Y upslope
substrata orient --use-chunk-transform --north-up

# Orient, then align along the slope and save that frame
substrata orient --slope
```

With `--slope`, the oriented cloud is then aligned along its slope exactly as `substrata intercepts --slope` does (a plane is fitted to the terrain and rotated flat about its inlier centroid, then X is aligned with the footprint's principal axis), and the result is saved in the YAML `world_transform`. Z is then height above the slope plane rather than depth (`depth_in_m` still works, as it uses the original coordinates), and later `intercepts --slope` runs are close to a no-op. It runs after any other option, including `--transform`.

With `--use-chunk-transform`, scale and orientation are taken from the `chunk_transform` stored in `<id>.meta.json` (written by `substrata metashape-export`) instead of being fitted to scalebars and camera depths. For a georeferenced chunk this transform maps to geocentric (ECEF) metres, so it is combined with a rotation into a local East-North-Up frame centred on the point cloud: +X east, +Y north, +Z ellipsoidal height (i.e. −depth when the camera references are depths). The cloud is then rotated about Z so +Y points upslope (shallow end at the top of top-down views, also for plots that are elongated downslope): of the plot footprint's long and short axes, the one closest to the fall line of a plane fitted to the terrain is aligned with +Y, so the plot edges stay parallel to X/Y (axis-aligned boxes fit snugly). The cloud is then shifted to positive X/Y; this rotation leaves scale and depth untouched. Add `--north-up` to skip the rotation and keep the georeferenced orientation (+X east, +Y north). The scale factor saved to the YAML is the uniform scale inside the chunk transform, and `up_vector`/`depth_offset`/`depth_per_unit` describe the same transform, so the depth residuals PDF still compares recorded and model depths. This option cannot be combined with `--manual` or `--markers`.

## Segmentation

Segment the point cloud by reusing a trained classifier: sample a grid of query points across the cloud, project each to its best camera photo, classify the patch around it, then propagate the labels to every point via nearest neighbour. The patch is as large as the classifier's training crops (recorded in the classifier by `substrata train`; `--crop-size` overrides it, and older classifiers without a recorded size use 224 px). With a point-feature classifier (`point_classifier.pt`, the `substrata train` default) no crops are cut. Each query keeps its `--views` best photos (default 3, by the same relevance as the best photo: close and central; with occlusion filtering every view is checked), every photo that is a view of some query gets one dense probability map (the photo is resampled at its median ground resolution over its views - the project's metres per model unit come from its `world_transform`, or its `scale_factor` when the transform is the identity - or, without one, by its width as `predict` and `masks` resample it; cut into overlapping 448 px tiles whose per-cell probabilities are blended with centre-weighted weights, so there are no tile seams), and a query's prediction is the mean of its views' probabilities read at its pixel, weighted by relevance as in `predict --method weighted`. Each point then also records how many photos were fused (`pred_n`) and their agreement (`pred_agreement`: the weight share of the views whose own label is the fused one; 1 = all agree) - low agreement flags occlusions, boundaries and ambiguous surfaces worth checking by hand. `--save-masks [DIR]` also keeps every photo's dense map as masks (default folder `<id>_seg[_<tag>]_masks/` in the plot folder; see *Masks* below for the format), e.g. for 2-D area measurements. `--per-point` instead classifies each query from a window around it in its best photo only (the classifier's training view; no fusion; its cost grows with the number of queries, the dense path's with the number of photos). For scale: on a 40 m diver plot (2,510 photos, 200 k queries at 5 cm, RTX 4080) the dense path took ~18 min with 3 views and ~11 min with 1; more views were a little more accurate on held-out annotations (`--views 5` for the best results, at more time). Saves the classified query points as an annotation set `<id>.annos/<id>_seg_points.csv`, the segmentation's source of truth: predictions in `pred_label`/`pred_conf`, `label` left empty to check by hand, and the photo and pixel each point was classified from in `cam_filepath`/`cam_x`/`cam_y`. Its sidecar records `type: segmentation`, the classifier, cell size, sampling, crop size and the label colours. It also writes a recoloured decimated cloud `<id>_seg_dec.ply` (category colour blended with each point's original luminance) in the project folder; the sidecar names the recoloured clouds made from the set (`seg_ply_dec`, `seg_ply`). `--tag NAME` keeps versions side by side: `<id>.annos/<id>_seg_NAME_points.csv` and `<id>_seg_NAME[_dec].ply`. `--from-points CSV` skips sampling and classification and only recolours from an existing points CSV, where a hand-checked `label` wins over `pred_label` (the clouds keep that set's tag). `--output-npz` also saves the compact `.npz` of `Segmentation.save` (not written by default; `Segmentation.from_annotations` reads the CSV).

Because the points CSV is an ordinary annotation set, its labels can be summarised like any other (e.g. cover per label over depth). With `voxel` sampling each point stands for about the same patch of surface (one per `--cell-size` cell), so a label's share of the points is its 3D (surface) cover; with `xy_grid` it is planar (top-down) cover. Query points that no photo saw, or that failed to classify, are left out, so the cover is of the observed surface. Query matching is fully vectorized and occlusion-aware; cost scales with the number of query points (the spacing), not the number of cloud points. With `--full-ply` it additionally stream-recolours the full-size PLY (memory-safe, never loaded whole).

A points CSV whose labels were checked by hand is not overwritten by a new run of the same version (use `--tag`, `--from-points` to recolour from it, or `--overwrite`).

Usage: `substrata segment [--input PLY] [--classifier PKL|PT] [--cell-size M] [--crop-size PX] [--sampling voxel|xy_grid] [--no-occlusion] [--views N] [--save-masks [DIR]] [--per-point] [--max-radius M] [--color LABEL=RRGGBB ...] [--full-ply [PLY]] [--tag NAME] [--overwrite] [--from-points CSV] [--output-npz NPZ] [--output-ply PLY]`

```bash
# Using default behavior (auto-detects project + classifier from CWD)
substrata segment

# Use a specific classifier and 10 cm query spacing
substrata segment --classifier crop_classifier.pkl --cell-size 0.10

# Point classifier: fuse 5 views per point and keep each photo's masks
substrata segment --classifier point_classifier.pt --views 5 --save-masks

# Manual category colours (repeatable): coral=red, macroalgae=green, substrate=yellow
substrata segment --color CS=ff0000 --color MA=00ff00 --color SU=ffff00

# Top-down sampling (faster; may leave gaps at oblique angles) and no occlusion filtering
substrata segment --sampling xy_grid --no-occlusion

# Also stream-recolour the full-size PLY to <id>_seg.ply
substrata segment --full-ply

# A second version, kept next to the first: <id>.annos/<id>_seg_v2_points.csv, <id>_seg_v2_dec.ply
substrata segment --tag v2 --cell-size 0.10

# After checking labels in that set by hand: recolour from it without classifying again
substrata segment --from-points <id>.annos/<id>_seg_v2_points.csv --full-ply
```

## Color calibration

Run color calibration from the project's ColorChecker marker positions, save a QC PDF, and store the affine color correction (matrix + offset) in the project YAML as `color_correction`. The YAML is updated in place: only that key changes, and comments and relative paths are kept; the fit's RMS residual (`fit_rmse`, RGB units over the fitted patches), the number of patches fitted (`fit_patches`) and its colour space (`fit_space`: sRGB 0-255) are recorded with it and shown in the PDF's summary page. Use `--no-save-yaml` to only write the PDF. The PLY is always loaded in full when `--input` is given (never stream-decimated).

The patches are sampled within 5 mm of their centres and 3 mm of the card plane (metres), so the cloud and the markers are both taken into the project's world frame (its `world_transform`), also with `-n` and with `--markers`; run `substrata orient` first (a project without a `world_transform` is assumed to be in metres, with a warning). Each patch's measured colour is the median over the cards; a card whose value of a patch lies more than 2.5 robust SDs (1.4826 x MAD) from the cross-card median is flagged as an outlier and left out of that patch's median. The affine fit is in gamma-encoded sRGB 0-255 (not linear light).

No PLY is written. Loading the project (`ProjectInitializer.initialize()`) applies `color_correction` to the point cloud in memory, so the original colours stay on disk. To write a colour-corrected PLY as an explicit extra step, use `substrata decimate -c`. Its output header is marked (`comment substrata color_correction applied`), and loading a marked PLY skips the YAML correction, so it is never applied twice. Decimating a marked PLY with `-c` again is refused.

Usage: `substrata colors [--input PLY] [--markers CSV] [--output-pdf PDF] [-n N] [-s | --no-save-yaml] [--exclude-index N ...] [--exclude-name NAME ...]`

```bash
# QC PDF + color_correction into the project YAML (auto-detects from CWD)
substrata colors

# QC PDF only; leave the YAML unchanged
substrata colors --no-save-yaml

# Stream-sample the default YAML PLY to N vertices for a faster QC pass
substrata colors -n 10000000

# Omit specific ColorChecker cards from the fit (by 0-based index or by name)
substrata colors --exclude-index 0 --exclude-index 3
substrata colors --exclude-name top-left

# Optional extra: write a colour-corrected decimated PLY
substrata decimate -c
```

## FireFish alignment

Run FireFish/Cameras alignment to determine up vector and generate output PDF. Initializes FireFish and Cameras, then determines the up vector based on camera depth data.

The time offset between the cameras and the FireFish log is searched on the records near the plot depth: `--target-depth M` is the depth below the surface in metres (`20` and `-20` both mean 20 m deep; default: from the folder name, `_20m_`), and only the longest stretch of records within `--depth-outlier-threshold` (default 4 m) of it is used. The same threshold also drops altitudes at or above it as outliers. A depth band without any record stops with an error naming the depths the log covers.

Usage: `substrata firefish [--firefish-file FILE] [--target-depth M] [--cam-depths-file CSV] [--depth-outlier-threshold M] [--cams-group GROUP] [--offset SEC] [--beam-half-angle DEG] [--distance-percentile P] [--sonar-offset X Y Z] [--depth-sensor-offset X Y Z] [--water-level M] [--recompute-camdists] [--input PLY] [--save-yaml]`

The camera-derived altitude (`camdist`, plotted against the FireFish altitude) is the distance from the camera to the point cloud within a cone around the camera view vector. It is matched to the FireFish Ping2 echosounder by default:

- `--beam-half-angle` (default 12.5): cone half-angle in degrees (the Ping2 has a 25° full −3 dB beamwidth).
- `--distance-percentile` (default 5): percentile of the in-beam distances, as a proxy for the sonar's first return (nearest surface). Use `-1` for the mean (the behaviour before these options existed, which used a 15° half-angle).
- `--sonar-offset X Y Z` (default `0 0 0`): sonar position relative to the camera centre in metres, in the camera frame (x: image right, y: image down, z: view direction).
- `--recompute-camdists`: `camdist` is otherwise only computed when the camdepths CSV does not exist yet.

The up vector is fitted by regressing the FireFish depths against camera positions. If the depth sensor is mounted away from the camera centre, `--depth-sensor-offset X Y Z` (metres, camera frame; default `0 0 0`) regresses the depths against the sensor positions instead, following each photo's orientation (lever-arm correction). For example, a sensor 15 cm behind the camera centre along the view axis is `0 0 -0.15`. With `--save-yaml` the offset is recorded in the YAML as `firefish_depth_sensor_offset`.

The FireFish measures depth below the water surface, which moves with the tide. `--water-level M` (default 0) gives the water level relative to mean sea level during the survey (e.g. from a nearby tide gauge; negative below MSL), and converts all FireFish depths to depth below MSL (depth + water level) before the time sync and regression. It is recorded in the YAML as `firefish_water_level`.

```bash
# Using default behavior (auto-detects from CWD, infers depth from directory name)
substrata firefish

# With explicit target depth (20 m below the surface)
substrata firefish --target-depth 20

# Filter cameras by group
substrata firefish --cams-group "group_name"

# Save results to YAML
substrata firefish --save-yaml

# With manual time offset
substrata firefish --offset 30

# Recompute camdist for a sonar mounted 15 cm from the camera (image-down direction)
substrata firefish --offset 30 --sonar-offset 0 0.15 0 --recompute-camdists

# Depth sensor 15 cm behind the camera centre (lever-arm correction)
substrata firefish --depth-sensor-offset 0 0 -0.15

# ... and reference depths to mean sea level (survey at 0.32 m below MSL)
substrata firefish --depth-sensor-offset 0 0 -0.15 --water-level -0.32
```

## Camera video creation

Create a video from cameras by drawing image matches. Optionally include annotations in the video.

Usage: `substrata cams2video [--input PLY] [--annotations CSV] [--cams-group GROUP] [--label] [--resolution WIDTH] [--output-mp4 MP4]`

```bash
# Using default behavior (auto-detects from CWD)
substrata cams2video

# With annotations
substrata cams2video --annotations annotations.csv

# Filter cameras by group
substrata cams2video --cams-group "group_name"

# Use label column from annotations
substrata cams2video --label

# Resize images to specific width
substrata cams2video --resolution 1920

# Specify output file
substrata cams2video --output-mp4 output.mp4
```

## Image sets and points on photos (sample-points)

An *image set* is a folder of photos without a 3-D model, laid out like a plot: `<id>/<id>.yaml` with `type: images` (optionally `photos_path`, `camera`, `gsd_mm`, `classes`), the photos in `<id>.photos/` (subfolders allowed), and its point sets in `<id>.annos/`. `substrata sample-points`, run in its folder, samples random points on every photo (`--n-points` per photo, `--stratified` for one per grid cell, `--seed` for the same points again, `--margin` px from the edges, default half a crop) into `<id>.annos/<id>[_TAG]_points.csv`: an annotation set with each point's photo and pixel (`cam_filepath`, `cam_x`, `cam_y`) and no 3-D coordinates, to label, predict and train on like any other. A folder with `<id>.photos/` but no `<id>.yaml` (or `<id>.ply`) is an image set already, with the defaults; `sample-points` writes its YAML (`type: images`) first. A set with labels is never overwritten (`--tag` for another, or `--overwrite`); `--training-only` marks points for training only.

`camera` names the photos' camera when it is not the one the classifier was trained on (e.g. `auv`): the set's sidecar then carries it as `domain`, and `substrata train` standardises those points with their camera's own feature statistics (as for `curate` pools). `gsd_mm` gives the photos' ground resolution for point classifiers (else they are scaled by their width).

Usage: `substrata sample-points [--n-points N] [--stratified] [--seed N] [--margin PX] [--tag NAME] [--training-only] [--overwrite]`

## Z-intercepts calculation

Find optimal box position, subdivide to grid, sample random points, and compute Z-intercepts. Optionally apply along-slope transform before processing.

Usage: `substrata intercepts [--input PLY] [--box-length M] [--box-width M] [--box-size M] [--search-radius M] [--slope] [--position [X,Y]] [--depth-range MIN MAX [--width-step M] [--depth-step M] [--seed N]] [--plan] [--tag NAME] [--training-only] [--overwrite]`

- `--tag NAME`: a new set next to the others, `<id>_<mode>_NAME_intercepts.csv` (and `<mode>_NAME_bbox.png`). An existing intercepts CSV with labels is never overwritten: the command stops and asks for a `--tag` (or `--overwrite`).
- `--training-only`: points for training only, not a survey (e.g. a box elsewhere in the plot): the sidecar gets `purpose: training_only`, so cover summaries can leave them out.
- `--depth-range MIN MAX`: instead of a box, a depth band (metres) across the plot's whole width: one intercept per cell of `--width-step` (across the slope, along the depth contours of a plane fitted to the band; default 1 m) by `--depth-step` (in depth; default 1 m), the cloud point nearest a random position in each cell (like an intercept, not a random cloud point, which would favour densely sampled parts); cells without cloud are left out. Saved as `<id>_depth[_TAG]_intercepts.csv`, the grid in the sidecar. No box is searched for, so it is quick on a big cloud (an AUV's 60-90 m band: seconds once the cloud is loaded): e.g. training data at depths the best box leaves out. With `--plan` it only reports the cells (with cloud, empty), the band's width and the cloud's depth range. `--seed` gives the same points again.
- `--plan`: only find the box (or take `--position`) and print it as one `INTERCEPTS_PLAN {json}` line: the box in the grid frame (x along its length, y across), the grid -> orig transform, the cloud's median height in the box, its corners in orig coords, the cloud's extent, and the cloud's point counts on the box search's 0.1 m grid (with the box's own count, to compare a moved box). Nothing is written. substrata-explore's intercepts tab uses it to preview and move the box before generating with `--position`.

```bash
# Using default behavior (top-down intercepts)
substrata intercepts

# With custom box dimensions
substrata intercepts --box-length 30.0 --box-width 5.0

# With custom grid cell size
substrata intercepts --box-size 0.25

# Apply along-slope transform
substrata intercepts --slope

# Custom search radius
substrata intercepts --search-radius 0.01
```

Writes `<id>_{topdown,slope}_bbox.png` in the project folder. The intercepts CSV is an annotation set, so it goes to `<id>.annos/<id>_{topdown,slope}_intercepts.csv`, with its sidecar YAML `<id>_{topdown,slope}_intercepts.csv.yaml` next to it. The sidecar records how the intercepts were generated, so `intercepts-plot` (and other tools) can replay the exact grid:

```yaml
type: intercepts
plot_id: cur_sna_20m_20200303
world_transform: [...]        # the project's world_transform at generation
intercepts:
  mode: slope                 # slope | topdown
  grid_transform: [...]       # slope alignment (world -> grid frame); slope only
  grid_bbox: [[x0, y0], [x1, y1]]   # in the grid frame
  cell_size: 0.2
  box_length: 25.0
  box_width: 4.0
  search_radius: 0.005
  position: auto              # or the --position [x, y]
  slope_elevation_deg: 23.4   # slope only
  bbox_orig: [...]            # the 4 box corners in orig coords (for display)
```

The grid frame is `grid_transform @ world_transform @ orig`. The CSV's `orig_*` columns are the reference; its `world_*` columns are in the grid frame. The slope normal is not stored: `substrata.slope_normal_from_transform(grid_transform)` gives it in world coords, or pass `grid_transform @ world_transform` to get it in orig coords.

**Annotation sidecars in general:** any annotations CSV may have an optional `<csv filename>.yaml` (the `.csv` is kept in the name, so a `<id>.csv` never clashes with the project's `<id>.yaml`). It holds the settings of that annotation set rather than the plot: `type`, `plot_id`, `classes`, `world_transform`, `last_highest_id` (the id high-water mark, so deleted ids are never reused) and, when not the default, `cropped_path` (default `<csv stem>.cropped`), `crop_size` and `classifier`. Paths are relative to the sidecar's folder. For the project's `annotations` file, `ProjectInitializer` takes these from the sidecar in preference to the legacy keys in `<id>.yaml`. A CSV without a sidecar works as before. Helpers: `substrata.read_annotation_sidecar` / `write_annotation_sidecar`.

## Points that photos aim at (cams2annos)

Targeted photos, e.g. close-ups (macro photos) of single colonies, each aim at one thing: the point where the ray through the photo's centre meets the cloud. `substrata cams2annos`, run in a plot's folder, finds that point for every camera of a camera group (`--group`, default `closeup`: the photos in `<id>.closeup/`; any group `predict --camera-group` takes) and saves them as an annotation set, `<id>.annos/<id>_<group>[_TAG]_targets.csv`, with each photo as its point's view (`cam_filepath`, `cam_x`, `cam_y`): one point per colony, to label from its close-up (e.g. to find all the Acroporas among the close-ups). The cameras need a pose and a calibration (aligned in Metashape, or for close-ups: `camsync --match-images`) and the plot a world transform (distances in metres). Close-ups posed by copying a GoPro frame's pose by time alone can aim metres away from what they show. A camera whose pose camsync estimated rather than matched (`pose_status` `estimated` in the meta JSON) gets its point all the same, marked `is_extrapolated` True (as `intercepts` marks a point it could only extrapolate), to check by hand.

The ray (lens distortion undone) is followed from the camera in steps of `--search-radius` (default 3 mm) up to `--max-distance` (default 5 m); the first cloud point within that radius of the ray is the hit. A ray through a gap in the cloud (between branches, an edge) is tried again 3x wider, up to `--max-search-radius` (default 3 cm). The hit is refined to the point of that surface nearest the ray, and the view's pixel is that point projected into the photo (on sa3_40m's close-ups typically 10-60 px from the centre of an 8726 px photo), so the point and its view agree. Cameras whose ray meets no cloud, or that cannot project, are listed and left out. `--pixel X Y` follows another pixel than the centre.

The CSV's extra columns: `search_radius` (the radius that hit), `target_distance` (camera to point, m) and `target_offset_px` (the view's pixel from the one followed). The sidecar has `type: camera_targets`, `purpose: training_only` (photos aimed at chosen things are not a random sample, so cover summaries leave them out) and the settings and missed cameras in `camera_targets`. A set with labels is never overwritten (`--tag` for another, or `--overwrite`).

In Python: `pcd.camera_targets(cams, pixel=None, search_radius=0.003, max_search_radius=0.03, max_distance=5, id_prefix=None)` returns the Annotations (each an `InterceptAnnotation` with its camera as `image_match`), with `missed` the cameras left out.

Usage: `substrata cams2annos [--group GROUP] [--pixel X Y] [--search-radius M] [--max-search-radius M] [--max-distance M] [--tag NAME] [--overwrite] [--input PLY]` (alias `cams-to-annos`)

## Intercepts plot

Re-plot a saved intercepts CSV as a 2D grid of cells colored by the majority annotation label per cell — no bounding boxes needed. The orientation and exact generation grid are read from the CSV's sidecar (`<csv>.yaml`, written by `intercepts`) automatically, so the grid is axis-aligned and the generation cells (`grid_bbox`) are replayed exactly. `--yaml` overrides this, e.g. for an older `<id>_slope_intercepts.yaml` (which holds the combined `world_transform` and, if recent enough, `grid_bbox` at the top level). Without either, the project's `world_transform` is used (fine for top-down intercepts) and the grid is derived from the annotation extent at `--grid-size`.

When the project point cloud can be found, it is loaded and oriented with the same transform as the annotations and scattered as a background behind the grid cells by default; pass `--hide-points` to disable that (use `--points`/`-n` to cap how many points are loaded).

Each run writes three artifacts sharing the same width (1800 px): the static grid PNG (`<stem>_grid.png`, or `--output`), an animated GIF of the grid filling in cell by cell (`<stem>_grid.gif`), and a positions plot of the annotation markers over the point-cloud ortho map (`<stem>_positions.png`, rendered as `annotations.show(pcd, color=True)`). The positions plot is skipped with a warning when no point cloud is available.

**Manual label colors (`--label-colors`):** supply a plain text file with one `label #hexcolor` per line (the label is everything before the last whitespace token, so labels may contain spaces) to override the automatic `tab20` palette. Any label not listed collapses into a single `OTHER` category; add an `OTHER #hex` row to set its color (default `#999999`). The colors apply consistently to the grid cells/legend, the animation, and the positions markers. Example file:

```
Coral  #e6194b
Sand   #f0e442
Algae  #3cb44b
OTHER  #999999
```

**Older intercepts files without a saved `grid_bbox`:** because each intercept is sampled at a random position *within* its generation cell, deriving a fresh grid from the annotation extent phase-misaligns with the generation grid and produces spurious empty ("No data") cells (and doubled-up cells). Two recovery options:

- `--fit-grid` (recommended, point-cloud independent): recovers the generation lattice directly from the intercepts by scanning the sub-cell origin offset for the best one-point-per-cell alignment (`measurements.get_bboxes_from_intercepts`). Robust and needs no point cloud. A small residual of empty cells remains — the irreducible effect of the intercept search radius nudging points across cell boundaries.
- `--box-length`/`--box-width` (matching the values used by `intercepts`): reconstruct the exact generation grid by re-running the same deterministic optimal-box search on the point cloud and subdividing at `--grid-size`. This only reproduces the generation grid if it sees a bit-identical point cloud, so omit `--points` (decimation shifts the recovered box); `--fit-grid` is usually the safer choice. Use `--position [x,y]` to skip the box search when the box position was set manually, and `--step-size` to match a non-default search resolution.

The animated GIF fills its cells column-by-column by default; `--animation-order` picks a different reveal order — `rows` (top-to-bottom raster), `scan` (columns with a moving scan line), `random` (cells pop in in random order), `spiral` (expand outward from the centre), or `categories` (reveal one label group at a time, most dominant first, ~1s per category — label grids only).

Usage: `substrata intercepts-plot [--intercepts CSV] [--yaml YAML] [--grid-size M] [--fit-grid] [--box-length M --box-width M] [--position [x,y]] [--step-size M] [--output PNG] [--label-colors FILE] [--animation-order ORDER] [--title STR] [--hide-points] [--points N]`

```bash
# Plot slope intercepts (orientation + grid from the CSV's sidecar YAML)
substrata intercepts-plot --intercepts cur_sna_20m_20230414.annos/cur_sna_20m_20230414_slope_intercepts.csv --grid-size 0.2

# Older slope intercepts: pass the generation YAML explicitly
substrata intercepts-plot --intercepts cur_sna_20m_20230414.annos/cur_sna_20m_20230414_slope_intercepts.csv \
                          --yaml cur_sna_20m_20230414_slope_intercepts.yaml --grid-size 0.2

# Coarser grid (majority vote per larger cell)
substrata intercepts-plot --intercepts cur_sna_20m_20230414.annos/cur_sna_20m_20230414_slope_intercepts.csv \
                          --yaml cur_sna_20m_20230414_slope_intercepts.yaml --grid-size 1.0

# Recover the grid for an older file lacking grid_bbox (fit from intercepts)
substrata intercepts-plot --intercepts cur_sna_20m_20230414.annos/cur_sna_20m_20230414_slope_intercepts.csv \
                          --yaml cur_sna_20m_20230414_slope_intercepts.yaml \
                          --grid-size 0.2 --fit-grid

# Alternative: reconstruct the exact grid via the point-cloud box search
substrata intercepts-plot --intercepts cur_sna_20m_20230414.annos/cur_sna_20m_20230414_slope_intercepts.csv \
                          --yaml cur_sna_20m_20230414_slope_intercepts.yaml \
                          --grid-size 0.2 --box-length 25 --box-width 4

# Auto-detect the project's files from the current directory
substrata intercepts-plot

# Cap the background scatter at 2M points (scatter is on by default)
substrata intercepts-plot --points 2000000

# Disable the background point-cloud scatter
substrata intercepts-plot --hide-points

# Custom output path
substrata intercepts-plot --intercepts topdown_intercepts.csv --output /tmp/grid.png

# Manual label colors (unlisted labels collapse into OTHER)
substrata intercepts-plot --fit-grid --label-colors label_colors.txt

# Animate the GIF one label group at a time (dominant first, ~1s per category)
substrata intercepts-plot --fit-grid --animation-order categories
```

## Point cloud alignment

Register a source to a target and print the alignment transform. `--source` and `--target` each take a PLY file, a project YAML (detected by the `.yaml`/`.yml` extension) or a project directory (resolved like other subcommands: its `<dirname>.yaml` if present, otherwise filename conventions).

Usage: `substrata align --source PLY|YAML|DIR --target PLY|YAML|DIR [--cameras [--max-residual M] [--refine]] [--pdf PDF] [--no-scaling] [--voxel-size V] [--points N] [--output-yaml YAML] [--as-parent | --as-baseline]`

The transform is always fitted between the **raw** (untransformed) models. Two matrices are printed:

- **raw source → raw target**: the fitted registration.
- **raw source → oriented target**: the same transform with the target's `world_transform` applied on top. This is only printed when the target YAML has a non-identity `world_transform`. It can be applied directly to source `orig_coords` (e.g. source annotations) to place them in the oriented target frame.

Alignment methods:

- **Point clouds** (default): statistical outlier removal, then FPFH features with global RANSAC (rigid), then ICP refinement. ICP also estimates a uniform scale unless `--no-scaling` is given. The global stage is rigid and FPFH uses fixed radii, so it can fail when the two raw models differ a lot in scale. Check `ransac_fitness`/`fitness` in the printed metrics, or use `--cameras`.
- **Matching cameras** (`--cameras`, project YAMLs only): uses the centres of cameras whose image names match in both projects. Names are compared by filename stem, case-insensitively, regardless of folder or camera group. The command then fits a least-squares similarity transform to those centres (rigid with `--no-scaling`). It needs at least 3 non-collinear shared cameras. Names that occur more than once in a project, and cameras without a pose, are skipped. Outlier pairs are rejected iteratively: fit, keep the cameras within the threshold, refit, until the inlier set is stable. This matters when a whole AUV run is placed differently in the two models, which shows up as a block of cameras with a near-constant offset. The default threshold is `median + 5 × 1.4826 × MAD` of the residuals; `--max-residual` sets a fixed threshold instead, in metres if the target has a `scale_factor`, otherwise in target units. The command lists the rejected cameras with their residuals, plus the RMS, median and maximum residual of the inliers. Add `--refine` to refine the camera-based fit with point-cloud ICP. The source is first moved into the target frame with the camera fit, then the target is cropped to the overlap (so a small plot inside a large model works), and the default voxel size is 0.5% of the overlap extent. Camera centres and reconstructed surfaces don't always agree exactly between two models (e.g. a few cm vertically), so use `--refine` when the surfaces should coincide, e.g. for annotations on the point cloud.

`--pdf` writes a QC report. Everything in it is drawn in the oriented target frame (metres when the target is scaled and oriented). The target cloud is loaded without the `--points` cap, so the overlap is dense enough for meaningful distances. The pages are:

1. **Summary**: inputs, method, camera counts and RMS, scale, and both matrices.
2. **Camera residuals** (`--cameras` only): matched cameras on a top-down map coloured by residual under the final transform, with rejected cameras marked, plus a histogram.
3. **Location in target**: the source footprint on a top-down RGB ortho of the whole target (full and zoomed), and x–z / y–z profiles of the target through the footprint with the source overlaid. Use this to check that the source sits where you expect, at the expected depth.
4. **Correspondence in the overlap**:
   - **C2C**: the distance from each source point to the nearest target point.
   - **dz**: the vertical distance from each source point to a plane fitted to its nearest target points. dz isn't computed on near-vertical surfaces or over gaps in the target.
   - Top-down maps and histograms of both. The histograms report the target point spacing in the overlap, which is the lowest C2C you can expect.
   - Two thin cross-sections with both clouds overlaid.

The plots come from `visualizations.plot_pcd_context`, `plot_pcd_correspondence` and `plot_camera_correspondence`. These take any two point clouds already in the same frame, so you can also use them outside `align`.

`--output-yaml` writes a copy of the source project YAML whose `world_transform` is the raw source → oriented target transform. This replaces the source's previous `world_transform`, so loading that YAML puts the source in the target's oriented frame.

`--as-parent` links the source to the target as its **parent** plot instead, and leaves the source `world_transform` (its local scale and orientation) untouched. It writes two keys to the source project YAML, or to a copy at `--output-yaml`. The rest of the file, including comments and relative paths, is kept as is.

- `parent_id`: the target's id.
- `parent_transform`: the fitted raw source → raw target transform, scale included. It maps the source's orig coords to the parent's orig coords, the frame the parent's cameras use. So points move between the two models exactly, e.g. annotations to reproject in the parent's images (see `substrata transform --to-parent`).

`--as-baseline` is for later **timepoints** of the same plot. The target is the baseline timepoint whose orientation and scale the source inherits, because part of that orientation is arbitrary and can't simply be redone. It writes the link and the inherited frame to the source project YAML (or a copy at `--output-yaml`), keeping the rest of the file as is:

- `baseline_id`: the target's id.
- `baseline_transform`: the fitted raw source → raw target transform, scale included. It never changes after fitting.
- `world_transform` = baseline `world_transform @ baseline_transform`, and the `scale_factor` and depth model (`up_vector`, `depth_offset`, `depth_per_unit`) carried over from the baseline. These are a **cache**, so the YAML still loads on its own. When the baseline is re-oriented, `substrata baseline` reports the cache as stale and `--update` refreshes it (see [Baseline timepoints](#baseline-timepoints-baseline)).

Unlike plain `--output-yaml`, which writes the same `world_transform` but no link, this keeps track of where the frame came from.

The command also prints the scale ratio implied between the two world frames, `parent.world_transform @ parent_transform @ inv(world_transform)`. A value of 1 means both plots' scale factors agree. In Python, `ProjectInitializer.to_parent_world_transform(parent)` gives that transform, `parent_scale_ratio(parent)` gives the ratio, and `find_parent_project()` locates the parent's folder by `parent_id`. It searches sibling and ancestor folders, plus any extra folders you pass.

```bash
# Align two point clouds
substrata align --source source.ply --target target.ply

# Limit points for faster processing
substrata align --source source.ply --target target.ply --points 5000000

# Focal plot onto the oriented AUV model, using AUV images included in both models
# (project directories work as well as YAMLs)
substrata align --source focal/ --target auv/auv.yaml --cameras \
    --output-yaml focal/focal_aligned.yaml

# Fixed 5 cm outlier threshold instead of the automatic one
substrata align --source focal/focal.yaml --target auv/auv.yaml --cameras --max-residual 0.05

# Same, refined with point-cloud ICP, with a QC report
substrata align --source focal/focal.yaml --target auv/auv.yaml --cameras --refine \
    --pdf focal/focal_align_qc.pdf

# Keep the focal plot's own world_transform; store the AUV model as its parent
substrata align --source focal/focal.yaml --target auv/auv.yaml --cameras --refine \
    --as-parent

# A later timepoint inherits the frame of the baseline timepoint
substrata align --source cur_sna_20m_20240305/ --target cur_sna_20m_20200303/ \
    --cameras --refine --as-baseline
```

### Which transform key means what

| keys | relation | `world_transform` of this plot |
| --- | --- | --- |
| `world_transform` only | own frame (scalebars + orientation) | its own |
| `parent_id` + `parent_transform` | placed inside a larger plot (e.g. AUV); orig → parent orig | its own (local scale kept) |
| `baseline_id` + `baseline_transform` | later timepoint of the same plot; orig → baseline orig | inherited: baseline `world_transform @ baseline_transform` (cached) |

A later timepoint needs no `parent_*` keys of its own: its placement in the parent is the baseline's `parent_transform @ baseline_transform`.

## Baseline timepoints (baseline)

Check that the `world_transform` of later timepoints still follows their baseline timepoint (`baseline_id`/`baseline_transform`, written by `align --as-baseline`), and refresh it when the baseline was re-oriented.

Usage: `substrata baseline [PROJECT ...] [--update] [--link BASELINE]`

`PROJECT` is a project YAML or directory, or a folder of them, searched two levels down (e.g. a site folder `<site>/<plot>/<timepoint>/`). The default is the current directory. Baselines are looked up among those projects first, then like `find_project` (sibling and ancestor folders).

- **Default (dry run)**: for each project with a `baseline_id`, prints `fresh`, `stale` (with how much the frame moved: rotation, origin shift, scale), or why it can't be checked. A stale cache is detected without any extra key: the baseline frame it was built from is `world_transform @ inv(baseline_transform)`. Chains (tp3 → tp2 → tp1) are followed, so tp3 is reported stale when tp1 changed.
- **`--update`**: rewrites the stale `world_transform`, `scale_factor` and depth model in place, baselines before their dependants. Only the YAMLs found under `PROJECT` are written. The `world_*` columns of those plots' annotation CSVs (and their sidecar `world_transform`) still follow the old frame. `orig_*` are unaffected.
- **`--link BASELINE`**: adds the link to YAMLs whose `world_transform` was built from the baseline before links existed (e.g. by `align --output-yaml`): `baseline_transform = inv(baseline world_transform) @ world_transform`. This is only right if the baseline hasn't been re-oriented since. `BASELINE` is a project directory, YAML or id. Only the projects named are linked, never a whole folder.

Re-orienting a later timepoint with `substrata orient` gives it its own frame; it prints a warning, since `baseline --update` would restore the baseline's. `substrata head` shows `baseline: <id> (fresh|stale|not found)` for a project with a `baseline_id`.

```bash
# Check all timepoints of a site after re-orienting a baseline
substrata baseline /data/cur_sna

# Refresh the stale ones
substrata baseline /data/cur_sna --update

# Link an existing timepoint that was aligned with --output-yaml
substrata baseline cur_sna_20m_20240305/ --link cur_sna_20m_20200303
```

## Image matches

Find image matches of annotations and output them to a PDF, in one of two layouts (`--layout`):

- **grid**: each annotation's first image match, cropped, several to a page. Optionally apply a transform to annotation coordinates before matching (`--transform`). Default: `<cwd_basename>_imagematches.pdf`.
- **pages**: one A4 page per annotation, to check what a close-up (macro) photo shows: top left the best overview photo zoomed on the point (`--zoom`, at least 0.5 m wide, wider to show 2.5x the close-up's frame), top right the close-up's full frame (a box marks the crop below), and across the bottom the close-up at 100% (1 photo pixel = 1 page dot at 300 dpi), centred on the point. The point is marked in each panel, with scale bars from the camera distance (world scale / fx) and a header with the plot, annotation id, label, depth, coordinates and the CSV columns of `--info` (default `label_conf label_source`). Without a close-up of a point, the top right shows the overview's full frame and the bottom the overview at 100%. Default: `<annotations stem>_pages.pdf`.

`--layout auto` (default) makes pages when the set's stored views (`cam_filepath`) are close-ups: a camera-targets set made by `substrata cams2annos` (each point stored with the photo aimed at it), or stored photos of a camera type other than the main one. Other sets get the grid. `--closeup GROUP` names the close-up cameras instead (a camera group, type or photo folder as `predict --camera-group` takes, e.g. `closeup`; implies pages): each point gets its stored view if that is a close-up, else the most relevant close-up camera that sees it. A stored close-up is used even when its camera is disabled, as camsync leaves the poses it estimated rather than matched (cams2annos marks their points `is_extrapolated`); its caption then says so, as the point may be off and the overview match is worth a check.

The overview is chosen as `substrata crops` chooses views (`--occlusion`, `--camera-type`): the most relevant view of the main camera type, among the cameras that are not close-ups, not flagged obstructed. `--labels` / `--ids` select annotations (both layouts).

Photos are shown with a display contrast (`--contrast`, default `stretch`; `substrata.imageadjust.auto_contrast`): `stretch` brightens with one linear stretch for all channels (colours kept; 0.5% clipped at each end), `stretch-rgb` stretches each channel (also removes a blue-green cast, which can shift true colours), `clahe` equalises local contrast, `none` leaves photos as they are. A photo's statistics come from what its top panel shows, so its 100% crop matches its full frame.

Usage: `substrata images [--layout auto|grid|pages] [--closeup GROUP] [--labels LABEL ...] [--ids ID ...] [--zoom M] [--contrast METHOD] [--info COLUMN ...] [--occlusion ...] [--camera-type TYPE] [--annotations CSV] [--input PLY] [--transform] [--size PX] [--pdf-output PDF]`

```bash
# Using default behavior (auto-detects from CWD)
substrata images

# With explicit annotations file
substrata images --annotations annotations.csv

# Apply transform to annotation coordinates (interactive; grid)
substrata images --transform

# Specify output PDF
substrata images --pdf-output imagematches.pdf

# One page per close-up labelled as Acropora (a cams2annos set: pages by default)
substrata images --annotations <id>.annos/<id>_closeup_targets.csv --labels CS_ACR

# Pages for any set, with the close-ups that see its points (others: overview only)
substrata images --annotations points.csv --closeup closeup --contrast clahe
```

In Python: `substrata.annotation_page(title, subtitle, overview, closeup)` renders a page from two `PageView`s (photo, pixel, metres per pixel) and `save_pages_pdf(pages, path)` writes them. `auto_contrast(img, method, reference=None)` adjusts any PIL image or array; new methods are added with `@register_contrast_method("name")` (a function of float RGB arrays in [0, 1]).

## Thumbnails

Write a downscaled copy of every camera photo, for annotation tools such as reefcat. Thumbnails belong to the model, so they go in `thumbnails_path` from the YAML, or `<id>.thumbnails/` in the model folder by default. They mirror the photo folders: `<thumbnails>/<photo folder>/<filename>`, e.g. `<id>.thumbnails/<id>.auv/IMG_0001.jpg`. Existing thumbnails are skipped, including old ones stored directly in the thumbnails folder.

Usage: `substrata thumbnails [--size W H] [--output DIR] [--overwrite] [--jobs N] [--photo-path-replace OLD NEW] [--local]`

```bash
# Default 1024x768 bounding box (aspect ratio is kept)
substrata thumbnails

# Regenerate all thumbnails at a different size
substrata thumbnails --size 1280 960 --overwrite
```

## Crops

Save a crop centred on each image match of every annotation (up to `--max-photos` per annotation, best matches first). Crops belong to an annotations file, not to the model: they go in the set's `cropped_path` (from its sidecar), or next to the annotations CSV in `<annotations csv stem>.cropped/` by default. With `--annotations`, the given CSV's own sidecar applies, so e.g. `--annotations <id>.annos/<id>_slope_intercepts.csv` crops into `<id>_slope_intercepts.cropped/`. The layout is `<crops>/<ann_id>/<photo folder>/<photo stem>_<x>_<y>.jpg`, e.g. `<id>_ann.cropped/<id>_17/<id>.auv/G0012345_1520_988.jpg`, so photos from `.photos`, `.auv` and `.closeup` folders never collide (see the layout in {doc}`projects`), and the pixel tells whether a crop still shows the point (after the point moves, or the plot is aligned anew, its view lands elsewhere). Older crops named after the photo only (`<photo folder>/<photo file>`, or `<crops>/<ann_id>/<photo file>`) are still found and skipped.

Matches whose crop would extend past the image edge are rejected, so every crop is centred on its annotation. Crops keep the photo's EXIF and are saved at JPEG quality 95.

**Which views.** Cameras are grouped into camera types by sensor: the same resolution (either orientation) and a focal length within 3% (`CAMERA_TYPE_FOCAL_TOL`), whatever their folder. Two identical survey cameras, or one AUV camera across missions, are one type; a macro close-up camera or a GoPro is another. The set's main camera type is the one that sees most points best (the type of each sampled point's most relevant view; `--camera-type TYPE` to choose, `all` for every camera alike as before). Per annotation, `--max-photos` (default 5) views of the main type are cropped: unobstructed ones by relevance first, skipping views whose crop would run past the photo edge (so the training view always has a crop). Views are tested for occlusion in relevance order only until enough unobstructed ones are found. The crops are written by a pool of threads, each photo decoded once for all of its crops, while the next points' views are chosen (about 10 points/s on a 61 Mpx photo set). Other camera types are not cropped: review tools (substrata-explore) show more views, 10 of the main type and up to 3 of each other type (e.g. close-ups for identification), and cut the ones without a crop from the photo on the fly. `substrata predict` still combines up to 10 views of any camera.

**Occlusion.** By default (`--occlusion auto`) the line of sight from each camera to the point is tested on the plot's stratified full-resolution PLY (`substrata stratify`; the same test substrata-explore uses): a view counts as obstructed when 3 or more subsample points lie on it (`OCCLUSION_BLOCKED_MIN`), and only the 40 most relevant views of a point are tested (`OCCLUSION_CHECK_VIEWS`). Nothing is dropped; obstructed views go last. Without a stratified PLY, `auto` falls back to the legacy test (`--occlusion cloud`): the YAML's point cloud is loaded and matches are dropped or flagged by reprojection error. `--occlusion none` (or `--no-ply`) skips the test. The obstruction flag is stored in the crop's EXIF.

**Stale crops.** Before cropping, a point's stored crops that no longer show it are removed: a crop named `<photo stem>_<x>_<y>.jpg` whose photo, projected anew, no longer puts the point at that pixel (the point moved, or the plot was aligned anew). Crops that still show it are kept, chosen or not; older crops without the pixel in their name are left alone. The count is reported as `stale`.

**Training views.** Each annotation's training view is its most relevant unobstructed view of the main camera type. It is written into the annotations CSV (`cam_filepath`, `cam_x`, `cam_y`, with `include_cam_in_training` = `True`) where `cam_filepath` is still empty; a view that is already set (e.g. the photo a manual annotation was made on) is kept. A CSV without `include_cam_in_training` gets the column, `True` for every row. The CSV is backed up first (`backup/`). Points whose tested views of the main type are all obstructed get no training view and are listed: they need a look. `--no-views` leaves the CSV alone.

Other commands that use image matches with a point cloud (`substrata images`, benthic-fraction measurements, annotated videos) keep the legacy point-cloud test.

Usage: `substrata crops [--size W H] [--max-photos N] [--output DIR] [--overwrite] [--occlusion auto|subsample|cloud|none] [--no-ply] [--camera-type TYPE|all] [--no-views] [--annotations CSV] [--jobs N] [--photo-path-replace OLD NEW] [--local]`

```bash
# Default 800x800 crops (or the YAML crop_size), up to 5 photos per annotation
substrata crops

# Larger crops for a specific annotations file, regenerating existing ones
substrata crops --annotations other_ann.csv --size 1000 --overwrite
```

## Training views after the lens model fix (reproject-views)

Since 0.2.0.dev1 substrata projects with Metashape's frame camera model exactly: P1 with x and P2 with y (they were
the other way round, OpenCV's order), plus K4, P3, P4 and the skew B2, checked against Metashape's own
`Calibration.project` and against the tie points of a Samoa project (`tests/test_lens_metashape.py`). A point's
pixel moves by a few px on most sensors, up to tens near the photo edges and hundreds on a badly calibrated one. A
view is also only in view up to the radius where the lens's distortion folds back: beyond it, points far outside the
photo's field landed inside the image (on sa3_20m's PGRAM02, points ~50° off the axis in the middle of the photo).

Views chosen anew (review, crops, predict) use the corrected model by themselves. The training views stored in
annotation CSVs (`cam_x`, `cam_y`) are moved once, in each plot folder:

```bash
substrata reproject-views --dry-run   # what would change, per CSV of <plot>.annos/
substrata reproject-views             # or name CSVs: substrata reproject-views <plot>.annos/x.csv
```

A row with a 3-D point and a training view on one of the plot's photos gets the corrected pixel when its stored
pixel is where the old model put the point (within `--tol`, 1 px); a view the corrected model does not see the point
in is cleared (`cam_filepath`, `cam_x`, `cam_y` emptied; `--keep-out-of-view` keeps them; a row left out of training
keeps its view), and its id listed in `backup/<csv>_cleared_views_<time>.txt` (its label may have been given on the
wrong crop). Other rows are left as they are and counted (a pixel placed elsewhere, a photo of another plot), as is
every other cell. The CSV is backed up first; substrata-explore's grouping features of the set (made at the old
pixels) go to `backup/` too. Running it again changes nothing. Then crop again (`substrata crops`, or
substrata-explore's crop job: the cleared points get a new training view, crops at the old pixels are pruned as
stale) and train again.

## Predict

Predict a label for every annotation with a trained classifier (`substrata train`). The views are chosen as `substrata crops` chooses them: of the main camera type (`--camera-type`; default: the type that sees most points best; `all`: every camera alike), tested for occlusion (`--occlusion auto|subsample|cloud|none`, as for crops; `--no-ply`: none), clear of the photo edge (views whose crop would run past it are skipped), up to `--max-photos` (default 10). Other camera types are not used. substrata-explore predicts (live, and "crop & predict") from the same views, so its predictions match these. Each view is classified, and the results are combined into one prediction per annotation:

- `weighted` (default): the mean of the class probabilities across crops, each crop weighted by its relevance (how close the camera is and how central the point is in the photo). Possibly obstructed crops are left out unless all crops are. A single overconfident crop cannot outvote several consistent ones.
- `first`: the prediction for the best-ranked crop only.

The combined prediction is written to the `pred_label`, `pred_conf`, `pred_n` (number of crops used), `pred_model`, `pred_agree` and `pred_top` columns of the annotations CSV (only those columns change). `pred_top` holds the five most likely classes of the combined prediction with their probabilities (`CST:0.620;CSE:0.200;...`), so the probability the classifier gives a point's own label can be read later (substrata-explore's QC screening; a class not among the five has at most the fifth's probability). `pred_agree` is the share of the crops used whose own prediction is the combined one: low when the views see different things, e.g. something in front of the point in some photos, so the point is worth a look through its views; the CSV is backed up to `backup/` first. Each crop's own prediction is also stored in its EXIF (with `pred_model`), unless `--no-write-crops` is given.

**Which classifier.** `pred_model` is the classifier's fingerprint, `<file stem>@<8 hex digits>` (e.g. `point_classifier@83d12e81`): the digits are a hash of the file's contents, label map included. A copied or moved classifier keeps its fingerprint; a retrained one gets a new one. So a stored prediction can be told apart from a stale one: substrata-explore marks predictions of an older version of a loaded classifier as stale and leaves them out of its prediction filters. `substrata predict` skips the annotations whose `pred_model` is already this classifier's (re-running after retraining redoes only the stale ones, and an interrupted run continues); `--redo` predicts them all. Predictions made before `pred_model` existed are treated as unknown.

The classifier input is cut from the centre of the stored crop when that crop is at least `--size` (default: the crop size the classifier was trained on, as recorded by `substrata train`; 224x224 for older classifiers), giving the same pixels as cropping the photo but much faster, and without needing the photos. Otherwise it is cut from the photo.

A point-feature classifier (`point_classifier.pt`) is also accepted: it reads each match's photo around the matched pixel itself (stored crops and `--size` are not used for classifying), at the match's ground resolution when the camera geometry is known (plain photos fall back to the classifier's fixed scale).

Do not run this while the same annotations file is open in reefcat, as both write the CSV.

Usage: `substrata predict [--classifier PKL|PT] [--size W H] [--max-photos N] [--method weighted|first] [--crops DIR] [--no-write-crops] [--redo] [--batch-size N] [--no-ply] [--occlusion auto|subsample|cloud|none] [--camera-type TYPE] [--annotations CSV] [--jobs N] [--photo-path-replace OLD NEW] [--local]`

```bash
# Classifier from the YAML; crops from `substrata crops`
substrata predict

# Explicit classifier, using only the best crop per annotation
substrata predict --classifier crop_classifier.pkl --method first
```

### From one camera group (`--camera-group`)

`--camera-group GROUP` predicts the set's points from the photos of **one camera group only, including cameras that are disabled in the project**. For example, it can use the AUV photos aligned into a diver plot, which the plot's cams.xml marks disabled so that they stay out of its point cloud. The results go to a **separate CSV**; the set's own CSV (labels, training views, its stored predictions) is not changed. This allows validation across camera systems, such as classifying the diver-labelled points of a plot from its AUV photos only.

Choosing the group (`--camera-group list` prints the plot's groups, with how many cameras each has and how many are enabled):

- **camera type**, as printed by crops and predict: one resolution, with the focal length within 3% (e.g. `'2848x2816 f3153'`);
- **camera system**: a sensor resolution (`2848x2816`), i.e. every type of that resolution, whatever its calibrated focal length (substrata-explore chooses cameras this way across plots);
- **camera group**, named by the photo folder and file-name prefix (`auv_PR`, `photos_PGRAM01`), or a folder's groups together (`auv`, `photos`);
- `enabled`: the enabled cameras, the ones `substrata segment` uses;
- `all`: every camera.

`type:` / `group:` / `system:` say which is meant when a name is several.

How views are chosen and combined (`--view-mode`):

- `segment` (default): as `substrata segment` classifies its query points (the same code, `segmentation.classify_query_points`), so each point gets what a cover map made from that group's photos gives at it.
  - Up to `--views` views (default `SEG_POINT_VIEWS` = 3), the most relevant (closest, most central).
  - The reprojection occlusion test on the YAML's point cloud (`--no-ply`: none). The PLY is loaded, which takes a few minutes for a large plot.
  - No photo-edge filter.
  - A point classifier fuses the views' dense per-photo probability maps: the relevance-weighted mean (weight = relevance_min / relevance), then the most probable class.
  - A crop classifier uses the best view.
  - As in segment, a view that fails the occlusion test is dropped without being replaced by the next one, so a point can end up with no view even though cameras see it.
- `predict`: as `substrata predict` above (the same views and consensus).
  - Up to `--views` views (default `--max-photos`, 10).
  - The stratified occlusion test (`--occlusion`).
  - Only views whose crop fits in the photo.
  - The relevance-weighted mean, with obstructed views left out unless all are.

The output is `<set folder>/predictions/<set stem>__<group>__<mode>[__<tag>].csv` (`--tag NAME`, or `--output-csv PATH`). It has one row per point of the set, in the set's order, with these columns:

- `id`, `label`: the set's label, copied for comparison.
- `pred_label`, `pred_conf`: the fused probability of the predicted class (threshold this).
- `pred_n`: the views used.
- `pred_agree`: in segment mode, the weight share of the views whose own label is the fused one; in predict mode, the unweighted share.
- `pred_top`.
- `views_available`: the group's cameras that have the point in view (in front, inside the image), before the occlusion test, the edge filter and the limit of views.
- `view_filepath`, `view_x`, `view_y`: the highest-weight view used (in predict mode, the most relevant unobstructed one).
- `camera_group`, `view_mode`, `pred_model`.

A point with no view has an empty `pred_label`, `pred_n` 0, and `views_available` showing whether any camera of the group saw it at all. The `predictions/` folder is not an annotation set (substrata-explore lists `.annos/*.csv` only).

`<output>.json` records the run: the set, the classifier and its fingerprint, the group, its kind, its number of cameras and how many of them are disabled, the mode, views, occlusion test, edge filter, crop size, substrata's version, the time, and the counts (`points`, `with_view_available`, `predicted`, `no_view`).

Usage: `substrata predict --camera-group GROUP [--view-mode segment|predict] [--views N] [--classifier PT|PKL] [--tag NAME] [--output-csv PATH] [--annotations CSV] [--no-ply] [--occlusion ...] [--batch-size N] [--jobs N]`

```bash
# The diver-labelled points of a plot from its AUV photos only, as the AUV cover map would classify them
cd sam_sa3_40m_20250928
substrata predict --annotations sam_sa3_40m_20250928.annos/sam_sa3_40m_20250928_slope_intercepts.csv \
    --camera-group auv --classifier ../../classifiers/sav03/point_classifier.pt --tag sav03
# -> sam_sa3_40m_20250928.annos/predictions/sam_sa3_40m_20250928_slope_intercepts__auv__segment__sav03.csv

# The same points from the diver photos, as substrata predict would (for comparison)
substrata predict --annotations ... --camera-group photos --view-mode predict --tag sav03

# Which groups the plot has
substrata predict --camera-group list
```

### Image-only mode (plain photos)

With `--image-dir` or `--points`, `predict` works on plain photos instead of a 3-D project (no point cloud, cameras or YAML; `--classifier` is required). Each point is classified from a crop centred on it, sized as for `predict` above.

- `--image-dir DIR`: random points are sampled on every image in `DIR` (searched recursively): `--n-points` per image (default 25), simple random or `--stratified` (one point per grid cell), kept at least half a crop from the image edges, reproducible with `--seed`. They are saved to `<output>/<DIR name>_points.csv` (`--output`, default CWD) and **reused** on later runs, so different classifiers are compared on the same points. Images added to the folder later get their own new points. `--resample` draws new points (the old file is backed up).
- `--points CSV`: predict at given points instead: a points CSV from an earlier run (updated in place), or a CoralNet export (`Name`, `Row`, `Column`, `Label code`). For a CoralNet export the images are searched in `--image-dir` (default: the CSV's folder), the result is written to `<output>/<CSV name>_points.csv`, and the predictions are compared with the CoralNet labels (agreement and a per-class report). Labels the classifier does not predict are reported and left out of the comparison.

With a point classifier (`.pt`) each point is read in a window resampled, in order of precedence: (1) to the classifier's resolution from the photos' ground resolution, `--gsd-mm MM` (classifiers trained in mm/px); (2) for photos of another camera (width more than 25% off the training photos' median), so that the window covers the same share of the photo width as on the training photos; (3) at the classifier's fixed scale. `substrata curate` pools, `substrata train` (for pool points) and `substrata masks` use the same rule, so predictions on a new camera match how its pool points were reviewed and trained. `--domain NAME` standardises the features with a camera domain's statistics (a `curate` pool whose labelled points went into training) instead of the training photos'.

The points CSV has the image (`cam_filepath`) and pixel position (`cam_x`, `cam_y`) of each point plus `pred_label`/`pred_conf` (the 3-D coordinate columns are `nan`); an existing file is backed up to `backup/` first. A `<points>_cover.csv` next to it gives, per image, the number of points and the percent cover of each predicted label, plus an `ALL` row over all points.

Usage: `substrata predict (--image-dir DIR | --points CSV) --classifier PKL|PT [--n-points N] [--stratified] [--seed N] [--resample] [--output DIR] [--gsd-mm MM] [--domain NAME] [--size W H] [--batch-size N] [--jobs N]`

```bash
# 25 stratified random points per photo; reused on later runs
substrata predict --image-dir photos --classifier crop_classifier.pkl --stratified

# Check a model against CoralNet's own annotations (images in frames/)
substrata predict --points annotations.csv --image-dir frames --classifier crop_classifier.pkl

# AUV frames with a diver-trained point classifier: their ground resolution, and the
# statistics of the AUV pool whose labelled points were trained on
substrata predict --image-dir auv_frames --classifier point_classifier.pt --gsd-mm 1.21 --domain auv
```

## Match annotations

Match random point annotations (e.g. a point-intercept CSV) to manual annotations of the same object (e.g. colony annotations in `<id>.annos/<id>_ann.csv`). Without `--manual`, the manual set is the `--random` sidecar's `matched_to`, else the project YAML's `annotations:`. Only random points whose label starts with `--label-prefix` (default `CS`, hard corals; comma-separated list allowed, `''` for all) are considered, and only cameras are loaded (not the point cloud).

For each random point, the nearest manual annotations within `--max-dist` (default 1 m, up to `--max-candidates`, default 5) are candidates. SAM2 runs two tests per candidate:

- fwd: SAM2 segments the candidate in the image it was annotated on, and the test checks whether the random point falls inside.
- rev: SAM2 segments the random point in its own image, and the test checks which candidates fall inside.

If a point is outside that photo, the most relevant camera that sees both points is used. SAM2's top-scoring mask is used, plus its 2nd mask as a fallback (e.g. the whole colony when the top mask covers one plate). A 2nd mask only counts together with a top-mask agreement in the other direction. SAM2 and a checkpoint are required (default `~/Github/sam2/checkpoints/sam2.1_hiera_large.pt`, see `settings.SAM2_CHECKPOINT`).

Writes a copy of the random CSV (`<random>_matched.csv`) with two added columns:

- `manual_id`: the id of the matched manual annotation.
- `manual_id_conf`, one of:
  - `high`: one candidate agrees both ways with top masks.
  - `medium`: one candidate agrees both ways using a 2nd mask, or agrees one way only.
  - `low`: several candidates agree (the nearest is chosen), or none agrees but the nearest is within `--fallback-dist` (5 cm).
  - empty: unmatched.

A candidate inside the random point's mask whose own mask excludes the random point is ignored, because it is typically hidden behind the colony.

Only the link is stored, not copies of the manual annotation, so a label corrected later in the manual CSV is never out of date in the matched CSV. Which manual CSV the `manual_id`s refer to is recorded in the matched CSV's sidecar (`<random>_matched.csv.yaml`) as `matched_to`, stored relative to the sidecar and repaired by `substrata path-repair`. The sidecar also carries over the random CSV's own sidecar (e.g. the intercepts settings).

To get the manual labels next to each random point, run `--export-labels OUT.csv` on the matched CSV. This needs no cameras or SAM2. It writes the matched CSV plus the manual set's current `manual_label`, `manual_label_conf` and a `manual_status` column:

- `ok`
- `no match`
- `missing`: the `manual_id` no longer exists in the manual set. Ids are never reused, so a deleted manual annotation can't silently point at another colony.

This export is a derived snapshot: regenerate it after editing either set. In Python, `join_manual_annotations(matched_csv)` returns the same rows.

Matched CSVs from before this change are migrated when passed back as `--random`: the copied `manual_label`/`manual_label_conf` columns are dropped and `match_conf` is renamed to `manual_id_conf`.

A review PDF (`<random>_matches.pdf`) has one page per flagged point: every point that is not `high`, including unmatched points that had candidates. `--review-all` adds the `high` matches. Pages are ordered low, medium, unmatched, high. Each page shows:

- The random point's image, with its mask in red.
- The assigned (or nearest) candidate's image, with its mask.
- A conflicting candidate, or else an overview that sees the most candidates.

Candidates are labelled by their id suffix, with one colour per candidate. Each page also gives the approximate surface area of each colony. To correct a match, edit `manual_id` (and `manual_id_conf`) directly in the matched CSV.

The matched CSV can be passed back as `--random`. Its `matched_to` then becomes the default `--manual`. Rows that already have a `manual_id` or a `manual_id_conf` are kept as they are, and only the remaining rows are matched. This covers matches from a previous run and matches you added by hand (e.g. with `manual_id_conf` set to `manual`). Use `--redo-all` to re-match every row instead. Rows outside `--label-prefix` are never changed.

Usage: `substrata match-annotations --random CSV [--manual CSV] [--label-prefix CS] [--max-dist M] [--max-candidates N] [--fallback-dist M] [--sam2-checkpoint PT] [--output CSV] [--pdf-output PDF] [--review-all] [--redo-all] [--no-pdf] [--photo-path-replace OLD NEW] [--local]`, or `substrata match-annotations --random MATCHED_CSV --export-labels OUT_CSV [--manual CSV]`

```bash
# Match random points to the project's manual annotations (auto-detects from CWD)
substrata match-annotations --random ton_ko1_40m_20240930.annos/ton_ko1_40m_20240930_slope_intercepts.csv

# Explicit manual annotations, and include every match in the review PDF
substrata match-annotations --random points.csv --manual colonies_ann.csv --review-all

# Matched points with the manual set's current labels, for analysis
substrata match-annotations --random points_matched.csv --export-labels points_labels.csv

# Match hard and soft corals
substrata match-annotations --random points.csv --label-prefix CS,SC

# Re-run on a (hand-edited) matched CSV: only rows without a match are assessed
substrata match-annotations --random points_matched.csv --output points_matched_v2.csv
```

## Camera time-sync (camsync)

Copy camera centers/transforms from a *pose-source* sensor (e.g. a GoPro) to an *updated-target* sensor (e.g. a macro camera) using EXIF time matching. Per-camera poses are written to the cameras meta JSON (backed up first). The `.cams.xml` keeps its sensor calibration; only its cameras' `enabled` flags are updated to match the meta JSON (backed up first). The relative timing of the two sensors can be given manually (`--time-offset` / `--pose-time`) or estimated automatically (`--auto-time` / `--auto-xyz` / `--auto-offsets`); auto modes print a detailed report and prompt before saving unless you pass `--yes`.

Each run writes a sanity-check PDF (`<id>_camsync.pdf`): a summary page, followed by the first time-matched pose/target image pairs side by side (five per page). `--pdf-pairs N` sets how many pairs to render (default 9).

Usage: `substrata camsync [--pose-source ID] [--updated-target ID] [--pose-date DATES] [--target-date DATES] [--time-offset SEC | --pose-time TIMESTAMP] [--xyz X,Y,Z] [--auto-time] [--auto-xyz] [--auto-offsets] [--spatial-max-dist M] [--min-spatial-pairs N] [--pdf-pairs N] [--yes]`

```bash
# Interactive: lists sensors and prompts for source/target and timing
substrata camsync

# Sync macro (sensor 1) to GoPro (sensor 0) with a known time offset
substrata camsync --pose-source 0 --updated-target 1 --time-offset 12.5

# Anchor timing by matching the earliest target image to a pose-source timestamp
substrata camsync -s 0 -u 1 --pose-time "2020-03-03 14:21:07"

# Estimate both the time offset and the pose-local xyz offset automatically
substrata camsync -s 0 -u 1 --auto-offsets --yes

# Add a fixed offset (meters) in the pose-source camera frame when copying pose
substrata camsync -s 0 -u 1 --time-offset 12.5 --xyz 0,0.12,0

# Render 20 image pairs in the sanity PDF instead of the default 9
substrata camsync -s 0 -u 1 --time-offset 12.5 --pdf-pairs 20
```

### Close-ups posed from image matches (`--match-images`)

Copying a pose by time only works if the two cameras look at the same thing at the same moment. A diver's close-ups (macro photos of single colonies) and the GoPro video they film do not: the clocks differ by an unknown offset, and the close-up camera points elsewhere than the GoPro (on sa3_40m about 10° lower, 0.1-0.3 m in front of it), so a copied pose can be metres off. `--match-images` poses each close-up from the GoPro frames themselves (`substrata.imagesync`):

1. **Time offset**: every close-up / frame pair within the offsets that could apply (`--offset-window MIN MAX`, default every offset at which a close-up falls within the frames' time span) is matched coarsely (SIFT, the whole frame at 1600 px). Each close-up that matches votes with its scores divided by its own best (so a few strong matches, such as a colour chart, cannot outvote the rest); the votes peak at the clocks' offset. The highest peaks (at most 3, 20 s apart) are then **confirmed** with the learned matcher on up to 12 close-ups each: the votes' peak is taken when it stands out (3x the median vote, nothing 20 s away above 0.6 of it, 3 close-ups whose own best is within 8 s) and at least 3 (and a quarter) of the sampled close-ups match there; otherwise the candidate that most close-ups match is taken, if that many match and twice as many as at the next candidate. If none is confirmed the command stops and asks for `--time-offset` (which skips the search). This also gives where a close-up's centre lies in a frame, how wide it appears there and how it is turned (sa4's GoPro was mounted upside down: its close-ups match turned 180°).
2. **Matching**: each close-up against the frames within `--frames` s (default 3) of its offset time with a learned matcher (kornia's LoFTR, `--matcher loftr`; or DISK + LightGlue), first in a crop round that spot, then across the rest of the frame, then turned the other ways; a close-up that still matches nothing is also tried in the frames up to `--wide-frames` s off (default 8: the diver lingered at the colony). The learned matchers cope with the blur, the scale difference and the lighting far better than SIFT (sa3_40m: 78 of 83 close-ups matched, against 30), but not with a photo turned round, hence the turning.
3. **Pose**: each match's frame pixel is followed along the frame's ray to the cloud, so the close-up's pixels see known 3-D points; PnP (RANSAC, then refined, the close-up's lens distortion undone) gives its pose.
4. **Checks**: enough PnP inliers, a small reprojection error (0.7% of the photo's width), a distance from the camera to the point it aims at within 0.5-2x the other close-ups' median, and, unless `--no-check`, the other matcher's pose aiming within 2 cm of the same point across the view (along it, a branch tip and the base below it are the same colony; where it gives a pose: the PDF says which were checked). Close-ups that fail get no pose.

A close-up whose time has no aligned frame near it (none within `--wide-frames` s) is reported as `no frames`, with the frames at those times that Metashape left unaligned. Either align those in Metashape (Align Selected Cameras) and export again, or add `--pose-frames`:

- **`--pose-frames`** poses those frames here first (`substrata.framepose`), then matches the close-ups with them as with aligned ones. A frame is matched (LoFTR) with the posed photos that see where it should be looking (the survey photos and the aligned frames; `--frame-photos SENSOR ...` picks the sensors, by default every sensor but `-s` and `-u` that shows the reef in at least 60% of the best one's detail, px per m at the range its photos were taken from: photos from much further away, such as an AUV's, mostly match nothing), ranked by how alike their view is; the matched pixels are followed to the cloud and the frame posed by PnP, pooled over up to 4 photos that agree. Gaps are filled from their edges in, 1 s at a time, each posed frame guiding the next; a frame is only used, and only guides the next, when photos with a pose of their own (not other posed frames) agree on it, so an error does not carry along the chain. A frame with no posed frame within 10 s is first placed by image retrieval (`substrata.frameretrieval`: the 20 survey photos most like it by DINOv2 descriptors, checked with LoFTR). A close-up that still matches nothing is tried once more after the unaligned frames among its own (within `--frames` s) are posed too: e.g. 1-2 s either side of its time, while the aligned frames 3 s off show the reef next to it. The frames' poses are only used for this run; they are not saved. sa3_40m, aligned frames hidden and posed again: 2-3 mm from Metashape's (median). sa4_60m: 101 of 129 unaligned frames posed, close-ups posed 45 of 46 (26 without), ~26 min.
- **`--frame-calibration OTHER.cams.xml`** gives the frames a lens where Metashape aligned none of them and so calibrated none (sa3_60m): the same sensor id's calibration in another plot shot with the same camera and settings (GoPro: 2-4 px apart between plots, against ~130 px of distortion). The camera may have been mounted the other way up there (sa4's GoPro is upside down relative to sa3's): `--frame-calibration-orientation auto` (default) poses a few frames, spread over the run, and keeps the lens as given or turned 180 deg, whichever fits their matches better (lower reprojection error for two thirds of them; otherwise kept as given, with a note); `same` or `turned` sets it. With `--pose-frames`, close-ups can then be posed in a plot where no frame was aligned.

Where a close-up aims and what it sees on the reef are accurate to millimetres. Its exact position is less certain: over a flat patch through a narrow lens, tilting the camera and shifting it sideways look almost the same, so the centre can be a few cm off (with a tilt that makes up for it).

A close-up without a good match still gets a pose, marked (`--no-estimate`: none):

- **`estimated`**: no match (or one rejected for its range or error). Its pose is the pose of the frame at its time + offset (or 1 s off) combined with the rig: where the close-up camera sits and points relative to the GoPro mounted on it, the median over the posed close-ups and their frames. On sa3_40m and sa4_40m, estimating each posed close-up from the others this way, it aimed 4-6 cm from where its match puts it (median; 8-12 cm at p90), against 21-23 cm for the frame's pose copied as it is. A diver who moved off before or after taking the close-up makes it worse.
- **`unverified`**: the two matchers disagree on where it aims (more than 2 cm apart); the main matcher's (LoFTR's) pose is kept.

`<id>_camsync_images.pdf` shows the offset's scores and the rig, then the close-up next to its frame with its centre (red) and where its pose aims (green), for every close-up not posed outright, every posed one at an unusual range (beyond 1.5x or 1/1.5x the median) and a selection of the others (`--pdf-pairs`). `--dry-run` stops there. Otherwise, after a confirmation (or `--yes`), the meta JSON and cams.xml are backed up to `backup/`, the posed close-ups get their pose (enabled), the estimated and unverified ones theirs too but disabled, with `pose_status` `estimated` / `unverified` in the meta JSON (the posed: `matched`), the others a null pose (disabled), and `<id>_camsync_poses.json` is written for Metashape. `substrata cams2annos` then gives each close-up's target point; those from estimated poses are marked `is_extrapolated`, to check and correct by hand.

Everything substrata needs is then in the plot's files: the poses are in the meta JSON (substrata reads camera poses from there; cams.xml only gets the enabled flags), so Metashape is not needed. `metashape-import-poses` only puts the same poses into the .psx, so that a later export from Metashape (which rewrites the meta JSON from the project) keeps them.

A close-up sensor that Metashape never calibrated (none of its photos aligned) is used with its nominal lens: the `focal_length` and `pixel_width` in cams.xml, no distortion. Needs the plot's point cloud and world transform, and kornia (`pip install kornia`, or the `match` extra); the matchers' weights are downloaded on first use, and a GPU makes it fast (about 4-11 min a plot of 50-90 close-ups).

```bash
substrata camsync -s 2 -u 1 --match-images --dry-run   # look at <id>_camsync_images.pdf
substrata camsync -s 2 -u 1 --match-images             # then save
substrata metashape-import-poses                       # optional: the poses into the .psx
```

### Poses into Metashape (metashape-import-poses)

`substrata metashape-import-poses [--psx PSX] [--poses JSON] [--chunk CHUNK] [--metashape PATH] [--dry-run] [--posed-only]` gives the cameras in the Metashape project the poses in `<id>_camsync_poses.json` (by camera key, checked against the label), by running `metashape_scripts/import_camera_poses.py` under Metashape's Python, and saves the project. Cameras marked `estimated` or `unverified` get their pose too and are disabled, so they can be found among the disabled cameras and checked (`--posed-only`: left as they are). Cameras without a pose in the file are left as they are. These cameras are not aligned by Metashape (they have no tie points), so Metashape keeps the given pose.

## Video tracks (track-video)

Places videos filmed over a plot on it, and finds where they stood still. Made for a **GRAV** camera on a sampling
dive: a GoPro on a float at the top of a weighted pole, looking down on a diver who moves the pole to a coral, lets
it settle and samples round it. The output tells, per stop, which video, when, and where in the plot: an annotator
then only has to put each sample on the right coral (in substrata-explore, next to the video), not search the plot
for where it was filmed (`substrata.videotrack`).

Usage: `substrata track-video VIDEO_OR_FOLDER [...] [--plot PLOT ...] [--out DIR] [--work DIR] [--probe N] [--proxy] [--no-annotations] [--no-poses] [--redo]`

```bash
# three divers' GRAV videos on the 60 m and 40 m plots; H.264 copies for the browser
substrata track-video Actioncams/HYD2511_GRAV02GOPRO_* Actioncams/HYD2511_GRAV03GOPRO_* \
    --plot sam_sa3_60m/sam_sa3_60m_20250928 --plot sam_sa3_40m/sam_sa3_40m_20250928 --out sav03_grav --proxy
substrata-explore --videos sav03_grav     # tracks and stops in 3D, the video player
```

1. **Keyframes**: a GoPro's HEVC has a keyframe every 1.001 s; those are decoded on their own (~20x real time, bound
   by reading the file) at full size: one frame a second (other videos: sampled at 1 fps). Videos shorter than 5 s
   are left out.
2. **Stops**: the camera's motion between frames (the 30th percentile of the optical flow, so a diver moving in
   front of the still reef does not count); still (< 1 px at 480 px wide) for 10 s or more is a stop.
3. **Which plots**: 12 frames per video (still ones, spread over it) are placed by a seed search (below) on each plot; a video
   is tracked on the plots where any was placed (`--probe 0`: on every plot).
4. **Placing a frame**: LoFTR (kornia, half precision) between the frame (1280 px) and the plot's aligned survey
   photos (640 px; sensors at least 0.6 as wide as the widest: no AUV or video frames), turned so the plot's +y is up
   in each. A seed: the frame against a grid of photos 1.5 m apart (960 / 480 px, 4 quarter turns), then against
   the photos round the best; tracking: the 5 photos nearest the last position (its turn first). 80 or more matches
   consistent with one fundamental matrix place it; the frame centre is mapped into that photo (a homography of the
   matches nearest it) and followed along the photo's ray to the cloud. While the camera is still, the position is
   kept and checked every 8 s; lost, a seed is tried at still frames, at most every 6 s. The frame's **heading**
   (where the top of the frame points) and **footprint** (how much reef it spans) come from points a little above
   and right of the matches' middle, mapped the same way.
5. **Sequences and estimates**: the videos of one camera (one folder) over one plot are one *sequence* (a dive's
   samples there, in order, over a GoPro's split files and restarts; ordered by the creation times). Its stops are
   numbered in order over its videos (`seq_stop`). A stop with no frame placed gets a rough position: interpolated
   in time between the sequence's frames placed before and after it (at most 10 min apart), else the nearest placed
   frame within 60 s on one side (`placed` = `estimated`).
6. **Camera poses** (`substrata.videopose`; `--no-poses`: none): where the camera was at each stop and where it
   looked, so a 3D viewer can show the plot as the video does. The stop's first and last frame placed by matching are
   matched against the survey photos within 3 m (up to 30); each match's 3D point comes from that photo's depth map
   (the cloud projected into it). One lens per camera setting (model, lens mode and frame size, from a GoPro's
   metadata) is calibrated from all those frames together with their poses (a fisheye with the sides' stretch of a
   GoPro's SuperView; fitted in pixels from the middle of the frame out), then each frame's pose; a stop takes its
   frame with most inliers. On SAV03: 3 px median reprojection error, the two frames of a stop 4 cm / 1.3 deg apart,
   cameras 1.6 m above the stop points; ~5 s a stop frame the first time (kept in `--work`).
7. **Outputs** (`--out`, default `./videotrack`):
   - `stops.csv`: per stop: sequence, diver (named in substrata-explore's video player: `labels.json` in `--out`),
     plot, `seq_stop`, video, its stop number, start (m:ss.s) and end (s into the
     file), duration, `placed` (matched / estimated / empty), x, y, z (the plot's world frame), frames placed by
     matching and their spread (or `est_gap_s`: s to the nearest placed frame), `heading_deg` (degrees clockwise
     from the world's +y) and `width_m`, the camera pose (`cam_x`, `cam_y`, `cam_z`; `cam_qx` .. `cam_qw`: a
     quaternion, camera -> world, x right, y down, z forward; `pose_px`), the point id in the plot's stops set, the
     video file;
   - `track.csv` (every placed frame), `videos.csv`, `<plot>_map.jpg` (the plot from above: each sequence as a line
     through its numbered stops, estimated ones `~n`, the placed frames faint), `README.txt` (the columns),
     `videotrack.json` (for substrata-explore; with each camera setting's lens calibration and field of view);
   - `proxies/<video>.mp4` with `--proxy` (720p H.264, NVENC when available): browsers on Linux mostly do not
     play HEVC;
   - per plot, the stops with a position as the point set `<id>.annos/<id>_video_stops.csv` (with its sidecar; the
     `sequence`, `seq_stop`, `video`, `stop`, … columns kept; an estimated position has `is_extrapolated` True, as
     intercepts and camera targets mark theirs), to move onto the sampled corals and label.
     An existing set is only added to: stops never written to it become new points and the stop columns of its
     points are brought up to date (backed up first), but its points are never moved, relabelled, removed, or added
     again once you removed them (`--no-annotations`: none written).

Keyframes, photo thumbnails (1/8) and each video's track per plot are kept in `--work` (default
`~/.cache/substrata/videotrack`) and reused: running again (other outputs, `--proxy`) only redoes what is missing
(`--redo`: track again). Needs ffmpeg and kornia (the `match` extra).

On SAV03 (2025; three divers, 40 m and 60 m) whole-frame retrieval (DINOv2) did not find the survey photos a GRAV
frame sees (the frame covers ~4 m of reef, a survey photo ~1 m, and a diver fills part of it), but LoFTR with a survey
photo that sees the same reef gave 500-1,200 consistent matches across the strobe / ambient light gap (unrelated
pairs: below ~45). A tracked frame takes ~0.1-0.8 s, a seed ~5-10 s (RTX 4080). The position is the frame centre's,
near the pole's foot: within a metre or so of the sampled corals.

## Path repair (path-repair)

A project records absolute paths in two places, and moving it breaks both at once. `<id>.yaml` records the project's own files (`ply`, `cams_xml`, `cams_meta_json`, `markers`, `annotations`, `photos_path`, …), and `<id>.meta.json` records each camera's image path as captured from Metashape at export time. Annotation sets (the point CSVs in `<id>.annos/`) add two more: the path keys of their sidecar YAMLs (`<csv>.yaml`: `classes`, `cropped_path`, `classifier`, `matched_to`) and the `cam_filepath` column of the CSVs (the image each point was placed on). `path-repair` checks and repairs all of them.

Phase A repairs the project file paths in the YAML, and the path keys of each annotation sidecar (written relative to the sidecar, e.g. `classes.csv` or `../classes.csv`). Phase B repairs the per-camera image paths in the meta JSON. Phase C repairs the `cam_filepath` cells of the point CSVs in `<id>.annos/` (any CSV with `id` and `x`/`orig_x`, `y`/`orig_y`, `z`/`orig_z` columns); it reuses phase B's folder mappings, so a folder answered once fixes both, and writes a backup of each changed CSV to `backup/`. The `.cams.xml` is never modified, since it stores image *labels*, not paths. Use `--yaml-only` (phase A) or `--cams-only` (phases B and C) to run part of it, and `--no-annotations` to leave the annotation sets alone.

Each broken path is resolved by trying, in order: the conventional filename in the project folder, then a recursive search by basename, then asking you. The YAML's `path:` key is repaired first, because every relative value beneath it resolves against that key — one stale `path:` is usually the whole problem. Repaired YAML entries that sit inside the project folder are written as **bare filenames**, so the YAML stays portable and survives the next move.

**Missing photo folders** (e.g. a local copy that deliberately leaves out the AUV or video-frame images) are asked about once per folder, not once per image. When none of a folder's images can be found, you choose: `s` skips the folder, `all` skips it and every further missing folder, `p` gives the folder's new location, `i` asks per image as before, and `a` aborts. Skipped folders are listed with their image counts in the report, and phase C skips the same folders without asking again. Their paths are left as they are, so a later run on a machine that has the images can still repair them. A folder of which only some images are missing is still asked per image.

The YAML is patched line by line, preserving comments, key order and any keys substrata does not itself use. (The initializer's own `save_config_to_yaml` rebuilds the file from scratch and drops anything it does not model, so it is not used here.)

All phases are planned in full before any writes, so aborting at any point leaves the project untouched. `--dry-run` prints the plan and writes nothing. For camera images, `--find` with `--replace` does a literal substring substitution and refuses to write unless every resulting path exists; `--find`, `--cams-group` and `--sensor-id` restrict which cameras are considered and apply to phase B only. `--find`/`--replace` also apply to the annotation images; with `--cams-group` or `--sensor-id`, phase C is skipped.

**Use this instead of `--local`.** That flag rebases every path onto the working directory by basename — unverified, and discarding any nested folder structure — and on `orient`, `scalebars`, `colors` and `firefish --save-yaml` it is then persisted into the YAML as a side effect. `path-repair` changes only the paths that are actually broken, and verifies each result before writing.

Related: `substrata segment --photo-path-replace OLD NEW` applies the same camera substitution for a single run without saving it; `substrata path-repair --find OLD --replace NEW` makes it permanent.

Usage: `substrata path-repair [--yaml-only | --cams-only] [--no-annotations] [--find STR] [--replace STR] [--cams-group NAME] [--sensor-id ID] [--root DIR] [--dry-run] [--yes]`

```bash
# Inspect what would change, without writing anything
substrata path-repair --dry-run

# Repair a project that has been moved or copied to this machine
substrata path-repair

# Repair only the project YAML, or only the camera images
substrata path-repair --yaml-only
substrata path-repair --cams-only

# Leave the annotation sidecars and CSVs alone
substrata path-repair --no-annotations

# Search elsewhere for the photos (e.g. an external drive)
substrata path-repair --root /Volumes/ReefArchive

# Only repair cameras whose stored path mentions the old drive
substrata path-repair --find /Volumes/OldDrive

# Only repair one camera group, or one sensor
substrata path-repair --cams-group auv
substrata path-repair --sensor-id 1

# Literal find/replace on camera paths; refuses to write unless every result exists
substrata path-repair --find "D:\photos" --replace /data/reefs/photos
```

## Transform annotations

Load annotations from a CSV, apply one or more transforms (entered interactively) to each annotation's `orig_coords`, and save the result. With `--inverse` the cumulative transform is inverted before being applied.

Instead of prompting, `--to-parent PROJECT` applies that project's `parent_transform` (see `align --as-parent`). This moves the project's annotations into its parent's orig frame. `--from-parent PROJECT` applies the inverse, moving annotations from the parent into the project.

Usage: `substrata transform INPUT [--output CSV] [--no-header] [--ignore-header] [--inverse] [--to-parent PROJECT | --from-parent PROJECT]`

```bash
# Transform annotations; prompts for the transform matrix/matrices
substrata transform annotations.csv

# Write to an explicit output path
substrata transform annotations.csv --output annotations_world.csv

# Apply the inverse of the entered transform(s)
substrata transform annotations.csv --inverse

# Move a focal plot's annotations into its parent (AUV) model
substrata transform focal/focal_ann.csv --to-parent focal/ --output auv/focal_ann_in_auv.csv
```

## Classifier training

Train a classifier from labelled annotations. There are two kinds, chosen with
`--arch`:

- **Point-feature classifier** (default, `--arch dinov2` = `dinov2_vitb14_reg`;
  or `dinov3` = `dinov3_vitb16`; sizes `dinov2-s/-l/-g`, `dinov3-s/-l/-h`, or
  any torch.hub name): a frozen self-supervised
  backbone (DINOv2 / DINOv3) looks at a context window around each annotated point, its
  local feature is read at the exact point pixel, and a linear probe on those
  features is the classifier (`point_classifier.pt`). The label then describes a
  feature cell of about 1-3 cm rather than a whole crop, which suits point
  annotations, and training takes seconds once the features are cached. See
  *Point-feature classifier* below.
- **Crop classifier** (`--arch resnet34`, `resnet50`, ... any torchvision
  architecture): a FastAI network fine-tuned on crops centred on the points
  (`crop_classifier.pkl`).

Both read the same annotations, label tree, label map and 80/10/10 split, and
`predict`, `segment` and the intercept measurements accept either file.

The command collates the `label` column across all annotation CSVs matching a
glob pattern in `--csv-path` (default CWD), renders them on the CATAMI hierarchy
from `classes.csv`, and shows which entries are **bolded** (the seed training
classes). It then verifies each unique `cam_filepath` directory, and only when
one is missing does it fall back to the `site/site_depth/model/<final-folder>`
convention under `--model-path` (or prompt for a substitution), writes a
consolidated `training_annotations.csv`, and generates `training_crops` /
`validation_crops` / `test_crops` (80/10/10, assigned deterministically per
photo: all points of one photo share a split; see *Train / validation / test
split* below). Finally it trains the model and
reports validation stats
(printed and written to a `<split>_stats.pdf` with a per-class report, a
row-normalised confusion matrix, and example classified crops per category —
one row per category with a red border on misclassified examples). The training
run itself is also captured in a separate `training_summary.pdf` (the run
settings, the per-training-class crop counts across the train/validation/test
splits, and the final-epoch metrics) — kept separate from the evaluation PDF
because `--validate` / `--test` can be re-run without training. Crops are cut
at the classifier's input resolution by default (`--crop-size`). Crop filenames
encode the annotation id, source image, and pixel centre, so a changed
annotation's stale crop is deleted and regenerated; emptied category folders are
cleaned up, and a few example paths are shown before any deletion as a safeguard
against pointing `--output` at the wrong directory. The filenames do not include
the crop size. Instead, the output folder records it in `training_crops.json`.
When `--crop-size` differs from the existing crops, all of them are replaced
(after confirmation; declining stops the run, rather than training on a mix of
sizes). Use a separate `--output` per crop size to keep several. The trained
classifier records its crop size too, so `predict` and `segment` crop the same
by default.

**Crops are decoupled from category selection.** Folders are named by the
*raw* label and crops are generated for every label *visible* in the tree —
not just the trained ones — so changing which categories you consider no longer
regenerates or deletes crops. Crop generation runs in parallel (`--jobs`,
default all cores). Empty or unreadable crops (e.g. from a 0-byte/corrupt source
image) are skipped at training/evaluation time with a warning of how many were
ignored, so a single bad image can't crash the run; zero-byte crops are also
regenerated on the next sync.

**Train / validation / test split.** The split is drawn per `split_key` (a hash
of it; 80/10/10), which `training_annotations.csv` records for every point.
`--split-by` sets it: `photo` (default) gives all points of a photo the same
split (the file stem of `cam_filepath`; CoralNet and pool points already carry
their image), `plot` all points of an annotation file's plot (its model name),
and `point` each point on its own (by its id, the behaviour before 2026-11).
Points of one photo look much alike, so a per-point split puts near-copies of
training points into validation and test and reports optimistic accuracies;
use `--split-by point` only to reproduce an older run. The grouping is recorded
in `training_summary.pdf` (and, for point classifiers, in the model's metadata,
`split_by`). Changing it moves crops between the split folders on the next run.

**Selection and collapsing live in an editable label map.** A
`training_label_map.csv` (`--label-map` to relocate it; default in `--output`)
holds one `label,count,training_class` row per *selected* label (the map is not
cluttered with the sub-threshold/excluded labels, even though their crops still
exist on disk). It is *seeded* from the bolded tree; blanking a row's
`training_class` by hand excludes that label. The seeding
is controlled by `--min-count` (a tip is bolded when its aggregated count, a
parent with subcategories when its own count, is at least N) / `--tips-only`, or by
`--include-classes` (an explicit list of category codes from the tree brackets,
which overrides `--min-count` / `--tips-only` and errors if any code is absent).
By default only the selected codes themselves are trained; add `--collapse` to
fold each selected class's non-selected descendants into it — e.g.
`--include-classes MAF --collapse` trains `MAFG`, `MAF_T`, … as `MAF`, whereas
without `--collapse` only `MAF` itself is trained and its descendants are
excluded. By default each run re-seeds the map from the current selection flags,
so changing `--include-classes` / `--collapse` / `--min-count` takes effect
immediately. Pass `--keep-map` to instead preserve an existing (hand-edited) map
and only append newly-seen labels. Both training and `--validate` / `--test`
read this map, so evaluation always uses the same classes as training.

**Filtering by annotation confidence.** Pass `--min-conf VALUE` to train only on
annotations whose `label_conf` is `>= VALUE`. Annotations with an *empty* (no
value) `label_conf` are **included by default**. Use `--min-conf-strict VALUE`
for the same threshold but treating an empty `label_conf` as `0`, so those
annotations are excluded whenever `VALUE > 0`. The two flags are mutually
exclusive. The confidence filter is applied while collating, and the number of
annotations dropped below the threshold is reported alongside those dropped for
missing camera fields.

**Annotation CSVs from several folders.** `--sets FILE` takes the annotation CSVs from a text file, one path per line, instead of `PATTERN` in `--csv-path`: e.g. the sets of several plots, as substrata-explore's train tool writes it (`training_sets.txt` in the run folder).

**Leaving image matches out of training.** An annotation row with
`include_cam_in_training` = `False` (also `0` / `no`) is left out: its image
match (`cam_filepath`, `cam_x`, `cam_y`) is not fit for training, e.g. the point
is hidden behind something in that photo. A missing column or an empty cell
counts as `True`. Left-out rows are not counted in the label list either, and
the run reports how many there were.

**Without a classes file.** `classes.csv` is only needed for the hierarchical
(CATAMI) tree. When `--classes` is not given and there is no `classes.csv` in the
CWD, the labels are listed as a flat table of counts instead and selected by
`--min-count` (or exactly `--include-classes`); `--tips-only` and `--collapse`
need the hierarchy and are unavailable. Use `--exclude-classes LABEL ...` (in
either mode) to leave labels out entirely, so no crops are generated for them.

**CoralNet point exports (image-only data).** To edit and train on a CoralNet
export from then on, convert it once with `substrata import-coralnet EXPORT.csv
[...] [--image-dir DIR] [--output CSV] [--overwrite]`: each export becomes
`<export stem>_points.csv` next to it (`id`, `label`, `cam_filepath`, `cam_x` =
`Column`, `cam_y` = `Row`, no 3-D position, `split_key` = image,
`include_cam_in_training` = `True`). That CSV is the annotation set to relabel,
include or exclude, and train on like any other (no `--coralnet`). An existing
output is never overwritten without `--overwrite` (its edits would be lost).

To train straight from the exports instead, use `--coralnet`: the matched
CSVs (default pattern `*.csv`) are CoralNet exports with `Name`, `Row`,
`Column` and `Label code` (or `Label`) columns: points on plain images, without
point cloud or cameras. The images are searched (recursively) in `--image-dir`
(default `--csv-path`); when it holds one subfolder per dataset, each CSV is
linked to the subfolder containing most of its images, and the link is reported
(`annotations_site.csv -> Site (213/213 images, 1065 points)`). Each export is
converted into a standard annotations CSV in `<output>/coralnet_points/`
(`Column` → `cam_x`, `Row` → `cam_y`, `split_key` = image), and points on
missing images are dropped. From there the run is identical to the 3-D workflow
(label tree or flat counts, label map, crops, training, stats). The split is
per image, so the points of one image never end up in both training and
validation. CoralNet label codes are matched against `classes.csv` as usual;
without a classes file they are used as-is.

**Starting weights.** By default the network (`--arch`) starts from torchvision's ImageNet weights. `--pretrained-weights CKPT` starts from another encoder checkpoint instead, such as the self-supervised BenthicNet ResNet-50 trained on seafloor imagery (`MLIsaac/benthicnet_bt_rn50_e600_prod` on HuggingFace); use a matching `--arch` (`resnet50`). Plain state dicts and training checkpoints (weights under `state_dict` with a `backbone.`/`encoder.` prefix) are both accepted, and a file that doesn't fit the architecture is refused.

To hand-tune before training, run `--prepare-only`: it generates the crops and
seeds the map, then **stops**. Edit `training_label_map.csv` (re-point a label to
another class, or blank one to exclude it), then re-run `substrata train
--keep-map` to train on the edited map (a plain `substrata train` would re-seed
it and discard your edits).

**Point-feature classifier.** With the default `--arch` (any `dinov2*`
backbone) no crops are needed: each point's photo is read directly (decoded once
per photo, in parallel), a 448 px context window around the point goes through
the backbone, and the feature is interpolated at the point. Steps 1-3 (label
tree, label map, `training_annotations.csv`) are as above; then:

- *Resolution.* Each photo is resampled before the backbone. When the points'
  ground resolution is known (camera distance / focal length, looked up from the
  project cameras next to the photos - the project YAML in or above the photo
  folder - or from a `cam_gsd_mm` column), the classifier works at a physical
  resolution in mm per model pixel, so a feature cell covers the same seabed area
  across cameras and distances; `--feature-mm-px MM` sets it, by default it is
  the median ground resolution at a scale where the window spans about a sixth of
  the photo width (e.g. ~0.7 mm/px for 61 MP diver photos). Without ground
  resolutions (e.g. CoralNet exports) a fixed scale is used instead
  (`--feature-scale S`, model px per photo px; automatic by default). A
  projected point must land within 5 px of its `cam_x`/`cam_y` for its distance
  to be trusted, so points reprojected from another model fall back gracefully.
  The ground resolutions found are cached in
  `point_features/gsd_cache.json`; delete it after adding or re-aligning
  projects. Retraining in a folder that already holds a point classifier keeps
  its resolution (curation changes the points, and a new automatic target would
  re-extract every feature); `--feature-mm-px` / `--feature-scale` (or removing
  the classifier) changes it. Points of another camera domain (see below) never
  set the automatic target.
- *Features are cached* in `<output>/point_features/` per backbone, keyed by
  point (id, photo, pixel) and resolution. Features are extracted for every
  visible label, like crops, so editing the label map, the class selection or
  `--collapse` never re-extracts; `--validate` / `--test` reuse the cache too.
- *Calibration statistics.* Features are standardised with the statistics of
  random patches of the training photos (also cached), which `substrata
  calibrate` can later swap for another camera system's. Points of another
  camera system among the training points (a `domain` column, e.g. AUV points
  labelled from a `curate --mine` pool) are standardised with their own
  statistics (the pool's), so diver and AUV points train one classifier; the
  classifier keeps these per-domain statistics. The patches are drawn per photo
  (by a hash of its name), so adding or removing a few photos only changes their
  own patches.
- *Curation.* Decisions recorded with `substrata curate` (`curation.jsonl` in
  the output folder) are applied while collating: excluded points are left out,
  relabelled points take their new label (also in the label tree counts), and
  accepted points are kept whatever their `label_conf`. A decision made with a
  confidence (keys 0-4 in the review grid) carries it as the point's
  `label_conf`, and `--min-conf` then applies to it. Points of the folder's
  pools (`curate/pools/*.csv`) that were given a label join the training.
  `--curation JSONL` adds another folder's store (read-only; the folder's own
  decisions win), `--no-curation` ignores them all.
- *DINOv3* (`--arch dinov3`, `dinov3-s/-l/-h`): Meta's weights are gated.
  Accept the licence at
  https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/, download
  `dinov3_vitb16_pretrain_lvd1689m-*.pth` (or the size you use) from the link you
  are sent, and put it in torch.hub's checkpoints folder
  (`~/.cache/torch/hub/checkpoints`), or set `SUBSTRATA_DINOV3_WEIGHTS` to the
  file, its folder or the download link. The backbone code comes from GitHub
  (`facebookresearch/dinov3`, once; `SUBSTRATA_DINOV3_HUB_REPO` for a local
  clone) and needs no packages beyond torch. A classifier keeps its backbone, so
  `predict`, `segment`, `curate` and substrata-explore need the same weights on
  that machine.
- *The probe* is a class-balanced logistic regression, its regularisation chosen
  by 5-fold cross-validation on macro-F1 over the training split, with the folds
  grouped by photo (all points of a photo in the same fold).
- *Outputs:* `point_classifier.pt` (`--model`; a small file - the backbone
  weights come from torch.hub, downloaded once to the torch cache; set
  `SUBSTRATA_POINT_HUB_REPO` to a local clone of `facebookresearch/dinov2` to
  work offline; DINOv3: see below), `training_summary.pdf` (settings, resolution, per-class point
  counts per split, validation and test accuracy / balanced accuracy /
  macro-F1) and `validation_points_stats.pdf` (per-class report and confusion
  matrix; `--test` writes `test_points_stats.pdf`).
- Crops are only generated when `--pdf` asks for the contact sheet;
  `--crop-size`, `--epochs` and `--pretrained-weights` apply to the crop
  classifier only. `--batch-size` sets the windows per backbone pass (lower it
  on small GPUs) and `--jobs` the photo-decoding threads.

**Inspecting the crops (`--pdf`).** `--pdf` writes `training_crops.pdf` once the
crops and label map are ready: one A4 page per training class, as an 11 × 15 grid
of crops. Each crop is captioned with its annotation id, so a mislabelled crop can
be traced back to its point. Captions also show the raw label when it was collapsed
into the class, and `val`/`test` for validation and test crops. The ids are PDF
text, so you can search for them. A class with more crops than `--pdf-max`
(default 165, one page) shows a fixed sample, the same on every run; `--pdf-max 0`
shows them all, over as many pages as needed. Combine it with `--prepare-only` to
inspect before training. After training, `--validate --pdf` or `--test --pdf`
rebuilds the PDF from the existing crops and map without collating again, and then
runs the evaluation.

Usage: `substrata train [PATTERN | --sets FILE] [--classes CSV] [--coralnet] [--image-dir DIR] [--csv-path DIR] [--model-path DIR] [--output DIR] [--split-by photo|plot|point] [--min-count N] [--tips-only] [--include-classes LABEL ...] [--exclude-classes LABEL ...] [--collapse] [--min-conf CONF | --min-conf-strict CONF] [--label-map CSV] [--keep-map] [--prepare-only] [--pdf] [--pdf-max N] [--crop-size PX] [--jobs N] [--arch ARCH] [--feature-mm-px MM | --feature-scale S] [--batch-size N] [--pretrained-weights CKPT] [--epochs N] [--model PATH] [--validate] [--test] [--yes] [--curation JSONL ...] [--no-curation]`

```bash
# Collate *_slope_intercepts.csv in CWD, confirm, seed the map, and train a
# point-feature classifier (point_classifier.pt)
substrata train

# Point features at a fixed physical resolution, with a larger backbone
substrata train --feature-mm-px 1.2 --arch dinov2_vitl14_reg

# Point features on CoralNet exports (no cameras): fixed scale for 9568 px photos
substrata train --coralnet --csv-path /data/coralnet --feature-scale 0.25

# The FastAI crop classifier instead (crop_classifier.pkl)
substrata train --arch resnet34

# Two-step: prepare crops + seed the label map, edit it, then train
substrata train --include-classes MAF_T MAENRC_C CSE --prepare-only
$EDITOR training_label_map.csv   # re-point / collapse / exclude labels
substrata train --keep-map       # trains on the edited map (edits preserved)

# Prepare, then inspect the crops per class before training
substrata train --classes classes.csv --min-count 20 --collapse --prepare-only --pdf

# Train parent classes, folding their sub-categories in (MAFG/MAF_T -> MAF)
substrata train --include-classes MAF MAEN --collapse

# Custom pattern; seed the map from labels with an aggregated count >= 50
substrata train "*_ann.csv" --min-count 50

# Only train on annotations with label_conf >= 0.8 (empty label_conf kept)
substrata train --min-conf 0.8

# Same, but treat an empty label_conf as 0 so those annotations are excluded
substrata train --min-conf-strict 0.8

# Keep a hand-edited map, only appending labels from newly added CSVs
substrata train --keep-map

# Reproduce a run made before 2026-11 (each point split on its own)
substrata train --split-by point

# CSVs in one dir, image projects in another, output elsewhere, bigger backbone
substrata train --csv-path /data/annotations --model-path /data/models \
    --output /data/training --arch resnet50 --epochs 20

# CoralNet exports in /data/coralnet, images in per-dataset subfolders of
# /data/coralnet_images; no classes.csv, so labels are selected by count
substrata train --coralnet --csv-path /data/coralnet \
    --image-dir /data/coralnet_images --exclude-classes Unc --min-count 20

# Start from BenthicNet's seafloor-pretrained ResNet-50 instead of ImageNet
substrata train --arch resnet50 --pretrained-weights "bt-600e-prod_rn50_epoch=599.ckpt"

# Re-run validation stats on an existing model (reads the same label map)
substrata train --validate --model crop_classifier.pkl

# Skip training; evaluate an existing model on the held-out test crops
substrata train --test --model crop_classifier.pkl

# Same for a point classifier (held-out test points; features from the cache)
substrata train --test

# Non-interactive (auto-confirm, deletions, path fallbacks)
substrata train --yes
```

## Curation (curate)

Curate the training points of a point-feature classifier. `substrata curate` runs
in a classifier folder (where `substrata train` ran, with its CSVs from any
number of projects) and works on what training sees: photos and pixels, no
point clouds.

**Decisions** go to `curation.jsonl` in the folder, one JSON line per decision:
`accept`, `relabel` (with the new label), `exclude` (with a reason: `mixed`,
`misplaced`, `unclear`, `photo`, `other`) or `clear`; an accept or relabel can
carry the reviewer's confidence (`label_conf`, 0-1). The file is append-only and
the latest line of a point wins. Points are keyed by the photo's file name and
the rounded pixel (`IMG_1234.jpg|4021|2210`), not by row or id, so decisions
survive regenerated training sets, other mixes of CSVs, renumbered ids and moved
photo folders. The next `substrata train` in the folder applies them (see
*Curation* under the point-feature classifier); copy the file, or pass
`--curation PATH`, to use them in another classifier folder.

**Scores.** `substrata curate` (no options) scores the training points of the
**train split** for review and writes `curate/points.csv` plus a thumbnail of each
model input window (`curate/tiles/`). Validation and test points are left out by
default: relabelling held-out points where the model disagrees with them would
tune their labels towards the model and inflate its validation / test metrics.
`--all-splits` scores and lists them too (e.g. to clean a set before a new split).
The points are read at exactly the resolution training used:

- `p_label`: the cross-validated probability of the point's own class (5 folds,
  grouped by photo so neighbouring points of a photo never vouch for each
  other) - low for label noise; `cv_pred` / `cv_conf` is the predicted class;
- `ring_agree`: the mean probability of the point's class at 8 positions 2
  feature cells (~2 cm at 0.7 mm/px) around it, read in the same window - low
  for points on boundaries and small objects, whose label will not hold for a
  patch; `ring_sim` is the feature similarity to that ring;
- `knn_label`: the share of the 10 most similar points (in other photos) with
  the same class;
- `novelty`: 1 - the mean similarity to those neighbours.

Features come from the training cache, so this costs one pass over the photos
for the neighbourhoods and thumbnails (cached too; ~30 s for 2,500 points).
In the samoa test (2,362 diver points, 14 classes), accuracy rose from 0.38 to
0.98 across the quartiles of `ring_agree` and from 0.28 to 0.98 across those of
`knn_label`.

**Pools.** `substrata curate --mine IMAGES --name NAME` samples unlabelled
points from photos (e.g. AUV frames of habitats without annotations; folders,
globs or a `.txt` list): `--mine-images` photos (2000), `--per-image` points on
a jittered grid (12), resampled to the classifier's resolution (`--gsd-mm` for
classifiers in mm/px). Their features are standardised with their own
statistics (`--domain`, default the pool name: one camera system), clustered
(PCA + k-means, `--clusters`, 60; largest first), predicted and compared with the
training points (novelty). The pool is `curate/pools/<name>.csv` (an annotation
CSV with an empty `label`, plus `cluster`, `pred_label`, `novelty`, ...), `.npz`
(features, domain statistics) and `.json` (per-cluster summary). Re-running
with other `--clusters` re-clusters the same points (`--resample` draws new
ones). Pool points given a label (relabel) join the next `substrata train` in
the folder, standardised with their pool's statistics.

*Image-only pools* take `substrata predict`'s inputs: `--image-dir DIR` samples
the photos under DIR (like `--mine DIR`; `--n-points` = `--per-image`, `--seed`),
and `--points CSV` uses the points of a points CSV (`cam_filepath`, `cam_x`,
`cam_y`, e.g. `predict`'s output) or a CoralNet export (photos under
`--image-dir`, default the CSV's folder) instead of sampled ones. Their labels
become the pool's `given_label`: shown in the review grid, where accepting one
labels the point with it (only reviewed points train). Scales follow the same
precedence as `predict`: the ground resolution first (`--gsd-mm`, or a
`cam_gsd_mm` column of a `--points` CSV; classifiers in mm/px); else photos of
another camera (width more than 25% off the training photos') are resampled so
the model window covers the same share of their width as on the training photos
(scale = the classifier's fixed scale × the training photos' median width / the
photo's width; e.g. 0.95 for 2816 px AUV frames with a diver classifier at 0.28
on 9568 px photos); else the fixed scale. `substrata train` resamples the pool's
labelled points the same way, and `predict --domain <pool domain>` classifies
new photos of that camera as they were trained.

**Reviewing.** `substrata curate --gui` (or `substrata-curate`, both from the
substrata-explore package) opens the review grid in the browser: queues
(suspect labels, mixed neighbourhoods, neighbour labels, unusual points, by
class), pools by cluster, a wider view of the photo around the point, class
hotkeys from `classes.csv`, and keys for accept / exclude / relabel / undo; see
substrata-explore's documentation. Without it, `--sheets` writes contact sheets
of the queues (`curate/review_sheets.pdf`; captions: id, label > predicted
class, label probability), and `--import CSV` reads decisions from a CSV
(`action`, `label`, `label_conf`, `reason`, and the point as `key`,
`cam_filepath`/`photo` + `cam_x`/`x` + `cam_y`/`y`, or `id`). `--status`
summarises the decisions. `--export CSV` writes the outcome per point, no
history (cleared points left out), sorted by photo and position: `photo`, `x`,
`y`, `id`, `action`, `orig_label`, `label` (the new label of a relabel),
`label_conf`, `reason`, `src` (training points or a pool), `time`, `by`, `key`.
Filter it on `action` to find the relabels to carry over into the original
annotation files; an edited copy reads back with `--import`. The review grid can also
write relabels into the annotation files directly ("also edit the source
annotations", off by default): `substrata train` records the files it read in
`training_sources.json`, and only image-only points (CoralNet exports, or
annotation CSVs without 3-D positions) are edited, after a backup of each file.

Usage: `substrata curate [--folder DIR] [--model PT] [--annotations CSV] [--all-splits] [{--mine IMAGES ... | --image-dir DIR | --points CSV [--image-dir DIR]} --name NAME [--gsd-mm MM] [--domain D] [--mine-images N] [--per-image/--n-points N] [--clusters K] [--seed N] [--resample]] [--status] [--export CSV] [--import CSV] [--sheets [--sheets-max N]] [--curation JSONL ...] [--batch-size N] [--jobs N] [--gui [GUI OPTIONS]]`

```bash
# In the classifier folder, after `substrata train`
substrata curate                 # score the training points
substrata curate --gui           # review them in the browser (port 8010)

# Mine AUV frames of 60-145 m (no diver annotations) into a clustered pool
substrata curate --mine /data/auv/deep_frames.txt --name auv_deep --domain auv \
    --gsd-mm 1.21 --clusters 40

# Image-only, as with `substrata predict`: a folder of photos (no ground
# resolution: the window keeps its share of the photo width) ...
substrata curate --image-dir /data/drop_cam/2024 --name dropcam --n-points 10
# ... or given points: predict's output, or a CoralNet export (labels kept as
# given labels, accepted or changed in the review grid)
substrata curate --points /data/coralnet/annotations.csv --image-dir /data/coralnet/images \
    --name coralnet_site4

# Without the GUI: contact sheets, then decisions from a CSV
substrata curate --sheets
substrata curate --import decisions.csv --status

# Apply the decisions
substrata train --keep-map --yes
```

## Masks (dense per-photo classification)

Classify every part of plain photos (no 3-D model needed, e.g. AUV frames) with a
point-feature classifier and save the result as masks. Each photo is resampled
(to the classifier's resolution: `--gsd-mm` gives the photos' ground
resolution for classifiers working in mm/px, otherwise the classifier's fixed
scale is used), cut into overlapping tiles and classified per feature cell
(about 14 resampled pixels; the tiles are blended so there are no seams). Use a
classifier calibrated to the camera system (`substrata calibrate`) for photos of
another camera than the training photos.

For each photo, in the output folder (`--output`, default `./masks`):

- `<stem>_labels.png`: palette PNG, one pixel per feature cell; the pixel value
  is the class index (order of `classes` in `masks.json`), shown in the class
  colour (`--color LABEL=RRGGBB` to set one).
- `<stem>_conf.png`: 8-bit confidence of that class (255 = 1.0).
- `<stem>_probs.npz` (with `--probs`): all class probabilities, uint8 (0-255).

`masks.json` indexes them: the classes and colours, and per photo the files,
`cell_px` (photo pixels per mask pixel - upsample by it to overlay a mask on its
photo), size, mean confidence and cover. `masks_cover.csv` has one row per photo
with the fraction of its cells per class (most likely class), e.g. for percent
cover per frame. Photos already in the index are skipped, so an interrupted run
continues where it stopped (`--overwrite` redoes them); the index is saved every
200 photos. The index records the classifier's fingerprint (`classifier_fingerprint`,
as `pred_model`); masks are only reused when made by the same classifier file and
`--gsd-mm` (a retrained classifier, even with the same classes, redoes all photos). Stems that repeat across folders are prefixed with their folder name.

Usage: `substrata masks IMAGES ... [--classifier PT] [--gsd-mm MM] [--output DIR] [--probs] [--color LABEL=RRGGBB ...] [--overwrite] [--jobs N]`

```bash
# AUV frames of one dive, with the AUV-calibrated classifier at 1.21 mm/px
substrata masks /data/auv/dive3 --classifier point_classifier_calibrated.pt \
    --gsd-mm 1.21 --output /data/auv/dive3_masks

# Photos like the training photos, keeping all probabilities
substrata masks "photos/*.jpg" --probs
```

## Classifier calibration (calibrate)

Adapt a point-feature classifier (`substrata train`, default `--arch`) to
another camera system, e.g. apply a classifier trained on diver photos to AUV
frames. The classifier standardises its backbone features with statistics of
random patches of its training photos; `calibrate` computes the same statistics
for photos of the other camera and writes a copy of the classifier that uses
them (`<classifier>_calibrated.pt`, or `--output`).

The calibration photos must show **the same area** as the training photos (e.g.
the AUV frames over the diver plots): no labels are needed, but statistics of
other habitats would shift the classes along with the camera. Once made, the
calibrated classifier is used for every photo of that camera system, including
other areas and depths. Classifiers with a physical resolution (mm/px) need the
ground resolution of the calibration photos (`--gsd-mm`, camera distance / focal
length, e.g. 1.21 for an AUV flying at ~3.8 m with a 3106 px focal length); at
prediction time the resolution of each photo comes from its camera geometry.
The calibration patches are resampled exactly as prediction resamples those
photos: by `--gsd-mm`, else (fixed-scale classifiers) by the photo width
relative to the training photos; each photo's scale is stored with the
calibration in the model's metadata (`calibration.scales`).

Usage: `substrata calibrate IMAGES ... [--classifier PT] [--gsd-mm MM] [--n-images N] [--per-image N] [--output PT] [--batch-size N] [--jobs N]`

```bash
# AUV frames over the diver plot (folders are searched recursively)
substrata calibrate /data/auv/plot_frames --classifier point_classifier.pt --gsd-mm 1.21

# Glob pattern, more photos, explicit output
substrata calibrate "/data/auv/dive3/*.png" --gsd-mm 1.2 --n-images 1000 \
    --output point_classifier_auv.pt
```
