# Command-line usage of `substrata`

This tutorial walks through the available command-line tools of the `substrata` Python package.

Note: All commands look for a YAML project file or default filenames in the current working directory, unless separately specified. Most commands use the `ProjectInitializer` to auto-detect project files based on the current directory name.

## Decimation of PLY files

Decimate a PLY file to reduce the number of points. With no arguments, uses initializer on CWD, output to `<id>_dec50M.ply`, target = 50,000,000 points.

Usage: `substrata decimate [--input PLY] [--output PLY] [--points N]`

```bash
# Using default behavior (auto-detects from CWD)
substrata decimate

# With explicit arguments
substrata decimate --input cur_sna_20m_20200303.ply --output cur_sna_20m_20200303_dec50M.ply --points 50000000

# Using short flags
substrata decimate --ply input.ply -n 10000000
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

Usage: `substrata repair [--input PLY] [--output PLY] [--local]`

```bash
# Repair in place (original kept as <input>_old.ply)
substrata repair --input pointcloud.ply

# Write the repaired copy to an explicit path (input left untouched)
substrata repair --input pointcloud.ply --output pointcloud_fixed.ply
```

## PLY file preview (head)

Show the first N vertex rows from a PLY file.

Usage: `substrata head [--input PLY] [-n N]`

```bash
# Show first 5 rows (default)
substrata head

# Show first 10 rows
substrata head --input pointcloud.ply -n 10
```

## Visual assessment of scalebars

Generate a scalebar PDF from a point cloud and marker annotations. Optionally save the computed scale factor to YAML.

Usage: `substrata scalebars [--input PLY] [--markers CSV] [--output_pdf PDF] [--points N] [--save_yaml]`

```bash
# Using default behavior (auto-detects from CWD)
substrata scalebars

# With explicit arguments
substrata scalebars --input cur_sna_20m_20200303_dec50M.ply --markers cur_sna_20m_20200303_markers.csv --output_pdf ~/scalebar_check.pdf

# Save scale factor to YAML
substrata scalebars --save_yaml

# Limit points loaded (stream decimation)
substrata scalebars --points 10000000
```

## Composite views

Save composite views PDF for a point cloud showing multiple perspectives.

Usage: `substrata views [--input PLY] [--output_pdf PDF]`

```bash
# Using default behavior
substrata views

# With explicit output path
substrata views --input pointcloud.ply --output_pdf views.pdf
```

## Orientation and scaling

Calculate and apply scale and orientation transforms, then save to YAML. Also generates composite views and camera depth residuals PDFs.

Usage: `substrata orient [--input PLY]`

```bash
# Using default behavior (auto-detects from CWD)
substrata orient

# With explicit PLY path
substrata orient --input pointcloud.ply
```

## Color calibration

Run color calibration from the project's ColorChecker marker positions, save a QC PDF, and optionally store the affine color correction (matrix + offset) in the project YAML. The PLY is always loaded in full when `--input` is given (never stream-decimated).

Usage: `substrata colors [--input PLY] [--markers CSV] [--output_pdf PDF] [-n N] [--save_yaml] [--exclude-index N ...] [--exclude-name NAME ...]`

```bash
# Using default behavior (auto-detects from CWD)
substrata colors

# Save the affine colour_correction into the project YAML
substrata colors --save_yaml

# Stream-sample the default YAML PLY to N vertices for a faster QC pass
substrata colors -n 10000000

# Omit specific ColorChecker cards from the fit (by 0-based index or by name)
substrata colors --exclude-index 0 --exclude-index 3
substrata colors --exclude-name top-left
```

## FireFish alignment

Run FireFish/Cameras alignment to determine up vector and generate output PDF. Initializes FireFish and Cameras, then determines the up vector based on camera depth data.

Usage: `substrata firefish [--firefish-file FILE] [--target-depth M] [--cam-depths-file CSV] [--depth-outlier-threshold M] [--cams_group GROUP] [--offset SEC] [--input PLY] [--save_yaml]`

```bash
# Using default behavior (auto-detects from CWD, infers depth from directory name)
substrata firefish

# With explicit target depth
substrata firefish --target-depth 20

# Filter cameras by group
substrata firefish --cams_group "group_name"

# Save results to YAML
substrata firefish --save_yaml

# With manual time offset
substrata firefish --offset 30
```

## Camera video creation

Create a video from cameras by drawing image matches. Optionally include annotations in the video.

Usage: `substrata cams2video [--input PLY] [--annotations CSV] [--cams_group GROUP] [--label] [--resolution WIDTH] [--output_mp4 MP4]`

```bash
# Using default behavior (auto-detects from CWD)
substrata cams2video

# With annotations
substrata cams2video --annotations annotations.csv

# Filter cameras by group
substrata cams2video --cams_group "group_name"

# Use label column from annotations
substrata cams2video --label

# Resize images to specific width
substrata cams2video --resolution 1920

# Specify output file
substrata cams2video --output_mp4 output.mp4
```

## Z-intercepts calculation

Find optimal box position, subdivide to grid, sample random points, and compute Z-intercepts. Optionally apply along-slope transform before processing.

Usage: `substrata intercepts [--input PLY] [--box-length M] [--box-width M] [--box-size M] [--search-radius M] [--slope]`

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

## Point cloud alignment

Register a source PLY to a target PLY and print the alignment transform.

Usage: `substrata align --source PLY --target PLY [--points N]`

```bash
# Align two point clouds
substrata align --source source.ply --target target.ply

# Limit points for faster processing
substrata align --source source.ply --target target.ply --points 5000000
```

## Image matches

Find image matches of annotations and output cropped images to PDF. Optionally apply a transform to annotation coordinates before matching.

Usage: `substrata images [--input PLY] [--annotations CSV] [--transform] [--pdf-output PDF]`

```bash
# Using default behavior (auto-detects from CWD)
substrata images

# With explicit annotations file
substrata images --annotations annotations.csv

# Apply transform to annotation coordinates (interactive)
substrata images --transform

# Specify output PDF
substrata images --pdf-output imagematches.pdf
```

## Camera time-sync (camsync)

Copy camera centers/transforms from a *pose-source* sensor (e.g. a GoPro) to an *updated-target* sensor (e.g. a macro camera) using EXIF time matching. Per-camera poses are written to the cameras meta JSON only — the `.cams.xml` is not modified (it is read for sensor calibration). The relative timing of the two sensors can be given manually (`--time-offset` / `--pose-time`) or estimated automatically (`--auto-time` / `--auto-xyz` / `--auto-offsets`); auto modes print a detailed report and prompt before saving unless you pass `--yes`.

Usage: `substrata camsync [--pose-source ID] [--updated-target ID] [--pose-date DATES] [--target-date DATES] [--time-offset SEC | --pose-time TIMESTAMP] [--xyz X,Y,Z] [--auto-time] [--auto-xyz] [--auto-offsets] [--spatial-max-dist M] [--min-spatial-pairs N] [--yes]`

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
```

## Transform annotations

Load annotations from a CSV, apply one or more transforms (entered interactively) to each annotation's `orig_coords`, and save the result. With `--inverse` the cumulative transform is inverted before being applied.

Usage: `substrata transform INPUT [--output CSV] [--no_header] [--ignore_header] [--inverse]`

```bash
# Transform annotations; prompts for the transform matrix/matrices
substrata transform annotations.csv

# Write to an explicit output path
substrata transform annotations.csv --output annotations_world.csv

# Apply the inverse of the entered transform(s)
substrata transform annotations.csv --inverse
```

## Classifier training

Train a FastAI image classifier on crops generated from labelled annotations.
The command collates the `label` column across all annotation CSVs matching a
glob pattern in `--csv-path` (default CWD), renders them on the CATAMI hierarchy
from `classes.csv`, and shows which entries are **bolded** (the seed training
classes). It then verifies each unique `cam_filepath` directory, and only when
one is missing does it fall back to the `site/site_depth/model/<final-folder>`
convention under `--model-path` (or prompt for a substitution), writes a
consolidated `training_annotations.csv`, and generates `training_crops` /
`validation_crops` / `test_crops` (80/10/10, assigned deterministically per
annotation id). Finally it trains the model and reports validation stats
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
against pointing `--output` at the wrong directory.

**Crops are decoupled from category selection.** Folders are named by the
*raw* label and crops are generated for every label *visible* in the tree —
not just the trained ones — so changing which categories you consider no longer
regenerates or deletes crops. Crop generation runs in parallel (`--jobs`,
default all cores). Empty or unreadable crops (e.g. from a 0-byte/corrupt source
image) are skipped at training/evaluation time with a warning of how many were
ignored, so a single bad image can't crash the run; zero-byte crops are also
regenerated on the next sync.

**Selection and collapsing live in an editable label map.** A
`training_label_map.csv` (`--label-map` to relocate it; default in `--output`)
holds one `label,count,training_class` row per *selected* label (the map is not
cluttered with the sub-threshold/excluded labels, even though their crops still
exist on disk). It is *seeded* from the bolded tree; blanking a row's
`training_class` by hand excludes that label. The seeding
is controlled by `--min-count` / `--tips_only`, or by
`--include-classes` (an explicit list of category codes from the tree brackets,
which overrides `--min-count` / `--tips_only` and errors if any code is absent).
By default only the selected codes themselves are trained; add `--collapse` to
fold each selected class's non-selected descendants into it — e.g.
`--include-classes MAF --collapse` trains `MAFG`, `MAF_T`, … as `MAF`, whereas
without `--collapse` only `MAF` itself is trained and its descendants are
excluded. By default each run re-seeds the map from the current selection flags,
so changing `--include-classes` / `--collapse` / `--min-count` takes effect
immediately. Pass `--keep-map` to instead preserve an existing (hand-edited) map
and only append newly-seen labels. Both training and `--validate` / `--test`
read this map, so evaluation always uses the same classes as training.

**Filtering by annotation confidence.** Pass `--min_conf VALUE` to train only on
annotations whose `label_conf` is `>= VALUE`. Annotations with an *empty* (no
value) `label_conf` are **included by default**. Use `--min_conf_strict VALUE`
for the same threshold but treating an empty `label_conf` as `0`, so those
annotations are excluded whenever `VALUE > 0`. The two flags are mutually
exclusive. The confidence filter is applied while collating, and the number of
annotations dropped below the threshold is reported alongside those dropped for
missing camera fields.

To hand-tune before training, run `--prepare-only`: it generates the crops and
seeds the map, then **stops**. Edit `training_label_map.csv` (re-point a label to
another class, or blank one to exclude it), then re-run `substrata train
--keep-map` to train on the edited map (a plain `substrata train` would re-seed
it and discard your edits).

Usage: `substrata train [PATTERN] [--classes CSV] [--csv-path DIR] [--model-path DIR] [--output DIR] [--min-count N] [--tips_only] [--include-classes LABEL ...] [--collapse] [--min_conf CONF | --min_conf_strict CONF] [--label-map CSV] [--keep-map] [--prepare-only] [--crop-size PX] [--jobs N] [--arch ARCH] [--epochs N] [--model PKL] [--validate] [--test] [--yes]`

```bash
# Collate *_slope_intercepts.csv in CWD, confirm, crop, seed the map, and train
substrata train

# Two-step: prepare crops + seed the label map, edit it, then train
substrata train --include-classes MAF_T MAENRC_C CSE --prepare-only
$EDITOR training_label_map.csv   # re-point / collapse / exclude labels
substrata train --keep-map       # trains on the edited map (edits preserved)

# Train parent classes, folding their sub-categories in (MAFG/MAF_T -> MAF)
substrata train --include-classes MAF MAEN --collapse

# Custom pattern; seed the map from labels with an aggregated count >= 50
substrata train "*_ann.csv" --min-count 50

# Only train on annotations with label_conf >= 0.8 (empty label_conf kept)
substrata train --min_conf 0.8

# Same, but treat an empty label_conf as 0 so those annotations are excluded
substrata train --min_conf_strict 0.8

# Keep a hand-edited map, only appending labels from newly added CSVs
substrata train --keep-map

# CSVs in one dir, image projects in another, output elsewhere, bigger backbone
substrata train --csv-path /data/annotations --model-path /data/models \
    --output /data/training --arch resnet50 --epochs 20

# Re-run validation stats on an existing model (reads the same label map)
substrata train --validate --model crop_classifier.pkl

# Skip training; evaluate an existing model on the held-out test crops
substrata train --test --model crop_classifier.pkl

# Non-interactive (auto-confirm, deletions, path fallbacks)
substrata train --yes
```

