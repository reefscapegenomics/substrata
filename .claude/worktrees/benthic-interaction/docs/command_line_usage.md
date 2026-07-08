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

To hand-tune before training, run `--prepare-only`: it generates the crops and
seeds the map, then **stops**. Edit `training_label_map.csv` (re-point a label to
another class, or blank one to exclude it), then re-run `substrata train
--keep-map` to train on the edited map (a plain `substrata train` would re-seed
it and discard your edits).

Usage: `substrata train [PATTERN] [--classes CSV] [--csv-path DIR] [--model-path DIR] [--output DIR] [--min-count N] [--tips_only] [--include-classes LABEL ...] [--collapse] [--label-map CSV] [--keep-map] [--prepare-only] [--crop-size PX] [--jobs N] [--arch ARCH] [--epochs N] [--model PKL] [--validate] [--test] [--yes]`

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

