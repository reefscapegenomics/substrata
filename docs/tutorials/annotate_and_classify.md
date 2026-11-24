# Point intercepts, labels and a classifier

This tutorial draws random point intercepts on a plot, labels them from the photos,
trains a classifier on the labels, and uses it to label new points and to segment
the whole cloud. It starts from a plot prepared as in {doc}`plot_workflow`: oriented
(`world_transform` in `<id>.yaml`) and stratified (`<id>.strat.npz`).

| Step | Command | Run in | Writes |
| --- | --- | --- | --- |
| 1 | `substrata intercepts --slope` | plot folder | `<id>.annos/<id>_slope_intercepts.csv` (+ sidecar), `<id>_slope_bbox.png` |
| 2 | `substrata crops` | plot folder | `<id>.annos/<id>_slope_intercepts.cropped/`; training views in the CSV |
| 3 | `substrata explore --edit` | plot folder | labels in the CSV |
| 4 | `substrata intercepts-plot` | plot folder | `<id>.annos/<id>_slope_intercepts_grid.png`, `.gif`, `_positions.png` |
| 5 | `substrata train` | classifier folder | `point_classifier.pt`, `training_summary.pdf`, ... |
| 6 | `substrata predict` | plot folder | `pred_*` columns in a set's CSV |
| 7 | `substrata segment` | plot folder | `<id>.annos/<id>_seg_points.csv`, `<id>_seg_dec.ply` |
| 8 | `substrata curate` | classifier folder | `curation.jsonl`, `curate/` |
| (9) | `substrata calibrate` | classifier folder | `point_classifier_calibrated.pt` |

Every point set is an *annotation set*: a CSV in `<id>.annos/` with an `id` column,
the raw coordinates (`orig_x`, `orig_y`, `orig_z`), a `label`, and the photo and
pixel the point is labelled on (`cam_filepath`, `cam_x`, `cam_y`). Next to it sits an
optional sidecar `<csv>.yaml` with the set's settings. See {doc}`../projects`.

## 1. Draw point intercepts

```bash
cd sam_sa3_40m_20250928
substrata intercepts --slope
```

Reads the YAML's point cloud and `world_transform` (it stops if the plot is not
oriented). What it computes:

1. With `--slope`, the cloud is rotated so that a plane fitted to the terrain is
   flat (as `orient --slope`). Without it, the grid is drawn top-down in the world
   frame.
2. A box of `--box-length` × `--box-width` (default 25 × 4 m) is placed where it holds
   the most points (counts on a 0.1 m grid), and divided into cells of `--box-size`
   (default 0.2 m).
3. One random XY position is drawn in each cell. The intercept is the cloud point
   below it: of the points within `--search-radius` (default 5 mm, horizontal
   distance) of that position, the one nearest to (x, y, median z of those
   points). When no point lies that close, the radius grows in 5 mm steps until
   some do, and the point is placed at (x, y, median z) with `is_extrapolated` True.

Writes `<id>.annos/<id>_slope_intercepts.csv`, its sidecar
`<id>_slope_intercepts.csv.yaml` (the box, the grid, the cell size and the slope
transform, so the exact grid can be replayed), and `<id>_slope_bbox.png` in the plot
folder. An intercepts CSV with labels is never overwritten: use `--tag NAME` for a
new set next to it.

**Check.** `<id>_slope_bbox.png` shows the box over the part of the plot you want to
survey. The command prints `Intercept points returned: N/M` and the number of
extrapolated points; extrapolated points sit over holes in the cloud and are worth a
look.

**Training points at other depths.** The best box can leave out parts of the plot.
For training data across a depth band, draw points over the plot's whole width
instead of in a box:

```bash
substrata intercepts --depth-range 38 42 --training-only --tag band
```

This places one intercept per cell of `--width-step` (across the slope, default 1 m)
by `--depth-step` (in depth, default 1 m), in `<id>_depth_band_intercepts.csv`.
`--training-only` marks the set `purpose: training_only` in its sidecar, so cover
summaries leave it out. With `--plan` it only reports the cells.

## 2. Crops and training views

```bash
substrata crops --annotations sam_sa3_40m_20250928.annos/sam_sa3_40m_20250928_slope_intercepts.csv
```

Reads the set, the cameras and the stratified PLY. For each point it chooses the
photos that see it best and cuts a crop (default 800 × 800 px) centred on the
point from each of up to `--max-photos` (default 5) of them.

How views are chosen:

- Cameras are grouped into camera types (same resolution, focal length within
  3 %). Only the main camera type is cropped: the one that sees most points best
  (`--camera-type` to choose another).
- Views are ranked by relevance: close to the point and the point near the photo
  centre. Views whose crop would run past the photo edge are skipped.
- Occlusion: the line of sight from the camera to the point is tested on the
  stratified PLY's subsample; a view with 3 or more points on it counts as
  obstructed and goes last. Without a stratified PLY the older reprojection test on
  the YAML's cloud is used.

Writes the crops to `<id>.annos/<id>_slope_intercepts.cropped/<point id>/<photo
folder>/<photo stem>_<x>_<y>.jpg`. It also writes each point's *training view*, its
most relevant unobstructed view, into the CSV (`cam_filepath`, `cam_x`, `cam_y`,
`include_cam_in_training` = True) where `cam_filepath` is still empty. The CSV is
backed up to `<id>.annos/backup/` first.

**Check.** The command lists points whose views are all obstructed: they get no
training view and need a look. Crops that no longer show their point (after a
re-alignment) are removed as `stale`.

## 3. Label the points

```bash
substrata explore --edit
```

Opens substrata-explore (a separate package) in the browser for the plot in the
current folder. Choose the intercepts set, look at each point in its crops and in
the photos, and give it a label. Label codes and their hotkeys come from
`classes.csv` in `<id>.annos/` (or `--classes CSV`). With `--edit` the labels are
saved into the set's `label` column (a backup goes to `backup/` once per run).
Without `--edit` nothing is written. `substrata explore -h` lists the options; the
viewer needs a stratified plot.

A set can also carry a `label_conf` column (0-1): how sure each label is.
`train --min-conf` can later leave out the unsure ones.

## 4. Map the labels

```bash
substrata intercepts-plot \
    --intercepts sam_sa3_40m_20250928.annos/sam_sa3_40m_20250928_slope_intercepts.csv
```

Replays the grid from the CSV's sidecar and colours each cell by its majority label
(`--grid-size 1.0` for coarser cells; `--label-colors FILE` to set colours). Writes
three files next to the CSV: `<stem>_grid.png`, an animated `<stem>_grid.gif`, and
`<stem>_positions.png` (the points over a top-down map of the cloud).

**Check.** Each cell of the replayed grid holds one point. Many empty ("No data")
cells mean the grid was not replayed exactly: older files without `grid_bbox` in
their sidecar need `--fit-grid`.

## 5. Train a classifier

Train in a folder of its own: it collects the labelled sets of one or more plots.

```bash
mkdir -p ~/classifiers/samoa_v1 && cd ~/classifiers/samoa_v1
cp /data/samoa/classes.csv .
ls /data/samoa/sam_*/*/*/*.annos/*_slope_intercepts.csv > training_sets.txt
substrata train --sets training_sets.txt --min-count 20
```

Without `--sets` it reads the CSVs matching `*_slope_intercepts.csv` (or a
pattern you give) in `--csv-path` (default: the current folder).

What it does:

1. Collects the `label` column of all sets and shows the labels on the CATAMI tree
   of `classes.csv`. The **bold** entries are the classes that will be trained:
   those with at least `--min-count` points, or exactly `--include-classes`. Labels
   below a trained class can be folded into it with `--collapse`. The choice is
   written to `training_label_map.csv`, which you can edit (`--prepare-only`, edit,
   then `--keep-map`).
2. Leaves out points without a training view, with `include_cam_in_training`
   False, or with `label_conf` below `--min-conf`.
3. Splits the points 80 / 10 / 10 into train, validation and test sets.
4. Trains the classifier and evaluates it on the validation set.

**The default classifier** (`--arch dinov2`) is a point-feature classifier: a frozen
DINOv2 backbone (ViT-B/14 with registers) looks at a 448 px window around each
point, its feature map is read at the point's pixel, and a linear probe (a
class-balanced logistic regression) on those features is the classifier. Each
photo is first resampled to one ground resolution (mm per pixel, from the camera
distance and focal length), so a feature cell covers about the same patch of reef
in every photo (about 1-3 cm). Features are standardised with the statistics of
random patches of the training photos. Features are cached in `point_features/`,
so retraining takes seconds. `--arch resnet34` (or another torchvision network)
trains the older crop classifier instead (`crop_classifier.pkl`).

**Why `--split-by photo`.** By default all points of one photo go to the same
split (a hash of the photo's file name decides which). Points of one photo look
alike: same light, same distance, often the same colony. With a per-point split
(`--split-by point`, the behaviour before November 2026), near-copies of training
points end up in validation and test, and the reported accuracy is too high. The
probe's regularisation is also chosen by cross-validation with the folds grouped by
photo. `--split-by plot` keeps whole plots together, to test transfer to new plots.

Writes, in the classifier folder:

- `point_classifier.pt`: the classifier (small: the backbone weights come from
  torch.hub);
- `training_annotations.csv`: the collated points with their split;
- `training_label_map.csv`: label to training class;
- `training_summary.pdf`: settings, resolution, points per class and split, and
  validation and test accuracy, balanced accuracy and macro-F1;
- `validation_points_stats.pdf`: per-class precision and recall, and the confusion
  matrix;
- `training_sources.json`: the CSVs read.

**Check.** In `validation_points_stats.pdf`, look at the per-class recall and the
confusion matrix rather than the overall accuracy: a common class can hide a rare
one that is never predicted. Classes that are confused with each other may need
merging (`--include-classes PARENT --collapse`) or more points. Evaluate on the test
set (`substrata train --test`, writes `test_points_stats.pdf`) once, at the end:
tuning on it makes it a second validation set.

## 6. Predict labels

In a plot folder, for any annotation set:

```bash
cd /data/samoa/sam_sa3/sam_sa3_40m/sam_sa3_40m_20250928
substrata predict --classifier ~/classifiers/samoa_v1/point_classifier.pt \
    --annotations sam_sa3_40m_20250928.annos/sam_sa3_40m_20250928_slope_intercepts.csv
```

Each point is classified in up to `--max-photos` (default 10) views, chosen as for
`crops`. The views' class probabilities are averaged, each weighted by its
relevance, and obstructed views are left out unless all are (`--method first` uses
the best view only).

Writes these columns into the set's CSV (backed up first; other columns are left
as they are): `pred_label`, `pred_conf` (the averaged probability of that class),
`pred_n` (views used), `pred_agree` (share of views whose own prediction is the
combined one), `pred_top` (the five most likely classes) and `pred_model` (the
classifier's file name and a hash of its contents). Points already predicted by
the same classifier file are skipped; `--redo` predicts them all again.

To avoid passing `--classifier` each time, name the classifier in the set's
sidecar, e.g. `classifier: /data/classifiers/samoa_v1/point_classifier.pt` (a
relative path is taken from the sidecar's folder). It then also records which model
labelled the set.

**Check.** Compare `pred_label` with `label` where both exist. Low `pred_agree` marks
points whose views disagree (something in front of the point in some photos, or a
point on a boundary): look at those first.

## 7. Segment the cloud

```bash
substrata segment --classifier ~/classifiers/samoa_v1/point_classifier.pt
```

Samples query points over the cloud (one per 5 cm voxel, `--cell-size`), classifies
each from its `--views` (default 3) best photos, and gives every cloud point the
label of its nearest query point. With a point classifier each photo is classified
once as a dense probability map, and a query's prediction is the relevance-weighted
mean of its views' probabilities.

Writes `<id>.annos/<id>_seg_points.csv` (the query points as an annotation set, with
`pred_label`, `pred_conf`, `pred_n`, `pred_agreement` and the photo each was
classified from) and `<id>_seg_dec.ply` (the decimated cloud coloured by class).
`--tag NAME` keeps versions side by side; `--full-ply` also colours the full cloud.

With `voxel` sampling each query point stands for about the same area of surface, so
a label's share of the query points is its cover of the 3-D surface. Points no photo
saw are left out, so it is the cover of the observed surface.

**Check.** Open `<id>_seg_dec.ply` in CloudCompare, or the points set in
substrata-explore. Low `pred_agreement` flags boundaries, occlusions and ambiguous
surfaces. Labels you correct by hand in the points CSV win over `pred_label`: recolour
with `substrata segment --from-points <id>.annos/<id>_seg_points.csv`.

## 8. Curate the training points

Label noise limits a classifier more than its architecture. In the classifier
folder:

```bash
cd ~/classifiers/samoa_v1
substrata curate          # scores the training points: curate/points.csv
substrata curate --gui    # review them in the browser
substrata train --keep-map --yes
```

`curate` scores each point of the train split: `p_label` (the cross-validated
probability of its own label: low for likely label errors), `ring_agree` (how well
its label holds 2 feature cells around it: low on boundaries and small objects),
`knn_label` (share of the 10 most similar points elsewhere with the same label) and
`novelty`. Validation and test points are left out, so their labels are not tuned
towards the model. In the review grid you accept, relabel or exclude points. The
decisions go to `curation.jsonl` (append-only, keyed by photo file name and pixel),
and the next `substrata train` in the folder applies them. Without the GUI:
`--sheets` writes contact sheets, `--import CSV` reads decisions.

## 9. Other cameras

The classifier standardises its features with statistics of its training photos.
For photos of another camera (e.g. AUV frames over the diver plots), compute those
statistics from unlabelled photos of the same area:

```bash
substrata calibrate /data/auv/plot_frames --classifier point_classifier.pt --gsd-mm 1.21
```

Writes `point_classifier_calibrated.pt` (or `--output`). `--gsd-mm` is the photos'
ground resolution (camera distance / focal length, in mm per pixel). Use the
calibrated file to predict on that camera. The photos must show the same area as the
training photos, or the statistics of another habitat shift the classes too.

## Common problems

- **`train` finds no CSVs.** The default pattern is `*_slope_intercepts.csv` in the
  current folder. Use `--sets FILE` or `--csv-path DIR` (and a pattern).
- **Photos not found while training.** `cam_filepath` is an absolute path. Repair the
  plot with `substrata path-repair` (it also fixes `cam_filepath` in the sets).
- **"No bolded training labels".** No label reaches `--min-count`; lower it or name the
  classes with `--include-classes`.
- **Few points have a training view.** The plot is not stratified (occlusion fell back
  to the older test), or the main camera type is not the survey camera: check the
  camera types `crops` prints and pass `--camera-type`.
- **Predictions on another camera are poor.** Calibrate the classifier (step 9), or
  label points of that camera too (a `curate --mine` pool).
- **`predict` skips everything.** The points already carry predictions of this
  classifier file (`pred_model`); use `--redo`.
