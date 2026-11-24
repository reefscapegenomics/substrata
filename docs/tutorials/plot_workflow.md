# From a Metashape project to a measurable plot

This tutorial takes one reconstruction from Agisoft Metashape to a plot that is
scaled to metres, levelled with +Z up, colour-corrected and ready to measure. All
steps run from the command line, inside the plot's folder. Each step says what it
reads, what it writes and how to check the result.

The examples use the plot `sam_sa3_40m_20250928` (site, location, depth, date).
Replace the id with your own.

| Step | Command | Writes (in the plot folder) |
| --- | --- | --- |
| 1 | `substrata metashape-export` | `<id>.ply`, `<id>_dec50M.ply`, `<id>.cams.xml`, `<id>.meta.json`, `<id>_markers.csv`, `<id>.yaml` |
| 2 | `substrata scalebars` | `<id>_scalebars.pdf` |
| 3 | `substrata orient` | `world_transform` and depth keys in `<id>.yaml`; `<id>_views.pdf`, `<id>_depth_residuals.pdf` |
| 4 | `substrata colors` | `color_correction` in `<id>.yaml`; `<id>_colorcal.pdf` |
| 5 | `substrata views` | `<id>_views.pdf` |
| 6 | `substrata stratify` | `<id>.ply` reordered in place, `<id>.strat.npz` |
| 7 | `substrata head` | nothing (prints) |
| (export) | `substrata decimate` | a decimated copy, e.g. for other software |

## The project folder

A plot is one folder, named after its id, and every file in it starts with that id.
substrata finds the files by name, or through the project file `<id>.yaml` when it
exists. Most commands take no file arguments: they work on the plot in the current
folder. {doc}`../projects` describes the layout and every YAML key.

Before step 1 the folder only needs the Metashape project:

```text
sam_sa3_40m_20250928/
└── sam_sa3_40m_20250928.psx      (and its .files folder)
```

The photos can stay where Metashape found them. Their paths are recorded in
`<id>.meta.json`. By convention they sit in folders next to the project files, one
per camera group: `<id>.photos/`, `<id>.auv/`, `<id>.closeup/`, `<id>.videoframes/`.
If you move them later, run `substrata path-repair`.

What the Metashape chunk needs for the later steps:

- a dense point cloud (for `.ply`);
- the scale-bar targets detected as markers, named as in `settings.RGL_SCALEBARS`
  (`target 3`, `target 4`, ...). `scalebars` and `orient` take the scale from them;
- a depth for each survey photo as the camera reference z (from a depth logger).
  `orient` levels the model by regressing these depths on the camera positions;
- for `colors`: the corner targets of the ColorChecker cards (`settings.RGL_COLOR_CALIBRATIONS`).

## 1. Export from Metashape

```bash
cd sam_sa3_40m_20250928
substrata metashape-export
```

Reads `<id>.psx` in the current folder (or `--psx PATH`). It runs an export script
under Metashape's own Python, so Metashape must be installed (`--metashape PATH`,
or the `METASHAPE_EXE` environment variable). Files are written into the current
folder, or into `-o OUTPUT_DIR`; never into a new subfolder.

Writes, all in Metashape's raw chunk coordinates:

- `<id>.ply`: the full point cloud;
- `<id>_dec50M.ply`: a random 50 M point subsample of it (seeded, so the same
  input gives the same file). Most commands load this copy;
- `<id>.cams.xml` and `<id>.meta.json`: camera calibrations, poses and photo paths;
- `<id>_markers.csv`: marker coordinates (`id,x,y,z,label`);
- `<id>.yaml`: a starter project file naming these files.

Existing files are skipped, so a re-run only fills in what is missing.
`--metadata-only` skips the point cloud (fast); `--overwrite` redoes everything.

**Check.** All six files exist. `<id>_markers.csv` lists the targets you expect.

## 2. Check the scale bars

```bash
substrata scalebars
```

Reads `<id>_markers.csv` and, for the figures only, the point cloud of the YAML.
Each scale bar is a pair of targets at a known distance (`settings.RGL_SCALEBARS`:
0.5 m bars and the 0.07 m / 0.20 m sides of the ColorChecker frames). Its scale
factor is the known length divided by the distance between the two markers in the
model. The plot's scale factor is the mean over all bars whose two targets were
found.

Writes `<id>_scalebars.pdf`: a summary page (number of bars, scale factor, variance
and standard error of the per-bar factors), then a top-down view of the cloud
around each target with the marker highlighted.

**Check.** Every target marker sits on its target in the figures. The spread of the
per-bar factors is small relative to the scale factor: compute
`sqrt(variance) / scale factor`, the coefficient of variation. A warning is printed
when the variance is above 0.005 (an absolute value, so it is only a coarse check).
A bar with a misplaced marker shows up as a target off-centre in its figure: fix
the marker in Metashape and export again (`metashape-export --metadata-only
--overwrite` rewrites the markers without the point cloud; it also rewrites
`<id>.cams.xml` and `<id>.meta.json`, so do this before any `camsync`).

This step does not change the project. `orient` computes the same scale factor
itself.

## 3. Scale and orient

```bash
substrata orient
```

Reads the YAML's point cloud, the cameras (positions and reference depths) and the
markers. It always starts from the raw chunk coordinates, so running it again
replaces the previous orientation instead of adding to it. It computes, in order:

1. **Scale**: the scale factor from the scale bars (step 2).
2. **Up vector and depth**: an ordinary least-squares fit of the camera depths on the
   camera positions, `depth ≈ depth_offset + up_vector · (x, y, z)`
   (`measurements.fit_depth_regression`). Depths are negative below the surface;
   positive depths are negated with a warning. The coefficient vector is the depth
   gradient, so it points up; the model is rotated so that it becomes +Z, and
   shifted so that z is depth in metres.
3. **Horizontal axes**: X along the principal axis of the cloud's footprint, then
   +Y turned upslope, then the cloud shifted to positive X and Y.

The result is one 4×4 `world_transform` (raw coordinates to metres), saved in
`<id>.yaml` with `scale_factor`, `up_vector`, `depth_offset` and `depth_per_unit`.
Only `world_transform` is applied when the project is loaded; the other keys are a
record. The YAML is patched in place: comments and other keys are kept.

Also writes `<id>_views.pdf` (composite views of the oriented cloud) and
`<id>_depth_residuals.pdf` (recorded against modelled depth per camera, and the
residuals).

Options:

- `--slope`: after orienting, rotate the cloud so that a plane fitted to the
  terrain is flat, and X runs along the footprint's principal axis. This is the
  frame `substrata intercepts --slope` samples in. Z is then height above the slope
  plane, not depth (an annotation's depth, `depth_in_m`, still comes from the
  depth model). Choose it when you want a slope-aligned grid and maps; keep in mind
  that every Z-based measurement (TPI, plane angles, gap fraction) is then relative
  to the slope plane.
- `--markers CSV`: fit the depth model to annotations with known depths instead of
  the cameras.
- `--use-chunk-transform [--north-up]`: for a georeferenced chunk, take scale and
  orientation from Metashape's chunk transform instead of fitting them.
- `--transform`: enter an extra 4×4 transform to apply after the fit.

**Check.** Open `<id>_depth_residuals.pdf`. The recorded and modelled camera depths
lie on a line, and the residuals have no trend with position. A trend or a cluster
of large residuals points to a depth logger out of sync with the photos, or to a
group of cameras placed wrongly by Metashape. In `<id>_views.pdf` the reef is level
(no tilt in the side views) and the shallow end is at the top of the top-down view.

## 4. Colour correction

```bash
substrata colors
```

Reads the point cloud (in the world frame, so run `orient` first) and the corner
markers of the ColorChecker cards. For each card the patches are sampled within
5 mm of their centres and 3 mm of the card plane. Each patch's colour is the median
over the cards, leaving out a card whose value lies more than 2.5 robust standard
deviations (1.4826 × MAD) from the cross-card median. An affine transform (3×3
matrix and offset) is then fitted from the measured to the reference colours, in
gamma-encoded sRGB 0-255.

Writes `<id>_colorcal.pdf` and the `color_correction` key in `<id>.yaml`, with the
fit's RMS residual (`fit_rmse`), the number of patches (`fit_patches`) and the colour
space (`fit_space`). No PLY is changed: loading the project applies the correction
in memory.

**Check.** In `<id>_colorcal.pdf`, the corrected patches match the reference
patches, and `fit_rmse` on the summary page is in line with your other plots. A
card that is shaded, covered or wrongly marked pulls the fit: leave it out with
`--exclude-index N` or `--exclude-name NAME` (e.g. `top-left`). `--no-save-yaml`
writes only the PDF.

## 5. Composite views

```bash
substrata views
```

Loads the project as every later command does (world transform and colour
correction applied) and writes `<id>_views.pdf`. This is the overall check of
steps 3 and 4: the reef is level, upslope is at the top, and the colours look
natural.

## 6. Stratify the full-resolution cloud

```bash
substrata stratify
```

Reorders `<id>.ply` in place so that its first N points are a uniform random
subsample and any region can be read without scanning the whole file. The points
themselves are unchanged (byte-identical, only reordered), so other software still
reads the file. Writes the index `<id>.strat.npz` next to it. The format is in
{doc}`../stratified_ply`.

The command prints the mode it will use and the RAM and disk space it needs. The
`memory` mode needs RAM for the whole cloud plus 8 bytes a point, and disk space for
one copy; the `stream` mode needs little RAM but space for two copies (`--work-dir`
or `--tmp-dir` can put one on another disk). The new file is written and
checksummed before it replaces the old one; `--keep-original` keeps the old one as
`backup_<name>.ply`.

Several tools need a stratified cloud: substrata-explore (`substrata explore`),
and the occlusion test that `crops`, `predict` and `images` use by default
(`--occlusion auto`).

## 7. Check the files

```bash
substrata head
```

Prints the first 5 points of `<id>.ply` (`-n N` for more). For a stratified PLY it
also prints the index summary: number of points and their properties, bounding box,
clusters, rounds, and the prefix sizes (how many leading points form each uniform
subsample). For a plot linked to a baseline timepoint (`baseline_id`), it says
whether the `world_transform` still follows the baseline.

At this point the folder holds:

```text
sam_sa3_40m_20250928/
├── sam_sa3_40m_20250928.yaml               # files, world_transform, color_correction
├── sam_sa3_40m_20250928.ply                # stratified full-resolution cloud
├── sam_sa3_40m_20250928.strat.npz          # its index
├── sam_sa3_40m_20250928_dec50M.ply         # decimated copy (loaded by most commands)
├── sam_sa3_40m_20250928.cams.xml
├── sam_sa3_40m_20250928.meta.json
├── sam_sa3_40m_20250928_markers.csv
├── sam_sa3_40m_20250928_scalebars.pdf
├── sam_sa3_40m_20250928_views.pdf
├── sam_sa3_40m_20250928_depth_residuals.pdf
└── sam_sa3_40m_20250928_colorcal.pdf
```

## Exporting a decimated copy

`metashape-export` already wrote `<id>_dec50M.ply`, which the YAML names and most
commands load. Run `decimate` yourself only to make a copy for other software, or
to remake that copy:

```bash
# A colour-corrected 10 M point copy for CloudCompare or Viscore
substrata decimate -c -n 10000000 --output sam_sa3_40m_20250928_dec10M_cc.ply
```

The sample is seeded (`--seed`, default 0, recorded in the header), so the same
input gives the same output. With `-c` the YAML's `color_correction` is applied to
the colours and the header is marked, so loading that file does not apply the
correction a second time. Results computed on a decimated cloud (orientation,
intercepts, neighbourhood metrics) depend on which points it holds; a decimated
copy made before November 2026 was unseeded and differs from a new one.

## Common problems

- **"No markers file found" or no scale factor.** The markers CSV is missing, or its
  labels do not match `settings.RGL_SCALEBARS`. For another scale-bar layout, compute
  the scale in Python (see {doc}`../notebooks/scaling_and_orientation`) and save it
  with `ProjectInitializer.save_config_to_yaml`.
- **Depths are positive.** They are negated with a warning; nothing to do.
- **The model is tilted or upside down in the views.** Check the camera depths in
  Metashape (reference z of the survey photos) and `<id>_depth_residuals.pdf`. Photos
  of another camera without depths are left out of the fit.
- **Photos not found** (crops, explore, camsync). The paths in `<id>.meta.json` are
  stale after a move: `substrata path-repair --dry-run`, then `substrata path-repair`.
- **`colors` fits badly.** One card is in shade or partly covered: exclude it.
  `colors` needs `orient` first, as it samples in metres.
- **Not enough disk space to stratify.** Use `--mode stream --work-dir /other/disk`,
  or `--out-dir` to write the stratified copy elsewhere.
- **Re-orienting after annotating.** Annotation CSVs store the raw coordinates
  (`orig_x`, `orig_y`, `orig_z`), so they follow a new `world_transform`. Crops and
  training views depend only on the raw coordinates and the cameras, so they stay
  valid. Intercept grids were drawn in the old frame, which their sidecar records.
