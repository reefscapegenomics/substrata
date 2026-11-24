# Measurements in Python

This tutorial computes terrain and habitat metrics around annotated points (e.g.
coral colonies) in Python: topographic position and ruggedness (TPI, TRI),
roughness (Ra, Rq), the orientation of the surface (plane angles), vector
dispersion, gap fraction and benthic fraction. For each metric a box says exactly
what is computed, in which units, from which reference, and what to watch for.

It starts from a plot prepared as in {doc}`plot_workflow` (scaled and oriented), with
an annotation set in `<id>.annos/`. The formulas below are those of
`substrata.measurements`; the {doc}`../api` has the full docstrings.

## Load the project

```python
import os
from substrata import *

plot_dir = "/data/samoa/sam_sa3/sam_sa3_40m/sam_sa3_40m_20250928"
proj = ProjectInitializer(path=plot_dir)   # reads <id>.yaml

print(proj.annotation_csvs())              # the sets in <id>.annos/
proj.use_annotations_file(os.path.join(
    plot_dir, "sam_sa3_40m_20250928.annos", "sam_sa3_40m_20250928_ann.csv"))

proj.initialize()   # loads proj.pcd, proj.cams, proj.annotations
```

`initialize()` loads the point cloud named in the YAML (normally the 50 M point
`<id>_dec50M.ply`), the cameras and the annotation set, and applies the
`world_transform`. Everything is then in the world frame: metres, +Z up. After a
plain `substrata orient`, Z is depth. After `orient --slope`, Z is height above the
plot's slope plane: every metric below that uses Z or "up" (TPI, TRI, plane angles,
gap fraction) is then relative to the slope, not to the horizontal.

`use_annotations_file` must come before `initialize()`. Without it, the set named by
the YAML's `annotations:` key is loaded.

## Choose annotations

`proj.annotations` is an `Annotations` container. The subset methods return a new
container that keeps the set's metadata:

```python
anns = proj.annotations.subset_by_label(["CS_ACR", "CS_POR"])
anns = anns.subset_by_depth_range(38, 42)     # depth in m, from the depth model
ann = proj.annotations["sam_sa3_40m_20250928_17"]   # one annotation by id

for ann in anns:
    print(ann.id, ann.label, ann.coords)      # coords: world frame, metres
```

## Neighbourhoods

Most metrics work on the points around an annotation, stored per annotation as
`ann.simple_pcd` (a `SimplePointCloud` with points, colours and normals).

```python
# A vertical cylinder of 0.5 m radius (XY distance only): for TPI / TRI
anns.get_point_cloud_by_radius(proj.pcd, radius=0.5, cylinder=True)

# A sphere of 0.15 m radius (the default): for roughness, plane angles, vector dispersion
anns.get_point_cloud_by_radius(proj.pcd, radius=0.15)
```

Use each shape only for its metrics. A sphere clips the TPI annulus on a slope (TPI
then returns NaN), and an unbounded cylinder on a steep slope takes in a long strip
of surface plus whatever lies above or below it: on the close-up targets of a 40 m
plot, a 0.15 m cylinder gave about three times the Ra of the sphere and a plane
elevation 38° steeper.

Each call replaces `simple_pcd`, so extract the neighbourhood a metric needs just
before measuring it. Radii are in metres. A point of the cloud exactly at the
annotation is left out.

**Per-colony radii.** With a surface area per colony (`SA_in_cm2`, from
`anns.load_measurements_from_csv(...)` or `get_mask_surface_area`),
`anns.get_radius_from_2D_surface_area()` sets `ann.radius = sqrt(SA / π)`. Then
`get_point_cloud_by_radius(proj.pcd, cylinder=True, radius_buffer=0.5)` takes each
colony's radius plus 0.5 m, and TPI / TRI and benthic fraction use `ann.radius` as
their inner radius.

**Full resolution.** The decimated cloud holds a few percent of the points. To read
the neighbourhoods from the full-resolution `<id>.ply` instead (one pass through the
file, never loaded whole):

```python
from substrata.frames import metres_per_unit

anns.load_undecimated_neighborhoods(
    proj.ply_full_path, radius=0.5, cylinder=True,
    scale_factor=metres_per_unit(proj.world_transform, proj.scale_factor))
```

`scale_factor` converts the radius from metres to the PLY's raw units; the
neighbourhoods are returned in the world frame.

**Point density.** All metrics below except TPI/TRI (weighted by area, see below)
weight each *point* equally, so they depend on point density. A 50 M point copy of a small plot is denser than one of a large plot.
Compare values computed at similar density: the
same decimation, or the same full-resolution source.

## Run a measurement

`measure_all` runs a metric for every annotation and stores the results in
`ann.measurements`:

```python
anns.get_point_cloud_by_radius(proj.pcd, radius=0.5, cylinder=True)
anns.measure_all("calc_tpi_and_tri", radius_inner=0.1, radius_outer=0.5)

anns.get_point_cloud_by_radius(proj.pcd, radius=0.15)
anns.measure_all("calc_roughness")
anns.measure_all("get_plane_angles")
anns.measure_all("get_vector_dispersion")

anns.measure_all("calc_gap_fraction", proj.pcd, max_radius=1.0)

for ann in anns:
    print(ann.id, ann.measurements["tpi_abs"], ann.measurements["Ra"])
anns.save(os.path.join(plot_dir, "sam_sa3_40m_20250928_measurements.csv"))
```

`save()` writes the set with one column per scalar measurement. Saving to a new file
leaves the annotation set itself as it was.

For QC images, pass `generate_images=True` (slow: several are rendered with a
headless browser; use a subset), then `ann.show_measurement_images()` or
`visualizations.save_measurement_visualizations_to_pdf(anns, "report.pdf",
orthomap=OrthoMap(proj.pcd, pixel_width=5000))`. A single annotation can be measured
with `ann.measure(calc_roughness)`.

## TPI and TRI

```python
anns.get_point_cloud_by_radius(proj.pcd, radius=0.5, cylinder=True)
anns.measure_all("calc_tpi_and_tri", radius_inner=0.1, radius_outer=0.5)
# or: annulus_width=0.2 (outer = inner + 0.2), center_z_from_inner=True,
#     raster_cell_size=None (weight per point), min_annulus_coverage=0.75
```

Keys: `tpi_abs`, `tpi_plane`, `std_annulus_z`, `std_annulus_plane`, `tri_abs`,
`tri_plane`.

```{admonition} What is computed: calc_tpi_and_tri
:class: note

**Neighbourhood.** The *annulus*: the points whose horizontal (XY) distance from the
focal point is between `radius_inner` and the outer radius (`radius_outer`, or
`radius_inner + annulus_width`; defaults 0.1 and 0.5 m). Z is unconstrained. The
inner radius keeps the focal object (the colony) out. The focal point is the
annotation; its height `z0` is its Z, or with `center_z_from_inner=True` the mean Z
of the points within `radius_inner`.

**TPI** (topographic position index; Weiss 2001), in metres, positive on crests and
negative in hollows:

- `tpi_abs = z0 - mean(z_i)` over the annulus points;
- `tpi_plane`: the signed perpendicular distance of the focal point from the plane
  fitted to the annulus points (total least squares, normal turned to +Z). It
  corrects for the slope of the habitat.

**TRI** (terrain ruggedness index; Wilson et al. 2007), in metres:

- `tri_abs = mean(|z_i - z0|)`. Wilson's formula compares a DEM cell with its
  immediate neighbours. Here the annulus is wide and excludes the focal object, so on
  a slope or a pedestal this also takes up the focal point's height offset: for a
  perfectly smooth annulus offset by H it gives about |H| = |TPI_abs|. It mixes
  position and ruggedness;
- `tri_plane = mean(|d_i|)`, with `d_i` the perpendicular distances of the annulus
  points from their fitted plane. Free of slope and offset: the recommended
  ruggedness measure. It is Ra (below) restricted to the annulus.

`std_annulus_z` is the standard deviation of the annulus Z (population SD);
`std_annulus_plane` that of the plane distances.

**Caveats.**

- *Annulus coverage.* The annulus is divided into 24 sectors × 4 equal-area rings;
  the coverage is the fraction of these cells that hold a point
  (`annulus_coverage`). Below `min_annulus_coverage` (default 0.75,
  `settings.TPI_MIN_ANNULUS_COVERAGE`) all six values are NaN, with a warning. This
  happens near the edge of the cloud, where a one-sided annulus on a slope gives a
  biased TPI (0.084 m on a smooth plane). It also happens when the neighbourhood was
  extracted with a radius smaller than the outer radius, or as a sphere. A very
  sparse cloud can read low too. `min_annulus_coverage=None` turns the check off.
- *Density.* The points of each 1 cm XY cell are replaced by their mean
  (`raster_cell_size`, default 0.01 m, `settings.TPI_RASTER_CELL_SIZE`), so every
  cell of the surface counts once rather than every point: densely photographed
  parts (often tops of structures) would otherwise weigh more. `raster_cell_size=None`
  weights per point, as before version 0.2.0 (TPI about 0.02 m different on a 40 m
  plot's colonies).
- The plane-based values need at least 3 annulus points, else NaN.
- *Frame.* Heights are world Z: depth after `orient`, height above the slope plane
  after `orient --slope`.
```

## Roughness: Ra and Rq

```python
anns.get_point_cloud_by_radius(proj.pcd, radius=0.15)
anns.measure_all("calc_roughness")      # keys: Ra, Rq
```

```{admonition} What is computed: calc_roughness
:class: note

A plane is fitted to all points of the neighbourhood (total least squares: through
the centroid, with the direction of least variance as its normal). With `d_i` the
perpendicular distance of point i from that plane:

- `Ra = mean(|d_i|)`, the arithmetical mean roughness;
- `Rq = sqrt(mean(d_i²))`, the root mean square roughness, more sensitive to large
  deviations.

Both in metres (the cloud's units). These are the surface-metrology definitions,
here on a 3-D neighbourhood rather than a profile. `get_dev_rugosity` (key
`dev_rug`) is the same value as Ra.

**Caveats.** The values grow with the neighbourhood size and depend on its shape
(sphere or cylinder): report the radius. A neighbourhood centred on a colony includes
the colony. Each point counts once, so point density matters. Ra is not the
"rugosity" of chain-and-tape methods (surface over planar length).
```

## Plane angles

```python
anns.get_point_cloud_by_radius(proj.pcd, radius=0.15)
anns.measure_all("get_plane_angles")
# keys: elevation, azimuth, theta, psi, plane_coeffs
```

```{admonition} What is computed: get_plane_angles
:class: note

The plane is the one of `calc_roughness`, with unit normal `n = (n_x, n_y, n_z)`.
The sign of the normal is chosen as follows. When the neighbourhood has point normals,
`n` is turned towards the mean point normal of the inliers: the points within 0.01 m
of the plane (`settings.DEFAULT_INLIER_RANGE`). Otherwise `n` is turned to +Z.

- `elevation = arccos(n_z)` in degrees: 0 for a surface facing up, 90 for a vertical
  one, 180 for one facing down (an overhang).
- `azimuth = atan2(n_y, n_x)` in degrees, 0-360, counter-clockwise from +X; None when
  the normal is nearly vertical. After `orient`, +X runs along the plot and +Y is
  upslope, so this is relative to the plot, not to north.
- `theta = atan2(n_x, n_z)` and `psi = atan2(-n_y, hypot(n_x, n_z))`: rotating +Z
  about +X by `psi`, then about +Y by `theta`, gives `n`.
- `plane_coeffs`: `[a, b, c, d]` with `a x + b y + c z + d = 0`.

**Caveats.** Without point normals the elevation is 0-90°: an overhang cannot be told
from the plane alone. With point normals the result relies on them being oriented
consistently (as stored in the PLY; they are carried through the world transform
correctly). Elevation is measured from world +Z, i.e. from the slope normal after
`orient --slope`.
```

## Vector dispersion

```python
anns.get_point_cloud_by_radius(proj.pcd, radius=0.15)
anns.measure_all("get_vector_dispersion")   # key: vector_disp
```

```{admonition} What is computed: get_vector_dispersion
:class: note

Fisher's spherical dispersion of the unit point normals (as used for reef
complexity by Young et al. 2017): `(N - R) / (N - 1)`, with N the number of normals
and R the length of their vector sum. 0 when all normals are parallel, 1 for normals
scattered uniformly, up to 2 for two opposite groups. Unitless.

**Caveats.** It needs one normal per point (a `TypeError` otherwise); the PLY must
have normals. The statistic is directional, not axial: the normals must be
consistently oriented (all facing out of the surface). Flipping half of a set of
identical normals gives 1.0. Each point counts once, so density matters, and it
grows with the neighbourhood size.
```

## Gap fraction

Gap fraction uses the whole cloud above each annotation, not its neighbourhood, so
pass the project's cloud:

```python
anns.measure_all("calc_gap_fraction", proj.pcd, max_radius=1.0)
# keys: gapF_raw, gapF_fill
```

```{admonition} What is computed: calc_gap_fraction
:class: note

The points above the annotation (`z > z0`, and within `max_radius` horizontally if
given) are projected onto a hemispherical image of `resolution` × `resolution`
pixels (default 200), in an equidistant projection: the distance from the image
centre is proportional to the zenith angle (0° at the centre, 90° at the edge of the
disk). Each point marks the one pixel it falls on as covered. On this binary image:

- `gapF_raw` = sky pixels / all pixels of the disk;
- `gapF_fill` = sky pixels connected (sharing an edge) to a seed / all pixels of the
  disk. The default seeds are the zenith and a point at 45° zenith towards -Y. Sky
  enclosed by cover (holes in a canopy, gaps under an overhang) does not count.

Both are fractions of sky over the whole disk (not over the benthos), 0-1.

**Caveats.** Pixels are equal steps of zenith angle, not equal solid angles: an
opaque canopy over zenith 0-45° gives 0.75, where the open fraction of the solid
angle is cos 45° = 0.71. Each point covers one pixel, so a sparse cloud or a fine
`resolution` leaves holes in solid surfaces that read as sky: `gapF_raw` rises with
lower density. Compare values only at the same resolution and similar density;
`gapF_fill` is less sensitive to such holes. "Above" is world +Z (the slope normal
after `orient --slope`). A seed that falls on cover is skipped with a warning.
```

## Benthic fraction

The share of a benthic class (e.g. sand) in an annulus around each colony, from a
trained classifier (see {doc}`annotate_and_classify`):

```python
clf = classification.get_image_classifier(
    "/data/classifiers/samoa_v1/point_classifier.pt")
anns.measure_all(calc_benthic_fraction, proj.pcd, proj.cams, clf, "SU",
                 annulus_width=0.2, sample_spacing=0.05)
# keys: benthic_fraction, benthic_fraction_interacting, benthic_target_class
```

```{admonition} What is computed: calc_benthic_fraction
:class: note

A regular XY grid of `sample_spacing` (default 0.05 m) is laid over the annulus
around the annotation (the same annulus as TPI; inner radius `ann.radius` if set,
else 0.1 m). Each grid position becomes a surface point (the z-intercept within 0.02 m
horizontally). Each surface point gets its best camera view (one, after an occlusion
test by reprojection), and the classifier labels that view.

- `fraction = n_target / n_classified`, or with `weight_by_probability=True` the mean
  probability of the target class over the classified samples.
- `fraction_interacting = Σ w·t / n_classified`. `t` is 1 for the target class (or
  its probability), and `w` is a height weight: 1 at or above the colony base (the
  10th percentile of Z within the inner radius), falling linearly to 0 at
  `base_falloff_depth` (default 0.20 m) below it. It counts the target class that is
  level with or above the colony base.

The grid is planar, so this is a planar (top-down) cover.

**Caveats.** Samples without a surface point (off the cloud, e.g. near its edge),
without a camera view, or not classified are left out of the denominator. The
result's `n_samples`, `n_intercepts`, `n_matched` and `n_classified` say how many
remained. There is no annulus coverage check as for TPI. Classifier errors pass
straight into the fraction. Accepts a crop classifier (`.pkl`) or a point classifier
(`.pt`).
```

## Common problems

- **All TPI values are NaN.** The neighbourhood is smaller than the outer radius, or
  was extracted as a sphere: use `cylinder=True` with a radius of at least the outer
  radius. Near the cloud's edge NaN is intended.
- **`get_vector_dispersion` raises `TypeError`.** The cloud has no normals: the PLY
  was written without them (`substrata head` lists its properties: `nx`, `ny`, `nz`).
- **Gap fraction of almost 1 everywhere.** `max_radius` is too small, or the
  annotations lie above the surface.
- **Values differ between plots for no visible reason.** Check the point density of
  the clouds the neighbourhoods came from, and the frame (`orient` or
  `orient --slope`).
- **Many annotations take long.** `measure_all` uses 4 threads by default
  (`n_jobs`); TPI and benthic fraction run one annotation at a time. Leave
  `generate_images` off for batch runs.
