# Getting Started

Most substrata workflows start from a **project**: a single directory that holds
one reconstruction — its point cloud, camera poses, scalebar markers, and
annotations — together with the scale and orientation needed to turn it into a
measurable, real-world model. The {class}`~substrata.initializer.ProjectInitializer`
class is the factory that ties these pieces together. Both the command-line
tools and the tutorial notebooks start from it.

This page shows the two ways to set up a project:

1. **Build it in Python, then save a YAML** — load the point cloud, cameras, and
   annotations, compute scale and orientation, and write out a project file you
   can reuse.
2. **Write the YAML by hand** — point at files you already have and describe the
   project declaratively. This section documents the minimum keys, the defaults,
   and the full schema.

Either way, the end result is the same: a `<project>.yaml` file that lets you
reload the whole project, fully scaled and oriented, with a single call.

## What a project looks like on disk

A project lives in its own directory. By convention the directory is named
`site_location_depth_date` (e.g. `cur_sna_20m_20200303`), and that name is the
project **id**. A typical layout:

```text
cur_sna_20m_20200303/
├── cur_sna_20m_20200303.yaml          # project file (optional but recommended)
├── cur_sna_20m_20200303.ply           # dense point cloud
├── cur_sna_20m_20200303_dec50M.ply    # decimated copy (50M points)
├── cur_sna_20m_20200303.cams.xml      # Metashape camera poses
├── cur_sna_20m_20200303.meta.json     # camera metadata (paths/centers/transforms)
├── cur_sna_20m_20200303_markers.csv   # scalebar target coordinates
├── cur_sna_20m_20200303_annotations.csv
└── classes.csv
```

When you open a project by directory, `ProjectInitializer` first looks for a
`<dirname>.yaml` file inside it. If that file exists, it is used. If not, the
initializer falls back to **default naming conventions** (the filenames above)
and only registers the files it actually finds.

## Avenue 1 — build the project in Python, then save the YAML

Use this when you have raw reconstruction outputs and still need to compute scale
and orientation. You load everything, let substrata derive the scale factor (from
scalebar markers) and the up vector / depth offset (from camera depths), and then
persist the result so you never have to recompute it.

```python
from substrata import *

# Point at the project directory. With no <dirname>.yaml present, the
# initializer registers files by the default naming conventions above.
proj = ProjectInitializer(
    path="/data/cur_sna_20m_20200303",
    max_points=50_000_000,   # optional cap on points loaded from the PLY
)

# Instantiate PointCloud, Cameras, markers and Annotations from the registered
# files. Pass apply_transform=False so nothing is transformed before we have
# computed the orientation.
proj.initialize(apply_transform=False)

# Compute scale (from scalebar markers) and orientation (up vector + depth
# offset, regressed from camera depths), then apply the combined transform to
# the point cloud, cameras, markers and annotations. plot=True shows the
# regression fit as a quick QC check.
proj.scale_and_orient(plot=True)

# Persist everything to a project file. The baked-in world_transform plus the
# scale_factor / up_vector / depth_offset are all written out.
proj.save_config_to_yaml(
    "/data/cur_sna_20m_20200303/cur_sna_20m_20200303.yaml"
)
```

A few things worth knowing:

- **`scale_and_orient()` needs markers and cameras.** Scale is computed from
  scalebar targets in the markers file (using the built-in RGL scalebar target
  definitions in {mod}`substrata.settings`), and the up vector is regressed from
  per-camera measured depths (`depth_sensor_m`). If your scalebars don't follow
  the RGL target layout, compute scale manually instead — see the
  {doc}`notebooks/scaling_and_orientation` tutorial — and set
  `proj.scale_factor` / `proj.world_transform` yourself before saving.
- **You can determine the up vector from annotation depths instead of cameras**
  by passing `depth_markers=` to `scale_and_orient()`.
- **Pass an explicit path to `save_config_to_yaml()`** when you opened the
  project by directory. In that case there is no existing YAML path to default
  to, so the filepath argument is required.
- **What gets saved is the cumulative `world_transform`.** On reload, only the
  4×4 `world_transform` is re-applied; the `scale_factor`, `up_vector`,
  `depth_offset`, and `depth_per_unit` are stored as read-only metadata.

Once the YAML exists, reloading the fully scaled, oriented project is a
two-liner (see Avenue 2 below).

## Avenue 2 — write the YAML by hand

Use this when the files already exist (and, if needed, scale/orientation are
already known) and you just want to describe the project declaratively. The
project file is plain YAML.

### Minimum project file

The only thing substrata strictly needs to do useful work is a point cloud.
Relative filenames are resolved against `path`, so the smallest practical file
is:

```yaml
path: "/data/cur_sna_20m_20200303/"
ply: "cur_sna_20m_20200303_dec50M.ply"
```

With this, `proj.initialize()` loads the point cloud and nothing else. Add the
key for each component you want loaded — cameras need **both** `cams_xml` and
`cams_meta_json`; markers and annotations each need their own key.

> **Tip:** filenames may be either relative (resolved against `path`) or
> absolute. If you use relative filenames you must set `path`. If you give
> absolute paths for every file, `path` is optional.

### Reload a fully configured project

A complete project file lets you reload everything — scaled and oriented — in
two lines. Because the file is named `<dirname>.yaml` and lives inside the
project directory, you can open it either by its path or by the directory:

```python
from substrata import *

# By explicit YAML path:
proj = ProjectInitializer(yaml="/data/cur_sna_20m_20200303/cur_sna_20m_20200303.yaml")

# ...or by directory (auto-detects <dirname>.yaml inside it):
proj = ProjectInitializer(path="/data/cur_sna_20m_20200303")

# Loads PointCloud / Cameras / markers / Annotations and applies the stored
# world_transform (and colour correction, if present) automatically.
proj.initialize()
```

### Full schema reference

Every key is optional except as noted under "Minimum" above. Keys you omit leave
the corresponding object unloaded (or fall back to the default shown).

| Key | Type | Purpose / default |
| --- | --- | --- |
| `path` | str | Base directory; relative filenames are resolved against it. |
| `id` | str | Project id (used to name outputs). Defaults to the directory name when opened by `path`. |
| `ply` | str | Point-cloud PLY file. Legacy alias: `pcd`. |
| `cams_xml` | str | Metashape camera poses (`chunk.exportCameras(...)`). Loaded only together with `cams_meta_json`. |
| `cams_meta_json` | str | Camera metadata (paths / centers / transforms) from the Metashape API. |
| `markers` | str | Scalebar target coordinates CSV. |
| `annotations` | str | Point-label annotations CSV. |
| `annotations_last_highest_id` | int | Highest annotation id issued so far (for assigning new ids). |
| `classes` | str | Class-definition CSV. |
| `classifier` | str | Trained crop-classifier file. |
| `photos_path` | str | Directory of source photos. |
| `cropped_path` | str | Directory of cropped image chips. |
| `crop_size` | int | Crop size in pixels. |
| `thumbnails_path` | str | Directory of thumbnails. |
| `scale_factor` | float | Real-world scale factor (read-only metadata). |
| `up_vector` | `[x, y, z]` | Up direction in cloud coordinates (read-only metadata). |
| `depth_offset` | float | Offset aligning the z-axis with depth (read-only metadata). |
| `depth_per_unit` | float | Vertical scale (depth units per model unit; read-only metadata). |
| `world_transform` | 4×4 list | Cumulative transform applied on `initialize()`. Defaults to the identity matrix. |
| `color_correction` | mapping | `matrix` (3×3 or 3×4) and `offset` (length-3) arrays applied to point-cloud colours on load. |

Defaults when opening by `path` with **no** YAML present (filenames are derived
from the directory `id`; only files that exist are registered):

| Component | Default filename |
| --- | --- |
| `ply` | `<id>_dec50M.ply`, else `<id>.ply` |
| `cams_xml` | `<id>.cams.xml` |
| `cams_meta_json` | `<id>.meta.json` |
| `markers` | `<id>_markers.csv` |
| `annotations` | `<id>_annotations.csv` |
| `photos_path` | `<id>.photos/` |
| `cropped_path` | `<id>.cropped/` |
| `thumbnails_path` | `<id>.thumbnails/` |
| `classes` | `classes.csv` |

### A complete example

```yaml
path: "/data/cur_sna_20m_20200303/"
id: "cur_sna_20m_20200303"
ply: "cur_sna_20m_20200303_dec50M.ply"
cams_xml: "cur_sna_20m_20200303.cams.xml"
cams_meta_json: "cur_sna_20m_20200303.meta.json"
markers: "cur_sna_20m_20200303_markers.csv"
annotations: "cur_sna_20m_20200303_annotations.csv"
annotations_last_highest_id: 620
classes: "classes.csv"
scale_factor: 1.0
up_vector: [0.0, 0.0, 1.0]
depth_offset: 0.0
depth_per_unit: 1.0
world_transform:
- [-3.75205861e-04, -2.11005752e-01, 3.98466962e-02, 1.62435295e+01]
- [1.75381757e-01, -2.32932058e-02, -1.21696317e-01, 4.42951957e+00]
- [1.23904908e-01, 3.23315100e-02, 1.72376260e-01, 0.00000000e+00]
- [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
```

## Moving a project between machines

Project files store paths, which won't match after copying a project to a
different machine or directory. To rebase every registered path onto a new base
directory, pass `local=True` (rebases onto the current working directory) or call
`proj.reset_paths_to_local("/new/base/dir")` and re-save:

```python
proj = ProjectInitializer(yaml="cur_sna_20m_20200303.yaml", local=True)
```

## Next steps

- {doc}`notebooks/scaling_and_orientation` — compute scale and orientation step
  by step, including custom (non-RGL) scalebar layouts.
- {doc}`command_line_usage` — drive the same workflows from the terminal; most
  CLI subcommands auto-detect the project in the current directory via
  `ProjectInitializer`.
- {doc}`notebooks/transferring_annotations` — move annotations between
  overlapping reconstructions.
