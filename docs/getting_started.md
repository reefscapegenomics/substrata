# Getting Started

**substrata** turns the outputs of a structure-from-motion pipeline — dense
point clouds, camera poses, and image annotations — into scaled, georeferenced
models that you can measure and analyze. Work centers on a **project**: a single
directory holding one reconstruction, which substrata loads, scales, orients, and
measures as a unit (see {doc}`projects` for how to create and configure one).

## Two ways to use substrata

There are two ways to work with substrata, and you can mix them freely on the
same project:

- **As a command-line tool** — run `substrata <subcommand>` from inside a project
  directory for simplified, high-level access to the common workflows (decimate,
  orient, scalebars, views, colors, train, and more). This is the quickest way to
  process a project without writing any code. See {doc}`command_line_usage` for
  the full list of subcommands.
- **As a Python package** — `from substrata import *` gives you direct,
  programmatic access to `ProjectInitializer`, `PointCloud`, `Cameras`,
  `Annotations`, and the measurement and visualization functions. Use this in
  scripts or Jupyter notebooks when you need full control or want to build custom
  analyses. See the tutorials below and the {doc}`api`.

## Install

Install substrata with conda (it depends on native libraries such as Open3D,
OpenCV, and PyTorch). See {doc}`installation` for the full instructions:

```bash
conda env create -f environment.yml
conda activate substrata
python -m pip install -e .
```

## A first taste — command line

Create a project folder from a Metashape `.psx`, then run a couple of high-level
commands from inside it:

```bash
# Initialize a project folder from a Metashape .psx (see the Projects page)
substrata metashape-export --psx cur_sna_20m_20200303.psx

cd cur_sna_20m_20200303
substrata orient       # compute scale + orientation, write <id>.yaml
substrata scalebars    # QC PDF of the scalebar fit
substrata views        # composite-views PDF
```

Most commands auto-detect the project in the current directory, so you rarely
need to pass file paths. See {doc}`command_line_usage` for every subcommand and
its options.

## A first taste — Python

Load a configured project and reach its objects. `initialize()` applies the
stored scale and orientation automatically:

```python
from substrata import *

proj = ProjectInitializer(path="/data/cur_sna_20m_20200303")
proj.initialize()        # loads and applies stored scale/orientation

proj.pcd                 # PointCloud
proj.cams                # Cameras
proj.annotations         # Annotations
```

From here you can compute measurements, transfer annotations, or render QC
reports. The tutorial notebooks walk through complete workflows end to end.

## Next steps

- {doc}`projects` — set up and configure your own project (the on-disk layout,
  the YAML project file, and the full schema).
- {doc}`command_line_usage` — every command-line subcommand and its options.
- {doc}`notebooks/scaling_and_orientation` — scale and orient a point cloud from
  scalebar markers.
- {doc}`notebooks/transferring_annotations` — transfer annotations between
  overlapping reconstructions.
- {doc}`notebooks/measurements_tpi` — topographic position index (TPI) and
  terrain ruggedness index (TRI).
- {doc}`notebooks/measurements_benthic_fraction` — benthic fraction around
  annotations, using the crop classifier.
- {doc}`api` — the complete API reference.
