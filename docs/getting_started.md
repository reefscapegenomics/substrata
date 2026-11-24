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

Put the Metashape `.psx` in a folder named after the project id, then run a
couple of high-level commands from inside it:

```bash
cd cur_sna_20m_20200303          # holds cur_sna_20m_20200303.psx
substrata metashape-export       # point cloud, cameras, markers, starter <id>.yaml
substrata scalebars    # QC PDF of the scalebar fit
substrata orient       # compute scale + orientation, write them to <id>.yaml
substrata views        # composite-views PDF
```

`metashape-export` writes into the current folder (or `-o DIR`), never into a new
subfolder. {doc}`tutorials/plot_workflow` goes through these steps and the checks
after each.

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
reports. The tutorials below walk through complete workflows end to end.

## Next steps

- {doc}`projects` — set up and configure your own project (the on-disk layout,
  the YAML project file, and the full schema).
- {doc}`tutorials/plot_workflow` — from a Metashape project to a scaled,
  oriented plot ready to measure, with the checks after each step.
- {doc}`tutorials/annotate_and_classify` — point intercepts, labelling, training
  a classifier, predicting and segmenting.
- {doc}`tutorials/closeups` — close-up (macro) photos of colonies: poses,
  annotations and one page per colony.
- {doc}`tutorials/measurements_in_python` — TPI/TRI, roughness, plane angles,
  vector dispersion, gap fraction and benthic fraction, and exactly what each
  computes.
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
