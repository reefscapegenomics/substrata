substrata Documentation
=======================

**substrata** is a Python package for 3-D geometry and point-cloud handling,
built for coral-reef photogrammetry workflows. It turns the outputs of a
structure-from-motion pipeline — dense point clouds, camera poses, and image
annotations — into scaled, georeferenced models that you can measure and
analyze.

A typical workflow starts from a *project*: a directory holding a single
reconstruction (point cloud, camera calibration, scalebar markers, and
annotations) that substrata loads, scales, orients, and measures as one unit.
From there you can derive terrain and ecology metrics, transfer annotations
between overlapping reconstructions, and generate QC reports.

What you can do with substrata
------------------------------

* **Point clouds** — load, decimate, and stream very large PLY files without
  holding them in memory, backed by ``open3d``.
* **Cameras** — read Metashape camera poses and sensors, match images, extract
  video frames, and time-sync cameras with depth logs.
* **Annotations** — work with CSV-backed point labels and scalebars. Manage
  labeled features (e.g. individual corals) or generate random and stratified
  point annotations for point-intercept sampling, then subset and filter them
  by label or class. Move freely between 2-D and 3-D: project 3-D annotations
  into the camera images that see them, and reconstruct 3-D coordinates from
  2-D image annotations by ray-casting onto the point cloud — the basis for
  transferring annotations between overlapping reconstructions.
* **Scaling & orientation** — recover real-world scale from scalebars and orient
  a reconstruction into a world coordinate frame.
* **Measurements** — compute terrain and ecology metrics such as TPI/TRI,
  rugosity, fractal dimension, surface area, gap fraction, and depth regression.
* **Visualization** — render composite views and QC report PDFs with ``open3d``
  and ``matplotlib``.

Installation
------------

Get started by installing substrata. See :doc:`installation` for detailed
instructions, including the conda environment used for the heavy dependencies
(``open3d``, ``opencv``, ``pytorch``).

Tutorials
---------

If you're new to substrata, these notebook tutorials walk through complete
workflows end to end:

* :doc:`notebooks/scaling_and_orientation` — scale and orient a point cloud
  from scalebar markers
* :doc:`notebooks/transferring_annotations` — transfer annotations between
  datasets/projects
* :doc:`notebooks/measurements_tpi` — topographic position index (TPI) and
  terrain ruggedness index (TRI)
* :doc:`notebooks/measurements_benthic_fraction` — benthic fraction: cover of a
  target benthic class around annotations, using the crop classifier

Many of these workflows can also be driven from the terminal; see
:doc:`command_line_usage`.

API Documentation
-----------------

The complete API reference is available in the :doc:`api` section, which
documents all modules, classes, and functions.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   command_line_usage

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks/scaling_and_orientation
   notebooks/transferring_annotations
   notebooks/measurements_tpi
   notebooks/measurements_benthic_fraction

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
