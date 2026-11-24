API Reference
=============

This page documents the public modules of the substrata package. The pages are
generated automatically from the source docstrings, so they stay in sync with
the code.

A good entry point is :class:`substrata.initializer.ProjectInitializer`, the
central factory that loads a project directory and lazily builds the main domain
objects — :class:`~substrata.pointclouds.PointCloud`,
:class:`~substrata.cameras.Cameras`, and
:class:`~substrata.annotations.Annotations`. ``from substrata import *`` gives the
public names of the core modules in one flat namespace (annotations, sidecars,
color_calibration, pathrepair, cameras, pointclouds, stratified, initializer,
measurements, visualizations, geom, settings, classification, pointfeatures,
ortho, animations, segmentation, imageadjust and annotationpages; they are imported
on first use). Import the other modules by name, e.g.
``from substrata import imagesync``.

Core Modules
------------

Primitives and the core domain objects: 4×4 transforms and vectors, the point
cloud and camera wrappers, annotations/scalebars, the FireFish depth sensor, and
the project factory that ties them together.

.. automodule:: substrata.geom
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.pointclouds
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.cameras
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.annotations
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.firefish
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.initializer
    :members:
    :undoc-members:
    :show-inheritance:

Analysis and Processing
-----------------------

Terrain and ecology metrics, color calibration from ColorChecker targets, the
classification (crop and point-feature classifiers) and segmentation tooling,
training-data curation (``substrata curate``), and predictions from one camera
group (``substrata predict --camera-group``).

.. automodule:: substrata.measurements
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.color_calibration
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.classification
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.pointfeatures
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.segmentation
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.curation
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.grouppredict
    :members:
    :undoc-members:
    :show-inheritance:

Project Files and Point-Cloud Storage
-------------------------------------

Annotation-set sidecar YAMLs, image sets (photos without a 3-D model), repair of
stale file and photo paths (``substrata path-repair``), and the stratified PLY
format with its line-of-sight occlusion test (``substrata stratify``).

.. automodule:: substrata.sidecars
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.imagesets
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.pathrepair
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.stratified
    :members:
    :undoc-members:
    :show-inheritance:

Camera Poses and Video
----------------------

Poses for photos Metashape could not align (close-ups and unaligned video frames,
``substrata camsync --match-images``), the move of stored training views to the
corrected lens model (``substrata reproject-views``), and videos placed on a plot
with their camera poses (``substrata track-video``).

.. automodule:: substrata.imagesync
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.framepose
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.frameretrieval
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.lensfix
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.videotrack
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.videopose
    :members:
    :undoc-members:
    :show-inheritance:

Visualization
-------------

Plotly/matplotlib figures, composite-view and QC PDF reports, fast 2-D
orthographic projections of point clouds, animated versions of those
orthographic maps, one page per annotation (``substrata images --layout pages``),
and display contrast for photos. The figures are in the ``substrata.visualizations``
package; import its front module (``from substrata import visualizations``), which
re-exports every figure function, rather than the submodules below.

.. automodule:: substrata.visualizations

.. automodule:: substrata.visualizations.clouds
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.colorcards
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.photos
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.depth
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.terrain
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.gapfraction
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.benthic
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.sampling
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.reports
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.runreports
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.visualizations.alignment
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.ortho
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.animations
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.annotationpages
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.imageadjust
    :members:
    :undoc-members:
    :show-inheritance:

Configuration and Utilities
---------------------------

Module-level constants (default radii/thresholds, scalebar targets, column
orders), the shared logger, helpers for 4×4 similarity transforms, and safe file
writing (atomic replacement and per-path locks).

.. automodule:: substrata.settings
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.logging
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.frames
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.fileio
    :members:
    :undoc-members:
    :show-inheritance:

Command Line Interface
----------------------

The ``substrata`` CLI entry point and its subcommand handlers. See
:doc:`command_line_usage` for usage-oriented documentation.

.. automodule:: substrata.cli
    :members:
    :undoc-members:
    :show-inheritance:
