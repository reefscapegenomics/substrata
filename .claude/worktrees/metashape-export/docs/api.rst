API Reference
=============

This page documents the public modules of the substrata package. The pages are
generated automatically from the source docstrings, so they stay in sync with
the code.

A good entry point is :class:`substrata.initializer.ProjectInitializer`, the
central factory that loads a project directory and lazily builds the main domain
objects — :class:`~substrata.pointclouds.PointCloud`,
:class:`~substrata.cameras.Cameras`, and
:class:`~substrata.annotations.Annotations`. Because ``__init__.py``
star-imports every module, all public names are also reachable from the flat
``substrata.*`` namespace.

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

Terrain and ecology metrics, color calibration from ColorChecker targets, and
the crop-classification and segmentation tooling.

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

.. automodule:: substrata.segmentation
    :members:
    :undoc-members:
    :show-inheritance:

Visualization
-------------

Open3D/matplotlib rendering, composite-view and QC PDF reports, and fast 2-D
orthographic projections of point clouds.

.. automodule:: substrata.visualizations
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.ortho
    :members:
    :undoc-members:
    :show-inheritance:

Configuration and Utilities
---------------------------

Module-level constants (default radii/thresholds, scalebar targets, column
orders) and the shared logger.

.. automodule:: substrata.settings
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: substrata.logging
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
