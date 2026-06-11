substrata Documentation
=======================

Welcome to the substrata documentation!

substrata is a Python package for 3-D geometry and point-cloud handling, providing
utilities for working with point clouds, camera calibration, annotations, and
geometric transformations.

Installation
------------

Get started by installing substrata. See :doc:`installation` for detailed instructions.

Tutorials
---------

If you're new to substrata or want to quickly orient and scale a point cloud, start here:

* :doc:`notebooks/scaling_and_orientation` - End-to-end scaling and orientation workflow
* :doc:`notebooks/transferring_annotations` - Transfer annotations between datasets/projects
* :doc:`notebooks/measurements_tpi_tri` - Topographic position index (TPI) and terrain ruggedness index (TRI)

You can also drive common workflows from the terminal; see :doc:`command_line_usage`.

API Documentation
-----------------

The complete API reference is available in the :doc:`api` section, which includes
documentation for all modules, classes, and functions.

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
   notebooks/measurements_tpi_tri

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
