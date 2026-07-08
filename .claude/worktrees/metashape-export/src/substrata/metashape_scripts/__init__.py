"""Standalone scripts run under Metashape's bundled Python interpreter.

Modules in this subpackage are executed by Metashape (``metashape.sh -r``) and
import only :mod:`Metashape` plus the standard library. They are deliberately
**not** star-imported by :mod:`substrata` (see ``substrata/__init__.py``), so
importing the ``substrata`` package in the conda environment never triggers an
``import Metashape``. The ``substrata metashape-export`` CLI command locates and
runs :mod:`substrata.metashape_scripts.export_project` via :mod:`subprocess`.
"""
