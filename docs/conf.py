import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'substrata'
copyright = '2025, Reefscape Genomics Lab'
author = 'Reefscape Genomics Lab'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",           # for Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",      # adds type hints in signature
]

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    # Heavy / non-tutorial notebooks kept out of the build (also gitignored).
    'notebooks/notebook_mcav_sint.ipynb',
    'notebooks/color_calibration.ipynb',
]

# Do not execute notebooks during doc builds (safer for RTD)
nb_execution_mode = "off"

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_mock_imports = [
    # Heavy/optional libs not guaranteed on RTD
    "open3d",
    "cv2",
    "sklearn",
    "scipy",
    "pandas",
    "matplotlib",
    "tqdm",
    "tqdm_joblib",
    "yaml",
    "exifread",
    "ffmpeg",
    "alphashape",
    "shapely",
    "joblib",
    "torch",
    "IPython",
    "PIL",
    "sam2",
    "psutil",
    "plotly",
    "fastai",
    "fpdf",
    # Submodules occasionally imported directly
    "mpl_toolkits.mplot3d",
]

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}