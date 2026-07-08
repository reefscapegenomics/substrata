# Installation

substrata relies on several heavy, native dependencies (Open3D, OpenCV,
PyTorch) that are best installed with conda. The recommended approach is to
create a conda environment from the provided `environment.yml` file and then
install substrata into it with pip.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/reefscapegenomics/substrata.git
   cd substrata
   ```

2. Create the conda environment (installs Python and all native
   dependencies from conda-forge):
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate substrata
   ```

4. Install the package in editable mode:
   ```bash
   python -m pip install -e .
   ```

That's it! You're ready to use substrata.

## Verifying the installation

Confirm the package imports and the `substrata` command-line entry point is on
your PATH:

```bash
python -c "import substrata"
substrata --help
```

To run the test suite (stdlib `unittest`, no extra dependencies):

```bash
python -m unittest discover -s tests -v
```

## Requirements

The `environment.yml` file pins Python 3.10 and pulls all dependencies from the
`conda-forge` channel, including:

- Open3D (point-cloud processing)
- OpenCV (image processing)
- PyTorch / torchvision and fastai (the crop classifier and ML features)
- NumPy, pandas, scikit-learn (data analysis)
- PyYAML, tqdm, ffmpeg, and other supporting libraries

A few pure-Python packages (e.g. `fpdf2`, `gspread`, `exifread`) are installed
via pip within the same environment. The only hard dependency declared in
`pyproject.toml` is NumPy; everything else comes from the conda environment, so
installing from `environment.yml` is strongly recommended over a bare
`pip install`.

For the complete list, see `environment.yml` in the repository root.

## Building the documentation (optional)

The docs are built with Sphinx and myst-nb. To build them locally:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

The rendered HTML is written to `docs/_build/html/`.
