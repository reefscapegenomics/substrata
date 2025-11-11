# Installation

substrata can be installed using conda and pip. The recommended approach is to create a conda environment from the provided `environment.yml` file.

## Quick Start

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate substrata
   ```

3. Install the package in editable mode:
   ```bash
   python -m pip install -e .
   ```

That's it! You're ready to use substrata.

## Requirements

The `environment.yml` file includes all necessary dependencies:
- Python 3.10
- Open3D (for point cloud processing)
- OpenCV (for image processing)
- PyTorch (for machine learning features)
- NumPy, Pandas, scikit-learn (for data analysis)
- And other supporting libraries

For a complete list of dependencies, see the `environment.yml` file in the repository root.

