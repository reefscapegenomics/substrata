"""Optional per-annotation-set metadata ("sidecar") YAML files.

An annotations/intercepts CSV may have a sidecar named after the full CSV filename
(``x_ann.csv`` -> ``x_ann.csv.yaml``), so it never clashes with the project's
``<id>.yaml``. A bare CSV without a sidecar stays fully supported. The sidecar
holds what belongs to the annotation set rather than to the plot, and only what
cannot be derived from the CSV:

    type: intercepts               # annotations | intercepts
    plot_id: cur_sna_20m_20200303
    classes: ../classes.csv        # paths are relative to the sidecar's folder
    world_transform: [[...]]       # plot transform used to generate/annotate
    last_highest_id: 620           # id high-water mark; deleted ids are not reused
    cropped_path: crops/           # optional, default <csv stem>.cropped
    crop_size: 800                 # optional
    classifier: model.pkl          # optional
    intercepts:                    # only for type: intercepts
      mode: slope                  # slope | topdown
      grid_transform: [[...]]      # world -> grid frame (slope alignment)
      grid_bbox: [[x0, y0], [x1, y1]]  # grid frame
      cell_size: 0.5
      box_length: 5.0
      box_width: 5.0
      search_radius: 0.01
      position: auto               # or [x, y]
      slope_elevation_deg: 23.4
      bbox_orig: [[x, y, z], ...]  # derived: grid_bbox corners in orig coords

The grid frame is ``grid_transform @ world_transform @ orig``.
"""

from __future__ import annotations

# Standard Library
import os
from typing import Any, Dict

# Third-Party Libraries
import numpy as np
import yaml

# Local Modules
from substrata import settings

# Sidecar keys that hold file/folder paths (stored relative to the sidecar)
SIDECAR_PATH_KEYS = ("classes", "cropped_path", "classifier")


def annotation_sidecar_path(csv_path: str) -> str:
    """Return the sidecar YAML path of an annotations CSV (``<csv>.yaml``)."""
    return csv_path + settings.ANNOTATION_SIDECAR_SUFFIX


def read_annotation_sidecar(csv_path: str) -> Dict[str, Any]:
    """Read the sidecar YAML of an annotations CSV.

    Path keys (``classes``, ``cropped_path``, ``classifier``) are resolved
    against the sidecar's folder and returned as absolute paths.

    Args:
        csv_path: Path to the annotations CSV.

    Returns:
        The sidecar contents, or an empty dict when there is no sidecar.
    """
    sidecar = annotation_sidecar_path(csv_path)
    if not os.path.isfile(sidecar):
        return {}
    with open(sidecar, "r") as f:
        config = yaml.safe_load(f) or {}
    folder = os.path.dirname(os.path.abspath(sidecar))
    for key in SIDECAR_PATH_KEYS:
        value = config.get(key)
        if value:
            path = os.path.expanduser(str(value))
            config[key] = os.path.normpath(os.path.join(folder, path))
    return config


def write_annotation_sidecar(
    csv_path: str, payload: Dict[str, Any], update: bool = True
) -> str:
    """Write (or update) the sidecar YAML of an annotations CSV.

    Path keys are stored relative to the sidecar's folder where possible, so the
    project folder can be moved; relative input paths are taken relative to the
    current directory. Numpy arrays/scalars are converted to plain YAML, and
    keys set to None are left out (or removed, when updating).

    Args:
        csv_path: Path to the annotations CSV.
        payload: Keys to write.
        update: If True, merge into an existing sidecar (keeping keys that
            ``payload`` does not set); otherwise replace it.

    Returns:
        The sidecar path.
    """
    sidecar = annotation_sidecar_path(csv_path)
    folder = os.path.dirname(os.path.abspath(sidecar))
    config = read_annotation_sidecar(csv_path) if update else {}
    config.update(payload)
    for key in SIDECAR_PATH_KEYS:
        value = config.get(key)
        if value:
            path = os.path.abspath(os.path.expanduser(str(value)))
            try:
                config[key] = os.path.relpath(path, folder)
            except ValueError:  # e.g. another drive on Windows
                config[key] = path
    config = {k: _plain(v) for k, v in config.items() if v is not None}
    with open(sidecar, "w") as f:
        yaml.dump(config, f, Dumper=_SidecarDumper, sort_keys=False)
    return sidecar


def _plain(value: Any) -> Any:
    """Convert numpy arrays/scalars (also nested) to plain Python values."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class _SidecarDumper(yaml.SafeDumper):
    """Write lists of scalars in flow style ([x, y, z]), matrices as rows."""


def _represent_list(dumper, data):
    flow = all(not isinstance(v, (list, dict)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_SidecarDumper.add_representer(list, _represent_list)
