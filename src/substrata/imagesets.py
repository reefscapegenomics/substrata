"""Image sets: photos without a 3-D model, laid out like a plot.

::

    <project>/.../<id>/                 a folder named like a plot
        <id>.yaml                       type: images (what marks it), photos_path, camera, gsd_mm, classes (optional)
                                        camera: a name for the photos' camera, when it is not the one the classifier
                                        was trained on (e.g. auv): their points then train as that camera's domain
                                        (standardised with its own feature statistics, as curate pools)
        <id>.photos/                    the photos (subfolders allowed), or photos_path elsewhere
        <id>.annos/<id>[_<tag>]_points.csv   points on the photos (substrata sample-points), with their sidecars

A folder named like a plot with ``<id>.photos/`` (or ``<id>.annos/`` with a ``*_points.csv``) and neither ``<id>.yaml``
nor ``<id>.ply`` is an image set too, with the defaults (no camera, gsd_mm or classes): ``is_image_set``.
``substrata sample-points`` writes its YAML; nothing else does.

A points CSV holds each point's photo and pixel (``cam_filepath``, ``cam_x``, ``cam_y``); the 3-D coordinate columns
are empty. Everything that works on a point's training view (thumbnails, labelling, predicting, grouping, training)
works on these points the same way.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

TYPE = "images"


def image_set_yaml(folder: str) -> Optional[str]:
    """``<folder>/<name>.yaml`` if it marks an image set (``type: images``), else None."""
    folder = os.path.abspath(folder)
    y = os.path.join(folder, f"{os.path.basename(folder)}.yaml")
    try:
        with open(y) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    return y if isinstance(data, dict) and data.get("type") == TYPE else None


def _unmarked_image_set(folder: str) -> bool:
    """No ``<id>.yaml`` or ``<id>.ply`` (not a plot), and ``<id>.photos/`` or a ``<id>.annos/*_points.csv``."""
    sid = os.path.basename(folder)
    if any(os.path.exists(os.path.join(folder, sid + ext)) for ext in (".yaml", ".ply")):
        return False
    if os.path.isdir(os.path.join(folder, f"{sid}.photos")):
        return True
    try:
        return any(e.name.endswith("_points.csv") for e in os.scandir(os.path.join(folder, f"{sid}.annos")))
    except OSError:
        return False


def is_image_set(folder: str) -> bool:
    """Whether ``folder`` is an image set: marked by its YAML (``type: images``), or unmarked (see the module)."""
    folder = os.path.abspath(folder)
    return image_set_yaml(folder) is not None or _unmarked_image_set(folder)


def read_image_set(folder_or_yaml: str) -> Dict[str, Any]:
    """An image set's settings: ``id``, ``folder``, ``yaml`` (None if unmarked), ``photos`` (absolute), ``annos``
    (``<id>.annos``), ``camera``, ``gsd_mm`` and ``classes`` (absolute, or None). Raises ValueError if it is not an
    image set."""
    p = os.path.abspath(folder_or_yaml)
    folder = os.path.dirname(p) if p.endswith((".yaml", ".yml")) else p
    y = image_set_yaml(folder)
    if y is None and not _unmarked_image_set(folder):
        raise ValueError(f"{folder} is not an image set (no {os.path.basename(folder)}.yaml with type: images, "
                         f"and no {os.path.basename(folder)}.photos/ folder)")
    data = {}
    if y is not None:
        with open(y) as f:
            data = yaml.safe_load(f) or {}
    sid = os.path.basename(folder)

    def path(v):
        v = os.path.expanduser(str(v)) if v else None
        return None if not v else v if os.path.isabs(v) else os.path.normpath(os.path.join(folder, v))
    return dict(id=sid, folder=folder, yaml=y, photos=path(data.get("photos_path")) or os.path.join(folder, f"{sid}.photos"),
                annos=os.path.join(folder, f"{sid}.annos"), camera=data.get("camera"),
                gsd_mm=float(data["gsd_mm"]) if data.get("gsd_mm") else None, classes=path(data.get("classes")))


def write_image_set_yaml(folder: str, **extra: Any) -> str:
    """Mark ``folder`` as an image set: ``<id>.yaml`` with ``type: images`` (and ``extra``); kept if it exists."""
    folder = os.path.abspath(folder)
    y = os.path.join(folder, f"{os.path.basename(folder)}.yaml")
    if not os.path.isfile(y):
        with open(y, "w") as f:
            yaml.safe_dump(dict(type=TYPE, **{k: v for k, v in extra.items() if v is not None}), f, sort_keys=False)
    return y
