"""Image sets: photos without a 3-D model, laid out like a plot.

::

    <project>/.../<id>/                 a folder named like a plot
        <id>.yaml                       type: images (what marks it), photos_path, camera, gsd_mm, classes (optional)
                                        camera: a name for the photos' camera, when it is not the one the classifier
                                        was trained on (e.g. auv): their points then train as that camera's domain
                                        (standardised with its own feature statistics, as curate pools)
        <id>.photos/                    the photos (subfolders allowed), or photos_path elsewhere
        <id>.annos/<id>[_<tag>]_points.csv   points on the photos (substrata sample-points), with their sidecars

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


def read_image_set(folder_or_yaml: str) -> Dict[str, Any]:
    """An image set's settings: ``id``, ``folder``, ``yaml``, ``photos`` (absolute), ``annos`` (``<id>.annos``),
    ``camera``, ``gsd_mm`` and ``classes`` (absolute, or None). Raises ValueError if it is not an image set."""
    p = os.path.abspath(folder_or_yaml)
    folder = os.path.dirname(p) if p.endswith((".yaml", ".yml")) else p
    y = image_set_yaml(folder)
    if y is None:
        raise ValueError(f"{folder} is not an image set (no {os.path.basename(folder)}.yaml with type: images)")
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
