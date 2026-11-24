"""Small helpers for 4x4 similarity transforms (NumPy only, no heavy imports).

Kept apart from ``geom`` so the light-weight modules (``imagesync``,
``framepose``, ``initializer``) can share one definition.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["uniform_scale", "metres_per_unit"]


def uniform_scale(transform: Any) -> float:
    """The uniform scale of a (similarity) transform.

    Computed as the cube root of ``|det|`` of the upper-left 3x3 block: the
    geometric mean of the axis scales, so it is the exact scale for a similarity
    transform (rotation x uniform scale, any translation) and the
    volume-preserving average for anything else. A reflection does not make it
    negative.

    Args:
        transform: A 4x4 (or 3x3) matrix or array-like (e.g. ``geom.Transform``).

    Returns:
        The scale factor (1.0 for an identity transform).
    """
    m = np.asarray(transform, dtype=float)
    return float(np.cbrt(abs(np.linalg.det(m[:3, :3]))))


def metres_per_unit(
    world_transform: Any, scale_factor: Optional[float] = None,
) -> Optional[float]:
    """Metres per original model unit of a project.

    The :func:`uniform_scale` of its ``world_transform``, except when that
    transform is missing or the identity (a project that was never oriented):
    then the project's ``scale_factor`` (from the scale bars), when it has a
    positive one. Returns None when neither gives a positive scale; an identity
    transform without a ``scale_factor`` gives 1.0 (the model is taken to be in
    metres).

    Args:
        world_transform: The project's 4x4 world transform, or None.
        scale_factor: The project's ``scale_factor``, or None.
    """
    identity = world_transform is None or np.allclose(
        np.asarray(world_transform, dtype=float), np.eye(4))
    if identity and scale_factor is not None:
        try:
            s = float(scale_factor)
        except (TypeError, ValueError):
            s = float("nan")
        if np.isfinite(s) and s > 0:
            return s
    if world_transform is None:
        return None
    s = uniform_scale(world_transform)
    return s if s > 0 else None
