"""Image adjustments for display: contrast methods for photos shown in review PDFs and figures.

A method is a function ``fn(img, ref, **kw) -> img`` on float RGB arrays in [0, 1], registered by name with
:func:`register_contrast_method`; :func:`auto_contrast` takes and returns a PIL image or a uint8 array. ``ref`` is
the image the statistics come from (by default ``img``): give several crops of one photo the same reference (e.g.
the whole photo, or the region a page shows first) and they get the same adjustment, so a 100% crop matches the
full frame it was cut from. The reference is downsampled for the statistics (``settings.CONTRAST_STATS_MAX_PX``).

Built in (``settings.CONTRAST_METHOD_DEFAULT`` is ``stretch``):

* ``none`` - unchanged.
* ``stretch`` - one linear stretch for all three channels, from the ``cutoff`` and ``100 - cutoff`` percentiles of
  the luminance: brighter and more contrast, colour balance kept.
* ``stretch-rgb`` - each channel stretched on its own percentiles (as PIL's ``ImageOps.autocontrast``): also
  removes a colour cast (blue-green water), which can shift true colours.
* ``clahe`` - contrast-limited adaptive histogram equalisation (OpenCV) of the lightness channel (CIELAB): local
  contrast in dark and bright parts alike; per image (``ref`` is not used).
"""

# Standard Library
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# Third-Party Libraries
import numpy as np

# Local Modules
import substrata.settings as settings  # (direct: `from substrata import` would load the whole package)

_CONTRAST_METHODS: Dict[str, Callable[..., np.ndarray]] = {}
_LUMA = np.array([0.299, 0.587, 0.114])


def register_contrast_method(name: str) -> Callable:
    """Decorator: register ``fn(img, ref, **kw)`` (float RGB in [0, 1]) as contrast method ``name``."""
    def deco(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        _CONTRAST_METHODS[name] = fn
        return fn
    return deco


def contrast_methods() -> List[str]:
    """The registered contrast method names."""
    return list(_CONTRAST_METHODS)


def _to_float(img: Any) -> np.ndarray:
    """An RGB image as float32 in 0-1, scaled by its dtype: integer images by
    their type's maximum (uint8 / 255, uint16 / 65535), float images as they
    are (taken to be 0-1 already)."""
    a = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3]
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return a.astype(np.float32)


def _stats_sample(ref: np.ndarray, max_px: int = settings.CONTRAST_STATS_MAX_PX) -> np.ndarray:
    """``ref`` subsampled (every n-th pixel) to at most ``max_px`` pixels for percentiles."""
    step = max(1, int(np.ceil(np.sqrt(ref.shape[0] * ref.shape[1] / max(max_px, 1)))))
    return ref[::step, ::step]


def _linear(img: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
    lo, hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


@register_contrast_method("none")
def _none(img: np.ndarray, ref: np.ndarray, **_kw) -> np.ndarray:
    return img


@register_contrast_method("stretch")
def _stretch(img: np.ndarray, ref: np.ndarray,
             cutoff: float = settings.CONTRAST_CUTOFF_PERCENT, **_kw) -> np.ndarray:
    lum = _stats_sample(ref) @ _LUMA
    lo, hi = np.percentile(lum, [cutoff, 100.0 - cutoff])
    return _linear(img, lo, hi)


@register_contrast_method("stretch-rgb")
def _stretch_rgb(img: np.ndarray, ref: np.ndarray,
                 cutoff: float = settings.CONTRAST_CUTOFF_PERCENT, **_kw) -> np.ndarray:
    s = _stats_sample(ref).reshape(-1, 3)
    lo, hi = np.percentile(s, cutoff, axis=0), np.percentile(s, 100.0 - cutoff, axis=0)
    return _linear(img, lo, hi)


@register_contrast_method("clahe")
def _clahe(img: np.ndarray, ref: np.ndarray, clip_limit: float = settings.CLAHE_CLIP_LIMIT,
           tile_grid: int = settings.CLAHE_TILE_GRID, **_kw) -> np.ndarray:
    import cv2  # (only this method needs OpenCV)

    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clip_limit, (tile_grid, tile_grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def auto_contrast(img: Any, method: Optional[str] = settings.CONTRAST_METHOD_DEFAULT,
                  reference: Any = None, **kw) -> Any:
    """``img`` with contrast method ``method`` applied (see the module docstring).

    Args:
        img: A PIL image or an RGB / grey array (uint8, or float in [0, 1]).
        method: A registered method name (:func:`contrast_methods`); None or ``"none"``: unchanged.
        reference: The image the statistics come from (default ``img``), e.g. the whole photo of a crop.
        **kw: Method options (``cutoff`` percent for the stretches; ``clip_limit``, ``tile_grid`` for clahe).

    Returns:
        The adjusted image, of ``img``'s kind (a PIL RGB image, or a uint8 array).

    Raises:
        ValueError: For an unknown method.
    """
    if method in (None, "none"):
        return img
    if method not in _CONTRAST_METHODS:
        raise ValueError(f"Unknown contrast method {method!r}; one of {contrast_methods()}")
    a = _to_float(img)
    ref = a if reference is None else _to_float(reference)
    out = (np.clip(_CONTRAST_METHODS[method](a, ref, **kw), 0, 1) * 255 + 0.5).astype(np.uint8)
    if hasattr(img, "convert"):
        from PIL import Image

        return Image.fromarray(out)
    return out
