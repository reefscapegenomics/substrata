"""Crops of photos around a pixel: the box arithmetic and the crop extraction.

Every classifier input crop (training crops, stored review crops, prediction
crops in ``classify_*`` and ``segment``) is cut with :func:`classifier_crop_box`,
so prediction sees the same pixels as training. The data paths (classification,
cameras, segmentation) import from here; ``visualizations.get_crop_img`` and
``classification.classifier_crop_box`` are the same functions, re-exported.

Only numpy and PIL are needed at import (OpenCV is imported by
:func:`encode_to_png_buffer` when called).
"""

from __future__ import annotations

# Standard Library
from io import BytesIO
from typing import Optional, Tuple

# Third-Party Libraries
import numpy as np

__all__ = [
    "classifier_crop_box",
    "clipped_crop_box",
    "get_crop_img",
    "encode_to_png_buffer",
]


def classifier_crop_box(
    x: float, y: float, crop_w: int, crop_h: int,
    img_w: Optional[int] = None, img_h: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Pixel box ``(left, top, right, bottom)`` of a crop centred on ``(x, y)``.

    ``left = int(x - crop_w / 2)`` (truncated), ``right = left + crop_w`` (and
    likewise for ``top`` / ``bottom``); clipped to the image when its size is
    given. Every classifier input crop (stored crops, photos in ``segment``, the
    training crops cut by :func:`get_crop_img`) uses this box, so prediction sees
    the same pixels as training.
    """
    left = int(x - crop_w / 2)
    top = int(y - crop_h / 2)
    right, bottom = left + int(crop_w), top + int(crop_h)
    if img_w is not None:
        left, right = max(0, left), min(int(img_w), right)
    if img_h is not None:
        top, bottom = max(0, top), min(int(img_h), bottom)
    return left, top, right, bottom


def clipped_crop_box(
    crop_x: float, crop_y: float, crop_w: int, crop_h: int,
    img_w: int, img_h: int, img_path: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    """:func:`classifier_crop_box` clipped to a ``img_w`` x ``img_h`` image.

    Raises:
        ValueError: When the crop falls completely outside the image
            (``img_path`` is named in the message).
    """
    left, top, right, bottom = classifier_crop_box(
        crop_x, crop_y, crop_w, crop_h, img_w, img_h)
    if left >= right or top >= bottom:
        raise ValueError(
            f"Crop area falls outside image bounds: center=({crop_x}, {crop_y}), "
            f"size=({crop_w}x{crop_h}), image={img_path}"
        )
    return left, top, right, bottom


def get_crop_img(img_path, crop_x, crop_y, crop_w, crop_h):
    """Crop centred on ``(crop_x, crop_y)`` as a PIL image (stored pixel grid).

    The box is :func:`classifier_crop_box` (the one every classifier input crop
    uses), clipped to the image. The photo file is closed before returning.

    Raises:
        ValueError: When the crop falls completely outside the image.
    """
    from PIL import Image

    with Image.open(img_path) as img:
        w_img, h_img = img.size
        box = clipped_crop_box(crop_x, crop_y, crop_w, crop_h, w_img, h_img,
                               img_path)
        cropped_img = img.crop(box)
        cropped_img.load()
    return cropped_img


def encode_to_png_buffer(image):
    """Encode an RGB image (PIL or array) as JPEG into a ``BytesIO``.

    (The name is historical: the buffer holds a JPEG.)
    """
    import cv2

    if hasattr(image, "size"):  # PIL image (numpy arrays too: taken as RGB)
        image = np.array(image)
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise IOError("Could not encode image to JPG.")
    return BytesIO(buffer.tobytes())
