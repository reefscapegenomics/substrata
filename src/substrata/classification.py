# Standard Library
from typing import Any, Dict, Optional, Tuple, Union
import os

# Third-Party Libraries
from PIL import Image
from fastai.vision.all import load_learner

# Local Modules
from substrata.logging import logger


def _center_crop(
    img: Image.Image, crop_size: Union[int, Tuple[int, int]]
) -> Image.Image:
    """
    Center-crop a PIL image to a given square size (int) or (width, height) tuple.

    Args:
        img: PIL image to crop.
        crop_size: Either an int (square) or a (width, height) tuple.

    Returns:
        Cropped PIL image.
    """
    w, h = img.size
    if isinstance(crop_size, int):
        cw = ch = min(crop_size, w, h)
    else:
        req_w, req_h = crop_size
        cw = min(req_w, w)
        ch = min(req_h, h)

    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    right = left + cw
    bottom = top + ch
    return img.crop((left, top, right, bottom))


def _ensure_learner(classifier: Union[str, Any]):
    """
    Ensure we have a FastAI learner object. If a string path is provided, load it.

    Args:
        classifier: Either a path to a .pkl learner or an already loaded learner.

    Returns:
        A FastAI Learner instance.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        RuntimeError: If loading the learner fails.
    """
    if isinstance(classifier, str):
        if not os.path.isfile(classifier):
            raise FileNotFoundError(f"Learner file not found: {classifier}")
        try:
            return load_learner(classifier)
        except Exception as e:
            raise RuntimeError(f"Failed to load learner from {classifier}: {e}") from e
    return classifier


def classify_image_match(
    image_match,
    classifier: Union[str, Any],
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run an image classifier on an ImageMatch and attach the result to the instance.

    - Loads a FastAI learner (if `classifier` is a path).
    - Optionally center-crops the source image to `crop_size`.
    - Calls learner.predict on the PIL image.
    - Stores the result on `image_match.classification` and returns it.

    Args:
        image_match: An instance of `cameras.ImageMatch`.
        classifier: Path to a FastAI .pkl learner or an already loaded learner.
        crop_size: Optional int (square) or (width, height) tuple for center crop.

    Returns:
        A dict with keys: 'label', 'confidence', 'probs', 'pred_idx'.
        Returns None on failure.
    """
    try:
        learn = _ensure_learner(classifier)
    except Exception as e:
        logger.error(str(e))
        return None

    if not hasattr(image_match, "filepath") or not image_match.filepath:
        logger.error("ImageMatch has no valid filepath.")
        return None
    if not os.path.isfile(image_match.filepath):
        logger.error(f"Image file not found: {image_match.filepath}")
        return None

    try:
        img = Image.open(image_match.filepath).convert("RGB")
        if crop_size is not None:
            img = _center_crop(img, crop_size)

        pred_class, pred_idx, pred_probs = learn.predict(img)

        # Build probabilities mapping if vocab is available
        probs_map = None
        try:
            vocab = getattr(getattr(learn, "dls", None), "vocab", None)
            if vocab is not None and pred_probs is not None:
                # Convert to {class_name: probability}
                probs = pred_probs.tolist()
                probs_map = {str(vocab[i]): float(probs[i]) for i in range(len(vocab))}
        except Exception:
            probs_map = None

        # Confidence as top-1 probability if available
        confidence = None
        if probs_map is not None:
            confidence = float(probs_map.get(str(pred_class), 0.0))
        elif pred_probs is not None:
            try:
                confidence = float(pred_probs[pred_idx].item())
            except Exception:
                pass

        result = {
            "label": str(pred_class),
            "confidence": confidence,
            "probs": probs_map,
            "pred_idx": int(pred_idx) if hasattr(pred_idx, "__int__") else pred_idx,
        }

        # Attach to the ImageMatch instance
        setattr(image_match, "classification", result)
        return result

    except Exception as e:
        logger.error(f"Classification failed for {image_match.filepath}: {e}")
        return None
