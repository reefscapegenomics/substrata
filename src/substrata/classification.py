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


def get_image_classifier(checkpoint: str, device: Optional[str] = None) -> Any:
    """
    Load a FastAI image classifier (learner), similar to get_sam2_predictor.

    Args:
        checkpoint: Path to the learner .pkl file.
        device: Optional 'cuda' or 'cpu'. If None, auto-detect.

    Returns:
        Loaded FastAI Learner.
    """
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Learner file not found: {checkpoint}")

    try:
        import torch  # lazy

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        dev = device or "cpu"

    # Load on CPU by default, then move if needed
    learn = load_learner(checkpoint, cpu=True)
    if dev != "cpu":
        try:
            learn.to(dev)  # type: ignore[attr-defined]
        except Exception:
            logger.warning("Could not move learner to device '%s'; using CPU.", dev)
    return learn


def _ensure_learner(classifier: Union[str, Any]):
    """
    Ensure we have a FastAI learner object. If a string path is provided, load it.

    Args:
        classifier: Either a path to a .pkl learner or an already loaded learner.

    Returns:
        A FastAI Learner instance.
    """
    if isinstance(classifier, str):
        return get_image_classifier(classifier)
    return classifier


def classify_image_match(
    image_match,
    classifier: Union[str, Any],
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    show: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run an image classifier on an ImageMatch and attach the result to the instance.

    - Accepts a loaded FastAI learner (recommended) or a path (will be loaded).
    - Optionally center-crops the source image to `crop_size`.
    - Calls learner.predict on the PIL image.
    - Stores the result on `image_match.classification` and returns it.

    Args:
        image_match: An instance of `cameras.ImageMatch`.
        classifier: Loaded FastAI learner or path to a .pkl learner.
        crop_size: Optional int (square) or (width, height) tuple for center crop.
        show: If True, display the image crop and classification result (Jupyter notebook).

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

        # --- Show the exact image crop and classification output if requested ---
        if show:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Classification: {result.get('label', '')}\nConfidence: {result.get('confidence', ''):.3f}")
                plt.show()
            except Exception as e:
                logger.warning(f"Could not display classification input crop: {e}")
        # --------------------------------------------------------------------------

        # Attach to the ImageMatch instance
        setattr(image_match, "classification", result)
        return result

    except Exception as e:
        logger.error(f"Classification failed for {image_match.filepath}: {e}")
        return None
