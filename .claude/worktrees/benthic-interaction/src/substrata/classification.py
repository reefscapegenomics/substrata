# Standard Library
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import csv
import hashlib
import os
from collections import Counter

# Third-Party Libraries
from PIL import Image
from tqdm.auto import tqdm

# Local Modules
from substrata import settings
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
        device: Optional 'cuda'/'mps'/'cpu'. If None, auto-detect (prefers
            CUDA, then Apple MPS, then CPU).

    Returns:
        Loaded FastAI Learner placed on the selected device.
    """
    from fastai.vision.all import load_learner  # lazy heavy import

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Learner file not found: {checkpoint}")

    try:
        import torch  # lazy

        if device is not None:
            dev = device
        elif torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    except Exception:  # noqa: BLE001 - torch import / probing failed.
        dev = device or "cpu"

    # load_learner(cpu=False) restores onto the default device; then move the
    # model AND the dataloaders explicitly (a fastai Learner has no ``.to``, so
    # the model must be moved via ``learn.model.to`` and batches via
    # ``learn.dls.device`` — otherwise inference silently runs on the CPU).
    learn = load_learner(checkpoint, cpu=(dev == "cpu"))
    if dev != "cpu":
        try:
            learn.model = learn.model.to(dev)
            if hasattr(learn.dls, "to"):
                learn.dls.to(dev)
            else:
                learn.dls.device = dev
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not move learner to device '%s' (%s); using CPU.", dev, e
            )
            try:
                learn.model = learn.model.to("cpu")
                learn.dls.device = "cpu"
            except Exception:  # noqa: BLE001
                pass
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


def _crop_for_match(image_match, crop_size) -> Optional[Image.Image]:
    """Return the classifier-input crop for an ImageMatch (or None on failure).

    Crops centred on the matched pixel ``(image_match.x, image_match.y)`` at
    ``crop_size`` (matching how training crops are cut); with ``crop_size=None``
    uses the whole image. Shared by :func:`classify_image_match` and the batched
    :func:`classify_image_matches_batch` so both feed the classifier the same
    pixels.
    """
    fp = getattr(image_match, "filepath", None)
    if not fp or not os.path.isfile(fp):
        return None
    try:
        if crop_size is not None:
            from substrata.visualizations import get_crop_img  # lazy heavy import

            if isinstance(crop_size, int):
                crop_w = crop_h = crop_size
            else:
                crop_w, crop_h = crop_size
            return get_crop_img(
                fp, image_match.x, image_match.y, crop_w, crop_h
            ).convert("RGB")
        return Image.open(fp).convert("RGB")
    except (OSError, ValueError):
        return None


def _learner_device(learn) -> str:
    """Best-effort device string for a fastai learner ('cuda', 'mps', 'cpu')."""
    try:
        param = next(learn.model.parameters())
        return str(param.device).split(":")[0]
    except Exception:  # noqa: BLE001
        try:
            return str(getattr(getattr(learn, "dls", None), "device", "unknown"))
        except Exception:  # noqa: BLE001
            return "unknown"


def _result_from_probs(prob_list, vocab) -> Dict[str, Any]:
    """Build the standard classification result dict from a probability vector."""
    pred_idx = int(max(range(len(prob_list)), key=lambda i: prob_list[i]))
    if vocab is not None and len(vocab) == len(prob_list):
        probs_map = {str(vocab[i]): float(prob_list[i]) for i in range(len(vocab))}
        label = str(vocab[pred_idx])
        confidence = float(probs_map[label])
    else:
        probs_map = None
        label = str(pred_idx)
        confidence = float(prob_list[pred_idx])
    return {
        "label": label, "confidence": confidence,
        "probs": probs_map, "pred_idx": pred_idx,
    }


def classify_image_matches_batch(
    image_matches: List[Any], classifier: Union[str, Any],
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    batch_size: int = 64, n_jobs: int = -1,
) -> List[Optional[Dict[str, Any]]]:
    """Classify many ImageMatches in batched GPU passes (fast path).

    Crops each match centred on ``(x, y)`` (like :func:`classify_image_match`),
    then runs a **single** ``learn.get_preds`` over all crops in batches of
    ``batch_size`` instead of a per-image ``learn.predict``. fastai's
    ``predict`` rebuilds a one-item dataloader on every call, so batching is
    typically an order of magnitude faster (more on GPU). The crops are loaded
    **in parallel** (``n_jobs`` threads) because decoding the full-resolution
    source photos is usually the real bottleneck - far more than the forward
    pass. Each match's ``.classification`` is set and a list of result dicts
    aligned with ``image_matches`` is returned (``None`` where the crop or
    inference failed). Falls back to per-image classification if batched
    inference is unavailable.

    Args:
        image_matches: ImageMatch instances to classify.
        classifier: Loaded fastai learner or path to a ``.pkl`` (loaded once).
        crop_size: Crop size (int or ``(w, h)``); ``None`` uses the full image.
        batch_size: Inference batch size.
        n_jobs: Threads for parallel crop loading (-1 = all cores).

    Returns:
        List of result dicts (or ``None``) aligned with ``image_matches``.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(image_matches)
    if not image_matches:
        return results
    try:
        learn = _ensure_learner(classifier)
    except Exception as e:  # noqa: BLE001
        logger.error(str(e))
        return results

    # Report the inference device so it is clear whether classification is
    # GPU-accelerated (a CPU learner is the usual reason batching feels slow).
    device = _learner_device(learn)
    accel = "GPU" if device not in ("cpu", "unknown") else "CPU"
    msg = (
        f"Classifying {len(image_matches)} image match(es) on {device} "
        f"({accel}, batch_size={batch_size})"
    )
    logger.info(msg)
    print(msg)

    # Load the crops in parallel (JPEG decode releases the GIL, so threads
    # scale). Decoding the full-resolution source photos is the dominant cost,
    # so this is the main speed lever - not the GPU forward pass. joblib
    # preserves input order, keeping results aligned with image_matches.
    from joblib import Parallel, delayed

    from substrata.logging import tqdm_joblib

    with tqdm_joblib(tqdm(
        total=len(image_matches),
        desc=f"Cropping image matches ({device})", unit="img",
    )):
        loaded = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_crop_for_match)(im, crop_size) for im in image_matches
        )
    crops, idx_map = [], []
    for i, crop in enumerate(loaded):
        if crop is not None:
            crops.append(crop)
            idx_map.append(i)
    if not crops:
        return results

    vocab = getattr(getattr(learn, "dls", None), "vocab", None)
    try:
        # Single batched forward pass over all crops (order preserved).
        dl = learn.dls.test_dl(crops, bs=batch_size)
        probs, _ = learn.get_preds(dl=dl)
        for pos, prob in zip(idx_map, probs.tolist()):
            res = _result_from_probs(prob, vocab)
            results[pos] = res
            setattr(image_matches[pos], "classification", res)
    except Exception as e:  # noqa: BLE001 - fall back to the per-image path.
        logger.warning(
            "Batched classification unavailable (%s); using per-image predict.", e
        )
        for pos in idx_map:
            results[pos] = classify_image_match(
                image_matches[pos], learn, crop_size
            )
    return results


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
        show: If True, display the image crop and classification result
            (Jupyter notebook).

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
        # Crop centred on the matched pixel (image_match.x, image_match.y),
        # matching how training crops are cut (generate_crop -> get_crop_img).
        img = _crop_for_match(image_match, crop_size)
        if img is None:
            logger.error(f"Could not crop image: {image_match.filepath}")
            return None

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
                plt.title(
                    f"Classification: {result.get('label', '')}\n"
                    f"Confidence: {result.get('confidence', ''):.3f}"
                )
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


# ---------------------------------------------------------------------------
# Classifier training (substrata train)
# ---------------------------------------------------------------------------
#
# The label-tree functions below mirror sandbox/annotations/count_labels.py so
# that `substrata train` renders the identical CATAMI hierarchy and uses the
# *bolded* tip/heavy-parent entries as the set of training labels. The bolding
# rule lives in a single helper (`_is_bold`) shared by render() and
# get_training_labels() so the displayed tree and the trained labels can never
# diverge. All tunable constants (column orders, ANSI styling, crop/image
# parameters) live in ``settings``.


def load_classes(path: str) -> Tuple[Dict, Dict, List[str]]:
    """Load the CATAMI class hierarchy from a classes CSV.

    Args:
        path: Path to the classes CSV (SPECIES_CODE / CATAMI_PARENT_ID /
            CPC_CODES / CATAMI_LEVEL_1..7 columns).

    Returns:
        Tuple of (nodes, children, roots) where
        ``nodes[species_code] = {cpc, name, parent}``,
        ``children[species_code]`` lists child species codes, and ``roots``
        lists species codes whose parent is not itself a node.
    """
    nodes: Dict[str, Dict[str, str]] = {}
    children: Dict[str, List[str]] = {}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # Skip blank/padding rows that have no species code.
            if not row["SPECIES_CODE"].strip():
                continue
            rows.append(row)

    codes = {row["SPECIES_CODE"].strip() for row in rows}

    for row in rows:
        code = row["SPECIES_CODE"].strip()
        # Display name = deepest non-empty CATAMI level for this row.
        name = code
        for col in settings.TRAIN_LEVEL_COLUMNS:
            if row.get(col, "").strip():
                name = row[col].strip()
        nodes[code] = {
            "cpc": row["CPC_CODES"].strip(),
            "name": name,
            "parent": row["CATAMI_PARENT_ID"].strip(),
        }
        children.setdefault(code, [])

    roots = []
    for code, node in nodes.items():
        parent = node["parent"]
        # A node is a root when its parent is empty, points to a code that
        # doesn't exist, or refers to itself.
        if parent and parent != code and parent in codes:
            children.setdefault(parent, []).append(code)
        else:
            roots.append(code)

    return nodes, children, roots


def count_labels(
    files: List[str], class_codes: Set[str]
) -> Tuple[Counter, Counter]:
    """Tally ``label`` column values across the given annotation CSVs.

    Args:
        files: Annotation CSV paths to scan.
        class_codes: Set of valid CPC codes (labels not in this set are
            collected separately as ``unknown``).

    Returns:
        Tuple ``(counts, unknown)`` of Counters keyed by label/CPC code.
    """
    counts: Counter = Counter()
    unknown: Counter = Counter()
    for path in files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if settings.TRAIN_LABEL_COLUMN not in (reader.fieldnames or []):
                continue
            for row in reader:
                label = (row.get(settings.TRAIN_LABEL_COLUMN) or "").strip()
                if not label:
                    continue
                if label in class_codes:
                    counts[label] += 1
                else:
                    unknown[label] += 1
    return counts, unknown


def subtree_totals(
    children: Dict, roots: List[str], direct: Dict
) -> Dict[str, int]:
    """Aggregate counts up the tree (``total = direct + sum of children``)."""
    total: Dict[str, int] = {}

    def walk(code: str) -> int:
        t = direct.get(code, 0)
        for child in children.get(code, []):
            t += walk(child)
        total[code] = t
        return t

    for root in roots:
        walk(root)
    return total


def _is_bold(
    has_visible_kids: bool, is_root: bool, direct_count: int,
    total_count: int, min_count: int, tips_only: bool,
) -> bool:
    """Whether a node is bolded (i.e. used as a training label).

    Shared by :func:`render` and :func:`get_training_labels` so the displayed
    tree and the training-label set always agree. ``min_count`` controls only
    bolding, not visibility - sub-threshold entries are still shown, just not
    bolded.

    A node is bolded when it is a *tip* (no visible subcategories and not a
    bare root-level category) whose aggregated count reaches ``min_count``,
    or - unless ``tips_only`` - a *heavy parent* (has visible children and its
    own direct count exceeds ``min_count``).
    """
    is_tip = not has_visible_kids and not is_root
    is_tip_bold = is_tip and total_count >= min_count
    is_heavy_parent = (
        has_visible_kids and direct_count > min_count and not tips_only
    )
    return is_tip_bold or is_heavy_parent


def _node_is_bold(
    node_label: str, has_visible_kids: bool, is_root: bool, direct_count: int,
    total_count: int, min_count: int, tips_only: bool,
    include_labels: Optional[Set[str]],
) -> bool:
    """Bolding decision for one node.

    When ``include_labels`` is given, a node is bolded iff its label is in that
    set (an explicit override of the count-based rules); otherwise the
    :func:`_is_bold` tip/heavy-parent rule applies.
    """
    if include_labels is not None:
        return node_label in include_labels
    return _is_bold(
        has_visible_kids, is_root, direct_count, total_count, min_count, tips_only
    )


def render(
    nodes: Dict, children: Dict, roots: List[str], direct: Dict, total: Dict,
    lines: List[str], min_count: int = 1, tips_only: bool = False,
    include_labels: Optional[Set[str]] = None,
) -> None:
    """Append the CATAMI tree to ``lines``.

    All nodes with at least ``settings.TRAIN_MIN_VISIBLE_COUNT`` occurrences are
    shown. Which are bolded (i.e. training labels) is decided by
    :func:`_node_is_bold`: by the ``min_count`` tip/heavy-parent rule, or - when
    ``include_labels`` is given - exactly the labels in that set.
    """
    vis = settings.TRAIN_MIN_VISIBLE_COUNT

    def walk(code, prefix, is_last, is_root=False):
        if total.get(code, 0) < vis:
            return
        node = nodes[code]
        connector = "└── " if is_last else "├── "
        d = direct.get(code, 0)
        t = total.get(code, 0)

        kids = [c for c in children.get(code, []) if total.get(c, 0) >= vis]
        kids.sort(key=lambda c: (-total.get(c, 0), nodes[c]["name"]))

        # Show the subtree total; add the node's own direct count if it also
        # has visible children that contributed.
        if kids:
            count_str = f"{t}" + (f" (self {d})" if d else "")
        else:
            count_str = f"{t}"
        label = node["cpc"] or code
        text = f"{node['name']} [{label}]: {count_str}"
        if _node_is_bold(
            label, bool(kids), is_root, d, t, min_count, tips_only, include_labels
        ):
            text = f"{settings.TRAIN_BOLD}{text}{settings.TRAIN_RESET}"
        lines.append(f"{prefix}{connector}{text}")

        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(kids):
            walk(child, new_prefix, i == len(kids) - 1)

    visible_roots = [r for r in roots if total.get(r, 0) >= vis]
    visible_roots.sort(key=lambda c: (-total.get(c, 0), nodes[c]["name"]))
    for i, root in enumerate(visible_roots):
        walk(root, "", i == len(visible_roots) - 1, is_root=True)


def get_training_labels(
    nodes: Dict, children: Dict, roots: List[str], direct: Dict, total: Dict,
    min_count: int = 1, tips_only: bool = False,
    include_labels: Optional[Set[str]] = None,
) -> Set[str]:
    """Return the labels of exactly the entries :func:`render` bolds.

    Walks the same visible tree as :func:`render` and collects the CPC code
    (falling back to the species code) of every bolded node. When
    ``include_labels`` is given, the result is that set restricted to labels
    that actually appear in the tree, so the caller can detect (and reject) any
    requested category that is absent.
    """
    labels: Set[str] = set()
    vis = settings.TRAIN_MIN_VISIBLE_COUNT

    def walk(code, is_root=False):
        if total.get(code, 0) < vis:
            return
        kids = [c for c in children.get(code, []) if total.get(c, 0) >= vis]
        d = direct.get(code, 0)
        t = total.get(code, 0)
        label = nodes[code]["cpc"] or code
        if _node_is_bold(
            label, bool(kids), is_root, d, t, min_count, tips_only, include_labels
        ):
            labels.add(label)
        for child in kids:
            walk(child)

    for root in roots:
        if total.get(root, 0) >= vis:
            walk(root, is_root=True)
    return labels


def get_label_collapse_map(
    nodes: Dict, children: Dict, roots: List[str], direct: Dict, total: Dict,
    min_count: int = 1, tips_only: bool = False,
    include_labels: Optional[Set[str]] = None, collapse: bool = False,
) -> Dict[str, str]:
    """Map each visible label to the training class it collapses into.

    Walks the same visible tree as :func:`render`/:func:`get_training_labels`,
    carrying the nearest *bolded* ancestor down each branch. Keys and values are
    CPC codes (the raw ``label`` values, which are also the crop folder names).

    A bolded node always maps to itself (it is its own training class). A
    *non-bolded* visible node is, by default, excluded (omitted from the map).
    When ``collapse`` is True it instead folds into its nearest bolded ancestor
    (if any) - so e.g. with ``include_labels={'MAF'}`` and ``collapse=True`` the
    descendants ``MAFG``/``MAF_T`` are trained as ``MAF``; without ``collapse``
    only ``MAF`` itself is trained and its descendants are excluded.

    Args:
        nodes, children, roots, direct, total: Label-tree structures from
            :func:`load_classes` / :func:`subtree_totals`.
        min_count: Bolding threshold (see :func:`_is_bold`).
        tips_only: Only bold tip entries (not heavy parents).
        include_labels: Explicit bolded set; when given it overrides the
            count-based rules (see :func:`_node_is_bold`).
        collapse: Fold non-bolded descendants into their nearest bolded
            ancestor instead of excluding them.

    Returns:
        Mapping ``raw_label -> training_class`` for the non-excluded labels.
    """
    mapping: Dict[str, str] = {}
    vis = settings.TRAIN_MIN_VISIBLE_COUNT

    def walk(code, nearest_bold: Optional[str], is_root=False):
        if total.get(code, 0) < vis:
            return
        kids = [c for c in children.get(code, []) if total.get(c, 0) >= vis]
        d = direct.get(code, 0)
        t = total.get(code, 0)
        label = nodes[code]["cpc"] or code
        if _node_is_bold(
            label, bool(kids), is_root, d, t, min_count, tips_only, include_labels
        ):
            nearest_bold = label
            mapping[label] = label
        elif collapse and nearest_bold is not None:
            mapping[label] = nearest_bold
        for child in kids:
            walk(child, nearest_bold)

    for root in roots:
        if total.get(root, 0) >= vis:
            walk(root, None, is_root=True)
    return mapping


def build_label_tree(
    classes_path: str, csv_files: List[str], min_count: int = 1,
    tips_only: bool = False, include_labels: Optional[Set[str]] = None,
    collapse: bool = False,
) -> Tuple[List[str], Set[str], Counter, Counter, Dict[str, str]]:
    """Render the CATAMI label tree and derive the training-label set.

    Combines :func:`load_classes`, :func:`count_labels`, :func:`subtree_totals`,
    :func:`render` and :func:`get_training_labels` into one call.

    Args:
        classes_path: Path to the classes CSV.
        csv_files: Annotation CSV paths to count labels across.
        min_count: Bold (i.e. select as a training label) only entries whose
            count reaches this; lower-count entries are still shown, unbolded.
        tips_only: Only bold tip entries (not heavy parents).
        include_labels: Explicit set of labels to bold/train on. When given it
            overrides ``min_count``/``tips_only``; the returned training-label
            set is this restricted to labels present in the tree, so the caller
            can detect any requested category that is absent.
        collapse: Fold non-selected descendants into their nearest selected
            ancestor in the collapse map instead of excluding them (see
            :func:`get_label_collapse_map`).

    Returns:
        Tuple ``(lines, training_labels, counts, unknown, collapse_map)`` where
        ``lines`` is the rendered tree (with ANSI bold styling),
        ``training_labels`` is the set of bolded CPC codes, ``counts`` maps CPC
        code -> count, ``unknown`` maps unrecognised label -> count, and
        ``collapse_map`` (see :func:`get_label_collapse_map`) maps each
        non-excluded raw label to the training class it collapses into.
    """
    nodes, children, roots = load_classes(classes_path)
    cpc_to_code = {n["cpc"]: code for code, n in nodes.items() if n["cpc"]}

    counts, unknown = count_labels(csv_files, set(cpc_to_code))

    direct: Counter = Counter()
    for cpc, n in counts.items():
        direct[cpc_to_code[cpc]] += n

    total = subtree_totals(children, roots, direct)

    lines: List[str] = []
    lines.append(f"Scanned {len(csv_files)} CSV file(s).")
    lines.append(f"Total annotations counted: {sum(counts.values())}")
    lines.append(f"Distinct labels found: {len(counts)}")
    lines.append("")
    if include_labels is not None:
        lines.append(
            "Label occurrences on the CATAMI hierarchy "
            "(bolded = --include-classes categories, used for training):"
        )
    elif min_count > 1:
        lines.append(
            "Label occurrences on the CATAMI hierarchy "
            f"(bolded = count >= {min_count}, used for training):"
        )
    else:
        lines.append("Label occurrences on the CATAMI hierarchy:")
    lines.append("")
    render(
        nodes, children, roots, direct, total, lines, min_count, tips_only,
        include_labels=include_labels,
    )

    if unknown:
        lines.append("")
        lines.append("Labels not found in classes.csv (not shown in tree):")
        for label, n in unknown.most_common():
            lines.append(f"  {label}: {n}")

    training_labels = get_training_labels(
        nodes, children, roots, direct, total, min_count, tips_only,
        include_labels=include_labels,
    )
    collapse_map = get_label_collapse_map(
        nodes, children, roots, direct, total, min_count, tips_only,
        include_labels=include_labels, collapse=collapse,
    )
    return lines, training_labels, counts, unknown, collapse_map


# ---------------------------------------------------------------------------
# Label map (selection + hierarchical collapse), decoupled from crops
# ---------------------------------------------------------------------------
#
# Crops are generated for every *visible* raw label (folder named by the raw
# label), independent of which categories are trained. Which labels are trained
# and how they collapse lives in an editable CSV (``label,count,training_class``)
# so changing the considered categories only re-reads this file and never
# regenerates crops. A blank ``training_class`` means the label is excluded from
# training. The file is seeded from the label tree (:func:`get_label_collapse_map`);
# the CLI re-seeds it from the current selection flags by default, or preserves
# hand edits and only appends new labels under ``--keep-map``
# (:func:`merge_label_map`).

_LABEL_MAP_COLUMNS = ("label", "count", "training_class")


def write_label_map(
    path: str, labels: Set[str], counts: Counter, collapse_map: Dict[str, str],
) -> None:
    """Write a fresh ``label,count,training_class`` map for ``labels``.

    One row per label in ``labels`` (sorted), with the seed ``training_class``
    taken from ``collapse_map`` (blank for any label not in it). Callers seed
    only the selected/training labels, so the file stays free of blank rows for
    excluded labels; a row can still be blanked by hand to drop that label. The
    ``count`` column is an informational hint for hand-editing.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_LABEL_MAP_COLUMNS)
        for label in sorted(labels):
            writer.writerow(
                [label, counts.get(label, 0), collapse_map.get(label, "")]
            )


def load_label_map(path: str) -> Dict[str, str]:
    """Read a label map, returning ``{raw_label: training_class}``.

    Rows with a blank ``training_class`` are treated as excluded and omitted, so
    the returned mapping contains only labels that participate in training.
    """
    mapping: Dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            training_class = (row.get("training_class") or "").strip()
            if label and training_class:
                mapping[label] = training_class
    return mapping


def merge_label_map(
    path: str, labels: Set[str], counts: Counter, collapse_map: Dict[str, str],
    reseed: bool = False,
) -> Dict[str, str]:
    """Seed or update the on-disk label map and return the active mapping.

    When the file does not exist (or ``reseed`` is True) it is written fresh
    from ``collapse_map``. Otherwise existing rows are preserved verbatim
    (keeping the user's hand edits) and only labels not already present are
    appended with their seed mapping - so newly added annotation CSVs surface
    their labels without clobbering prior edits.

    Args:
        path: Map CSV path.
        labels: Visible raw labels that should be represented.
        counts: Label -> count (for the informational ``count`` column).
        collapse_map: Seed ``raw_label -> training_class`` mapping.
        reseed: Overwrite any existing file from the seed mapping.

    Returns:
        The active ``{raw_label: training_class}`` mapping (excluded labels
        omitted), as read back from disk.
    """
    if reseed or not os.path.isfile(path):
        write_label_map(path, labels, counts, collapse_map)
        return load_label_map(path)

    with open(path, newline="") as f:
        existing_rows = list(csv.DictReader(f))
    known = {(r.get("label") or "").strip() for r in existing_rows}
    new_labels = sorted(lab for lab in labels if lab not in known)
    if new_labels:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            for label in new_labels:
                writer.writerow(
                    [label, counts.get(label, 0), collapse_map.get(label, "")]
                )
        logger.info(
            "Added %d new label(s) to %s (existing edits preserved).",
            len(new_labels), path,
        )
    return load_label_map(path)


# ---------------------------------------------------------------------------
# Collation, path repair and crop generation
# ---------------------------------------------------------------------------


def model_name_from_filename(filename: str, pattern: str) -> str:
    """Derive a model name from an annotation filename and a glob pattern.

    The model name is the filename with the literal text following the ``*``
    in ``pattern`` stripped from its end. For ``pattern='*_slope_intercepts.csv'``
    and ``filename='ton_ko1_05m_20241005_slope_intercepts.csv'`` this returns
    ``'ton_ko1_05m_20241005'``.

    Args:
        filename: Annotation CSV filename (basename or path).
        pattern: Glob pattern with a single ``*`` wildcard.

    Returns:
        The derived model name.
    """
    base = os.path.basename(filename)
    suffix = pattern.split("*")[-1] if "*" in pattern else ""
    if suffix and base.endswith(suffix):
        return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def convention_dir_for_model(model_path: str, model_name: str, last_folder: str) -> str:
    """Build the standard directory path for a model's image folder.

    Follows the ``site_location_depth_date`` naming convention, nesting by
    site_location / site_location_depth / model_name. For ``model_name=
    'ton_ko1_05m_20241005'`` and ``last_folder='ton_ko1_05m_20241005.photos'``
    this returns
    ``<model_path>/ton_ko1/ton_ko1_05m/ton_ko1_05m_20241005/<last_folder>``.

    Args:
        model_path: Base directory holding the model project folders.
        model_name: ``site_location_depth_date`` model name.
        last_folder: Final path component to preserve (e.g. ``<id>.photos``).

    Returns:
        The candidate directory path.
    """
    parts = model_name.split("_")
    # site_location = first two underscore-parts; +depth = first three.
    site_loc = "_".join(parts[:2]) if len(parts) >= 2 else model_name
    site_loc_depth = "_".join(parts[:3]) if len(parts) >= 3 else model_name
    return os.path.join(
        model_path, site_loc, site_loc_depth, model_name, last_folder
    )


def resolve_cam_dirs(
    cam_dirs: Set[str], model_path: str, model_name: str,
    prompt: bool = True,
) -> Dict[str, str]:
    """Resolve (and if needed remap) camera image directories.

    For each directory in ``cam_dirs`` that exists, it maps to itself. For a
    missing directory, tries the standard convention path
    (:func:`convention_dir_for_model`) preserving the original final folder.
    If that also does not exist and ``prompt`` is True, asks the user for a
    substitution directory (the original final folder is appended to whatever
    base the user supplies, preserving the final-folder + filename layout).

    Args:
        cam_dirs: Unique directories referenced by ``cam_filepath`` entries.
        model_path: Base directory for convention-based fallback.
        model_name: Model name used to build the convention path.
        prompt: Whether to interactively prompt for unresolved directories.

    Returns:
        Mapping ``old_dir -> new_dir`` for every input directory.

    Raises:
        FileNotFoundError: If a directory cannot be resolved and ``prompt`` is
            False (or the user provides no substitution).
    """
    mapping: Dict[str, str] = {}
    for old_dir in sorted(cam_dirs):
        if old_dir and os.path.isdir(old_dir):
            mapping[old_dir] = old_dir
            continue
        last_folder = os.path.basename(old_dir.rstrip("/")) if old_dir else ""
        candidate = convention_dir_for_model(model_path, model_name, last_folder)
        if os.path.isdir(candidate):
            logger.info("Remapped missing dir %s -> %s", old_dir, candidate)
            mapping[old_dir] = candidate
            continue
        if not prompt:
            raise FileNotFoundError(
                f"Camera directory not found and no fallback exists: {old_dir!r} "
                f"(tried convention path {candidate!r})."
            )
        print(
            f"\nCamera directory not found:\n  {old_dir}\n"
            f"Convention fallback also missing:\n  {candidate}"
        )
        sub = input(
            "Enter a substitution base directory (the final folder "
            f"{last_folder!r} will be appended), or blank to abort: "
        ).strip()
        if not sub:
            raise FileNotFoundError(
                f"No substitution provided for missing directory {old_dir!r}."
            )
        new_dir = os.path.join(sub, last_folder)
        if not os.path.isdir(new_dir):
            print(f"Warning: substituted directory does not exist: {new_dir}")
        mapping[old_dir] = new_dir
    return mapping


def collate_training_annotations(
    csv_files: List[str], pattern: str, keep_labels: Set[str],
    output_path: str, model_path: str, prompt: bool = True,
) -> Tuple[int, int]:
    """Write a consolidated training annotations CSV.

    Keeps every row whose ``label`` is in ``keep_labels`` (the *visible* labels,
    not just the trained ones - selection/collapse is applied later via the
    label map), rewrites ``cam_filepath`` directories via
    :func:`resolve_cam_dirs`, prefixes integer-only ``id`` values with the model
    name, and drops rows lacking the camera fields needed to make a crop.

    Args:
        csv_files: Annotation CSV paths.
        pattern: Glob pattern used to derive the per-file model name.
        keep_labels: CPC codes to keep (typically the visible labels).
        output_path: Destination CSV path.
        model_path: Base directory for camera-directory fallback.
        prompt: Whether to prompt for unresolved camera directories.

    Returns:
        Tuple ``(n_written, n_dropped_missing_cam)``.
    """
    # First pass: collect kept rows and the unique camera directories per file.
    kept: List[Dict[str, str]] = []
    dir_mapping: Dict[str, str] = {}
    for path in csv_files:
        model_name = model_name_from_filename(path, pattern)
        file_dirs: Set[str] = set()
        rows: List[Dict[str, str]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("label") or "").strip()
                if label not in keep_labels:
                    continue
                cam_fp = (row.get("cam_filepath") or "").strip()
                if cam_fp:
                    file_dirs.add(os.path.dirname(cam_fp))
                row["_model_name"] = model_name
                rows.append(row)
        # Resolve this file's directories (per-model convention fallback).
        file_map = resolve_cam_dirs(file_dirs, model_path, model_name, prompt)
        dir_mapping.update(file_map)
        kept.extend(rows)

    n_written = 0
    n_dropped = 0
    with open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(settings.TRAIN_ANN_COLUMNS)
        for row in kept:
            cam_fp = (row.get("cam_filepath") or "").strip()
            cam_x = (row.get("cam_x") or "").strip()
            cam_y = (row.get("cam_y") or "").strip()
            if not cam_fp or not cam_x or not cam_y:
                n_dropped += 1
                continue
            old_dir = os.path.dirname(cam_fp)
            new_dir = dir_mapping.get(old_dir, old_dir)
            new_fp = os.path.join(new_dir, os.path.basename(cam_fp))

            ann_id = (row.get("id") or "").strip()
            if ann_id.isdigit():
                ann_id = f"{row['_model_name']}_{ann_id}"

            writer.writerow([
                ann_id,
                row.get("orig_x", ""),
                row.get("orig_y", ""),
                row.get("orig_z", ""),
                row.get("label", ""),
                row.get("label_conf", ""),
                row.get("world_x", ""),
                row.get("world_y", ""),
                row.get("world_z", ""),
                new_fp,
                cam_x,
                cam_y,
                row.get("depth", ""),
            ])
            n_written += 1
    return n_written, n_dropped


def split_for_id(ann_id: str, split: Tuple[int, int, int]) -> int:
    """Deterministically assign an annotation id to a split.

    Args:
        ann_id: Annotation id.
        split: ``(train, validation, test)`` percentages summing to 100.

    Returns:
        ``0`` for train, ``1`` for validation, ``2`` for test.
    """
    h = int(hashlib.md5(ann_id.encode()).hexdigest(), 16) % 100
    if h < split[0]:
        return 0
    if h < split[0] + split[1]:
        return 1
    return 2


def crop_filename(ann_id: str, cam_filepath: str, cam_x: float, cam_y: float) -> str:
    """Build the encoded crop filename ``<id>_<camstem>_<x>_<y>.jpg``.

    Encoding the camera filename and integer pixel centre in the name means a
    changed annotation produces a different filename, so the stale old crop is
    detected and removed while the new one is generated.
    """
    cam_stem = os.path.splitext(os.path.basename(cam_filepath))[0]
    return f"{ann_id}_{cam_stem}_{int(round(cam_x))}_{int(round(cam_y))}.jpg"


def plan_crops(
    annotations_path: str, output_dir: str,
    split: Tuple[int, int, int] = settings.TRAIN_SPLIT,
    crop_dirs: Tuple[str, str, str] = settings.TRAIN_CROP_DIRS,
) -> Dict[str, Dict[str, Any]]:
    """Compute the expected crop file for every annotation.

    Args:
        annotations_path: Consolidated training annotations CSV.
        output_dir: Base output directory holding the crop folders.
        split: Train/validation/test percentages.
        crop_dirs: Folder names for the three splits.

    Returns:
        Mapping ``abs_crop_path -> {cam_filepath, cam_x, cam_y, label, id}``.
    """
    expected: Dict[str, Dict[str, Any]] = {}
    with open(annotations_path, newline="") as f:
        for row in csv.DictReader(f):
            ann_id = (row.get("id") or "").strip()
            label = (row.get("label") or "").strip()
            cam_fp = (row.get("cam_filepath") or "").strip()
            cam_x = (row.get("cam_x") or "").strip()
            cam_y = (row.get("cam_y") or "").strip()
            if not (ann_id and label and cam_fp and cam_x and cam_y):
                continue
            cx, cy = float(cam_x), float(cam_y)
            split_dir = crop_dirs[split_for_id(ann_id, split)]
            fname = crop_filename(ann_id, cam_fp, cx, cy)
            abs_path = os.path.join(output_dir, split_dir, label, fname)
            expected[abs_path] = {
                "cam_filepath": cam_fp,
                "cam_x": cx,
                "cam_y": cy,
                "label": label,
                "id": ann_id,
            }
    return expected


def existing_crops(
    output_dir: str, crop_dirs: Tuple[str, str, str] = settings.TRAIN_CROP_DIRS,
) -> Set[str]:
    """Return the set of existing crop image paths under the crop folders.

    Zero-byte files are treated as not present, so a previously botched (empty)
    crop is regenerated on the next sync rather than lingering.
    """
    found: Set[str] = set()
    for split_dir in crop_dirs:
        base = os.path.join(output_dir, split_dir)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for name in files:
                if not name.lower().endswith(settings.TRAIN_CROP_IMAGE_EXTS):
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getsize(path) == 0:
                        continue
                except OSError:
                    continue
                found.add(path)
    return found


def _is_unreadable_image(path: str) -> bool:
    """Whether an image file is empty or cannot be decoded by PIL.

    Used to skip corrupt/zero-byte crops (e.g. produced from a truncated or
    0-byte source image) so they do not crash training/evaluation.
    """
    try:
        if os.path.getsize(path) == 0:
            return True
        with Image.open(path) as im:
            im.verify()
        return False
    except (OSError, ValueError):
        return True


def filter_readable_images(files: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Partition image paths into (readable, unreadable).

    Args:
        files: Image file paths (str or path-like).

    Returns:
        Tuple ``(good, bad)`` preserving input order.
    """
    good: List[Any] = []
    bad: List[Any] = []
    for fp in tqdm(files, desc="Checking crops", unit="img"):
        if _is_unreadable_image(str(fp)):
            bad.append(fp)
        else:
            good.append(fp)
    return good, bad


def generate_crop(
    cam_filepath: str, cam_x: float, cam_y: float, crop_size: int, out_path: str
) -> bool:
    """Write a single square crop centred on ``(cam_x, cam_y)``.

    Args:
        cam_filepath: Source image path.
        cam_x: Crop centre x (pixels).
        cam_y: Crop centre y (pixels).
        crop_size: Square crop width/height (pixels).
        out_path: Destination JPEG path.

    Returns:
        True on success, False if the source image is missing/unreadable or
        the crop falls outside the image bounds.

    Note:
        Delegates the actual centred crop to
        :func:`substrata.visualizations.get_crop_img` (the shared pixel-centred
        crop helper) rather than reimplementing the box maths here.
    """
    from substrata.visualizations import get_crop_img  # lazy heavy import

    if not os.path.isfile(cam_filepath):
        logger.warning("Source image not found, skipping crop: %s", cam_filepath)
        return False
    try:
        if os.path.getsize(cam_filepath) == 0:
            logger.warning("Source image is empty, skipping crop: %s", cam_filepath)
            return False
    except OSError:
        return False
    try:
        crop = get_crop_img(cam_filepath, cam_x, cam_y, crop_size, crop_size)
        crop = crop.convert("RGB")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        crop.save(out_path, "JPEG", quality=settings.TRAIN_CROP_JPEG_QUALITY)
    except (OSError, ValueError) as e:
        logger.warning("Could not crop %s: %s", cam_filepath, e)
        # Remove any partial/empty output so it is not treated as a valid crop.
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return False
    return True


def sync_crops(
    annotations_path: str, output_dir: str, crop_size: int,
    split: Tuple[int, int, int] = settings.TRAIN_SPLIT,
    crop_dirs: Tuple[str, str, str] = settings.TRAIN_CROP_DIRS,
    delete_stale: bool = True, prompt: bool = True,
    n_jobs: int = settings.TRAIN_CROP_JOBS,
) -> Dict[str, int]:
    """Incrementally regenerate crops to match the annotations.

    Deletes crops that no longer correspond to any annotation (after optional
    confirmation) and generates only the crops not already present. Crop
    generation runs in parallel (each crop is independent and writes its own
    file).

    Args:
        annotations_path: Consolidated training annotations CSV.
        output_dir: Base output directory.
        crop_size: Square crop size in pixels.
        split: Train/validation/test percentages.
        crop_dirs: Folder names for the three splits.
        delete_stale: Whether to remove redundant crops.
        prompt: Whether to confirm deletion interactively.
        n_jobs: Parallel workers for crop generation (-1 = all cores).

    Returns:
        Stats dict with ``generated``, ``skipped_existing``, ``deleted``,
        ``failed`` counts.
    """
    expected = plan_crops(annotations_path, output_dir, split, crop_dirs)
    existing = existing_crops(output_dir, crop_dirs)

    stale = sorted(existing - set(expected))
    deleted = 0
    removed_dirs = 0
    if stale and delete_stale:
        print(f"\n{len(stale)} redundant crop(s) no longer match any annotation.")
        # Show a few examples before deleting: if the wrong output directory
        # was given, this surfaces it before any files are removed.
        n_preview = min(settings.TRAIN_DELETE_PREVIEW, len(stale))
        print(f"Examples of files that would be deleted (showing {n_preview}):")
        for p in stale[:n_preview]:
            print(f"  {p}")
        do_delete = True
        if prompt:
            ans = input(f"Delete {len(stale)} redundant crop(s)? [y/N]: ").strip()
            do_delete = ans.lower() in ("y", "yes")
        if do_delete:
            for p in stale:
                try:
                    os.remove(p)
                    deleted += 1
                except OSError as e:
                    logger.warning("Could not delete %s: %s", p, e)
            removed_dirs = _remove_empty_category_dirs(output_dir, crop_dirs)

    to_make = sorted(set(expected) - existing)
    generated = 0
    failed = 0
    if to_make:
        from joblib import Parallel, delayed

        from substrata.logging import tqdm_joblib

        # Threads (not processes): generate_crop is dominated by JPEG
        # decode/encode and disk I/O, which release the GIL, and a thread
        # backend avoids re-importing the heavy visualizations module (and
        # pickling args) per worker.
        with tqdm_joblib(
            tqdm(total=len(to_make), desc="Generating crops", unit="crop")
        ):
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(generate_crop)(
                    expected[p]["cam_filepath"], expected[p]["cam_x"],
                    expected[p]["cam_y"], crop_size, p,
                )
                for p in to_make
            )
        generated = sum(1 for r in results if r)
        failed = len(results) - generated

    return {
        "generated": generated,
        "skipped_existing": len(expected) - len(to_make),
        "deleted": deleted,
        "removed_dirs": removed_dirs,
        "failed": failed,
    }


def _remove_empty_category_dirs(
    output_dir: str, crop_dirs: Tuple[str, str, str],
) -> int:
    """Remove now-empty category subfolders under each split crop directory.

    After stale crops are deleted a label folder can be left empty (e.g. a
    category that no longer has any annotations); these are removed so the crop
    tree only contains categories that are actually present.

    Returns:
        Number of empty category directories removed.
    """
    removed = 0
    for split_dir in crop_dirs:
        base = os.path.join(output_dir, split_dir)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            cat_dir = os.path.join(base, name)
            if os.path.isdir(cat_dir) and not os.listdir(cat_dir):
                try:
                    os.rmdir(cat_dir)
                    removed += 1
                except OSError as e:
                    logger.warning("Could not remove empty dir %s: %s", cat_dir, e)
    return removed


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def count_crops_by_class(
    output_dir: str, label_map: Dict[str, str],
    crop_dirs: Tuple[str, str, str] = settings.TRAIN_CROP_DIRS,
) -> Dict[str, Counter]:
    """Tally crops per training class for each split.

    Walks the crop split folders (named by the raw label), collapses each raw
    label to its training class via ``label_map`` (raw labels absent from the
    map are skipped, matching what training/evaluation use), and counts non-
    empty image files.

    Args:
        output_dir: Base directory holding the crop folders.
        label_map: ``raw_label -> training_class`` mapping.
        crop_dirs: Folder names for the three splits.

    Returns:
        Mapping ``split_dir -> Counter(training_class -> count)``.
    """
    result: Dict[str, Counter] = {split: Counter() for split in crop_dirs}
    for split in crop_dirs:
        base = os.path.join(output_dir, split)
        if not os.path.isdir(base):
            continue
        for raw_label in os.listdir(base):
            cls = label_map.get(raw_label)
            if cls is None:
                continue
            cat_dir = os.path.join(base, raw_label)
            if not os.path.isdir(cat_dir):
                continue
            for name in os.listdir(cat_dir):
                if not name.lower().endswith(settings.TRAIN_CROP_IMAGE_EXTS):
                    continue
                path = os.path.join(cat_dir, name)
                try:
                    if os.path.getsize(path) == 0:
                        continue
                except OSError:
                    continue
                result[split][cls] += 1
    return result


def final_metrics_from_learner(learn: Any) -> Dict[str, Any]:
    """Best-effort extraction of the last-epoch metrics from a fastai learner.

    Reads ``learn.recorder`` defensively (no fastai import needed) and returns a
    ``{metric_name: value}`` dict for the final epoch, or ``{}`` if unavailable.
    """
    out: Dict[str, Any] = {}
    try:
        rec = learn.recorder
        names = [
            n for n in getattr(rec, "metric_names", []) if n not in ("epoch", "time")
        ]
        values = list(getattr(rec, "values", []) or [])
        if names and values:
            for name, val in zip(names, values[-1]):
                try:
                    out[name] = round(float(val), 4)
                except (TypeError, ValueError):
                    out[name] = val
    except Exception:  # noqa: BLE001 - summary is best-effort, never fatal
        return {}
    return out


def write_training_summary_pdf(
    pdf_path: str, info: List[Tuple[str, Any]], counts_by_split: Dict[str, Counter],
    crop_dirs: Tuple[str, str, str] = settings.TRAIN_CROP_DIRS,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a training-run summary to a multi-page PDF.

    Page 1 is a text summary of the run settings, the per-split crop totals and
    (if given) the final metrics. The remaining pages show a horizontal stacked
    bar chart of per-training-class crop counts (train/validation/test),
    paginated ``settings.TRAIN_SUMMARY_CLASSES_PER_PAGE`` classes at a time and
    sorted by total count.

    Args:
        pdf_path: Destination PDF path.
        info: Ordered ``(label, value)`` rows of run settings to print.
        counts_by_split: ``split_dir -> Counter(class -> count)`` (see
            :func:`count_crops_by_class`).
        crop_dirs: Folder names for the three splits (train/val/test order).
        metrics: Optional ``{metric_name: value}`` from the final epoch.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    train_name, valid_name, test_name = crop_dirs
    classes = sorted(
        set().union(*[set(c) for c in counts_by_split.values()])
        if counts_by_split else set(),
        key=lambda c: (
            -sum(counts_by_split.get(s, {}).get(c, 0) for s in crop_dirs), c
        ),
    )

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Page 1: settings + totals + final metrics.
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle("Training summary", fontsize=12, y=0.98)
        lines = [f"{k}: {v}" for k, v in info]
        lines.append("")
        lines.append("Crops per split:")
        for split in crop_dirs:
            tot = sum(counts_by_split.get(split, {}).values())
            lines.append(f"  {split}: {tot}")
        grand_total = sum(
            sum(counts_by_split.get(s, {}).values()) for s in crop_dirs
        )
        lines.append(f"  total: {grand_total}")
        if metrics:
            lines.append("")
            lines.append("Final-epoch metrics:")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")
        ax.text(
            0.02, 0.96, "\n".join(lines), transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Remaining pages: per-class stacked bar chart of crop counts.
        per_page = settings.TRAIN_SUMMARY_CLASSES_PER_PAGE
        for start in range(0, len(classes), per_page):
            page = classes[start:start + per_page]
            tr = np.array([counts_by_split.get(train_name, {}).get(c, 0) for c in page])
            va = np.array([counts_by_split.get(valid_name, {}).get(c, 0) for c in page])
            te = np.array([counts_by_split.get(test_name, {}).get(c, 0) for c in page])
            y = np.arange(len(page))
            fig, ax = plt.subplots(figsize=(8.5, max(3, len(page) * 0.25 + 1)))
            ax.barh(y, tr, color="#4c72b0", label=train_name)
            ax.barh(y, va, left=tr, color="#dd8452", label=valid_name)
            ax.barh(y, te, left=tr + va, color="#55a868", label=test_name)
            ax.set_yticks(y)
            ax.set_yticklabels(page, fontsize=6)
            ax.invert_yaxis()  # largest class on top
            ax.set_xlabel("Crops")
            ax.set_title("Crops per training class")
            ax.legend(fontsize=7, loc="lower right")
            totals = tr + va + te
            for i, t in enumerate(totals):
                ax.text(t, i, f" {int(t)}", va="center", fontsize=6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    logger.info("Wrote training summary PDF: %s", pdf_path)


def train_classifier(
    output_dir: str, model_path: str, label_map: Dict[str, str],
    arch: str = settings.TRAIN_DEFAULT_ARCH,
    epochs: int = settings.TRAIN_DEFAULT_EPOCHS,
    crop_dirs: Tuple[str, str, str] = settings.TRAIN_CROP_DIRS,
) -> Any:
    """Train a fastai image classifier on the generated crops.

    Builds a DataBlock over the training/validation crop folders (the test
    folder is excluded from items), fine-tunes the model, and exports the
    learner to ``model_path``. Crop folders are named by the *raw* label;
    ``label_map`` collapses each raw label into its training class (via
    ``get_y``) and crops whose raw label is excluded (absent from ``label_map``)
    are dropped from the items, so changing the considered categories needs only
    a different map - not regenerated crops.

    Args:
        output_dir: Base directory holding the crop folders.
        model_path: Destination ``.pkl`` path for the exported learner.
        label_map: ``raw_label -> training_class`` mapping (excluded labels
            absent); see :func:`get_label_collapse_map` / :func:`load_label_map`.
        arch: torchvision architecture name (e.g. ``resnet34``).
        epochs: Number of fine-tuning epochs.
        crop_dirs: Folder names for the three splits.

    Returns:
        The trained fastai Learner.
    """
    import torchvision  # lazy
    from fastai.vision.all import (
        DataBlock, ImageBlock, CategoryBlock, GrandparentSplitter,
        get_image_files, parent_label, vision_learner, accuracy, error_rate,
        Resize,
    )

    train_name, valid_name, _test_name = crop_dirs

    def _items(path):
        # Only include train/validation folders; exclude the test folder.
        files = get_image_files(path, folders=[train_name, valid_name])
        # Drop crops whose raw label is excluded from training (not in the map).
        files = [f for f in files if parent_label(f) in label_map]
        # Drop empty/corrupt crops so a single bad image can't crash training.
        good, bad = filter_readable_images(list(files))
        if bad:
            logger.warning(
                "Ignoring %d empty/unreadable crop image(s) during training.",
                len(bad),
            )
            print(f"Ignored {len(bad)} empty/unreadable crop image(s).")
        return good

    def _get_y(o):
        # Collapse the raw-label folder name into its training class.
        return label_map[parent_label(o)]

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=_items,
        get_y=_get_y,
        splitter=GrandparentSplitter(
            train_name=train_name, valid_name=valid_name
        ),
        item_tfms=Resize(settings.TRAIN_IMAGE_SIZE),
    )
    dls = dblock.dataloaders(output_dir)

    arch_fn = getattr(torchvision.models, arch)
    learn = vision_learner(dls, arch_fn, metrics=[accuracy, error_rate])
    learn.fine_tune(epochs)
    learn.export(model_path)
    logger.info("Exported trained learner to %s", model_path)
    return learn


def _write_example_pages(pdf, plt, np, labels, examples):
    """Append example-classification pages to an open ``PdfPages``.

    One row per category (in ``labels`` order, matching the confusion matrix),
    with up to ``settings.TRAIN_EXAMPLES_PER_CLASS`` example crops as columns.
    Each crop's title is the predicted label; misclassified crops get a red
    title and a red border. Rows are paginated
    ``settings.TRAIN_EXAMPLE_ROWS_PER_PAGE`` categories at a time.
    """
    per_class = settings.TRAIN_EXAMPLES_PER_CLASS
    rows_per_page = settings.TRAIN_EXAMPLE_ROWS_PER_PAGE

    # Keep confusion-matrix order; skip labels with no example crops.
    cats = [lab for lab in labels if examples.get(lab)]
    if not cats:
        return

    for start in range(0, len(cats), rows_per_page):
        page_cats = cats[start:start + rows_per_page]
        fig, axes = plt.subplots(
            len(page_cats), per_class,
            figsize=(per_class * 1.1 + 1.0, len(page_cats) * 1.25),
            squeeze=False,
        )
        fig.suptitle(
            "Example classifications (title = prediction; red = misclassified)",
            fontsize=11, y=0.99,
        )
        for r, cat in enumerate(page_cats):
            samples = examples.get(cat, [])[:per_class]
            for c in range(per_class):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(
                        cat, rotation=0, ha="right", va="center", fontsize=7,
                    )
                if c >= len(samples):
                    ax.axis("off")
                    continue
                fp, pred = samples[c]
                try:
                    im = np.asarray(Image.open(fp).convert("RGB"))
                    ax.imshow(im)
                except OSError:
                    ax.text(
                        0.5, 0.5, "(unreadable)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=5,
                    )
                wrong = pred != cat
                ax.set_title(
                    pred, fontsize=5, color="red" if wrong else "black",
                )
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("red" if wrong else "0.6")
                    spine.set_linewidth(2.0 if wrong else 0.5)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)


def _write_stats_pdf(
    pdf_path: str, split_folder: str, n: int, acc: float, report: str,
    cm: Any, labels: List[str],
    examples: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> None:
    """Render the evaluation stats to a multi-page PDF.

    Page 1 is a text summary (accuracy + per-class precision/recall/F1); page 2
    is the confusion-matrix heatmap, row-normalised so per-class structure stays
    visible despite large differences in class size; the remaining pages show
    example classified crops per category (see :func:`_write_example_pages`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Page 1: text summary.
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle(f"Classifier stats - {split_folder}", fontsize=12, y=0.98)
        text = (
            f"Crops evaluated: {n}\n"
            f"Overall accuracy: {acc:.4f}\n\n"
            f"{report}"
        )
        ax.text(
            0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: confusion matrix heatmap. Colour by row-normalised values
        # (fraction of each true class) so the diagonal and per-class errors
        # stay visible regardless of how many images back each class; a few
        # huge classes would otherwise saturate a raw-count colour scale and
        # wash out every other cell.
        cm = np.asarray(cm)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm, row_sums, out=np.zeros(cm.shape, dtype=float),
            where=row_sums != 0,
        )

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5),
                                        max(5, len(labels) * 0.5)))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        # Keep absolute counts in view via the y-axis (n = images per class).
        ax.set_yticklabels(
            [f"{lab} (n={int(row_sums[i])})" for i, lab in enumerate(labels)],
            fontsize=6,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            "Confusion matrix (row-normalised; n = images per true class)"
        )
        fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class"
        )

        # Annotate raw counts only when there are few enough classes to read.
        if len(labels) <= settings.TRAIN_CM_ANNOTATE_MAX:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    count = int(cm[i, j])
                    if count == 0:
                        continue
                    ax.text(
                        j, i, str(count), ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                    )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Remaining pages: example classified crops per category.
        if examples:
            _write_example_pages(pdf, plt, np, labels, examples)
    logger.info("Wrote classifier stats PDF: %s", pdf_path)


def report_classifier_stats(
    classifier: Union[str, Any], crops_dir: str, split_folder: str,
    label_map: Dict[str, str], pdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a classifier over a crop split folder and report stats.

    Prints overall accuracy, a per-class precision/recall/F1 report and the
    confusion matrix, and optionally writes the same to a PDF. Crop folders are
    named by the raw label; ``label_map`` collapses each into its training class
    so evaluation classes match those the model was trained on, and crops whose
    raw label is excluded (absent from ``label_map``) are skipped.

    Args:
        classifier: A loaded fastai Learner or a path to a ``.pkl`` learner.
        crops_dir: Base directory holding the crop folders.
        split_folder: Which split to evaluate (e.g. ``validation_crops`` or
            ``test_crops``).
        label_map: ``raw_label -> training_class`` mapping (excluded labels
            absent); see :func:`load_label_map`.
        pdf_path: Optional path to write a stats PDF to.

    Returns:
        Dict with ``accuracy``, ``report`` (sklearn text), ``n``, the sorted
        label set, and ``pdf_path`` (or None).
    """
    from fastai.vision.all import get_image_files
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
    )

    learn = _ensure_learner(classifier)
    split_dir = os.path.join(crops_dir, split_folder)
    files = get_image_files(split_dir)
    if not files:
        raise SystemExit(f"No crops found under {split_dir}.")

    # Skip crops whose raw label is excluded from training (not in the map);
    # otherwise their collapsed true label would be undefined.
    files = [
        f for f in files
        if os.path.basename(os.path.dirname(str(f))) in label_map
    ]
    if not files:
        raise SystemExit(
            f"No crops under {split_dir} match the label map "
            "(all excluded). Check training_label_map.csv."
        )

    # Drop empty/corrupt crops so a single bad image can't crash evaluation.
    files, bad = filter_readable_images(list(files))
    if bad:
        logger.warning(
            "Ignoring %d empty/unreadable crop image(s) during evaluation.",
            len(bad),
        )
        print(f"Ignored {len(bad)} empty/unreadable crop image(s).")
    if not files:
        raise SystemExit(f"No readable crops under {split_dir}.")

    y_true: List[str] = []
    y_pred: List[str] = []
    # Per true-class example crops (path, predicted label) for the PDF gallery.
    examples: Dict[str, List[Tuple[str, str]]] = {}
    per_class = settings.TRAIN_EXAMPLES_PER_CLASS
    for fp in tqdm(files, desc=f"Evaluating {split_folder}", unit="crop"):
        raw_label = os.path.basename(os.path.dirname(str(fp)))
        true_label = label_map[raw_label]
        img = Image.open(str(fp)).convert("RGB")
        pred_class, _idx, _probs = learn.predict(img)
        y_true.append(true_label)
        y_pred.append(str(pred_class))
        bucket = examples.setdefault(true_label, [])
        if len(bucket) < per_class:
            bucket.append((str(fp), str(pred_class)))

    labels = sorted(set(y_true) | set(y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n=== Stats on {split_folder} ({len(files)} crops) ===")
    print(f"Overall accuracy: {acc:.4f}\n")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)

    if pdf_path is not None:
        _write_stats_pdf(
            pdf_path, split_folder, len(files), acc, report, cm, labels,
            examples=examples,
        )

    return {
        "accuracy": acc, "report": report, "n": len(files),
        "labels": labels, "pdf_path": pdf_path,
    }
