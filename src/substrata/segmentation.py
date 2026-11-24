# Standard Library
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Libraries
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# NOTE: ``torch`` and ``sam2`` are heavy/optional and are imported lazily (inside
# the functions that need them, via ``_require_sam2``) so that importing this
# module — and therefore ``substrata`` as a whole — does not require them. This is
# what lets ``segmentation`` be added to the package's star-imports.

# Local Modules
from substrata.logging import logger

# Only the point-cloud segmentation API is star-exported into the flat
# ``substrata.*`` namespace (this module is in ``__init__``'s star-import list).
# The SAM2 image helpers below (including a ``Mask`` class that would otherwise
# collide with ``cameras.Mask``) stay accessible via ``substrata.segmentation.*``
# but are deliberately kept out of ``__all__``.
__all__ = [
    "Segmentation",
    "sample_query_points",
    "segment_point_cloud",
    "recolor_ply_file",
]


class Mask:
    """
    Mask object holding the output of SAM2 prediction
    """

    def __init__(self, mask, score, logits):
        self.vals = mask
        self.score = score
        self.logits = logits
        self.area_in_px = cv2.countNonZero(self.vals)
        self.area_in_cm2 = None


def get_sam2_predictor(sam2_checkpoint=None, model_cfg=None):
    """
    Initialize SAM2 segmentation options

    from https://github.com/facebookresearch/sam2

    Args:
        sam2_checkpoint: Path to the SAM2 checkpoint (default
            ``settings.SAM2_CHECKPOINT``).
        model_cfg: SAM2 model config (default ``settings.SAM2_MODEL_CFG``).
    """
    import torch  # lazy: heavy/optional, not required to import the package
    from substrata import settings

    if sam2_checkpoint is None:
        sam2_checkpoint = settings.SAM2_CHECKPOINT
    if model_cfg is None:
        model_cfg = settings.SAM2_MODEL_CFG

    build_sam2, SAM2ImagePredictor = _require_sam2()

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def show_mask(mask, ax, random_color=False, borders=True):
    """
    Show the mask on the image
    from https://github.com/facebookresearch/sam2
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    Show the annotated points on the image
    from https://github.com/facebookresearch/sam2
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker=".",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker=".",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    """
    Show the annotated box on the image
    from https://github.com/facebookresearch/sam2
    """
    import matplotlib.pyplot as plt  # lazy: figures only

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    """
    Show the image masks
    from https://github.com/facebookresearch/sam2
    """
    import matplotlib.pyplot as plt  # lazy: figures only

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def get_sam2_masks(
    image_filepath, pixel_x, pixel_y, sam_predictor=None, visualize=False
):
    """
    Get a SAM2 prediction of a single point and return Mask instances.
    """
    if sam_predictor is None:
        # user didn't pass a predictor → build on demand
        sam_predictor = get_sam2_predictor()

    image = Image.open(image_filepath)
    image = np.array(image.convert("RGB"))

    sam_predictor.set_image(image)
    masks, scores, logits = sam_predictor.predict(
        point_coords=np.array([[pixel_x, pixel_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    # Sort masks based on scores (descending order)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # Create Mask instances
    mask_objects = [Mask(masks[i], scores[i], logits[i]) for i in range(len(masks))]

    if visualize:
        show_masks(
            image,
            masks,
            scores,
            point_coords=np.array([[pixel_x, pixel_y]]),
            input_labels=np.array([1]),
            borders=True,
        )

    return mask_objects


def get_sam2_masks_multi(image_filepath, pixels, sam_predictor, max_dim=None):
    """
    Get SAM2 masks for several single-point prompts on the same image.

    The image is embedded once (``set_image``) and each ``(x, y)`` prompt is then
    predicted separately, which is much faster than calling
    :func:`get_sam2_masks` per point. The image is downscaled to ``max_dim``
    before embedding (SAM2 works at 1024 px internally), so the returned masks
    are at reduced resolution: each ``Mask`` gets a ``scale`` attribute (full-res
    pixels per mask pixel) to convert full-res pixel coordinates into mask
    indices (``mask.vals[int(y / scale), int(x / scale)]``).

    Args:
        image_filepath: Path to the image.
        pixels: Sequence of ``(x, y)`` prompts in full-resolution pixels.
        sam_predictor: A predictor from :func:`get_sam2_predictor`.
        max_dim: Max image dimension used for embedding (default
            ``settings.SAM2_MAX_IMAGE_DIM``).

    Returns:
        One list of ``Mask`` objects per prompt (sorted by score, highest first).
    """
    from substrata import settings

    if max_dim is None:
        max_dim = settings.SAM2_MAX_IMAGE_DIM
    image = Image.open(image_filepath)
    full_w, full_h = image.size
    scale = max(1.0, max(full_w, full_h) / float(max_dim))
    if scale > 1.0:
        target = (int(round(full_w / scale)), int(round(full_h / scale)))
        image.draft("RGB", target)  # fast JPEG DCT downscaling (no-op otherwise)
        image = image.convert("RGB").resize(target, Image.BILINEAR)
    image = np.array(image.convert("RGB"))
    scale = full_w / float(image.shape[1])

    sam_predictor.set_image(image)
    results = []
    for x, y in pixels:
        masks, scores, logits = sam_predictor.predict(
            point_coords=np.array([[x / scale, y / scale]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        order = np.argsort(scores)[::-1]
        mask_objects = []
        for i in order:
            m = Mask(masks[i].astype(np.uint8), scores[i], logits[i])
            m.scale = scale
            mask_objects.append(m)
        results.append(mask_objects)
    return results


def _require_sam2():
    """Import SAM2 only when needed."""
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception as e:
        raise ImportError(
            "SAM2 is not installed. Install it only if you use SAM2 features:\n"
            "  pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'\n"
            "or follow the project’s installation instructions."
        ) from e
    return build_sam2, SAM2ImagePredictor


# ===========================================================================
# Point-cloud segmentation via image-match classification
# ===========================================================================
#
# Classify/segment a whole point cloud by reusing the trained per-annotation
# crop classifier: sample a grid of query points across the cloud, project each
# to its best camera photo, cut a 224 px patch, classify it, then propagate the
# labels to every cloud point via nearest neighbour. Unlike the SAM2 code above
# this is a 3-D point-cloud operation; its only heavy dependency (fastai) is
# imported lazily, so this module stays importable without torch/sam2.

def _parse_hex(s: str) -> Tuple[int, int, int]:
    """``#rrggbb`` (or ``rrggbb``) as a 0-255 ``(r, g, b)`` tuple."""
    h = str(s).strip().lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def build_label_colors(
    codebook: List[str], overrides: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> Dict[str, Tuple[int, int, int]]:
    """Build a ``{label: (r, g, b)}`` 0-255 map (tab20 defaults + manual overrides).

    Mirrors the ``label_colors``/``tab20`` convention used across ortho and
    annotations rendering. ``overrides`` (e.g. ``{"CS": (255, 0, 0)}``) win over
    the auto colours; labels not in ``overrides`` get a stable tab20 colour.
    """
    import matplotlib.pyplot as plt  # lazy: colour map only

    cmap = plt.get_cmap("tab20")
    colors = {
        lbl: tuple(int(255 * c) for c in cmap(i % 20)[:3])
        for i, lbl in enumerate(codebook)
    }
    if overrides:
        colors.update({k: tuple(v) for k, v in overrides.items() if k in codebook})
    return colors


_build_label_colors = build_label_colors  # deprecated alias (renamed 2026-11-25)


@dataclass
class Segmentation:
    """A point-cloud segmentation result — standalone, not attached to a cloud.

    The source of truth is the small set of classified *query points*
    (``query_coords`` + ``query_labels``); per-point labels for a full cloud are
    a derived cache produced on demand by :meth:`propagate`. Persisted compactly
    to ``.npz`` so it costs zero memory when not loaded.

    Attributes:
        query_coords: ``(Nq, 3)`` float32 world-frame coordinates of classified
            query points (kept only where a label was assigned).
        query_labels: ``(Nq,)`` object array of category label strings.
        query_conf: ``(Nq,)`` float classifier confidences.
        codebook: ordered unique labels; the integer code of a label is its index.
        label_colors: ``{label: (r, g, b)}`` 0-255 display colours.
        point_codes: optional ``(N,)`` int cache of per-point label codes for a
            specific cloud (``-1`` = unlabeled). Derived; not persisted by default.
        query_cams: optional ``(Nq,)`` object array of the photo each query was
            classified from (``""`` if unknown).
        query_px: optional ``(Nq, 2)`` float pixel (x, y) of each query in that
            photo (``nan`` if unknown).
        query_n_views: optional ``(Nq,)`` int number of photos (views) fused
            into each query's prediction (point-feature classifiers).
        query_agreement: optional ``(Nq,)`` float weight share of those views
            whose own label is the fused one (1 = all views agree).
    """

    query_coords: np.ndarray
    query_labels: np.ndarray
    query_conf: np.ndarray
    codebook: List[str]
    label_colors: Dict[str, Tuple[int, int, int]]
    point_codes: Optional[np.ndarray] = None
    query_cams: Optional[np.ndarray] = None
    query_px: Optional[np.ndarray] = None
    query_n_views: Optional[np.ndarray] = None
    query_agreement: Optional[np.ndarray] = None
    _tree: Any = field(default=None, repr=False, compare=False)
    _codes: Any = field(default=None, repr=False, compare=False)

    @property
    def n_queries(self) -> int:
        return len(self.query_coords)

    @classmethod
    def from_query_labels(
        cls,
        query_coords: np.ndarray,
        labels: np.ndarray,
        conf: np.ndarray,
        label_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        cams: Optional[np.ndarray] = None,
        px: Optional[np.ndarray] = None,
        n_views: Optional[np.ndarray] = None,
        agreement: Optional[np.ndarray] = None,
    ) -> "Segmentation":
        """Build a Segmentation, keeping only successfully-classified queries.

        ``cams`` / ``px``: optional photo filepath and pixel (x, y) per query
        (see :attr:`query_cams` / :attr:`query_px`); ``n_views`` /
        ``agreement``: optional multi-view fusion details
        (:attr:`query_n_views` / :attr:`query_agreement`).
        """
        matched = np.array([bool(l) for l in labels])
        if not matched.any():
            raise ValueError(
                "No query points were classified. Nothing to segment — most "
                "likely the camera photo filepaths do not resolve on disk "
                "(check cams.set_filepath_replace(old, new))."
            )
        qc = np.asarray(query_coords)[matched].astype(np.float32)
        ql = np.asarray(labels, dtype=object)[matched]
        qcf = np.asarray(conf, dtype=float)[matched]
        codebook = sorted(set(ql.tolist()))
        colors = build_label_colors(codebook, label_colors)
        qcam = None if cams is None else np.asarray(cams, dtype=object)[matched]
        qpx = None if px is None else np.asarray(px, dtype=float)[matched]
        qnv = None if n_views is None else np.asarray(n_views, dtype=int)[matched]
        qag = None if agreement is None else np.asarray(agreement, float)[matched]
        return cls(qc, ql, qcf, codebook, colors, query_cams=qcam, query_px=qpx,
                   query_n_views=qnv, query_agreement=qag)

    @classmethod
    def from_annotations(
        cls, csv_path: str, world_transform: Optional[np.ndarray] = None
    ) -> "Segmentation":
        """Read a segmentation from its annotation set (``<id>_seg[_<tag>]_points.csv``).

        The CSV written by :meth:`to_annotations` (with its sidecar) is the
        segmentation's source of truth. A point's label is its ``label`` when set
        (checked by hand, confidence 1), else its ``pred_label`` (``pred_conf``);
        points with neither are left out. Colours come from the sidecar's
        ``segmentation.colors``; labels not in it get a tab20 colour.

        Args:
            csv_path: The annotations CSV.
            world_transform: If given, the 4x4 transform applied to the CSV's
                orig coords (e.g. the plot's current ``pcd.world_transform``);
                otherwise the query points stay in the orig (raw PLY) frame.

        Returns:
            A :class:`Segmentation` (with :attr:`query_cams` / :attr:`query_px`).
        """
        import csv

        from substrata import sidecars

        with open(csv_path, newline="") as f:
            rows = csv.reader(f)
            head = [h.strip().strip('"') for h in next(rows)]
            col = {h: i for i, h in enumerate(head)}
            for need in ("orig_x", "orig_y", "orig_z"):
                if need not in col:
                    raise ValueError(f"{csv_path}: no {need} column")

            def get(r, name):
                i = col.get(name)
                return r[i].strip() if i is not None and i < len(r) else ""

            xyz, labels, conf, cams, px, nviews, agree = [], [], [], [], [], [], []
            for r in rows:
                if not r:
                    continue
                label, c = get(r, "label"), 1.0
                if not label:
                    label, c = get(r, "pred_label"), get(r, "pred_conf")
                    c = float(c) if c else np.nan
                if not label:
                    continue
                xyz.append([float(get(r, k)) for k in ("orig_x", "orig_y", "orig_z")])
                labels.append(label)
                conf.append(c)
                cams.append(get(r, "cam_filepath"))
                x, y = get(r, "cam_x"), get(r, "cam_y")
                px.append([float(x) if x else np.nan, float(y) if y else np.nan])
                nv, ag = get(r, "pred_n"), get(r, "pred_agreement")
                nviews.append(int(float(nv)) if nv else 0)
                agree.append(float(ag) if ag else np.nan)
        if not labels:
            raise ValueError(f"{csv_path}: no labelled or predicted points")
        qc = np.asarray(xyz, dtype=float)
        if world_transform is not None:
            wt = np.asarray(world_transform, dtype=float)
            qc = qc @ wt[:3, :3].T + wt[:3, 3]
        ql = np.array(labels, dtype=object)
        codebook = sorted(set(labels))
        hex_colors = (
            sidecars.read_annotation_sidecar(csv_path).get("segmentation") or {}
        ).get("colors") or {}
        colors = build_label_colors(
            codebook, {k: _parse_hex(v) for k, v in hex_colors.items()}
        )
        return cls(
            qc.astype(np.float32), ql, np.asarray(conf, dtype=float), codebook,
            colors, query_cams=np.array(cams, dtype=object),
            query_px=np.asarray(px, dtype=float),
            query_n_views=(np.array(nviews, dtype=int)
                           if "pred_n" in col else None),
            query_agreement=(np.array(agree, dtype=float)
                             if "pred_agreement" in col else None),
        )

    def hex_colors(self) -> Dict[str, str]:
        """The label colours as ``{label: "#rrggbb"}`` (for the sidecar)."""
        return {
            lbl: "#{:02x}{:02x}{:02x}".format(*self.label_colors[lbl])
            for lbl in self.codebook
        }

    def _query_tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.query_coords)
        return self._tree

    def propagate(
        self, points: np.ndarray, max_radius: Optional[float] = None
    ) -> np.ndarray:
        """Assign each point the label of its nearest classified query.

        Args:
            points: ``(N, 3)`` world-frame coordinates.
            max_radius: if given, points farther than this from any query are
                left unlabeled (``-1``); default ``None`` labels every point
                (gap-free).

        Returns:
            ``(N,)`` int16 array of label codes (``-1`` = unlabeled).
        """
        if self.n_queries == 0:
            raise ValueError("Segmentation has no query points to propagate.")
        if self._codes is None:  # label code of each query (cached: propagate runs per chunk)
            index = {l: i for i, l in enumerate(self.codebook)}
            self._codes = np.array(
                [index[l] for l in self.query_labels], dtype=np.int16
            )
        code_of_query = self._codes
        dist, idx = self._query_tree().query(points, k=1, workers=-1)
        codes = code_of_query[idx].astype(np.int16)
        if max_radius is not None:
            codes[dist > max_radius] = -1
        return codes

    def recolor(
        self,
        orig_colors: np.ndarray,
        point_codes: Optional[np.ndarray] = None,
        value_floor: float = 0.3,
        unlabeled: str = "gray",
    ) -> np.ndarray:
        """Blend each point's category colour with its original luminance.

        The category base colour is modulated by the point's own luminance
        (floored at ``value_floor`` so the tint stays visible in dark areas), so
        surface relief survives the recolouring. Returns ``(N, 3)`` 0-1 RGB.

        Args:
            orig_colors: ``(N, 3)`` original colours (0-1 or 0-255).
            point_codes: per-point label codes; defaults to ``self.point_codes``.
            value_floor: minimum luminance scaling for labeled points.
            unlabeled: ``"gray"`` (dim grayscale) or ``"keep"`` (original colour).
        """
        if point_codes is None:
            point_codes = self.point_codes
        if point_codes is None:
            raise ValueError("No point_codes given; call propagate() first.")
        oc = np.asarray(orig_colors, dtype=float)
        if oc.size and oc.max() > 1.0:
            oc = oc / 255.0
        lum = oc @ np.array([0.299, 0.587, 0.114])
        v = value_floor + (1.0 - value_floor) * lum
        palette = (
            np.array([self.label_colors[l] for l in self.codebook], dtype=float)
            / 255.0
        )
        out = np.zeros_like(oc)
        labeled = point_codes >= 0
        out[labeled] = palette[point_codes[labeled]] * v[labeled, None]
        if unlabeled == "gray":
            out[~labeled] = (lum[~labeled] * 0.6)[:, None]
        else:  # "keep": original colour
            out[~labeled] = oc[~labeled]
        return np.clip(out, 0.0, 1.0)

    def to_annotations(
        self, world_transform: Optional[np.ndarray] = None, id_prefix: str = "seg"
    ):
        """The classified query points as an annotation set.

        Each point becomes an annotation ``<id_prefix>_<i>`` whose prediction is
        in ``pred_label`` / ``pred_conf`` (``pred_n`` = the number of photos
        fused, 1 for one crop; with multi-view fusion also ``pred_agreement``,
        the weight share of the views agreeing with the prediction); its
        ``label`` stays empty, as for :meth:`Annotations.predict`, so the set can
        be checked by hand. The photo and pixel it was classified from (if
        known) go in ``cam_filepath`` / ``cam_x`` / ``cam_y``. Read it back
        with :meth:`from_annotations`.

        Args:
            world_transform: The 4x4 transform of the frame the query points are
                in (``pcd.world_transform``), to also store orig coords.
            id_prefix: Prefix of the annotation ids (e.g. the project id).

        Returns:
            An :class:`~substrata.annotations.Annotations`.
        """
        from substrata.annotations import Annotation, Annotations

        anns = Annotations()
        for i, (xyz, label, conf) in enumerate(
            zip(self.query_coords, self.query_labels, self.query_conf)
        ):
            if not label:
                continue
            ann = Annotation(np.asarray(xyz, float), id=f"{id_prefix}_{i}")
            ann.label, ann.label_conf = "", ""
            n_views = 1
            if self.query_n_views is not None:
                n_views = int(self.query_n_views[i])
            ann.pred_label, ann.pred_conf, ann.pred_n = str(label), float(conf), n_views
            if self.query_agreement is not None and np.isfinite(
                self.query_agreement[i]
            ):
                ann.meta_data["pred_agreement"] = round(
                    float(self.query_agreement[i]), 4)
            if self.query_cams is not None and self.query_cams[i]:
                ann.cam_filepath = str(self.query_cams[i])
                if self.query_px is not None and np.isfinite(self.query_px[i]).all():
                    ann.cam_x, ann.cam_y = (float(v) for v in self.query_px[i])
            anns.append(ann)
        if world_transform is not None and not np.allclose(
            world_transform, np.eye(4)
        ):
            anns.reverse_transform(world_transform)
        return anns

    def summary(self, point_codes: Optional[np.ndarray] = None) -> Dict[str, int]:
        """Return {label: count}. Uses ``point_codes`` if given, else query labels."""
        if point_codes is not None:
            uniq, cnt = np.unique(point_codes[point_codes >= 0], return_counts=True)
            return {self.codebook[c]: int(n) for c, n in zip(uniq, cnt)}
        uniq, cnt = np.unique(self.query_labels, return_counts=True)
        return {str(u): int(n) for u, n in zip(uniq, cnt)}

    def save(self, path: str, include_point_codes: bool = False) -> str:
        """Persist to a compact ``.npz`` (query points are tiny)."""
        payload = dict(
            query_coords=self.query_coords,
            query_labels=np.array(self.query_labels, dtype=object),
            query_conf=self.query_conf,
            codebook=np.array(self.codebook, dtype=object),
            color_values=np.array(
                [self.label_colors[l] for l in self.codebook], dtype=np.int16
            ),
        )
        if include_point_codes and self.point_codes is not None:
            payload["point_codes"] = self.point_codes
        if self.query_n_views is not None:
            payload["query_n_views"] = self.query_n_views
        if self.query_agreement is not None:
            payload["query_agreement"] = self.query_agreement
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str) -> "Segmentation":
        d = np.load(path, allow_pickle=True)
        codebook = d["codebook"].tolist()
        colors = {
            lbl: tuple(int(c) for c in d["color_values"][i])
            for i, lbl in enumerate(codebook)
        }
        return cls(
            d["query_coords"], d["query_labels"], d["query_conf"], codebook, colors,
            point_codes=d["point_codes"] if "point_codes" in d.files else None,
            query_n_views=(d["query_n_views"] if "query_n_views" in d.files
                           else None),
            query_agreement=(d["query_agreement"] if "query_agreement" in d.files
                             else None),
        )


def sample_query_points(
    pcd, cell_size: float, method: str = "voxel", rep: str = "highest"
) -> np.ndarray:
    """Sample one representative query point per cell across the cloud.

    Args:
        pcd: a :class:`~substrata.pointclouds.PointCloud` (uses ``pcd.points`` in
            the world frame, and ``pcd.o3d_pcd`` for voxel downsampling).
        cell_size: cell / voxel side in metres (== target spatial resolution).
        method: ``"voxel"`` (3-D voxel downsample — covers vertical faces and
            overhangs, the default) or ``"xy_grid"`` (one point per top-down XY
            cell — faster, but leaves gaps at oblique viewing angles).
        rep: for ``xy_grid``, ``"highest"`` (max-z surface point, best for camera
            matching) or ``"centroid"``.

    Returns:
        ``(Nq, 3)`` array of query coordinates (world frame).
    """
    pts = np.asarray(pcd.points)
    if method == "voxel":
        down = pcd.o3d_pcd.voxel_down_sample(cell_size)
        return np.asarray(down.points)
    if method != "xy_grid":
        raise ValueError(f"unknown sampling method: {method!r}")

    ix = np.floor(pts[:, 0] / cell_size).astype(np.int64)
    iy = np.floor(pts[:, 1] / cell_size).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    key = ix * (iy.max() + 1) + iy  # unique per XY cell

    if rep == "highest":
        order = np.lexsort((pts[:, 2], key))
        k_sorted = key[order]
        last = np.ones(len(k_sorted), dtype=bool)
        last[:-1] = k_sorted[1:] != k_sorted[:-1]
        return pts[order[last]]

    uniq, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, pts)
    counts = np.bincount(inv, minlength=len(uniq))[:, None]
    return sums / counts


def _match_points_to_cameras_multi(
    query_coords: np.ndarray,
    cams,
    world_transform: Optional[np.ndarray] = None,
    pcd=None,
    occlusion: bool = True,
    views: int = 1,
    intercept_radius: Optional[float] = None,
    discard_threshold: Optional[float] = None,
    verbose: bool = True,
    include_disabled: bool = False,
    return_available: bool = False,
):
    """Find each query point's ``views`` best cameras (vectorized, occlusion-aware).

    For every camera, project *all* query points at once
    (:meth:`~substrata.cameras.Camera.project_points`) and keep, per query, the
    ``views`` cameras with the lowest ``relevance`` (closest, most central) —
    O(cameras) numpy passes, no per-query Python loop. When ``occlusion`` is
    on, the reprojection ray-march (``pcd.get_intercept``) runs once per kept
    view, and views whose line of sight is blocked by a nearer surface are
    dropped. This is the production replacement for
    ``Annotations.get_first_image_matches``, which looped every camera per query
    and ran the ray-march per candidate (~15 min for 40 k queries).

    Disabled cameras (``enabled`` False) are skipped unless
    ``include_disabled`` (e.g. a camera group chosen for prediction, see
    :mod:`substrata.grouppredict`).

    Returns:
        ``(cam, x, y, depth, rel, cam_list)``: ``(Nq, views)`` arrays sorted by
        relevance (column 0 = best view), with ``cam`` an int index into
        ``cam_list`` (``-1`` = no view / blocked, ``rel`` then ``inf``),
        ``x``/``y`` pixel coordinates and ``depth`` the point depth along the
        camera axis in original-frame units. With ``return_available``, also
        ``available``: ``(Nq,)`` the number of cameras that have the point in
        view (in front, inside the image), before the occlusion test and the
        limit of ``views``.
    """
    from substrata import settings

    if intercept_radius is None:
        intercept_radius = settings.DEFAULT_INTERCEPT_SEARCH_RADIUS
    if discard_threshold is None:
        discard_threshold = settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD
    views = max(1, int(views))

    query_coords = np.asarray(query_coords, dtype=float)
    n = len(query_coords)

    # Camera projection uses each point's *original* (pre-world-transform)
    # coords; occlusion uses the world-frame coords + world pcd. Derive the
    # original-frame coords once.
    if world_transform is not None and not np.allclose(world_transform, np.eye(4)):
        inv = np.linalg.inv(np.asarray(world_transform, dtype=float))
        homog = np.hstack([query_coords, np.ones((n, 1))])
        orig = (homog @ inv.T)[:, :3]
    else:
        orig = query_coords

    from .cameras import can_project

    cam_list = list(cams)
    top_rel = np.full((n, views), np.inf)
    top_cam = np.full((n, views), -1, dtype=int)
    top_x = np.zeros((n, views))
    top_y = np.zeros((n, views))
    top_depth = np.zeros((n, views))
    rows = np.arange(n)
    available = np.zeros(n, dtype=int)

    iterator = (
        tqdm(cam_list, desc="Projecting cameras", unit="cam") if verbose else cam_list
    )
    for ci, cam in enumerate(iterator):
        if (getattr(cam, "enabled", True) is False and not include_disabled) or not can_project(cam):
            continue  # (disabled, or no calibrated sensor / no pose of its own)
        x, y, depth, rel, in_view = cam.project_points(orig, use_orig_coords=True)
        available += in_view & (depth > 0)
        # Replace each query's worst kept view when this camera beats it.
        worst = np.argmax(top_rel, axis=1)
        better = in_view & (depth > 0) & (rel < top_rel[rows, worst])
        if not better.any():
            continue
        q = rows[better]
        j = worst[better]
        top_rel[q, j] = rel[better]
        top_cam[q, j] = ci
        top_x[q, j] = x[better]
        top_y[q, j] = y[better]
        top_depth[q, j] = depth[better]

    if occlusion and pcd is not None:
        pcd.build_kd_tree()
        kept = np.argwhere(top_cam >= 0)
        it = (
            tqdm(kept, desc="Occlusion check", unit="view") if verbose else kept
        )
        n_ray_fail = 0
        first_error = None
        for qi, j in it:
            cam = cam_list[top_cam[qi, j]]
            try:
                vector = cam.pixel_to_ray(float(top_x[qi, j]), float(top_y[qi, j]))
                intercept = pcd.get_intercept(
                    cam.coords, search_radius=intercept_radius,
                    vector=vector, always_return=True,
                )
            except Exception as e:  # noqa: BLE001 - the view is dropped (counted)
                intercept = None
                n_ray_fail += 1
                first_error = first_error or e
            if intercept is None or float(
                np.linalg.norm(query_coords[qi] - intercept.coords)
            ) > discard_threshold:  # no hit, or line of sight blocked
                top_cam[qi, j] = -1
                top_rel[qi, j] = np.inf
        if n_ray_fail:
            logger.warning(
                "Occlusion check: %d of %d view(s) dropped because their ray "
                "could not be cast (first error: %r)", n_ray_fail, len(kept),
                first_error)

    order = np.argsort(top_rel, axis=1, kind="stable")
    take = (lambda a: np.take_along_axis(a, order, axis=1))
    out = (take(top_cam), take(top_x), take(top_y), take(top_depth),
           take(top_rel), cam_list)
    return out + (available,) if return_available else out


def match_points_to_cameras(
    query_coords: np.ndarray,
    cams,
    world_transform: Optional[np.ndarray] = None,
    pcd=None,
    occlusion: bool = True,
    intercept_radius: Optional[float] = None,
    discard_threshold: Optional[float] = None,
    verbose: bool = True,
):
    """Find each query point's best camera (see :func:`_match_points_to_cameras_multi`).

    Returns:
        ``(best_cam, best_x, best_y, best_depth, cam_list)`` where ``best_cam``
        is an ``(Nq,)`` int index into ``cam_list`` (``-1`` = unmatched or
        blocked), ``best_x``/``best_y`` are ``(Nq,)`` pixel coordinates, and
        ``best_depth`` is the ``(Nq,)`` point depth of the chosen camera (in
        original-frame units) — used to report the physical crop footprint.
    """
    cam, x, y, depth, _rel, cam_list = _match_points_to_cameras_multi(
        query_coords, cams, world_transform=world_transform, pcd=pcd,
        occlusion=occlusion, views=1, intercept_radius=intercept_radius,
        discard_threshold=discard_threshold, verbose=verbose,
    )
    return cam[:, 0], x[:, 0], y[:, 0], depth[:, 0], cam_list


_match_points_to_cameras = match_points_to_cameras  # deprecated alias (renamed 2026-11-25)


def _classify_crops(
    query_coords: np.ndarray,
    best_cam: np.ndarray,
    best_x: np.ndarray,
    best_y: np.ndarray,
    cam_list,
    learn,
    crop_size: int = 224,
    batch_size: int = 64,
    flush_batch: int = 16384,
    verbose: bool = True,
):
    """Decode each photo once, cut its crops, and classify them in flushed batches.

    Groups matched queries by source photo so every full-res JPEG is decoded a
    single time (the dominant cost) and slices its patches from the in-memory
    array. Crops are classified and **freed** in bounded batches of
    ``flush_batch`` rather than accumulating every patch for the whole cloud
    before a single forward pass — that unbounded list was ~150 KB × millions of
    queries and blew the 125 GB budget at fine ``cell_size``. Peak crop memory is
    now ~``flush_batch`` patches (+ one decoded photo), independent of cloud size.
    Returns ``(labels, conf)`` aligned with ``query_coords``
    (unmatched/failed → ``""`` / ``nan``).
    """
    from substrata import classification, imagecrops

    n = len(query_coords)
    labels = np.full(n, "", dtype=object)
    conf = np.full(n, np.nan)

    by_file: Dict[str, list] = defaultdict(list)
    for qi in np.nonzero(best_cam >= 0)[0]:
        cam = cam_list[best_cam[qi]]
        by_file[cam.filepath].append((int(qi), best_x[qi], best_y[qi]))
    if not by_file:
        return labels, conf

    example_fp = next(iter(by_file))
    if verbose:
        print(
            f"  {len(by_file)} unique photos; example: {example_fp} "
            f"(exists={os.path.isfile(example_fp) if example_fp else 'n/a'})"
        )

    vocab = getattr(getattr(learn, "dls", None), "vocab", None)
    crops: list = []
    crop_qids: list = []
    n_open_fail = n_oob = n_classified = 0

    def _flush():
        """Classify the buffered crops, write results, and free the buffer."""
        nonlocal crops, crop_qids, n_classified
        if not crops:
            return
        dl = learn.dls.test_dl(crops, bs=batch_size)
        # Silence fastai's per-call progress bar (this runs many times); no-op
        # for any learner without ``no_bar`` (e.g. test doubles).
        cm = learn.no_bar() if hasattr(learn, "no_bar") else nullcontext()
        with cm:
            probs, _ = learn.get_preds(dl=dl)
        for qi, p in zip(crop_qids, probs.tolist()):
            res = classification.result_from_probs(p, vocab)
            labels[qi] = res["label"]
            conf[qi] = res["confidence"]
        n_classified += len(crops)
        crops, crop_qids = [], []

    it = (
        tqdm(by_file.items(), desc="Decoding photos", unit="photo")
        if verbose else by_file.items()
    )
    for fp, group in it:
        try:
            with Image.open(fp) as im:
                arr = np.asarray(im.convert("RGB"))
        except (OSError, ValueError):
            n_open_fail += 1
            continue
        h, w = arr.shape[:2]
        for qi, x, y in group:
            # the training crops' box (imagecrops.classifier_crop_box)
            lft, tp, rgt, bot = imagecrops.classifier_crop_box(
                x, y, crop_size, crop_size, w, h)
            if lft >= rgt or tp >= bot:
                n_oob += 1
                continue
            crops.append(Image.fromarray(arr[tp:bot, lft:rgt]))
            crop_qids.append(qi)
            if len(crops) >= flush_batch:
                _flush()  # bound peak memory: classify + free mid-photo
    _flush()  # remainder

    if verbose:
        if n_open_fail:
            print(
                f"  WARNING: {n_open_fail}/{len(by_file)} photos failed to open "
                "— check camera filepaths / cams.set_filepath_replace(...)"
            )
        print(f"  classified {n_classified} crops ({n_oob} dropped as out-of-bounds)")
    return labels, conf


def _report_crop_footprint(
    best_cam, best_depth, cam_list, crop_size, cell_size, depth_scale=1.0
):
    """Print the physical size a classifier crop covers on the substrate.

    A crop is a fixed ``crop_size``-px cut from the full-res photo, so on the
    substrate it spans ``crop_size * depth / fx`` (``fx`` = focal length in px,
    ``depth`` = camera-to-point distance). ``best_depth`` is in the *original*
    (pre-world-transform) frame, so ``depth_scale`` (world metres per original
    unit) converts it to metres. This is the real resolution floor: sampling
    much finer than the footprint just re-classifies heavily overlapping crops.
    Uses the depths already computed during matching.
    """
    matched = best_cam >= 0
    if not matched.any():
        return
    fxs = [
        c.sensor.fx for c in cam_list
        if getattr(c, "sensor", None) is not None
        and getattr(c.sensor, "fx", None)
    ]
    if not fxs:
        return
    fx = float(np.median(fxs))
    depths = np.abs(best_depth[matched].astype(float)) * depth_scale
    fp = crop_size * depths / fx  # metres
    lo, med, hi = np.percentile(fp, [5, 50, 95])
    print(
        f"[footprint] crop ~= {med * 100:.1f} cm on substrate "
        f"(5-95 pct {lo * 100:.1f}-{hi * 100:.1f} cm; "
        f"crop {crop_size}px, fx {fx:.0f}px, depth {np.median(depths):.2f} m)"
    )
    if cell_size < 0.5 * med:
        print(
            f"  NOTE: cell_size {cell_size * 100:.1f} cm is well below the crop "
            f"footprint — adjacent queries classify overlapping patches "
            f"(oversampling; little extra detail)."
        )


def _classify_point_features(
    best_cam: np.ndarray,
    best_x: np.ndarray,
    best_y: np.ndarray,
    best_depth: np.ndarray,
    cam_list,
    clf,
    depth_scale: float,
    batch_size: int = 64,
    verbose: bool = True,
):
    """Classify matched queries with a point-feature classifier.

    Each query is classified at its pixel in its best photo; its ground
    resolution (``depth x depth_scale / fx``) sets the resampling of
    classifiers with a physical target resolution. Photos are decoded once and
    windows are streamed through the backbone (bounded memory).

    Returns:
        ``(labels, conf)`` aligned with the queries (unmatched/failed ->
        ``""`` / ``nan``).
    """
    from substrata import pointfeatures

    n = len(best_cam)
    labels = np.full(n, "", dtype=object)
    conf = np.full(n, np.nan)
    qi = np.nonzero(best_cam >= 0)[0]
    if not len(qi):
        return labels, conf
    paths = [cam_list[best_cam[i]].filepath for i in qi]
    gsd = None
    if clf.target_mm_px:
        gsd = []
        for i in qi:
            sensor = getattr(cam_list[best_cam[i]], "sensor", None)
            gsd.append(pointfeatures.ground_resolution_mm(
                best_depth[i], getattr(sensor, "fx", None), depth_scale))
        if verbose:
            g = np.array([np.nan if v is None else v for v in gsd])
            print(f"[classify] ground resolution median {np.nanmedian(g):.3f} "
                  f"mm/px -> target {clf.target_mm_px:.3f} mm/px")
    probs, ok = clf.classify_points(
        paths, best_x[qi], best_y[qi], gsd_mm=gsd, batch_size=batch_size,
        progress=verbose,
    )
    k = np.nanargmax(np.where(ok[:, None], probs, -1), axis=1)
    classes = np.array(clf.classes, dtype=object)
    labels[qi[ok]] = classes[k[ok]]
    conf[qi[ok]] = probs[ok, k[ok]]
    return labels, conf


def _classify_dense_views(
    view_cam: np.ndarray,
    view_x: np.ndarray,
    view_y: np.ndarray,
    view_depth: np.ndarray,
    view_rel: np.ndarray,
    cam_list,
    clf,
    depth_scale: float,
    mask_dir: Optional[str] = None,
    mask_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    verbose: bool = True,
):
    """Classify queries from dense per-photo maps, fusing several views.

    Every photo that is a view of some query gets one dense probability map
    (:func:`substrata.pointfeatures.iter_dense_maps`; resampled at the photo's
    median ground resolution over its views when the classifier works in
    mm/px). Each view's probabilities are read at its pixel, and a query's are
    the mean over its views weighted by ``relevance_min / relevance`` (as
    ``predict --method weighted``).

    Args:
        view_cam, view_x, view_y, view_depth, view_rel: ``(Nq, V)`` views from
            :func:`_match_points_to_cameras_multi`.
        cam_list: Cameras indexed by ``view_cam``.
        clf: Point-feature classifier.
        depth_scale: World metres per original-frame unit.
        mask_dir: If given, write each photo's masks there
            (:func:`substrata.pointfeatures.save_dense_masks`) plus an index.
        mask_colors: ``{label: (r, g, b)}`` mask colours.

    Returns:
        Dict of ``(Nq,)`` arrays: ``labels`` (``""`` unclassified), ``conf``
        (fused probability), ``n_views`` (views used), ``agreement`` (weight
        share of the views whose own label is the fused one), ``cam`` / ``x`` /
        ``y`` (the highest-weight view used; ``-1`` / nan if none), ``probs``
        (``(Nq, classes)`` fused probabilities, rows of 0 where unclassified).
    """
    import json

    from substrata import pointfeatures

    n, _v = view_cam.shape
    k = len(clf.classes)
    valid = view_cam >= 0
    rel_min = np.where(valid, view_rel, np.inf).min(axis=1)
    weight = np.where(valid, rel_min[:, None] / np.where(valid, view_rel, 1.0), 0.0)

    photos = np.unique(view_cam[valid])
    entries = {ci: np.argwhere(view_cam == ci) for ci in photos}
    gsds = []
    for ci in photos:
        q, j = entries[ci].T
        g = None
        if clf.target_mm_px:
            fx = getattr(getattr(cam_list[ci], "sensor", None), "fx", None)
            gs = [pointfeatures.ground_resolution_mm(d, fx, depth_scale)
                  for d in view_depth[q, j]]
            gs = [x for x in gs if x is not None]
            g = float(np.median(gs)) if gs else None
        gsds.append(g)
    # without a ground resolution a photo is resampled by its width, as
    # predict and masks resample it (pointfeatures.photo_scale)
    widths = pointfeatures.image_widths(
        [cam_list[ci].filepath for ci, g in zip(photos, gsds) if g is None])
    scales = [pointfeatures.photo_scale(clf, g, widths.get(cam_list[ci].filepath))
              for ci, g in zip(photos, gsds)]
    if verbose:
        print(f"[dense] {len(photos):,} photos for {int(valid.any(1).sum()):,} "
              f"queries ({valid.sum():,} views); median scale "
              f"{np.median(scales) if scales else float('nan'):.3f}")

    fused = np.zeros((n, k), np.float32)
    wsum = np.zeros(n)
    view_lab = np.full(view_cam.shape, -1, dtype=int)
    paths = [cam_list[ci].filepath for ci in photos]
    stems = pointfeatures.unique_stems(paths) if mask_dir else None
    index = {}
    for i, dm in pointfeatures.iter_dense_maps(clf, paths, scales, progress=verbose):
        if dm is None:
            continue
        q, j = entries[photos[i]].T
        p = dm.sample(view_x[q, j], view_y[q, j])
        w = weight[q, j]
        np.add.at(fused, q, p * w[:, None])
        np.add.at(wsum, q, w)
        view_lab[q, j] = p.argmax(1)
        if mask_dir:
            files = pointfeatures.save_dense_masks(dm, mask_dir, stems[i],
                                                   mask_colors or {})
            index[paths[i]] = dict(files, cell_px=dm.cell_px, scale=dm.scale,
                                   width=dm.width, height=dm.height)
    if mask_dir:
        with open(os.path.join(mask_dir, "masks.json"), "w") as f:
            json.dump({"classes": list(clf.classes),
                       "colors": {c: list(v) for c, v in (mask_colors or {}).items()},
                       "photos": index}, f, indent=1)

    ok = wsum > 0
    probs = np.divide(fused, wsum[:, None], out=np.zeros_like(fused),
                      where=ok[:, None])
    lab = probs.argmax(1)
    classes = np.array(clf.classes, dtype=object)
    labels = np.where(ok, classes[lab], "").astype(object)
    conf = np.where(ok, probs[np.arange(n), lab], np.nan)
    used = view_lab >= 0
    agree = np.where(used & (view_lab == lab[:, None]), weight, 0.0).sum(1)
    agreement = np.where(ok, agree / np.where(ok, wsum, 1.0), np.nan)
    # The highest-weight view actually used (for cam_filepath / cam_x / cam_y).
    best = np.argmax(np.where(used, weight, -1.0), axis=1)
    rows = np.arange(n)
    cam = np.where(ok, view_cam[rows, best], -1)
    return {
        "labels": labels, "conf": conf, "n_views": used.sum(1),
        "agreement": agreement, "cam": cam, "probs": probs,
        "x": np.where(ok, view_x[rows, best], np.nan),
        "y": np.where(ok, view_y[rows, best], np.nan),
    }


def segment_point_cloud(
    pcd,
    cams,
    classifier,
    *,
    world_transform: Optional[np.ndarray] = None,
    cell_size: Optional[float] = None,
    sampling: str = "voxel",
    occlusion: bool = True,
    label_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    crop_size: Optional[int] = None,
    batch_size: int = 64,
    flush_batch: int = 16384,
    views: Optional[int] = None,
    dense: bool = True,
    mask_dir: Optional[str] = None,
    scale_factor: Optional[float] = None,
    verbose: bool = True,
) -> Segmentation:
    """Segment a point cloud by classifying camera patches at sampled query points.

    Samples query points across ``pcd`` and matches them to camera photos (fast
    vectorized, occlusion-aware). With a crop classifier, each query's best
    photo is classified from a crop around it. With a point-feature classifier
    (``.pt``), each photo that sees queries gets one dense probability map,
    and every query fuses its ``views`` best photos (weighted by relevance;
    see :func:`_classify_dense_views`), which also gives each point its number
    of views and their agreement. Returns a :class:`Segmentation`. Propagate its
    labels to a cloud with :meth:`Segmentation.propagate`, recolour with
    :meth:`Segmentation.recolor`, or recolour a full-size PLY with
    :func:`recolor_ply_file`.

    Args:
        pcd: decimated :class:`~substrata.pointclouds.PointCloud` (world frame).
        cams: project :class:`~substrata.cameras.Cameras`.
        classifier: a loaded fastai learner or a path to a ``.pkl`` (loaded via
            :func:`substrata.classification.get_image_classifier`).
        world_transform: project world transform; defaults to
            ``pcd.world_transform``.
        cell_size: query spacing / voxel size in metres
            (default ``settings.DEFAULT_SEG_CELL_SIZE``).
        sampling: ``"voxel"`` (default) or ``"xy_grid"`` (see
            :func:`sample_query_points`).
        occlusion: run reprojection occlusion filtering (drops points seen
            through a nearer surface).
        label_colors: optional ``{label: (r, g, b)}`` overrides (0-255).
        crop_size: patch size in px (default ``settings.TRAIN_CROP_SIZE``).
        batch_size: inference batch size.
        flush_batch: max crops held in memory before a forward pass frees them
            (bounds peak RAM regardless of ``cell_size`` / cloud size).
        views: point classifiers: photos fused per query (default
            ``settings.SEG_POINT_VIEWS``).
        dense: point classifiers: dense per-photo maps (default); False
            classifies each query's best view from its own window instead.
        mask_dir: point classifiers (dense): also save each photo's masks here.
        scale_factor: the project's ``scale_factor`` (metres per model unit),
            used for ground resolutions when ``world_transform`` is the
            identity (see :func:`substrata.pointfeatures.metres_per_unit`).

    Returns:
        A :class:`Segmentation` (classified query points + codebook + colours).
    """
    from substrata import classification, settings

    if cell_size is None:
        cell_size = settings.DEFAULT_SEG_CELL_SIZE
    if crop_size is None:
        crop_size = settings.TRAIN_CROP_SIZE
    if world_transform is None:
        world_transform = getattr(pcd, "world_transform", None)

    learn = (
        classification.get_image_classifier(classifier)
        if isinstance(classifier, str)
        else classifier
    )

    qc = sample_query_points(pcd, cell_size, method=sampling)
    if verbose:
        print(
            f"[sample] {len(qc):,} query points "
            f"({sampling}, {cell_size * 100:.0f} cm)"
        )

    res = classify_query_points(
        qc, cams, learn, world_transform=world_transform, pcd=pcd,
        occlusion=occlusion, label_colors=label_colors, crop_size=crop_size,
        batch_size=batch_size, flush_batch=flush_batch, views=views,
        dense=dense, mask_dir=mask_dir, cell_size=cell_size,
        scale_factor=scale_factor, verbose=verbose,
    )
    px = np.column_stack([res["x"], res["y"]]).astype(float)
    if res["dense"]:
        return Segmentation.from_query_labels(
            qc, res["labels"], res["conf"], label_colors=label_colors,
            cams=res["cam_paths"], px=px, n_views=res["n_views"],
            agreement=res["agreement"],
        )
    return Segmentation.from_query_labels(
        qc, res["labels"], res["conf"], label_colors=label_colors,
        cams=res["cam_paths"], px=px,
    )


def classify_query_points(
    qc: np.ndarray,
    cams,
    learn,
    *,
    world_transform: Optional[np.ndarray] = None,
    pcd=None,
    occlusion: bool = True,
    label_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    crop_size: Optional[int] = None,
    batch_size: int = 64,
    flush_batch: int = 16384,
    views: Optional[int] = None,
    dense: bool = True,
    mask_dir: Optional[str] = None,
    cell_size: Optional[float] = None,
    include_disabled: bool = False,
    scale_factor: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Match world-frame query points ``qc`` to camera views and classify them, as
    :func:`segment_point_cloud` does (its matching and classification; also used
    by ``substrata predict --view-mode segment``, :mod:`substrata.grouppredict`).

    A point-feature classifier with ``dense``: each query's ``views`` best views
    (default ``settings.SEG_POINT_VIEWS``; relevance order, the reprojection
    occlusion test on ``pcd`` when ``occlusion``), fused from dense per-photo
    maps (:func:`_classify_dense_views`). Otherwise the best view only: a
    window at its pixel (point classifier) or a ``crop_size`` crop.

    Args:
        qc: ``(Nq, 3)`` query points in the world frame of ``world_transform``.
        cams: Cameras (a :class:`~substrata.cameras.Cameras` or a list).
        learn: Loaded classifier.
        include_disabled: Also use cameras with ``enabled`` False.
        cell_size: Only for the crop footprint report of a crop classifier.
        scale_factor: The project's ``scale_factor``: metres per model unit
            when ``world_transform`` is the identity (ground resolutions).
        Others: as for :func:`segment_point_cloud`.

    Returns:
        Dict of ``(Nq,)`` arrays: ``labels`` (``""`` unclassified), ``conf``,
        ``n_views`` (views used), ``agreement`` (dense: weight share of the views
        whose own label is the fused one; else None), ``cam_paths`` (photo of
        the highest-weight view used, ``""`` none), ``x`` / ``y`` (its pixel,
        nan none), ``available`` (cameras with the point in view before the
        occlusion test and the limit of views); ``probs`` (dense: ``(Nq,
        classes)``, else None), ``classes`` and ``dense`` (bool).
    """
    from substrata import pointfeatures, settings

    if crop_size is None:
        crop_size = settings.TRAIN_CROP_SIZE
    qc = np.asarray(qc, dtype=float).reshape(-1, 3)
    point_clf = pointfeatures.is_point_classifier(learn)
    # Depths are in the original frame; convert to metres with the project's
    # metres per model unit (the world_transform's linear scale, else its
    # scale_factor; pointfeatures.metres_per_unit).
    depth_scale = pointfeatures.metres_per_unit(world_transform, scale_factor) or 1.0
    use_pcd = pcd if occlusion else None
    classes = getattr(learn, "classes", None)
    if classes is None:
        classes = getattr(getattr(learn, "dls", None), "vocab", None)
    classes = [str(c) for c in classes] if classes is not None else []

    if point_clf and dense:
        n_views = int(views or settings.SEG_POINT_VIEWS)
        vc, vx, vy, vd, vr, cam_list, available = _match_points_to_cameras_multi(
            qc, cams, world_transform=world_transform, pcd=use_pcd,
            occlusion=occlusion, views=n_views, verbose=verbose,
            include_disabled=include_disabled, return_available=True,
        )
        if verbose:
            print(f"[match] {int((vc[:, 0] >= 0).sum()):,}/{len(qc):,} queries "
                  f"matched, {(vc >= 0).sum(1).mean():.2f} of {n_views} views "
                  "per query on average")
        mask_colors = None
        if mask_dir:
            mask_colors = build_label_colors(list(learn.classes), label_colors)
        res = _classify_dense_views(
            vc, vx, vy, vd, vr, cam_list, learn, depth_scale,
            mask_dir=mask_dir, mask_colors=mask_colors, verbose=verbose,
        )
        labels = res["labels"]
        if verbose:
            done = labels != ""
            print(f"[classify] {int(done.sum()):,} classified; views agree "
                  f"fully for {np.mean(res['agreement'][done] >= 0.999):.0%}")
        cams_of = np.array(
            [cam_list[c].filepath if c >= 0 else "" for c in res["cam"]],
            dtype=object,
        )
        return dict(labels=labels, conf=res["conf"], n_views=res["n_views"],
                    agreement=res["agreement"], cam_paths=cams_of, x=res["x"],
                    y=res["y"], available=available, probs=res["probs"],
                    classes=list(learn.classes), dense=True)

    best_cam, best_x, best_y, best_depth, _rel, cam_list, available = _match_points_to_cameras_multi(
        qc, cams, world_transform=world_transform, pcd=use_pcd,
        occlusion=occlusion, views=1, verbose=verbose,
        include_disabled=include_disabled, return_available=True,
    )
    best_cam, best_x, best_y, best_depth = best_cam[:, 0], best_x[:, 0], best_y[:, 0], best_depth[:, 0]
    if verbose:
        print(f"[match] {int((best_cam >= 0).sum()):,}/{len(qc):,} queries matched")
        if not point_clf:
            _report_crop_footprint(
                best_cam, best_depth, cam_list, crop_size,
                cell_size if cell_size is not None else settings.DEFAULT_SEG_CELL_SIZE,
                depth_scale,
            )

    if point_clf:
        labels, conf = _classify_point_features(
            best_cam, best_x, best_y, best_depth, cam_list, learn, depth_scale,
            batch_size=batch_size, verbose=verbose,
        )
    else:
        labels, conf = _classify_crops(
            qc, best_cam, best_x, best_y, cam_list, learn,
            crop_size=crop_size, batch_size=batch_size, flush_batch=flush_batch,
            verbose=verbose,
        )
    if verbose:
        print(f"[classify] {int((labels != '').sum()):,} classified")

    cams_of = np.array(
        [cam_list[c].filepath if c >= 0 else "" for c in best_cam], dtype=object
    )
    x = np.where(best_cam >= 0, best_x, np.nan).astype(float)
    y = np.where(best_cam >= 0, best_y, np.nan).astype(float)
    return dict(labels=labels, conf=conf, n_views=(labels != "").astype(int),
                agreement=None, cam_paths=cams_of, x=x, y=y, available=available,
                probs=None, classes=classes, dense=False)


def recolor_ply_file(
    input_ply: str,
    output_ply: str,
    segmentation: Segmentation,
    *,
    world_transform: Optional[np.ndarray] = None,
    max_radius: Optional[float] = None,
    value_floor: float = 0.3,
    unlabeled: str = "keep",
    chunk_bytes: int = 64 << 20,
    verbose: bool = True,
) -> str:
    """Stream a full-size PLY and rewrite its colours from a segmentation.

    Reads ``input_ply`` in chunks (never loading it fully), propagates the
    segmentation's labels to each chunk's points, blends the category colour
    with the point's original luminance, and writes the recoloured cloud to
    ``output_ply`` — type-preserving (integer colours over their type's full
    range, float 0-1). The header is read with
    :func:`substrata.stratified.read_ply_header`; the output is written
    atomically.

    Args:
        input_ply: path to the full-size PLY (raw, original frame).
        output_ply: destination path.
        segmentation: a :class:`Segmentation` (query points are in world frame).
        world_transform: transform mapping the PLY's raw coords into the world
            frame the segmentation lives in (default: the pcd world transform you
            used when segmenting). If ``None``/identity, coords are used as-is.
        max_radius: propagation cap (``None`` = label every point).
        value_floor / unlabeled: passed to :meth:`Segmentation.recolor`
            (``unlabeled="keep"`` retains original colour for unlabeled points).
        chunk_bytes: streaming chunk size.
    """
    from substrata.fileio import atomic_write
    from substrata.stratified import ply_rgb, read_ply_header

    wt = None
    if world_transform is not None and not np.allclose(world_transform, np.eye(4)):
        wt = np.asarray(world_transform, dtype=float)

    with open(input_ply, "rb") as fin:
        header = read_ply_header(fin)
        fmt, n_vertices, vprops = header.fmt, header.n_vertices, header.props
        rec_size, dtype = header.record_size, header.dtype
        if not header.has("red", "green", "blue"):
            raise ValueError("Input PLY has no red/green/blue properties to recolour.")

        recs_per_chunk = max(1, chunk_bytes // rec_size)

        with atomic_write(output_ply, "wb") as fout:
            fout.write(_seg_output_header(fmt, vprops, n_vertices))
            remaining = n_vertices
            bar = (
                tqdm(total=n_vertices, desc="Recolouring PLY", unit="pt")
                if verbose else None
            )
            while remaining > 0:
                k = min(recs_per_chunk, remaining)
                buf = fin.read(k * rec_size)
                if len(buf) < k * rec_size:
                    raise ValueError("Unexpected EOF while reading PLY vertex data.")
                recs = np.frombuffer(buf, dtype=dtype, count=k).copy()

                xyz = np.column_stack(
                    [recs["x"], recs["y"], recs["z"]]
                ).astype(float)
                if wt is not None:
                    xyz = (np.hstack([xyz, np.ones((k, 1))]) @ wt.T)[:, :3]

                codes = segmentation.propagate(xyz, max_radius=max_radius)
                orig_rgb = ply_rgb(recs).astype(float)  # 0-255 whatever the type
                new_rgb = segmentation.recolor(
                    orig_rgb, codes, value_floor=value_floor, unlabeled=unlabeled,
                )  # 0-1

                col_dtype = recs["red"].dtype
                if col_dtype.kind in "ui":  # integers: full range of the type
                    top = float(np.iinfo(col_dtype).max)
                    vals = np.round(new_rgb * top).astype(col_dtype)
                else:  # float/double stored 0-1
                    vals = new_rgb.astype(col_dtype)
                recs["red"] = vals[:, 0]
                recs["green"] = vals[:, 1]
                recs["blue"] = vals[:, 2]

                fout.write(recs.tobytes())
                remaining -= k
                if bar is not None:
                    bar.update(k)
            if bar is not None:
                bar.close()
    return output_ply


def _seg_output_header(fmt: str, vprops, vertex_count: int) -> bytes:
    """Clean minimal PLY header for the recoloured output (all props preserved)."""
    lines = [b"ply\n", f"format {fmt} 1.0\n".encode("ascii")]
    lines.append(b"comment recoloured by substrata segmentation\n")
    lines.append(f"element vertex {vertex_count}\n".encode("ascii"))
    for t, name in vprops:
        lines.append(f"property {t} {name}\n".encode("ascii"))
    lines.append(b"end_header\n")
    return b"".join(lines)
