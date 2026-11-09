"""Point-feature classifier: frozen DINOv2 features read at the annotated pixel.

Instead of classifying a crop centred on a point (the fastai crop classifier),
a context window around the point is passed through a frozen self-supervised
backbone (DINOv2) and the backbone's *local* feature is interpolated at the
exact point pixel. A linear (logistic-regression) probe on these features is
the classifier. The label then describes a feature cell of ~14 model pixels
rather than a whole crop, which fits point annotations far better, and the
probe trains in seconds.

Scale. Each window is resampled from the photo by a per-point ``scale`` (model
px per native px). A classifier either has a physical target resolution
(``target_mm_px``; each photo is resampled by its ground resolution / target,
so a feature cell covers the same seabed area across cameras and distances)
or a fixed ``scale_factor`` (photos without camera geometry). The ground
resolution of a point is ``camera depth x world scale / focal length``.

Calibration. Features are standardised with statistics of random patches of
the training photos (``source_*``). ``substrata calibrate`` adds statistics of
another camera system's photos of the same area (``target_*``), so a model
trained on one camera can be applied to another.

The model file (``.pt``) is a plain tensor/str dict, loaded with
``torch.load(weights_only=True)``; the backbone weights come from torch.hub.
"""

from __future__ import annotations

# Standard Library
import copy
import csv
import glob
import json
import os
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Third-Party Libraries
import numpy as np
from PIL import Image
from tqdm import tqdm

# Local Modules
from substrata import settings
from substrata.logging import logger

Image.MAX_IMAGE_PIXELS = None

__all__ = [
    "PointFeatureClassifier", "ScalePlan", "is_point_arch", "is_point_classifier",
    "is_point_classifier_path", "load_backbone", "forward_feature_grid",
    "sample_feature_grid", "window_box", "read_window", "image_windows",
    "extract_point_features", "world_scale", "ground_resolution_mm",
    "image_match_gsd_mm", "estimate_gsd_from_projects", "fit_point_probe",
    "read_training_points", "plan_training_scales", "resolve_training_gsd",
    "auto_scale_factor", "cached_features", "random_patches",
    "feature_statistics", "evaluate_predictions", "train_point_classifier",
    "evaluate_point_classifier", "report_point_evaluation",
    "calibrate_point_classifier", "list_images",
]

POINT_MODEL_FORMAT = "substrata-point-classifier"
POINT_MODEL_VERSION = 1
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESAMPLE = Image.LANCZOS
_BACKBONES: Dict[Tuple[str, str, str], Any] = {}


def is_point_arch(arch: Optional[str]) -> bool:
    """Whether a ``train --arch`` value selects the point-feature classifier."""
    return bool(arch) and arch.lower().startswith(settings.POINT_ARCH_PREFIX)


def is_point_classifier(obj: Any) -> bool:
    """Whether ``obj`` is a :class:`PointFeatureClassifier` (vs a fastai learner)."""
    return bool(getattr(obj, "is_point_classifier", False))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


def default_device(device: Optional[str] = None) -> str:
    """``device`` or the best available torch device (cuda, mps, cpu)."""
    if device:
        return device
    import torch  # lazy

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone(
    name: str, device: Optional[str] = None, hub_repo: Optional[str] = None,
) -> Any:
    """Load (once per process) a frozen DINOv2 backbone from torch.hub.

    Args:
        name: torch.hub entry point, e.g. ``dinov2_vitb14_reg``.
        device: Torch device (default: best available).
        hub_repo: Hub repo or local clone; default the ``SUBSTRATA_POINT_HUB_REPO``
            environment variable, else ``settings.POINT_HUB_REPO``. A local
            directory works offline.

    Returns:
        The backbone in eval mode on ``device``.
    """
    import torch  # lazy

    repo = hub_repo or os.environ.get("SUBSTRATA_POINT_HUB_REPO") \
        or settings.POINT_HUB_REPO
    dev = default_device(device)
    key = (repo, name, dev)
    if key not in _BACKBONES:
        source = "local" if os.path.isdir(repo) else "github"
        logger.info("Loading backbone %s from %s (%s)", name, repo, source)
        model = torch.hub.load(repo, name, source=source)
        _BACKBONES[key] = model.eval().to(dev)
    return _BACKBONES[key]


def _model_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        return "cpu"


def forward_feature_grid(model: Any, windows: np.ndarray) -> Any:
    """Backbone feature grid of a batch of windows.

    Args:
        model: DINOv2-style backbone (``forward_features`` returning
            ``x_norm_patchtokens``; ``patch_size`` attribute).
        windows: ``(B, H, W, 3)`` uint8 RGB windows.

    Returns:
        ``(B, C, H / patch, W / patch)`` float32 torch tensor.
    """
    import torch  # lazy

    dev = _model_device(model)
    x = torch.from_numpy(np.ascontiguousarray(windows)).to(dev)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor(_IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad(), torch.autocast(
        device_type="cuda" if dev.startswith("cuda") else "cpu",
        dtype=torch.float16, enabled=dev.startswith("cuda"),
    ):
        out = model.forward_features(x)
    tokens = out["x_norm_patchtokens"].float()
    p = int(model.patch_size)
    b, _n, c = tokens.shape
    h, w = x.shape[2] // p, x.shape[3] // p
    return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)


def sample_feature_grid(feat: Any, xy: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly interpolate a feature grid at pixel positions.

    Feature cell ``j`` is centred on window pixel ``(j + 0.5) * size / w``
    (``align_corners=False``), so a point is read at its exact position.

    Args:
        feat: ``(B, C, h, w)`` feature grid of ``size`` x ``size`` windows.
        xy: ``(B, 2)`` point pixel positions within each window.
        size: Window size in pixels.

    Returns:
        ``(B, C)`` float32 array.
    """
    import torch  # lazy
    import torch.nn.functional as F  # lazy

    b, c = feat.shape[:2]
    pos = torch.as_tensor(np.asarray(xy, dtype=np.float32), device=feat.device)
    grid = (2 * pos / size - 1).view(b, 1, 1, 2)
    out = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.view(b, c).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def window_box(
    cx: float, cy: float, size: float, width: int, height: int,
) -> Tuple[float, float, float, float]:
    """A ``size`` square box centred on ``(cx, cy)``, shifted to lie inside the
    image where it fits (so the point is off-centre near an edge, but every
    window pixel is real image)."""
    x0 = min(max(cx - size / 2, 0.0), max(width - size, 0.0))
    y0 = min(max(cy - size / 2, 0.0), max(height - size, 0.0))
    return x0, y0, x0 + size, y0 + size


def read_window(im: Image.Image, box: Sequence[float], out_px: int) -> Image.Image:
    """Resample ``box`` of ``im`` to ``out_px`` square; outside areas are black."""
    w, h = im.size
    if box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h:
        return im.resize((out_px, out_px), _RESAMPLE, box=tuple(box))
    crop = im.crop(tuple(int(round(v)) for v in box))
    return crop.resize((out_px, out_px), _RESAMPLE)


def image_windows(
    path: str, points: np.ndarray, scales: np.ndarray, window: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Decode a photo once and cut one context window per point.

    JPEGs are decoded at a reduced size when every point's scale allows it
    (much faster for large photos).

    Args:
        path: Photo path.
        points: ``(n, 2)`` native pixel positions.
        scales: ``(n,)`` model px per native px.
        window: Output window size (model px).

    Returns:
        ``(windows (n, window, window, 3) uint8, xy (n, 2))`` with each point's
        position in its window, or None if the photo cannot be read.
    """
    try:
        with Image.open(path) as im:
            width, height = im.size
            s_min = float(np.min(scales))
            if s_min < 1:
                im.draft("RGB", (max(1, int(width * s_min)),
                                 max(1, int(height * s_min))))
            im = im.convert("RGB")
            f = im.size[0] / width
            n = len(points)
            wins = np.empty((n, window, window, 3), np.uint8)
            xy = np.empty((n, 2), np.float32)
            for i, ((cx, cy), s) in enumerate(zip(points, scales)):
                size = window / s * f
                box = window_box(cx * f, cy * f, size, *im.size)
                wins[i] = np.asarray(read_window(im, box, window))
                k = window / size
                xy[i] = ((cx * f - box[0]) * k, (cy * f - box[1]) * k)
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    return wins, xy


def extract_point_features(
    paths: Sequence[str], xs: Sequence[float], ys: Sequence[float],
    scales: Sequence[float], model: Any,
    window: int = settings.POINT_WINDOW_PX,
    batch_size: int = settings.POINT_BATCH_SIZE,
    n_jobs: int = -1, prefetch: int = settings.POINT_IMAGE_PREFETCH,
    desc: str = "Point features", progress: bool = True,
    on_batch: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Backbone features at many points, decoding each photo once.

    Photos are decoded in parallel threads, at most ``prefetch`` ahead of the
    GPU, so memory stays bounded however many points there are.

    Args:
        paths, xs, ys: Photo path and native pixel position per point.
        scales: Model px per native px, per point.
        model: Backbone (see :func:`forward_feature_grid`).
        window: Window size (model px).
        batch_size: Windows per forward pass.
        n_jobs: Decoding threads (-1 = all cores).
        prefetch: Photos decoded ahead of the GPU.
        desc: Progress-bar label.
        progress: Show a progress bar.
        on_batch: Optional ``callback(indices, features)`` per forward batch;
            when given, features are not accumulated (returned array is None),
            e.g. to classify millions of points without holding the features.

    Returns:
        ``(features (n, C) float32 or None, ok (n,) bool)``; rows of points on
        unreadable photos stay zero with ``ok`` False.
    """
    n = len(paths)
    ok = np.zeros(n, bool)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    scales = np.asarray(scales, float)
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, p in enumerate(paths):
        groups.setdefault(p, []).append(i)
    feats = None if on_batch else np.zeros((n, int(model.embed_dim)), np.float32)
    jobs = (os.cpu_count() or 1) if n_jobs in (None, -1) else max(1, int(n_jobs))
    buf_idx: List[np.ndarray] = []
    buf_win: List[np.ndarray] = []
    buf_xy: List[np.ndarray] = []

    def flush():
        if not buf_idx:
            return
        idx = np.concatenate(buf_idx)
        wins = np.concatenate(buf_win)
        xy = np.concatenate(buf_xy)
        for s in range(0, len(idx), batch_size):
            sl = slice(s, s + batch_size)
            pf = sample_feature_grid(
                forward_feature_grid(model, wins[sl]), xy[sl], window
            )
            if on_batch is not None:
                on_batch(idx[sl], pf)
            else:
                feats[idx[sl]] = pf
        ok[idx] = True
        buf_idx.clear()
        buf_win.clear()
        buf_xy.clear()

    def load(item):
        path, idx = item
        idx = np.asarray(idx)
        pts = np.column_stack([xs[idx], ys[idx]])
        return idx, image_windows(path, pts, scales[idx], window)

    items = list(groups.items())
    bar = tqdm(total=len(items), desc=desc, unit="photo", disable=not progress)
    n_buf = 0
    with ThreadPoolExecutor(jobs) as ex:
        for start in range(0, len(items), prefetch):
            for idx, res in ex.map(load, items[start:start + prefetch]):
                bar.update(1)
                if res is None:
                    continue
                buf_idx.append(idx)
                buf_win.append(res[0])
                buf_xy.append(res[1])
                n_buf += len(idx)
                if n_buf >= batch_size:
                    flush()
                    n_buf = 0
    flush()
    bar.close()
    return feats, ok


# ---------------------------------------------------------------------------
# Ground resolution (mm per native pixel)
# ---------------------------------------------------------------------------


def world_scale(world_transform: Any) -> Optional[float]:
    """Linear scale of a 4x4 world transform (world units per original unit)."""
    if world_transform is None:
        return None
    lin = np.asarray(world_transform, dtype=float)[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(lin))))
    return s if s > 0 else None


def ground_resolution_mm(depth: float, fx: float, scale: float) -> Optional[float]:
    """Ground resolution (mm/px) from a camera depth in original model units.

    Args:
        depth: Point depth along the camera axis (original model units).
        fx: Focal length in pixels.
        scale: World (metre) units per original unit (:func:`world_scale`).
    """
    try:
        g = float(depth) * float(scale) / float(fx) * 1000.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return g if np.isfinite(g) and g > 0 else None


def image_match_gsd_mm(match: Any) -> Optional[float]:
    """Ground resolution (mm/px) at an :class:`~substrata.cameras.ImageMatch`.

    Uses the match depth, or projects its annotation's original coordinates
    when the match has none (matches on the annotated-on photo). Returns None
    when the camera geometry is unavailable (e.g. plain photos).
    """
    cam = getattr(match, "cam", None)
    sensor = getattr(cam, "sensor", None)
    fx = getattr(sensor, "fx", None)
    scale = world_scale(getattr(getattr(cam, "parent", None), "world_transform", None))
    if not fx or scale is None:
        return None
    depth = getattr(match, "depth", None)
    if depth is None:
        ann = getattr(match, "annotation", None)
        coords = getattr(ann, "orig_coords", None)
        if coords is None or not hasattr(cam, "project_points"):
            return None
        try:
            _x, _y, d, *_ = cam.project_points(
                np.asarray(coords, float).reshape(1, 3), use_orig_coords=True
            )
            depth = float(np.atleast_1d(d)[0])
        except Exception:  # noqa: BLE001 - geometry is optional here
            return None
    return ground_resolution_mm(depth, fx, scale)


def _find_project_yaml(photo_dir: str, levels: int = 3) -> Optional[str]:
    """The project YAML in or above a photo folder (``<dir>/<dirname>.yaml``
    preferred; annotation sidecars ``*.csv.yaml`` are ignored)."""
    d = os.path.abspath(photo_dir)
    for _ in range(levels + 1):
        named = os.path.join(d, os.path.basename(d) + ".yaml")
        if os.path.isfile(named):
            return named
        cands = [
            p for p in sorted(glob.glob(os.path.join(d, "*.yaml")))
            if not p.endswith(".csv" + settings.ANNOTATION_SIDECAR_SUFFIX)
        ]
        if len(cands) == 1:
            return cands[0]
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def estimate_gsd_from_projects(
    rows: Sequence[Dict[str, str]], tol_px: float = settings.POINT_GSD_PIXEL_TOL,
) -> np.ndarray:
    """Ground resolution (mm/px) of annotation rows from their projects' cameras.

    For each photo folder the project YAML is looked up in or above it, its
    cameras are loaded, and each row's original coordinates are projected into
    its photo. A row only gets a value when the projection lands within
    ``tol_px`` of its ``cam_x``/``cam_y`` (guards against coordinates from
    another model, e.g. points reprojected from a different plot).

    Args:
        rows: Training annotation rows (``orig_x/y/z``, ``cam_filepath``,
            ``cam_x``, ``cam_y``).
        tol_px: Projection tolerance in pixels.

    Returns:
        ``(n,)`` float array, NaN where unknown.
    """
    out = np.full(len(rows), np.nan)
    by_dir: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        fp = (r.get("cam_filepath") or "").strip()
        if fp:
            by_dir[os.path.dirname(fp)].append(i)
    projects: Dict[str, Any] = {}
    for photo_dir, idx in by_dir.items():
        yaml_path = _find_project_yaml(photo_dir)
        if yaml_path is None:
            continue
        if yaml_path not in projects:
            projects[yaml_path] = _load_project_cameras(yaml_path)
        loaded = projects[yaml_path]
        if loaded is None:
            continue
        cams, scale = loaded
        by_photo: Dict[str, List[int]] = defaultdict(list)
        for i in idx:
            by_photo[os.path.basename(rows[i]["cam_filepath"])].append(i)
        for name, pidx in by_photo.items():
            cam = cams.get(name) or cams.get(os.path.splitext(name)[0])
            if cam is None or getattr(cam, "sensor", None) is None:
                continue
            try:
                xyz = np.array([[float(rows[i]["orig_x"]), float(rows[i]["orig_y"]),
                                 float(rows[i]["orig_z"])] for i in pidx])
                px, py, depth, *_ = cam.project_points(xyz, use_orig_coords=True)
            except (KeyError, ValueError, TypeError):
                continue
            px, py, depth = (np.atleast_1d(np.asarray(v, float))
                             for v in (px, py, depth))
            for j, i in enumerate(pidx):
                try:
                    dx = px[j] - float(rows[i]["cam_x"])
                    dy = py[j] - float(rows[i]["cam_y"])
                except (KeyError, ValueError, TypeError):
                    continue
                if np.hypot(dx, dy) <= tol_px:
                    g = ground_resolution_mm(depth[j], cam.sensor.fx, scale)
                    if g is not None:
                        out[i] = g
    return out


def _load_project_cameras(yaml_path: str) -> Optional[Tuple[Dict[str, Any], float]]:
    """``({filename: Camera}, world scale)`` of a project, or None."""
    try:
        from substrata import cameras
        from substrata.initializer import ProjectInitializer

        init = ProjectInitializer(yaml=yaml_path)
        meta = getattr(init, "cams_meta_json_filepath", None)
        xml = getattr(init, "cams_xml_filepath", None)
        if not (meta and xml and os.path.isfile(meta) and os.path.isfile(xml)):
            return None
        scale = world_scale(getattr(init, "world_transform", None))
        if scale is None:
            scale = float(getattr(init, "scale_factor", None) or 0) or None
        if scale is None:
            return None
        print(f"Ground resolution: loading cameras of {os.path.basename(yaml_path)}")
        cams = cameras.Cameras(meta, xml)
        return {c.filename: c for c in cams.data.values()}, scale
    except Exception as e:  # noqa: BLE001 - any failure means "unknown"
        logger.warning("Could not load cameras of %s: %s", yaml_path, e)
        return None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _as_np(v, dtype=np.float32):
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=dtype)


@dataclass
class PointFeatureClassifier:
    """A linear probe on frozen backbone features read at points.

    ``logits = ((features - mean) / std) @ weights.T + bias`` with the
    calibration statistics (``target_*`` if set, else ``source_*``).

    Attributes:
        backbone: torch.hub entry point of the backbone.
        classes: Class names (probe output order).
        weights: ``(K, C)`` probe weights.
        bias: ``(K,)`` probe bias.
        source_mean, source_std: Feature statistics of the training photos.
        window: Context window size (model px).
        target_mm_px: Physical target resolution (mm per model px), or None
            for a fixed ``scale_factor``.
        scale_factor: Model px per native px when no ground resolution is
            known (and always, without ``target_mm_px``).
        target_mean, target_std: Statistics of another camera system's photos
            (``substrata calibrate``), or None.
        label_map: ``raw_label -> class`` used in training.
        meta: Free-form training/calibration metadata.
    """

    backbone: str
    classes: List[str]
    weights: np.ndarray
    bias: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    window: int = settings.POINT_WINDOW_PX
    target_mm_px: Optional[float] = None
    scale_factor: float = 1.0
    target_mean: Optional[np.ndarray] = None
    target_std: Optional[np.ndarray] = None
    label_map: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    hub_repo: Optional[str] = None
    device: Optional[str] = None

    is_point_classifier = True

    # -- fastai-learner-like surface used by existing callers --------------
    @property
    def vocab(self) -> List[str]:
        return list(self.classes)

    @property
    def calibrated(self) -> bool:
        return self.target_mean is not None

    # -- scale ---------------------------------------------------------------
    def scales_for(
        self, gsd_mm: Optional[Sequence[Optional[float]]], n: int,
    ) -> np.ndarray:
        """Per-point scale (model px per native px) from ground resolutions.

        Points without a ground resolution (or classifiers without a target
        resolution) use ``scale_factor``.
        """
        out = np.full(n, float(self.scale_factor))
        if self.target_mm_px and gsd_mm is not None:
            g = np.array([np.nan if v is None else v for v in gsd_mm], float)
            good = np.isfinite(g) & (g > 0)
            out[good] = g[good] / float(self.target_mm_px)
        return out

    # -- inference -----------------------------------------------------------
    def standardise(self, feats: np.ndarray) -> np.ndarray:
        mean = self.target_mean if self.calibrated else self.source_mean
        std = self.target_std if self.calibrated else self.source_std
        return (np.asarray(feats, np.float32) - mean) / std

    def predict_proba_features(self, feats: np.ndarray) -> np.ndarray:
        """Class probabilities ``(n, K)`` for raw backbone features."""
        z = self.standardise(feats) @ self.weights.T + self.bias
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def backbone_model(self) -> Any:
        return load_backbone(self.backbone, self.device, self.hub_repo)

    def classify_points(
        self, paths: Sequence[str], xs: Sequence[float], ys: Sequence[float],
        gsd_mm: Optional[Sequence[Optional[float]]] = None,
        batch_size: int = settings.POINT_BATCH_SIZE, n_jobs: int = -1,
        progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify points on photos.

        Args:
            paths, xs, ys: Photo path and native pixel position per point.
            gsd_mm: Optional ground resolution (mm/px) per point.
            batch_size: Windows per forward pass.
            n_jobs: Decoding threads (-1 = all cores).
            progress: Show a progress bar.

        Returns:
            ``(probs (n, K) float32, ok (n,) bool)``; failed points have
            NaN probabilities.
        """
        n = len(paths)
        if self.target_mm_px and gsd_mm is None:
            logger.warning(
                "No ground resolution for the points; using the fixed scale "
                "%.3f instead of %.3f mm/px.", self.scale_factor, self.target_mm_px
            )
        scales = self.scales_for(gsd_mm, n)
        probs = np.full((n, len(self.classes)), np.nan, np.float32)

        def on_batch(idx, pf):
            probs[idx] = self.predict_proba_features(pf)

        _f, ok = extract_point_features(
            paths, xs, ys, scales, self.backbone_model(), self.window,
            batch_size=batch_size, n_jobs=n_jobs, desc="Classifying points",
            progress=progress, on_batch=on_batch,
        )
        return probs, ok

    def results(
        self, probs: np.ndarray, ok: np.ndarray,
    ) -> List[Optional[Dict[str, Any]]]:
        """Standard result dicts (``label``, ``confidence``, ``probs``,
        ``pred_idx``), None for failed points."""
        out: List[Optional[Dict[str, Any]]] = []
        for p, good in zip(probs, ok):
            if not good:
                out.append(None)
                continue
            k = int(np.argmax(p))
            out.append({
                "label": self.classes[k], "confidence": float(p[k]),
                "probs": {c: float(v) for c, v in zip(self.classes, p)},
                "pred_idx": k,
            })
        return out

    def with_calibration(
        self, mean: np.ndarray, std: np.ndarray, info: Dict[str, Any],
    ) -> "PointFeatureClassifier":
        """A copy standardising with another camera system's statistics."""
        clf = copy.deepcopy(self)
        clf.target_mean = np.asarray(mean, np.float32)
        clf.target_std = np.asarray(std, np.float32)
        clf.meta = dict(clf.meta, calibration=info)
        return clf

    # -- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        import torch  # lazy

        def t(v):
            return None if v is None else torch.from_numpy(np.asarray(v, np.float32))

        state = {
            "format": POINT_MODEL_FORMAT, "version": POINT_MODEL_VERSION,
            "backbone": self.backbone, "classes": list(self.classes),
            "weights": t(self.weights), "bias": t(self.bias),
            "source_mean": t(self.source_mean), "source_std": t(self.source_std),
            "target_mean": t(self.target_mean), "target_std": t(self.target_std),
            "window": int(self.window),
            "target_mm_px": None if self.target_mm_px is None
            else float(self.target_mm_px),
            "scale_factor": float(self.scale_factor),
            "label_map": json.dumps(self.label_map),
            "meta": json.dumps(self.meta, default=str),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state, path)
        logger.info("Saved point classifier to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PointFeatureClassifier":
        import torch  # lazy

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Classifier file not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("format") != POINT_MODEL_FORMAT:
            raise ValueError(f"{path} is not a substrata point classifier.")
        return cls(
            backbone=state["backbone"], classes=list(state["classes"]),
            weights=_as_np(state["weights"]), bias=_as_np(state["bias"]),
            source_mean=_as_np(state["source_mean"]),
            source_std=_as_np(state["source_std"]),
            window=int(state["window"]), target_mm_px=state.get("target_mm_px"),
            scale_factor=float(state["scale_factor"]),
            target_mean=_as_np(state.get("target_mean")),
            target_std=_as_np(state.get("target_std")),
            label_map=json.loads(state.get("label_map") or "{}"),
            meta=json.loads(state.get("meta") or "{}"),
            device=device,
        )


def is_point_classifier_path(path: str) -> bool:
    """Whether a classifier path is a point classifier file (``.pt``/``.pth``)."""
    return str(path).lower().endswith((".pt", ".pth"))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def fit_point_probe(
    z: np.ndarray, y: Sequence[str], cs: Sequence[float] = settings.POINT_PROBE_CS,
    cv: int = settings.POINT_PROBE_CV, seed: int = 0,
) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Fit a class-balanced logistic-regression probe on standardised features.

    The regularisation ``C`` is chosen by stratified cross-validation on
    macro-F1. A per-feature standardisation of ``z`` is folded into the
    returned weights, so the probe is a single linear layer.

    Returns:
        ``(classes, weights (K, C), bias (K,), chosen C)``.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    m = z.mean(0)
    s = z.std(0) + 1e-6
    zs = (z - m) / s
    folds = min(cv, int(min(Counter(y.tolist()).values())))
    if folds >= 2 and len(cs) > 1:
        lr = LogisticRegressionCV(
            Cs=list(cs), cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
            class_weight="balanced", scoring="f1_macro", max_iter=5000,
        ).fit(zs, y)
        c = float(np.atleast_1d(lr.C_)[0])
    else:
        c = float(sorted(cs)[len(cs) // 2])
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(zs, y)
    classes = [str(k) for k in lr.classes_]
    w = np.asarray(lr.coef_, np.float64)
    b = np.asarray(lr.intercept_, np.float64)
    if len(classes) == 2 and w.shape[0] == 1:  # binary: softmax over [0, w]
        w = np.vstack([np.zeros_like(w), w])
        b = np.concatenate([[0.0], b])
    w_fold = w / s
    b_fold = b - (w * (m / s)).sum(1)
    return classes, w_fold.astype(np.float32), b_fold.astype(np.float32), c


# ---------------------------------------------------------------------------
# Training (substrata train --arch dinov2_*)
# ---------------------------------------------------------------------------


SPLIT_NAMES = ("train", "validation", "test")


def read_training_points(
    ann_path: str, label_map: Optional[Dict[str, str]],
    split: Tuple[int, int, int] = settings.TRAIN_SPLIT,
) -> List[Dict[str, Any]]:
    """Training annotation rows with a class (``_cls``) and split (``_split``).

    Rows lacking camera fields are skipped. With a ``label_map``, rows whose
    label it excludes are skipped too; with ``None`` every labelled row is kept
    (``_cls`` None where unmapped) - features are extracted for all visible
    labels, like crops, so editing the map never needs re-extraction. The split
    is drawn exactly as for crops (:func:`substrata.classification.split_for_id`).
    """
    from substrata import classification  # lazy (circular)

    rows = []
    with open(ann_path, newline="") as f:
        for r in csv.DictReader(f):
            label = (r.get("label") or "").strip()
            cls = label_map.get(label) if label_map is not None else None
            if not label or (label_map is not None and not cls):
                continue
            try:
                r["_x"] = float(r["cam_x"])
                r["_y"] = float(r["cam_y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (r.get("cam_filepath") or "").strip():
                continue
            key = (r.get("split_key") or "").strip() or (r.get("id") or "").strip()
            r["_cls"] = cls
            r["_split"] = classification.split_for_id(key, split)
            rows.append(r)
    return rows


def row_key(r: Dict[str, Any]) -> str:
    """Stable cache key of a point: id, photo and rounded pixel position."""
    return (f"{(r.get('id') or '').strip()}|{os.path.basename(r['cam_filepath'])}|"
            f"{r['_x']:.1f}|{r['_y']:.1f}")


def _median_image_width(paths: Sequence[str], max_n: int = 200) -> Optional[float]:
    widths = []
    for p in list(dict.fromkeys(paths))[:max_n]:
        try:
            with Image.open(p) as im:
                widths.append(im.size[0])
        except OSError:
            continue
    return float(np.median(widths)) if widths else None


def auto_scale_factor(
    paths: Sequence[str], window: int = settings.POINT_WINDOW_PX,
) -> float:
    """Scale at which the window spans ``POINT_CONTEXT_FRACTION`` of the median
    image width (at most 1: never upsample)."""
    w = _median_image_width(paths)
    if not w:
        return 1.0
    return float(min(1.0, window / (w * settings.POINT_CONTEXT_FRACTION)))


def resolve_training_gsd(
    rows: Sequence[Dict[str, Any]], cache_path: Optional[str] = None,
) -> np.ndarray:
    """Ground resolution (mm/px) per training row, NaN where unknown.

    Uses a ``cam_gsd_mm`` column when present, else
    :func:`estimate_gsd_from_projects`; results are cached in ``cache_path``.
    """
    cache: Dict[str, Optional[float]] = {}
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    out = np.full(len(rows), np.nan)
    todo = []
    for i, r in enumerate(rows):
        raw = r.get("cam_gsd_mm")
        if isinstance(raw, str):
            raw = raw.strip()
        try:
            v = float(raw) if raw not in (None, "") else None
        except ValueError:
            v = None
        if v is not None and v > 0:
            out[i] = v
            continue
        k = row_key(r)
        if k in cache:
            out[i] = np.nan if cache[k] is None else cache[k]
        else:
            todo.append(i)
    if todo:
        est = estimate_gsd_from_projects([rows[i] for i in todo])
        for i, v in zip(todo, est):
            out[i] = v
            cache[row_key(rows[i])] = None if not np.isfinite(v) else float(v)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
    return out


def _fill_gsd(rows: Sequence[Dict[str, Any]], gsd: np.ndarray) -> np.ndarray:
    """Fill unknown ground resolutions with their photo's, else the overall median."""
    g = gsd.copy()
    by_photo: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_photo[r["cam_filepath"]].append(i)
    for idx in by_photo.values():
        vals = g[idx]
        if np.isfinite(vals).any() and not np.isfinite(vals).all():
            med = np.nanmedian(vals)
            g[[i for i in idx if not np.isfinite(g[i])]] = med
    if np.isfinite(g).any():
        g[~np.isfinite(g)] = np.nanmedian(g)
    return g


@dataclass
class ScalePlan:
    """How training photos are resampled: per-row scales and the model fields."""

    scales: np.ndarray
    target_mm_px: Optional[float]
    scale_factor: float
    description: str


def plan_training_scales(
    rows: Sequence[Dict[str, Any]], output_dir: str,
    feature_mm_px: Optional[float] = None, feature_scale: Optional[float] = None,
    window: int = settings.POINT_WINDOW_PX,
) -> ScalePlan:
    """Choose the training resolution.

    ``feature_scale`` -> fixed scale. Otherwise ground resolutions are looked
    up (:func:`resolve_training_gsd`); with enough of them
    (``POINT_GSD_MIN_FRACTION``) a physical target is used (``feature_mm_px``,
    or the median ground resolution at the automatic scale), else the
    automatic fixed scale (:func:`auto_scale_factor`).
    """
    paths = [r["cam_filepath"] for r in rows]
    auto = auto_scale_factor(paths, window)
    if feature_scale:
        s = float(feature_scale)
        return ScalePlan(np.full(len(rows), s), None, s,
                         f"fixed scale {s:.3f} (--feature-scale)")
    gsd = resolve_training_gsd(
        rows, os.path.join(output_dir, settings.POINT_FEATURES_DIR, "gsd_cache.json")
    )
    frac = float(np.isfinite(gsd).mean()) if len(gsd) else 0.0
    if frac < settings.POINT_GSD_MIN_FRACTION:
        if feature_mm_px:
            raise ValueError(
                f"--feature-mm-px needs a ground resolution for at least "
                f"{settings.POINT_GSD_MIN_FRACTION:.0%} of the points, but only "
                f"{frac:.0%} could be derived from project cameras (or a "
                "cam_gsd_mm column)."
            )
        return ScalePlan(np.full(len(rows), auto), None, auto,
                         f"fixed scale {auto:.3f} (automatic; ground resolution "
                         f"known for only {frac:.0%} of points)")
    g = _fill_gsd(rows, gsd)
    med = float(np.median(g))
    # The automatic target is rounded so it stays put as the data grow a little
    # (a changed target changes every scale and re-extracts every feature).
    target = float(feature_mm_px) if feature_mm_px else float(f"{med / auto:.3g}")
    return ScalePlan(
        g / target, target, med / target,
        f"{target:.3f} mm/px (median ground resolution {med:.3f} mm/px, known "
        f"for {frac:.0%} of points{'' if feature_mm_px else '; automatic'})",
    )


def _cache_file(output_dir: str, backbone: str, window: int, kind: str) -> str:
    return os.path.join(output_dir, settings.POINT_FEATURES_DIR,
                        f"{kind}_{backbone}_w{window}.npz")


def cached_features(
    keys: Sequence[str], paths: Sequence[str], xs: Sequence[float],
    ys: Sequence[float], scales: Sequence[float], cache_path: str,
    model_loader: Callable[[], Any], window: int,
    batch_size: int = settings.POINT_BATCH_SIZE, n_jobs: int = -1,
    desc: str = "Point features",
) -> Tuple[np.ndarray, np.ndarray]:
    """Features for keyed points, extracting only those not in the cache.

    Keys include the scale, so a changed resolution re-extracts. The cache is
    rewritten with exactly the current keys (stale entries dropped); points on
    unreadable photos are not cached and are retried next time.

    Returns:
        ``(features (n, C) float32, ok (n,) bool)``.
    """
    full_keys = [f"{k}|{s:.5f}" for k, s in zip(keys, scales)]
    have: Dict[str, int] = {}
    old = None
    if os.path.isfile(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            old = z["feats"]
            have = {k: i for i, k in enumerate(z["keys"].tolist())}
        except (OSError, ValueError, KeyError):
            have, old = {}, None
    missing = [i for i, k in enumerate(full_keys) if k not in have]
    n = len(full_keys)
    dim = old.shape[1] if old is not None and len(old) else None
    feats = None
    ok = np.zeros(n, bool)
    if missing:
        print(f"{desc}: {n - len(missing)} cached, extracting {len(missing)}")
        model = model_loader()
        dim = int(model.embed_dim)
        new, new_ok = extract_point_features(
            [paths[i] for i in missing], [xs[i] for i in missing],
            [ys[i] for i in missing], [scales[i] for i in missing], model, window,
            batch_size=batch_size, n_jobs=n_jobs, desc=desc,
        )
    else:
        print(f"{desc}: all {n} cached")
    feats = np.zeros((n, dim or 1), np.float32)
    for i, k in enumerate(full_keys):
        if k in have:
            feats[i] = old[have[k]]
            ok[i] = True
    if missing:
        feats[missing] = new
        ok[missing] = new_ok
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    keep = np.nonzero(ok)[0]
    np.savez(cache_path, keys=np.array([full_keys[i] for i in keep]),
             feats=feats[keep].astype(np.float16))
    return feats, ok


def random_patches(
    paths: Sequence[str], n_images: int = settings.POINT_CALIB_IMAGES,
    per_image: int = settings.POINT_CALIB_PER_IMAGE,
    margin: float = settings.POINT_CALIB_MARGIN, seed: int = 0,
) -> List[Tuple[str, float, float]]:
    """Random positions in a random sample of photos (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(paths))
    pick = [uniq[i] for i in sorted(rng.choice(len(uniq), min(n_images, len(uniq)),
                                               replace=False))]
    out = []
    for p in pick:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except OSError:
            continue
        for _ in range(per_image):
            out.append((p, float(rng.uniform(w * margin, w * (1 - margin))),
                        float(rng.uniform(h * margin, h * (1 - margin)))))
    return out


def feature_statistics(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and spread used for standardisation."""
    return feats.mean(0).astype(np.float32), (feats.std(0) + 1e-6).astype(np.float32)


def _calibration_features(
    rows: Sequence[Dict[str, Any]], scales: np.ndarray, output_dir: str,
    backbone: str, model_loader, window: int, batch_size: int, n_jobs: int,
) -> np.ndarray:
    """Features of random patches of the training photos (each photo at the
    median scale of its points)."""
    photo_scale: Dict[str, List[float]] = defaultdict(list)
    for r, s in zip(rows, scales):
        photo_scale[r["cam_filepath"]].append(s)
    patches = random_patches(list(photo_scale))
    keys = [f"{os.path.basename(p)}|{x:.1f}|{y:.1f}" for p, x, y in patches]
    sc = [float(np.median(photo_scale[p])) for p, _x, _y in patches]
    feats, ok = cached_features(
        keys, [p for p, _x, _y in patches], [x for _p, x, _y in patches],
        [y for _p, _x, y in patches], sc,
        _cache_file(output_dir, backbone, window, "calibration"), model_loader,
        window, batch_size, n_jobs, desc="Calibration patches",
    )
    return feats[ok]


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str],
) -> Dict[str, Any]:
    """Accuracy, balanced accuracy, macro-F1, report and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score,
    )

    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def train_point_classifier(
    ann_path: str, label_map: Dict[str, str], output_dir: str, model_path: str,
    backbone: str = settings.TRAIN_DEFAULT_ARCH,
    feature_mm_px: Optional[float] = None, feature_scale: Optional[float] = None,
    window: int = settings.POINT_WINDOW_PX,
    batch_size: int = settings.POINT_BATCH_SIZE, n_jobs: int = -1,
    hub_repo: Optional[str] = None,
) -> Tuple[PointFeatureClassifier, Dict[str, Any]]:
    """Train and save a point-feature classifier from a training annotations CSV.

    Steps: read rows (label map, split) -> choose the resolution
    (:func:`plan_training_scales`) -> point features (cached, incremental) ->
    calibration statistics from random patches of the training photos ->
    probe on the train split -> evaluate validation and test.

    Returns:
        ``(classifier, info)`` with ``info`` holding ``rows``, ``plan``,
        ``counts`` (split -> Counter(class)), ``evaluation`` (split -> metrics
        dict of :func:`evaluate_predictions`) and ``n_failed``.
    """
    # Features (and the resolution) cover every collated point, whatever the
    # label map selects, so editing the map never re-extracts.
    rows = read_training_points(ann_path, None)
    for r in rows:
        r["_cls"] = label_map.get(r["label"].strip())
    if not any(r["_cls"] for r in rows):
        raise ValueError(f"No usable training points in {ann_path}.")
    plan = plan_training_scales(rows, output_dir, feature_mm_px, feature_scale, window)
    print(f"Resolution: {plan.description}")

    def loader():
        return load_backbone(backbone, hub_repo=hub_repo)

    feats, ok = cached_features(
        [row_key(r) for r in rows], [r["cam_filepath"] for r in rows],
        [r["_x"] for r in rows], [r["_y"] for r in rows], plan.scales,
        _cache_file(output_dir, backbone, window, "points"), loader, window,
        batch_size, n_jobs, desc="Point features",
    )
    calib = _calibration_features(rows, plan.scales, output_dir, backbone, loader,
                                  window, batch_size, n_jobs)
    if len(calib) < 2:
        calib = feats[ok]
    mean, std = feature_statistics(calib)
    z = (feats - mean) / std
    y = np.array([r["_cls"] or "" for r in rows])
    split = np.array([r["_split"] for r in rows])
    ok = ok & (y != "")  # unmapped labels: features only
    train = ok & (split == 0)
    if len(set(y[train])) < 2:
        raise ValueError("Training needs at least two classes in the train split.")
    print(f"Fitting the probe on {int(train.sum())} training points ...")
    classes, w, b, c = fit_point_probe(z[train], y[train])
    clf = PointFeatureClassifier(
        backbone=backbone, classes=classes, weights=w, bias=b,
        source_mean=mean, source_std=std, window=window,
        target_mm_px=plan.target_mm_px, scale_factor=plan.scale_factor,
        label_map=dict(label_map), hub_repo=hub_repo,
        meta={
            "resolution": plan.description, "probe_C": c,
            "n_train": int(train.sum()), "n_calibration_patches": int(len(calib)),
            "annotations": os.path.abspath(ann_path),
        },
    )
    clf.save(model_path)
    probs = clf.predict_proba_features(feats)
    pred = np.array(classes)[probs.argmax(1)]
    evaluation = {}
    for s in (1, 2):
        sel = ok & (split == s)
        if sel.any():
            evaluation[SPLIT_NAMES[s]] = evaluate_predictions(
                y[sel].tolist(), pred[sel].tolist())
    counts = {name: Counter(y[ok & (split == s)].tolist())
              for s, name in enumerate(SPLIT_NAMES)}
    n_failed = int(sum(1 for r, good in zip(rows, ok) if r["_cls"] and not good))
    return clf, {"rows": rows, "plan": plan, "counts": counts,
                 "evaluation": evaluation, "n_failed": n_failed}


def evaluate_point_classifier(
    clf: PointFeatureClassifier, ann_path: str, output_dir: str, split_name: str,
    batch_size: int = settings.POINT_BATCH_SIZE, n_jobs: int = -1,
) -> Dict[str, Any]:
    """Re-evaluate a saved point classifier on one split of a training CSV
    (``train --validate`` / ``--test``), at the classifier's own resolution."""
    # Same points, resolution and feature cache as training (all cached unless
    # the CSV changed since).
    rows = read_training_points(ann_path, None)
    s = SPLIT_NAMES.index(split_name)
    sel = np.array([r["_split"] == s and bool(clf.label_map.get(r["label"].strip()))
                    for r in rows])
    if not sel.any():
        raise ValueError(f"No {split_name} points in {ann_path}.")
    gsd = None
    if clf.target_mm_px:
        gsd = _fill_gsd(rows, resolve_training_gsd(rows, os.path.join(
            output_dir, settings.POINT_FEATURES_DIR, "gsd_cache.json")))
    scales = clf.scales_for(gsd, len(rows))
    feats, ok = cached_features(
        [row_key(r) for r in rows], [r["cam_filepath"] for r in rows],
        [r["_x"] for r in rows], [r["_y"] for r in rows], scales,
        _cache_file(output_dir, clf.backbone, clf.window, "points"),
        clf.backbone_model, clf.window, batch_size, n_jobs,
        desc="Point features",
    )
    use = sel & ok
    probs = clf.predict_proba_features(feats[use])
    pred = np.array(clf.classes)[probs.argmax(1)]
    y = [clf.label_map[r["label"].strip()] for r, u in zip(rows, use) if u]
    return evaluate_predictions(y, pred.tolist())


def report_point_evaluation(
    result: Dict[str, Any], split_name: str, pdf_path: Optional[str] = None,
) -> None:
    """Print an evaluation and optionally write the classifier stats PDF."""
    from substrata import classification  # lazy (circular)

    print(f"\n=== Stats on {split_name} ({result['n']} points) ===")
    print(f"Overall accuracy: {result['accuracy']:.4f}  "
          f"balanced: {result['balanced_accuracy']:.4f}  "
          f"macro-F1: {result['macro_f1']:.4f}\n")
    print(result["report"])
    if pdf_path:
        classification._write_stats_pdf(
            pdf_path, split_name, result["n"], result["accuracy"],
            f"Balanced accuracy: {result['balanced_accuracy']:.4f}\n"
            f"Macro-F1: {result['macro_f1']:.4f}\n\n{result['report']}",
            result["confusion"], result["labels"], examples=None,
        )
        print(f"Wrote stats PDF: {pdf_path}")


# ---------------------------------------------------------------------------
# Calibration (substrata calibrate)
# ---------------------------------------------------------------------------


def list_images(inputs: Sequence[str]) -> List[str]:
    """Photo paths from files, folders (recursive) and glob patterns."""
    exts = settings.CORALNET_IMAGE_EXTS
    out: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                out += [os.path.join(root, f) for f in files
                        if f.lower().endswith(exts)]
        elif os.path.isfile(item):
            out.append(item)
        else:
            out += [p for p in glob.glob(item, recursive=True)
                    if p.lower().endswith(exts)]
    return sorted(set(out))


def calibrate_point_classifier(
    clf: PointFeatureClassifier, images: Sequence[str],
    gsd_mm: Optional[float] = None, n_images: int = settings.POINT_CALIB_IMAGES,
    per_image: int = settings.POINT_CALIB_PER_IMAGE,
    batch_size: int = settings.POINT_BATCH_SIZE, n_jobs: int = -1, seed: int = 0,
) -> PointFeatureClassifier:
    """Calibrate a classifier to another camera system.

    Computes feature statistics of random patches of ``images`` - photos of the
    *same area* as the training photos (e.g. AUV frames over the diver plots;
    statistics from other habitats would shift the classes) - and returns a
    copy that standardises with them.

    Args:
        clf: Trained point classifier.
        images: Photo paths of the target camera system.
        gsd_mm: Ground resolution (mm/px) of these photos; required for
            classifiers with a physical target resolution.
        n_images, per_image: Sampling of random patches.
    """
    if clf.target_mm_px and not gsd_mm:
        raise ValueError(
            f"This classifier works at {clf.target_mm_px:.3f} mm/px; pass the "
            "ground resolution of the calibration photos (--gsd-mm)."
        )
    patches = random_patches(images, n_images, per_image, seed=seed)
    if not patches:
        raise ValueError("No readable calibration photos.")
    scale = gsd_mm / clf.target_mm_px if clf.target_mm_px else clf.scale_factor
    feats, ok = extract_point_features(
        [p for p, _x, _y in patches], [x for _p, x, _y in patches],
        [y for _p, _x, y in patches], np.full(len(patches), scale),
        clf.backbone_model(), clf.window, batch_size=batch_size, n_jobs=n_jobs,
        desc="Calibration patches",
    )
    mean, std = feature_statistics(feats[ok])
    info = {"n_patches": int(ok.sum()),
            "n_images": len({p for p, _x, _y in patches}),
            "gsd_mm": gsd_mm, "scale": float(scale)}
    return clf.with_calibration(mean, std, info)
