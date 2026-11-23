"""Placing video frames that Metashape did not align: which survey photos see them, and a lens calibration from
another plot.

Made for a diver's GoPro frames (sensor 2: 3840 x 2160, wide angle, 1 fps) where Metashape aligned few or none
(sa3_60m: 0 of 701), so ``camsync --match-images`` has no posed frame to pose a close-up from. Posing such a frame
from the plot's aligned survey photos (Sony; matched, their pixels followed to the cloud, PnP: as a close-up is
posed from frames, :mod:`substrata.imagesync`) needs two things this module gives:

1. **A lens** (:func:`sensor_from_cams_xml`, :func:`rotated_180`, :func:`calibration_difference`): the GoPro's
   calibration from a plot where Metashape aligned its frames. The same GoPro and settings give the same lens:
   sa3_40m, sa4_40m and sa4_60m agree within 2.4-3.7 px (median over the image; 0.07-0.11 deg) once sa4's is turned
   round: on the sa4 dives the GoPro was mounted upside down, which mirrors the principal point offset (cx, cy) and
   the tangential terms (p1, p2). Without the turn they differ by ~25 px (0.7 deg). The distortion itself moves
   pixels by ~130 px (median), so a borrowed calibration is far better than none.
2. **Which survey photos to match a frame against** (:class:`GlobalIndex`, :func:`verify_inliers`;
   :func:`candidates_seeing`). With no pose at all, by image similarity: DINOv2 global descriptors (class token) of
   the survey photos and the frame, the 20 most alike, each verified by LoFTR (:func:`verify_inliers`, the frame
   turned 4 ways). Similarity alone is a weak cue across the cameras (GoPro: blue, vignetted, its own light; Sony:
   strobes): on sa3_40m a survey photo that sees the frame (its centre and most of it) was the most alike for 21% of
   frames, among the 20 most alike for 57%; with LoFTR, the best-verified candidate (>= 50 inliers) overlapped the
   frame for 24 of 25 frames that had one (60% of frames), ~14 s a frame. That only has to work once per run of
   frames: once a frame is placed, its 1-fps neighbours' survey photos are those that see the cloud points it sees
   (:func:`candidates_seeing`, no image read): on sa3_40m the next frame's were among those 20 for 100% of frames
   1 s on, 94% 5 s on, 84% 10 s on, and walking the 788 frames that way (searching again after 10 s without a
   placed frame) reached 594 of the 608 frames that see the survey photos' reef, with 6 searches.
   The AUV's photos (sensor 5) almost never see what a GoPro frame sees there: index the diver's survey photos.

Nothing here poses a frame: that is :mod:`substrata.imagesync`'s matching and PnP with the frame as the target and
the survey photos as the sources.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DINOV2_HUB = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
DINOV2_MODEL = "dinov2_vitb14_reg"
DESCRIPTOR_SHORT_SIDE = 224  # px, images resized to (no better at 448, nor ViT-L, GeM, grey, cropped or turned)
FRAME_CROP_X = 0.0  # the frame's left and right share cut off (cutting the GoPro's dark corners did not help)
CANDIDATES = 20  # survey photos a frame is matched against
VERIFY_FRAME_WIDTH = 960  # px: frame and survey photo at about the same scale (GoPro ~35 px/cm, Sony ~64 px/cm full)
VERIFY_SURVEY_WIDTH = 1280
VERIFY_MIN_INLIERS = 50  # LoFTR matches consistent with one fundamental matrix: a frame and a survey photo it sees
VERIFY_THRESHOLD_PX = 2.0


# --------------------------------------------------------------------------- 1. a lens from another plot


_CAL_TERMS = ("f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")


def calibration_params(cams_xml: str, sensor_id: Optional[int] = None, label: Optional[str] = None
                       ) -> Tuple[Dict[str, float], Dict[str, int], str]:
    """The adjusted frame calibration of a sensor in a Metashape cams.xml (by ``sensor_id``, else the only
    sensor with ``label``): (terms ``f``, ``cx``, ... ``p4``, missing ones 0; resolution ``width``/``height``;
    the sensor's label). Raises ValueError when the sensor or its calibration is not there."""
    root = ET.parse(cams_xml).getroot()
    found = []
    for s in root.iter("sensor"):
        if (sensor_id is not None and s.get("id") == str(sensor_id)) or (
                sensor_id is None and label is not None and s.get("label") == label):
            found.append(s)
    if len(found) != 1:
        raise ValueError(f"{cams_xml}: {len(found)} sensors match id {sensor_id} / label {label!r}")
    s = found[0]
    cal = s.find('.//calibration[@type="frame"][@class="adjusted"]')
    if cal is None:
        raise ValueError(f"{cams_xml}: sensor {s.get('id')} has no adjusted calibration (none of its photos aligned)")
    res = cal.find("resolution")
    if res is None:
        res = s.find("resolution")
    params = {k: 0.0 for k in _CAL_TERMS}
    for c in cal:
        if c.tag in params and c.text:
            params[c.tag] = float(c.text)
    return params, {"width": int(res.get("width")), "height": int(res.get("height"))}, s.get("label", "")


def rotated_params(params: Dict[str, float]) -> Dict[str, float]:
    """The calibration terms of the same lens with the camera turned 180 deg about its axis (images upside
    down): the principal point offset and the tangential terms change sign; f, B1/B2 and the radial terms stay."""
    out = dict(params)
    for k in ("cx", "cy", "p1", "p2"):
        out[k] = -float(params.get(k, 0.0))
    return out


def sensor_from_cams_xml(cams_xml: str, sensor_id: Optional[int] = None, *, label: Optional[str] = None,
                         as_sensor_id: Optional[int] = None, rotate180: bool = False):
    """A :class:`substrata.cameras.Sensor` with the calibration of a sensor in another plot's cams.xml (e.g. the
    GoPro's from a plot where its frames aligned), optionally turned 180 deg (:func:`rotated_params`: the GoPro
    mounted the other way up). ``as_sensor_id``: the id it gets (default the source's). ``calibration_class``
    ``"borrowed"``."""
    from .cameras import Sensor

    params, res, lab = calibration_params(cams_xml, sensor_id, label)
    if rotate180:
        params = rotated_params(params)
    sid = as_sensor_id if as_sensor_id is not None else int(sensor_id) if sensor_id is not None else -1
    s = Sensor(sid, lab, res, params)
    s.calibration_class = "borrowed"
    s.borrowed_from = (os.path.abspath(cams_xml), sensor_id, bool(rotate180))
    return s


def rotated_180(sensor):
    """A copy of ``sensor`` turned 180 deg (:func:`rotated_params`)."""
    from .cameras import Sensor

    p = {k: float(getattr(sensor, "cx_metashape" if k == "cx" else "cy_metashape" if k == "cy" else k, 0.0))
         for k in _CAL_TERMS}
    s = Sensor(sensor.sensor_id, sensor.label, {"width": sensor.width, "height": sensor.height}, rotated_params(p))
    s.calibration_class = getattr(sensor, "calibration_class", "adjusted")
    return s


def calibration_difference(a, b, *, nx: int = 41, ny: int = 23, inner: float = 0.7) -> Dict[str, float]:
    """How differently two sensors (same resolution) image the world: over a grid of ``a``'s pixels, each
    pixel's ray through ``a`` imaged through ``b`` (px apart) and the angle between the two sensors' rays at the
    same pixel (deg). ``inner``: the central share of the image for ``px_max_inner`` (the corners of a wide lens
    are dark and may not be imaged at all). Returns ``px_median``, ``px_max_inner``, ``deg_median``,
    ``deg_max``, ``n_unimaged`` (grid pixels either lens model cannot image)."""
    from .cameras import _lens, _undistort

    if (a.width, a.height) != (b.width, b.height):
        raise ValueError("the sensors' resolutions differ")
    u, v = np.meshgrid(np.linspace(0, a.width, nx), np.linspace(0, a.height, ny))
    u, v = u.ravel(), v.ravel()
    xa, ya = _undistort(u, v, a)
    xd, yd = _lens(xa, ya, b)
    px = np.hypot(b.fx * xd + b.b2 * yd + b.cx - u, b.fy * yd + b.cy - v)
    xb, yb = _undistort(u, v, b)
    ra = np.stack([xa, ya, np.ones_like(xa)], 1)
    rb = np.stack([xb, yb, np.ones_like(xb)], 1)
    ra /= np.linalg.norm(ra, axis=1)[:, None]
    rb /= np.linalg.norm(rb, axis=1)[:, None]
    deg = np.degrees(np.arccos(np.clip((ra * rb).sum(1), -1.0, 1.0)))
    ok = np.isfinite(px) & np.isfinite(deg)
    centre = (np.abs(u - a.width / 2) <= inner * a.width / 2) & (np.abs(v - a.height / 2) <= inner * a.height / 2)
    return dict(px_median=float(np.median(px[ok])), px_max_inner=float(np.max(px[ok & centre])),
                deg_median=float(np.median(deg[ok])), deg_max=float(np.max(deg[ok])), n_unimaged=int((~ok).sum()))


def give_sensor(cams: Iterable[Any], sensor, sensor_id: Optional[int] = None) -> int:
    """Give ``sensor`` to the cameras of ``sensor_id`` (default ``sensor.sensor_id``) among ``cams`` (a
    :class:`substrata.cameras.Cameras` also gets it in its ``sensors``). They still see nothing until they
    have a pose. Returns how many cameras got it."""
    from .cameras import _projection_changed

    sid = sensor.sensor_id if sensor_id is None else sensor_id
    if hasattr(cams, "sensors") and isinstance(getattr(cams, "sensors"), dict):
        cams.sensors[sid] = sensor
    data = cams.data.values() if hasattr(cams, "data") else cams
    n = 0
    for c in data:
        if getattr(c, "sensor_id", None) == sid:
            c.sensor = sensor
            n += 1
    _projection_changed()
    return n


# --------------------------------------------------------------------------- 2. which survey photos


def top_k(query: np.ndarray, index: np.ndarray, k: int = CANDIDATES, mean: Optional[np.ndarray] = None
          ) -> Tuple[np.ndarray, np.ndarray]:
    """The ``k`` index rows most like each query (cosine, after taking off ``mean``, default the index's mean:
    the descriptors all share a large common part). ``query``: (n, d), or (n, r, d) for r versions of each query
    (e.g. turned), of which the best counts. Returns (indices (n, k), similarities (n, k)), best first."""
    q = np.asarray(query, float)
    x = np.asarray(index, float)
    mu = x.mean(0) if mean is None else np.asarray(mean, float)
    x = x - mu
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    q = q - mu
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    sim = q @ x.T if q.ndim == 2 else np.einsum("nrd,md->nrm", q, x).max(1)
    k = min(int(k), x.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return idx, np.take_along_axis(sim, idx, 1)


class GlobalIndex:
    """DINOv2 global descriptors (the class token) of a set of photos, to find those most like a query photo
    (:func:`top_k`). The weights come from the local torch hub cache (``DINOV2_HUB``; nothing is downloaded).

    ~35 ms a photo for a 9568 x 6376 survey photo (most of it reading the JPEG at 1/8); 2,865 survey photos ~100 s."""

    def __init__(self, model: str = DINOV2_MODEL, short_side: int = DESCRIPTOR_SHORT_SIDE,
                 device: Optional[str] = None, hub_dir: str = DINOV2_HUB):
        import torch

        if not os.path.isdir(hub_dir):
            raise FileNotFoundError(f"No DINOv2 hub repository at {hub_dir} (torch.hub.load('facebookresearch/dinov2', ...) once)")
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load(hub_dir, model, source="local").to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.short_side = int(short_side)
        self.keys: List[Any] = []
        self.descriptors: Optional[np.ndarray] = None

    def _tensor(self, path: str, crop_x: float = 0.0):
        from PIL import Image

        im = Image.open(path)
        im.draft("RGB", (self.short_side * 3, self.short_side * 3))
        im = im.convert("RGB")
        if crop_x:
            w, h = im.size
            im = im.crop((int(crop_x * w), 0, int((1 - crop_x) * w), h))
        w, h = im.size
        s = self.short_side / min(w, h)
        im = im.resize((max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)), Image.BILINEAR)
        t = self.torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        mean = self.torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = self.torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t - mean) / std

    def describe(self, paths: Sequence[str], *, crop_x: float = 0.0, turns: Sequence[int] = (0,),
                 num_workers: int = 8) -> np.ndarray:
        """Descriptors of the photos at ``paths``: (n, len(turns), d) for the photos turned by ``turns`` quarter
        turns (the survey photos lie every way round relative to a frame); (n, d) for one turn."""
        torch = self.torch
        outer = self

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(paths)

            def __getitem__(self, i):
                return outer._tensor(paths[i], crop_x)

        out = []
        dl = torch.utils.data.DataLoader(_DS(), batch_size=1, num_workers=num_workers)
        dtype = next(self.model.parameters()).dtype
        for x in dl:
            x = x.to(self.device, dtype)
            rows = []
            for k in turns:
                with torch.inference_mode():
                    o = self.model.forward_features(torch.rot90(x, int(k), (2, 3)))
                rows.append(o["x_norm_clstoken"].float().cpu().numpy()[0])
            out.append(rows)
        d = np.asarray(out, np.float32)
        return d[:, 0] if len(turns) == 1 else d

    def build(self, cams: Sequence[Any], **kw) -> "GlobalIndex":
        """Index the photos of ``cams`` (cameras with a ``filepath``)."""
        self.keys = list(cams)
        self.descriptors = self.describe([c.filepath for c in self.keys], **kw)
        return self

    def query(self, paths: Sequence[str], k: int = CANDIDATES, *, crop_x: float = FRAME_CROP_X,
              turns: Sequence[int] = (0,)) -> List[List[Tuple[Any, float]]]:
        """Per query photo, the ``k`` indexed cameras most like it, best first: [(camera, similarity)]."""
        if self.descriptors is None:
            raise RuntimeError("build() the index first")
        qd = self.describe(paths, crop_x=crop_x, turns=turns)
        idx, sim = top_k(qd, self.descriptors, k)
        return [[(self.keys[j], float(s)) for j, s in zip(ri, rs)] for ri, rs in zip(idx, sim)]

    def save(self, path: str) -> None:
        np.savez_compressed(path, descriptors=self.descriptors,
                            keys=np.array([str(getattr(c, "cam_id", c)) for c in self.keys]))


def candidates_seeing(survey: Sequence[Any], points: np.ndarray, k: int = CANDIDATES, *,
                      use_orig_coords: bool = False) -> List[Tuple[Any, int]]:
    """The survey cameras that see the most of ``points`` (3-D, e.g. where a placed neighbouring frame's grid of
    pixels meets the cloud), most first: [(camera, points seen)], at most ``k``. No image is read."""
    from .cameras import pixel_hits

    counts: Dict[int, int] = {}
    keep: Dict[int, Any] = {}
    for p in np.asarray(points, float).reshape(-1, 3):
        if not np.all(np.isfinite(p)):
            continue
        for c, *_rest in pixel_hits(list(survey), p, use_orig_coords=use_orig_coords):
            counts[id(c)] = counts.get(id(c), 0) + 1
            keep[id(c)] = c
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:k]
    return [(keep[i], n) for i, n in ranked]


def verify_inliers(matcher, frame_gray: np.ndarray, survey_gray: np.ndarray, *, turns: Sequence[int] = (0, 1, 2, 3),
                   threshold: float = VERIFY_THRESHOLD_PX) -> Tuple[int, int]:
    """Whether a frame sees what a survey photo sees: the most matches (``matcher``: an
    :class:`substrata.imagesync.LearnedMatcher`, LoFTR) consistent with one fundamental matrix over the frame
    turned by ``turns`` quarter turns (a survey photo may lie any way round). Both greyscale, sides multiples of
    8, at about the same scale (``VERIFY_*_WIDTH``). Returns (inliers, the quarter turn that gave them);
    ``VERIFY_MIN_INLIERS`` or more: it sees it (sa3_40m: no unrelated pair reached 43; 67% of the best related
    pairs reached 50)."""
    import cv2

    best, best_k = 0, 0
    for k in turns:
        g = np.ascontiguousarray(np.rot90(frame_gray, int(k)))
        a, b = matcher.match(g, survey_gray)
        if len(a) < 8:
            continue
        _F, mask = cv2.findFundamentalMat(np.float32(a), np.float32(b), cv2.USAC_MAGSAC, threshold, 0.999, 5000)
        n = 0 if mask is None else int(mask.sum())
        if n > best:
            best, best_k = n, int(k)
    return best, best_k
