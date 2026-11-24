from __future__ import annotations

# Standard Library
import copy
import csv
import datetime
import importlib
import json
import logging
import operator
import os
import re
import xml.etree.ElementTree as ET

# Third-Party Libraries
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from joblib import Parallel, delayed
import exifread

# Local Modules
from substrata import settings, geom, measurements
from substrata.compat import renamed_kwargs as _renamed_kwargs
from substrata.fileio import atomic_write as _atomic_write
from substrata.fileio import atomic_write_text as _atomic_write_text
from substrata.fileio import imread as _imread
from substrata.imagecrops import classifier_crop_box, get_crop_img
from substrata.pathrepair import path_basename
from substrata.logging import tqdm_joblib

logger = logging.getLogger(__name__)


def _atomic_write_json(data: dict, output_path: str) -> str:
    """Write ``data`` as indented JSON to ``output_path`` atomically
    (:func:`substrata.fileio.atomic_write`: an interrupted write cannot truncate
    the original file; its permission mode is kept).

    Args:
        data: JSON-serializable mapping to write.
        output_path: Destination path (created or replaced).

    Returns:
        str: The absolute path written.
    """
    return _atomic_write_text(output_path, json.dumps(data, indent=2) + "\n")


def _sync_camera_enabled_to_cams_xml(cameras: Cameras, xml_path: str) -> None:
    """Update ``<camera enabled="...">`` from in-memory :attr:`Camera.enabled`.

    Writes the XML file atomically. Intended for workflows such as ``camsync`` that
    re-enable cameras after pose transfer.

    Args:
        cameras: Loaded cameras whose ``enabled`` flags should be reflected in XML.
        xml_path: Path to the Metashape ``.cams.xml`` export.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError) as e:
        logger.warning("Could not parse %s for enabled sync: %s", xml_path, e)
        return
    root = tree.getroot()
    updated = 0
    for elem in root.findall(".//camera"):
        cid = elem.get("id")
        if cid is None or cid not in cameras.data:
            continue
        cam = cameras.data[cid]
        if not hasattr(cam, "enabled"):
            continue
        val = "true" if bool(cam.enabled) else "false"
        if elem.get("enabled") != val:
            elem.set("enabled", val)
            updated += 1
    if updated == 0:
        return
    xml_abs = os.path.abspath(xml_path)
    with _atomic_write(xml_abs, "wb") as xf:
        tree.write(xf, encoding="utf-8", xml_declaration=True)
    print(f"Updated {updated} <camera enabled> flags in {xml_abs}")


# --- Crop / thumbnail files ----------------------------------------------------

# EXIF UserComment payloads start with an 8-byte character-code prefix.
_EXIF_ASCII_PREFIX = b"ASCII\x00\x00\x00"
_CROP_STATUSES = ("generated", "skipped", "rejected", "failed")


def _size_pair(size) -> tuple:
    """Return ``(w, h)`` for an int (square) or a ``(w, h)`` pair."""
    if isinstance(size, (int, float)):
        return int(size), int(size)
    w, h = size
    return int(w), int(h)


def photo_folder_name(filepath: str) -> str | None:
    """Name of the folder holding a photo (e.g. ``ton_tof.auv``), if any.

    Handles both ``/`` and ``\\`` separators, as Metashape may store Windows
    paths.
    """
    parts = [p for p in re.split(r"[\\/]", filepath or "") if p]
    return parts[-2] if len(parts) >= 2 else None


_photo_folder_name = photo_folder_name  # deprecated alias (renamed 2026-11-25)


def view_crop_name(cam_filepath: str, x: float, y: float) -> str:
    """File name of a view's crop: ``<photo stem>_<x>_<y>.jpg`` (integer pixel).

    The pixel in the name tells whether a stored crop still shows the point: after
    the point moves (or the plot is aligned anew) its projection lands elsewhere,
    so the old crop no longer matches the current view.
    """
    stem = os.path.splitext(re.split(r"[\\/]", cam_filepath or "")[-1])[0]
    return f"{stem}_{int(round(float(x)))}_{int(round(float(y)))}.jpg"


def find_view_crop(
    root: str, ann_id: str, cam_filepath: str, x: float, y: float,
) -> str | None:
    """An annotation's stored crop of one view (photo + pixel), if there is one.

    Looks for ``<root>/<ann_id>/<photo folder>/<photo stem>_<x>_<y>.jpg``, then for
    the older names without the pixel (``<photo folder>/<photo file>``, and the
    legacy ``<root>/<ann_id>/<photo file>``), whose pixel is unknown.
    """
    base = os.path.join(root, str(ann_id))
    folder = photo_folder_name(cam_filepath)
    sub = os.path.join(base, folder) if folder else base
    photo = re.split(r"[\\/]", cam_filepath or "")[-1]
    return _first_existing(
        os.path.join(sub, view_crop_name(cam_filepath, x, y)),
        os.path.join(sub, photo), os.path.join(base, photo),
    )


def _first_existing(*paths) -> str | None:
    """Return the first of ``paths`` that is an existing file, else None."""
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


# --- Projection (3-D points -> pixels) -----------------------------------------
# One core shared by Camera.get_pixel_coords (a point, one camera), Camera.project_points
# (many points, one camera) and pixel_hits (a point, many cameras): camera_points_to_pixels and
# _relevance. The poses are inverted per call for one camera (always its current
# pose); for many cameras they are stacked once (CameraStack), cached on their
# Cameras and rebuilt after invalidate_projection_cache.

_SENSOR_FIELDS = ("k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4", "b2", "fx", "fy", "cx", "cy", "width", "height",
                  "r2_max")
_STACKS_KEPT = 2  # camera stacks kept per Cameras (or for cameras without one, in _HITS_CACHE)
_PROJ_STATE = {"gen": 0}  # bumped when poses or sensors change in place: every stack is rebuilt
_HITS_CACHE: dict = {}  # stacks of camera lists that do not belong to a Cameras


def invalidate_projection_cache() -> None:
    """Camera poses, sensors or the cameras of a :class:`Cameras` changed: stacks are rebuilt on next use."""
    _PROJ_STATE["gen"] += 1
    _HITS_CACHE.clear()


_projection_changed = invalidate_projection_cache  # deprecated alias (renamed 2026-11-25)


def lens_distortion(x, y, s):
    """Metashape's frame camera distortion of normalised coordinates ``(x, y)`` (``X/Z``, ``Y/Z``) through
    sensor ``s``: the distorted normalised coordinates ``(x', y')``. Agisoft Metashape manual, appendix
    "Camera models" (frame cameras)::

        r2 = x^2 + y^2
        x' = x (1 + K1 r2 + K2 r2^2 + K3 r2^3 + K4 r2^4) + (P1 (r2 + 2 x^2) + 2 P2 x y) (1 + P3 r2 + P4 r2^2)
        y' = y (1 + K1 r2 + K2 r2^2 + K3 r2^3 + K4 r2^4) + (P2 (r2 + 2 y^2) + 2 P1 x y) (1 + P3 r2 + P4 r2^2)

    Note the tangential terms: Metashape's P1 goes with x (OpenCV's p2) and P2 with y (OpenCV's p1).
    Checked against Metashape's own ``Calibration.project`` (tests/test_lens_metashape.py)."""
    r2 = x * x + y * y
    radial = 1 + r2 * (s.k1 + r2 * (s.k2 + r2 * (s.k3 + r2 * s.k4)))
    tang = 1 + r2 * (s.p3 + r2 * s.p4)
    x_dist = x * radial + (s.p1 * (r2 + 2 * x * x) + 2 * s.p2 * x * y) * tang
    y_dist = y * radial + (s.p2 * (r2 + 2 * y * y) + 2 * s.p1 * x * y) * tang
    return x_dist, y_dist


_lens = lens_distortion  # deprecated alias (renamed 2026-11-25)


def _lens_jacobian(x, y, s):
    """The derivatives of :func:`lens_distortion` at ``(x, y)``: ``(dx'/dx, dx'/dy, dy'/dx, dy'/dy)``."""
    r2 = x * x + y * y
    radial = 1 + r2 * (s.k1 + r2 * (s.k2 + r2 * (s.k3 + r2 * s.k4)))
    d_radial = s.k1 + r2 * (2 * s.k2 + r2 * (3 * s.k3 + r2 * 4 * s.k4))  # (d radial / d r2)
    tang = 1 + r2 * (s.p3 + r2 * s.p4)
    d_tang = s.p3 + 2 * s.p4 * r2
    tx = s.p1 * (r2 + 2 * x * x) + 2 * s.p2 * x * y
    ty = s.p2 * (r2 + 2 * y * y) + 2 * s.p1 * x * y
    xx = radial + 2 * x * x * d_radial + (6 * s.p1 * x + 2 * s.p2 * y) * tang + 2 * x * tx * d_tang
    xy = 2 * x * y * d_radial + (2 * s.p1 * y + 2 * s.p2 * x) * tang + 2 * y * tx * d_tang
    yx = 2 * x * y * d_radial + (2 * s.p2 * x + 2 * s.p1 * y) * tang + 2 * x * ty * d_tang
    yy = radial + 2 * y * y * d_radial + (6 * s.p2 * y + 2 * s.p1 * x) * tang + 2 * y * ty * d_tang
    return xx, xy, yx, yy


def _r2_max(k1, k2, k3, k4, fx, width, height) -> float:
    """The largest squared normalised radius a sensor images: below the radius where its radial distortion
    folds back (``d/dr [r (1 + K1 r^2 + ... + K4 r^8)] = 1 + 3 K1 r^2 + 5 K2 r^4 + 7 K3 r^6 + 9 K4 r^8`` turns
    0; beyond it, points outside the lens's field would land inside the image again), and within 1.01 times
    the image's longer side of the axis (a bound where it never folds)."""
    bound = 1.01 * max(width, height) ** 2 / fx**2
    roots = np.roots([9 * k4, 7 * k3, 5 * k2, 3 * k1, 1.0]) if any((k1, k2, k3, k4)) else []
    folds = [float(t.real) for t in roots if abs(t.imag) < 1e-12 and t.real > 0]
    return min([bound, *folds])


def pixels_to_normalized(x_img, y_img, s, iterations: int = 20):
    """The normalised coordinates ``(x, y)`` that sensor ``s`` images at pixel ``(x_img, y_img)``: the
    inverse of :func:`camera_points_to_pixels` (Newton's method on :func:`lens_distortion`, from the undistorted guess). Scalars or
    arrays. Converges to ~1e-12 px over the image for the calibrations Metashape produces (closer than
    Metashape's own ``unproject``); NaN where there is no such point (a pixel the lens model cannot reach:
    the far corners of a strongly distorted photo; or one beyond the fold, see :func:`_r2_max`)."""
    y_d = (np.asarray(y_img, float) - s.cy) / s.fy
    x_d = (np.asarray(x_img, float) - s.cx - s.b2 * y_d) / s.fx
    x, y = x_d.copy(), y_d.copy()
    for _ in range(iterations):
        ex, ey = lens_distortion(x, y, s)
        ex, ey = ex - x_d, ey - y_d
        a, b, c, d = _lens_jacobian(x, y, s)
        det = a * d - b * c
        x = x - (d * ex - b * ey) / det
        y = y - (a * ey - c * ex) / det
    ex, ey = lens_distortion(x, y, s)
    off = np.hypot(ex - x_d, ey - y_d) * s.fy > 1e-4  # (px: not converged, no solution)
    bad = off | ~(x * x + y * y < s.r2_max)
    if np.ndim(bad):
        x, y = np.where(bad, np.nan, x), np.where(bad, np.nan, y)
    elif bad:
        x, y = np.float64(np.nan), np.float64(np.nan)
    return x, y


_undistort = pixels_to_normalized  # deprecated alias (renamed 2026-11-25)


def camera_points_to_pixels(xc, yc, zc, s):
    """Pixels of camera-frame points ``(xc, yc, zc)`` through sensor ``s`` (a :class:`Sensor`, or a
    :class:`CameraStack`, row by row): Metashape's frame camera model (:func:`lens_distortion`: K1-K4 radial, P1-P4
    tangential), then ``u = cx + x' fx + y' B2`` (fx = f + B1: affinity; B2: skew), ``v = cy + y' fy``.
    Scalars or arrays (broadcast).

    Pixel coordinates are continuous, in Metashape's convention: ``(0, 0)`` is the top-left corner of the
    top-left pixel, pixel ``(i, j)`` (column, row) covers ``[i, i + 1) x [j, j + 1)`` and its centre is
    ``(i + 0.5, j + 0.5)``; the grid is the image as stored (EXIF orientation not applied, see
    :func:`substrata.fileio.imread`).

    Returns:
        ``(x_img, y_img, in_view)``, unrounded. In view: in front of the camera (one behind it can project
        into the image too), within the radius the lens images (``r2_max``: before its distortion folds
        back into the image, see :func:`_r2_max`) and inside the image: ``0 <= x < width``,
        ``0 <= y < height``.
    """
    x_norm = xc / zc
    y_norm = yc / zc
    r2 = x_norm**2 + y_norm**2
    x_dist, y_dist = lens_distortion(x_norm, y_norm, s)
    x_img = s.fx * x_dist + s.b2 * y_dist + s.cx
    y_img = s.fy * y_dist + s.cy
    in_view = (
        (zc > 0)
        & (r2 < s.r2_max)
        & (x_img >= 0) & (x_img < s.width)
        & (y_img >= 0) & (y_img < s.height)
    )
    return x_img, y_img, in_view


_distort = camera_points_to_pixels  # deprecated alias (renamed 2026-11-25)


def int_pixel(v, size):
    """The integer pixel coordinate of an in-view continuous coordinate ``v`` (:func:`camera_points_to_pixels`): ``v``
    rounded (half to even), at most ``size - 1``.

    Rounded, not floored: a crop of even size centred on the integer coordinate then has the point within
    half a pixel of its centre (the crop's centre is a pixel corner), and the crop names
    (:func:`view_crop_name`) of stored crops stay as they are. The pixel *containing* the point, to index an
    image array with, is ``floor(v)`` instead. Clamped, so ``v`` in ``[size - 0.5, size)`` gives the last
    pixel, never ``size``. Scalars or arrays.
    """
    return np.minimum(np.rint(v), np.asarray(size, float) - 1)


_int_pixel = int_pixel  # deprecated alias (renamed 2026-11-25)


def _relevance(dist_sq, x_img, y_img, fx, fy, width, height):
    """Relevance of a view (a cost: lower is better): the squared distance from the camera centre to the point
    over ``fx * fy`` (the area one pixel covers there, for a surface facing the camera), times
    (10 + the pixel's distance from the image centre, L1).

    Every term of the first factor is a squared length, so scaling the chunk scales every cost alike: the
    ranking of views, and the relevance weights of ``predict --method weighted`` (``relevance_min /
    relevance``), do not depend on the chunk's units. Until November 2026 the first factor was
    ``(|depth| + dist_sq) / (fx * fy)``, which mixed a length with a squared length; on sa3_40m (model units
    of 0.19 m) 97 % of 400 points keep the same best view."""
    return (dist_sq / (fx * fy)) * (
        10 + np.abs(x_img - 0.5 * width) + np.abs(y_img - 0.5 * height)
    )


def _put_pose_status(entry: dict, cam) -> None:
    """A meta JSON camera entry gets the camera's ``pose_status`` (where its pose came from, e.g. camsync
    --match-images' ``matched`` / ``estimated`` / ``unverified``); a camera whose ``pose_status`` is None has it
    removed; one without the attribute keeps what the file has."""
    if not hasattr(cam, "pose_status"):
        return
    if cam.pose_status:
        entry["pose_status"] = str(cam.pose_status)
    else:
        entry.pop("pose_status", None)


def can_project(cam, use_orig_coords: bool = True) -> bool:
    """Whether ``cam`` can project points: it has a calibrated sensor and a pose of its own.

    Not a camera whose sensor has no calibration in cams.xml (``sensor`` None: e.g. close-ups and video frames
    left unaligned), nor one loaded with a placeholder pose (a null centre or transform in meta.json:
    ``missing_pose_from_meta``), nor one without a pose in the frame asked for. Such cameras see nothing:
    :func:`pixel_hits`, :meth:`Camera.get_pixel_coords` and :meth:`Camera.project_points` treat every point as
    out of their view.
    """
    if getattr(cam, "sensor", None) is None or getattr(cam, "missing_pose_from_meta", False):
        return False
    attr = "orig_camera_transform" if use_orig_coords else "camera_transform"
    return not hasattr(cam, attr) or getattr(cam, attr) is not None  # (a camera double may carry none)


class CameraStack:
    """The projection arrays of a list of cameras, one row per camera, in one frame.

    Only the cameras that can project (:func:`can_project`: a calibrated sensor and a pose) are stacked.
    Attributes: ``cams`` (those cameras), ``inv`` ``(M, 4, 4)`` inverse poses (world to camera), ``centres``
    ``(M, 3)`` camera centres, and the sensors' ``_SENSOR_FIELDS`` (k1-k4, p1-p4, b2, fx fy cx cy width
    height) as ``(M,)``
    arrays; ``use_orig_coords`` the frame. Made by :meth:`Cameras.projection_arrays` and :func:`pixel_hits`
    (cached; see :func:`_stack_for`). Raises AttributeError / TypeError / ValueError when a camera has no
    ``sensor`` attribute at all (a double) or a pose cannot be read as a 4x4 matrix or inverted.
    """

    def __init__(self, cams, use_orig_coords: bool = True):
        # (a camera double without a sensor attribute raises below: pixel_hits then asks it one at a time)
        self.cams = [c for c in cams if not hasattr(c, "sensor") or can_project(c, use_orig_coords)]
        self.use_orig_coords = bool(use_orig_coords)
        self.gen = _PROJ_STATE["gen"]
        sensors = [c.sensor for c in self.cams]  # (first: a camera without one fails before the inversion)
        for a in _SENSOR_FIELDS:
            setattr(self, a, np.array([float(getattr(s, a)) for s in sensors]))
        poses = [c.orig_camera_transform if use_orig_coords else c.camera_transform for c in self.cams]
        centres = [c.orig_coords if use_orig_coords else c.coords for c in self.cams]
        self.inv = (np.linalg.inv(np.stack([np.asarray(t, float).reshape(4, 4) for t in poses])) if poses
                    else np.zeros((0, 4, 4)))
        self.centres = (np.stack([np.asarray(c, float).reshape(3) for c in centres]) if centres
                        else np.zeros((0, 3)))

    def __len__(self) -> int:
        return len(self.cams)


def _stack_for(cams: list, use_orig_coords: bool = True, owner=None):
    """The :class:`CameraStack` of ``cams`` (a non-empty list), made once per list of cameras and frame.

    Kept on ``owner`` (the :class:`Cameras` they come from; by default the first camera's parent), the last
    ``_STACKS_KEPT`` lists, else in ``_HITS_CACHE``. A list is looked up by its length and a sample of its
    cameras, and a hit is checked against the whole list (the cameras are held, so their ids cannot be
    reused): two lists of the same length that differ only between the sampled cameras get their own
    stacks. Poses and sensors are taken as fixed: what
    changes them in place (world transforms, camera sync, reading sensors, replacing cameras) calls
    :func:`invalidate_projection_cache`. ``enabled`` is not stacked, nor are cameras that cannot project
    (:func:`can_project`). None if a pose cannot be read or inverted (then pixel_hits goes one camera at a
    time); that is remembered too.
    """
    n = len(cams)
    key = (bool(use_orig_coords), n, tuple(id(c) for c in cams[:: max(1, n // 64)]), id(cams[-1]))
    owner = owner if owner is not None else getattr(cams[0], "parent", None)
    store = owner.__dict__.setdefault("_proj_stacks", {}) if isinstance(owner, Cameras) else _HITS_CACHE
    gen = _PROJ_STATE["gen"]
    got = store.get(key)
    # (the key samples the list: a hit is confirmed on the whole list, by identity - C speed, ~1 ms for 60k)
    if got is not None and got[0] == gen and len(got[2]) == n and all(map(operator.is_, got[2], cams)):
        return got[1]
    try:
        st = CameraStack(cams, use_orig_coords)
    except (AttributeError, TypeError, ValueError):
        st = None
    for k in [k for k, v in store.items() if v[0] != gen or k == key]:  # (stale ones go)
        del store[k]
    while len(store) >= _STACKS_KEPT:
        del store[next(iter(store))]
    store[key] = (gen, st, list(cams))
    return st


def pixel_hits(cams, coords, use_orig_coords: bool = True, enabled_only: bool = True) -> list:
    """The cameras that see 3-D point ``coords`` in their image.

    As :meth:`Camera.get_pixel_coords` gives them one camera at a time, but in one set of numpy operations
    over all cameras: choosing views among tens of thousands of cameras (an AUV's) per point took ~1 s that
    way. Same projection core (Metashape's lens model, the in-view test, the relevance metric), on the
    cameras' stacked poses and sensors (:class:`CameraStack`, cached on their :class:`Cameras`), in either
    frame. Cameras without a calibrated sensor or a pose of their own (:func:`can_project`) see nothing.

    Args:
        cams: Cameras (a list, any iterable, or a :class:`Cameras`).
        coords: The 3-D point, in the frame of ``use_orig_coords``.
        use_orig_coords: The cameras' original poses (else their current, world-frame ones).
        enabled_only: Skip cameras whose ``enabled`` is False (read afresh on every call).

    Returns:
        ``[(camera, x, y, depth, relevance)]`` in the order of ``cams``: ``x``, ``y`` int pixels
        (:func:`int_pixel`), ``depth`` and ``relevance`` floats.
    """
    owner = cams if isinstance(cams, Cameras) else None
    cams = cams if isinstance(cams, list) else list(cams)
    if not cams:
        return []
    st = _stack_for(cams, use_orig_coords, owner)
    if st is None:
        out = []
        for cam in cams:
            if enabled_only and getattr(cam, "enabled", True) is False:
                continue
            x, y, depth, rel = cam.get_pixel_coords(coords, use_orig_coords=use_orig_coords)
            if x is not None:
                out.append((cam, x, y, depth, rel))
        return out
    p = np.append(np.asarray(coords, float).ravel()[:3], 1.0)
    proj = st.inv @ p
    with np.errstate(divide="ignore", invalid="ignore"):
        xi, yi, ok = camera_points_to_pixels(proj[:, 0], proj[:, 1], proj[:, 2], st)
    idx = np.flatnonzero(ok)
    if enabled_only:  # (read afresh, of the hits only: cheap, and never stale)
        idx = np.array([i for i in idx.tolist() if getattr(st.cams[i], "enabled", True) is not False], int)
    if not len(idx):
        return []
    dist_sq = ((st.centres[idx] - p[:3]) ** 2).sum(1)
    rm = _relevance(dist_sq, xi[idx], yi[idx],
                    st.fx[idx], st.fy[idx], st.width[idx], st.height[idx])
    xs, ys = int_pixel(xi[idx], st.width[idx]), int_pixel(yi[idx], st.height[idx])
    return [(st.cams[i], int(x), int(y), float(proj[i, 2]), r)
            for i, x, y, r in zip(idx.tolist(), xs.tolist(), ys.tolist(), rm.tolist())]


def _file_names(folder: str) -> set:
    """The names of the files in ``folder`` (empty if it cannot be listed)."""
    try:
        with os.scandir(folder) as it:
            return {e.name for e in it if e.is_file()}
    except OSError:
        return set()


def count_crop_statuses(statuses) -> dict:
    """Count crop/thumbnail statuses, always including every status key."""
    counts = {status: 0 for status in _CROP_STATUSES}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


_count_statuses = count_crop_statuses  # deprecated alias (renamed 2026-11-25)


def assign_camera_types(cams, focal_tol: float = settings.CAMERA_TYPE_FOCAL_TOL):
    """Group cameras into camera types and set ``cam.camera_type`` on each.

    A type is a sensor resolution (either orientation) with a focal length
    within ``focal_tol`` (fraction): e.g. two identical cameras of one survey,
    or an AUV camera across missions, are one type, whatever their folders; a
    macro close-up camera or a GoPro is another. Cameras without a sensor get
    ``camera_type`` None.

    Args:
        cams: Cameras (a :class:`Cameras` or a list).
        focal_tol: Relative focal-length tolerance within a type.

    Returns:
        The type names (``"<long>x<short> f<focal px>"``), most cameras first.
    """
    cams = list(cams.data.values()) if hasattr(cams, "data") else list(cams)
    types: list = []  # [short, long, fx, name, n]
    for cam in cams:
        sensor = getattr(cam, "sensor", None)
        fx = getattr(sensor, "fx", None)
        if sensor is None or not fx or not sensor.width or not sensor.height:
            cam.camera_type = None
            continue
        short, long_ = sorted((int(sensor.width), int(sensor.height)))
        for t in types:
            same_lens = abs(float(fx) - t[2]) <= focal_tol * t[2]
            if (t[0], t[1]) == (short, long_) and same_lens:
                break
        else:
            t = [short, long_, float(fx), f"{long_}x{short} f{round(float(fx))}", 0]
            types.append(t)
        t[4] += 1
        cam.camera_type = t[3]
    return [t[3] for t in sorted(types, key=lambda t: -t[4])]


def save_photo_crops(matches, root=None, crop_size=None, overwrite: bool = False) -> list:
    """Save the crops of several matches of one photo, decoding the photo once.

    As :meth:`ImageMatch.save_crop` for each match (all of the same camera /
    photo): crops that would extend past the image edge are rejected, the
    photo's EXIF is kept (minus its thumbnail) with ``potentially_obstructed``
    in the UserComment, and crops are saved at ``settings.CROP_JPEG_QUALITY``.

    Args:
        matches: Image matches of one photo.
        root: Crops directory (defaults as in :meth:`ImageMatch.crop_path`).
        crop_size: Crop size ``(w, h)`` or int (default
            ``settings.CROP_SIZE_DEFAULT``).
        overwrite: Replace existing crops (an older-named one is removed).

    Returns:
        Per match ``"generated"``, ``"skipped"`` (already exists),
        ``"rejected"`` (too close to the image edge) or ``"failed"`` (photo
        missing or unreadable).
    """
    out = [None] * len(matches)
    crop_w, crop_h = _size_pair(crop_size if crop_size is not None else settings.CROP_SIZE_DEFAULT)
    todo = []
    for i, m in enumerate(matches):
        r = m._crops_root(root)
        existing = m.find_crop(r)
        if existing and not overwrite:
            out[i] = "skipped"
        else:
            todo.append((i, m, r, existing))
    if not todo:
        return out
    src = matches[todo[0][0]].cam.filepath
    if not src or not os.path.isfile(src):
        for i, *_ in todo:
            out[i] = "failed"
        return out
    try:
        with Image.open(src) as img:  # (decoded on the first crop, once)
            exif_src = img.info.get("exif")
            for i, m, r, existing in todo:
                if not m.is_clear_from_image_edge((crop_w, crop_h), img.size):
                    out[i] = "rejected"
                    continue
                # The classifier crop box (imagecrops.get_crop_img's), so a centre
                # crop of this crop equals a crop cut from the photo directly.
                crop = img.crop(classifier_crop_box(m.x, m.y, crop_w, crop_h))
                if crop.mode not in ("RGB", "L"):
                    crop = crop.convert("RGB")
                exif_bytes = _exif_with_crop_metadata(exif_src, {
                    "potentially_obstructed": bool(getattr(m, "potentially_obstructed", False)),
                    "pred_label": "",
                    "pred_conf": 0.0,
                })
                out_path = m.crop_path(r)
                _save_image_atomic(
                    crop, out_path, quality=settings.CROP_JPEG_QUALITY, exif=exif_bytes)
                if existing and os.path.abspath(existing) != os.path.abspath(out_path):
                    os.remove(existing)
                out[i] = "generated"
    except (OSError, ValueError) as e:
        logger.warning("Could not crop %s: %s", src, e)
        out = [o if o is not None else "failed" for o in out]
    return out


def _save_image_atomic(img, output_path: str, **save_kwargs) -> None:
    """Save a PIL image as JPEG atomically, so readers never see a partial file
    (:func:`substrata.fileio.atomic_write`)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with _atomic_write(output_path, "wb") as fh:
        img.save(fh, format="JPEG", **save_kwargs)


def _exif_with_crop_metadata(exif_source, data: dict) -> bytes:
    """Return EXIF bytes carrying ``data`` as JSON in the UserComment tag.

    Keeps the other tags of ``exif_source`` (EXIF bytes or a filepath) where
    possible, but drops the embedded thumbnail, which shows the whole photo.

    Args:
        exif_source: Existing EXIF bytes or a JPEG filepath; may be empty/None.
        data: JSON-serializable crop metadata.

    Returns:
        bytes: EXIF block for ``PIL.Image.save(exif=...)`` or ``piexif.insert``.
    """
    import piexif

    payload = dict(data)
    payload.setdefault("v", settings.CROP_METADATA_VERSION)
    comment = _EXIF_ASCII_PREFIX + json.dumps(payload).encode("ascii")

    exif_dict = None
    if exif_source:
        try:
            exif_dict = piexif.load(exif_source)
        except Exception:  # noqa: BLE001 - unreadable EXIF: start afresh
            exif_dict = None
    if exif_dict:
        exif_dict["thumbnail"] = None
        exif_dict["1st"] = {}
        exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = comment
        # piexif expects rationals; some cameras store plain ints here
        for tag in (piexif.ImageIFD.XResolution, piexif.ImageIFD.YResolution):
            value = exif_dict.get("0th", {}).get(tag)
            if isinstance(value, int):
                exif_dict["0th"][tag] = (value, 1)
        try:
            return piexif.dump(exif_dict)
        except Exception:  # noqa: BLE001 - malformed tags: keep only ours
            pass
    return piexif.dump({"Exif": {piexif.ExifIFD.UserComment: comment}})


def read_crop_metadata(path: str) -> dict | None:
    """Read the metadata JSON stored in a crop's EXIF UserComment.

    Reads crops written by reefcat (no ``"v"`` key) and by
    :meth:`ImageMatch.save_crop` alike. Typical keys: ``potentially_obstructed``,
    ``pred_label``, ``pred_conf`` and ``pred_model`` (the classifier's fingerprint,
    :func:`classification.classifier_fingerprint`).

    Args:
        path: Crop image filepath.

    Returns:
        dict | None: The metadata, or None if absent or unreadable.
    """
    try:
        import piexif

        exif_dict = piexif.load(path)
        raw = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    except Exception:  # noqa: BLE001 - missing file or EXIF
        return None
    if not raw or not raw.startswith(b"ASCII"):
        return None
    try:
        text = raw[len(_EXIF_ASCII_PREFIX):].decode("ascii").rstrip("\x00")
        data = json.loads(text) if text else None
    except (UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def update_crop_metadata(path: str, updates: dict) -> None:
    """Merge ``updates`` into a crop's metadata without re-encoding the JPEG."""
    import piexif

    data = read_crop_metadata(path) or {}
    data.update(updates)
    piexif.insert(_exif_with_crop_metadata(path, data), path)


_update_crop_metadata = update_crop_metadata  # deprecated alias (renamed 2026-11-25)


def _nominal_sensor(sensor_elem, sensor_id, label, resolution) -> "Sensor | None":
    """A sensor Metashape did not calibrate (none of its photos aligned, e.g. close-ups) from its nominal lens: the
    ``focal_length`` (mm) and ``pixel_width`` (mm) properties in cams.xml, the principal point at the centre and no
    distortion (``calibration_class`` "nominal"). None without those properties. Its cameras still have no pose
    until one is given (e.g. ``camsync --match-images``), so they see nothing until then."""
    props = {p.get("name"): p.get("value") for p in sensor_elem.findall("property")}
    try:
        f_mm, px_mm = float(props.get("focal_length")), float(props.get("pixel_width"))
    except (TypeError, ValueError):
        return None
    if not (f_mm > 0 and px_mm > 0):
        return None
    params = {k: 0.0 for k in ("cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")}
    sensor = Sensor(sensor_id, label, resolution, dict(params, f=f_mm / px_mm))
    sensor.calibration_class = "nominal"
    return sensor


class Sensor:
    """Class that holds calibration parameters for a specific camera sensor (Metashape's frame camera
    model: see :func:`lens_distortion`)."""

    # (defaults for the optional terms, also for a Sensor pickled before they were read)
    b1 = b2 = k4 = p3 = p4 = 0.0

    def __init__(self, sensor_id, label, resolution, calibration_params):
        """Initialize a Sensor instance.

        Args:
            sensor_id (int): Unique sensor identifier.
            label (str): Sensor label/description.
            resolution (dict): Dictionary with 'width' and 'height' keys.
            calibration_params (dict): Dictionary containing calibration parameters.
        """
        self.sensor_id = sensor_id
        self.label = label
        self.width = resolution["width"]
        self.height = resolution["height"]

        # Calibration parameters
        self.f = calibration_params["f"]
        self.cx_metashape = calibration_params["cx"]
        self.cy_metashape = calibration_params["cy"]
        self.b1 = calibration_params.get("b1", 0.0)
        self.b2 = calibration_params.get("b2", 0.0)
        self.k1 = calibration_params["k1"]
        self.k2 = calibration_params["k2"]
        self.k3 = calibration_params["k3"]
        self.k4 = calibration_params.get("k4", 0.0)
        self.p1 = calibration_params["p1"]
        self.p2 = calibration_params["p2"]
        self.p3 = calibration_params.get("p3", 0.0)
        self.p4 = calibration_params.get("p4", 0.0)

        # Derived parameters
        self.fx = self.f + self.b1
        self.fy = self.f
        self.cx = self.width / 2 + self.cx_metashape
        self.cy = self.height / 2 + self.cy_metashape
        # "adjusted": Metashape's calibration (cams.xml); "nominal": the lens's focal length only (_nominal_sensor)
        self.calibration_class = "adjusted"

    @property
    def r2_max(self) -> float:
        """The largest squared normalised radius this sensor images (:func:`_r2_max`), worked out once
        for its terms."""
        key = (self.k1, self.k2, self.k3, self.k4, self.fx, self.width, self.height)
        got = self.__dict__.get("_r2_max")
        if got is None or got[0] != key:
            got = (key, _r2_max(*key))
            self.__dict__["_r2_max"] = got
        return got[1]


class Cameras:
    """Container class that holds a collection of Camera objects."""

    def __init__(self, cams_meta_filepath=None, cams_xml_filepath=None):
        """Initialize the Cameras container.

        Depending on the input filepaths, the container is filled from a JSON
        metadata file and/or an XML file containing sensor parameters.

        Args:
            cams_meta_filepath (str, optional): Path to the cameras metadata file.
            cams_xml_filepath (str, optional): Path to the cameras XML file.
        """
        self.data = {}
        self.sensors = {}  # Dict mapping sensor_id -> Sensor instance
        self.world_transform = np.eye(4)
        if cams_meta_filepath:
            self.cams_meta_filepath = cams_meta_filepath
            self.get_cams_from_file(cams_meta_filepath)
        if cams_xml_filepath:
            self.cams_xml_filepath = cams_xml_filepath
            self.get_cam_sensor_parameters_from_file(cams_xml_filepath)

    def __str__(self) -> str:
        """Return a concise, human-readable summary of the cameras collection."""
        total = len(self.data)
        sensors_count = len(getattr(self, "sensors", {}) or {})
        wt_is_identity = self.world_transform_is_identity
        groups = self.group_names
        # Count cameras per group
        group_counts = {}
        for cam in self.data.values():
            g = getattr(cam, "group", None)
            if g is None:
                continue
            group_counts[g] = group_counts.get(g, 0) + 1

        # Basic availability stats
        num_with_coords = sum(
            1 for cam in self.data.values() if getattr(cam, "coords", None) is not None
        )
        num_with_datetime = sum(
            1
            for cam in self.data.values()
            if getattr(cam, "datetime", None) is not None
        )
        num_with_depth = sum(
            1
            for cam in self.data.values()
            if getattr(cam, "depth_sensor_m", None) is not None
        )
        num_enabled = sum(
            1 for cam in self.data.values() if getattr(cam, "enabled", True) is True
        )

        # Coordinate bounds (if available)
        try:
            coords_list = [
                cam.coords
                for cam in self.data.values()
                if getattr(cam, "coords", None) is not None
            ]
            if coords_list:
                C = np.vstack(coords_list)
                cmin = C.min(axis=0)
                cmax = C.max(axis=0)
                extent = cmax - cmin
                bb_str = (
                    f"[min=({cmin[0]:.3f}, {cmin[1]:.3f}, {cmin[2]:.3f}), "
                    f"max=({cmax[0]:.3f}, {cmax[1]:.3f}, {cmax[2]:.3f}), "
                    f"extent=({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})]"
                )
            else:
                bb_str = "unavailable"
        except Exception:
            bb_str = "unavailable"

        lines = [
            "Cameras(",
            f"  count={total}, sensors={sensors_count}, enabled={num_enabled},",
            f"  with_coords={num_with_coords}, with_datetime={num_with_datetime}, with_depth_sensor_m={num_with_depth},",
            f"  world_transform={'identity' if wt_is_identity else 'non-identity'},",
            f"  groups={groups if groups else '[]'},",
            f"  groups_counts={group_counts if group_counts else {}},",
            f"  coords_bounds={bb_str}",
            ")",
        ]
        return "\n".join(lines)

    @property
    def coords(self):
        """List of coordinates for every camera in the collection."""
        return [camera.coords for camera in self.data.values()]

    # @property
    # def depths(self):
    #     return [camera.depth for camera in self.data.values()]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        invalidate_projection_cache()

    def __delitem__(self, key):
        del self.data[key]
        invalidate_projection_cache()

    def __getstate__(self):
        """Pickled / copied without the cached projection stacks (rebuilt on use)."""
        state = self.__dict__.copy()
        state.pop("_proj_stacks", None)
        return state

    def projection_arrays(self, use_orig_coords: bool = True) -> CameraStack | None:
        """The cameras' poses and sensors stacked as arrays (:class:`CameraStack`), in ``data`` order.

        Made once and cached (shared with :func:`pixel_hits` for the same cameras); rebuilt after poses,
        sensors or the cameras change through this module (world transforms, camera sync, reading sensors,
        setting or deleting cameras). Only the cameras that can project (:func:`can_project`); None if there
        are no cameras or a pose cannot be read or inverted.

        Args:
            use_orig_coords: The original poses (else the current, world-frame ones).
        """
        cams = list(self.data.values())
        return _stack_for(cams, use_orig_coords, owner=self) if cams else None

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        """Iterate over the cameras (a fresh iterator each time, so loops can nest)."""
        return iter(self.data.values())

    def __len__(self):
        """Return the number of cameras in the collection."""
        return len(self.data)

    def items(self):
        """Return a view of ``(cam_id, Camera)`` pairs in the collection."""
        return self.data.items()

    @property
    def world_transform_is_identity(self) -> bool:
        """Check if the world_transform is the identity matrix."""
        return np.allclose(self.world_transform, np.eye(4))

    def _filename_prefixes(self) -> set:
        """Unique filename prefixes across all cameras (used by Camera.group).

        Cached per ``data`` dict and size, so deriving every camera's group is
        linear instead of quadratic in the number of cameras.
        """
        cache = getattr(self, "_prefix_cache", None)
        if cache is None or cache[0] is not self.data or cache[1] != len(self.data):
            prefixes = {_filename_prefix(cam.filename) for cam in self.data.values()}
            cache = (self.data, len(self.data), prefixes)
            self._prefix_cache = cache
        return cache[2]

    @property
    def group_names(self):
        """Sorted list of unique group names across all cameras."""
        return sorted(
            {cam.group for cam in self.data.values() if hasattr(cam, "group")}
        )

    def show(self, pcd, color=False):
        """Show the camera positions over a point-cloud ortho map.

        Args:
            pcd (PointCloud): The pointcloud to draw as background.
            color (bool): Draw the background in true colour when True;
                otherwise (default) desaturate it to grey so the red camera
                markers stand out.

        Returns:
            PIL.Image.Image: Top-down ortho map with camera markers (red,
            filled/hollow by camera group).
        """
        from substrata.ortho import OrthoMap

        return OrthoMap(pcd).show(
            highlights=self,
            fill_by_group=True,
            grayscale=not color,
        )

    def append(self, cam):
        """Add a camera to the collection and set it as its parent.

        Args:
            cam (Camera): The camera to add. Its ``cam_id`` is used as the key.

        Raises:
            ValueError: If a camera with the same ``cam_id`` already exists.
        """
        if cam.cam_id in self.data:
            raise ValueError(f"Camera with id {cam.cam_id} already exists.")
        else:
            self.data[cam.cam_id] = cam
            self.data[cam.cam_id].parent = self
            invalidate_projection_cache()
            # TO DO: any other changes (eg transforms) to be implemented on append?

    @_renamed_kwargs(transform_matrix="transform")
    def apply_transform(self, transform):
        """Apply a 4x4 transform to all camera coordinates and their poses.

        ``world_transform`` becomes ``transform @ world_transform``.

        Args:
            transform (np.ndarray): A 4x4 homogeneous transformation matrix.
        """
        for cam_id in self.data:
            self.data[cam_id].transform_coords(transform)
        self.world_transform = np.dot(np.array(transform), self.world_transform)

    transform_coords = apply_transform  # deprecated alias (renamed 2026-11-25)

    @_renamed_kwargs(transform_matrix="transform")
    def reverse_transform(self, transform):
        """Set the original poses from the current ones and the transform between them.

        For cameras whose ``coords`` / ``camera_transform`` are in a frame
        ``transform`` maps the original (model) frame to:
        ``orig = inv(T) @ current``, and ``world_transform`` becomes ``T``
        (``current = T @ orig`` then holds, whatever it was before).

        Args:
            transform (np.ndarray): The 4x4 transform from the original
                frame to the current one.
        """
        T = np.asarray(transform, dtype=float)
        for cam_id in self.data:
            self.data[cam_id].reverse_transform_coords(T)
        self.world_transform = T.copy()

    get_original_coords = reverse_transform  # deprecated alias (renamed 2026-11-25)

    def subset(self, length):
        """Return a new container with the first ``length`` cameras.

        Args:
            length (int): Number of cameras to include, taken in insertion order.

        Returns:
            Cameras: New container with the selected cameras.
        """
        cameras_subset = self._empty_like()
        for cam_id in list(self.data.keys())[:length]:
            cameras_subset.data[cam_id] = self.data[cam_id]
        return cameras_subset

    def subset_by_filename_prefix(self, prefix):
        """Return a subset of cameras with filenames starting with a prefix.

        Args:
            prefix (str): Prefix to filter camera IDs.

        Returns:
            Cameras: New container with matching cameras.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            if cam.filename.startswith(prefix):
                cameras_subset.data[cam.cam_id] = cam
        return cameras_subset

    def subset_by_group(self, group_name):
        """Return a subset of cameras that belong to a given group.

        Args:
            group_name (str): Name of the group (as given by `Camera.group`).

        Returns:
            Cameras: New container with cameras from the requested group.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            if getattr(cam, "group", None) == group_name:
                cameras_subset.data[cam.cam_id] = cam
        return cameras_subset

    def subset_by_sensor(self, sensor_id):
        """Return a subset of cameras that use a specific sensor.

        Args:
            sensor_id (int): The sensor ID to filter by.

        Returns:
            Cameras: New container with cameras using the specified sensor.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            if getattr(cam, "sensor_id", None) == sensor_id:
                cameras_subset.data[cam.cam_id] = cam
        if len(cameras_subset.data) == 0:
            print(f"No cameras found with sensor_id={sensor_id}")
        return cameras_subset

    def subset_by_dates(self, dates: list[str]) -> "Cameras":
        """Return a subset of cameras whose EXIF date matches one of ``dates``.

        Matches the first 10 characters of ``cam.datetime`` (``YYYY:MM:DD``).
        Input dates may use either ``YYYY-MM-DD`` or ``YYYY:MM:DD`` form and
        are normalized internally. Cameras without ``cam.datetime`` set are
        excluded. Assumes :meth:`get_datetime_originals` has already been
        called on this container.

        Args:
            dates: Iterable of date strings (``YYYY-MM-DD`` or ``YYYY:MM:DD``).

        Returns:
            Cameras: New container with cameras whose EXIF date matches.
        """
        wanted = {d.replace("-", ":")[:10] for d in dates}
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            dt = getattr(cam, "datetime", None)
            if dt is None:
                continue
            if str(dt)[:10] in wanted:
                cameras_subset.data[cam.cam_id] = cam
        return cameras_subset

    def datetime_date_counts(self) -> dict[str, int]:
        """Count cameras per EXIF date (``YYYY:MM:DD``).

        Assumes :meth:`get_datetime_originals` has already been called.
        Cameras without ``cam.datetime`` set are skipped.

        Returns:
            dict[str, int]: Ordered mapping of ``YYYY:MM:DD`` to camera count,
            sorted ascending by date.
        """
        counts: dict[str, int] = {}
        for cam in self.data.values():
            dt = getattr(cam, "datetime", None)
            if dt is None:
                continue
            day = str(dt)[:10]
            counts[day] = counts.get(day, 0) + 1
        return dict(sorted(counts.items()))

    # Convenience alias to match the requested call-site: Cameras.group("name")
    def group(self, group_name):
        """Return a subset of cameras belonging to ``group_name``.

        Convenience alias for :meth:`subset_by_group`.

        Args:
            group_name (str): Name of the group (as given by ``Camera.group``).

        Returns:
            Cameras: New container with cameras from the requested group.
        """
        return self.subset_by_group(group_name)

    def _empty_like(self) -> "Cameras":
        """An empty container with this one's metadata (``world_transform``,
        ``sensors``, ``up_vector``, file paths, ...), for the subset helpers.

        The subset helpers put the *same* :class:`Camera` objects in the new
        container without re-parenting them: a camera's ``parent`` stays the
        container it was loaded into, so its ``group`` (and anything else
        derived from its parent) does not change when a subset is taken.
        Arrays, dicts and lists are copied (one level); private caches (the
        projection stacks, the filename prefixes) are not carried over.
        """
        subset = Cameras()
        for attr, val in self.__dict__.items():
            if attr == "data" or attr.startswith("_"):
                continue
            if isinstance(val, np.ndarray):
                val = val.copy()
            elif isinstance(val, (dict, list)):
                val = type(val)(val)
            try:
                setattr(subset, attr, val)
            except Exception:  # noqa: BLE001 - e.g. a read-only property
                pass
        return subset

    def filter_by_ids(self, cam_ids):
        """Return a new Cameras container containing only the specified camera ids.

        Copies container metadata (see :meth:`_empty_like`); the cameras keep
        their parent.
        """
        subset = self._empty_like()
        for cid in cam_ids:
            cam = self.data.get(cid)
            if cam is not None:
                subset.data[cid] = cam
        return subset

    def match_by_name(self, other: "Cameras") -> list[tuple[str, "Camera", "Camera"]]:
        """Pair cameras in this container with same-named cameras in ``other``.

        Cameras are matched on their image filename stem (case-insensitive,
        extension ignored), irrespective of folder or camera group, so e.g. AUV
        images included in a focal-plot model pair with the same images in the
        AUV model. Cameras without a pose (placeholder from the meta JSON) are
        ignored, and names occurring more than once in either container are
        skipped as ambiguous.

        Args:
            other: Cameras container to match against.

        Returns:
            Sorted list of ``(name, self_camera, other_camera)`` tuples.
        """

        def _index(cams: "Cameras", label: str) -> dict:
            by_name, dupes = {}, set()
            for cam in cams.data.values():
                if getattr(cam, "missing_pose_from_meta", False):
                    continue
                name = os.path.splitext(cam.filename)[0].lower()
                if name in by_name:
                    dupes.add(name)
                by_name[name] = cam
            if dupes:
                logger.warning(
                    "Skipping %d camera name(s) occurring more than once in %s: %s",
                    len(dupes),
                    label,
                    sorted(dupes),
                )
            return {k: v for k, v in by_name.items() if k not in dupes}

        own, theirs = _index(self, "source"), _index(other, "target")
        return [(name, own[name], theirs[name]) for name in sorted(own.keys() & theirs)]

    def reset_depth_sensor_m(self):
        """Reset the depth_sensor_m attribute for all cameras to None."""
        for cam in self.data.values():
            cam.depth_sensor_m = None

    def set_depth_sensor_m(self, depth_by_cam_id):
        """Set depth_sensor_m for cameras from a mapping of cam_id -> depth (meters)."""
        for cam_id, depth in depth_by_cam_id.items():
            cam = self.data.get(cam_id)
            if cam is not None:
                cam.depth_sensor_m = float(depth)

    @classmethod
    def from_image_dir(
        cls,
        image_dir: str,
        exts: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
        recursive: bool = False,
    ) -> "Cameras":
        """Build a Cameras container from a folder of photographs (no poses).

        Each image becomes a pose-less :class:`Camera` (``camera_transform`` and
        ``coords`` are None, ``missing_pose_from_meta`` is True), keyed by its
        filename stem. Useful for 2D-only workflows, e.g. classifying random
        points on photos with :meth:`get_random_image_points`.

        Args:
            image_dir: Directory containing the images.
            exts: Image file extensions to include (case-insensitive).
            recursive: Whether to include images in subdirectories.

        Returns:
            Cameras: Container with one camera per image, sorted by filepath.
        """
        exts = tuple(ext.lower() for ext in exts)
        filepaths = []
        if recursive:
            for root, _dirs, files in os.walk(image_dir):
                filepaths.extend(os.path.join(root, f) for f in files)
        else:
            filepaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        filepaths = sorted(
            os.path.abspath(f)
            for f in filepaths
            if os.path.isfile(f) and f.lower().endswith(exts)
        )

        cams = cls()
        for filepath in filepaths:
            cam_id = os.path.splitext(os.path.relpath(filepath, image_dir))[0]
            cams.data[cam_id] = Camera(cams, cam_id, path=filepath)
            cams.data[cam_id].missing_pose_from_meta = True
        if not cams.data:
            logger.warning("No images %s found in %s", exts, image_dir)
        return cams

    @classmethod
    def from_annotation_images(cls, annotations) -> "Cameras":
        """Build pose-less cameras for the images that annotations were placed on.

        The counterpart of :meth:`get_random_image_points` for reloaded points:
        each annotation with ``cam_filepath``/``cam_x``/``cam_y`` (e.g. from a
        saved points CSV) gets an :class:`ImageMatch` at that pixel, on a
        pose-less :class:`Camera` per image. Annotations without these fields
        are left without an image match.

        Args:
            annotations: Annotations whose image positions to attach.

        Returns:
            Cameras: One camera per referenced image, keyed by its filepath.
        """
        cams = cls()
        for ann in annotations:
            if not ann.cam_filepath or ann.cam_x is None or ann.cam_y is None:
                continue
            cam = cams.data.get(ann.cam_filepath)
            if cam is None:
                cam = Camera(cams, ann.cam_filepath, path=ann.cam_filepath)
                cam.missing_pose_from_meta = True
                cams.data[ann.cam_filepath] = cam
            ann.image_match = ImageMatch(cam, ann.cam_x, ann.cam_y, None, None, ann)
            ann.image_matches = [ann.image_match]
        return cams

    def get_random_image_points(
        self,
        n_points_per_image: int,
        margin: int = 0,
        stratified: bool = False,
        seed: int | None = None,
    ):
        """Sample random 2D points on every camera's image.

        See :meth:`Camera.get_random_image_points`. Each camera gets its own
        seed derived from ``seed``, so the result is reproducible.

        Args:
            n_points_per_image: Number of points per image.
            margin: Minimum distance (pixels) of points from the image edges,
                e.g. ``crop_size // 2`` so that crops stay within the image.
            stratified: Use stratified random sampling (one point per grid cell).
            seed: Optional random seed.

        Returns:
            Annotations: All sampled points, each with an ``image_match``.
        """
        # Via sys.modules: the package attribute is shadowed by `__future__`
        annotations = importlib.import_module("substrata.annotations")

        image_points = annotations.Annotations()
        cams = list(self.data.values())
        cam_seeds = np.random.SeedSequence(seed).spawn(len(cams))
        for cam, cam_seed in zip(tqdm(cams), cam_seeds):
            for ann in cam.get_random_image_points(
                n_points_per_image, margin=margin, stratified=stratified,
                seed=cam_seed,
            ):
                image_points.append(ann)

        print(
            f"Image points returned: {len(image_points)} "
            f"({n_points_per_image} per image, {len(cams)} images)"
        )
        return image_points

    def get_cams_from_file(self, cams_meta_filepath):
        """Load cameras from a JSON file and store them in the container.

        Entries with a valid ``path`` but null ``center``/``transform`` (e.g. disabled
        or unaligned in Metashape) are still loaded using a placeholder pose so they
        can receive poses later (e.g. ``camsync``). Such cameras have
        ``missing_pose_from_meta`` set to True.

        Args:
            cams_meta_filepath (str): Path to the JSON file with camera metadata.
        """
        invalidate_projection_cache()  # (cameras (re)loaded)
        with open(cams_meta_filepath, "r") as f:
            data = json.load(f)
        n_placeholder = 0
        for cam_id, cam_data in data["cameras"].items():
            if cam_data.get("path") is None or cam_data.get("path") == "null":
                continue
            center = cam_data.get("center")
            transform = cam_data.get("transform")
            pose_ok = (
                center is not None
                and center != "null"
                and transform is not None
                and transform != "null"
            )
            if pose_ok:
                self.data[cam_id] = Camera(
                    self,
                    cam_id,
                    cam_data["transform"],
                    cam_data["center"],
                    cam_data["path"],
                )
                if "reference" in cam_data:
                    self.data[cam_id].reference = cam_data["reference"]
                    # Ensure depth is negative (in meters); none without a reference z
                    ref_z = _reference_z(cam_data["reference"])
                    self.data[cam_id].depth_sensor_m = (
                        None if ref_z is None else -abs(ref_z)
                    )
                if "reference_accuracy" in cam_data:
                    self.data[cam_id].reference_acc = cam_data["reference_accuracy"]
                    self.data[cam_id].depth_acc = _reference_z(
                        cam_data["reference_accuracy"]
                    )
                if "center_crs" in cam_data:
                    self.data[cam_id].center_crs = cam_data["center_crs"]
                if "enabled" in cam_data:
                    self.data[cam_id].enabled = bool(cam_data["enabled"])
                if cam_data.get("pose_status"):
                    self.data[cam_id].pose_status = str(cam_data["pose_status"])
            else:
                self.data[cam_id] = Camera(
                    self,
                    cam_id,
                    np.eye(4).tolist(),
                    [0.0, 0.0, 0.0],
                    cam_data["path"],
                )
                self.data[cam_id].missing_pose_from_meta = True
                if "enabled" in cam_data:
                    self.data[cam_id].enabled = bool(cam_data["enabled"])
                n_placeholder += 1
        if n_placeholder:
            logger.info(
                "Loaded %s cameras with placeholder pose (null center/transform in meta)",
                n_placeholder,
            )

    def save(
        self,
        cams_meta_filepath: str | None = None,
        cams_xml_filepath: str | None = None,
    ) -> None:
        """Persist camera poses to the meta JSON file.

        Merges ``center`` and ``transform`` for each loaded camera from in-memory
        state (preferring ``orig_coords`` / ``orig_camera_transform`` when set).
        If a camera has an ``enabled`` attribute, it is written to JSON as well.
        Other keys for each camera entry are preserved from the existing file.

        Args:
            cams_meta_filepath: Output path. Defaults to :attr:`cams_meta_filepath`.
            cams_xml_filepath: Optional path to ``.cams.xml``. If provided (or set on
                the container) and the file exists, ``<camera enabled="...">`` is
                updated from each in-memory :attr:`Camera.enabled` (poses are not
                written to XML here).

        Raises:
            ValueError: If no output path is known or the file has no ``cameras`` key.
        """
        xml_path = cams_xml_filepath or getattr(self, "cams_xml_filepath", None)
        out_path = cams_meta_filepath or getattr(self, "cams_meta_filepath", None)
        if not out_path:
            raise ValueError("No cams_meta_filepath; pass cams_meta_filepath=...")

        with open(out_path, "r") as f:
            data = json.load(f)
        if "cameras" not in data:
            raise ValueError(f"Invalid cameras meta JSON (no 'cameras' key): {out_path}")

        cameras_out = copy.deepcopy(data["cameras"])
        for cam_id, cam in self.data.items():
            if cam_id not in cameras_out:
                logger.warning(
                    "Camera %s in memory but not in meta JSON; skipping save for it",
                    cam_id,
                )
                continue
            if getattr(cam, "missing_pose_from_meta", False):
                # Still only the placeholder pose from load; keep the null pose (or write one: a pose taken
                # away, imagesync.clear_pose)
                if getattr(cam, "pose_cleared", False):
                    cameras_out[cam_id]["center"] = None
                    cameras_out[cam_id]["transform"] = None
                    cameras_out[cam_id].pop("center_crs", None)
                if hasattr(cam, "enabled"):
                    cameras_out[cam_id]["enabled"] = bool(cam.enabled)
                _put_pose_status(cameras_out[cam_id], cam)
                continue
            center = cam.orig_coords if cam.orig_coords is not None else cam.coords
            transform = (
                cam.orig_camera_transform
                if cam.orig_camera_transform is not None
                else cam.camera_transform
            )
            if center is None or transform is None:
                logger.warning("Skipping %s: missing center or transform", cam_id)
                continue
            center = np.asarray(center, dtype=float)
            was = cameras_out[cam_id].get("center")
            if was is None or was == "null" or not np.allclose(np.asarray(was, dtype=float), center, rtol=0, atol=1e-9):
                # A new pose (camsync): the CRS centre exported from Metashape no longer belongs to it
                cameras_out[cam_id].pop("center_crs", None)
            cameras_out[cam_id]["center"] = center.tolist()
            cameras_out[cam_id]["transform"] = (
                np.asarray(transform, dtype=float).reshape(4, 4).tolist()
            )
            if hasattr(cam, "enabled"):
                cameras_out[cam_id]["enabled"] = bool(cam.enabled)
            _put_pose_status(cameras_out[cam_id], cam)

        data["cameras"] = cameras_out

        out_abs = _atomic_write_json(data, out_path)
        print(f"Saved camera metadata to {out_abs}")

        if xml_path and os.path.isfile(xml_path):
            _sync_camera_enabled_to_cams_xml(self, xml_path)

    def save_paths(
        self,
        cams_meta_filepath: str | None = None,
        only_cam_ids: set | None = None,
    ) -> int:
        """Persist camera image filepaths to the meta JSON file.

        Merges ``path`` for each loaded camera from :attr:`Camera.orig_filepath`
        into the existing file, leaving every other key (``center``,
        ``transform``, ``enabled``, ...) exactly as found on disk.

        Unlike :meth:`save`, cameras without a pose are written too. :meth:`save`
        skips them, since it exists to write poses; but a camera left unaligned
        in Metashape still has an image path, and is in fact among the most
        likely to have a stale one. The ``.cams.xml`` is never touched here: it
        stores image labels, not paths.

        Args:
            cams_meta_filepath: Output path. Defaults to
                :attr:`cams_meta_filepath`.
            only_cam_ids: Optional set of camera ids to write. ``None`` writes
                every loaded camera.

        Returns:
            int: Number of camera entries whose ``path`` value changed on disk.

        Raises:
            ValueError: If no output path is known or the file has no
                ``cameras`` key.
        """
        out_path = cams_meta_filepath or getattr(self, "cams_meta_filepath", None)
        if not out_path:
            raise ValueError("No cams_meta_filepath; pass cams_meta_filepath=...")

        with open(out_path, "r") as f:
            data = json.load(f)
        if "cameras" not in data:
            raise ValueError(
                f"Invalid cameras meta JSON (no 'cameras' key): {out_path}"
            )

        cameras_out = copy.deepcopy(data["cameras"])
        n_changed = 0
        for cam_id, cam in self.data.items():
            if only_cam_ids is not None and cam_id not in only_cam_ids:
                continue
            if cam_id not in cameras_out:
                logger.warning(
                    "Camera %s in memory but not in meta JSON; skipping save for it",
                    cam_id,
                )
                continue
            new_path = cam.orig_filepath
            if not new_path:
                continue
            if cameras_out[cam_id].get("path") != new_path:
                cameras_out[cam_id]["path"] = new_path
                n_changed += 1

        data["cameras"] = cameras_out
        out_abs = _atomic_write_json(data, out_path)
        print(f"Saved camera paths to {out_abs}")
        return n_changed

    def set_filepaths(self, path_by_cam_id: dict) -> None:
        """Set image filepaths for cameras from a mapping of cam_id -> filepath.

        Args:
            path_by_cam_id: Mapping of camera id to new image filepath.
        """
        for cam_id, path in path_by_cam_id.items():
            cam = self.data.get(cam_id)
            if cam is not None:
                cam.set_filepath(path)

    @_renamed_kwargs(use_orig="use_orig_filepath")
    def missing_image_paths(self, use_orig_filepath: bool = False) -> dict:
        """Cameras whose image file does not exist on disk.

        Zero-byte files are *not* reported as missing: locating an image and
        validating it are separate jobs.

        Args:
            use_orig_filepath: Check :attr:`Camera.orig_filepath` (the value persisted to
                the meta JSON) rather than the resolved :attr:`Camera.filepath`,
                which may have container-level find/replace rules applied.

        Returns:
            dict: Mapping of camera id to the filepath that was checked, for
            every camera whose image is absent or unset.
        """
        missing, listed = {}, {}
        for cam_id, cam in self.data.items():
            path = cam.orig_filepath if use_orig_filepath else cam.filepath
            if not path:
                missing[cam_id] = path
                continue
            folder, name = os.path.split(path)
            names = listed.get(folder)
            if names is None:
                names = listed[folder] = _file_names(folder)
            # (each folder listed once: one lookup per photo is slow where folders are not indexed, e.g.
            # exFAT, 63k photos ~90 s; a name not listed is checked by itself, e.g. another case)
            if name not in names and not os.path.isfile(path):
                missing[cam_id] = path
        return missing

    def get_cam_sensor_parameters_from_file(self, cams_xml_filepath):
        """Parse XML file and create Sensor objects, then assign to cameras.

        Args:
            cams_xml_filepath (str): Path to the XML file with sensor parameters.
        """
        tree = ET.parse(cams_xml_filepath)
        root = tree.getroot()

        # 1. Parse all sensors and create Sensor objects
        sensors_section = root.find(".//sensors")
        if sensors_section is None:
            raise ValueError(f"No sensors section found in {cams_xml_filepath}")

        for sensor_elem in sensors_section.findall("sensor"):
            sensor_id = int(sensor_elem.get("id"))
            label = sensor_elem.get("label", "unknown")

            # Parse resolution
            resolution_elem = sensor_elem.find("resolution")
            if resolution_elem is None:
                logger.warning(
                    f"No resolution found for sensor {sensor_id}, skipping..."
                )
                continue

            resolution = {
                "width": int(resolution_elem.get("width")),
                "height": int(resolution_elem.get("height")),
            }

            # Parse calibration
            calibration = sensor_elem.find(
                './/calibration[@type="frame"][@class="adjusted"]'
            )
            if calibration is not None:
                # Helper function to safely get float values
                def get_float_or_zero(elem, name, default=0.0):
                    found_elem = elem.find(name)
                    if found_elem is not None and found_elem.text is not None:
                        try:
                            return float(found_elem.text)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid {name} value for sensor {sensor_id}, using default {default}"
                            )
                            return default
                    else:
                        logger.warning(
                            f"Missing {name} for sensor {sensor_id}, using default {default}"
                        )
                        return default

                calib_params = {
                    "f": get_float_or_zero(calibration, "f"),
                    "cx": get_float_or_zero(calibration, "cx"),
                    "cy": get_float_or_zero(calibration, "cy"),
                    "k1": get_float_or_zero(calibration, "k1"),
                    "k2": get_float_or_zero(calibration, "k2"),
                    "k3": get_float_or_zero(calibration, "k3"),
                    "p1": get_float_or_zero(calibration, "p1"),
                    "p2": get_float_or_zero(calibration, "p2"),
                }
                for name in ("k4", "p3", "p4"):  # (optional: Metashape writes them only when fitted)
                    elem = calibration.find(name)
                    try:
                        calib_params[name] = float(elem.text) if elem is not None and elem.text else 0.0
                    except ValueError:
                        logger.warning(f"Invalid {name} value for sensor {sensor_id}, using default 0.0")
                        calib_params[name] = 0.0

                # Handle optional b1, b2 parameters
                b1_elem = calibration.find("b1")
                if b1_elem is not None and b1_elem.text is not None:
                    try:
                        calib_params["b1"] = float(b1_elem.text)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid b1 value for sensor {sensor_id}, using default 0.0"
                        )
                        calib_params["b1"] = 0.0
                else:
                    calib_params["b1"] = 0.0

                b2_elem = calibration.find("b2")
                if b2_elem is not None and b2_elem.text is not None:
                    try:
                        calib_params["b2"] = float(b2_elem.text)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid b2 value for sensor {sensor_id}, using default 0.0"
                        )
                        calib_params["b2"] = 0.0
                else:
                    calib_params["b2"] = 0.0

                # Create Sensor instance
                sensor = Sensor(sensor_id, label, resolution, calib_params)
                self.sensors[sensor_id] = sensor
            else:
                sensor = _nominal_sensor(sensor_elem, sensor_id, label, resolution)
                if sensor is not None:
                    self.sensors[sensor_id] = sensor
                    logger.info(
                        f"No calibration for sensor {sensor_id}: its nominal lens (f {sensor.f:.0f} px, "
                        "no distortion)"
                    )
                else:
                    logger.warning(
                        f"No calibration found for sensor {sensor_id}, skipping..."
                    )

        # 2. Parse cameras and assign sensor references
        cameras_section = root.find(".//cameras")
        if cameras_section is None:
            logger.warning("No cameras section found in XML file!")
            return

        assigned_count = 0
        xml_cam_ids = []
        json_cam_ids = list(self.data.keys())

        for camera_elem in cameras_section.findall(".//camera"):
            cam_id = camera_elem.get("id")
            sensor_id = int(camera_elem.get("sensor_id"))
            xml_cam_ids.append(cam_id)

            # Find matching camera in our data and assign sensor
            static_no_sensor_message_printed = getattr(
                self, "_no_sensor_message_printed", False
            )
            if cam_id in self.data:
                invalidate_projection_cache()  # (sensors changed: restack)
                self.data[cam_id].sensor_id = sensor_id
                self.data[cam_id].sensor = self.sensors.get(sensor_id)
                if self.data[cam_id].sensor is not None:
                    assigned_count += 1
                else:
                    if not getattr(self, "_no_sensor_message_printed", False):
                        logger.warning(
                            f"No sensor found for at least one camera (e.g. {cam_id} {sensor_id})"
                        )
                        self._no_sensor_message_printed = True
            else:
                pass  # Too many warnings for some projects - TODO
                # logger.warning(f"Camera {cam_id} from XML not found in loaded cameras")

        if not self.sensors:
            raise ValueError(f"No valid sensors found in {cams_xml_filepath}")

    def load_camera_attributes(self, input_filepath):
        """Load camera attributes from a CSV file.

        Reads the cache :meth:`save_camera_attributes` writes: each camera's
        ``datetime``, ``camdist`` (float) and depth (``depth_sensor_m``). The
        photo path in the file is not applied (the cameras' own paths, e.g.
        after a path repair, are current; the cache's may be older): a camera
        whose path differs is only counted in a log message.

        Args:
            input_filepath (str): Path to the CSV file with camera attributes.
        """
        cam_counter = 0
        not_found_counter = 0
        other_path = 0

        def number(value):
            return float(value) if value not in (None, "", "None", "nan") else None

        with open(input_filepath, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cam_id = row.get("cam_id", row.get("id"))
                if cam_id in self.data:
                    cam = self.data[cam_id]
                    path = row.get("path", row.get("label"))
                    if path and path_basename(path) != cam.filename:
                        other_path += 1
                    cam.datetime = row.get("datetime") or None
                    cam.camdist = number(row.get("camdist"))
                    cam.depth_sensor_m = number(row.get("depth"))
                    cam_counter += 1
                else:
                    not_found_counter += 1
        if other_path:
            logger.warning(
                f"{other_path} camera(s) in {input_filepath} name another photo "
                "than the camera's; their cached values were applied by camera id"
            )
        if not_found_counter > 0:
            logger.warning(
                f"File had {not_found_counter} cameras that were not found..."
            )
        if cam_counter == 0:
            logger.warning(f"No cameras found in file {input_filepath}")

    @_renamed_kwargs(output_filepath="output_path")
    def save_camera_attributes(self, output_path):
        """Save camera attributes to a CSV file.

        Writes camera ID, path, datetime, distance, depth, predicted depth,
        and depth residual for each camera.

        Args:
            output_path (str): Path to the output CSV file.
        """
        with _atomic_write(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "id",
                "orig_x",
                "orig_y",
                "orig_z",
                "label",
                "datetime",
                "camdist",
                "depth",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cam in self.data.values():
                writer.writerow(
                    {
                        "id": cam.cam_id,
                        "orig_x": cam.orig_coords[0],
                        "orig_y": cam.orig_coords[1],
                        "orig_z": cam.orig_coords[2],
                        "label": cam.orig_filepath,
                        "datetime": getattr(cam, "datetime", None),
                        "camdist": getattr(cam, "camdist", None),
                        # Keep CSV column name 'depth' for compatibility,
                        # but read from Camera.depth_sensor_m
                        "depth": getattr(cam, "depth_sensor_m", None),
                    }
                )

    def get_timematches(self, other_cams):
        """Get the timematches between the cameras.

        Args:
            other_cams (Cameras): The other cameras to match against.
        """
        timematches = []
        for cam in self.data.values():
            other_cam = other_cams.get_camera_by_datetime(
                cam.datetime, getattr(cam, "datetime_subsec", None)
            )
            if other_cam is not None:
                timematches.append((cam, other_cam))
            else:
                raise ValueError(
                    f"No camera found for camera {cam.cam_id} with datetime {cam.datetime}"
                )
        return timematches

    def get_centers_and_transforms_based_on_timematch(
        self, other_cams, offset_xyz=None
    ):
        """Adopt centers and transforms from ``other_cams`` via exact datetime match.

        Cameras in ``self`` whose ``datetime`` has no exact string match in
        ``other_cams`` are left untouched, tagged with
        ``missing_pose_from_timematch = True``, and reported via a single
        ``logger.warning``. A ``ValueError`` is raised only if zero cameras
        matched (which almost always indicates a wrong ``--time-offset``).

        Args:
            other_cams: The other ``Cameras`` to match against.
            offset_xyz: ``[x, y, z]`` offset in the camera's local frame, applied
                to the matched pose before writing back to ``self``.

        Returns:
            tuple[list[str], list[str]]: ``(matched_cam_ids, unmatched_cam_ids)``.
        """
        n_matched = 0
        matched_ids: list[str] = []
        unmatched: list[tuple[str, str]] = []  # (cam_id, datetime)
        for cam in self.data.values():
            other_cam = other_cams.get_camera_by_datetime(
                cam.datetime, getattr(cam, "datetime_subsec", None)
            )
            if other_cam is not None:
                # (copies as float arrays: a pose read from JSON is a nested list,
                # and the two cameras must not share one array)
                other_T = np.array(other_cam.camera_transform, dtype=float).reshape(4, 4)
                other_C = np.array(other_cam.coords, dtype=float).reshape(3)
                if offset_xyz is not None:
                    # The offset is in the camera's frame: rotate it into the world
                    # (the pose's nearest proper rotation: any scale/shear removed)
                    R = orthonormal_rotation_from_camera_transform(other_T)
                    world_offset = R @ np.asarray(offset_xyz, dtype=float).reshape(3)
                    cam.coords = other_C + world_offset
                    cam.camera_transform = other_T
                    cam.camera_transform[:3, 3] += world_offset
                    cam.reverse_transform_coords(other_cam.parent.world_transform)
                else:
                    cam.coords = other_C
                    cam.camera_transform = other_T
                    cam.orig_coords = np.array(other_cam.orig_coords, dtype=float).reshape(3)
                    cam.orig_camera_transform = np.array(
                        other_cam.orig_camera_transform, dtype=float
                    ).reshape(4, 4)
                    invalidate_projection_cache()  # (orig pose changed: restack)
                if hasattr(cam, "missing_pose_from_meta"):
                    delattr(cam, "missing_pose_from_meta")
                if hasattr(cam, "missing_pose_from_timematch"):
                    delattr(cam, "missing_pose_from_timematch")
                matched_ids.append(str(cam.cam_id))
                n_matched += 1
            else:
                cam.missing_pose_from_timematch = True
                unmatched.append((str(cam.cam_id), str(cam.datetime)))

        unmatched_ids = [cid for cid, _ in unmatched]

        if unmatched:
            from substrata.firefish import get_time_diff_s

            other_dts = sorted(
                str(c.datetime)
                for c in other_cams.data.values()
                if getattr(c, "datetime", None) is not None
            )
            first_id, first_dt = unmatched[0]
            closest_info = ""
            if other_dts:
                try:
                    diffs = [
                        (abs(get_time_diff_s(d, first_dt)), d) for d in other_dts
                    ]
                    _, closest = min(diffs, key=lambda x: x[0])
                    signed = get_time_diff_s(closest, first_dt)
                    closest_info = (
                        f"\n  Closest other_cams datetime to first unmatched: "
                        f"{closest} (off by {signed:+.1f}s)."
                    )
                except Exception:
                    pass
            range_info = (
                f"\n  other_cams datetime range: [{other_dts[0]} .. {other_dts[-1]}] "
                f"(n={len(other_dts)})"
                if other_dts
                else "\n  other_cams have no datetimes set."
            )
            preview = ", ".join(f"{cid}@{dt}" for cid, dt in unmatched[:6])
            more = "" if len(unmatched) <= 6 else f", ... (+{len(unmatched) - 6} more)"
            unmatched_list = f"\n  Unmatched: {preview}{more}"
            summary = (
                f"Time-match: {len(unmatched)}/{len(self.data)} target cameras "
                f"had no exact datetime match in other_cams "
                f"(matched {n_matched}/{len(self.data)})."
                f"{range_info}"
                f"{closest_info}"
                f"{unmatched_list}"
                "\n  Matching is exact on DateTimeOriginal; these cameras may "
                "fall outside the other_cams time range (e.g. source video "
                "cut out early) or indicate a wrong --time-offset."
            )
            if n_matched == 0:
                raise ValueError(
                    "Time-match failed: 0 cameras matched. " + summary
                )
            logger.warning(
                "%s\n  Unmatched cameras were left untouched and tagged with "
                "missing_pose_from_timematch=True.",
                summary,
            )

        return matched_ids, unmatched_ids

    def set_filepath_replace(self, find_str, replace_str):
        """Set a find/replace pair for adjusting filepaths.

        Args:
            find_str (str): The string to search for in the filepaths.
            replace_str (str): The string to replace find_str with.
        """
        self.filepath_replace = [find_str, replace_str]

    def get_cam_dists(
        self,
        pcd,
        beam_angle: float,
        n_jobs: int = -1,
        backend: str = "threading",
        scale_factor: float = 1.0,
        distance_percentile: float | None = None,
        origin_offset=None,
    ):
        """Calculate camera distances to a point cloud.

        For each camera, compute the distance to the given point cloud based on
        the provided beam angle. If ``n_jobs != 1`` (default -1: all CPUs), this
        will parallelize the computation using joblib with a threading backend
        by default, which avoids pickling the non-serializable point cloud object.

        Args:
            pcd: A point cloud object.
            beam_angle (float): The beam half-angle (degrees) for distance calculation.
            n_jobs (int): Number of parallel workers. Use 1 for sequential; -1 for all CPUs.
            backend (str): joblib backend. Use "threading" to avoid pickling ``pcd``.
            scale_factor (float, optional): Value to scale the result distance by. Default is 1.0.
            distance_percentile (float, optional): Percentile (0-100) of the in-beam
                distances to use instead of the mean (e.g. 5 to mimic a sonar first
                return). Default None (mean).
            origin_offset (sequence of 3 floats, optional): Offset ``(x, y, z)`` of the
                beam origin from the camera centre in the camera frame (x: image right,
                y: image down, z: view direction), in metres (converted to point cloud
                units using ``scale_factor``). Default None (camera centre).
        """
        if origin_offset is not None:
            origin_offset = np.asarray(origin_offset, dtype=float) / scale_factor
        kwargs = dict(
            distance_percentile=distance_percentile, origin_offset=origin_offset
        )
        cams_list = list(self.data.values())
        if n_jobs in (None, 1):
            for cam in tqdm(cams_list, desc="Calculating camera distances..."):
                cam.camdist = pcd.get_cam_dist(cam, beam_angle, **kwargs) * scale_factor
        else:
            with tqdm_joblib(
                tqdm(total=len(cams_list), desc="Calculating camera distances...")
            ):
                dists = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(pcd.get_cam_dist)(cam, beam_angle, **kwargs)
                    for cam in cams_list
                )
            for cam, dist in zip(cams_list, dists):
                cam.camdist = dist * scale_factor

    def make_thumbnails(
        self,
        root: str,
        max_size=None,
        overwrite: bool = False,
        n_jobs: int = -1,
    ) -> dict:
        """Write a thumbnail for every camera (see :meth:`Camera.make_thumbnail`).

        Photos are decoded in parallel threads (JPEG decoding releases the GIL).

        Args:
            root: Thumbnails directory (typically ``<id>.thumbnails``).
            max_size: Bounding box ``(w, h)`` (or int); the aspect ratio is kept.
                Default: ``settings.THUMBNAIL_MAX_SIZE``.
            overwrite: Replace existing thumbnails.
            n_jobs: Number of threads (-1 = all cores, 1 = sequential).

        Returns:
            dict: Number of thumbnails per status (``generated``, ``skipped``,
            ``failed``).
        """
        cams_list = list(self.data.values())
        with tqdm_joblib(tqdm(total=len(cams_list), desc="Creating thumbnails")):
            statuses = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(cam.make_thumbnail)(root, max_size, overwrite)
                for cam in cams_list
            )
        counts = count_crop_statuses(statuses)
        counts.pop("rejected", None)
        return counts

    @_renamed_kwargs(offset_secs="offset_s")
    def get_datetime_originals(self, offset_s=None):
        """Read every camera's EXIF time into ``cam.datetime`` (``YYYY:MM:DD
        HH:MM:SS``, see :meth:`Camera.get_datetime_original`) and its
        sub-second part, when the photo has one, into ``cam.datetime_subsec``
        (seconds, else None; used to tell photos of the same second apart in
        :meth:`get_camera_by_datetime`)."""
        for cam in tqdm(
            self.data.values(), desc="Retrieving timestamps from camera files..."
        ):
            cam.datetime, cam.datetime_subsec = cam._read_datetime_original(offset_s)

    def earliest_exif_datetime(
        self, cam_ids: list[str] | None = None
    ) -> tuple[str | None, str | None]:
        """Earliest EXIF DateTimeOriginal among cameras (does not set ``cam.datetime``).

        Args:
            cam_ids: Camera ids to consider; default is all keys in ``self.data``.

        Returns:
            Tuple ``(earliest_datetime_str, cam_id)``, or ``(None, None)`` if none found.
        """
        from substrata import firefish

        ids = cam_ids if cam_ids is not None else list(self.data.keys())
        best_t: float | None = None
        best_dt: str | None = None
        best_id: str | None = None
        for cid in ids:
            cam = self.data.get(cid)
            if cam is None:
                continue
            dt = cam.get_datetime_original(None)
            if dt is None:
                continue
            t = firefish.get_unix_time(dt)
            if best_t is None or t < best_t:
                best_t = t
                best_dt = dt
                best_id = cid
        return (best_dt, best_id)

    def get_camera_by_filename(self, filename):
        """Get a camera object by its filename.

        Args:
            filename (str): The filename to search for.

        Returns:
            Camera: The matching Camera object, or None if not found.
        """
        for cam in self.data.values():
            if cam.filename == filename or cam.filename == filename + ".jpg":
                return cam
        return None

    def get_camera_by_filepath(self, filepath):
        """Get a camera object by its filepath.

        Args:
            filepath (str): The filepath to search for.
        """
        for cam in self.data.values():
            if cam.filepath == filepath:
                return cam
            elif cam.orig_filepath == filepath:
                return cam
        return None

    def get_camera_by_datetime(self, datetime, subsec=None):
        """The camera whose ``datetime`` equals ``datetime`` (to the second).

        When several cameras share that second, the one whose
        ``datetime_subsec`` is nearest ``subsec`` is taken if both are known;
        otherwise the first in order, with a warning (once per second of
        time).

        Args:
            datetime (str): The datetime to search for.
            subsec (float, optional): The sub-second part of the time searched.

        Returns:
            Camera or None.
        """
        if datetime is None:
            return None
        hits = self._cameras_by_datetime().get(str(datetime), [])
        if len(hits) <= 1:
            return hits[0] if hits else None
        known = [c for c in hits if getattr(c, "datetime_subsec", None) is not None]
        if subsec is not None and known:
            return min(known, key=lambda c: abs(c.datetime_subsec - subsec))
        warned = self.__dict__.setdefault("_ambiguous_times_warned", set())
        if datetime not in warned:
            warned.add(datetime)
            logger.warning(
                f"{len(hits)} cameras share the time {datetime} (no sub-second "
                f"time to tell them apart); taking {hits[0].cam_id}"
            )
        return hits[0]

    def _cameras_by_datetime(self) -> dict:
        """``{datetime string: [cameras]}``, in ``data`` order (rebuilt when a
        camera's ``datetime`` or the cameras change)."""
        stamp = tuple((id(c), getattr(c, "datetime", None)) for c in self.data.values())
        cached = getattr(self, "_datetime_index", None)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        index: dict = {}
        for cam in self.data.values():
            dt = getattr(cam, "datetime", None)
            if dt is not None:
                index.setdefault(str(dt), []).append(cam)
        self._datetime_index = (stamp, index)
        return index

    def get_time_delta_between_first_and_last_photo(self):
        """Calculate the time delta between the first and last photos.

        Returns:
            int: The difference in seconds between the first and last camera
                timestamps.
        """
        import substrata.firefish as firefish

        times = [
            firefish.get_unix_time(cam.datetime)
            for cam in self.data.values()
            if getattr(cam, "datetime", None) is not None
        ]
        if not times:
            raise ValueError("No camera has a datetime (run get_datetime_originals)")
        return int(max(times) - min(times))

    def get_up_vector_from_camera_depths(
        self,
        depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
        plot=False,
        sensor_offset=None,
        scale_factor=1.0,
    ):
        """Compute the up vector using least-squares regression on camera depths.

        Fits a linear regression between the camera 3D points and their sensor depths to
        find the dominant depth direction. Also stores predicted depths and errors.

        Args:
            plot (bool): If True, create a visualization of the regression fit.
            sensor_offset (sequence of 3 floats, optional): Position of the depth
                sensor relative to the camera centre in the camera frame (x: image
                right, y: image down, z: view direction), in metres. Depths are then
                regressed against the sensor positions instead of the camera centres
                (lever-arm correction). Default None (camera centres).
            scale_factor (float): Metres per coordinate unit, used to convert
                ``sensor_offset`` to coordinate units. Default 1.0.

        Returns:
            np.ndarray: The coefficient vector representing the up vector.
        """
        # Filter cameras to those with depth_sensor_m and coords, and if they have a depth accuracy threshold
        # ensure that it is below the threshold
        cams_filtered = [
            cam
            for cam in self.data.values()
            if hasattr(cam, "depth_sensor_m")
            and hasattr(cam, "coords")
            and cam.depth_sensor_m is not None
            and cam.coords is not None
            and (
                not hasattr(cam, "depth_acc")
                or cam.depth_acc <= depth_accuracy_threshold
            )
        ]
        print(f"Found {len(cams_filtered)} matching cameras/depths for regression")

        # Conduct regression on the filtered cameras
        cam_ids = [cam.cam_id for cam in cams_filtered]
        offset_units = sensor_offset_units(sensor_offset, scale_factor)
        points = np.array([cam.sensor_coords(offset_units) for cam in cams_filtered])
        depths = np.array([cam.depth_sensor_m for cam in cams_filtered])

        res = measurements.fit_depth_regression(points, depths)

        # 4) Plot the regression fit if requested
        if plot:
            from substrata import visualizations

            visualizations.plot_depth_regression(
                res.depths, res.depths_pred, errors=res
            )

        # Print summary statistics
        print(
            f"  Up vector: [{res.up_vector[0]}, {res.up_vector[1]}, {res.up_vector[2]}]"
        )
        print(f"  Depth offset: {res.depth_offset:.4f} m")
        print(f"  Depth per unit: {res.depth_per_unit:.4f} m")
        print(f"  Mean squared error: {res.mse:.4f} m²")
        print(f"  Root mean squared error: {res.rmse:.4f} m")
        print(f"  Mean absolute error: {res.mae:.4f} m")
        print(f"  R²: {res.r2:.4f}")
        print(f"  Number of matches: {len(cams_filtered)}")

        # 5) Return the *flipped-if-needed* up vector and error metrics, plus number of matches
        return (
            res.up_vector,
            res.depth_offset,
            res.depth_per_unit,
            res.mse,
            res.rmse,
            res.mae,
            res.r2,
            len(cams_filtered),
        )

    @property
    def depth_residuals(self):
        """Access depth residual analysis methods."""
        from substrata.measurements import DepthResidualAnalyzer

        return DepthResidualAnalyzer(self)

    def get_depths_and_estimated_depths(
        self, depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD
    ):
        """
        Get the sensor depths (.depth_sensor_m) and predicted depths (.depth_in_m)
        for the cameras.

        Args:
            depth_accuracy_threshold (float): The depth accuracy threshold.

        Returns:
            tuple: A tuple containing the depths and predicted depths.
        """
        return self.depth_residuals.get_depths_and_estimated_depths(
            depth_accuracy_threshold=depth_accuracy_threshold,
            use_accuracy_filter=True,
        )

    def get_depths_and_z_coords(
        self, depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD
    ):
        """
        Get the recorded camera depths (.depth_sensor_m) and current z-coordinates
        (.coords[2]) for the cameras.

        Args:
            depth_accuracy_threshold (float): The depth accuracy threshold.

        Returns:
            tuple: A tuple containing the depths and z-coordinates.
        """
        return self.depth_residuals.get_depths_and_z_coords(
            depth_accuracy_threshold=depth_accuracy_threshold,
            use_accuracy_filter=True,
        )

    def show_depth_vs_est_depth_residuals(self, width=15, height=5):
        """Show residuals between predicted depth_in_m and the original recorded camera depths.

        These residuals are calculated based on recorded camera depths and the predicted depths from the regression model.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
        """
        return self.depth_residuals.show_depth_vs_est_depth_residuals(
            width=width,
            height=height,
            depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
            use_accuracy_filter=True,
        )

    def show_z_vs_depth_residuals(self, width=15, height=5):
        """Show residuals between camera z-coordinates and the original recorded camera depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
        """
        return self.depth_residuals.show_z_vs_depth_residuals(
            width=width,
            height=height,
            depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
            use_accuracy_filter=True,
        )

    def save_depth_residuals_pdf(
        self, filepath=None, width=15, height=5, recalculate=True
    ):
        """Save camera depth residuals visualization as a PDF.

        These residuals are calculated based on recorded camera depths and the predicted depths from the regression model.

        Args:
            filepath (str, optional): Path to save the PDF file. If None, generates
                a default filename based on the cameras metadata filepath.
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            recalculate (bool): If True, recalculate the depth residuals (default: True)

        Returns:
            str: The filepath where the PDF was saved.
        """
        return self.depth_residuals.save_depth_residuals_pdf(
            filepath=filepath,
            width=width,
            height=height,
            depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
            use_accuracy_filter=True,
        )


def orthonormal_rotation_from_camera_transform(
    camera_transform: np.ndarray | list,
) -> np.ndarray:
    """Return a proper 3x3 rotation from a 4x4 camera transform (same logic as camsync)."""
    rotation_matrix = np.asarray(camera_transform, dtype=float).reshape(4, 4)[:3, :3]
    u, _, vt = np.linalg.svd(rotation_matrix)
    r_mat = u @ vt
    if np.linalg.det(r_mat) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        r_mat = u @ vt
    return r_mat


def spatial_nearest_time_offset_report(
    target_cams: Cameras,
    pose_cams: Cameras,
    *,
    spatial_max_dist_m: float,
    min_pairs: int,
    scale_factor: float | None = 1.0,
) -> dict:
    """Estimate EXIF time offset from nearest-neighbor pose cameras in 3D (same chunk).

    Assumes both subsets share one reconstruction frame. ``spatial_max_dist_m`` is
    compared to ``euclidean(center_T, center_P) * scale_factor`` (metric distance in
    meters if coords are model units and ``scale_factor`` converts to meters).

    Args:
        target_cams: Cameras to update (e.g. macro); must have ``cam.coords`` and EXIF
            times available (e.g. via ``get_datetime_originals()``).
        pose_cams: Pose source cameras (e.g. GoPro).
        spatial_max_dist_m: Max metric distance (m) for a pair to count as inlier.
        min_pairs: Minimum number of inlier pairs with valid times required for ``ok``.
        scale_factor: Multiplier from stored coords to meters (1.0 if already metric).

    Returns:
        Dict with ``ok``, ``median_k_sec``, ``pairs`` (per-target rows), ``stats``,
        ``n_inliers``, ``n_targets``, and optional ``reason`` if not ok.
    """
    from substrata import firefish

    sf = float(scale_factor) if scale_factor is not None else 1.0
    pose_list = list(pose_cams.data.values())
    n_targets = len(target_cams.data)
    if not pose_list or n_targets == 0:
        return {
            "ok": False,
            "median_k_sec": None,
            "pairs": [],
            "stats": {},
            "n_inliers": 0,
            "n_targets": n_targets,
            "reason": "empty pose or target subset",
        }

    centers_pose = np.array(
        [np.asarray(c.coords, dtype=float).ravel()[:3] for c in pose_list]
    )
    tree = cKDTree(centers_pose)
    pairs: list[dict] = []
    k_inliers: list[float] = []

    for tcam in target_cams.data.values():
        if getattr(tcam, "missing_pose_from_meta", False):
            dt_t = getattr(tcam, "datetime", None) or tcam.get_datetime_original(
                None
            )
            pairs.append(
                {
                    "target_id": tcam.cam_id,
                    "pose_id": None,
                    "dist_stored": None,
                    "dist_metric_m": None,
                    "inlier": False,
                    "dt_target": dt_t,
                    "dt_pose": None,
                    "k_sec": None,
                    "skip_reason": "missing_pose_in_meta",
                }
            )
            continue
        ct = np.asarray(tcam.coords, dtype=float).ravel()[:3]
        dist_stored, j = tree.query(ct, k=1)
        pcam = pose_list[int(j)]
        dist_metric_m = float(dist_stored) * sf
        inlier = dist_metric_m <= float(spatial_max_dist_m)

        dt_t = getattr(tcam, "datetime", None) or tcam.get_datetime_original(None)
        dt_p = getattr(pcam, "datetime", None) or pcam.get_datetime_original(None)
        k_sec: float | None = None
        if dt_t is not None and dt_p is not None and inlier:
            k_sec = float(
                firefish.get_unix_time(dt_p) - firefish.get_unix_time(dt_t)
            )
            k_inliers.append(k_sec)

        pairs.append(
            {
                "target_id": tcam.cam_id,
                "pose_id": pcam.cam_id,
                "dist_stored": float(dist_stored),
                "dist_metric_m": dist_metric_m,
                "inlier": inlier,
                "dt_target": dt_t,
                "dt_pose": dt_p,
                "k_sec": k_sec,
            }
        )

    stats: dict[str, float | None] = {}
    median_k: float | None = None
    if k_inliers:
        arr = np.array(k_inliers, dtype=float)
        median_k = float(np.median(arr))
        stats = {
            "median": median_k,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    n_inliers = len(k_inliers)
    ok = n_inliers >= int(min_pairs) and median_k is not None
    reason = None
    if not ok:
        reason = (
            f"need >= {min_pairs} inlier pairs with valid EXIF; got {n_inliers}"
        )

    return {
        "ok": ok,
        "median_k_sec": median_k,
        "pairs": pairs,
        "stats": stats,
        "n_inliers": n_inliers,
        "n_targets": n_targets,
        "reason": reason,
    }


def xyz_offset_datetime_matches_report(
    target_cams: Cameras,
    pose_cams: Cameras,
    *,
    scale_factor: float | None = 1.0,
) -> dict:
    """Estimate median pose-local ``[x,y,z]`` offset from time-matched camera pairs.

    For each target camera with ``cam.datetime``, finds the pose camera with the same
    datetime. Computes ``delta_world = (C_pose - C_target) * scale_factor`` (scale
    when coords are in model units), then ``offset_xyz = R^T @ delta_world`` using the
    orthonormalized rotation from the pose camera transform (matches
    ``get_centers_and_transforms_based_on_timematch``).

    Args:
        target_cams: Target subset (datetimes must match pose after time offset).
        pose_cams: Pose source subset.
        scale_factor: Multiply center difference by this (1.0 if coords already meters).

    Returns:
        Dict with ``ok``, ``median_xyz``, ``mean_xyz``, ``rows`` (one dict per target),
        and ``unmatched_ids`` if any target has no pose at same datetime.
    """
    sf = float(scale_factor) if scale_factor is not None else 1.0
    rows: list[dict] = []
    locals_list: list[np.ndarray] = []
    unmatched: list[str] = []

    for tcam in target_cams.data.values():
        pose = pose_cams.get_camera_by_datetime(
            tcam.datetime, getattr(tcam, "datetime_subsec", None)
        )
        if pose is None:
            unmatched.append(str(tcam.cam_id))
            rows.append(
                {
                    "target_id": tcam.cam_id,
                    "pose_id": None,
                    "error": "no pose camera with same datetime",
                }
            )
            continue
        if getattr(tcam, "missing_pose_from_meta", False):
            rows.append(
                {
                    "target_id": tcam.cam_id,
                    "pose_id": pose.cam_id,
                    "skip_reason": (
                        "missing_pose_in_meta (median xyz uses cameras with pose only; "
                        "this camera omitted)"
                    ),
                }
            )
            continue
        ct = np.asarray(tcam.coords, dtype=float).ravel()[:3]
        cp = np.asarray(pose.coords, dtype=float).ravel()[:3]
        delta_w = (cp - ct) * sf
        r_mat = orthonormal_rotation_from_camera_transform(pose.camera_transform)
        off = r_mat.T @ delta_w
        locals_list.append(off)
        rows.append(
            {
                "target_id": tcam.cam_id,
                "pose_id": pose.cam_id,
                "delta_world": delta_w.tolist(),
                "offset_xyz_local": off.tolist(),
            }
        )

    ok = len(unmatched) == 0
    median_xyz: list[float] | None = None
    mean_xyz: list[float] | None = None
    if locals_list:
        arr = np.stack(locals_list, axis=0)
        median_xyz = np.median(arr, axis=0).tolist()
        mean_xyz = np.mean(arr, axis=0).tolist()
    elif ok:
        median_xyz = [0.0, 0.0, 0.0]
        mean_xyz = [0.0, 0.0, 0.0]

    reason = None if ok else f"unmatched target ids: {unmatched}"

    return {
        "ok": ok,
        "median_xyz": median_xyz,
        "mean_xyz": mean_xyz,
        "rows": rows,
        "unmatched_ids": unmatched,
        "reason": reason,
    }


class Camera:
    """Class that holds information about a single camera."""

    def __init__(
        self, parent=None, cam_id=None, camera_transform=None, coords=None, path=None
    ):
        """Initialize a Camera instance.

        Args:
            parent (Cameras): The parent Cameras container.
            cam_id (str): Unique camera identifier.
            camera_transform (list or np.ndarray): The camera transform matrix.
            coords (list or np.ndarray): The camera center coordinates.
            path (str): Path to the camera image.
        """

        self.parent = parent
        self.cam_id = cam_id
        self.camera_transform = self.orig_camera_transform = camera_transform
        self.coords = self.orig_coords = coords
        self.depth_sensor_m = None  # Depth in meters as measured by a sensor
        self.orig_filepath = path
        self.filename = os.path.basename(path)
        self.sensor_id = None  # Will be set during XML parsing
        self.sensor = None  # Will reference the Sensor instance

    @property
    def vector(self):
        """Obtain the camera vector from the transform.

        Returns:
            np.ndarray: A normalized 3D vector.
        """
        transform_matrix = np.array(self.camera_transform).reshape((4, 4))
        camera_vector = transform_matrix[:3, 2]
        camera_vector /= np.linalg.norm(camera_vector)
        return camera_vector

    @_renamed_kwargs(orig="use_orig_coords")
    def sensor_coords(self, offset=None, use_orig_coords=False):
        """Position of a rig-mounted sensor, given its offset from this camera.

        Args:
            offset (sequence of 3 floats, optional): Sensor position relative to
                the camera centre in the camera frame (x: image right, y: image
                down, z: view direction), in coordinate units. None returns the
                camera centre.
            use_orig_coords (bool): Use the original (untransformed) coordinates and pose
                instead of the current ones.

        Returns:
            np.ndarray: Sensor coordinates.
        """
        coords = self.orig_coords if use_orig_coords else self.coords
        if offset is None or coords is None:
            return coords
        transform = self.orig_camera_transform if use_orig_coords else self.camera_transform
        return geom.offset_in_camera_frame(coords, transform, offset)

    @property
    def filepath(self):
        """Resolved image filepath for this camera.

        Returns the original filepath unless the parent container defines
        ``filepath_replace`` or ``filename_prefix`` rules, in which case the
        updated filepath is returned.

        Returns:
            str: The image filepath.
        """
        if (
            not hasattr(self.parent, "filepath_replace")
            or not self.parent.filepath_replace
        ) and (
            not hasattr(self.parent, "filename_prefix")
            or not self.parent.filename_prefix
        ):
            return self.orig_filepath
        else:
            return self._get_updated_filepath()

    def get_random_image_points(
        self,
        n_points: int,
        margin: int = 0,
        stratified: bool = False,
        seed=None,
        id_prefix: str | None = None,
    ):
        """Sample random 2D points on this camera's image.

        Requires only the image itself (no pose or point cloud). Each point is
        returned as an :class:`~substrata.annotations.Annotation` with NaN 3D
        coords and an :class:`ImageMatch` at the sampled pixel, so it can be
        classified with ``Annotations.classify_image_matches`` and saved as an
        annotations CSV (pixel positions in ``cam_filepath``/``cam_x``/``cam_y``).

        Args:
            n_points: Number of points to sample.
            margin: Minimum distance (pixels) of points from the image edges,
                e.g. ``crop_size // 2`` so that crops stay within the image.
            stratified: If True, use stratified random sampling: the image is
                divided into a near-square grid with at least ``n_points`` cells
                and one point is drawn in each of ``n_points`` random cells.
                Otherwise use simple random sampling.
            seed: Optional seed (int or ``np.random.SeedSequence``).
            id_prefix: Prefix for annotation IDs (defaults to the cam_id or the
                image filename stem); IDs are ``<prefix>_<index>``.

        Returns:
            Annotations: The sampled points.
        """
        # Via sys.modules: the package attribute is shadowed by `__future__`
        annotations = importlib.import_module("substrata.annotations")

        with Image.open(self.filepath) as img:
            img_w, img_h = img.size
        x_min, y_min = margin, margin
        x_max, y_max = img_w - margin, img_h - margin
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Margin {margin} too large for image {self.filename} "
                f"({img_w}x{img_h})"
            )

        rng = np.random.default_rng(seed)
        if stratified:
            # Near-square grid matching the sampling area's aspect ratio
            aspect = (x_max - x_min) / (y_max - y_min)
            n_cols = max(1, int(round(np.sqrt(n_points * aspect))))
            n_rows = int(np.ceil(n_points / n_cols))
            cells = rng.choice(n_rows * n_cols, size=n_points, replace=False)
            rows, cols = np.divmod(np.sort(cells), n_cols)
            cell_w = (x_max - x_min) / n_cols
            cell_h = (y_max - y_min) / n_rows
            xs = x_min + (cols + rng.random(n_points)) * cell_w
            ys = y_min + (rows + rng.random(n_points)) * cell_h
        else:
            xs = rng.uniform(x_min, x_max, n_points)
            ys = rng.uniform(y_min, y_max, n_points)

        if id_prefix is None:
            id_prefix = self.cam_id or os.path.splitext(self.filename)[0]
        image_points = annotations.Annotations()
        for idx, (x, y) in enumerate(zip(xs, ys)):
            ann = annotations.Annotation(
                [np.nan, np.nan, np.nan], id=f"{id_prefix}_{idx}"
            )
            ann.image_match = ImageMatch(self, float(x), float(y), None, None, ann)
            ann.image_matches = [ann.image_match]
            ann.cam_filepath = self.filepath
            ann.cam_x, ann.cam_y = float(x), float(y)
            image_points.append(ann)
        return image_points

    def set_filepath(self, path: str) -> None:
        """Set this camera's stored image filepath and refresh derived state.

        Assigns :attr:`orig_filepath` (the value persisted to the meta JSON) and
        recomputes :attr:`filename`, which is otherwise cached at construction
        and would go stale. Note that :attr:`group` is derived from the
        filepath, so group membership may change as a result of this call.

        Args:
            path: New image filepath.
        """
        self.orig_filepath = path
        self.filename = path_basename(path)
        # Invalidate the parent's cached filename prefixes (see Camera.group)
        if getattr(self, "parent", None) is not None:
            self.parent._prefix_cache = None

    @property
    def group(self):
        """Derive a grouping label from the camera's filepath.

        Rule:

        - Use the name of the immediate containing folder by default.
        - If that folder name contains a '.', use the text after the first '.'
          as the group name (e.g., "20241008.auv" -> "auv").
        - Further split the group by prefixing with the first part of the filename
          if there is an underscore in the filename (e.g., "PR_20250608.jpg" -> "PR").
        """
        try:
            folder = os.path.basename(os.path.dirname(self.filepath))
            if not folder:
                return None
            # Handle folder group extraction
            if "." in folder:
                parts = folder.split(".", 1)
                folder_group = parts[1] if len(parts) > 1 and parts[1] else folder
            else:
                folder_group = folder

            # Handle filename prefix extraction
            filename = self.filename
            filename_prefix = _filename_prefix(filename)

            # Determine if there are multiple unique filename prefixes in the parent
            parent = getattr(self, "parent", None)
            if parent is not None and hasattr(parent, "_filename_prefixes"):
                unique_prefixes = parent._filename_prefixes()
            else:
                # If no parent, fallback to just this camera's prefix
                unique_prefixes = {filename_prefix}

            if len(unique_prefixes) > 1 and "_" in filename:
                group = f"{folder_group}_{filename_prefix}"
            else:
                group = folder_group
            return group
        except Exception:
            return None

    def thumbnail_path(self, root: str) -> str:
        """Path of this camera's thumbnail: ``<root>/<photo folder>/<filename>``.

        Args:
            root: Thumbnails directory (typically ``<id>.thumbnails`` in the model
                folder).

        Returns:
            str: The thumbnail filepath (which may not exist yet).
        """
        folder = photo_folder_name(self.filepath)
        if folder is None:
            return os.path.join(root, self.filename)
        return os.path.join(root, folder, self.filename)

    def find_thumbnail(self, root: str) -> str | None:
        """Existing thumbnail for this camera, in the current or legacy layout.

        The legacy layout stored every thumbnail directly in ``root``.

        Args:
            root: Thumbnails directory.

        Returns:
            str | None: The thumbnail filepath, or None if there is none.
        """
        return _first_existing(
            self.thumbnail_path(root), os.path.join(root, self.filename)
        )

    def make_thumbnail(
        self,
        root: str,
        max_size=None,
        overwrite: bool = False,
    ) -> str:
        """Write a downscaled copy of this camera's photo to :meth:`thumbnail_path`.

        Args:
            root: Thumbnails directory.
            max_size: Bounding box ``(w, h)`` (or int); the aspect ratio is kept.
                Default: ``settings.THUMBNAIL_MAX_SIZE``.
            overwrite: Replace an existing thumbnail (a legacy-layout one is
                removed).

        Returns:
            str: ``"generated"``, ``"skipped"`` (already exists) or ``"failed"``.
        """
        existing = self.find_thumbnail(root)
        if existing and not overwrite:
            return "skipped"
        src = self.filepath
        if not src or not os.path.isfile(src):
            return "failed"
        if max_size is None:
            max_size = settings.THUMBNAIL_MAX_SIZE
        out_path = self.thumbnail_path(root)
        try:
            with Image.open(src) as img:
                img.thumbnail(_size_pair(max_size))
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                _save_image_atomic(img, out_path, quality=90)
        except (OSError, ValueError) as e:
            logger.warning("Could not create thumbnail for %s: %s", src, e)
            return "failed"
        if existing and os.path.abspath(existing) != os.path.abspath(out_path):
            os.remove(existing)
        return "generated"

    @property
    def depth_in_m(self) -> float:
        """Depth in meters from the 3D regression (orig frame).

        Uses the regression model stored on the parent:
            depth ≈ depth_offset + up_vector · orig_coords

        Where ``up_vector`` is the 3-D coefficient vector returned by the
        regression (not necessarily unit-length), and ``orig_coords`` are the
        camera centers in the original coordinate frame used for the fit.

        Returns:
            float: Depth value in meters, or None if inputs are missing.
        """
        if self.orig_coords is None:
            print(f"No coordinates found for camera {self.cam_id}")
            return None

        if (
            not hasattr(self.parent, "up_vector")
            or self.parent.up_vector is None
            or not hasattr(self.parent, "depth_offset")
            or self.parent.depth_offset is None
        ):
            print("Cameras 'up_vector' and/or depth_offset not set")
            return None

        # (a geom.Vector up_vector is homogeneous: use its xyz)
        up = self.parent.up_vector
        up = np.asarray(getattr(up, "xyz", up), dtype=float).ravel()[:3]
        return float(
            self.parent.depth_offset
            + float(np.dot(up, np.asarray(self.orig_coords, dtype=float).ravel()[:3]))
        )

    @_renamed_kwargs(transform_matrix="transform")
    def transform_coords(self, transform):
        """Apply a transformation to the camera coordinates and transform.

        Args:
            transform (np.ndarray): A 4x4 homogeneous transformation matrix.
        """
        if self.coords is not None:
            invalidate_projection_cache()  # (world pose changed: restack)
            self.coords = self.__transform_coords(self.coords, transform)
            self.camera_transform = np.dot(transform, self.camera_transform)

    def reverse_transform_coords(self, transform):
        """Restore original coordinates by applying the inverse
        transformation.

        Args:
            transform (np.ndarray): The transformation matrix to invert.
        """
        invalidate_projection_cache()  # (orig pose changed: restack)
        inverse_transform = np.linalg.inv(transform)
        self.orig_coords = self.__transform_coords(self.coords, inverse_transform)
        self.orig_camera_transform = np.dot(
            inverse_transform, self.camera_transform
        )  ### CHECK: changed this from camera_transform to orig_camera_transform

    def _projection_pose(self, use_orig_coords):
        """``(centre, 4x4 float pose)`` in the original or the current frame."""
        if use_orig_coords:
            cam_coords, cam_transform = self.orig_coords, self.orig_camera_transform
        else:
            cam_coords, cam_transform = self.coords, self.camera_transform
        return cam_coords, np.array(cam_transform, dtype=float).reshape((4, 4))

    def get_pixel_coords(
        self, coords, use_orig_coords=False, required_to_be_in_view=True
    ):
        """Compute the image pixel coordinates of one 3D point.

        Projects the point through the camera's current pose (inverted per call)
        and applies the lens distortion (Metashape's frame camera model, :func:`lens_distortion`): the projection core
        shared with :meth:`project_points` (many points) and
        :func:`pixel_hits` (many cameras), which give the same values.

        Args:
            coords (list or np.ndarray): The 3D point.
            use_orig_coords (bool): If True, use the original coordinates.
            required_to_be_in_view (bool): If True, only return the pixel coordinates if the point is in view.

        Returns:
            tuple: (x, y, depth, relevance metric), x and y int pixels
                   (:func:`int_pixel`: rounded half to even), or (None, None, None, None) if out of view
                   (behind the camera, beyond the image borders), and always for a camera that cannot
                   project (:func:`can_project`: no calibrated sensor or no pose).
        """
        if not can_project(self, use_orig_coords):
            return None, None, None, None
        cam_coords, cam_transform = self._projection_pose(use_orig_coords)
        coords = np.asarray(coords, dtype=float).ravel()[:3]
        cam_coords = np.asarray(cam_coords, dtype=float).ravel()[:3]  # (a list, as read from the meta JSON)
        proj_point = np.dot(np.linalg.inv(cam_transform), np.append(coords, 1))
        s = self.sensor
        x_img, y_img, in_view = camera_points_to_pixels(proj_point[0], proj_point[1], proj_point[2], s)
        if in_view or not required_to_be_in_view:
            dist_sq = np.sum((cam_coords - coords) ** 2)
            rm = _relevance(dist_sq, x_img, y_img, s.fx, s.fy, s.width, s.height)
            if in_view:
                x_img, y_img = int_pixel(x_img, s.width), int_pixel(y_img, s.height)
            else:
                x_img, y_img = np.rint(x_img), np.rint(y_img)
            return (int(x_img), int(y_img), float(proj_point[2]), rm)
        else:
            return None, None, None, None

    def project_points(self, coords, use_orig_coords=False):
        """Vectorized :meth:`get_pixel_coords` over many 3D points at once.

        Projects an ``(N, 3)`` array of 3D points through this camera in a single
        set of numpy operations (the pose is inverted once, not per point). This
        is the fast path used by point-cloud segmentation, where every query
        point must be tested against every camera; the per-point
        :meth:`get_pixel_coords` is far too slow at that scale. The projection
        core (Metashape's lens model, ``in_view`` test and ``relevance``
        metric) is shared with :meth:`get_pixel_coords` and :func:`pixel_hits` -
        see ``tests/test_projection_equivalence.py``.

        Args:
            coords: ``(N, 3)`` array (or a single ``(3,)`` point) of 3D points.
            use_orig_coords: If True, use the original (pre-world-transform)
                pose/centre; otherwise the current world-frame pose/centre.

        Returns:
            Tuple ``(x, y, depth, relevance, in_view)`` of length-``N`` arrays
            (``x``/``y`` are integer-valued float pixels, :func:`int_pixel` where in view,
            ``in_view`` is boolean). For a single input point, scalar values are
            returned. Unlike :meth:`get_pixel_coords`, out-of-view points are
            *not* dropped - the caller filters on ``in_view``. A camera that
            cannot project (:func:`can_project`) gives NaN for ``x``, ``y``,
            ``depth`` and ``relevance`` and ``in_view`` False.
        """
        pts = np.asarray(coords, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)  # (N, 3)
        if not can_project(self, use_orig_coords):
            nan = np.full(pts.shape[0], np.nan)
            if single:
                return np.nan, np.nan, np.nan, np.nan, False
            return nan, nan.copy(), nan.copy(), nan.copy(), np.zeros(pts.shape[0], bool)

        cam_coords, cam_transform = self._projection_pose(use_orig_coords)
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N, 4)
        proj = homog @ np.linalg.inv(cam_transform).T          # (N, 4)
        depth = proj[:, 2]
        s = self.sensor
        x_img, y_img, in_view = camera_points_to_pixels(proj[:, 0], proj[:, 1], depth, s)

        cam_center = np.asarray(cam_coords, dtype=float).ravel()[:3]
        dist_sq = np.sum((cam_center - pts) ** 2, axis=1)
        relevance = _relevance(dist_sq, x_img, y_img, s.fx, s.fy, s.width, s.height)

        x_px = np.where(in_view, int_pixel(x_img, s.width), np.rint(x_img))
        y_px = np.where(in_view, int_pixel(y_img, s.height), np.rint(y_img))
        if single:
            return (
                float(x_px[0]), float(y_px[0]), float(depth[0]),
                float(relevance[0]), bool(in_view[0]),
            )
        return x_px, y_px, depth, relevance, in_view

    def pixel_to_ray(self, x_img, y_img, use_optimization=False, iterations=20):
        """
        Compute the 3D ray (origin and direction) for a given image pixel.

        This function undistorts the pixel coordinate and converts it to a
        normalized 3D direction vector in the camera coordinate system. The ray
        is then transformed to world coordinates using self.transform, which
        defines the camera's orientation. The camera origin is given by
        self.center.

        Args:
            x_img (float): The x-coordinate in the image.
            y_img (float): The y-coordinate in the image.
            use_optimization (bool): If True, use an optimization routine for
                the undistortion step.
            iterations (int): Number of iterations if not using optimization.

        Returns:
            tuple: (origin, direction) where origin is a 3D point (the camera
            center) and direction is a normalized 3D vector in world coordinates.
        """
        # The lens distortion undone (the inverse of camera_points_to_pixels; use_optimization is no longer needed:
        # Newton's method converges in a few steps either way).
        x_norm, y_norm = (float(v) for v in pixels_to_normalized(x_img, y_img, self.sensor, iterations))

        # Create a direction vector in camera space. Here, we assume the
        # image plane is at unit distance (z = 1).
        vec_cam = np.array([x_norm, y_norm, 1.0])

        # Check for invalid values
        if np.any(np.isnan(vec_cam)) or np.any(np.isinf(vec_cam)):
            raise ValueError(
                f"Invalid camera vector computed for pixel ({x_img}, {y_img}): {vec_cam}"
            )

        vec_cam_norm = np.linalg.norm(vec_cam)
        if vec_cam_norm == 0:
            raise ValueError(f"Zero-length camera vector for pixel ({x_img}, {y_img})")

        vec_cam /= vec_cam_norm

        # Transform the direction vector from camera to world coordinates.
        # For directions, apply only the rotation part of self.transform.
        transform_matrix = np.array(self.camera_transform, dtype=float).reshape((4, 4))
        R = transform_matrix[:3, :3]
        vec_world = R.dot(vec_cam)

        vec_world_norm = np.linalg.norm(vec_world)
        if vec_world_norm == 0:
            raise ValueError(f"Zero-length world vector for pixel ({x_img}, {y_img})")

        vec_world /= vec_world_norm

        return vec_world

    def pixel_to_point(
        self, x_img, y_img, pcd, search_radius=0.001, reprojection_threshold=None
    ):
        """Back-project an image pixel onto the point cloud.

        Casts a ray from the camera centre through the pixel, finds where it
        intercepts the point cloud, and validates the result by re-projecting
        the 3D intercept back to image space.

        Args:
            x_img (float): Pixel x-coordinate in the image.
            y_img (float): Pixel y-coordinate in the image.
            pcd (PointCloud): Point cloud to intercept the ray against.
            search_radius (float): Search radius for the ray-cloud intercept, in
                the cloud's units (metres once the world transform is applied;
                the 0.001 default is 1 mm there, but 1 model unit before).
            reprojection_threshold (float, optional): Maximum allowed pixel
                distance between the input pixel and the re-projected 3D
                intercept. If exceeded, the match is rejected.

        Returns:
            tuple | None: ``(coords, [back_x, back_y], pixel_distance)`` where
            ``coords`` is the 3D intercept, ``[back_x, back_y]`` its re-projected
            pixel, and ``pixel_distance`` the reprojection error; ``None`` if the
            back-projection fails or exceeds ``reprojection_threshold``.
        """
        back_vector = self.pixel_to_ray(x_img, y_img)
        back_point = pcd.get_intercept(
            self.coords,
            vector=back_vector,
            search_radius=search_radius,
            always_return=True,
        )
        if back_point is None:
            logger.warning(
                f"Back-projection error for {self.cam_id} [{x_img}, {y_img}] failed ..."
            )
            return None

        back_x, back_y, *_ = self.get_pixel_coords(back_point.coords)
        if back_x is None:
            logger.warning(
                f"Back-projection error for {self.cam_id} [{x_img}, {y_img}] when querying 3D intercept {back_point.coords} ..."
            )
            return None

        pixel_distance = np.sqrt((back_x - x_img) ** 2 + (back_y - y_img) ** 2)
        if (
            reprojection_threshold is not None
            and pixel_distance > reprojection_threshold
        ):
            logger.warning(
                f"Back-projection error for {self.cam_id} [{x_img}, {y_img}] resulting in [{back_x}, {back_y}] with distance {pixel_distance:.2f} pixels ..."
            )
            return None

        return back_point.coords, [back_x, back_y], pixel_distance

    def calc_pixel_scale(self, annotations):
        """Calculate the pixel scale based on 3D annotation points.

        Computes distances between 3D annotation points and their corresponding
        2D projection distances.

        Args:
            annotations (Annotations): A collection of annotation points.

        Returns:
            np.ndarray: An array of computed pixel scales.
        """
        point_dists = annotations.get_eucl_distance_matrix()
        pixel_coords = [
            self.get_pixel_coords(ann.orig_coords, use_orig_coords=True)
            for ann in annotations
        ]
        # (a point out of view has no pixel: NaN, so its pairs drop out below)
        pixel_coords_2d = np.array(
            [(np.nan, np.nan) if p[0] is None else p[:2] for p in pixel_coords], dtype=float
        )
        pixel_dists = pd.DataFrame(
            squareform(pdist(pixel_coords_2d, metric="euclidean"))
        )
        pixel_scale_matrix = point_dists / pixel_dists
        pixels = pixel_scale_matrix.values[
            np.triu_indices_from(pixel_scale_matrix, k=1)
        ]
        pixels = pixels[~np.isnan(pixels)]
        return pixels

    def get_image_matches(
        self,
        anns,
        pcd=None,
        use_orig_coords=True,
        intercept_radius=settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        reprojection_threshold_uncertain=settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
        reprojection_threshold_discard=settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
    ):
        """The annotations this camera sees, as image matches.

        Args:
            anns: Annotations (any iterable of :class:`Annotation`).
            pcd: Optional point cloud for the reprojection occlusion test
                (:func:`keep_by_reprojection`, as
                :meth:`Annotation.get_image_matches` with ``pcd``).
            use_orig_coords: Project the annotations' ``orig_coords`` through
                the original pose (else ``coords`` through the current pose).
            intercept_radius: Search radius of the reprojection test.
            reprojection_threshold_uncertain: Error from which a match is
                flagged ``potentially_obstructed``.
            reprojection_threshold_discard: Error above which it is dropped.

        Returns:
            list[ImageMatch]: Sorted by relevance (unobstructed first with
            ``pcd``); empty if the camera sees none of them.
        """
        image_matches = []
        for ann in anns:
            coords = ann.orig_coords if use_orig_coords else ann.coords
            x, y, depth, relevance = self.get_pixel_coords(
                coords, use_orig_coords=use_orig_coords
            )
            if x is None:
                continue
            image_match = ImageMatch(self, x, y, depth, relevance, annotation=ann)
            if pcd is None or keep_by_reprojection(
                image_match, pcd, intercept_radius,
                reprojection_threshold_uncertain, reprojection_threshold_discard,
            ):
                image_matches.append(image_match)
        if pcd is not None:
            return sorted(
                image_matches, key=lambda m: (m.potentially_obstructed, m.relevance)
            )
        return sorted(image_matches, key=lambda m: m.relevance)

    def render(self, highlight_pixels=None, orient=False, square=False,
               highlight_radius: int = 50, highlight_outline_width: int = 10):
        """Return the camera image as a PIL Image with optional highlight markers.

        Args:
            highlight_pixels: A single ``[x, y]`` coordinate or an ``(N, 2)``
                array of pixel coordinates to highlight with red circles.
            orient: If True, rotate the image so that the world X-axis points
                right, making the image orientation roughly consistent with a
                top-down OrthoMap.  The rotation angle is derived from the
                camera_transform: the image x-axis (``T[:3, 0]``) and y-axis
                (``T[:3, 1]``) are used to project the world X direction
                ``[1, 0, 0]`` onto the image plane; ``atan2`` then gives the
                clockwise offset from image-right, which is corrected by a
                counter-clockwise PIL rotation.
            square: If True, crop the (possibly rotated) image to the largest
                possible square centred on the image centre.  When
                ``orient=True`` the square is the largest axis-aligned square
                that fits within the actual image content (no rotation
                whitespace), computed analytically from the original image
                dimensions and rotation angle.  A warning is logged if any
                highlight falls outside the resulting square.
            highlight_radius: Radius in pixels of the circle drawn at each
                highlight position (default 50).
            highlight_outline_width: Stroke width in pixels of the circle
                outline (default 10, matching the selected-mask contour
                thickness used in ``get_crop_img_from_masks``).

        Returns:
            PIL Image, or None if the file is missing or unreadable.
        """
        if not self.filepath or not os.path.isfile(self.filepath):
            logger.warning("Camera.render: file not found: %s", self.filepath)
            return None
        try:
            image = Image.open(self.filepath).convert("RGB")
        except OSError as e:
            logger.warning("Camera.render: could not open %s: %s", self.filepath, e)
            return None

        orig_W, orig_H = image.size

        # Parse highlight pixels into a float (N, 2) array for coordinate tracking.
        hp_draw = None
        is_single = False
        if highlight_pixels is not None:
            hp_draw = np.array(highlight_pixels, dtype=float)
            if hp_draw.ndim == 1:
                hp_draw = hp_draw[np.newaxis, :]
                is_single = True

        # ── Step 1: orientation rotation ─────────────────────────────────
        angle_deg = 0.0
        if orient and self.camera_transform is not None:
            try:
                T = np.array(self.camera_transform, dtype=float).reshape((4, 4))
                cam_right = T[:3, 0] / np.linalg.norm(T[:3, 0])
                cam_down = T[:3, 1] / np.linalg.norm(T[:3, 1])
                world_x = np.array([1.0, 0.0, 0.0])
                angle_deg = float(np.degrees(
                    np.arctan2(np.dot(world_x, cam_down), np.dot(world_x, cam_right))
                ))
            except Exception as e:
                logger.warning("Camera.render: orient failed: %s", e)

        if angle_deg != 0.0:
            image = image.rotate(angle_deg, expand=True)
            rot_W, rot_H = image.size
            # Transform highlight pixels from original to rotated canvas coordinates.
            if hp_draw is not None:
                a_rad = np.radians(angle_deg)
                cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
                dx = hp_draw[:, 0] - orig_W / 2.0
                dy = hp_draw[:, 1] - orig_H / 2.0
                hp_draw = np.column_stack([
                    dx * cos_a + dy * sin_a + rot_W / 2.0,
                    -dx * sin_a + dy * cos_a + rot_H / 2.0,
                ])
        else:
            rot_W, rot_H = orig_W, orig_H

        # ── Step 2: square crop ───────────────────────────────────────────
        crop_offset_x, crop_offset_y = 0, 0
        if square:
            if angle_deg != 0.0:
                # Largest axis-aligned square inscribed in the rotated content
                # (avoids whitespace corner pixels).  Derived from the constraint
                # that all four corners of the square must map back into the
                # original W×H image when un-rotated:
                #   s ≤ min(W, H) / (|cos a| + |sin a|)
                #   s ≤ max(W, H) / ||cos a| - |sin a||
                a_abs = abs(np.radians(angle_deg)) % (np.pi / 2)
                cos_a, sin_a = np.cos(a_abs), np.sin(a_abs)
                diff_cs = abs(cos_a - sin_a)
                s1 = min(orig_W, orig_H) / (cos_a + sin_a)
                s2 = max(orig_W, orig_H) / diff_cs if diff_cs > 1e-8 else float("inf")
                crop_size = int(min(s1, s2))
            else:
                crop_size = min(rot_W, rot_H)

            cx, cy = rot_W / 2.0, rot_H / 2.0

            half = crop_size // 2
            left = int(max(0, min(round(cx - half), rot_W - crop_size)))
            top = int(max(0, min(round(cy - half), rot_H - crop_size)))
            right = left + crop_size
            bottom = top + crop_size

            if hp_draw is not None:
                outside = (
                    (hp_draw[:, 0] < left) | (hp_draw[:, 0] >= right)
                    | (hp_draw[:, 1] < top) | (hp_draw[:, 1] >= bottom)
                )
                if outside.any():
                    logger.warning(
                        "Camera.render: %d highlight(s) fall outside the square crop",
                        int(outside.sum()),
                    )

            image = image.crop((left, top, right, bottom))
            crop_offset_x, crop_offset_y = left, top

        # ── Step 3: draw highlights ───────────────────────────────────────
        if hp_draw is not None:
            draw = ImageDraw.Draw(image)
            for pixel in hp_draw:
                x = int(round(pixel[0])) - crop_offset_x
                y = int(round(pixel[1])) - crop_offset_y
                draw.ellipse((x - highlight_radius, y - highlight_radius,
                              x + highlight_radius, y + highlight_radius),
                             fill=None, outline=(255, 0, 0),
                             width=highlight_outline_width)

        return image

    def show(self, highlight_pixels=None, orient=False, square=False):
        """Display the camera image with optional highlight markers."""
        img = self.render(highlight_pixels=highlight_pixels, orient=orient, square=square)
        if img is not None:
            import matplotlib.pyplot as plt

            plt.imshow(np.array(img))
            plt.axis("off")
            plt.show()

    @_renamed_kwargs(offset_secs="offset_s")
    def get_datetime_original(self, offset_s=None):
        """Retrieve the DateTimeOriginal from the image file EXIF data, with optional offset.

        Read with exifread (``EXIF DateTimeOriginal``, else ``Image DateTime``),
        else with PIL (DateTimeOriginal from the Exif IFD, else the IFD0
        DateTime), else from a ``..._YYYYMMDD_HHMMSS`` file name suffix (e.g.
        video frames extracted by ffmpeg). The value has whole seconds; the
        sub-second part (``SubSecTimeOriginal``) is read by
        :meth:`Cameras.get_datetime_originals` into ``datetime_subsec``.

        Args:
            offset_s (float or int, optional): If provided, apply this many seconds to the EXIF datetime.

        Returns:
            str or None: The (possibly offset) DateTimeOriginal value as a string
            (``YYYY:MM:DD HH:MM:SS``), else None.
        """
        return self._read_datetime_original(offset_s)[0]

    def _read_datetime_original(self, offset_s=None):
        """``(datetime string, sub-second fraction or None)`` of the photo (see
        :meth:`get_datetime_original`)."""
        if not os.path.isfile(self.filepath):
            logger.error(f"Image file not found: {self.filepath}")
            return None, None

        def shifted(value):
            if offset_s is None:
                return value
            dt = datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            return (dt + datetime.timedelta(seconds=offset_s)).strftime(
                "%Y:%m:%d %H:%M:%S"
            )

        # exifread first (also reads files PIL cannot)
        try:
            with open(self.filepath, "rb") as f:
                tags = exifread.process_file(f, details=False)
            for key, sub in (("EXIF DateTimeOriginal", "EXIF SubSecTimeOriginal"),
                             ("Image DateTime", "EXIF SubSecTime")):
                if key in tags:
                    return shifted(str(tags[key]).strip()), _subsec(tags.get(sub))
        except Exception as e:  # noqa: BLE001 - unreadable EXIF: try PIL
            logger.debug("exifread could not read %s: %s", self.filepath, e)

        # Fallback to PIL: DateTimeOriginal (0x9003) and SubSecTimeOriginal
        # (0x9291) live in the Exif IFD (0x8769), DateTime (0x0132) in IFD0
        try:
            with Image.open(self.filepath) as image:
                exif_data = image.getexif()
                exif_ifd = exif_data.get_ifd(0x8769) if exif_data else {}
                candidates = (
                    (exif_ifd.get(0x9003), exif_ifd.get(0x9291)),
                    (exif_data.get(0x0132) if exif_data else None, exif_ifd.get(0x9290)),
                )
                for value, sub in candidates:
                    if value:
                        return shifted(str(value).strip()), _subsec(sub)
        except (OSError, ValueError) as e:
            logger.debug("PIL could not read EXIF of %s: %s", self.filepath, e)

        # Final fallback: parse "..._YYYYMMDD_HHMMSS" suffix from the filename
        # (e.g. for ffmpeg-extracted video frames that carry no EXIF segment).
        match = re.search(
            r"_(\d{8})_(\d{6})(?=\.[^.]+$|$)", os.path.basename(self.filepath)
        )
        if match:
            try:
                dt = datetime.datetime.strptime(
                    match.group(1) + match.group(2), "%Y%m%d%H%M%S"
                )
                return shifted(dt.strftime("%Y:%m:%d %H:%M:%S")), None
            except ValueError:
                pass

        logger.error(f"No exif data for: {self.filepath}")
        return None, None

    def has_coords_datetime(self):
        """Whether the camera has both ``coords`` and ``datetime`` set.

        Returns:
            bool: True if both attributes exist and are not None.
        """
        return (
            hasattr(self, "coords")
            and self.coords is not None
            and hasattr(self, "datetime")
            and self.datetime is not None
        )

    def _get_updated_filepath(self):
        """Update the original filepath based on the parent replacement rules.

        Returns:
            str: The updated filepath.
        """
        updated = self.orig_filepath

        # Apply filepath_replace if set on parent
        if (
            hasattr(self.parent, "filepath_replace")
            and self.parent.filepath_replace
            and len(self.parent.filepath_replace) >= 2
        ):
            find_str, replace_str = (
                self.parent.filepath_replace[0],
                self.parent.filepath_replace[1],
            )
            if find_str:  # (an empty replace_str removes find_str)
                updated = updated.replace(find_str, replace_str)
            elif replace_str:
                updated = self.__replace_base_path(updated, replace_str)

        # Apply filename_prefix if set on parent
        filename_prefix = getattr(self.parent, "filename_prefix", None)
        if filename_prefix:
            dirname = os.path.dirname(updated)
            basename = os.path.basename(updated)
            updated = os.path.join(dirname, f"{filename_prefix}{basename}")

        return self.__reformat_filepath_according_to_os(updated)

    @staticmethod
    def __transform_coords(coords, transform):
        hom_coords = np.array([coords[0], coords[1], coords[2], 1], dtype=float)
        return np.array(np.dot(transform, hom_coords)[0:3], dtype=float)

    @staticmethod
    def __replace_base_path(orig_filepath, base_path):
        """Replace the base folder of a filepath.

        Args:
            orig_filepath (str): The original filepath.
            base_path (str): The new base path.

        Returns:
            str: The updated filepath.
        """
        match_string = base_path.rstrip("/").split("/")[-1]
        found = orig_filepath.find(match_string)
        if not match_string or found < 0:  # (the base folder's name is not in the path: unchanged)
            logger.warning(
                f"Base path {base_path!r}: folder {match_string!r} not in {orig_filepath}"
            )
            return orig_filepath
        replace_index = found + len(match_string)
        return base_path.rstrip("/") + orig_filepath[replace_index:]

    @staticmethod
    def __reformat_filepath_according_to_os(filepath):
        """Reformat the filepath to match the current OS.

        Returns:
            str: The reformatted filepath.
        """
        path_parts = re.split(r"[\\/]+", filepath)
        if len(path_parts[0]) == 2 and path_parts[0][1] == ":":
            path_parts[0] += "\\"
        else:
            path_parts[0] += "/"
        return os.path.join(*path_parts)


def keep_by_reprojection(
    image_match, pcd, intercept_radius, threshold_uncertain, threshold_discard
) -> bool:
    """The legacy (reprojection) occlusion test of one image match.

    The ray through the match's pixel is intersected with ``pcd``
    (:meth:`ImageMatch.get_reprojection_error`); the distance from that
    intercept to the annotation is the reprojection error. A match without an
    intercept, or with an error above ``threshold_discard``, is dropped
    (False); one with an error from ``threshold_uncertain`` on is kept with
    ``potentially_obstructed`` True, the others with it False. Shared by
    :meth:`Camera.get_image_matches` and :meth:`Annotation.get_image_matches`.

    Returns:
        bool: Whether to keep the match.
    """
    image_match.get_reprojection_error(pcd, intercept_radius)
    error = image_match.reprojection_error
    if error is None or error > threshold_discard:
        return False
    image_match.potentially_obstructed = bool(error >= threshold_uncertain)
    return True


class LocalMask:
    """Lightweight mask compatible with the segmentation.py Mask class.

    Must live at module level so that instances (held by annotations) remain
    picklable when sent to joblib worker processes.

    Args:
        mask_vals: Binary mask array (uint8, non-zero inside the mask).
        score: Confidence score of the mask.
        logits: Optional raw logits associated with the mask.
    """

    def __init__(self, mask_vals: np.ndarray, score: float = 1.0, logits=None):
        self.vals = mask_vals
        self.score = score
        self.logits = logits
        self.area_in_px = cv2.countNonZero(mask_vals)
        self.area_in_cm2 = None
        self.radius_m = None  # set for circular masks only


class ImageMatch:
    """Class that holds information about an image match."""

    def __init__(self, cam, x, y, depth, relevance, annotation=None):
        self.annotation = annotation
        self.cam = cam
        self.x = x
        self.y = y
        self.depth = depth
        self.relevance = relevance
        self.pixel_scale = None
        self.pixel_scales = None
        self.masks = None
        self.mask = None  # selected mask for measurements

    @property
    def filepath(self):
        """The photo's path: the camera's current one (after a path repair too)."""
        return self.cam.filepath

    @property
    def filename(self):
        """The photo's file name (the camera's)."""
        return self.cam.filename

    def set_image_mask_id(self, mask_id):
        """Select the mask to use for measurements by its index.

        Args:
            mask_id (int): Index into ``self.masks`` of the mask to select.
        """
        self.mask = self.masks[mask_id]

    def get_reprojection_error(self, pcd, intercept_radius):
        """Check if the image match is obstructed"""
        # Find intercept of camera->pixel vector with pointcloud
        vector = self.cam.pixel_to_ray(self.x, self.y)
        reprojection_intercept = pcd.get_intercept(
            self.cam.coords,
            vector=vector,
            search_radius=intercept_radius,
            always_return=True,
        )
        # Calculate reprojection error if intercept found
        if reprojection_intercept is None:
            # print("Warning: no neighboring point found")
            self.reprojection_error = None
            self.reprojection_coords = None
            return None
        else:
            self.reprojection_error = np.linalg.norm(
                self.annotation.coords - reprojection_intercept.coords
            )
            self.reprojection_coords = reprojection_intercept.coords
            return self.reprojection_error

    def calc_pixel_scale_from_crosshair(self, measure_dist=0.01):
        """
        Calculate the pixel scale from 3D crosshair points.

        Args:
            measure_dist (float, optional): Distance for crosshair offset.
        """
        crosshair_anns = self.annotation.get_crosshair_points(
            self.cam.vector, measure_dist
        )
        crosshair_anns.reverse_transform(self.cam.parent.world_transform)
        self.pixel_scales = self.cam.calc_pixel_scale(crosshair_anns)
        self.pixel_scale = np.mean(self.pixel_scales)

    @property
    def pixels_per_mm(self):
        """
        Returns the number of pixels per millimeter based on self.pixel_scale.

        Returns:
            float: Pixels per mm, or None if pixel_scale is not set.
        """
        if self.pixel_scale is None:
            self.calc_pixel_scale_from_crosshair()
        return 1.0 / (self.pixel_scale * 1000.0)

    def get_sam2_masks(self, sam_predictor):
        """
        Get the SAM2 masks for the annotation in the image.

        Args:
            sam_predictor: Predictor object for SAM2 segmentation.
        """
        from substrata import segmentation

        self.masks = segmentation.get_sam2_masks(
            self.filepath, self.x, self.y, sam_predictor
        )
        self.mask = self.masks[0]

    def classify(self, classifier, crop_size=None):
        """
        Classify the image match using a FastAI learner.

        Args:
            classifier: Loaded FastAI learner or path to a .pkl learner.
            crop_size: Optional int (square) or (width, height) tuple for center crop.

        Returns:
            dict: Classification result with label, confidence, probabilities, and pred_idx.
        """
        from substrata.classification import classify_image_match

        return classify_image_match(self, classifier, crop_size)

    def _crops_root(self, root: str | None) -> str:
        """Resolve the crops directory: ``root`` or the annotations' crops root."""
        if root is not None:
            return root
        parent = getattr(self.annotation, "parent", None)
        root = getattr(parent, "crops_root", None)
        if root is None:
            raise ValueError(
                "No crops directory known: pass root=, set Annotations.crops_root, "
                "or load the annotations from a file."
            )
        return root

    def crop_path(self, root: str | None = None) -> str:
        """Path of this match's crop:
        ``<root>/<ann_id>/<photo folder>/<photo stem>_<x>_<y>.jpg``
        (:func:`view_crop_name`).

        Args:
            root: Crops directory. Defaults to the ``crops_root`` of the
                annotation's parent :class:`~substrata.annotations.Annotations`.

        Returns:
            str: The crop filepath (which may not exist yet).
        """
        root = self._crops_root(root)
        folder = photo_folder_name(self.cam.filepath)
        parts = [root, str(self.annotation.id)]
        if folder is not None:
            parts.append(folder)
        return os.path.join(*parts, view_crop_name(self.cam.filepath, self.x, self.y))

    def find_crop(self, root: str | None = None) -> str | None:
        """Existing crop for this match, also under the older names without the
        pixel (:func:`find_view_crop`).

        Args:
            root: Crops directory (defaults as in :meth:`crop_path`).

        Returns:
            str | None: The crop filepath, or None if there is none.
        """
        root = self._crops_root(root)
        return find_view_crop(
            root, self.annotation.id, self.cam.filepath, self.x, self.y)

    def crop_metadata(self, root: str | None = None) -> dict | None:
        """Metadata stored in this match's crop (see :func:`read_crop_metadata`)."""
        path = self.find_crop(root)
        return read_crop_metadata(path) if path else None

    def is_clear_from_image_edge(self, crop_size, image_size=None) -> bool:
        """Whether a ``crop_size`` crop centred on this match fits in the image.

        Args:
            crop_size: Crop size ``(w, h)`` or int.
            image_size: Image ``(w, h)``; defaults to the camera sensor size.

        Returns:
            bool: True if the crop lies fully inside the image.
        """
        crop_w, crop_h = _size_pair(crop_size)
        if image_size is None:
            image_size = (self.cam.sensor.width, self.cam.sensor.height)
        img_w, img_h = image_size
        # (the crop covers [x - w/2, x + w/2) x [y - h/2, y + h/2) in continuous pixel coordinates)
        return (
            self.x - crop_w / 2 >= 0
            and self.x + crop_w / 2 <= img_w
            and self.y - crop_h / 2 >= 0
            and self.y + crop_h / 2 <= img_h
        )

    def save_crop(
        self,
        root: str | None = None,
        crop_size=None,
        overwrite: bool = False,
    ) -> str:
        """Save the crop centred on this match to :meth:`crop_path`.

        Crops that would extend past the image edge are rejected, so every
        stored crop is centred on the match. The source photo's EXIF is kept
        (minus its thumbnail), with ``potentially_obstructed`` stored in the
        UserComment. Crops are saved at ``settings.CROP_JPEG_QUALITY``.

        Args:
            root: Crops directory (defaults as in :meth:`crop_path`).
            crop_size: Crop size ``(w, h)`` or int. Default:
                ``settings.CROP_SIZE_DEFAULT``.
            overwrite: Replace an existing crop (a legacy-layout one is removed).

        Returns:
            str: ``"generated"``, ``"skipped"`` (already exists), ``"rejected"``
            (too close to the image edge) or ``"failed"`` (photo missing or
            unreadable).
        """
        return save_photo_crops([self], root, crop_size, overwrite)[0]

    def get_mask_surface_area(self, predictor=None):
        """
        Calculate the surface area of the mask in the image.

        Args:
            predictor: Optional predictor for SAM2.

        Returns:
            float: Surface area in cm^2.
        """
        if not self.pixel_scale:
            self.calc_pixel_scale_from_crosshair()
        if not self.masks and predictor:
            self.get_sam2_masks(predictor)
        if not self.pixel_scale or not self.masks:
            raise ValueError("Scale or masks not available for calculation")
        self.mask.area_in_cm2 = self.mask.area_in_px * (self.pixel_scale**2) * 10000
        return self.mask.area_in_cm2

    def create_circular_masks(self, radii):
        """Create circular masks of specified radii around the match point.

        Args:
            radii (list): List of radii in meters. Each entry creates a separate mask.

        Returns:
            None: Masks are stored in self.masks, with the first mask also
                assigned to self.mask.
        """
        if not self.pixel_scale:
            self.calc_pixel_scale_from_crosshair()

        if not self.pixel_scale:
            raise ValueError(
                "Pixel scale not available. Call calc_pixel_scale_from_crosshair() first."
            )

        if not radii or len(radii) == 0:
            raise ValueError("radii must be a non-empty list")

        img = _imread(self.filepath)
        if img is None:
            raise ValueError(f"Cannot load image: {self.filepath}")

        img_height, img_width = img.shape[:2]

        if not all(isinstance(r, (int, float)) for r in radii):
            raise ValueError("radii must be a list of scalar values (e.g. [0.1, 0.2])")

        mask_objects = []
        for idx, radius_m in enumerate(radii):
            radius_px = round(radius_m / self.pixel_scale)

            mask_vals = np.zeros((img_height, img_width), dtype=np.uint8)
            centre = (int(np.floor(self.x)), int(np.floor(self.y)))  # (the pixel holding the point)
            cv2.circle(mask_vals, centre, radius_px, 255, thickness=-1)

            mask_obj = LocalMask(mask_vals, score=1.0)
            mask_obj.radius_m = radius_m
            mask_obj.area_in_px = cv2.countNonZero(mask_vals)
            mask_obj.area_in_cm2 = mask_obj.area_in_px * (self.pixel_scale**2) * 10000

            mask_objects.append(mask_obj)

            print(f"Created circular mask {idx+1}: radius {radius_m:.4f} m")
            print(f"Center: ({self.x}, {self.y}), radius: {radius_px} px")
            print(
                f"Mask area: {mask_obj.area_in_px} pixels = {mask_obj.area_in_cm2/10000:.4f} m²"
            )

        self.masks = np.array(mask_objects)
        self.mask = mask_objects[0]

    def render(self, crop_w=1000, crop_h=1000, single_mask=False, orient=False,
               circular_image_mask=None, show_text=True, show_radius=False,
               show_point=True):
        """Return the cropped image match as a PIL Image.

        Args:
            crop_w: Crop width in pixels (default 1000).
            crop_h: Crop height in pixels (default 1000).
            single_mask: Whether to use only the primary mask when masks are present.
            orient: If True, rotate so that the world X-axis points right, using
                the camera_transform from :attr:`cam`.
            circular_image_mask: Radius in pixels of a circular mask centred on
                the annotation.  Pixels outside the circle are replaced with a
                grey background.  Provide this only when no circular mask has been
                set via :meth:`create_circular_masks`; when one is present its
                radius is used automatically.
            show_text: Overlay the surface area of the chosen SAM2 mask (when
                known).
            show_radius: Draw a line (in the annotation colour) with the length
                of the chosen mask's equivalent-circle radius.
            show_point: Draw the annotation point (default True).

        Returns:
            PIL Image.
        """
        # Detect a circular mask created by create_circular_masks (has radius_m).
        # Such masks use the fixed-center crop path + automatic grey-out rather
        # than the SAM2 contour-overlay path.
        is_circular = (
            getattr(self, "mask", None) is not None
            and hasattr(self.mask, "radius_m")
            and self.mask.radius_m is not None
        )

        angle_deg = 0.0
        if orient and getattr(self.cam, "camera_transform", None) is not None:
            try:
                T = np.array(self.cam.camera_transform, dtype=float).reshape((4, 4))
                cam_right = T[:3, 0] / np.linalg.norm(T[:3, 0])
                cam_down = T[:3, 1] / np.linalg.norm(T[:3, 1])
                world_x = np.array([1.0, 0.0, 0.0])
                angle_deg = float(np.degrees(
                    np.arctan2(np.dot(world_x, cam_down), np.dot(world_x, cam_right))
                ))
            except Exception as e:
                logger.warning("ImageMatch.render: orient failed: %s", e)

        if getattr(self, "masks", None) is not None and len(self.masks) > 0 and not is_circular:
            if angle_deg != 0.0:
                # Inflate the canvas so that after rotating in-place (expand=False)
                # the central crop_w × crop_h region has no black corners.
                # pad_ratio shrinks the mask within the inflated canvas so it still
                # occupies crop_w pixels — exactly matching the final crop size.
                a_abs = abs(np.radians(angle_deg)) % (np.pi / 2)
                cos_a, sin_a = np.cos(a_abs), np.sin(a_abs)
                inflate = int(max(crop_w, crop_h) * (cos_a + sin_a)) + 2
                rotate_pad = (cos_a + sin_a - 1.0) / 2.0
                from substrata import visualizations  # lazy: figures only

                arr = visualizations.get_crop_img_from_masks(
                    self, inflate, inflate, single_mask=single_mask,
                    pad_ratio=rotate_pad, show_text=show_text,
                    show_radius=show_radius, show_point=show_point,
                )
                pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                pil_img = pil_img.rotate(angle_deg, expand=False)
                iw, ih = pil_img.size
                left = (iw - crop_w) // 2
                top = (ih - crop_h) // 2
                pil_img = pil_img.crop((left, top, left + crop_w, top + crop_h))
            else:
                from substrata import visualizations  # lazy: figures only

                arr = visualizations.get_crop_img_from_masks(
                    self, crop_w, crop_h, single_mask=single_mask,
                    show_text=show_text, show_radius=show_radius,
                    show_point=show_point,
                )
                pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        else:
            pil_img = get_crop_img(
                self.cam.filepath, self.x, self.y, crop_w, crop_h
            )
            if not isinstance(pil_img, Image.Image):
                pil_img = Image.fromarray(pil_img)
            if angle_deg != 0.0:
                orig_W, orig_H = pil_img.size
                pil_img = pil_img.rotate(angle_deg, expand=True)
                pil_img = self._inscribed_square_crop(pil_img, orig_W, orig_H, angle_deg)
            if show_point:
                draw = ImageDraw.Draw(pil_img)
                cx, cy = pil_img.size[0] // 2, pil_img.size[1] // 2
                draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8],
                             fill=(255, 0, 0), outline=(0, 0, 0))

        # Apply grey-out circle: prefer the stored circular-mask radius, fall back
        # to the explicit circular_image_mask argument (pixels).
        grey_r: int | None = None
        if is_circular and self.pixel_scale is not None:
            grey_r = int(self.mask.radius_m / self.pixel_scale)
        elif circular_image_mask is not None:
            grey_r = int(circular_image_mask)
        if grey_r is not None:
            cx, cy = pil_img.size[0] // 2, pil_img.size[1] // 2
            alpha = Image.new("L", pil_img.size, 0)
            ImageDraw.Draw(alpha).ellipse(
                [cx - grey_r, cy - grey_r, cx + grey_r, cy + grey_r], fill=255
            )
            grey = Image.new("RGB", pil_img.size, (180, 180, 180))
            pil_img = Image.composite(pil_img, grey, alpha)

        return pil_img

    @staticmethod
    def _inscribed_square_crop(
        img: "Image.Image",
        orig_W: int,
        orig_H: int,
        angle_deg: float,
    ) -> "Image.Image":
        """Return the largest whitespace-free square crop of a rotated image."""
        a_abs = abs(np.radians(angle_deg)) % (np.pi / 2)
        cos_a, sin_a = np.cos(a_abs), np.sin(a_abs)
        diff_cs = abs(cos_a - sin_a)
        s = int(min(
            min(orig_W, orig_H) / (cos_a + sin_a),
            max(orig_W, orig_H) / max(diff_cs, 1e-8),
        ))
        rot_W, rot_H = img.size
        left = (rot_W - s) // 2
        top = (rot_H - s) // 2
        return img.crop((left, top, left + s, top + s))

    def show(
        self,
        crop_w=1000,
        crop_h=1000,
        single_mask=False,
        orient=False,
        show_text=True,
        show_radius=False,
        show_point=True,
    ):
        """Display the image match and its attributes.

        ``show_text`` / ``show_radius`` / ``show_point`` are passed to
        :meth:`render`.
        """
        print(f"Image match for camera {self.cam.cam_id} at {self.x}, {self.y}")
        print(f"Depth: {self.depth}, Relevance: {self.relevance}")
        if hasattr(self, "potentially_obstructed"):
            print(f"Obstructed: {self.potentially_obstructed}")
            print(f"Reprojection error: {self.reprojection_error}")
            print(f"Reprojection coords: {self.reprojection_coords}")
        else:
            print("No obstruction check performed")
        if self.pixel_scale:
            print(f"Pixel scale: {self.pixel_scale}")
            print(f"Pixels per mm: {self.pixels_per_mm}")
        if hasattr(self, "classification") and self.classification:
            cls = self.classification
            label = cls.get("label")
            conf = cls.get("confidence")
            print(f"Classification: {label}")
            if conf is not None:
                try:
                    print(f"Classification confidence: {float(conf):.3f}")
                except Exception:
                    print(f"Classification confidence: {conf}")
            probs = cls.get("probs")
            if isinstance(probs, dict) and len(probs) > 0:
                try:
                    top_items = sorted(
                        probs.items(), key=lambda kv: kv[1], reverse=True
                    )[:5]
                    print(
                        "Top probabilities: "
                        + ", ".join(
                            [
                                (
                                    f"{k}: {float(v):.3f}"
                                    if isinstance(v, (float, int))
                                    else f"{k}: {v}"
                                )
                                for k, v in top_items
                            ]
                        )
                    )
                except Exception:
                    pass
        img = self.render(
            crop_w=crop_w, crop_h=crop_h, single_mask=single_mask,
            orient=orient, show_text=show_text, show_radius=show_radius,
            show_point=show_point,
        )

        import matplotlib.pyplot as plt

        plt.imshow(np.array(img))
        plt.show()


def _reference_z(reference) -> float | None:
    """The z (depth) of a meta JSON ``reference`` / ``reference_accuracy`` entry, None when missing
    (a null value, or a shorter or null list)."""
    try:
        z = reference[2]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        return None if z is None or z == "null" else float(z)
    except (TypeError, ValueError):
        return None


def _subsec(value) -> float | None:
    """An EXIF SubSecTime* value (digits: the fraction's decimals, e.g. ``"07"`` = 0.07 s) as a fraction
    of a second; None if missing or not digits."""
    text = str(value).strip().rstrip("\x00 ") if value is not None else ""
    return float("0." + text) if text.isdigit() else None


def _filename_prefix(filename: str) -> str:
    """Prefix used for camera grouping: text before the first '_', else the stem."""
    if "_" in filename:
        return filename.split("_", 1)[0]
    return os.path.splitext(filename)[0]


def sensor_offset_units(sensor_offset, scale_factor=1.0):
    """Convert a camera-frame sensor offset in metres to coordinate units.

    Returns None when there is no (or an all-zero) offset.
    """
    if sensor_offset is None:
        return None
    offset = np.asarray(sensor_offset, dtype=float)
    if not np.any(offset):
        return None
    return offset / scale_factor


_sensor_offset_units = sensor_offset_units  # deprecated alias (renamed 2026-11-25)
