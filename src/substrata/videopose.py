"""The camera pose of video frames placed on a plot (substrata.videotrack): where the camera was and where it looked,
so that a 3D viewer can show the plot as the video does.

A frame placed on a plot is matched (LoFTR) against the survey photos round it; each match's survey photo pixel is
looked up in that photo's depth map (the cloud near it projected into it) for its 3D point. From these 2D-3D
correspondences one **lens model** per camera setting is fitted together with every frame's pose (robustly, in
pixels, from the middle of the frame outwards), then each frame's **pose** with that lens (PnP, then refined on its
inliers).

The lens model maps a pixel to a ray: ``x = (u - cx) / f``, ``y = (v - cy) / (f asp)``; the sides' stretch undone
``xs = x (1 + a1 x^2 + a2 x^4)`` (a GoPro's SuperView stretches its 4:3 sensor image more at the sides); a fisheye
``theta = r (1 + k1 r^2 + k2 r^4)`` with ``r = |(xs, y)|`` (the angle from the optical axis). It is valid up to
:func:`max_angle` (beyond, the polynomial folds back); outside, :func:`project` gives NaN. A plain lens fits with
a1 = a2 = k = 0 (an equidistant fisheye) or near it.

On SAV03 (2025; three HERO13 Black, 1080p SuperView, no stabilisation; 44 stop frames on 60 m): one lens for all three
cameras, 3.1 px median reprojection error (1920 x 1080), 3.2 px on frames held out of the calibration; the pose
fitted on one half of a frame put the other half's matches within 5.3 px (median); the two frames of a stop (the pole
standing still) 3.6 cm and 1.3 deg apart; cameras 1.6 m above the stop points (the pole), 6 deg from straight down;
the fitted field of view under water 80 x 63 deg (a flat port's refraction of the 147 deg diagonal in air predicts
~82 deg across).
"""

from __future__ import annotations

import json
import logging
import os
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PHOTO_RADIUS = 3.0  # m: survey photos round the frame's position matched for its correspondences ...
MAX_PHOTOS = 30  # ... at most this many (nearest first)
MIN_PHOTO_MATCHES = 25  # a photo's matches (consistent with one homography, loosely) to use them
HOMOGRAPHY_PX = 15.0  # px (survey photo at the match width): that loose consistency
DEPTH_REDUCE = 16  # the survey photos' depth maps at 1/16 of their size
DEPTH_BOX = 1.6  # m: the cloud within this (x, y) of a survey photo's camera projected into its depth map
MIN_FRAMES = 12  # frames with correspondences needed to calibrate a lens
CAL_POINTS = 400  # correspondences per frame in the joint fit (a random subset)
INLIER_PX = 6.0  # reprojection error (px) of an inlier for a frame's pose
MIN_INLIERS = 30  # a pose needs this many
LENS_TERMS = ("f", "asp", "cx", "cy", "a1", "a2", "k1", "k2")
STAGES = ((450, 4, 20.0), (700, 6, 12.0), (2000, 8, 8.0), (2000, 8, 6.0))  # (radius px, lens terms free, inlier px)


# --------------------------------------------------------------------------- the lens


def rays(p: Sequence[float], uv: np.ndarray) -> np.ndarray:
    """Unit rays (camera frame: x right, y down, z forward) of pixels ``uv`` (N x 2) through lens ``p``."""
    f, asp, cx, cy, a1, a2, k1, k2 = p
    uv = np.asarray(uv, float)
    x = (uv[:, 0] - cx) / f
    y = (uv[:, 1] - cy) / (f * asp)
    xs = x * (1 + a1 * x**2 + a2 * x**4)
    r = np.hypot(xs, y) + 1e-12
    th = r * (1 + k1 * r**2 + k2 * r**4)
    s = np.sin(th) / r
    return np.stack([s * xs, s * y, np.cos(th)], 1)


def project(p: Sequence[float], d: np.ndarray) -> np.ndarray:
    """Pixels of directions ``d`` (camera frame, N x 3) through lens ``p`` (the inverse of :func:`rays`); NaN for a
    direction the lens does not image (beyond :func:`max_angle`, or behind)."""
    f, asp, cx, cy, a1, a2, k1, k2 = p
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    th = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])
    r = th.copy()
    for _ in range(30):
        r = r - (r * (1 + k1 * r**2 + k2 * r**4) - th) / (1 + 3 * k1 * r**2 + 5 * k2 * r**4)
    xs, y = r * np.cos(phi), r * np.sin(phi)
    x = xs.copy()
    for _ in range(30):
        x = x - (x * (1 + a1 * x**2 + a2 * x**4) - xs) / (1 + 3 * a1 * x**2 + 5 * a2 * x**4)
    uv = np.stack([cx + f * x, cy + f * asp * y], 1)
    back = rays(p, uv)
    bad = ~((back * d).sum(1) > np.cos(np.radians(0.2))) | ~np.isfinite(uv).all(1)  # (no way back: folded)
    bad |= (r < 0) | (r > _fold(k1, k2)) | (np.abs(x) > _fold(a1, a2))  # (past where a polynomial turns back)
    uv[bad] = np.nan
    return uv


def _fold(c1: float, c2: float) -> float:
    """Where t (1 + c1 t^2 + c2 t^4) stops increasing (its derivative's first zero), inf if it does not (to 3)."""
    t = np.linspace(0, 3, 3001)
    z = np.nonzero(1 + 3 * c1 * t**2 + 5 * c2 * t**4 <= 0)[0]
    return float(t[z[0]]) if len(z) else np.inf


def max_angle(p: Sequence[float], size: Tuple[int, int]) -> float:
    """The angle (deg) from the optical axis up to which the lens is valid: the frame's corners, or where the
    fisheye polynomial turns back, whichever is less."""
    w, h = size
    corner = rays(p, np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]]))
    a = float(np.degrees(np.arccos(np.clip(corner[:, 2], -1, 1))).max())
    k1, k2 = p[6], p[7]
    rt = _fold(k1, k2)
    if np.isfinite(rt):
        a = min(a, float(np.degrees(rt * (1 + k1 * rt**2 + k2 * rt**4))))
    return a


def fov(p: Sequence[float], size: Tuple[int, int]) -> Dict[str, float]:
    """The field of view (deg): ``across`` / ``down`` / ``diagonal`` (the frame's edges and corners) and ``middle_v``
    (the vertical field of view of a plain perspective view with the lens' focal length at the middle: what a 3D
    viewer shows to match the middle of the frame)."""
    w, h = size
    e = rays(p, np.array([[0, h / 2], [w, h / 2], [w / 2, 0], [w / 2, h], [0, 0], [w, h]], float))
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))  # noqa: E731
    return {"across": ang(e[0], e[1]), "down": ang(e[2], e[3]), "diagonal": ang(e[4], e[5]),
            "middle_v": float(np.degrees(2 * np.arctan(h / 2 / (p[0] * p[1]))))}


def video_lens(path: str) -> Dict[str, Any]:
    """The camera and lens setting a video was filmed with, from a GoPro's metadata (GPMF in the file's first or
    last 8 MB): ``model`` (e.g. "HERO13 Black"), ``lens`` (VFOV: W wide, S SuperView, L linear, ...), ``zfov``
    (diagonal field of view in air, deg). Empty for other videos."""
    out: Dict[str, Any] = {}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            chunks = [fh.read(8 << 20)]
            if size > 8 << 20:
                fh.seek(max(0, size - (8 << 20)))
                chunks.append(fh.read())
    except OSError:
        return out
    for d in chunks:
        for key, name in ((b"DVNM", "model"), (b"VFOV", "lens"), (b"ZFOV", "zfov")):
            i = d.find(key)
            if i < 0 or name in out or i + 8 > len(d):
                continue
            t, sz, n = chr(d[i + 4]), d[i + 5], struct.unpack(">H", d[i + 6:i + 8])[0]
            raw = d[i + 8:i + 8 + sz * n]
            try:
                out[name] = raw.decode().strip("\x00 ").strip() if t == "c" else round(struct.unpack(">f", raw[:4])[0], 1)
            except (UnicodeDecodeError, struct.error):
                pass
    return out


def lens_key(info: Dict[str, Any], lens: Dict[str, Any]) -> str:
    """One lens calibration per camera model, lens setting and frame size (e.g. ``HERO13-Black_S_1920x1080``)."""
    parts = [str(lens.get("model") or "camera").replace(" ", "-"), str(lens.get("lens") or "lens"),
             f"{info.get('width', 0)}x{info.get('height', 0)}"]
    return "_".join(parts)


# --------------------------------------------------------------------------- correspondences


class Correspondences:
    """2D (frame px) - 3D (world) correspondences of frames placed on one plot (``matcher``: a
    :class:`substrata.videotrack.PlotMatcher`), the survey photos' depth maps kept."""

    def __init__(self, matcher):
        self.m = matcher
        self.P = np.asarray(matcher.pcd.points, float)
        self._depth: Dict[int, np.ndarray] = {}

    def depth_map(self, i: int) -> np.ndarray:
        """Survey photo i's depth map (1/DEPTH_REDUCE size): per pixel the index of the nearest cloud point, -1."""
        zb = self._depth.get(i)
        if zb is not None:
            return zb
        cam = self.m.photos[i]
        c = np.asarray(cam.coords, float)
        idx = np.nonzero((np.abs(self.P[:, 0] - c[0]) < DEPTH_BOX) & (np.abs(self.P[:, 1] - c[1]) < DEPTH_BOX))[0]
        x, y, dep, _rel, iv = cam.project_points(self.P[idx])
        ok = iv & (dep > 0)
        idx, x, y, dep = idx[ok], x[ok], y[ok], dep[ok]
        w, h = cam.sensor.width // DEPTH_REDUCE + 1, cam.sensor.height // DEPTH_REDUCE + 1
        xi, yi = (x / DEPTH_REDUCE).astype(int), (y / DEPTH_REDUCE).astype(int)
        keep = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        idx, xi, yi, dep = idx[keep], xi[keep], yi[keep], dep[keep]
        order = np.argsort(-dep)  # (far first: the nearest is written last)
        zb = np.full((h, w), -1, np.int64)
        zb[yi[order], xi[order]] = idx[order]
        if len(self._depth) > 400:
            self._depth.clear()
        self._depth[i] = zb
        return zb

    def lookup(self, i: int, px: np.ndarray) -> np.ndarray:
        """Cloud point indices at survey photo i's pixels ``px`` (sensor px; -1: none within a pixel of the map)."""
        zb = self.depth_map(i)
        out = np.full(len(px), -1, np.int64)
        bi = (np.asarray(px, float) / DEPTH_REDUCE).astype(int)
        h, w = zb.shape
        for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            need = out < 0
            if not need.any():
                break
            out[need] = zb[np.clip(bi[need, 1] + dy, 0, h - 1), np.clip(bi[need, 0] + dx, 0, w - 1)]
        return out

    def frame(self, frame_path: str, xy: Sequence[float], photo: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
        """The correspondences of a frame placed at ``xy`` (world) on the plot, ``photo`` the survey photo it was
        placed with (its quarter turn is found again against it): {uv (N x 2, frame px), xyz (N x 3), photo (N),
        size (w, h)}; None when its photo gives too few matches."""
        import cv2

        from . import videotrack as vt

        m = self.m
        im = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        h0, w0 = im.shape
        fg = m._prep(im, vt.MATCH_FRAME_WIDTH)
        fh, fw = fg.shape
        pi = m._by_name.get(photo) if photo else None
        if pi is None:
            return None
        n0, k, _a, _b = m.verify(fg, m.photo_gray(pi, vt.MATCH_PHOTO_WIDTH), [0, 1, 2, 3])
        if n0 < vt.MIN_INLIERS:
            return None
        ft = np.ascontiguousarray(np.rot90(fg, k))
        d = np.linalg.norm(m.C[:, :2] - np.asarray(xy, float)[:2], axis=1)
        UV, XYZ, PH = [], [], []
        for i in [int(i) for i in np.argsort(d)[:MAX_PHOTOS] if d[i] < PHOTO_RADIUS]:
            pg = m.photo_gray(i, vt.MATCH_PHOTO_WIDTH)
            A, B = m._match(ft, pg)
            if len(A) < MIN_PHOTO_MATCHES:
                continue
            _H, mk = cv2.findHomography(np.float32(A), np.float32(B), cv2.USAC_MAGSAC, HOMOGRAPHY_PX,
                                        maxIters=5000, confidence=0.999)
            if mk is None or mk.sum() < MIN_PHOTO_MATCHES:
                continue
            sel = mk.ravel().astype(bool)
            Af = vt.unturn(A[sel], k, fw, fh) * (w0 / fw)
            ph, pw = (pg.shape[1], pg.shape[0]) if m.turn[i] % 2 else pg.shape
            cam = m.photos[i]
            Bf = vt.unturn(B[sel], m.turn[i], pw, ph) * (cam.sensor.width / pw)
            idx = self.lookup(i, Bf)
            ok = idx >= 0
            UV.append(Af[ok])
            XYZ.append(self.P[idx[ok]])
            PH.append(np.full(int(ok.sum()), i))
        if not UV:
            return None
        return {"uv": np.concatenate(UV), "xyz": np.concatenate(XYZ), "photo": np.concatenate(PH),
                "size": np.array([w0, h0])}


# --------------------------------------------------------------------------- poses and the calibration


def _rot(rv):
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rv).as_matrix()


def residuals(rv, C, p, uv, X) -> np.ndarray:
    """Reprojection errors (px, x and y) of 3D points X through pose (rotation vector world -> camera, centre)
    and lens p against pixels uv."""
    d = (np.asarray(X, float) - C) @ _rot(rv).T
    d = np.where(d[:, 2:3] > 1e-3, d, np.c_[d[:, :2], np.full(len(d), 1e-3)])
    uvp = project(p, d)
    uvp = np.where(np.isfinite(uvp), uvp, 1e4)  # (outside the lens: far off)
    return (uvp - uv).ravel()


def pixel_errors(p, uv, X, rv, C) -> np.ndarray:
    return np.linalg.norm(residuals(rv, C, p, uv, X).reshape(-1, 2), axis=1)


def pnp(p, uv, X, thr_px: float = 8.0):
    """A first pose: perspective PnP (RANSAC) on the lens' rays. (rotation vector, centre, inliers) or None."""
    import cv2

    r = rays(p, uv)
    ok = r[:, 2] > 0.25
    if ok.sum() < 12:
        return None
    nrm = (r[ok, :2] / r[ok, 2:]).astype(np.float64)
    good, rv, tv, inl = cv2.solvePnPRansac(np.asarray(X, np.float64)[ok], nrm, np.eye(3), None,
                                           reprojectionError=thr_px / p[0], iterationsCount=3000, confidence=0.999,
                                           flags=cv2.SOLVEPNP_EPNP)
    if not good or inl is None or len(inl) < 12:
        return None
    R, _ = cv2.Rodrigues(rv)
    return rv.ravel(), (-R.T @ tv).ravel(), len(inl)


def fit_pose(p, uv, X, init=None, scale: float = 3.0):
    """The pose (rotation vector, centre) minimising the reprojection errors (robust), from ``init`` or PnP."""
    from scipy.optimize import least_squares

    if init is None:
        init = pnp(p, uv, X)
        if init is None:
            return None
    sol = least_squares(lambda q: residuals(q[:3], q[3:], p, uv, X), np.r_[init[0], init[1]], loss="soft_l1",
                        f_scale=scale)
    return sol.x[:3], sol.x[3:]


def frame_pose(p, uv, X) -> Optional[Dict[str, Any]]:
    """A frame's pose with lens p: fitted, its inliers (< INLIER_PX) kept, fitted again. {rv (world -> camera
    rotation vector), C (centre), px (median error of the inliers), n (inliers), share} or None."""
    uv, X = np.asarray(uv, float), np.asarray(X, float)
    rr = np.hypot(uv[:, 0] - p[2], uv[:, 1] - p[3])
    ps = fit_pose(p, uv, X) or fit_pose(p, uv[rr < 450], X[rr < 450])
    if ps is None:
        return None
    e = pixel_errors(p, uv, X, *ps)
    inl = e < INLIER_PX
    if inl.sum() < MIN_INLIERS:
        return None
    ps = fit_pose(p, uv[inl], X[inl], init=ps)
    e = pixel_errors(p, uv, X, *ps)
    inl = e < INLIER_PX
    if inl.sum() < MIN_INLIERS:
        return None
    return {"rv": ps[0], "C": ps[1], "px": float(np.median(e[inl])), "n": int(inl.sum()), "share": float(inl.mean())}


def calibrate(frames: Sequence[Dict[str, np.ndarray]], size: Tuple[int, int], *, seed: int = 0,
              log=None) -> Optional[Dict[str, Any]]:
    """One lens for ``frames`` (correspondences, :meth:`Correspondences.frame`) of one camera setting: the focal
    length from a grid (perspective PnP on the frames' middles, most inliers), then the lens and all poses fitted
    together, in pixels, robustly, from the middle of the frame outwards (``STAGES``: the points within a radius,
    the lens terms freed in turn: f, asp, cx, cy; k1, a1; k2, a2). {params, size, frames, px (median error), fov,
    max_angle} or None (too few frames)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    log = log or (lambda s: None)
    w0, h0 = size
    frames = [f for f in frames if f is not None and len(f["uv"]) >= 60]
    if len(frames) < MIN_FRAMES:
        log(f"lens: {len(frames)} frames with correspondences (< {MIN_FRAMES}): not calibrated")
        return None
    best = None
    for f in np.arange(600, 2601, 100):
        p = np.array([f, 1.0, w0 / 2, h0 / 2, 0, 0, 0, 0], float)
        n = 0
        for fr in frames[::2]:
            cen = np.hypot(fr["uv"][:, 0] - w0 / 2, fr["uv"][:, 1] - h0 / 2) < 350
            got = pnp(p, fr["uv"][cen], fr["xyz"][cen], 6.0)
            n += got[2] if got else 0
        if best is None or n > best[1]:
            best = (float(f), n)
    p = np.array([best[0], 1.0, w0 / 2, h0 / 2, 0, 0, 0, 0], float)
    rng = np.random.default_rng(seed)
    order = [0, 1, 2, 3, 6, 4, 7, 5]  # f, asp, cx, cy, then k1, a1, then k2, a2
    for rmax, nfree, thr in STAGES:
        poses, subs = [], []
        for fr in frames:
            rr = np.hypot(fr["uv"][:, 0] - w0 / 2, fr["uv"][:, 1] - h0 / 2)
            ps = fit_pose(p, fr["uv"], fr["xyz"]) or fit_pose(p, fr["uv"][rr < 450], fr["xyz"][rr < 450])
            if ps is None:
                continue
            idx = np.nonzero((pixel_errors(p, fr["uv"], fr["xyz"], *ps) < thr) & (rr < rmax))[0]
            if len(idx) < 20:
                continue
            poses.append((fr, ps))
            subs.append(idx if len(idx) <= CAL_POINTS else rng.choice(idx, CAL_POINTS, replace=False))
        if len(poses) < MIN_FRAMES // 2:
            log(f"lens: only {len(poses)} frames posed: not calibrated")
            return None
        x0 = np.r_[p, np.concatenate([np.r_[ps[0], ps[1]] for _fr, ps in poses])]

        def res(x):
            q = x[:8]
            return np.concatenate([residuals(x[8 + 6 * j: 11 + 6 * j], x[11 + 6 * j: 14 + 6 * j], q,
                                             fr["uv"][s_], fr["xyz"][s_]) for j, ((fr, _ps), s_) in enumerate(zip(poses, subs))])

        nres = sum(2 * len(s_) for s_ in subs)
        J = lil_matrix((nres, len(x0)), dtype=int)
        row = 0
        for j, s_ in enumerate(subs):
            J[row:row + 2 * len(s_), :8] = 1
            J[row:row + 2 * len(s_), 8 + 6 * j: 14 + 6 * j] = 1
            row += 2 * len(s_)
        lo = np.r_[[400, 0.7, w0 / 2 - 250, h0 / 2 - 250, -1, -2, -1, -2], np.full(len(x0) - 8, -np.inf)]
        hi = np.r_[[3000, 1.4, w0 / 2 + 250, h0 / 2 + 250, 1, 2, 1, 2], np.full(len(x0) - 8, np.inf)]
        x0[:8] = np.clip(x0[:8], lo[:8] + 1e-6, hi[:8] - 1e-6)
        for j in order[nfree:]:  # (terms not free yet: held where they are)
            lo[j], hi[j] = x0[j] - 1e-6, x0[j] + 1e-6
        sol = least_squares(res, x0, loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=60, jac_sparsity=J,
                            bounds=(lo, hi))
        p = sol.x[:8]
        log(f"lens: stage r < {rmax} px, {nfree} terms, {len(poses)} frames: "
            + ", ".join(f"{k} {v:.3f}" for k, v in zip(LENS_TERMS, p)))
    errs = [fp["px"] for fp in (frame_pose(p, fr["uv"], fr["xyz"]) for fr in frames) if fp]
    return {"params": [float(v) for v in p], "terms": list(LENS_TERMS), "size": [int(w0), int(h0)],
            "frames": len(frames), "posed": len(errs), "px": float(np.median(errs)) if errs else None,
            "fov": fov(p, size), "max_angle": max_angle(p, size)}


def pose_record(fp: Dict[str, Any]) -> Dict[str, Any]:
    """A frame pose as stored: the camera centre (world) and its orientation as a quaternion (camera -> world,
    x y z w; the camera's x right, y down, z forward), the error (px) and inliers."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_rotvec(fp["rv"]).inv().as_quat()  # (world -> camera inverted)
    return {"C": [float(v) for v in fp["C"]], "q": [float(v) for v in q], "px": round(fp["px"], 2), "n": fp["n"]}


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def stop_pose(poses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A stop's pose from its frames' (the pole stood still): the one with most inliers."""
    poses = [p for p in poses if p]
    return max(poses, key=lambda p: p["n"]) if poses else None


def camera_axes(q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The camera's right, down and forward axes (world) of a stored quaternion (camera -> world)."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(q).as_matrix()
    return R[:, 0], R[:, 1], R[:, 2]


__all__ = ["rays", "project", "max_angle", "fov", "video_lens", "lens_key", "Correspondences", "calibrate",
           "frame_pose", "pose_record", "stop_pose", "camera_axes"]
