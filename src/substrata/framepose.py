"""Poses for video frames Metashape left unaligned, from image matches with posed photos of the same reef.

Made for the GoPro frames that ``camsync --match-images`` (:mod:`substrata.imagesync`) poses close-ups from: where
Metashape aligned none of the frames round a close-up's time (sa4_60m: 147 of its 478 frames, in stretches of up to
a minute), the close-up has nothing to be matched with. Such a frame is posed here much as a close-up is there:

1. **Where to look** (:func:`anchors`): the posed frames nearest in time (before and after) aim at points on the
   cloud; the frame saw the reef round (and between) those, from about where the nearer one looked from.
2. **Candidates** (:meth:`FramePoser.candidates`): the posed photos that see those points (survey photos of other
   camera groups and the posed frames themselves), ranked by how alike their view is to the frame's expected one:
   the direction they look from (past ~50 deg apart the matcher mostly fails) and the detail they show (px per m: an
   aligned frame from 2 m away shows a colony too small to match).
3. **Matching** (LoFTR, :class:`imagesync.LearnedMatcher`, half precision): the frame against each candidate, turned
   by quarter turns to the candidate's in-plane orientation (the learned matchers do not match a photo turned
   round). On sa3_40m (frames at close-up times, 8 candidates each) LoFTR gave 15 or more correct matches for 75% of
   the pairs, DISK + LightGlue for 40%.
4. **Pose** (:func:`source_pose`, :func:`pool`): each match's pixel in the candidate is followed along its ray to the
   cloud (:class:`DepthLookup`), so the frame's pixels see known 3-D points; PnP per candidate (a candidate whose
   pose the diver could not have swum to from the neighbours is dropped), then PnP on the pooled inliers of those
   that agree (the frame's lens distortion undone). Once posed, the photos that see what the frame sees, from its
   direction, are tried too.

:func:`pose_frames` poses the frames a run of close-ups needs, **chained**: the unaligned frames between them and the
nearest aligned frames are posed too, from the edges of the gap in, each a neighbour 1 s off (and a photo to match)
for the next. From neighbours 30 s off a frame's expected place can be metres out (sa3_40m: 2 of 4 posed); 1 s off it
is not.

The pose is in the cloud's (world) frame, as :func:`imagesync.set_pose` takes it. Needs ``kornia``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import imagesync as isy
from .frames import uniform_scale

logger = logging.getLogger(__name__)

FRAME_WIDTH = 960  # px, the frame resized to for matching (LoFTR, fp16: ~3.4 GB of GPU memory with a photo)
SOURCE_WIDTH = 1200  # px, a posed photo resized to
LOCAL_RADIUS = 2.5  # m, the cloud round the anchors used for depths
LOCAL_VOXEL = 0.004  # m, that cloud thinned to
CELL_SPACING = 1.5  # a depth map's cell: this many thinned points' spacing (px at the photo's median depth)
MAX_CANDIDATES = 40  # posed photos tried
MAX_SOURCES = 8  # agreeing candidates pooled
SOURCE_MIN_INLIERS = 25  # a candidate's own PnP inliers for it to count
MIN_POOLED = 2  # photos whose poses agree, for a pose ...
SINGLE_MIN_INLIERS = 150  # ... or this many inliers from one
PNP_THRESHOLD = 0.0025  # of the frame's width (~10 px of 3840)
MIN_INLIERS = 40  # pooled PnP inliers for a pose
AGREE_M = 0.05  # a candidate's own pose centre within this of the best candidate's to be pooled
MAX_GAP_S = 120  # aligned frames at most this far off in time are used as anchors
MAX_SPEED = 0.5  # m/s, the diver's: a pose further than this x the time from its neighbours is implausible
CHAIN_SOURCE_MAX_RANGE = 1.2  # m, a chained frame further from what it aims at is not matched against
SEED_GAP_S = 10  # a frame with no posed frame nearer than this (s) is placed by image retrieval (seed) first
SEED_CANDIDATES = 20  # survey photos most like the frame (frameretrieval.GlobalIndex) verified for a seed
SEED_MAX_CANDIDATES = 16  # photos then tried for its pose
ORIENT_FRAMES = 6  # frames posed to tell which way round a borrowed lens fits (choose_orientation)
ORIENT_SHARE = 2 / 3  # ... the share of them that must fit the other way round better to turn it
PHOTO_MIN_DETAIL = 0.6  # a photo sensor showing the reef in less than this share of the best one's detail is not
# matched with by default (photo_sensors; sa4_60m: an AUV's photos from 3.4 m, 0.44 of the frames' / 0.49 of the
# survey photos'; the survey photos 0.79-0.91 of the frames')


# --------------------------------------------------------------------------- geometry


def world_to_cam(T: np.ndarray) -> np.ndarray:
    """The rotation world -> camera of a current-frame camera transform ``T`` (which carries the world
    transform's scale: ``T[:3, :3]`` = s R_cw)."""
    M = np.asarray(T, float).reshape(4, 4)[:3, :3]
    return (M / uniform_scale(M)).T


def rotation_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """The angle between two rotations (deg)."""
    c = (np.trace(np.asarray(Ra) @ np.asarray(Rb).T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


pose_transform = isy.pose_transform  # (one definition, shared with imagesync.set_pose)


def roll_turns(T_view: np.ndarray, T_source: np.ndarray) -> int:
    """Quarter turns (np.rot90's k) that bring the image of a view with pose ``T_view`` to the in-plane
    orientation of a photo with ``T_source``, both looking roughly the same way (as :func:`imagesync.quarter_turns`
    turns a close-up: from the angle at which the view's x axis appears in the photo)."""
    Rv, Rs = world_to_cam(T_view), world_to_cam(T_source)
    xv = Rs @ Rv.T @ np.array([1.0, 0.0, 0.0])  # the view's x axis in the photo's camera frame
    ang = np.degrees(np.arctan2(xv[1], xv[0]))  # (image y is down: positive = clockwise in the image)
    return isy.quarter_turns(ang)


class DepthLookup:
    """A camera's depth map of a (local, thinned) cloud: the 3-D point each of its pixels sees.

    ``sensor`` and ``T`` (its current-frame camera transform) are enough, so a pose being tried can be used as
    well as a camera's own. The cloud's points are projected (the lens as :func:`cameras.camera_points_to_pixels`) into cells of
    ``cell`` px, keeping the nearest; empty cells and those seen through a gap (deeper than their neighbours) take
    the nearest of the 3 x 3 cells round them. A pixel's point is then where its ray (the lens undone) reaches
    that depth."""

    def __init__(self, sensor, T: np.ndarray, points: np.ndarray, *, spacing: float = LOCAL_VOXEL,
                 cell: Optional[int] = None):
        from scipy.ndimage import minimum_filter
        from .cameras import camera_points_to_pixels

        self.sensor, self.T = sensor, np.asarray(T, float).reshape(4, 4)
        Ti = np.linalg.inv(self.T)
        pc = np.asarray(points, float) @ Ti[:3, :3].T + Ti[:3, 3]
        front = pc[:, 2] > 0
        pc = pc[front]
        x, y, inv = camera_points_to_pixels(pc[:, 0], pc[:, 1], pc[:, 2], sensor)
        x, y, z = x[inv], y[inv], pc[inv, 2]
        self.n_points = int(len(z))
        if cell is None:
            # (the thinned points' spacing in px at the median depth; the transform's scale turns m into its units)
            s = uniform_scale(self.T)
            zmed = float(np.median(z)) if len(z) else 1.0
            cell = int(np.clip(CELL_SPACING * sensor.f * spacing / max(zmed * s, 1e-6), 2, 64))
        self.cell = int(cell)
        w, h = int(np.ceil(sensor.width / self.cell)) + 1, int(np.ceil(sensor.height / self.cell)) + 1
        zb = np.full(w * h, np.inf)
        if len(z):
            flat = (y // self.cell).astype(int) * w + (x // self.cell).astype(int)
            np.minimum.at(zb, flat, z)
        zb = zb.reshape(h, w)
        mn = minimum_filter(zb, size=3, mode="constant", cval=np.inf)
        self.depth = np.where(np.isinf(zb) | (zb > 1.05 * mn), mn, zb)

    def lift(self, px: np.ndarray) -> np.ndarray:
        """The 3-D points (world frame) seen at pixels ``px`` (N x 2, full resolution); NaN rows where the map has
        no depth or the lens model has no ray."""
        from .cameras import pixels_to_normalized

        px = np.asarray(px, float).reshape(-1, 2)
        h, w = self.depth.shape
        ix = np.clip((px[:, 0] // self.cell).astype(int), 0, w - 1)
        iy = np.clip((px[:, 1] // self.cell).astype(int), 0, h - 1)
        z = self.depth[iy, ix]
        xn, yn = pixels_to_normalized(px[:, 0], px[:, 1], self.sensor)
        bad = ~np.isfinite(z) | ~np.isfinite(xn) | ~np.isfinite(yn)
        z, xn, yn = np.where(bad, 0.0, z), np.where(bad, 0.0, xn), np.where(bad, 0.0, yn)
        out = np.stack([xn * z, yn * z, z], 1) @ self.T[:3, :3].T + self.T[:3, 3]
        out[bad] = np.nan
        return out


def local_cloud(pcd, centres: np.ndarray, radius: float = LOCAL_RADIUS, voxel: float = LOCAL_VOXEL) -> np.ndarray:
    """The cloud's points within ``radius`` (m, world frame) of any of ``centres``, thinned to ``voxel``."""
    import open3d as o3d

    P = pcd.points
    centres = np.atleast_2d(np.asarray(centres, float))
    lo, hi = centres.min(0) - radius, centres.max(0) + radius
    box = np.all((P >= lo) & (P <= hi), axis=1)
    sub = P[box]
    if len(sub):
        d = np.min(np.stack([np.linalg.norm(sub - c, axis=1) for c in centres], 0), 0)
        sub = sub[d <= radius]
    if not len(sub):
        return sub
    q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub))
    return np.asarray(q.voxel_down_sample(voxel).points)


def aim_of(R: np.ndarray, C: np.ndarray, pcd) -> Optional[np.ndarray]:
    """Where a pose's centre ray first meets the cloud (3, 9, 27 mm search radius), else None."""
    ray = np.asarray(R, float).T @ np.array([0.0, 0.0, 1.0])
    r = isy.RAY_SEARCH_RADIUS
    while r <= isy.AIM_MAX_SEARCH_RADIUS:
        hit = pcd.get_intercept(np.asarray(C, float), r, vector=ray, max_dist_from_origin=isy.RAY_MAX_DISTANCE)
        if hit is not None:
            return np.asarray(hit.coords, float)
        r *= 3
    return None


def camera_aim(cam, pcd) -> Optional[np.ndarray]:
    """Where a posed camera's centre ray first meets the cloud."""
    return aim_of(world_to_cam(cam.camera_transform), np.asarray(cam.coords, float), pcd)


# --------------------------------------------------------------------------- where to look


def neighbours(t: _dt.datetime, posed_frames: Dict[_dt.datetime, Any], *, max_gap: float = MAX_GAP_S,
               exclude: Sequence[Any] = ()) -> List[Tuple[float, Any]]:
    """The posed frames nearest in time before and after ``t`` (at most ``max_gap`` s off): [(dt s, camera)]."""
    ex = {id(c) for c in exclude}
    before = [(ts, c) for ts, c in posed_frames.items() if ts < t and id(c) not in ex]
    after = [(ts, c) for ts, c in posed_frames.items() if ts > t and id(c) not in ex]
    out = []
    if before:
        ts, c = max(before, key=lambda v: v[0])
        out.append(((ts - t).total_seconds(), c))
    if after:
        ts, c = min(after, key=lambda v: v[0])
        out.append(((ts - t).total_seconds(), c))
    return [(d, c) for d, c in out if abs(d) <= max_gap]


def anchors(nbrs: Sequence[Tuple[float, Any]], pcd, step: float = 0.5) -> np.ndarray:
    """Points on the cloud the frame most likely saw: its neighbours' aim points, and points every ``step`` m on
    the line between them (the diver swam from one to the other)."""
    aims = []
    for _d, c in nbrs:
        a = camera_aim(c, pcd)
        if a is None:  # (fallback: 0.6 m along its view)
            a = np.asarray(c.coords, float) + 0.6 * world_to_cam(c.camera_transform).T @ np.array([0, 0, 1.0])
        aims.append(a)
    if len(aims) == 2:
        n = int(np.ceil(np.linalg.norm(aims[1] - aims[0]) / step))
        if n > 1:
            aims = [aims[0] + (aims[1] - aims[0]) * f for f in np.linspace(0, 1, n + 1)]
    return np.array(aims, float).reshape(-1, 3)


def expected_pose(t: _dt.datetime, nbrs: Sequence[Tuple[float, Any]]) -> Optional[np.ndarray]:
    """A transform for the frame from its neighbours (the nearer one's): for the turns to try first."""
    if not nbrs:
        return None
    _d, c = min(nbrs, key=lambda v: abs(v[0]))
    return np.asarray(c.camera_transform, float).reshape(4, 4)


# --------------------------------------------------------------------------- posing


class FramePoser:
    """Poses frames from posed photos (``sources``: any posed cameras, e.g. the survey photos and the aligned
    frames) and the cloud ``pcd`` (world frame). Keeps the matcher and the photos' features between frames."""

    def __init__(self, pcd, sources: Sequence[Any], *, device: Optional[str] = None,
                 max_candidates: int = MAX_CANDIDATES, max_sources: int = MAX_SOURCES, cache: int = 300):
        from collections import OrderedDict
        from .cameras import can_project

        self.pcd = pcd
        self.sources = [c for c in sources if can_project(c, use_orig_coords=False)]
        self.m = isy.LearnedMatcher("loftr", device)
        self.max_candidates, self.max_sources = max_candidates, max_sources
        self._img: "OrderedDict[Any, Any]" = OrderedDict()
        self._cache = cache
        W = np.asarray(pcd.world_transform, float)
        self.scale = uniform_scale(W)

    # ---- images and matching
    @staticmethod
    def _image(path: str, full_width: int, width: int):
        g = isy.resize_to_width(isy.read_gray(path, isy.jpeg_reduction_for(full_width, width)), width)
        return np.ascontiguousarray(g[: g.shape[0] // 8 * 8])

    def source_image(self, cam):
        """A posed photo, grey, ``SOURCE_WIDTH`` px wide (a frame: ``FRAME_WIDTH``), and its scale to full
        resolution; the last ``cache`` kept."""
        key = id(cam)
        if key in self._img:
            self._img.move_to_end(key)
            return self._img[key]
        width = min(cam.sensor.width, FRAME_WIDTH if cam.sensor.width <= 4000 else SOURCE_WIDTH)
        g = self._image(cam.filepath, cam.sensor.width, width)
        self._img[key] = (g, cam.sensor.width / g.shape[1])
        while len(self._img) > self._cache:
            self._img.popitem(last=False)
        return self._img[key]

    def _match(self, g0, g1) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.m.torch
        if self.m.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return self.m.match(g0, g1)
        return self.m.match(g0, g1)

    # ---- candidates
    @staticmethod
    def match_focal(cam_or_sensor, is_frame: bool = False) -> float:
        """A photo's focal length (px) at the resolution it is matched at."""
        s = getattr(cam_or_sensor, "sensor", None) or cam_or_sensor
        width = min(s.width, FRAME_WIDTH if (is_frame or s.width <= 4000) else SOURCE_WIDTH)
        return float(s.f) * width / s.width

    def candidates(self, pts: np.ndarray, *, view: Optional[np.ndarray] = None, density: Optional[float] = None,
                   exclude: Sequence[Any] = (), limit: Optional[int] = None) -> List[Any]:
        """The posed photos that see points ``pts`` (world frame), best first. Ranked by how alike they are to the
        frame's expected view: the direction they look at the points from (``view``: from the frame's expected
        centre to the points; past ~50 deg apart the matcher mostly fails, sa3_40m), and the detail they show there
        (``density``: the frame's expected px per m at matching resolution; a GoPro frame from 2 m away shows a
        colony too small), then nearer their image centre. Without ``view`` / ``density``: those seeing more of the
        points, nearer their image centre."""
        from .cameras import pixel_hits

        ex = {id(c) for c in exclude}
        score: Dict[int, List[Tuple[float, float]]] = {}
        cams: Dict[int, Any] = {}
        v = None if view is None else np.asarray(view, float) / np.linalg.norm(view)
        for p in np.atleast_2d(pts):
            for cam, x, y, _depth, _rel in pixel_hits(self.sources, p, use_orig_coords=False):
                if id(cam) in ex:
                    continue
                w, h = cam.sensor.width, cam.sensor.height
                centre = 1.0 - np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2)) / np.sqrt(2)
                u = p - np.asarray(cam.coords, float)
                dist = float(np.linalg.norm(u))
                cost = -0.5 * centre
                if v is not None:
                    cost += float(np.degrees(np.arccos(np.clip(u @ v / dist, -1, 1)))) / 15.0
                if density:
                    cost += 1.5 * abs(np.log2(self.match_focal(cam) / dist / density))
                score.setdefault(id(cam), []).append((cost, centre))
                cams[id(cam)] = cam
        if v is not None or density:
            key = lambda k: min(c for c, _ in score[k])  # noqa: E731
        else:
            key = lambda k: (-len(score[k]), -max(c for _a, c in score[k]))  # noqa: E731
        ranked = sorted(score, key=key)
        return [cams[k] for k in ranked[: (limit or self.max_candidates)]]

    # ---- one frame
    def pose(self, frame, sensor, pts: np.ndarray, *, expected_T: Optional[np.ndarray] = None,
             nbrs: Sequence[Tuple[float, Any]] = (), exclude: Sequence[Any] = (), seed: int = 0,
             limit: Optional[int] = None) -> Dict[str, Any]:
        """Pose ``frame`` (a camera with ``filepath``; ``sensor`` its calibrated sensor) expected to see the reef
        round ``pts`` (:func:`anchors`), looking from about where ``expected_T`` (:func:`expected_pose`) does,
        from the candidates that see those (:meth:`candidates`). A photo whose own pose of the frame is further from
        the neighbours ``nbrs`` than the diver could swim (:func:`plausible`) is not used. Once posed, the photos
        that see the frame's own view from its own direction are tried as well (up to ``max_sources`` agreeing).

        Returns:
            dict: ``ok``; when ok ``R`` (world -> camera), ``C``, ``T`` (current-frame transform), ``inliers``,
            ``error`` (median reprojection error, px), ``aim``, ``range``, ``n_pooled``; always ``sources``
            ([(filename, own inliers)]), ``tried``, ``seconds`` and ``reason`` when not ok.
        """
        t0 = time.time()
        out: Dict[str, Any] = dict(ok=False, sources=[], tried=0)
        view = density = None
        if expected_T is not None:
            Te = np.asarray(expected_T, float).reshape(4, 4)
            P_ = np.asarray(pts, float).reshape(-1, 3)
            view = P_.mean(0) - Te[:3, 3]
            rng = float(np.min(np.linalg.norm(P_ - Te[:3, 3], axis=1)))  # (the nearer neighbour's range)
            density = self.match_focal(sensor, is_frame=True) / max(rng, 0.1)
        cands = self.candidates(pts, view=view, density=density, exclude=exclude, limit=limit)
        if not cands:
            out.update(reason="no posed photo sees the expected place", seconds=time.time() - t0)
            return out
        local = local_cloud(self.pcd, pts)
        if not len(local):
            out.update(reason="no cloud at the expected place", seconds=time.time() - t0)
            return out
        g = self._image(frame.filepath, sensor.width, FRAME_WIDTH)
        fsc = sensor.width / g.shape[1]
        views = {k: np.ascontiguousarray(np.rot90(g, k)) for k in range(4)}
        per: List[Dict[str, Any]] = []
        tried = set()

        def attempt(cam, T_guess):
            out["tried"] += 1
            tried.add(id(cam))
            sg, ssc = self.source_image(cam)
            k_first = roll_turns(T_guess, cam.camera_transform) if T_guess is not None else 0
            turns = [k_first] if T_guess is not None else [0, 1, 2, 3]
            for k in turns:
                a, b = self._match(views[k], sg)
                if len(a) < SOURCE_MIN_INLIERS:
                    continue
                a = isy.to_sensor_px(isy.unrotate(a, k, g.shape[1], g.shape[0]), fsc)
                got = source_pose(sensor, cam, a, isy.to_sensor_px(b, ssc), local, seed=seed)
                if got is None:
                    continue
                ok, _note = plausible(dict(ok=True, C=got["C"]), nbrs)
                if not ok:
                    continue
                got["turns"] = (k_first, k)
                per.append(got)
                return

        def enough():  # (and MIN_POOLED of them photos with a pose of their own: a chained frame matches best,
            # but would carry an error on)
            good = [v for v in per if v["inliers"] >= 2 * SOURCE_MIN_INLIERS]
            return len(good) >= self.max_sources and sum(not chained(v["cam"]) for v in good) >= MIN_POOLED

        for cam in cands:
            attempt(cam, expected_T)
            if enough():
                break
        if not per:
            out.update(reason=f"no candidate matched ({out['tried']} tried)", seconds=time.time() - t0)
            return out
        res = pool(sensor, per, scale=self.scale, pcd=self.pcd, max_sources=self.max_sources, seed=seed)
        if res["ok"] and (res["n_pooled"] < self.max_sources or not res["anchored"]) and not enough():
            # (refined: the photos that see what the frame sees, from its direction)
            dl = DepthLookup(sensor, res["T"], local)
            grid = np.array([(x, y) for x in np.linspace(0.2, 0.8, 4) * sensor.width
                             for y in np.linspace(0.2, 0.8, 3) * sensor.height])
            P = dl.lift(grid)
            P = P[np.isfinite(P).all(1)]
            if len(P):
                rng = float(np.median(np.linalg.norm(P - res["C"], axis=1)))
                more = [c for c in self.candidates(P, view=P.mean(0) - res["C"], exclude=exclude,
                                                   density=self.match_focal(sensor, is_frame=True) / max(rng, 0.1))
                        if id(c) not in tried][: self.max_candidates // 2]
                for cam in more:
                    attempt(cam, res["T"])
                    if enough():
                        break
                res = pool(sensor, per, scale=self.scale, pcd=self.pcd, max_sources=self.max_sources, seed=seed)
        out.update(res)
        if out["ok"] and out["n_pooled"] < MIN_POOLED and out["inliers"] < SINGLE_MIN_INLIERS:
            out.update(ok=False, reason=f"only {out['n_pooled']} photo(s) agree, {out['inliers']} inliers (need "
                                        f"{MIN_POOLED}, or {SINGLE_MIN_INLIERS} inliers)")
        out["turns"] = [v["turns"] for v in per]  # (expected, matched) per photo with a pose
        out["seconds"] = time.time() - t0
        return out


    # ---- a frame with no posed neighbour
    def place(self, frame, sensor, index, *, k: int = SEED_CANDIDATES, exclude: Sequence[Any] = ()
              ) -> Dict[str, Any]:
        """Pose ``frame`` with no posed frame near it in time (:mod:`substrata.frameretrieval`): the ``k`` photos
        of ``index`` (a :class:`frameretrieval.GlobalIndex` of posed survey photos) most like it are matched with
        it (LoFTR, the frame turned 4 ways) until one shares ``VERIFY_MIN_INLIERS`` matches consistent with one
        fundamental matrix (unrelated pairs on sa3_40m: at most 42); the cloud points those matches see there are
        where the frame looks, and it is posed from the photos that see them (:meth:`pose`).

        Returns:
            :meth:`pose`'s dict, plus ``placed_by`` ((survey photo filename, its matches)) or ``reason``."""
        import cv2
        from . import frameretrieval as fr

        t0 = time.time()
        hits = index.query([frame.filepath], k)[0] if len(getattr(index, "cams", (1,))) else []
        g = self._image(frame.filepath, sensor.width, FRAME_WIDTH)
        best = None
        for n_tried, (cam, _sim) in enumerate(hits, 1):
            if getattr(cam, "sensor", None) is None:
                continue
            sg, ssc = self.source_image(cam)
            for kt in range(4):
                a, b = self._match(np.ascontiguousarray(np.rot90(g, kt)), sg)
                if len(a) < 8:
                    continue
                _F, mask = cv2.findFundamentalMat(np.float32(a), np.float32(b), cv2.USAC_MAGSAC,
                                                  fr.VERIFY_THRESHOLD_PX, 0.999, 5000)
                n = 0 if mask is None else int(mask.sum())
                if n >= fr.VERIFY_MIN_INLIERS and (best is None or n > best[1]):
                    best = (cam, n, isy.to_sensor_px(b[mask.ravel().astype(bool)], ssc))
            if best is not None:
                break
        if best is None:
            return dict(ok=False, reason=f"no survey photo like it matched ({len(hits)} tried)",
                        seconds=time.time() - t0, tried=len(hits))
        cam, n, bpx = best
        aim = camera_aim(cam, self.pcd)
        if aim is None:
            return dict(ok=False, reason=f"{cam.filename} aims at no cloud", seconds=time.time() - t0)
        dl = DepthLookup(cam.sensor, cam.camera_transform, local_cloud(self.pcd, aim[None]))
        P = dl.lift(bpx)
        P = P[np.isfinite(P).all(1)]
        if not len(P):
            return dict(ok=False, reason=f"{cam.filename}'s matches see no cloud", seconds=time.time() - t0)
        pts = P[np.random.default_rng(0).choice(len(P), min(12, len(P)), replace=False)]
        r = self.pose(frame, sensor, pts, exclude=exclude, limit=SEED_MAX_CANDIDATES)
        r["placed_by"] = (cam.filename, n)
        r["seconds"] = time.time() - t0
        return r


def source_pose(sensor, cam, a: np.ndarray, b: np.ndarray, local: np.ndarray, *, seed: int = 0
                ) -> Optional[Dict[str, Any]]:
    """The frame's pose from its matches with one posed photo ``cam``: ``a`` (frame px) <-> ``b`` (``cam``'s px,
    lifted to the ``local`` cloud through ``cam``'s depth map, :class:`DepthLookup`), PnP through the frame's
    ``sensor``. None with fewer than ``SOURCE_MIN_INLIERS`` inliers; else ``inliers``, ``R``, ``C``, ``obj`` /
    ``img`` (the inlier 3-D points and frame px) and ``cam``."""
    P = DepthLookup(cam.sensor, cam.camera_transform, local).lift(b)
    ok = np.isfinite(P).all(1)
    if ok.sum() < SOURCE_MIN_INLIERS:
        return None
    got = isy.pose_from_points(P[ok], np.asarray(a, float)[ok], sensor, threshold=PNP_THRESHOLD, seed=seed)
    if got is None or len(got[2]) < SOURCE_MIN_INLIERS:
        return None
    R, C, inl, _err = got
    return dict(inliers=int(len(inl)), R=R, C=C, obj=P[ok][inl], img=np.asarray(a, float)[ok][inl], cam=cam)


def chained(cam) -> bool:
    """Whether ``cam`` is a frame posed here (:func:`posed_copy`) rather than a photo with a pose of its own."""
    return bool(getattr(cam, "framepose_chained", False))


def pool(sensor, per: Sequence[Dict[str, Any]], *, scale: float, pcd=None, max_sources: int = MAX_SOURCES,
         seed: int = 0) -> Dict[str, Any]:
    """The frame's pose from the photos' own poses (:func:`source_pose`): those whose centre is within ``AGREE_M``
    of a reference pooled, PnP on their inliers. The reference is the best (most inliers) of the photos with a pose
    of their own (survey photos, aligned frames) when enough of those agree (``MIN_POOLED``, or one with
    ``SINGLE_MIN_INLIERS``): a frame posed here (:func:`chained`) always matches its neighbour best, and would
    otherwise carry a wrong step on along the chain (sa3_40m, a 30 s gap: one side's chain 3.5 cm off after one
    step, 40 cm after three, while the photos disagreed). Without them, the best of all.

    Returns:
        dict: ``ok``, ``sources`` ([(filename, own inliers)]), ``pooled`` (filenames), ``anchored`` (the reference
        has a pose of its own), and when ok ``R``, ``C``, ``T`` (with the world transform's ``scale``),
        ``inliers``, ``error``, ``n_pooled``, and with ``pcd`` ``aim`` / ``range``; ``reason`` when not ok.
    """
    per = sorted(per, key=lambda v: -v["inliers"])
    own = [v for v in per if not chained(v["cam"])]
    anchored = False
    if own:
        agree = [v for v in own if np.linalg.norm(v["C"] - own[0]["C"]) <= AGREE_M]
        anchored = len(agree) >= MIN_POOLED or agree[0]["inliers"] >= SINGLE_MIN_INLIERS
    Cb = (own if anchored else per)[0]["C"]
    keep = [v for v in per if np.linalg.norm(v["C"] - Cb) <= AGREE_M][:max_sources]
    out: Dict[str, Any] = dict(ok=False, sources=[(v["cam"].filename, int(v["inliers"])) for v in per],
                               pooled=[v["cam"].filename for v in keep], anchored=anchored)
    got = isy.pose_from_points(np.concatenate([v["obj"] for v in keep]), np.concatenate([v["img"] for v in keep]),
                               sensor, threshold=PNP_THRESHOLD, seed=seed)
    if got is None or len(got[2]) < MIN_INLIERS:
        out["reason"] = f"pooled PnP: {0 if got is None else len(got[2])} inliers (need {MIN_INLIERS})"
        return out
    R, C, inl, err = got
    aim = aim_of(R, C, pcd) if pcd is not None else None
    out.update(ok=True, R=R, C=C, T=pose_transform(R, C, scale), inliers=int(len(inl)), error=float(np.median(err)),
               aim=aim, range=None if aim is None else float(np.linalg.norm(aim - C)), n_pooled=len(keep),
               obj=np.concatenate([v["obj"] for v in keep]), img=np.concatenate([v["img"] for v in keep]))
    return out


def plausible(result: Dict[str, Any], nbrs: Sequence[Tuple[float, Any]], *, max_speed: float = MAX_SPEED,
              margin: float = 0.3) -> Tuple[bool, str]:
    """Whether a pose fits its neighbours in time: its centre within ``max_speed`` x the time (+ ``margin`` m) of
    each. (ok, note)."""
    if not result.get("ok"):
        return False, result.get("reason", "no pose")
    for d, c in nbrs:
        dist = float(np.linalg.norm(np.asarray(result["C"]) - np.asarray(c.coords, float)))
        if dist > max_speed * abs(d) + margin:
            return False, f"{dist:.2f} m from the frame {abs(d):.0f} s off"
    return True, ""


def posed_copy(cam, R: np.ndarray, C: np.ndarray, world_transform: np.ndarray):
    """A copy of camera ``cam`` with the pose ``R`` / ``C`` (world frame; :func:`imagesync.set_pose`), ``cam``
    itself untouched: a frame posed here, used to pose the next ones."""
    import copy

    c = copy.copy(cam)
    isy.set_pose(c, R, C, world_transform)
    c.enabled = True
    c.framepose_chained = True  # (its pose is ours: pool() prefers photos with a pose of their own)
    return c


class LazyIndex:
    """A :class:`frameretrieval.GlobalIndex` of ``cams`` built on its first query (indexing ~35 ms a photo)."""

    def __init__(self, cams: Sequence[Any], *, device: Optional[str] = None, progress: bool = True):
        self.cams, self.device, self.progress, self._index = list(cams), device, progress, None

    def query(self, paths: Sequence[str], k: int, **kw):
        if self._index is None:
            from .frameretrieval import GlobalIndex

            if self.progress:
                print(f"  (indexing {len(self.cams)} photos to place frames by retrieval)", flush=True)
            self._index = GlobalIndex(device=self.device).build(self.cams)
        return self._index.query(paths, k, **kw)


def gap_frames(t: _dt.datetime, frames_by_time: Dict[_dt.datetime, Any], posed_times: set,
               max_gap: float = MAX_GAP_S) -> List[_dt.datetime]:
    """The unaligned frames' times from ``t`` out to the nearest posed frame on either side (at most ``max_gap`` s):
    those a chain from the posed frames passes through to reach ``t``. A side that reaches no posed frame within
    ``max_gap`` (or a missing second first) adds none."""
    out = [t]
    for step in (-1, 1):
        ts, run = t, []
        for _ in range(int(max_gap)):
            ts = ts + _dt.timedelta(seconds=step)
            if ts in posed_times:
                out += run
                break
            if ts not in frames_by_time:
                break
            run.append(ts)
    return sorted(set(out))


def pose_frames(frames: Sequence[Any], all_frames: Sequence[Any], sources: Sequence[Any], pcd, *,
                sensor=None, device: Optional[str] = None, max_gap: float = MAX_GAP_S, chain: bool = True,
                unaligned: Sequence[Any] = (), max_sources: int = MAX_SOURCES, place: bool = False,
                index=None, keep_matches: bool = False, progress: bool = True) -> Dict[str, Dict[str, Any]]:
    """Pose the unaligned ``frames`` (cameras with ``datetime``) from the posed photos ``sources`` and the aligned
    frames among ``all_frames`` (their neighbours in time give where to look).

    With ``chain`` (default) the unaligned frames between each and the nearest aligned frames (:func:`gap_frames`)
    are posed too, nearest an aligned (or already posed) frame first, each posed frame then a neighbour and a
    photo to match for the next: every step has a neighbour 1 s off, where the frame can only have moved a little
    (sa3_40m: a frame posed from neighbours 30 s off was found for 2 of 4, 1 s off for every one). ``unaligned``:
    frames to treat as unaligned though they have a pose (a check against Metashape's). ``sensor``: the frames'
    calibrated sensor (default: that of an aligned frame, else of any frame: e.g. one borrowed from another plot,
    :func:`frameretrieval.give_sensor`).

    With ``place``, a frame with no posed frame within ``SEED_GAP_S`` s is first placed by image retrieval
    (:meth:`FramePoser.place`; ``index``: a :class:`frameretrieval.GlobalIndex` of ``sources``, built when first
    needed): the frames near it are then chained from it. This poses frames where Metashape aligned none.

    Returns:
        dict: per frame ``cam_id`` (``frames`` and those chained through) the result of :meth:`FramePoser.pose`
        (``anchored``: photos with a pose of their own agree on it, :func:`pool`) plus ``plausible`` / ``note``
        (:func:`plausible`), ``neighbours`` ([(dt s, filename)]), ``asked`` (one of ``frames``), ``steps`` (s to
        the nearest aligned frame) and, for a frame placed by retrieval, ``placed_by``; with ``keep_matches``
        ``obj`` / ``img`` (the pooled 3-D points and frame px; :func:`choose_orientation`). Use a pose only when ``ok``, ``plausible`` and ``anchored``. Nothing is written
        to the cameras (:func:`imagesync.set_pose` does that, with ``R`` and ``C``).
    """
    from .cameras import can_project

    skip = {id(c) for c in unaligned}
    by_t = isy.by_time(all_frames)
    aligned = {t: c for t, c in by_t.items() if can_project(c, use_orig_coords=False) and id(c) not in skip}
    if sensor is None:
        sensor = next((c.sensor for c in list(aligned.values()) + list(all_frames) if c.sensor is not None), None)
    if sensor is None:
        raise ValueError("No aligned frame and no sensor: the frames' lens is unknown")
    W = np.asarray(pcd.world_transform, float)
    poser = FramePoser(pcd, [c for c in list(sources) + list(aligned.values()) if id(c) not in skip], device=device,
                       max_sources=max_sources)
    posed = dict(aligned)
    asked = {}
    for f in frames:
        t = isy.parse_time(getattr(f, "datetime", None))
        if t is not None:
            asked[t] = f
    todo = set()
    for t in asked:
        todo |= set(gap_frames(t, by_t, set(aligned), max_gap)) if chain else {t}
    out: Dict[str, Dict[str, Any]] = {}
    for f in frames:
        if isy.parse_time(getattr(f, "datetime", None)) is None:
            out[str(f.cam_id)] = dict(ok=False, reason="no time", plausible=False, asked=True)

    def gap_to(t, times):
        return min((abs((t - u).total_seconds()) for u in times), default=np.inf)

    while todo:
        t = min(todo, key=lambda u: (gap_to(u, posed), u))
        todo.discard(t)
        f = by_t[t]
        steps = gap_to(t, aligned)
        nb = neighbours(t, posed, max_gap=max_gap, exclude=[f])
        r = None
        if place and (not nb or min(abs(d) for d, _c in nb) > SEED_GAP_S):
            if index is None:
                index = LazyIndex([c for c in sources if id(c) not in skip], device=device, progress=progress)
            r = poser.place(f, sensor, index, exclude=[f])
            r["plausible"], r["note"] = (r.get("ok", False), r.get("reason", "")) if not r.get("ok") else (True, "")
            if not (r.get("ok") and r.get("anchored")) and nb:
                r = None  # (then from the neighbours, as without place)
        if r is None and not nb:
            r = dict(ok=False, reason=f"no posed frame within {max_gap:.0f} s", plausible=False)
        elif r is None:
            r = poser.pose(f, sensor, anchors(nb, pcd), expected_T=expected_pose(t, nb), nbrs=nb, exclude=[f])
            r["plausible"], r["note"] = plausible(r, nb)
            r["neighbours"] = [(d, c.filename) for d, c in nb]
        if chain and r.get("ok") and r["plausible"] and r.get("anchored"):
            # (only a frame the photos with a pose of their own agree on is chained on: a stretch where none
            # does ends the chain rather than carrying an error along it)
            c = posed_copy(f, r["R"], r["C"], W)
            posed[t] = c
            if r.get("range") is not None and r["range"] <= CHAIN_SOURCE_MAX_RANGE:
                # (a frame looking across the reef is posed less well, sa3_40m: a neighbour still, not a photo)
                poser.sources.append(c)
        r.update(asked=t in asked, steps=steps)
        if not keep_matches:
            r.pop("obj", None)
            r.pop("img", None)
        out[str(f.cam_id)] = r
        if progress:
            print(f"  {f.filename}: " + ("placed by retrieval, " if r.get("placed_by") and r.get("ok") else "")
                  + (f"posed ({r['inliers']} inliers from {r.get('n_pooled')} photos, error "
                                         f"{r['error']:.1f} px)" + ("" if r["plausible"] else f", implausible: {r['note']}")
                                         if r.get("ok") else f"no pose ({r.get('reason')})")
                  + f" [{r.get('seconds', 0):.0f} s]", flush=True)
    return out


def choose_orientation(frames: Sequence[Any], all_frames: Sequence[Any], sources: Sequence[Any], pcd, sensor,
                       other, *, n: int = ORIENT_FRAMES, tries: int = 3, index=None, unaligned: Sequence[Any] = (),
                       device: Optional[str] = None, progress: bool = True) -> Dict[str, Any]:
    """Which of two lenses fits the frames: ``sensor`` or ``other`` (a borrowed calibration and the same turned
    180 deg, :func:`frameretrieval.rotated_180`: the camera may have been mounted the other way up there). Frames
    spread over ``frames`` are posed (unchained; placed by retrieval where no posed frame is near) through
    ``sensor`` until ``n`` are (at most ``tries`` x ``n`` tried); their pooled matches are posed again through
    each lens, and ``other`` is chosen when its median reprojection error is the lower for ``ORIENT_SHARE`` of
    them (at least 3). Matches gathered through the wrong lens still fit the right one better (sa4_60m, whose GoPro
    was upside down relative to sa3_40m's, with sa3_40m's lens unturned: 10 of 10 frames, median error 1.75-3.51 px
    against 2.33-3.59).

    Returns:
        dict: ``sensor`` (the chosen), ``turned`` (``other`` chosen), ``better`` / ``n`` (frames ``other`` fitted
        better / frames compared), ``clear`` (enough frames, and they agree), ``errors`` ([(filename, sensor's px,
        other's px)]). Pass the same ``index`` (:class:`LazyIndex`) to :func:`pose_frames` after, so the photos are
        indexed once."""
    fs = sorted(frames, key=lambda c: (isy.parse_time(getattr(c, "datetime", None)) or _dt.datetime.min, c.filename))
    # (spread over the run: the middle first, then the quarters, eighths, ...)
    order, seen = [], set()
    for depth in range(1, 12):
        for i in np.linspace(0, len(fs) - 1, 2 ** depth + 1)[1:-1].round().astype(int):
            if i not in seen:
                seen.add(i)
                order.append(fs[i])
    order += [f for i, f in enumerate(fs) if i not in seen]
    errors: List[Tuple[str, float, float]] = []
    tried = 0
    while len(errors) < n and tried < min(len(order), tries * n):
        batch = order[tried: tried + n - len(errors)]
        tried += len(batch)
        res = pose_frames(batch, all_frames, sources, pcd, sensor=sensor, device=device, chain=False, place=True,
                          index=index, keep_matches=True, unaligned=unaligned, progress=progress)
        by_id = {str(c.cam_id): c for c in batch}
        for cid, r in res.items():
            if not (r.get("ok") and r.get("anchored")) or r.get("obj") is None:
                continue
            e = []
            for s in (sensor, other):
                got = isy.pose_from_points(r["obj"], r["img"], s, threshold=PNP_THRESHOLD, seed=0)
                e.append(np.inf if got is None else float(np.median(got[3])))
            errors.append((by_id[cid].filename if cid in by_id else cid, e[0], e[1]))
    better = sum(b < a for _f, a, b in errors)
    worse = len(errors) - better
    clear = len(errors) >= 3 and max(better, worse) >= ORIENT_SHARE * len(errors)
    turned = clear and better > worse
    return dict(sensor=other if turned else sensor, turned=turned, better=better, n=len(errors), clear=clear,
                errors=errors)


def photo_detail(cams: Sequence[Any], pcd, *, sample: int = 40, seed: int = 0) -> Optional[float]:
    """How much detail posed photos ``cams`` (one sensor) show of the reef: the median px per m, at the
    resolution they are matched at, at the range they aim at (``sample`` of them). None if none aims at the cloud."""
    if not cams:
        return None
    rng = np.random.default_rng(seed)
    pick = [cams[i] for i in rng.choice(len(cams), min(sample, len(cams)), replace=False)]
    r = []
    for c in pick:
        a = camera_aim(c, pcd)
        if a is not None:
            r.append(float(np.linalg.norm(a - np.asarray(c.coords, float))))
    if not r:
        return None
    return FramePoser.match_focal(pick[0]) / max(float(np.median(r)), 1e-3)


def photo_sensors(photos: Sequence[Any], pcd, *, min_share: float = PHOTO_MIN_DETAIL
                  ) -> Tuple[List[Any], Dict[str, Optional[float]]]:
    """The posed ``photos`` worth matching frames with by default: those of the sensors (``sensor_id``) whose detail
    (:func:`photo_detail`) is at least ``min_share`` of the most detailed sensor's: photos from much further away
    (e.g. an AUV's, sa4_60m: 316 px/m against the survey photos' 563-652) mostly match nothing. The most detailed
    sensor is always kept. (Not against the frames' own detail: a GoPro held close to the reef, sa3_20m ~1,200 px/m,
    would leave out the survey photos too.) Returns (photos kept, {sensor id: detail})."""
    def sid_of(c):
        sid = getattr(c, "sensor_id", None)
        return str(sid if sid is not None else getattr(getattr(c, "sensor", None), "sensor_id", None))

    groups: Dict[str, List[Any]] = {}
    for c in photos:
        groups.setdefault(sid_of(c), []).append(c)
    detail = {sid: photo_detail(cs, pcd) for sid, cs in groups.items()}
    top = max((d for d in detail.values() if d), default=None)
    keep = {sid for sid, d in detail.items() if top is None or d is None or d >= min_share * top}
    return [c for c in photos if sid_of(c) in keep], detail
