"""Where a video was filmed over a plot, and where it stood still: GRAV videos of coral sampling onto the plot.

Made for the GRAV camera of a sampling dive: a GoPro on a float at the top of a weighted pole, looking down on the
diver who moves the pole to a coral, lets it settle and samples (sometimes nothing, sometimes several corals round
the pole). The video is placed on a plot (a substrata project with aligned survey photos and a cloud) so the stops
can be shown in the plot and an annotator only has to put each sample on the right coral, not find where it was.

Steps (:func:`track_videos` runs them all; ``substrata track-video`` is its command):

1. **Keyframes** (:func:`extract_keyframes`): a GoPro's HEVC has a keyframe every 1.001 s; those are decoded alone
   (``-skip_frame nokey``, ~20x real time, read-bound) at full size: one frame a second. Other videos (keyframes
   further apart) are sampled at 1 fps.
2. **Motion** (:func:`frame_motion`): the low (30th) percentile of the optical flow between neighbouring frames, in
   px of a 480-wide frame: the reef is still when the pole stands, even with the diver moving in front of it.
   Stops (:func:`find_stops`): stillness (< ``STILL_FLOW_PX``) for ``MIN_STOP_S`` or more.
3. **Placing a frame** (:class:`PlotMatcher`): LoFTR matches (half precision) between the frame and the plot's
   aligned survey photos. The photos are turned (quarter turns) so the plot's +y is up in all of them, so a frame
   needs only its own turn. Where nothing near is known (a *seed*), the frame is matched against a sparse grid of
   photos over the plot (``SEED_SPACING``) at low resolution, then against the photos round the best; when tracking,
   against the photos nearest the last position. A photo with ``MIN_INLIERS`` matches consistent with one
   fundamental matrix places it: the frame centre is mapped into that photo through a homography of the matches
   nearest the centre and followed along its ray to the cloud.
4. **Tracking** (:func:`track_frames`): every frame while the camera moves; while it is still, the last position
   is kept and checked every ``STILL_RECHECK_S``; lost, a seed is tried at still frames (where the diver stops).

On SAV03 (2025, 40 m and 60 m, three divers) whole-frame image retrieval (DINOv2) did not find the right survey
photos (the frame covers ~4 m, a survey photo ~1 m, and a diver fills part of it), but LoFTR between a GRAV frame
(1280 px) and a survey photo that sees the same reef (640 px) gave 500-1,200 consistent matches across the
strobe-lit / ambient-light gap; unrelated pairs stay below ~45. Tracked frames take ~0.1-0.2 s, a seed a few s.

6. **Camera poses** (:func:`pose_stops`, :mod:`substrata.videopose`): each stop's camera position and orientation,
   with one lens calibrated per camera setting.
5. **Sequences** (:func:`sequence_stops`): a camera's videos (one folder) over a plot are one sequence; its stops
   are numbered in order, and a stop with no frame placed gets a position estimated from the placed frames before
   and after it. Each placed frame also gets the frame's heading and footprint (:meth:`PlotMatcher._pose`).

Outputs (:func:`write_outputs`): ``stops.csv`` (video, start time, duration, position per stop), ``track.csv``
(every placed frame), ``videos.csv``, a map per plot (``<plot>_map.jpg``), ``README.txt`` and
``videotrack.json`` (for substrata-explore's video player); optionally H.264 proxies of the videos (browsers do
not play a GoPro's HEVC everywhere) and, per plot, the stops as an annotation set
(``<id>.annos/<id>_video_stops.csv``) to move onto the sampled corals and label.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import imagesync as _isy
from .compat import renamed_kwargs as _renamed_kwargs
from .fileio import atomic_write as _atomic_write
from .fileio import imread as _imread

logger = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".mov")
MIN_VIDEO_S = 5.0  # shorter videos (a GoPro's accidental clips) are left out
KEYFRAME_MAX_GAP_S = 1.5  # keyframes further apart than this: sample at 1 fps instead
MOTION_WIDTH = 480  # px, the frame width optical flow is measured at
STILL_FLOW_PX = 1.0  # 30th percentile of the flow (px at MOTION_WIDTH) below which the camera is still
MIN_STOP_S = 10.0  # a stop: still for at least this long ...
STOP_GAP_S = 3.0  # ... moving for at most this long within it
MATCH_FRAME_WIDTH = 1280  # px: a GRAV frame (~4 m of reef) and a survey photo (~1 m) at about the same scale
MATCH_PHOTO_WIDTH = 640
SEED_FRAME_WIDTH = 960  # px, the same for the seed search
SEED_PHOTO_WIDTH = 480
SEED_SPACING = 1.5  # m between the survey photos a seed is searched in
MIN_INLIERS = 80  # matches consistent with one fundamental matrix to place a frame
SEED_MIN_INLIERS = 40  # a seed photo's matches for its neighbours to be tried
GOOD_INLIERS = 160  # stop trying further photos / turns once one gives this many
NEAR_PHOTOS = 5  # survey photos tried round a position
NEAR_RADIUS = 2.5  # m
CENTRE_MATCHES = 40  # matches nearest the frame centre used to map it into the photo
STILL_RECHECK_S = 8.0  # still: the kept position is checked this often
TRACK_MEMORY_S = 20.0  # a position this old is still used to look round
RESEED_S = 6.0  # lost: a seed is tried at most this often
PROBE_FRAMES = 12  # frames tried per video and plot to tell whether the video is on the plot
PHOTO_MIN_DETAIL = 0.6  # survey photos of sensors narrower than this share of the widest are not used (AUV, video)
THUMB_REDUCE = 8  # survey photos are cached at 1/8 (JPEG draft decode)
RAY_RADIUS = 0.01  # m, a ray's hit on the cloud
SPIKE_WINDOW_S = 4.0  # a placed frame is compared with those within this many s ...
SPIKE_M = 1.0  # ... and dropped when further than this (+ MAX_SPEED x the time) from their median
MAX_SPEED = 0.5  # m/s, how fast the pole moves at most
POSE_OFFSET = 0.06  # share of the frame height: the points above and right of the centre mapped for the heading
POSE_FRAMES = 3  # frames of a stop placed again for its heading (tracks kept from before headings were recorded)
EST_MAX_GAP_S = 600.0  # an unplaced stop between placed frames this far apart (s) gets an estimated position ...
EST_ONE_SIDE_S = 60.0  # ... or, placed frames on one side only, when the nearest is this close
MATCHED = ("seed", "track", "back", "fill")  # how a frame placed by matching (not carried while still)
MAP_RESOLUTION = 0.005  # m/px of the plot maps
PROXY_HEIGHT = 720  # px, H.264 proxies for the browser


# --------------------------------------------------------------------------- videos and frames


def find_videos(paths: Sequence[str]) -> List[str]:
    """The video files in ``paths`` (files, or folders searched recursively), sorted; hidden files left out."""
    out = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isfile(p):
            out.append(p)
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            out += [os.path.join(root, f) for f in sorted(files)
                    if not f.startswith(".") and f.lower().endswith(VIDEO_EXTS)]
    return sorted(dict.fromkeys(out))


def probe_video(path: str) -> Dict[str, Any]:
    """Duration (s), fps, size and creation time (as the camera wrote it) of a video, from ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s.get("codec_type") == "video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    return {"duration": float(d["format"].get("duration", 0)), "fps": float(num) / float(den or 1),
            "width": int(v["width"]), "height": int(v["height"]), "codec": v.get("codec_name"),
            "creation_time": (d["format"].get("tags") or {}).get("creation_time")}


def video_keys(videos: Sequence[str]) -> Dict[str, str]:
    """A short unique name per video: its file stem, with its folder's name in front where stems repeat."""
    stems: Dict[str, int] = {}
    for v in videos:
        s = os.path.splitext(os.path.basename(v))[0]
        stems[s] = stems.get(s, 0) + 1
    return {v: (os.path.splitext(os.path.basename(v))[0] if stems[os.path.splitext(os.path.basename(v))[0]] == 1
                else os.path.basename(os.path.dirname(v)) + "_" + os.path.splitext(os.path.basename(v))[0])
            for v in videos}


@_renamed_kwargs(out_dir="output_dir")
def extract_keyframes(video: str, output_dir: str, info: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
    """One frame a second of ``video`` as JPEGs in ``output_dir`` (done once: a ``.done`` file marks it): the
    keyframes when they are at most ``KEYFRAME_MAX_GAP_S`` apart (a GoPro's), else frames sampled at 1 fps.
    Returns [(path, time s)], also written as ``frames.csv``."""
    listing = os.path.join(output_dir, "frames.csv")
    if os.path.exists(os.path.join(output_dir, ".done")) and os.path.exists(listing):
        with open(listing) as f:
            return [(os.path.join(output_dir, r["file"]), float(r["t_s"])) for r in csv.DictReader(f)]
    info = info or probe_video(video)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    subprocess.run(base + ["-skip_frame", "nokey", "-i", video, "-fps_mode", "passthrough", "-frame_pts", "1",
                           "-q:v", "3", os.path.join(output_dir, "f_%07d.jpg")], check=True)
    files = sorted(f for f in os.listdir(output_dir) if f.startswith("f_"))
    nums = np.array([int(f[2:9]) for f in files])
    fps = info["fps"]
    if len(nums) < 2 or np.median(np.diff(nums)) / fps > KEYFRAME_MAX_GAP_S:
        for f in files:
            os.remove(os.path.join(output_dir, f))
        subprocess.run(base + ["-i", video, "-vf", "fps=1", "-fps_mode", "passthrough", "-frame_pts", "1", "-q:v",
                               "3", os.path.join(output_dir, "f_%07d.jpg")], check=True)
        files = sorted(f for f in os.listdir(output_dir) if f.startswith("f_"))
        rows = [(f, float(int(f[2:9]))) for f in files]  # (pts in s at 1 fps)
    else:
        rows = [(f, n / fps) for f, n in zip(files, nums)]
    with _atomic_write(listing, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "t_s"])
        w.writerows([(name, "%.3f" % t) for name, t in rows])
    open(os.path.join(output_dir, ".done"), "w").close()
    return [(os.path.join(output_dir, name), t) for name, t in rows]


def _flow_p30(a, b) -> float:
    import cv2

    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)
    return float(np.percentile(np.hypot(fl[..., 0], fl[..., 1]), 30))


def frame_motion(frames: Sequence[Tuple[str, float]], cache: Optional[str] = None) -> np.ndarray:
    """Per frame, the camera's motion since the frame before: the 30th percentile of the dense optical flow (px at
    ``MOTION_WIDTH``); NaN for the first. Cached in ``cache`` (a CSV) when given."""
    import cv2

    if cache and os.path.exists(cache):
        with open(cache) as f:
            vals = [float(r["flow_p30"]) for r in csv.DictReader(f)]
        if len(vals) == len(frames):
            return np.array(vals)
    out, prev = [], None
    for path, _t in frames:
        im = _imread(path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (MOTION_WIDTH, round(im.shape[0] * MOTION_WIDTH / im.shape[1])), interpolation=cv2.INTER_AREA)
        out.append(np.nan if prev is None else _flow_p30(prev, im))
        prev = im
    out = np.array(out)
    if cache:
        with _atomic_write(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "flow_p30"])
            w.writerows([("%.3f" % t, "%.3f" % v) for (_p, t), v in zip(frames, out)])
    return out


def find_stops(times: Sequence[float], motion: Sequence[float], *, still_px: float = STILL_FLOW_PX,
               min_s: float = MIN_STOP_S, gap_s: float = STOP_GAP_S) -> List[Tuple[float, float]]:
    """Stops: runs of still frames (motion < ``still_px``) lasting at least ``min_s``, bridging moves of at most
    ``gap_s``. Returns [(start s, end s)]."""
    runs: List[List[float]] = []
    for t, m in zip(times, motion):
        if not (m < still_px):
            continue
        if runs and t - runs[-1][1] <= gap_s:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [(a, b) for a, b in runs if b - a >= min_s]


# --------------------------------------------------------------------------- the plot's survey photos


def survey_photos(cams: Sequence[Any], min_detail: float = PHOTO_MIN_DETAIL) -> List[Any]:
    """The aligned (posed, calibrated) survey photos to place frames with: those of the sensors at least
    ``min_detail`` as wide as the widest (an AUV's or a video's frames are left out)."""
    from .cameras import can_project

    posed = [c for c in cams if c.sensor is not None and can_project(c, use_orig_coords=False)]
    if not posed:
        return []
    widest = max(c.sensor.width for c in posed)
    return [c for c in posed if c.sensor.width >= min_detail * widest]


def _thumb_job(args):
    src, dst = args
    from PIL import Image

    if not os.path.exists(dst):
        with Image.open(src) as im:
            im.draft("RGB", (im.width // THUMB_REDUCE, im.height // THUMB_REDUCE))
            im = im.convert("RGB")
            with _atomic_write(dst, "wb") as fh:
                im.save(fh, format="JPEG", quality=90)
    return dst


def thumb_name(cam) -> str:
    """The file name of a photo's cached thumbnail: ``<photo folder>__<file name>`` (photos of the same name in
    different folders, e.g. two dives' ``GOPR0001.JPG``, get their own), the file name alone for a photo
    without a folder."""
    import re

    parts = [p for p in re.split(r"[\\/]", getattr(cam, "filepath", None) or "") if p]
    return f"{parts[-2]}__{cam.filename}" if len(parts) >= 2 else cam.filename


def cache_thumbnails(cams: Sequence[Any], folder: str, workers: int = 8,
                     progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Small copies (1/``THUMB_REDUCE``) of the photos of ``cams`` in ``folder`` (read once; a survey photo on a
    spinning disk takes ~0.1-1 s), named by :func:`thumb_name`."""
    os.makedirs(folder, exist_ok=True)
    jobs = [(c.filepath, os.path.join(folder, thumb_name(c))) for c in cams
            if not os.path.exists(os.path.join(folder, thumb_name(c)))]
    if not jobs:
        return
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(_thumb_job, jobs, chunksize=4), 1):
            if progress:
                progress(i, len(jobs))


def north_turn(cam) -> int:
    """The quarter turns (``np.rot90``, counter-clockwise) that bring the plot's +y (world frame) to the top of
    ``cam``'s photo: the image direction closest to +y (up, right, down, left) taken from rays round the centre."""
    cx, cy = cam.sensor.width / 2, cam.sensor.height / 2
    d = max(cam.sensor.width, cam.sensor.height) / 20

    def ray(x, y):
        v = np.asarray(cam.pixel_to_ray(x, y), float)
        return v / np.linalg.norm(v)

    c = ray(cx, cy)
    up, right = (ray(cx, cy - d) - c)[:2], (ray(cx + d, cy) - c)[:2]
    dirs = [up, right, -up, -right]  # what comes to the top after k turns
    return int(np.argmax([v[1] / (np.linalg.norm(v) + 1e-12) for v in dirs]))


# Pixels of an image turned by ``np.rot90(img, k)`` back to the unturned ``w`` x ``h`` image's (one definition,
# imagesync's)
unturn = _isy.unrotate


class PlotMatcher:
    """Places video frames on one plot: its survey photos (turned north-up, cached small), LoFTR, its cloud."""

    def __init__(self, project, thumb_dir: str, *, device: Optional[str] = None, workers: int = 8,
                 progress: Optional[Callable[[int, int], None]] = None):
        import cv2
        import torch

        from .imagesync import LearnedMatcher

        self.cv2, self.torch = cv2, torch
        self.project, self.pcd = project, project.pcd
        self.photos = survey_photos(list(project.cams.data.values()))
        if not self.photos:
            raise ValueError(f"{project.id}: no aligned survey photos")
        self.thumb_dir = thumb_dir
        cache_thumbnails(self.photos, thumb_dir, workers=workers, progress=progress)
        self.C = np.array([np.asarray(c.coords, float) for c in self.photos])
        self.turn = [north_turn(c) for c in self.photos]
        self.seeds = self._seed_grid(SEED_SPACING)
        self.matcher = LearnedMatcher("loftr", device=device)
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self._gray: Dict[Tuple[int, int], np.ndarray] = {}
        # survey photos by file name (as tracks give them, "photo"): a name several photos have is left out
        names: Dict[str, List[int]] = {}
        for i, c in enumerate(self.photos):
            names.setdefault(c.filename, []).append(i)
        self._by_name = {f: ii[0] for f, ii in names.items() if len(ii) == 1}
        if len(self._by_name) < len(names):
            logger.warning(f"{len(names) - len(self._by_name)} survey photo file name(s) occur in more than one "
                           "folder: not looked up by name")

    def _seed_grid(self, spacing: float) -> List[int]:
        order = np.lexsort((self.C[:, 1], self.C[:, 0]))
        picked: List[int] = []
        for i in order:
            if not picked or np.min(np.linalg.norm(self.C[picked, :2] - self.C[i, :2], axis=1)) >= spacing:
                picked.append(int(i))
        return picked

    def _prep(self, im, width):
        cv2 = self.cv2
        h = int(round(im.shape[0] * width / im.shape[1] / 8) * 8)
        return self.clahe.apply(cv2.resize(im, (width // 8 * 8, h), interpolation=cv2.INTER_AREA))

    def photo_gray(self, i: int, width: int) -> np.ndarray:
        """Survey photo ``i``, greyscale, ``width`` px wide (before turning), turned north-up."""
        key = (i, width)
        g = self._gray.get(key)
        if g is None:
            im = _imread(os.path.join(self.thumb_dir, thumb_name(self.photos[i])), self.cv2.IMREAD_GRAYSCALE)
            g = np.ascontiguousarray(np.rot90(self._prep(im, width), self.turn[i]))
            if len(self._gray) > 4000:
                self._gray.clear()
            self._gray[key] = g
        return g

    def _match(self, a, b):
        with self.torch.autocast(self.matcher.device.type, dtype=self.torch.float16,
                                 enabled=self.matcher.device.type == "cuda"):
            return self.matcher.match(a, b)

    def verify(self, frame_g: np.ndarray, photo_g: np.ndarray, turns: Sequence[int]):
        """The most matches consistent with one fundamental matrix over the frame turned by ``turns``:
        (inliers, turn, frame px (turned), photo px)."""
        cv2 = self.cv2
        best = (0, int(turns[0]), None, None)
        for k in turns:
            A, B = self._match(np.ascontiguousarray(np.rot90(frame_g, k)), photo_g)
            if len(A) < 8:
                continue
            _F, mk = cv2.findFundamentalMat(np.float32(A), np.float32(B), cv2.USAC_MAGSAC, 2.0, 0.999, 5000)
            n = 0 if mk is None else int(mk.sum())
            if n > best[0]:
                sel = mk.ravel().astype(bool)
                best = (n, int(k), A[sel], B[sel])
            if n >= GOOD_INLIERS:
                break
        return best

    def place(self, frame_path: str, near: Optional[np.ndarray] = None, turn: Optional[int] = None, *,
              n_photos: int = NEAR_PHOTOS, radius: float = NEAR_RADIUS,
              first: Sequence[str] = ()) -> Dict[str, Any]:
        """Place one frame. ``near``: (x, y) to look round (the ``n_photos`` survey photos nearest it, within
        ``radius`` m; else a seed search over the plot); ``turn``: the frame's last turn, tried first; ``first``:
        survey photos (file names) tried before those. Returns a dict with ``ok``; when placed ``x, y, z`` (world
        frame: where the frame centre meets the cloud), ``inliers``, ``photo``, ``turn``, ``heading_deg`` and
        ``width_m`` when measured (:meth:`_pose`); ``tried``: photos matched."""
        cv2 = self.cv2
        im = _imread(frame_path, cv2.IMREAD_GRAYSCALE)
        turns = [turn] + [k for k in range(4) if k != turn] if turn is not None else list(range(4))
        tried = 0
        if near is None:
            fl = self._prep(im, SEED_FRAME_WIDTH)
            best = (0, None, None)
            for group in ([turns[0]], turns[1:]) if turn is not None else (turns,):
                for i in self.seeds:
                    n, k, _a, _b = self.verify(fl, self.photo_gray(i, SEED_PHOTO_WIDTH), group)
                    tried += 1
                    if n > best[0]:
                        best = (n, k, i)
                if best[0] >= SEED_MIN_INLIERS:
                    break
            if best[0] < SEED_MIN_INLIERS:
                return {"ok": False, "seed_inliers": best[0], "tried": tried}
            near, turns = self.C[best[2], :2], [best[1]] + [k for k in range(4) if k != best[1]]
        d = np.linalg.norm(self.C[:, :2] - np.asarray(near, float)[:2], axis=1)
        cand = [int(i) for i in np.argsort(d)[:n_photos] if d[i] < radius]
        pre = [self._by_name[f] for f in first if f in self._by_name]
        cand = pre + [i for i in cand if i not in pre]
        fg = self._prep(im, MATCH_FRAME_WIDTH)
        best = (0,)
        for i in cand:
            n, k, A, B = self.verify(fg, self.photo_gray(i, MATCH_PHOTO_WIDTH), turns)
            tried += 1
            if n > best[0]:
                best = (n, k, A, B, i)
                turns = [k] + [t for t in range(4) if t != k]
            if n >= GOOD_INLIERS:
                break
        if best[0] < MIN_INLIERS:
            return {"ok": False, "inliers": best[0], "tried": tried}
        n, k, A, B, i = best
        cam = self.photos[i]
        fh, fw = fg.shape
        Af = unturn(A, k, fw, fh)
        pg = self.photo_gray(i, MATCH_PHOTO_WIDTH)  # turned north-up
        ph, pw = (pg.shape[1], pg.shape[0]) if self.turn[i] % 2 else pg.shape
        Bf = _isy.to_sensor_px(unturn(B, self.turn[i], pw, ph), cam.sensor.width / pw)
        ctr = np.array([fw / 2, fh / 2])
        nearest = np.argsort(np.linalg.norm(Af - ctr, axis=1))[:CENTRE_MATCHES]
        P, pose = None, {}
        H, _m = cv2.findHomography(np.float32(Af[nearest]), np.float32(Bf[nearest]), cv2.USAC_MAGSAC, 30.0)
        if H is not None:
            px = cv2.perspectiveTransform(np.float32([[ctr]]), H)[0, 0]
            if 0 <= px[0] < cam.sensor.width and 0 <= px[1] < cam.sensor.height:
                P = self._to_cloud(cam, px)
            pose = self._pose(cam, H, Af[nearest].mean(0), fw, fh)
        if P is None:
            P = self._to_cloud(cam, Bf[nearest[0]])
        if P is None:
            return {"ok": False, "inliers": n, "tried": tried, "note": "no cloud hit"}
        return {"ok": True, "x": float(P[0]), "y": float(P[1]), "z": float(P[2]), "inliers": int(n),
                "photo": cam.filename, "turn": int(k), "tried": tried, **pose}

    def _pose(self, cam, H, base, fw, fh) -> Dict[str, float]:
        """The frame's heading and footprint: frame pixel ``base`` (the middle of the matches used: inside the
        photo) and points a little above and right of it (``POSE_OFFSET``) mapped into the photo (``H``) and onto
        the cloud: ``heading_deg`` where the top of the frame points (degrees from the world's +y, clockwise: a
        compass bearing when +y is north), ``width_m`` how much reef the frame spans across (m, along the reef).
        Empty when the two directions disagree (a mismatch or too steep a reef)."""
        d = POSE_OFFSET * fh
        base = np.asarray(base, float)
        pts = self.cv2.perspectiveTransform(np.float32([[base, base + (0, -d), base + (d, 0)]]), H)[0]
        P, U, R = (self._to_cloud(cam, q) for q in pts)
        if P is None or U is None or R is None:
            return {}
        u, r = (U - P)[:2], (R - P)[:2]
        nu, nr = float(np.linalg.norm(u)), float(np.linalg.norm(r))
        if nu < 1e-4 or nr < 1e-4:
            return {}
        a, b = u / nu, np.array([-r[1], r[0]]) / nr  # (right turned a quarter anticlockwise: up, seen from above)
        l3 = (float(np.linalg.norm(U - P)) + float(np.linalg.norm(R - P))) / 2
        if float(a @ b) < 0.8 or not 0.5 < nu / nr < 2.0:
            return {}
        up = a + b
        return {"heading_deg": round(float(np.degrees(np.arctan2(up[0], up[1]))) % 360.0, 1),
                "width_m": round(l3 / d * fw, 3)}

    def _to_cloud(self, cam, px):
        from .imagesync import to_cloud

        return to_cloud(cam, px, self.pcd, radius=RAY_RADIUS)


# --------------------------------------------------------------------------- tracking

POSE_KEYS = ("heading_deg", "width_m")


def _take(d: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def track_frames(matcher: PlotMatcher, frames: Sequence[Tuple[str, float]], motion: Sequence[float], *,
                 progress: Optional[Callable[[int, int, int], None]] = None) -> List[Dict[str, Any]]:
    """Place ``frames`` ([(path, t)]) of one video on ``matcher``'s plot. Returns one row per frame:
    ``t_s``, ``how`` (seed / track / still / lost / skipped) and, placed, ``x, y, z, inliers, photo, turn``."""
    rows: List[Dict[str, Any]] = []
    last: Optional[Dict[str, Any]] = None
    last_seed, last_check, turn = -1e9, -1e9, None
    placed = 0
    for j, ((path, t), m) in enumerate(zip(frames, motion)):
        still = bool(m < STILL_FLOW_PX)
        row: Dict[str, Any] = {"t_s": round(t, 3), "frame": os.path.basename(path), "flow": round(float(m), 3)
                               if np.isfinite(m) else ""}
        recent = last is not None and t - last["t"] <= TRACK_MEMORY_S
        if recent and still and t - last_check < STILL_RECHECK_S and t - last["t"] < 30:
            row.update(_take(last, ("x", "y", "z", "photo") + POSE_KEYS), how="still")
        else:
            r = None
            if recent:
                r = matcher.place(path, near=np.array([last["x"], last["y"]]), turn=last["turn"])
                row["how"] = "track"
            if (r is None or not r["ok"]) and still and t - last_seed >= RESEED_S:
                last_seed = t
                r = matcher.place(path, turn=turn)
                row["how"] = "seed"
            if r is not None and r["ok"]:
                row.update(_take(r, ("x", "y", "z", "inliers", "photo", "turn") + POSE_KEYS))
                last, last_check, turn = dict(r, t=t), t, r["turn"]
                placed += 1
            else:
                row["how"] = "lost" if r is not None else "skipped"
                if r is not None and r.get("inliers"):
                    row["inliers"] = r["inliers"]
        rows.append(row)
        if progress:
            progress(j + 1, len(frames), placed)
    return rows


def backfill(matcher: PlotMatcher, frames: Sequence[Tuple[str, float]], motion: Sequence[float],
             rows: List[Dict[str, Any]]) -> int:
    """The forward pass misses the frames before a seed (a seed is only tried at still frames, every few s):
    walk back from each placed frame through the unplaced ones before it, tracking as forwards (still frames
    take the position, checked every ``STILL_RECHECK_S``). Changes ``rows`` in place; returns frames placed."""
    last: Optional[Dict[str, Any]] = None
    last_check = 1e18
    placed = 0
    for j in range(len(rows) - 1, -1, -1):
        row, (path, t) = rows[j], frames[j]
        if "x" in row:
            if row.get("how") in ("seed", "track", "back"):
                last = _take(row, ("x", "y", "z", "photo") + POSE_KEYS)
                last.update(t=t, turn=row.get("turn"))
                last_check = t
            continue
        if last is None or last["t"] - t > TRACK_MEMORY_S:
            continue
        nxt_still = j + 1 < len(motion) and bool(motion[j + 1] < STILL_FLOW_PX)  # (frame j+1 moved since j?)
        if nxt_still and last_check - t < STILL_RECHECK_S and last["t"] - t < 30:
            row.update(_take(last, ("x", "y", "z", "photo") + POSE_KEYS), how="still")
            continue
        r = matcher.place(path, near=np.array([last["x"], last["y"]]), turn=last.get("turn"))
        if r["ok"]:
            row.update(_take(r, ("x", "y", "z", "inliers", "photo", "turn") + POSE_KEYS), how="back")
            last = dict(r, t=t)
            last_check = t
            placed += 1
    return placed


def fill_stops(matcher: PlotMatcher, frames: Sequence[Tuple[str, float]], rows: List[Dict[str, Any]],
               stops: Sequence[Tuple[float, float]], tries: int = 4) -> int:
    """Stops with no frame placed: the diver works along the plot, so look between the positions placed just
    before and after the stop (a wider search: 15 photos within 4 m of each, and of their middle), on up to
    ``tries`` frames of the stop. A frame placed this way is ``how = "fill"``. Returns stops filled."""
    times = np.array([t for _p, t in frames])
    placed = [(r["t_s"], r) for r in rows if r.get("how") in ("seed", "track", "back", "fill")]
    filled = 0
    for t0, t1 in stops:
        idx = [j for j in range(len(rows)) if t0 - 1 <= times[j] <= t1 + 1]
        if any("x" in rows[j] for j in idx) or not idx:
            continue
        before = [r for t, r in placed if t < t0][-1:]
        after = [r for t, r in placed if t > t1][:1]
        guesses = [np.array([r["x"], r["y"]]) for r in before + after]
        if len(guesses) == 2:
            guesses.insert(1, (guesses[0] + guesses[1]) / 2)
        if not guesses:
            continue
        turn = (before or after)[0].get("turn")
        hit = None
        for j in [idx[int(i)] for i in np.linspace(0, len(idx) - 1, min(tries, len(idx)))]:
            for g in guesses:
                r = matcher.place(frames[j][0], near=g, turn=turn, n_photos=15, radius=4.0)
                if r["ok"]:
                    hit = r
                    break
            if hit:
                break
        if hit:
            for j in idx:
                rows[j].update(_take(hit, ("x", "y", "z", "photo") + POSE_KEYS), how="still")
            rows[idx[0]].update(how="fill", inliers=hit["inliers"], turn=hit["turn"])
            placed.append((rows[idx[0]]["t_s"], rows[idx[0]]))
            placed.sort(key=lambda p: p[0])
            filled += 1
    return filled


def add_poses(matcher: PlotMatcher, frames: Sequence[Tuple[str, float]], rows: List[Dict[str, Any]],
              stops: Sequence[Tuple[float, float]], per_stop: int = POSE_FRAMES) -> int:
    """Headings for stops whose frames have none (tracks made before :meth:`PlotMatcher.place` returned them): up
    to ``per_stop`` of the stop's matched frames placed again round their position (kept when within 0.5 m of
    it). A stop is tried once (``pose_tried``). Changes ``rows`` in place; returns stops given a heading."""
    done = 0
    for t0, t1 in stops:
        inside = [j for j, r in enumerate(rows) if t0 - 1 <= r["t_s"] <= t1 + 1 and "x" in r]
        matched = [j for j in inside if rows[j].get("how") in MATCHED]
        if not matched or any("heading_deg" in rows[j] or rows[j].get("pose_tried") for j in inside):
            continue
        got = []
        for j in [matched[int(i)] for i in np.linspace(0, len(matched) - 1, min(per_stop, len(matched)))]:
            r = rows[j]
            r["pose_tried"] = 1
            res = matcher.place(frames[j][0], near=np.array([r["x"], r["y"]]), turn=r.get("turn"), n_photos=3,
                                radius=1.5, first=[r["photo"]] if r.get("photo") else ())
            if res["ok"] and "heading_deg" in res and np.hypot(res["x"] - r["x"], res["y"] - r["y"]) < 0.5:
                r.update(_take(res, POSE_KEYS))
                got.append(r)
        done += bool(got)
    return done


def circular_mean_deg(angles: Sequence[float]) -> Optional[float]:
    """The mean direction of ``angles`` (degrees), None for none (or directions that cancel out)."""
    a = np.radians(np.asarray([x for x in angles if x is not None and np.isfinite(x)], float))
    if not len(a):
        return None
    s, c = float(np.sin(a).mean()), float(np.cos(a).mean())
    if np.hypot(s, c) < 1e-3:
        return None
    return float(np.degrees(np.arctan2(s, c))) % 360.0


def despike(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``rows`` without the positions that disagree with their neighbours: a frame placed by matching more than
    ``SPIKE_M`` (plus ``MAX_SPEED`` x the time apart) from the median of the frames placed within
    ``SPIKE_WINDOW_S`` (at least 2 of them) is a mismatch (a reef 3-D enough to map the frame centre wrongly):
    its position is left out (``how`` = "outlier"), as are the still frames that copied it. Returns copies."""
    out = [dict(r) for r in rows]
    placed = [i for i, r in enumerate(out) if r.get("how") in ("seed", "track", "back", "fill")]
    T = np.array([out[i]["t_s"] for i in placed])
    P = np.array([[out[i]["x"], out[i]["y"]] for i in placed]) if placed else np.zeros((0, 2))
    bad = set()
    for n, i in enumerate(placed):
        near = [m for m in np.nonzero(np.abs(T - T[n]) <= SPIKE_WINDOW_S)[0] if m != n]
        if len(near) < 2:
            continue
        med = np.median(P[near], 0)
        dt = float(np.median(np.abs(T[near] - T[n])))
        if np.linalg.norm(P[n] - med) > SPIKE_M + MAX_SPEED * dt:
            bad.add(i)
    copying = False
    for i, r in enumerate(out):
        if i in bad:
            copying = True
        elif r.get("how") != "still":
            copying = False
        if i in bad or (copying and r.get("how") == "still"):
            for k in ("x", "y", "z"):
                r.pop(k, None)
            r["how"] = "outlier" if i in bad else "lost"
    return out


def probe(matcher: PlotMatcher, frames: Sequence[Tuple[str, float]], motion: Sequence[float],
          n: int = PROBE_FRAMES) -> int:
    """How many of ``n`` frames (still ones, spread over the video) a seed search places on the plot (it stops at
    2)."""
    still = [i for i, m in enumerate(motion) if m < STILL_FLOW_PX]
    pool = still if len(still) >= n else list(range(len(frames)))
    pick = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(n, len(pool)))] if pool else []
    hits = 0
    for i in pick:
        if matcher.place(frames[i][0])["ok"]:
            hits += 1
            if hits >= 2:  # (enough to track it; a video may be on the plot for only part of its length)
                break
    return hits


def stop_rows(video: str, plot_id: str, rows: Sequence[Dict[str, Any]], stops: Sequence[Tuple[float, float]]
              ) -> List[Dict[str, Any]]:
    """Per stop of ``video``: its times and the median position of the frames placed by matching within it (else
    of those carried while still), ``spread_m`` = their median distance from it (xy), ``placed`` = "matched" (else
    ""), ``heading_deg`` / ``width_m`` the frames' mean heading and median footprint (see
    :meth:`PlotMatcher._pose`)."""
    out = []
    for i, (t0, t1) in enumerate(stops, 1):
        inside = [r for r in rows if t0 - 1 <= r["t_s"] <= t1 + 1 and "x" in r]
        matched = [r for r in inside if r.get("how") in ("seed", "track", "back", "fill")]
        use = matched or inside
        row = {"video": video, "plot": plot_id, "stop": i, "start": _timecode(t0), "t_start_s": round(t0, 1),
               "t_end_s": round(t1, 1), "duration_s": round(t1 - t0, 1), "n_matched": len(matched),
               "placed": "", "x": "", "y": "", "z": "", "spread_m": "", "heading_deg": "", "width_m": ""}
        if use:
            P = np.array([[r["x"], r["y"], r["z"]] for r in use], float)
            med = np.median(P, 0)
            row.update({"x": round(float(med[0]), 3), "y": round(float(med[1]), 3), "z": round(float(med[2]), 3),
                        "spread_m": round(float(np.median(np.linalg.norm(P[:, :2] - med[:2], axis=1))), 3),
                        "placed": "matched"})
            posed = [r for r in inside if "heading_deg" in r]
            h = circular_mean_deg([r["heading_deg"] for r in posed])
            if h is not None:
                row.update(heading_deg=round(h, 1), width_m=round(float(np.median([r["width_m"] for r in posed])), 2))
        out.append(row)
    return out


def _start_time(info: Dict[str, Any]) -> Optional[float]:
    """When a video started (POSIX s, from the creation time its camera wrote), None if unknown."""
    ct = info.get("creation_time")
    if not ct:
        return None
    try:
        return _dt.datetime.fromisoformat(str(ct).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def video_starts(vids: Sequence[str], infos: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Each video's start (s) on one clock, to order a camera's videos: the creation times; without them (for any
    video) the videos of a folder by name, one after the other."""
    t = {v: _start_time(infos[v]) for v in vids}
    if all(x is not None for x in t.values()):
        return t
    out, end = {}, {}
    for v in sorted(vids, key=lambda v: (os.path.dirname(v), os.path.basename(v))):
        d = os.path.dirname(v)
        out[v] = end.get(d, 0.0)
        end[d] = out[v] + float(infos[v]["duration"]) + 1.0
    return out


def sequence_stops(stops: List[Dict[str, Any]], tracks: Dict[Tuple[str, str], List[Dict[str, Any]]],
                   vids: Sequence[str], infos: Dict[str, Dict[str, Any]]) -> None:
    """The videos of one camera (one folder) over one plot are one *sequence* (a dive's samples there, in
    order, over a GoPro's split files and restarts). Sets on each stop (in place) ``sequence`` (the folder's
    name), ``plot`` (an unplaced stop: the plot its video has most frames placed on) and ``seq_stop`` (its number
    in the sequence, by time). An unplaced stop of a video on a plot gets an estimated position (``placed`` =
    "estimated", ``est_gap_s`` = s to the nearest placed frame): interpolated in time between the sequence's placed
    frames before and after it (at most ``EST_MAX_GAP_S`` apart), else the nearest placed frame on one side within
    ``EST_ONE_SIDE_S``."""
    start = video_starts(vids, infos)
    plot_of = {}
    for v in vids:
        n = {pid: sum("x" in r for r in rows) for (vv, pid), rows in tracks.items() if vv == v}
        if n and max(n.values()) > 0:
            plot_of[v] = max(n, key=n.get)
    frames: Dict[Tuple[str, str], List[Tuple[float, float, float, float]]] = {}
    for (v, pid), rows in tracks.items():
        frames.setdefault((os.path.dirname(v), pid), []).extend(
            (start[v] + r["t_s"], r["x"], r["y"], r["z"]) for r in rows if "x" in r)
    seq = {k: (np.array([f[0] for f in sorted(fs)]), np.array([f[1:] for f in sorted(fs)]))
           for k, fs in frames.items()}
    for s in stops:
        v = s["video_path"]
        folder = os.path.dirname(v)
        s["sequence"] = os.path.basename(folder)
        if not s.get("plot"):
            s["plot"] = plot_of.get(v, "")
        s.setdefault("est_gap_s", "")
        if s.get("x") != "" or not s["plot"] or (folder, s["plot"]) not in seq:
            continue
        T, X = seq[(folder, s["plot"])]
        a0, a1 = start[v] + s["t_start_s"], start[v] + s["t_end_s"]
        i, j = int(np.searchsorted(T, a0)), int(np.searchsorted(T, a1, "right"))
        b = i - 1 if i > 0 else None
        a = j if j < len(T) else None
        if b is not None and a is not None and T[a] - T[b] <= EST_MAX_GAP_S:
            u = float(np.clip(((a0 + a1) / 2 - T[b]) / max(T[a] - T[b], 1e-6), 0, 1))
            P, gap = X[b] + u * (X[a] - X[b]), min(a0 - T[b], T[a] - a1)
        elif b is not None and a0 - T[b] <= EST_ONE_SIDE_S:
            P, gap = X[b], a0 - T[b]
        elif a is not None and T[a] - a1 <= EST_ONE_SIDE_S:
            P, gap = X[a], T[a] - a1
        else:
            continue
        s.update(placed="estimated", x=round(float(P[0]), 3), y=round(float(P[1]), 3), z=round(float(P[2]), 3),
                 est_gap_s=round(float(gap), 1))
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for s in stops:
        groups.setdefault((s["sequence"], s["plot"]), []).append(s)
    for g in groups.values():
        g.sort(key=lambda s: start[s["video_path"]] + s["t_start_s"])
        for n, s in enumerate(g, 1):
            s["seq_stop"] = n


def pose_stops(stops: List[Dict[str, Any]], tracks: Dict[Tuple[str, str], List[Dict[str, Any]]],
               frames: Dict[str, List[Tuple[str, float]]], projects: Sequence[Any], keys: Dict[str, str],
               infos: Dict[str, Dict[str, Any]], work_dir: str, *, redo: bool = False, device: Optional[str] = None,
               workers: int = 8, log: Callable[[str], None] = print) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The camera pose of each stop placed by matching (:mod:`substrata.videopose`): correspondences of the stop's
    first and last matched frame with the plot (kept in ``<work_dir>/poses``), one lens per camera setting
    (calibrated from all of its frames, kept), each frame's pose; the stop's is its frame with most inliers. Sets on
    the stops ``cam_x, cam_y, cam_z`` (world), ``cam_qx .. cam_qw`` (camera -> world; x right, y down, z forward)
    and ``pose_px`` (median reprojection error). Returns (lenses {key: calibration}, video -> lens key)."""
    from . import videopose as vp

    pdir = os.path.join(work_dir, "poses")
    by_plot = {pr.id: pr for pr in projects}
    fpath = {v: {os.path.basename(path): path for path, _t in frames[v]} for v in frames}
    lens_of = {v: vp.lens_key(infos[v], vp.video_lens(v)) for v in frames}
    jobs = []
    for s in stops:
        rows = tracks.get((s["video_path"], s["plot"])) if s.get("placed") == "matched" else None
        if not rows:
            continue
        inside = [r for r in rows if s["t_start_s"] - 1 <= r["t_s"] <= s["t_end_s"] + 1 and r.get("how") in MATCHED
                  and "x" in r and r.get("photo")]
        for r in [inside[0], inside[-1]] if len(inside) > 1 else inside:
            jobs.append((s, s["video_path"], s["plot"], r))
    jobs.sort(key=lambda j: j[2])
    corr: Dict[Tuple[str, str], Optional[Dict[str, np.ndarray]]] = {}
    cur, finder, t0, last = None, None, time.time(), 0.0
    for n, (s, v, pid, r) in enumerate(jobs):
        cp = os.path.join(pdir, pid, keys[v], os.path.splitext(r["frame"])[0] + ".npz")
        if os.path.exists(cp) and not redo:
            with np.load(cp) as z:
                corr[(v, r["frame"])] = {k: z[k] for k in z.files} if z.files else None
            continue
        if cur != pid:
            cur, finder = pid, None  # (one plot's matcher at a time)
            finder = vp.Correspondences(PlotMatcher(by_plot[pid], os.path.join(work_dir, "plots", pid, "thumbs"),
                                                    device=device, workers=workers))
        path = fpath.get(v, {}).get(r["frame"])
        c = finder.frame(path, (r["x"], r["y"]), r.get("photo")) if path else None
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        np.savez(cp, **(c or {}))  # (empty: none)
        corr[(v, r["frame"])] = c
        if time.time() - last > 30:
            last = time.time()
            log(f"  poses: correspondences of {n + 1}/{len(jobs)} stop frames ({time.time() - t0:.0f} s)")
    finder = None
    lenses: Dict[str, Any] = {}
    for key in sorted(set(lens_of.values())):
        fr = [c for (v, _f), c in corr.items() if lens_of[v] == key and c is not None]
        lp = os.path.join(pdir, f"lens_{key}.json")
        cal = vp.load_json(lp)
        if redo or cal is None or cal.get("frames", 0) < 0.8 * len(fr):
            if not fr:
                continue
            size = tuple(int(x) for x in fr[0]["size"])
            log(f"lens {key}: calibrating from {len(fr)} frames ...")
            cal = vp.calibrate(fr, size, log=log)
            if cal is None:
                continue
            vp.save_json(lp, cal)
        lenses[key] = cal
        log(f"lens {key}: {cal['posed']}/{cal['frames']} frames posed, median error {cal['px']:.1f} px; field of view "
            f"{cal['fov']['across']:.0f} x {cal['fov']['down']:.0f} deg")
    posed = 0
    by_stop: Dict[int, List[Dict[str, Any]]] = {}
    for s, v, _pid, r in jobs:
        c, cal = corr.get((v, r["frame"])), lenses.get(lens_of[v])
        if c is None or cal is None:
            continue
        fp = vp.frame_pose(cal["params"], c["uv"], c["xyz"])
        if fp:
            by_stop.setdefault(id(s), []).append(vp.pose_record(fp))
    for s in stops:
        sp = vp.stop_pose(by_stop.get(id(s), []))
        if sp:
            posed += 1
            s.update(cam_x=round(sp["C"][0], 3), cam_y=round(sp["C"][1], 3), cam_z=round(sp["C"][2], 3),
                     cam_qx=round(sp["q"][0], 5), cam_qy=round(sp["q"][1], 5), cam_qz=round(sp["q"][2], 5),
                     cam_qw=round(sp["q"][3], 5), pose_px=sp["px"])
    log(f"poses: {posed} of {sum(1 for s in stops if s.get('placed') == 'matched')} stops placed by matching "
        f"({time.time() - t0:.0f} s)")
    return lenses, {keys[v]: k for v, k in lens_of.items()}


def _timecode(t: float) -> str:
    h, rem = divmod(float(t), 3600)
    m, s = divmod(rem, 60)
    return ("%d:%02d:%04.1f" % (h, m, s)) if h else ("%d:%04.1f" % (m, s))


# --------------------------------------------------------------------------- outputs


def plot_map(pcd, world_transform: Optional[np.ndarray] = None, resolution: float = MAP_RESOLUTION
             ) -> Tuple[np.ndarray, Dict[str, float]]:
    """A top-down image of the cloud (the top-most point per pixel, world frame) and its geometry
    (``x0``, ``y_top``, ``res``): pixel (col, row) is at x = x0 + (col + .5) res, y = y_top - (row + .5) res."""
    import cv2

    P = np.asarray(pcd.points, float)
    C = np.asarray(pcd.colors) if getattr(pcd, "colors", None) is not None else np.full((len(P), 3), 0.6)
    C = (np.asarray(C, float) * (255 if np.max(C) <= 1 else 1)).astype(np.uint8)
    x0, x1 = np.percentile(P[:, 0], [0.01, 99.99])
    y0, y1 = np.percentile(P[:, 1], [0.01, 99.99])
    w, h = int((x1 - x0) / resolution) + 1, int((y1 - y0) / resolution) + 1
    col = ((P[:, 0] - x0) / resolution).astype(np.int64)
    row = ((y1 - P[:, 1]) / resolution).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    order = np.argsort(P[ok, 2])
    img = np.zeros((h * w, 3), np.uint8)
    img[(row[ok] * w + col[ok])[order]] = C[ok][order]
    img = img.reshape(h, w, 3)
    empty = img.sum(2) == 0
    img[empty] = cv2.dilate(img, np.ones((3, 3), np.uint8))[empty]
    return img, {"x0": float(x0), "y_top": float(y1), "res": resolution}


COLOURS = [(0, 200, 255), (255, 140, 0), (80, 255, 80), (255, 80, 220), (255, 255, 80), (80, 160, 255)]


def smooth_path(P: np.ndarray, per: int = 12) -> np.ndarray:
    """A smooth line through points ``P`` (N x d) in order (centripetal Catmull-Rom, ``per`` points a segment)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        for t in np.linspace(t1, t2, per, endpoint=i == len(ext) - 3):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.array(out)


def draw_map(img: np.ndarray, geo: Dict[str, float], tracks: Dict[str, List[Dict[str, Any]]],
             stops: Sequence[Dict[str, Any]], path: str, title: str = "",
             colours: Optional[Dict[str, int]] = None) -> None:
    """The plot map: each sequence (stops' ``sequence``; ``colours``: sequence -> colour index, default by
    order) as a smooth line through its stops in order, the stops as circles numbered ``seq_stop`` (estimated
    ones thinner, ``~n``), the videos' placed frames (``tracks``: video -> rows) faint; legend and scale bar;
    written to ``path`` (JPEG)."""
    import cv2

    can = (img.astype(np.float32) * 0.7).astype(np.uint8)
    res = geo["res"]
    lw = max(2, int(0.012 / res))
    seqs = list(dict.fromkeys(s.get("sequence", "") for s in stops))
    col_i = colours or {q: i for i, q in enumerate(seqs)}
    vid_seq = {s["video"]: s.get("sequence", "") for s in stops}

    def colour(q):
        return COLOURS[col_i.get(q, 0) % len(COLOURS)]

    def px(x, y):
        return int((x - geo["x0"]) / res), int((geo["y_top"] - y) / res)

    faint = can.copy()
    for v, rows in tracks.items():
        prev = None
        for r in rows:
            if "x" not in r or r.get("how") == "still":
                continue
            p = px(r["x"], r["y"])
            if prev is not None and r["t_s"] - prev[1] <= 15:
                cv2.line(faint, prev[0], p, colour(vid_seq.get(v, "")), max(1, lw // 2), cv2.LINE_AA)
            prev = (p, r["t_s"])
    can = cv2.addWeighted(faint, 0.35, can, 0.65, 0)
    pos = [s for s in stops if s["x"] != ""]
    for q in seqs:
        st = sorted((s for s in pos if s.get("sequence", "") == q), key=lambda s: s.get("seq_stop", 0))
        if len(st) > 1:
            line = smooth_path(np.array([[s["x"], s["y"]] for s in st]))
            pts = np.array([px(x, y) for x, y in line], np.int32)
            cv2.polylines(can, [pts], False, (0, 0, 0), lw + 3, cv2.LINE_AA)
            cv2.polylines(can, [pts], False, colour(q), lw, cv2.LINE_AA)
    fs = 0.004 / res * 1.2
    for s in pos:
        est = s.get("placed") == "estimated"
        p = px(s["x"], s["y"])
        rad = int(0.15 / res)
        c = colour(s.get("sequence", ""))
        if not est:
            cv2.circle(can, p, rad, (255, 255, 255), lw + 2, cv2.LINE_AA)
        cv2.circle(can, p, rad, c, lw if not est else max(1, lw // 2), cv2.LINE_AA)
        label = ("~" if est else "") + str(s.get("seq_stop", s["stop"]))
        cv2.putText(can, label, (p[0] + rad + 4, p[1] + rad // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    max(2, lw) + 3, cv2.LINE_AA)
        cv2.putText(can, label, (p[0] + rad + 4, p[1] + rad // 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255) if not est else c, max(2, lw), cv2.LINE_AA)
    pad = int(0.6 / res) + 40 * len(seqs)
    out = np.zeros((can.shape[0] + pad, can.shape[1], 3), np.uint8)
    out[: can.shape[0]] = can
    y = can.shape[0] + int(0.25 / res)
    one_m = int(1 / res)
    cv2.line(out, (40, y), (40 + one_m, y), (255, 255, 255), lw + 2)
    cv2.putText(out, "1 m", (50 + one_m, y + 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (80 + 2 * one_m, y + 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    for qi, q in enumerate(seqs):
        yy = y + int(0.3 / res) + 40 * qi + int(fs * 30)
        n = sum(1 for s in stops if s.get("sequence", "") == q)
        nm = sum(1 for s in stops if s.get("sequence", "") == q and s.get("placed") == "matched")
        ne = sum(1 for s in stops if s.get("sequence", "") == q and s.get("placed") == "estimated")
        cv2.circle(out, (60, yy - 8), 12, colour(q), -1)
        cv2.putText(out, f"{q}: {n} stops ({nm} matched, {ne} estimated ~)", (90, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 2, cv2.LINE_AA)
    if out.shape[1] > 6000:
        sc = 6000 / out.shape[1]
        out = cv2.resize(out, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        with _atomic_write(path, "wb") as fh:
            fh.write(buf.tobytes())


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    """Write ``rows`` (dicts) as a CSV, atomically (:func:`substrata.fileio.atomic_write`)."""
    fields = list(fields or (list(dict.fromkeys(k for r in rows for k in r)) if rows else []))
    with _atomic_write(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


STOP_FIELDS = ["sequence", "diver", "plot", "seq_stop", "video", "stop", "start", "t_start_s", "t_end_s", "duration_s", "placed",
               "x", "y", "z", "n_matched", "spread_m", "est_gap_s", "heading_deg", "width_m", "cam_x", "cam_y", "cam_z",
               "cam_qx", "cam_qy", "cam_qz", "cam_qw", "pose_px", "annotation_id", "video_path"]
TRACK_FIELDS = ["video", "plot", "frame", "t_s", "how", "x", "y", "z", "inliers", "photo", "heading_deg", "width_m",
                "flow"]

README = """substrata track-video: where GRAV videos were filmed over the plots, and where they stood still
=====================================================================================

stops.csv   one row per stop of a video: the camera (pole) stood still for {min_stop:.0f} s or more.
            sequence     the camera's folder: its videos over one plot are one sequence (a dive's samples
                         there, in order); seq_stop the stop's number in it (by time, over the videos)
            diver        the diver the camera filmed (named in substrata-explore's video player: labels.json)
            plot         the plot of the stop
            video        the video's name (file stem); video_path its file; stop its number in the video
            start        time into the video where the stop starts (m:ss.s), also t_start_s / t_end_s (s)
            duration_s   how long it stood still
            placed       matched: frames of the stop were placed by image matching; x, y, z where the middle
                         of the frame meets the reef, in the plot's world frame (m; z = depth)
                         estimated: none were; x, y, z interpolated in time between the sequence's placed
                         frames before and after it (est_gap_s: s to the nearest; a rough position)
                         empty: not placed (the video is on no plot, or nothing placed near in time)
            n_matched    frames of the stop placed by image matching; spread_m their median distance from x, y
            heading_deg  where the top of the video frame points (degrees clockwise from the world's +y, i.e.
                         a compass bearing when +y is north); width_m how much reef the frame spans across
            cam_x, cam_y, cam_z   where the camera was (world frame), cam_qx .. cam_qw its orientation (a
                         quaternion, camera -> world; the camera's x right, y down, z forward): the stop's camera
                         pose (videotrack.json: the lens), pose_px its median reprojection error (px)
            annotation_id   the stop's point in the plot's annotation set <id>.annos/<id>_video_stops.csv
track.csv   every frame (one a second) placed on a plot: time (s), how (seed / track / back / fill: by image
            matching; still: position kept while the camera did not move), x, y, z, inliers (matches), photo
            (survey photo), heading_deg / width_m (as for the stops; when they could be measured)
videos.csv  the videos: n (their number in substrata-explore's player), sequence, duration, creation time (as
            the camera wrote it), plots, frames placed
<plot>_map.jpg   the plot from above: each sequence (one colour per camera) as a line through its stops in
            order, the stops numbered seq_stop (estimated ones with ~), the placed frames faint
videotrack.json  the same for substrata-explore (substrata-explore --videos <this folder>), with each camera
            setting's lens calibration (lenses: substrata.videopose's model, its field of view)
proxies/    H.264 copies of the videos for the browser (if made with --proxy)

The position is that of the frame centre, which is near (not exactly at) the pole's foot: the sampled corals are
round it, typically within a metre. Times are seconds into each video file (a GoPro splits long recordings into
several files).
"""


ANN_STOP_FIELDS = ["sequence", "diver", "seq_stop", "video", "stop", "t_start_s", "t_end_s", "duration_s",
                   "is_extrapolated", "n_matched", "spread_m", "est_gap_s", "heading_deg"]
# (is_extrapolated: True for an estimated position, as substrata's intercepts and camera targets mark theirs)
_ANN_BASE = ["id", "orig_x", "orig_y", "orig_z", "label", "label_conf", "world_x", "world_y", "world_z"]


def write_stops_annotations(project, stops: Sequence[Dict[str, Any]], track_dir: str) -> Optional[str]:
    """The stops of ``project`` with a position (matched or estimated) as a point set
    ``<id>.annos/<id>_video_stops.csv`` (with its sidecar), to move onto the sampled corals and label in
    substrata-explore. A set that exists (it may hold the annotator's work) is only added to: a stop never written
    to it before becomes a new point, and the stop columns of its points (``ANN_STOP_FIELDS`` but video and stop:
    sequence, seq_stop, is_extrapolated (an estimated position), ...) are brought up to date; its points are never moved, relabelled, removed or
    added again once removed (the sidecar lists the stops written). The file is backed up first (``backup/``).
    Sets ``annotation_id`` in ``stops``. Returns the path when written, else None."""
    from .sidecars import read_annotation_sidecar, write_annotation_sidecar

    folder = os.path.join(project.path, f"{project.id}.annos")
    path = os.path.join(folder, f"{project.id}_video_stops.csv")
    exists = os.path.exists(path)
    W = np.asarray(project.world_transform, float)
    Wi = np.linalg.inv(W)
    rows: List[Dict[str, Any]] = []
    fields = list(_ANN_BASE)
    last, written = -1, set()
    if exists:
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or fields)
            rows = list(rd)
        side = read_annotation_sidecar(path) or {}
        try:
            last = int(side.get("last_highest_id", -1))
        except (TypeError, ValueError):
            last = -1
        written = set((side.get("video_stops") or {}).get("written") or [])
        pre = f"{project.id}_"
        for r in rows:
            i = str(r.get("id", ""))
            if i.startswith(pre) and i[len(pre):].isdigit():
                last = max(last, int(i[len(pre):]))
    index = {(r.get("video"), str(r.get("stop"))): r for r in rows if r.get("video") and r.get("stop")}
    written |= {f"{v}:{st}" for v, st in index}
    added = changed = 0
    for s in stops:
        k = (s["video"], str(s["stop"]))
        r = index.get(k)
        info = {f: s.get(f, "") for f in ANN_STOP_FIELDS}
        info["is_extrapolated"] = s.get("placed") == "estimated"
        if r is None and s["x"] != "" and f"{k[0]}:{k[1]}" not in written:
            last += 1
            w = np.array([s["x"], s["y"], s["z"], 1.0])
            o = Wi @ w
            r = {"id": f"{project.id}_{last}", "orig_x": o[0], "orig_y": o[1], "orig_z": o[2], "label": "",
                 "label_conf": "", "world_x": w[0], "world_y": w[1], "world_z": w[2], **info}
            rows.append(r)
            index[k] = r
            written.add(f"{k[0]}:{k[1]}")
            added += 1
        elif r is not None:
            for f, val in info.items():
                if f not in ("video", "stop") and str(r.get(f, "")) != str(val):
                    r[f] = val
                    changed += 1
        s["annotation_id"] = r["id"] if r is not None else ""
    for r in rows:  # (sets written before is_extrapolated had "placed")
        if r.pop("placed", None) is not None:
            changed += 1
    if not rows or (exists and not added and not changed):
        return None
    fields = [f for f in dict.fromkeys(fields + ANN_STOP_FIELDS + [f for r in rows for f in r]) if f != "placed"]
    os.makedirs(folder, exist_ok=True)
    if exists:
        bdir = os.path.join(folder, "backup")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(path, os.path.join(bdir, os.path.basename(path) + _dt.datetime.now().strftime("_%Y%m%d_%H%M%S.bak")))
    write_csv(path, rows, fields)
    vs = {"track_dir": os.path.abspath(track_dir), "written": sorted(written)}
    if exists:
        write_annotation_sidecar(path, {"last_highest_id": last, "video_stops": vs}, update=True)
    else:
        write_annotation_sidecar(path, {"type": "annotations", "plot_id": project.id, "world_transform": W,
                                        "last_highest_id": last, "video_stops": vs}, update=False)
    logger.info("%s: %d stops added, %d stop values updated", path, added, changed)
    return path


@_renamed_kwargs(out="output_path")
def make_proxy(video: str, output_path: str, height: int = PROXY_HEIGHT) -> str:
    """An H.264 copy of ``video`` (``height`` px, a keyframe a second, fast start) for browsers; NVENC when
    available, else libx264. Done once."""
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + ".part.mp4"
    common = ["-vf", f"scale=-2:{height}", "-g", "30", "-an", "-movflags", "+faststart"]
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    try:
        subprocess.run(base + common + ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26", tmp], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(base + common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24", tmp], check=True)
    os.replace(tmp, output_path)
    return output_path


# --------------------------------------------------------------------------- the whole run


@_renamed_kwargs(out_dir="output_dir")
def track_videos(videos: Sequence[str], projects: Sequence[Any], output_dir: str, work_dir: str, *,
                 probe_frames: int = PROBE_FRAMES, proxies: bool = False, annotations: bool = True,
                 redo: bool = False, device: Optional[str] = None, workers: int = 8, poses: bool = True,
                 log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Keyframes and motion of every video, which plots each is on (``probe_frames`` seeds per plot; 0: track on
    every plot), its track and stops there; then the outputs in ``output_dir`` (see the module docstring).
    ``projects``: initialised :class:`ProjectInitializer` s (cloud and cameras loaded, world frame). Per video and
    plot the track is kept in ``work_dir`` and not made again unless ``redo``."""
    if log is None:
        def log(msg: str) -> None:
            print(msg, flush=True)
    os.makedirs(output_dir, exist_ok=True)
    keys = video_keys(videos)
    infos, frames, motion = {}, {}, {}
    for v in videos:
        info = probe_video(v)
        if info["duration"] < MIN_VIDEO_S:
            log(f"{keys[v]}: {info['duration']:.1f} s, left out")
            continue
        infos[v] = info
        fdir = os.path.join(work_dir, "frames", keys[v])
        t0 = time.time()
        frames[v] = extract_keyframes(v, fdir, info)
        motion[v] = frame_motion(frames[v], os.path.join(fdir, "motion.csv"))
        still = float(np.mean(motion[v][1:] < STILL_FLOW_PX)) if len(motion[v]) > 1 else 0.0
        log(f"{keys[v]}: {info['duration'] / 60:.1f} min, {len(frames[v])} frames, still {still:.0%} "
            f"({time.time() - t0:.0f} s)")
    vids = list(infos)
    tracks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    all_stops: List[Dict[str, Any]] = []
    for pr in projects:
        tdir = os.path.join(work_dir, "tracks", pr.id)
        os.makedirs(tdir, exist_ok=True)
        cached = {}
        for v in vids:
            tp = os.path.join(tdir, keys[v] + ".json")
            if not redo and os.path.exists(tp):
                with open(tp) as f:
                    cached[v] = json.load(f)
        need = [v for v in vids if v not in cached or (not (cached[v].get("refined") and cached[v].get("posed"))
                                                        and any("x" in r for r in cached[v]["rows"]))]
        matcher = None
        if need:
            log(f"{pr.id}: survey photos ...")
            matcher = PlotMatcher(pr, os.path.join(work_dir, "plots", pr.id, "thumbs"), device=device,
                                  workers=workers)
            log(f"{pr.id}: {len(matcher.photos)} survey photos, {len(matcher.seeds)} seed photos")
        for v in vids:
            tp = os.path.join(tdir, keys[v] + ".json")
            d = cached.get(v)
            if d is None:
                t0 = time.time()
                hits = probe(matcher, frames[v], motion[v], probe_frames) if probe_frames else -1
                rows: List[Dict[str, Any]] = []
                if hits != 0:
                    last_print = [0.0]

                    def prog(i, n, placed, _k=keys[v]):
                        if time.time() - last_print[0] > 30 or i == n:
                            last_print[0] = time.time()
                            log(f"  {_k} on {pr.id}: {i}/{n} frames, {placed} placed")

                    rows = track_frames(matcher, frames[v], motion[v], progress=prog)
                d = {"video": v, "probe_hits": hits, "rows": rows}
                log(f"{keys[v]} on {pr.id}: probe {hits}/{probe_frames}, "
                    f"{sum(1 for r in rows if 'x' in r)} of {len(frames[v])} frames placed ({time.time() - t0:.0f} s)")
            dirty = v not in cached
            rows = d["rows"]
            if not d.get("refined") and any("x" in r for r in rows):
                dirty = True
                t0 = time.time()
                nb = backfill(matcher, frames[v], motion[v], rows)
                st = find_stops([r["t_s"] for r in rows], motion[v])
                nf = fill_stops(matcher, frames[v], rows, st)
                d["refined"] = True
                log(f"{keys[v]} on {pr.id}: {nb} frames placed backwards, {nf} stops filled "
                    f"({time.time() - t0:.0f} s)")
            if not d.get("posed") and any("x" in r for r in rows):
                dirty = True
                t0 = time.time()
                np_ = add_poses(matcher, frames[v], rows, find_stops([r["t_s"] for r in rows], motion[v]))
                d["posed"] = True
                if np_:
                    log(f"{keys[v]} on {pr.id}: headings for {np_} stops ({time.time() - t0:.0f} s)")
            if dirty:
                with _atomic_write(tp, "w") as f:
                    json.dump(d, f)
            rows = despike(rows)
            if any("x" in r for r in rows):
                tracks[(v, pr.id)] = rows
                st = find_stops([r["t_s"] for r in rows], motion[v])
                for s in stop_rows(keys[v], pr.id, rows, st):
                    s["video_path"] = v
                    all_stops.append(s)
    # videos on no plot: their stops all the same (unplaced), so every stop of every video is listed
    tracked = {v for (v, _pid) in tracks}
    for v in vids:
        if v not in tracked:
            st = find_stops([t for _p, t in frames[v]], motion[v])
            for s in stop_rows(keys[v], "", [], st):
                s["video_path"] = v
                all_stops.append(s)
    # a stop placed on several plots (a video over two plots): keep it on the plot with most matched frames
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for s in all_stops:
        k = (s["video"], s["stop"])
        if k not in best or (s["n_matched"], s["x"] != "") > (best[k]["n_matched"], best[k]["x"] != ""):
            best[k] = s
    # stops never placed are listed once (plot empty until sequence_stops gives them their video's)
    stops = [s if s["x"] != "" else dict(s, plot="") for s in best.values()]
    stops.sort(key=lambda s: (s["video"], s["stop"]))
    sequence_stops(stops, tracks, vids, infos)
    lenses, lens_of = (pose_stops(stops, tracks, frames, projects, keys, infos, work_dir, redo=redo, device=device,
                                  workers=workers, log=log) if poses else ({}, {}))
    return write_outputs(output_dir, work_dir, vids, keys, infos, frames, tracks, stops, projects,
                         proxies=proxies, annotations=annotations, log=log, lenses=lenses, lens_of=lens_of)


@_renamed_kwargs(out_dir="output_dir")
def read_divers(output_dir: str) -> Dict[str, str]:
    """sequence -> the diver its camera filmed, from ``<output_dir>/labels.json`` (set in substrata-explore's video
    player); {} without."""
    try:
        with open(os.path.join(output_dir, "labels.json")) as f:
            return {str(k): str(v) for k, v in (json.load(f).get("divers") or {}).items() if v}
    except (OSError, ValueError, AttributeError):
        return {}


@_renamed_kwargs(out_dir="output_dir")
def write_outputs(output_dir, work_dir, vids, keys, infos, frames, tracks, stops, projects, *, proxies, annotations,
                  log, lenses=None, lens_of=None):
    by_plot = {pr.id: pr for pr in projects}
    divers = read_divers(output_dir)
    for s in stops:
        s["diver"] = divers.get(s.get("sequence", ""), "")
    if annotations:
        for pid, pr in by_plot.items():
            p = write_stops_annotations(pr, [s for s in stops if s["plot"] == pid], output_dir)
            if p:
                log(f"{pid}: stops written as annotation set {p}")
    write_csv(os.path.join(output_dir, "stops.csv"), stops, STOP_FIELDS)
    trows = [dict(r, video=keys[v], plot=pid) for (v, pid), rows in tracks.items() for r in rows if "x" in r]
    write_csv(os.path.join(output_dir, "track.csv"), trows, TRACK_FIELDS)
    vrows = []
    for v in vids:
        on = [pid for (vv, pid) in tracks if vv == v]
        vrows.append({"n": len(vrows) + 1, "video": keys[v], "sequence": os.path.basename(os.path.dirname(v)),
                      "video_path": v, "duration_s": round(infos[v]["duration"], 1),
                      "creation_time": infos[v].get("creation_time") or "", "frames": len(frames[v]),
                      "plots": " ".join(on),
                      "frames_placed": sum(1 for pid in on for r in tracks[(v, pid)] if "x" in r),
                      "stops": sum(1 for s in stops if s["video"] == keys[v]),
                      "stops_placed": sum(1 for s in stops if s["video"] == keys[v] and s["x"] != "")})
    write_csv(os.path.join(output_dir, "videos.csv"), vrows)
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(README.format(min_stop=MIN_STOP_S))
    proxy_paths = {}
    if proxies:
        for v in vids:
            t0 = time.time()
            proxy_paths[v] = make_proxy(v, os.path.join(output_dir, "proxies", keys[v] + ".mp4"))
            log(f"{keys[v]}: proxy ({time.time() - t0:.0f} s)")
    seq_colours = {q: i for i, q in enumerate(dict.fromkeys(os.path.basename(os.path.dirname(v)) for v in vids))}
    starts = video_starts(vids, infos)
    for pid, pr in by_plot.items():
        tr = {keys[v]: tracks[(v, p)] for (v, p) in tracks if p == pid}
        if not tr:
            continue
        img, geo = plot_map(pr.pcd)
        draw_map(img, geo, tr, [s for s in stops if s["plot"] == pid], os.path.join(output_dir, f"{pid}_map.jpg"),
                 title=pid, colours=seq_colours)
    meta = {"created": _dt.datetime.now().isoformat(timespec="seconds"),
            "plots": [{"id": pr.id, "yaml": getattr(pr, "yaml_path", None), "path": pr.path} for pr in projects],
            "videos": [{"key": keys[v], "path": v, "proxy": proxy_paths.get(v), "duration": infos[v]["duration"],
                        "sequence": os.path.basename(os.path.dirname(v)), "start": starts[v],
                        "lens": (lens_of or {}).get(keys[v]),
                        "fps": infos[v]["fps"], "creation_time": infos[v].get("creation_time"),
                        "plots": [pid for (vv, pid) in tracks if vv == v]} for v in vids],
            "lenses": lenses or {},
            "settings": {"still_flow_px": STILL_FLOW_PX, "min_stop_s": MIN_STOP_S, "min_inliers": MIN_INLIERS}}
    with _atomic_write(os.path.join(output_dir, "videotrack.json"), "w") as f:
        json.dump(meta, f, indent=1)
    nm = sum(1 for s in stops if s.get("placed") == "matched")
    ne = sum(1 for s in stops if s.get("placed") == "estimated")
    log(f"{len(stops)} stops: {nm} placed by matching, {ne} estimated; outputs in {output_dir}")
    return meta
