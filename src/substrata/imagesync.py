"""Poses for photos Metashape could not align, from image matches with an aligned camera group filmed at the same time.

Made for close-ups (macro photos of single colonies) taken by a diver who also films the reef (a GoPro's video
frames, aligned in Metashape): ``substrata camsync --match-images``. The close-ups' poses were copied from the GoPro
frame at the same EXIF time (``camsync``), but the two clocks differ by an unknown offset and the close-up camera
does not point where the GoPro points, so a copied pose can be metres off. Here each close-up is matched to the
GoPro frames themselves:

1. **Time offset** (:func:`estimate_offset`): every (close-up, frame) pair within a window of offsets is matched
   (SIFT, coarse: the whole frame and the close-up, both small) and scored by the number of matches consistent with
   one homography. Each close-up that matches votes with its scores / its best score (so a few strong matches, e.g.
   a colour chart, cannot outvote the rest); the votes peak at the clocks' offset. The highest peaks are then
   confirmed with the learned matcher on a sample of close-ups (:func:`confirm_offset`). Close-ups that match
   clearly also give where a close-up's centre lies in its frame (``aim``), how wide it appears there
   (``width_px``) and how it is turned.
2. **Matching** (:func:`match_targets`): each close-up against the frames within ``frames`` s of its offset time,
   cropped round ``aim`` (then the rest of the frame; then frames up to ``wide`` s off), with a learned matcher
   (kornia's LoFTR by default; DISK + LightGlue as the second opinion). These do not match a photo turned round, so
   the close-ups are first turned the way step 1 found them in the frames (by quarter turns: sa4's GoPro was upside
   down). On sa3_40m's 83 close-ups LoFTR matched 78, SIFT 30.
3. **Pose** (:func:`solve_pose`): each match's frame pixel is followed along the frame's ray to the plot's cloud,
   giving 3-D points seen at the close-up's pixels; PnP (RANSAC, then refined; the close-up's lens distortion
   undone) gives the close-up's pose in the cloud's frame.
4. **Checks** (:func:`assess`): enough PnP inliers, a small reprojection error, a distance from the camera to the
   point it aims at (``range``) like the other close-ups', and, where the second matcher also gives a pose, the
   two aiming at the same spot (across the view: along it, a branch tip and the base below it are the same spot).

:func:`set_pose` writes a pose into a camera (current and original frames); :func:`pose_from_images` runs 1-4.
The learned matchers need ``kornia`` (``pip install kornia``; a GPU makes them fast).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TIME_FORMAT = "%Y:%m:%d %H:%M:%S"  # (EXIF DateTimeOriginal, as Camera.get_datetime_original returns it)

# Stage 1 (offset): SIFT on the frame's centre and the close-up, both small
COARSE_SOURCE_WIDTH = 1600  # px, the whole frame resized to (a close-up may lie anywhere in it: sa4's near the top)
COARSE_SOURCE_FEATURES = 6000
COARSE_TARGET_WIDTH = 700  # px, the close-up resized to
COARSE_MIN_SCORE = 15  # homography inliers for a close-up's own best offset to count
COARSE_MAX_TARGETS = 120  # close-ups used (evenly spread) to find the offset
OFFSET_MIN_PROMINENCE = 3.0  # the votes' peak / their median
OFFSET_MAX_AMBIGUITY = 0.6  # the highest votes more than OFFSET_ELSEWHERE_S from the peak / the peak
OFFSET_ELSEWHERE_S = 20
OFFSET_MIN_AGREE = 3  # close-ups whose own clear best offset is within OFFSET_AGREE_S of the peak
OFFSET_AGREE_S = 8  # (a diver lingers at a colony: its best frame can be several s from the clocks' offset)
OWN_OFFSET_S = 15  # a close-up's own clear best offset this near the peak: matched round it
CONFIRM_CANDIDATES = 3  # the votes' highest peaks (OFFSET_ELSEWHERE_S apart) that may be confirmed
CONFIRM_SAMPLE = 12  # close-ups matched with the learned matcher at a candidate offset
CONFIRM_MIN = 3  # of those, matched for the candidate to be confirmed ...
CONFIRM_SHARE = 0.25  # ... and at least this share of them
FRAME_POSE_SOURCES = 4  # --pose-frames: agreeing photos pooled per frame (framepose; sa4_60m)
POSED_STATUSES = ("posed", "unverified", "estimated")  # the statuses that come with a pose (only "posed" trusted)
RIG_MIN = 5  # posed close-ups needed for the rig (the close-up's pose relative to its frame's) to estimate others
RIG_FRAME_S = 1  # an estimated close-up's frame: at its time + offset, or this many s off

# Stage 2 (matching)
MATCH_FRAMES = 3  # frames tried on either side of the offset time (s)
MATCH_WIDE_FRAMES = 8  # ... and, for a close-up that matches none of those, up to this far off
MATCH_MAX_SIDE = 1280  # px, the frame's crop resized to at most
MATCH_TARGET_OVERSIZE = 1.3  # the close-up a little bigger than it appears in the frame
MATCH_MIN_INLIERS = 20  # homography inliers for a match
MATCH_GOOD = 100  # matches enough not to look in the rest of the frame
HOMOGRAPHY_THRESHOLD_PX = 6.0  # in the frame's px

# Stage 3-4 (pose and checks)
PNP_MAX_POINTS = 150
PNP_THRESHOLD = 0.007  # of the close-up's width (~60 px of 8726; 1 mm on the reef ~40 px at 0.3 m)
PNP_MIN_INLIERS = 8
MAX_ERROR = 0.007  # median reprojection error, of the close-up's width
RANGE_RATIO = (0.5, 2.0)  # a pose's range within these times the median range of the poses
MAX_DISAGREEMENT_M = 0.02  # the two matchers' aim points at most this far apart
RAY_SEARCH_RADIUS = 0.003  # m, a frame ray's hit on the cloud (as cams2annos)
RAY_MAX_DISTANCE = 6.0  # m
AIM_MAX_SEARCH_RADIUS = 0.03  # m, the close-up's centre ray (3, 9, 27 mm)


# --------------------------------------------------------------------------- times


def parse_time(value) -> Optional[_dt.datetime]:
    """A camera's ``datetime`` (EXIF string) as a datetime, None if missing or unreadable."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    try:
        return _dt.datetime.strptime(str(value).strip(), TIME_FORMAT)
    except ValueError:
        return None


def by_time(cams) -> Dict[_dt.datetime, Any]:
    """The cameras by their ``datetime`` (the first of several at the same second), those without one left out."""
    out: Dict[_dt.datetime, Any] = {}
    for c in (cams.data.values() if hasattr(cams, "data") else cams):
        t = parse_time(getattr(c, "datetime", None))
        if t is not None and t not in out:
            out[t] = c
    return out


def overlap_window(target_times: Sequence[_dt.datetime], source_times: Sequence[_dt.datetime]) -> Tuple[int, int]:
    """The offsets (s, source time - target time) at which some close-up falls within the frames' time span."""
    return (int((min(source_times) - max(target_times)).total_seconds()),
            int((max(source_times) - min(target_times)).total_seconds()))


def peak(ks: np.ndarray, curve: np.ndarray) -> Dict[str, float]:
    """The offset of the summed scores' highest value and its prominence (that value / the median of the
    offsets that scored; 0 when nothing scored)."""
    ks, curve = np.asarray(ks), np.asarray(curve, float)
    j = int(np.argmax(curve))
    med = float(np.median(curve[curve > 0])) if (curve > 0).any() else 0.0
    prom = 0.0 if curve[j] <= 0 else (float(curve[j] / med) if med > 0 else float("inf"))
    return dict(offset=int(ks[j]), score=float(curve[j]), median=med, prominence=prom)


# --------------------------------------------------------------------------- images


def _cv2():
    import cv2
    return cv2


def _gray(path: str, reduce: int = 1):
    cv2 = _cv2()
    flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
            8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduce]
    img = cv2.imread(path, flag)
    if img is None:
        raise OSError(f"Cannot read {path}")
    return img


def _resize_width(img, width: int):
    cv2 = _cv2()
    h, w = img.shape[:2]
    return cv2.resize(img, (int(width), max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def _reduce_for(width_full: int, width_wanted: float) -> int:
    """The JPEG reduction (1, 2, 4, 8) that still leaves at least ``width_wanted`` px."""
    for r in (8, 4, 2):
        if width_full / r >= width_wanted:
            return r
    return 1


def _sift(img, n: int = 3000):
    cv2 = _cv2()
    img = cv2.createCLAHE(2.0, (8, 8)).apply(img)
    kp, des = cv2.SIFT_create(nfeatures=n).detectAndCompute(img, None)
    return np.float32([k.pt for k in kp]).reshape(-1, 2), des


def _coarse_source(path: str):
    """SIFT on a frame, small: (points in the frame's full-resolution px, descriptors)."""
    w_full = _image_width(path)
    img = _gray(path, _reduce_for(w_full, COARSE_SOURCE_WIDTH))
    p, d = _sift(_resize_width(img, COARSE_SOURCE_WIDTH), COARSE_SOURCE_FEATURES)
    return p * (w_full / COARSE_SOURCE_WIDTH), d


def _image_width(path: str) -> int:
    from PIL import Image

    with Image.open(path) as im:
        return im.size[0]


def _coarse_target(path: str, width_full: int):
    """SIFT on a close-up, small: (points in the close-up's full-resolution px, descriptors)."""
    img = _gray(path, _reduce_for(width_full, COARSE_TARGET_WIDTH))
    sc = COARSE_TARGET_WIDTH / img.shape[1] * (img.shape[1] / width_full)
    p, d = _sift(_resize_width(img, COARSE_TARGET_WIDTH))
    return p / sc, d


def homography_inliers(A: np.ndarray, B: np.ndarray, threshold: float = HOMOGRAPHY_THRESHOLD_PX, usac: bool = False):
    """(n inliers, H, inlier mask) of the matches A -> B (H maps A to B); (0, None, None) for too few."""
    cv2 = _cv2()
    if len(A) < 8:
        return 0, None, None
    method = cv2.USAC_MAGSAC if usac else cv2.RANSAC
    H, mask = cv2.findHomography(np.float32(A), np.float32(B), method, threshold, maxIters=5000, confidence=0.999)
    if H is None or mask is None:
        return 0, None, None
    k = mask.ravel().astype(bool)
    return int(k.sum()), H, k


def _match_sift(a, b, ratio: float = 0.75):
    cv2 = _cv2()
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0, None
    m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=50)).knnMatch(da, db, k=2)
    g = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
    if len(g) < 8:
        return 0, None
    n, H, _k = homography_inliers(pa[[x.queryIdx for x in g]], pb[[x.trainIdx for x in g]])
    return n, H


def circular_median_deg(angles: Sequence[float]) -> float:
    """The angle (degrees, -180..180) nearest to most of ``angles``: the one minimising the summed angular
    distances to them (each of them tried)."""
    a = np.asarray(angles, float)
    d = np.abs((a[:, None] - a[None, :] + 180) % 360 - 180).sum(1)
    return float(a[int(np.argmin(d))])


def quarter_turns(rotation_deg: Optional[float]) -> int:
    """``np.rot90``'s k (counter-clockwise quarter turns) that brings a close-up turned ``rotation_deg`` (image
    coordinates, y down: the homography's angle) in a frame closest to the frame's orientation."""
    if rotation_deg is None:
        return 0
    return int(np.round(-rotation_deg / 90.0)) % 4


def unrotate(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Points in an image turned by ``np.rot90(img, k)`` back to the image before (``w`` x ``h`` px)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    k %= 4
    if k == 0:
        return p.copy()
    if k == 1:
        return np.c_[w - 1 - y, x]
    if k == 2:
        return np.c_[w - 1 - x, h - 1 - y]
    return np.c_[y, h - 1 - x]


def map_point(H, xy) -> np.ndarray:
    p = np.asarray(H, float) @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


# --------------------------------------------------------------------------- 1. offset


def estimate_offset(
    targets: Sequence[Any],
    sources: Dict[_dt.datetime, Any],
    window: Optional[Tuple[int, int]] = None,
    *,
    max_targets: int = COARSE_MAX_TARGETS,
    n_jobs: int = -1,
    progress: bool = True,
) -> Dict[str, Any]:
    """The clocks' offset (s; a close-up taken at EXIF time t is in the frame of time t + offset), by matching close-ups
    to the frames (module docstring, step 1).

    Args:
        targets: The close-ups (cameras with ``datetime``, ``filepath`` and ``sensor``).
        sources: The frames by time (:func:`by_time`).
        window: The offsets tried, (min, max) s; default every offset at which a close-up falls within the frames'
            span (:func:`overlap_window`).
        max_targets: Close-ups used, evenly spread over their order.

    Returns:
        dict: ``ok``, ``offset``, ``prominence`` (and ``score``, ``median``), ``ambiguity`` (the highest votes
        elsewhere / the peak), ``ks`` / ``curve`` (the votes, smoothed) / ``raw`` (the summed scores) / ``scores``,
        ``rows`` (per close-up: ``name``, ``cam_id``, ``best`` offset, ``score``, ``clear``, ``agrees``, ``own``: its
        best offset when clear and near the peak, else None), ``n_targets``, ``n_agree``,
        ``aim`` (the close-ups' centre in a frame, its px; median of those that agree) and ``width_px`` (how wide a
        close-up appears there) and ``rotation_deg`` (how the close-ups are turned in a frame: e.g. ~180 when the
        GoPro was mounted upside down), None when none agrees; ``reason`` when not ``ok``.
    """
    from joblib import Parallel, delayed

    timed = [(c, parse_time(getattr(c, "datetime", None))) for c in targets]
    timed = [(c, t) for c, t in timed if t is not None]
    if not timed or not sources:
        return dict(ok=False, reason="no close-ups or frames with a time", offset=None, aim=None, width_px=None,
                    rows=[], n_targets=0, n_agree=0)
    if len(timed) > max_targets:
        timed = [timed[i] for i in np.linspace(0, len(timed) - 1, max_targets).round().astype(int)]
    lo, hi = window if window is not None else overlap_window([t for _c, t in timed], list(sources))
    lo, hi = int(lo), int(hi)
    jobs = []  # (target index, frame time, offset)
    for i, (_c, t) in enumerate(timed):
        for ts in sources:
            k = int((ts - t).total_seconds())
            if lo <= k <= hi:
                jobs.append((i, ts, k))
    if not jobs:
        return dict(ok=False, reason=f"no frame within offsets {lo}..{hi} s of any close-up", offset=None, aim=None,
                    width_px=None, rows=[], n_targets=len(timed), n_agree=0)
    need = sorted({ts for _i, ts, _k in jobs})
    if progress:
        print(f"Time offset: {len(timed)} close-ups x frames within {lo}..{hi} s ({len(jobs):,} pairs, "
              f"{len(need)} frames)", flush=True)
    sf = dict(zip(need, Parallel(n_jobs=n_jobs)(delayed(_coarse_source)(sources[ts].filepath) for ts in need)))
    tf = Parallel(n_jobs=n_jobs)(delayed(_coarse_target)(c.filepath, c.sensor.width) for c, _t in timed)
    res = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_match_sift)(tf[i], sf[ts]) for i, ts, _k in jobs)

    ks = np.arange(lo, hi + 1)
    S = np.zeros((len(timed), len(ks)))
    Hs: Dict[Tuple[int, int], Any] = {}
    for (i, _ts, k), (n, H) in zip(jobs, res):
        if n > S[i, k - lo]:
            S[i, k - lo] = n
            Hs[i, k] = H
    # each close-up that matches at all votes once (its scores / its best score: a few that match strongly, e.g. a
    # colour chart, must not outvote the rest), smoothed over +-1 s
    mx = S.max(1)
    use = mx >= COARSE_MIN_SCORE
    votes = (S[use] / mx[use, None]).sum(0) if use.any() else np.zeros(len(ks))
    curve = np.convolve(votes, np.ones(3), "same")
    pk = peak(ks, curve)
    away_pk = np.abs(ks - pk["offset"]) > OFFSET_ELSEWHERE_S
    ambiguity = float(curve[away_pk].max() / pk["score"]) if away_pk.any() and pk["score"] > 0 else 1.0
    rows, aims, widths, angles = [], [], [], []
    for i, (c, _t) in enumerate(timed):
        j = int(np.argmax(S[i]))
        best, score = int(ks[j]), float(S[i, j])
        away = np.abs(ks - best) > OFFSET_AGREE_S
        second = float(S[i, away].max()) if away.any() else 0.0
        clear = score >= COARSE_MIN_SCORE and score >= 1.5 * second
        agrees = clear and abs(best - pk["offset"]) <= OFFSET_AGREE_S
        rows.append(dict(name=c.filename, cam_id=str(c.cam_id), best=best, score=score, clear=clear, agrees=agrees,
                         own=best if clear and abs(best - pk["offset"]) <= OWN_OFFSET_S else None))
        H = Hs.get((i, best))
        if agrees and H is not None:
            w, h = c.sensor.width, c.sensor.height
            aims.append(map_point(H, (w / 2, h / 2)))
            widths.append(float(np.linalg.norm(map_point(H, (w, h / 2)) - map_point(H, (0, h / 2)))))
            angles.append(float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))))
    n_agree = sum(r["agrees"] for r in rows)
    ok = (pk["prominence"] >= OFFSET_MIN_PROMINENCE and ambiguity <= OFFSET_MAX_AMBIGUITY
          and n_agree >= min(OFFSET_MIN_AGREE, len(rows)))
    out = dict(ok=ok, ks=ks, curve=curve, raw=S.sum(0), scores=S, rows=rows, n_targets=len(timed), n_agree=n_agree,
               ambiguity=ambiguity, window=(lo, hi), **pk,
               aim=tuple(np.median(aims, 0).tolist()) if aims else None,
               width_px=float(np.median(widths)) if widths else None,
               rotation_deg=circular_median_deg(angles) if angles else None,
               candidates=candidate_offsets(ks, curve))
    if not ok:
        out["reason"] = (f"no clear offset: the votes' peak at {pk['offset']} s is {pk['prominence']:.1f}x their "
                         f"median (need {OFFSET_MIN_PROMINENCE:g}), elsewhere they reach {ambiguity:.2f} of it (at "
                         f"most {OFFSET_MAX_AMBIGUITY:g}), {n_agree} close-up(s) agree (need {OFFSET_MIN_AGREE})")
    return out


def frame_geometry(sources: Dict[_dt.datetime, Any]) -> Tuple[int, int, Optional[float]]:
    """(width, height, focal length px or None) of the frames: from the first frame with a sensor, else (no
    calibration: frames Metashape never aligned) from an image's size."""
    for c in sources.values():
        if getattr(c, "sensor", None) is not None:
            return int(c.sensor.width), int(c.sensor.height), float(c.sensor.f)
    from PIL import Image

    with Image.open(next(iter(sources.values())).filepath) as im:
        return int(im.size[0]), int(im.size[1]), None


def default_width_px(sources: Dict[_dt.datetime, Any], target) -> float:
    """How wide a close-up appears in a frame without a match to tell: as wide as its field at the frame's focal
    length, about half that as the frame is further from the reef (without the frames' focal length: 11% of the
    frame's width, as on sa3/sa4)."""
    fw, _fh, f = frame_geometry(sources)
    if f:
        return 0.5 * f * target.sensor.width / target.sensor.f
    return 0.11 * fw


def candidate_offsets(ks: np.ndarray, curve: np.ndarray, n: int = CONFIRM_CANDIDATES,
                      sep: float = OFFSET_ELSEWHERE_S) -> List[int]:
    """The votes' ``n`` highest peaks, at least ``sep`` s apart, highest first (none where nothing voted)."""
    ks = np.asarray(ks)
    c = np.asarray(curve, float).copy()
    out: List[int] = []
    while len(out) < n and len(c):
        j = int(np.argmax(c))
        if not c[j] > 0:
            break
        out.append(int(ks[j]))
        c[np.abs(ks - ks[j]) <= sep] = -np.inf
    return out


def confirmed(matched: int, sampled: int) -> bool:
    """Whether ``matched`` of ``sampled`` close-ups matching at an offset confirm it (:data:`CONFIRM_MIN`, and
    :data:`CONFIRM_SHARE` of them)."""
    return sampled > 0 and matched >= max(min(CONFIRM_MIN, sampled), int(np.ceil(CONFIRM_SHARE * sampled)))


def confirm_offset(
    targets: Sequence[Any],
    sources: Dict[_dt.datetime, Any],
    off: Dict[str, Any],
    *,
    matcher: str = "loftr",
    frames: int = MATCH_FRAMES,
    device: Optional[str] = None,
    sample: int = CONFIRM_SAMPLE,
    progress: bool = True,
) -> Dict[str, Any]:
    """Step 1's candidate offsets (``off["candidates"]``, :func:`estimate_offset`) checked with the learned
    matcher: a sample of close-ups that have frames at a candidate's time are matched there (:func:`match_targets`).
    The votes' peak is taken when it was clear (``off["ok"]``) and is confirmed (:func:`confirmed`); otherwise every
    candidate is tried, and the one most close-ups match is taken when it is confirmed and they are at least twice
    the next candidate's.

    Returns:
        dict: ``offset`` (None when none is taken: ``reason``), ``tried`` (per candidate: ``offset``, ``matched``,
        ``sampled``), and from the taken candidate's matches ``aim`` / ``width_px`` (as in :func:`estimate_offset`)
        and ``turns`` (how the close-ups matched, :func:`quarter_turns`), None without matches.
    """
    cands = list(off.get("candidates") or [])
    out: Dict[str, Any] = dict(offset=None, tried=[], aim=None, width_px=None, turns=None)
    if not cands:
        out["reason"] = off.get("reason") or "no candidate offset"
        return out
    fw, fh, _f = frame_geometry(sources)
    width0 = off.get("width_px") or default_width_px(sources, targets[0])
    turns0 = quarter_turns(off.get("rotation_deg"))
    timed = [(c, parse_time(getattr(c, "datetime", None))) for c in targets]
    timed = [(c, t) for c, t in timed if t is not None]
    runs = {}

    def run(k):
        have = [c for c, t in timed
                if any(t + _dt.timedelta(seconds=k + d) in sources for d in range(-frames, frames + 1))]
        pick = [have[i] for i in np.unique(np.linspace(0, len(have) - 1, min(sample, len(have))).round().astype(int))] \
            if have else []
        top = off.get("offset") is not None and abs(k - off["offset"]) <= OFFSET_AGREE_S and off.get("aim")
        aim = off["aim"] if top else (fw / 2, fh / 2)
        ms = match_targets(pick, sources, k, aim, width0, matcher=matcher, frames=frames, device=device,
                           turns=turns0, progress=False) if pick else {}
        n = sum(m["n"] >= MATCH_MIN_INLIERS for m in ms.values())
        runs[k] = (ms, pick)
        out["tried"].append(dict(offset=int(k), matched=int(n), sampled=len(pick)))
        if progress:
            print(f"  offset {k} s: {n} of {len(pick)} close-ups match ({matcher})", flush=True)
        return n, len(pick)

    if progress:
        print(f"Confirming the offset with {matcher} (up to {sample} close-ups each)", flush=True)
    n, m = run(cands[0])
    if off.get("ok") and confirmed(n, m):
        chosen = cands[0]
    else:
        for k in cands[1:]:
            run(k)
        ranked = sorted(out["tried"], key=lambda r: -r["matched"])
        best, second = ranked[0], (ranked[1]["matched"] if len(ranked) > 1 else 0)
        if confirmed(best["matched"], best["sampled"]) and best["matched"] >= 2 * second:
            chosen = best["offset"]
        else:
            out["reason"] = ("no offset confirmed: " + ", ".join(
                f"{r['matched']} of {r['sampled']} close-ups match at {r['offset']} s" for r in out["tried"])
                + f" (need {CONFIRM_MIN} and {CONFIRM_SHARE:.0%} of them, and twice the next offset's)")
            return out
    out["offset"] = int(chosen)
    ms, pick = runs[chosen]
    by_id = {str(c.cam_id): c for c in pick}
    aims, widths, ks_ = [], [], []
    for cid, mt in ms.items():
        if mt["n"] >= MATCH_MIN_INLIERS:
            c = by_id[cid]
            w, h = c.sensor.width, c.sensor.height
            aims.append(map_point(mt["H"], (w / 2, h / 2)))
            widths.append(float(np.linalg.norm(map_point(mt["H"], (w, h / 2)) - map_point(mt["H"], (0, h / 2)))))
            ks_.append(int(mt["turns"]))
    if aims:
        out.update(aim=tuple(np.median(aims, 0).tolist()), width_px=float(np.median(widths)),
                   turns=int(np.bincount(ks_).argmax()))
    return out


# --------------------------------------------------------------------------- 2. matching


class LearnedMatcher:
    """kornia's LoFTR (``"loftr"``, outdoor weights) or DISK + LightGlue (``"lightglue"``) on one device; the
    weights are downloaded on first use."""

    def __init__(self, kind: str = "loftr", device: Optional[str] = None):
        try:
            import torch
            import kornia.feature as KF
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("Matching close-ups needs kornia: pip install kornia") from e
        self.torch, self.KF, self.kind = torch, KF, kind
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if kind == "loftr":
            self.model = KF.LoFTR(pretrained="outdoor").to(self.device).eval()
        elif kind == "lightglue":
            self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
            self.model = KF.LightGlueMatcher("disk").to(self.device).eval()
        else:
            raise ValueError(f"Unknown matcher {kind!r} (loftr or lightglue)")

    def _tensor(self, gray):
        t = self.torch.from_numpy(np.ascontiguousarray(gray)).float().div(255)[None, None]
        return t.to(self.device)

    def features(self, gray):
        """DISK features of a grayscale image (LightGlue only; LoFTR needs none)."""
        if self.kind != "lightglue":
            return None
        with self.torch.inference_mode():
            rgb = self._tensor(gray).repeat(1, 3, 1, 1)
            f = self.disk(rgb, n=2048, pad_if_not_divisible=True)[0]
        return f.keypoints, f.descriptors

    def match(self, gray0, gray1, f0=None, f1=None) -> Tuple[np.ndarray, np.ndarray]:
        """Matched points (image 0 px, image 1 px)."""
        with self.torch.inference_mode():
            if self.kind == "loftr":
                out = self.model({"image0": self._tensor(gray0), "image1": self._tensor(gray1)})
                return out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy()
            (k0, d0), (k1, d1) = f0 or self.features(gray0), f1 or self.features(gray1)
            ones = lambda k: self.torch.ones(1, len(k), 1, 1, device=self.device)  # noqa: E731
            l0 = self.KF.laf_from_center_scale_ori(k0[None], ones(k0))
            l1 = self.KF.laf_from_center_scale_ori(k1[None], ones(k1))
            _d, idx = self.model(d0, d1, l0, l1, hw1=gray0.shape[:2], hw2=gray1.shape[:2])
            idx = idx.cpu().numpy()
            return k0[idx[:, 0]].cpu().numpy(), k1[idx[:, 1]].cpu().numpy()


def crop_box(aim: Tuple[float, float], width_px: float, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """The frame's crop matched against a close-up: round ``aim``, ~4 close-up widths wide (at least 1600 x 1200
    px, 4:3), inside the frame. (x0, y0, w, h)."""
    cw = int(min(frame_w, max(1600, 4 * width_px)))
    ch = int(min(frame_h, max(1200, 3 * width_px)))
    x0 = int(np.clip(aim[0] - cw / 2, 0, frame_w - cw))
    y0 = int(np.clip(aim[1] - ch / 2, 0, frame_h - ch))
    return x0, y0, cw, ch


def frame_tiles(aim: Tuple[float, float], width_px: float, frame_w: int, frame_h: int) -> List[Tuple[int, int, int, int]]:
    """The crops of a frame a close-up is looked for in: first the one round ``aim`` (:func:`crop_box`), then a
    grid of crops of that size covering the frame, overlapping by 1.5 close-up widths (so a close-up lies whole in
    one), nearest to ``aim`` first."""
    first = crop_box(aim, width_px, frame_w, frame_h)
    _x, _y, cw, ch = first
    ov = int(1.5 * width_px)

    def starts(full, size):
        if size >= full:
            return [0]
        n = int(np.ceil((full - size) / max(1, size - ov))) + 1
        return sorted({int(round(v)) for v in np.linspace(0, full - size, n)})

    grid = [(x, y, cw, ch) for y in starts(frame_h, ch) for x in starts(frame_w, cw)]
    grid.sort(key=lambda t: (t[0] + cw / 2 - aim[0]) ** 2 + (t[1] + ch / 2 - aim[1]) ** 2)
    return [first] + [t for t in grid if t != first]


def match_targets(
    targets: Sequence[Any],
    sources: Dict[_dt.datetime, Any],
    offset: int,
    aim: Tuple[float, float],
    width_px: float,
    *,
    matcher: str = "loftr",
    frames: int = MATCH_FRAMES,
    device: Optional[str] = None,
    near: Optional[Dict[str, Tuple[float, float]]] = None,
    own: Optional[Dict[str, int]] = None,
    turns: int = 0,
    wide: int = 0,
    progress: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Each close-up matched against the frames within ``frames`` s of its time + ``offset`` (module docstring,
    step 2; or of its own offset ``own[cam_id]``): first in the crop of each frame round ``aim`` (or round ``near[cam_id]``, e.g. where another matcher
    found it); when that gives fewer than ``MATCH_GOOD`` matches, in the rest of the frame too (:func:`frame_tiles`).
    The close-up is turned by ``turns`` quarter turns first (:func:`quarter_turns`: the learned matchers do not
    match a photo turned upside down); one that still does not match is tried the other three ways round too, and
    then (with ``wide`` > ``frames``) in the frames up to ``wide`` s off.
    Returns, per close-up ``cam_id``, its best frame: ``n`` (homography inliers), ``source`` (the frame's camera),
    ``dt`` (its time - the close-up's - offset, s), ``A`` / ``B`` (the inlier matches in the close-up's and the
    frame's full-resolution px) and ``H`` (close-up -> frame); ``n`` 0 and no ``source`` without a frame
    (``no_frames`` True: no aligned frame at all within ``wide`` (or ``frames``) s)."""
    from collections import OrderedDict

    m = LearnedMatcher(matcher, device)
    fw, fh, _f = frame_geometry(sources)
    _x, _y, cw, ch = crop_box(aim, width_px, fw, fh)
    sc = min(1.0, MATCH_MAX_SIDE / max(cw, ch))
    size = (int(cw * sc) // 8 * 8, int(ch * sc) // 8 * 8)
    tw = max(160, int(width_px * sc * MATCH_TARGET_OVERSIZE) // 8 * 8)
    frames_cache: "OrderedDict[_dt.datetime, Any]" = OrderedDict()

    def frame(ts):
        if ts not in frames_cache:
            frames_cache[ts] = _gray(sources[ts].filepath)
            if len(frames_cache) > 4 * (2 * frames + 1):
                frames_cache.popitem(last=False)
        return frames_cache[ts]

    def tile(ts, box):
        x0, y0, w, h = box
        return _cv2().resize(frame(ts)[y0:y0 + h, x0:x0 + w], size, interpolation=_cv2().INTER_AREA)

    timed = [(c, parse_time(getattr(c, "datetime", None))) for c in targets]
    if progress:
        print(f"Matching {len(timed)} close-ups with {matcher} ({m.device}), turned {90 * (turns % 4)} deg: frames "
              f"within {frames} s, crops "
              f"{cw}x{ch} px (round {tuple(int(v) for v in aim)} first; the rest of the frame when that fails), "
              f"scaled {sc:.2f}", flush=True)
    out: Dict[str, Dict[str, Any]] = {}
    n_wide = n_turned = n_further = 0
    for c, t in timed:
        best: Dict[str, Any] = dict(n=0)
        cid = str(c.cam_id)
        if t is not None:
            img = _resize_width(_gray(c.filepath, _reduce_for(c.sensor.width, tw)), tw)
            img = img[: img.shape[0] // 8 * 8]
            ih, iw = img.shape
            tsc = c.sensor.width / tw
            turned = {}

            def view(k):
                if k not in turned:
                    g = np.ascontiguousarray(np.rot90(img, k))
                    turned[k] = (g, m.features(g))
                return turned[k]

            tiles = frame_tiles((near or {}).get(cid, aim), width_px, fw, fh)
            k0 = (own or {}).get(cid, offset)
            stamps = [t + _dt.timedelta(seconds=k0 + d) for d in range(-frames, frames + 1)]
            stamps = [ts for ts in stamps if ts in sources]

            def attempt(k, box, stamps=stamps):
                nonlocal best
                gray, ft = view(k)
                for ts in stamps:
                    g = tile(ts, box)
                    a, b = m.match(gray, g, ft, m.features(g))
                    if len(a) < 8:
                        continue
                    A, B = unrotate(a, k, iw, ih) * tsc, b / sc + [box[0], box[1]]
                    n, H, keep = homography_inliers(A, B, usac=True)
                    if n > best["n"]:
                        best = dict(n=n, source=sources[ts], dt=int((ts - t).total_seconds()) - offset, A=A[keep],
                                    B=B[keep], H=H, turns=k)

            for ti, box in enumerate(tiles if stamps else []):
                if ti == 1:
                    if best["n"] >= MATCH_GOOD:
                        break
                    n_wide += 1
                attempt(turns % 4, box)
                if ti >= 1 and best["n"] >= MATCH_GOOD:
                    break
            if stamps and best["n"] < MATCH_MIN_INLIERS:
                n_turned += 1
                for k in [(turns + j) % 4 for j in (2, 1, 3)]:
                    attempt(k, tiles[0])
                    if best["n"] >= MATCH_MIN_INLIERS:
                        break
            if best["n"] < MATCH_MIN_INLIERS and wide > frames:
                # the diver lingered at the colony, or the frames at its time are not aligned: frames further off
                more = [t + _dt.timedelta(seconds=k0 + d) for d in range(-wide, wide + 1) if abs(d) > frames]
                more = [ts for ts in more if ts in sources]
                if more:
                    n_further += 1
                    for box in tiles:
                        attempt(turns % 4, box, more)
                        if best["n"] >= MATCH_GOOD:
                            break
                stamps = stamps + more
            if not stamps:
                best["no_frames"] = True
        out[cid] = best
    if progress and (n_wide or n_turned or n_further):
        print(f"  {n_wide} close-up(s) looked for in the whole frame, {n_turned} also turned the other ways"
              + (f", {n_further} also in frames {frames + 1}-{wide} s off" if wide > frames else ""), flush=True)
    return out


# --------------------------------------------------------------------------- 3. pose


def to_cloud(cam, px, pcd, radius: float = RAY_SEARCH_RADIUS) -> Optional[np.ndarray]:
    """Where the ray through pixel ``px`` of ``cam`` first meets ``pcd`` (both in the cloud's frame), else None."""
    v = cam.pixel_to_ray(float(px[0]), float(px[1]))
    hit = pcd.get_intercept(np.asarray(cam.coords, float), radius, vector=v, max_dist_from_origin=RAY_MAX_DISTANCE)
    return None if hit is None else np.asarray(hit.coords, float)


def pose_from_points(obj: np.ndarray, img: np.ndarray, sensor, *, threshold: float = PNP_THRESHOLD, seed: int = 0):
    """PnP for a camera with ``sensor`` seeing 3-D points ``obj`` at its pixels ``img`` (lens distortion undone):
    RANSAC (EPnP), then refined on the inliers. Returns (R world -> camera, centre, inlier indices, per-inlier
    reprojection errors in px) or None."""
    cv2 = _cv2()
    from .cameras import _undistort

    obj = np.asarray(obj, np.float64)
    img = np.asarray(img, np.float64)
    if len(obj) < max(PNP_MIN_INLIERS, 6):
        return None
    xn, yn = _undistort(img[:, 0], img[:, 1], sensor)
    norm = np.stack([xn, yn], 1).astype(np.float64)
    K = np.eye(3)
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, norm, K, None, reprojectionError=threshold * sensor.width / sensor.f,
                                             iterationsCount=2000, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < PNP_MIN_INLIERS:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], norm[inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    pr = obj[inl] @ R.T + tvec.ravel()
    if (pr[:, 2] <= 0).any():
        return None
    err = np.linalg.norm(pr[:, :2] / pr[:, 2:] - norm[inl], axis=1) * sensor.f
    return R, -R.T @ tvec.ravel(), inl, err


def solve_pose(target, source, A: np.ndarray, B: np.ndarray, pcd, *, sensor=None,
               max_points: int = PNP_MAX_POINTS, seed: int = 0) -> Optional[Dict[str, Any]]:
    """The close-up's pose from its matches with a frame (module docstring, step 3): ``R`` (world -> camera, the
    cloud's frame), ``C`` (centre), ``inliers`` / ``points``, ``error`` (median reprojection error, px), ``aim`` (where
    its centre ray meets the cloud, None if nowhere) and ``range`` (camera to ``aim``); None without a pose."""
    sensor = sensor or target.sensor
    A, B = np.asarray(A, float), np.asarray(B, float)
    if len(A) > max_points:
        keep = np.random.default_rng(seed).choice(len(A), max_points, replace=False)
        A, B = A[keep], B[keep]
    pts = [to_cloud(source, b, pcd) for b in B]
    ok = np.array([p is not None for p in pts], bool)
    if ok.sum() < PNP_MIN_INLIERS:
        return None
    got = pose_from_points(np.array([p for p in pts if p is not None]), A[ok], sensor, seed=seed)
    if got is None:
        return None
    R, C, inl, err = got
    ray = R.T @ np.array([0.0, 0.0, 1.0])
    # (the ray through the photo's centre: the lens leaves the principal point's ray straight; through a gap in
    # the cloud, tried 3x wider, as cams2annos)
    hit, r = None, RAY_SEARCH_RADIUS
    while hit is None and r <= AIM_MAX_SEARCH_RADIUS:
        hit = pcd.get_intercept(C, r, vector=ray, max_dist_from_origin=3.0)
        r *= 3
    aim = None if hit is None else np.asarray(hit.coords, float)
    return dict(R=R, C=C, inliers=int(len(inl)), points=int(ok.sum()), error=float(np.median(err)), aim=aim,
                range=None if aim is None else float(np.linalg.norm(aim - C)))


# --------------------------------------------------------------------------- 4. checks


def aim_apart(p: Dict[str, Any], q: Dict[str, Any]) -> float:
    """How far apart two poses of a close-up aim, across ``p``'s view: ``q``'s aim point from ``p``'s centre ray
    (m). Along the ray they may differ more: on a branching colony one ray stops at a branch tip, the other a few
    cm deeper, both on the same colony."""
    ray = np.asarray(p["R"], float).T @ np.array([0.0, 0.0, 1.0]) if "R" in p else None
    d = np.asarray(q["aim"], float) - np.asarray(p["aim"], float)
    if ray is None:
        return float(np.linalg.norm(d))
    return float(np.linalg.norm(d - (d @ ray) * ray))


def assess(poses: Dict[str, Optional[Dict[str, Any]]], second: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
           *, width: Optional[Dict[str, int]] = None, max_error: float = MAX_ERROR,
           range_ratio: Tuple[float, float] = RANGE_RATIO,
           max_disagreement: float = MAX_DISAGREEMENT_M) -> Dict[str, Tuple[str, str]]:
    """Each close-up's (status, note): ``posed`` (and the second matcher agrees: note ``checked``; or gave no pose:
    ``unchecked``) or ``rejected`` (the note says why), or ``no pose``. The matchers agree when their aims are within
    ``max_disagreement`` across the view (:func:`aim_apart`). ``width``: the close-ups' widths (px), for
    ``max_error`` (a share of the width); without it the error is not checked."""
    width = width or {}
    ranges = [p["range"] for p in poses.values() if p and p.get("range")]
    med = float(np.median(ranges)) if ranges else None
    out: Dict[str, Tuple[str, str]] = {}
    for k, p in poses.items():
        if not p:
            out[k] = ("no pose", "")
            continue
        if p.get("aim") is None:
            out[k] = ("rejected", "its centre ray meets no cloud")
            continue
        if k in width and p["error"] > max_error * width[k]:
            out[k] = ("rejected", f"reprojection error {p['error']:.0f} px")
            continue
        if med and not (range_ratio[0] * med <= p["range"] <= range_ratio[1] * med):
            out[k] = ("rejected", f"range {p['range']:.2f} m (the poses' median {med:.2f} m)")
            continue
        q = (second or {}).get(k)
        if second is None:
            out[k] = ("posed", "")
        elif not q or q.get("aim") is None:
            out[k] = ("posed", "unchecked")
        else:
            d = aim_apart(p, q)
            out[k] = (("posed", "checked") if d <= max_disagreement
                      else ("rejected", f"the matchers aim {d * 100:.1f} cm apart"))
    return out


# --------------------------------------------------------------------------- writing a pose


def set_pose(cam, R: np.ndarray, C: np.ndarray, world_transform: np.ndarray) -> None:
    """Give ``cam`` the pose ``R`` (world -> camera) / centre ``C`` in the current (world) frame, and the matching
    original-frame pose (what ``Cameras.save`` writes to the meta JSON). The world transform's scale is carried in
    the current frame's camera transform, as for cameras loaded from the project."""
    from .cameras import _projection_changed

    W = np.asarray(world_transform, float)
    s = float(np.linalg.norm(W[:3, 0]))
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R, float).T
    T[:3, 3] = np.asarray(C, float)
    cam.camera_transform = T
    cam.coords = np.asarray(C, float).copy()
    cam.reverse_transform_coords(W)
    for flag in ("missing_pose_from_meta", "missing_pose_from_timematch", "pose_cleared"):
        if hasattr(cam, flag):
            delattr(cam, flag)
    _projection_changed()


def clear_pose(cam) -> None:
    """Leave ``cam`` without a pose (``Cameras.save`` then writes a null pose) and disabled."""
    from .cameras import _projection_changed

    cam.missing_pose_from_meta = True
    cam.pose_cleared = True
    cam.enabled = False
    _projection_changed()


# --------------------------------------------------------------------------- all steps


def pose_from_images(
    targets: Sequence[Any],
    sources: Sequence[Any],
    pcd,
    *,
    offset: Optional[int] = None,
    window: Optional[Tuple[int, int]] = None,
    matcher: str = "loftr",
    check: Optional[str] = "lightglue",
    frames: int = MATCH_FRAMES,
    wide: int = MATCH_WIDE_FRAMES,
    frames_all: Optional[Sequence[Any]] = None,
    pose_frames: bool = False,
    photos: Sequence[Any] = (),
    frame_sensor_other=None,
    estimate: bool = True,
    device: Optional[str] = None,
    n_jobs: int = -1,
    progress: bool = True,
) -> Dict[str, Any]:
    """Steps 1-4 of the module docstring for close-ups ``targets`` and aligned frames ``sources`` (cameras with
    ``datetime``; both in ``pcd``'s frame). ``frames_all``: all the frames, aligned or not (default ``sources``):
    the time offset is found and confirmed with these (it needs no poses), the close-ups are posed from
    ``sources``. With ``offset`` the offset is not searched for (only round it, for the
    close-ups' place in a frame). With ``pose_frames``, the unaligned frames of ``frames_all`` at the times of
    close-ups that have no aligned frame within ``frames`` s are posed first (:func:`substrata.framepose.pose_frames`,
    from the posed ``photos``, e.g. the survey photos, and the aligned frames) and matched as aligned ones are;
    ``frame_sensor_other``: the frames' lens turned 180 deg (a borrowed calibration): the better fitting is used
    (:func:`substrata.framepose.choose_orientation`; given to all of ``frames_all``). With ``estimate`` (default),
    a close-up without a good match is still given a pose, marked (:func:`estimate_poses`).
    Nothing is written to the cameras (:func:`set_pose` does that; frames are posed on copies).

    Returns:
        dict: ``offset`` (step 1's result, :func:`estimate_offset`), ``time_offset`` (s), ``matches`` /
        ``poses`` (by the main matcher), ``check_poses`` (the second's, or None), ``status`` (:func:`assess`;
        ``no frames`` for a close-up without an aligned frame within ``wide`` s of its time + offset),
        ``frame_poses`` (with ``pose_frames``: :func:`substrata.framepose.pose_frames`' results, by frame ``cam_id``;
        ``used``: the pose was used). ``offset["ok"]`` False (and no matches): stopped at step 1.
    """
    src = by_time(sources)
    src_all = by_time(frames_all) if frames_all else src
    if offset is not None:
        window = (int(offset) - 5, int(offset) + 5)
    off = estimate_offset(targets, src_all, window, n_jobs=n_jobs, progress=progress)
    out = dict(offset=off, time_offset=None, matches={}, poses={}, check_poses=None, status={}, confirm=None,
               frame_poses=None)
    conf: Dict[str, Any] = {}
    if offset is not None:
        k = int(offset)
    else:
        conf = confirm_offset(targets, src_all, off, matcher=matcher, frames=frames, device=device, progress=progress)
        out["confirm"] = conf
        k = conf["offset"]
        if k is None:
            off["reason"] = conf.get("reason") or off.get("reason")
            return out
    out["time_offset"] = k
    fw, fh, _f = frame_geometry(src_all)
    near_peak = off.get("offset") is not None and abs(k - off["offset"]) <= OFFSET_AGREE_S
    sift_aim = off.get("aim") if near_peak else None
    aim = sift_aim or conf.get("aim") or (fw / 2, fh / 2)
    # (without a clear match: as wide as the close-up's field at the frame's focal length, about half that as
    # the frame is further away)
    width_px = ((off.get("width_px") if sift_aim else None) or conf.get("width_px")
                or default_width_px(src_all, targets[0]))
    own = {r["cam_id"]: r["own"] for r in off.get("rows", [])
           if r.get("own") is not None and abs(r["own"] - k) <= OWN_OFFSET_S}
    turns = conf["turns"] if conf.get("turns") is not None else quarter_turns(off.get("rotation_deg"))
    out.update(aim=aim, width_px=width_px, turns=turns)
    tby = {str(c.cam_id): c for c in targets}
    fp_state: Dict[str, Any] = {}

    def pose_needed(need: List[Any]) -> None:
        """(--pose-frames) pose the unaligned frames ``need`` (and those chaining them to aligned ones); used ones
        join ``src`` as posed copies."""
        from . import framepose

        if "index" not in fp_state:
            fp_state["index"] = framepose.LazyIndex(photos, device=device, progress=progress)
        index = fp_state["index"]
        if frame_sensor_other is not None and "orientation" not in out:
            sensor = next(c.sensor for c in frames_all if getattr(c, "sensor", None) is not None)
            if progress:
                print(f"Which way round the frames' lens fits: {framepose.ORIENT_FRAMES} frames posed both ways",
                      flush=True)
            ori = framepose.choose_orientation(need, frames_all, photos, pcd, sensor, frame_sensor_other,
                                               index=index, device=device, progress=progress)
            ori["errors"] = [tuple(e) for e in ori["errors"]]
            out["orientation"] = {k_: v for k_, v in ori.items() if k_ != "sensor"}
            if ori["turned"]:
                from .frameretrieval import give_sensor

                give_sensor(frames_all, ori["sensor"])
            if progress:
                print(f"  the lens {'turned 180 deg' if ori['turned'] else 'as given'}: {ori['better']} of "
                      f"{ori['n']} frames fit it turned better" + ("" if ori["clear"] else
                      " (not clear: kept as given; --frame-calibration-orientation to choose)"), flush=True)
        fr = framepose.pose_frames(need, frames_all, photos, pcd, device=device, max_sources=FRAME_POSE_SOURCES,
                                   place=True, index=index, progress=progress)
        by_id = {str(c.cam_id): c for c in frames_all}
        W = np.asarray(pcd.world_transform, float)
        for cid, r in fr.items():
            r["used"] = bool(r.get("ok") and r.get("plausible") and r.get("anchored"))
            if r["used"]:
                c = framepose.posed_copy(by_id[cid], r["R"], r["C"], W)
                src[parse_time(c.datetime)] = c
        out["frame_poses"] = {**(out.get("frame_poses") or {}), **fr}

    def match_and_pose(tg: Sequence[Any]):
        """Steps 2-3 for close-ups ``tg``: (matches, poses, the second matcher's poses or None)."""
        ms = match_targets(tg, src, k, aim, width_px, matcher=matcher, frames=frames, device=device,
                           own=own, turns=turns, wide=wide, progress=progress)
        ps = {cid: (solve_pose(tby[cid], m["source"], m["A"], m["B"], pcd) if m["n"] >= MATCH_MIN_INLIERS else None)
              for cid, m in ms.items()}
        sec = None
        if check:
            posed = [tby[cid] for cid, p in ps.items() if p]
            near = {cid: tuple(map_point(ms[cid]["H"], (tby[cid].sensor.width / 2, tby[cid].sensor.height / 2)))
                    for cid, p in ps.items() if p}
            own2 = {cid: k + ms[cid]["dt"] for cid in near}
            cm2 = match_targets(posed, src, k, aim, width_px, matcher=check, frames=frames, device=device, near=near,
                                own=own2, turns=turns, progress=progress) if posed else {}
            sec = {cid: (solve_pose(tby[cid], m["source"], m["A"], m["B"], pcd) if m["n"] >= MATCH_MIN_INLIERS
                         else None) for cid, m in cm2.items()}
        return ms, ps, sec

    if pose_frames and frames_all:
        need = unaligned_frames([c for c in targets if not _has_frame(c, src, k, frames)], frames_all, k, frames)
        if need:
            if progress:
                print(f"Posing {len(need)} unaligned frame(s) at the times of close-ups without an aligned frame "
                      f"(and those between them and the nearest aligned ones) from {len(photos)} photo(s)", flush=True)
            pose_needed(need)
    matches, poses, second = match_and_pose(targets)
    out.update(matches=matches, poses=poses, check_poses=second)
    status = assess(poses, second, width={cid: c.sensor.width for cid, c in tby.items()})
    if pose_frames and frames_all:
        # (a close-up that matched none of its frames may have unaligned ones among them, e.g. 1-2 s either side
        # of its time while aligned frames 3 s off show the reef next to it: pose those, and match it again)
        tried = set(out.get("frame_poses") or {})
        retry = [tby[cid] for cid, (s_, _n) in status.items() if s_ in ("no pose", "rejected")]
        need = [f for f in unaligned_frames(retry, frames_all, k, frames) if str(f.cam_id) not in tried]
        if need:
            if progress:
                print(f"Posing {len(need)} unaligned frame(s) at the times of {len(retry)} close-up(s) not posed, "
                      "to match them again", flush=True)
            pose_needed(need)
            m2, p2, s2 = match_and_pose(retry)
            for cid, m in m2.items():
                if p2.get(cid) and m["n"] > matches[cid]["n"]:
                    matches[cid], poses[cid] = m, p2[cid]
                    if second is not None:
                        second[cid] = (s2 or {}).get(cid)
            status = assess(poses, second, width={cid: c.sensor.width for cid, c in tby.items()})
    for cid, m in matches.items():
        if m.get("no_frames") and status.get(cid, ("",))[0] == "no pose":
            status[cid] = ("no frames", f"no aligned frame within {max(wide, frames)} s of its time")
        elif getattr(m.get("source"), "framepose_chained", False) and status.get(cid, ("",))[0] == "posed":
            s_, note = status[cid]
            status[cid] = (s_, ", ".join(v for v in (note, "frame posed by substrata") if v))
    if estimate:
        out["rig"] = estimate_poses(targets, src, k, matches, poses, status, pcd)
    out["status"] = status
    return out


def rig_offset(targets: Sequence[Any], matches: Dict[str, Any], poses: Dict[str, Any], status: Dict[str, Tuple[str, str]]
               ) -> Optional[Dict[str, Any]]:
    """The rig: a close-up's pose relative to its frame's (the GoPro mounted on the close-up camera), from the posed
    close-ups (``status`` posed) and the frames they matched: ``R`` (close-up axes from the frame's, world -> camera
    convention) and ``t`` (the close-up's centre in the frame's axes, m), the mean rotation and median offset;
    ``n``, ``spread_m`` (the offsets' median distance from ``t``) and ``spread_deg``. None with fewer than
    ``RIG_MIN``. The close-ups' own poses carry their tilt / shift ambiguity (a few cm, a few deg), the median
    evens it out."""
    from scipy.spatial.transform import Rotation

    from .framepose import world_to_cam

    rel_R, rel_t = [], []
    for c in targets:
        cid = str(c.cam_id)
        f = (matches.get(cid) or {}).get("source")
        p = poses.get(cid)
        if status.get(cid, ("",))[0] != "posed" or f is None or not p:
            continue
        Rf = world_to_cam(f.camera_transform)
        rel_R.append(np.asarray(p["R"], float) @ Rf.T)
        rel_t.append(Rf @ (np.asarray(p["C"], float) - np.asarray(f.coords, float)))
    if len(rel_t) < RIG_MIN:
        return None
    rot = Rotation.from_matrix(np.asarray(rel_R))
    mean = rot.mean()
    t = np.median(np.asarray(rel_t), axis=0)
    return dict(R=mean.as_matrix(), t=t, n=len(rel_t),
                spread_m=float(np.median(np.linalg.norm(np.asarray(rel_t) - t, axis=1))),
                spread_deg=float(np.degrees(np.median((mean.inv() * rot).magnitude()))))


def estimate_poses(targets: Sequence[Any], src: Dict[_dt.datetime, Any], offset: int, matches: Dict[str, Any],
                   poses: Dict[str, Any], status: Dict[str, Tuple[str, str]], pcd) -> Optional[Dict[str, Any]]:
    """Poses, marked, for the close-ups without a good match (in place in ``poses`` / ``status``):

    - ``unverified``: the two matchers disagree on where it aims (``rejected``, :data:`MAX_DISAGREEMENT_M`); the
      main matcher's pose is kept.
    - ``estimated``: no pose, or one rejected for its range, error or no cloud: the pose of the frame at its time +
      ``offset`` (or ``RIG_FRAME_S`` s off) and the rig (:func:`rig_offset`). On sa3_40m / sa4_40m (each posed
      close-up estimated from the others): it aims 4-6 cm (median; 8-12 cm p90) from its matched pose's aim,
      against 21-23 cm for the frame's pose copied as it is. A diver who moved off before or after the close-up
      makes it worse: look at it.

    Returns the rig (without its matrices: ``n``, ``spread_m``, ``spread_deg``) or None (too few posed close-ups:
    no estimates; the unverified are marked still)."""
    from .framepose import aim_of, world_to_cam

    for cid, (s, note) in list(status.items()):
        if s == "rejected" and note.startswith("the matchers aim"):
            status[cid] = ("unverified", note)
    rig = rig_offset(targets, matches, poses, status)
    if rig is None:
        return None
    for c in targets:
        cid = str(c.cam_id)
        s, note = status.get(cid, ("no pose", ""))
        if s not in ("no pose", "rejected"):
            continue
        t = parse_time(getattr(c, "datetime", None))
        if t is None:
            continue
        hit = next(((d, src[t + _dt.timedelta(seconds=int(offset) + d)])
                    for d in sorted(range(-RIG_FRAME_S, RIG_FRAME_S + 1), key=abs)
                    if t + _dt.timedelta(seconds=int(offset) + d) in src), None)
        if hit is None:
            continue
        d, f = hit
        Rf = world_to_cam(f.camera_transform)
        R = rig["R"] @ Rf
        C = np.asarray(f.coords, float) + Rf.T @ rig["t"]
        aim = aim_of(R, C, pcd)
        poses[cid] = dict(R=R, C=C, aim=aim, range=None if aim is None else float(np.linalg.norm(aim - C)),
                          inliers=0, error=None, frame=f, dt=d)
        why = "no match" if s == "no pose" else f"not matched well: {note}"
        status[cid] = ("estimated", f"{why}; from frame {f.filename} ({d:+d} s) and the rig")
    return {k: v for k, v in rig.items() if k in ("n", "spread_m", "spread_deg")}


def _has_frame(target, src: Dict[_dt.datetime, Any], offset: int, within: int) -> bool:
    """Whether ``src`` (posed frames by time) has a frame within ``within`` s of ``target``'s time + ``offset``."""
    t = parse_time(getattr(target, "datetime", None))
    return t is not None and any(t + _dt.timedelta(seconds=int(offset) + d) in src for d in range(-within, within + 1))


def unaligned_frames(targets: Sequence[Any], frames_all: Sequence[Any], offset: int, within: int) -> List[Any]:
    """The frames (of ``frames_all``, all the frame cameras, aligned or not) within ``within`` s of the
    ``targets``' times + ``offset`` that have no pose: aligning them in Metashape would let those close-ups be
    posed. Sorted by time."""
    from .cameras import can_project

    by_t = by_time(frames_all)
    out = {}
    for c in targets:
        t = parse_time(getattr(c, "datetime", None))
        if t is None:
            continue
        for d in range(-within, within + 1):
            ts = t + _dt.timedelta(seconds=int(offset) + d)
            f = by_t.get(ts)
            if f is not None and not can_project(f, use_orig_coords=False):
                out[ts] = f
    return [out[ts] for ts in sorted(out)]


# --------------------------------------------------------------------------- outputs


def poses_payload(report: Dict[str, Any], targets: Sequence[Any], *, pose_source=None, updated_target=None) -> dict:
    """The poses file (``<id>_camsync_poses.json``) for Metashape (``metashape_scripts/import_camera_poses.py``):
    per close-up (by its camera key, as in the meta JSON) its ``label``, ``path``, ``status`` / ``note`` and, when
    posed, ``center`` and ``transform`` in the chunk's own frame (the cameras' original frame, as Metashape stores
    a camera's transform). Read after :func:`set_pose`."""
    cams = {}
    for c in targets:
        cid = str(c.cam_id)
        status, note = report["status"].get(cid, ("no pose", ""))
        m = report["matches"].get(cid) or {}
        p = report["poses"].get(cid) or {}
        row = dict(label=os.path.splitext(c.filename)[0], path=c.filepath, status=status, note=note,
                   matches=int(m.get("n", 0)))
        if m.get("source") is not None:
            row["frame"] = m["source"].filename
            if getattr(m["source"], "framepose_chained", False):
                row["frame_posed_by"] = "substrata"
        if p:
            row.update(inliers=p["inliers"], error_px=None if p.get("error") is None else round(p["error"], 1),
                       range_m=None if p.get("range") is None else round(p["range"], 4))
        if status == "estimated" and p.get("frame") is not None:
            row["frame"] = p["frame"].filename
        if status in POSED_STATUSES:
            row["center"] = np.asarray(c.orig_coords, float).tolist()
            row["transform"] = np.asarray(c.orig_camera_transform, float).reshape(4, 4).tolist()
        cams[cid] = row
    off = report["offset"]
    return dict(type="camsync_match_images", pose_source=pose_source, updated_target=updated_target,
                time_offset_sec=report["time_offset"], offset_prominence=off.get("prominence"),
                n_posed=sum(r["status"] == "posed" for r in cams.values()),
                n_marked=sum(r["status"] in ("unverified", "estimated") for r in cams.values()),
                n_cameras=len(cams), cameras=cams)


def _thumb(path: str, size: int):
    from PIL import Image

    im = Image.open(path)
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    im.thumbnail((size, size))
    return im


def write_report_pdf(report: Dict[str, Any], targets: Sequence[Any], path: str, *, title: str = "",
                     max_posed: int = 9, per_page: int = 3) -> str:
    """The run's quality check: a summary page with the offset's scores, then close-ups next to their frame (the
    close-up's centre mapped into the frame by the matches' homography: red; where its pose aims: green). Every
    close-up that was not posed is shown, and up to ``max_posed`` posed ones (spread over the run)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    off = report["offset"]
    status = report["status"]
    counts: Dict[str, int] = {}
    for s, _n in status.values():
        counts[s] = counts.get(s, 0) + 1
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        lines = [title or "camsync --match-images", "",
                 f"Time offset: {report['time_offset']} s (a close-up taken at t is in the frame of t + offset)"
                 if report["time_offset"] is not None else f"No time offset: {off.get('reason', '')}",
                 f"  from the images: peak {off.get('offset')} s, {off.get('prominence', 0):.1f}x the median vote, "
                 f"{off.get('n_agree', 0)} of {off.get('n_targets', 0)} close-ups agree; offsets tried "
                 f"{off.get('window')}",
                 "  confirmed with the learned matcher: " + ", ".join(
                     f"{r['matched']} of {r['sampled']} close-ups match at {r['offset']} s"
                     for r in (report.get("confirm") or {}).get("tried", [])) if (report.get("confirm") or {}).get("tried")
                 else "",
                 f"  close-up centre in a frame ~{tuple(int(v) for v in report['aim'])} px, ~{report['width_px']:.0f} px "
                 f"wide, turned {off.get('rotation_deg') or 0:.0f} deg (matched turned {90 * report.get('turns', 0)} deg)"
                 if report.get("aim") else "",
                 "  unaligned frames posed by substrata (--pose-frames): {} of {}".format(
                     sum(bool(r.get("used")) for r in report["frame_poses"].values()), len(report["frame_poses"]))
                 if report.get("frame_poses") else "",
                 "  rig (close-up relative to its frame) from {n} posed close-ups: spread {a:.1f} cm, {b:.1f} deg; "
                 "close-ups without a good match estimated from it".format(
                     n=report["rig"]["n"], a=100 * report["rig"]["spread_m"], b=report["rig"]["spread_deg"])
                 if report.get("rig") else "", "",
                 "Close-ups: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]
        rng = [p["range"] for cid, p in report["poses"].items() if p and status.get(cid, ("",))[0] == "posed"]
        if rng:
            lines.append(f"  posed: camera to reef {np.median(rng):.2f} m (median; {min(rng):.2f}-{max(rng):.2f})")
        fig.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
        if off.get("ks") is not None and len(off["ks"]):
            ax = fig.add_axes([0.08, 0.08, 0.86, 0.45])
            ax.plot(off["ks"], off["curve"], lw=0.8)
            if off.get("offset") is not None:
                ax.axvline(off["offset"], color="red", lw=0.8, ls="--")
            ax.set_xlabel("offset (s): frame time - close-up time")
            ax.set_ylabel("votes (each close-up's matches / its best)")
        pdf.savefig(fig)
        plt.close(fig)

        rows = sorted(targets, key=lambda c: c.filename)
        posed = [c for c in rows if status.get(str(c.cam_id), ("",))[0] == "posed"]
        keep = {str(c.cam_id) for c in rows if status.get(str(c.cam_id), ("",))[0] != "posed"}
        if rng:  # (and the posed ones at an unusual range)
            med = float(np.median(rng))
            keep |= {str(c.cam_id) for c in posed if (report["poses"].get(str(c.cam_id)) or {}).get("range")
                     and not (med / 1.5 <= report["poses"][str(c.cam_id)]["range"] <= 1.5 * med)}
        if posed and max_posed > 0:
            keep |= {str(posed[i].cam_id) for i in np.linspace(0, len(posed) - 1, min(max_posed, len(posed))).round()
                     .astype(int)}
        show = [c for c in rows if str(c.cam_id) in keep]
        for start in range(0, len(show), per_page):
            fig, axes = plt.subplots(per_page, 2, figsize=(8.3, 11.7))
            for r, c in enumerate(show[start:start + per_page]):
                cid = str(c.cam_id)
                s, note = status.get(cid, ("no pose", ""))
                m = report["matches"].get(cid) or {}
                p = report["poses"].get(cid) or {}
                a0, a1 = axes[r]
                a0.imshow(_thumb(c.filepath, 700))
                head = f"{c.filename}: {s}" + (f" ({note})" if note else "")
                info = f"{m.get('n', 0)} matches"
                if p and s != "estimated":
                    info += f", {p['inliers']} pose inliers, error {p['error']:.0f} px" + (
                        f", range {p['range']:.2f} m" if p.get("range") else "")
                elif p:
                    info += f"; estimated: range {p['range']:.2f} m" if p.get("range") else "; estimated"
                a0.set_title(head + "\n" + info, fontsize=7, loc="left")
                src = m.get("source")
                if s == "estimated" and p.get("frame") is not None and p.get("aim") is not None:
                    src = p["frame"]
                    x, y, *_ = src.get_pixel_coords(p["aim"])
                    big = _thumb(src.filepath, max(src.sensor.width, src.sensor.height))
                    if x is not None:
                        half = 1.5 * max(report.get("width_px") or 400, 200)
                        box = (int(x - half), int(y - 0.6 * half), int(x + half), int(y + 0.6 * half))
                        a1.imshow(big.crop(box), extent=(box[0], box[2], box[3], box[1]))
                        a1.plot(x, y, "x", color="lime", ms=10, mew=2)
                    else:
                        a1.imshow(big)
                    a1.set_title(f"{src.filename} ({p.get('dt', 0):+d} s from the offset time): where the estimate "
                                 "aims (green)", fontsize=7, loc="left")
                elif src is not None and m.get("H") is not None:
                    hc = map_point(m["H"], (c.sensor.width / 2, c.sensor.height / 2))
                    big = _thumb(src.filepath, max(src.sensor.width, src.sensor.height))
                    half = 1.5 * max(report.get("width_px") or 400, 200)
                    box = (int(hc[0] - half), int(hc[1] - 0.6 * half), int(hc[0] + half), int(hc[1] + 0.6 * half))
                    a1.imshow(big.crop(box), extent=(box[0], box[2], box[3], box[1]))
                    a1.plot(*hc, "+", color="red", ms=14, mew=2)
                    if p and p.get("aim") is not None:
                        x, y, *_ = src.get_pixel_coords(p["aim"])
                        if x is not None:
                            a1.plot(x, y, "x", color="lime", ms=10, mew=2)
                    a1.set_title(f"{src.filename} ({m.get('dt', 0):+d} s from the offset time)", fontsize=7, loc="left")
                for a in (a0, a1):
                    a.set_xticks([])
                    a.set_yticks([])
            for r in range(len(show[start:start + per_page]), per_page):
                axes[r][0].axis("off")
                axes[r][1].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return path
