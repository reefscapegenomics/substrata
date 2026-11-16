"""Training views written before substrata 0.2's lens model, moved to where the corrected model projects them.

Until 0.2 substrata projected with P1 and P2 the other way round from Metashape (OpenCV's order) and without K4
and B2 (affinity B1 was there; see ``cameras._lens``). The training view an annotation CSV stores (``cam_filepath``,
``cam_x``, ``cam_y``: the photo and pixel its training crop is cut at) was worked out with that model, a few pixels
from the right one on most sensors and up to hundreds near the edges of a badly calibrated one. ``substrata
reproject-views`` (:func:`reproject_views`) moves them: for each row with a 3-D point (``orig_x/y/z``) and a training
view in a photo of the plot, the pixel is projected again with the corrected model, when the stored one is where the
old model put the point (within ``tol_px``: a view written by substrata or substrata-explore from the point). Rows
whose pixel is somewhere else (a view placed by hand on another photo's pixel, one from another plot or model) are
left as they are and counted; rows already at the corrected pixel are left too, so it can run again.

A view the corrected model does not see the point in at all (``out_of_view``) is cleared (``cam_filepath``,
``cam_x``, ``cam_y`` emptied) when the old model wrote it: mostly points far outside the photo's field that a lens
folding back near its edge put inside the image (on sa3_20m's PGRAM02 points ~50 degrees off the axis landed in the
middle of the photo, half of that camera's training views), and a few just over the image border now. Their crop
showed another place than the point. A cleared row gets a new training view from the cameras that do see it where
views are written for points without one (``substrata crops``, substrata-explore's crop job); its label may have been
given looking at the wrong crop: the ids are listed (``backup/<csv>_cleared_views_<time>.txt``) to look at again. A
row left out of training keeps its view (writing a new one would include it again); ``keep_out_of_view`` keeps all.

Nothing else in the row changes (labels, included in training, predictions). The CSV is backed up first
(``backup/``); its grouping features (``<csv stem>.features.npz``, substrata-explore: made at the old pixels) go to
``backup/`` too. Crops named after the old pixels are not used any more (the crops / substrata-explore's crop job make
them anew and prune the stale ones); a classifier trained on the old pixels is best trained again.
"""

from __future__ import annotations

import glob
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import annotations as annotations_mod
from . import settings
from .cameras import _distort

# A stored pixel within this many px of where the old model put the 3-D point was written from it (substrata stores
# the rounded pixel: up to 0.71 px from the unrounded projection).
TOL_PX = 1.0
FEATURES_SUFFIX = ".features.npz"  # substrata-explore's grouping features next to a set's CSV


def legacy_pixels(xc, yc, zc, s):
    """The pixels substrata before 0.2 gave camera-frame points: P1 / P2 in OpenCV's order, no K4, no B2. Only to
    recognise the training views it wrote."""
    x, y = xc / zc, yc / zc
    r2 = x**2 + y**2
    radial = 1 + s.k1 * r2 + s.k2 * r2**2 + s.k3 * r2**3
    x_dist = x * radial + 2 * s.p1 * x * y + s.p2 * (r2 + 2 * x**2)
    y_dist = y * radial + s.p1 * (r2 + 2 * y**2) + 2 * s.p2 * x * y
    return s.fx * x_dist + s.cx, s.fy * y_dist + s.cy


def _camera_index(cams) -> Dict[str, Any]:
    """Cameras by photo file name and by its stem."""
    out: Dict[str, Any] = {}
    for c in getattr(cams, "data", {}).values():
        name = os.path.basename(str(getattr(c, "filepath", "") or getattr(c, "filename", "") or ""))
        if name and getattr(c, "sensor", None) is not None:
            out.setdefault(name, c)
            out.setdefault(os.path.splitext(name)[0], c)
    return out


def _cell(row: List[str], col: Dict[str, int], name: str) -> str:
    i = col.get(name)
    return row[i].strip() if i is not None and i < len(row) else ""


def plan(csv_path: str, cams, tol_px: float = TOL_PX) -> Dict[str, Any]:
    """What :func:`reproject_views` would do to one CSV: ``{"moves": {id: (x, y)}, "shifts": [px], "counts":
    {...}, "clear": [ids]}``. Counts: ``rows``, ``with_view`` (a 3-D point and a training view), ``moved``,
    ``current`` (already at the corrected pixel), ``elsewhere`` (the stored pixel is not where the old model put the
    point: kept), ``not_in_plot`` (the photo is not one of this plot's cameras), ``out_of_view`` (the old model's view,
    not in view by the corrected one: to clear, see the module docstring), ``out_of_view_kept`` (such a view of a row
    left out of training, or one not written by the old model: kept)."""
    rows, _eol, low = annotations_mod._read_csv_rows(csv_path)
    col = {c: i for i, c in enumerate(low)}
    counts = dict(rows=len(rows) - 1, with_view=0, moved=0, current=0, elsewhere=0, not_in_plot=0, out_of_view=0,
                  out_of_view_kept=0)
    moves: Dict[str, tuple] = {}
    clear: List[str] = []
    shifts: List[float] = []
    need = ("id", "orig_x", "orig_y", "orig_z", "cam_filepath", "cam_x", "cam_y")
    if not all(c in col for c in need):
        return dict(moves=moves, shifts=shifts, counts=counts, clear=clear)
    inc = col.get(settings.TRAIN_INCLUDE_COLUMN.lower())
    index = _camera_index(cams)
    for r in rows[1:]:
        try:
            xyz = np.array([float(_cell(r, col, c)) for c in ("orig_x", "orig_y", "orig_z")])
            sx, sy = float(_cell(r, col, "cam_x")), float(_cell(r, col, "cam_y"))
        except ValueError:
            continue
        fp = _cell(r, col, "cam_filepath")
        if not fp or not np.isfinite(xyz).all():
            continue
        counts["with_view"] += 1
        name = fp.replace("\\", "/").rsplit("/", 1)[-1]
        cam = index.get(name) or index.get(os.path.splitext(name)[0])
        if cam is None:
            counts["not_in_plot"] += 1
            continue
        _centre, pose = cam._projection_pose(True)
        p = np.linalg.inv(pose) @ np.append(xyz, 1.0)
        s = cam.sensor
        with np.errstate(divide="ignore", invalid="ignore"):
            nx, ny, ok = _distort(p[0], p[1], p[2], s)
            ox, oy = legacy_pixels(p[0], p[1], p[2], s)
        if not ok:
            excluded = inc is not None and _cell(r, col, settings.TRAIN_INCLUDE_COLUMN.lower()
                                                 ).lower() in ("false", "0", "no")
            if np.hypot(sx - ox, sy - oy) <= tol_px and not excluded:
                clear.append(_cell(r, col, "id"))
                counts["out_of_view"] += 1
            else:
                counts["out_of_view_kept"] += 1
            continue
        new = (float(int(round(float(nx)))), float(int(round(float(ny)))))
        if (sx, sy) == new:
            counts["current"] += 1
        elif np.hypot(sx - ox, sy - oy) <= tol_px:
            moves[_cell(r, col, "id")] = new
            shifts.append(float(np.hypot(new[0] - sx, new[1] - sy)))
            counts["moved"] += 1
        else:
            counts["elsewhere"] += 1
    return dict(moves=moves, shifts=shifts, counts=counts, clear=clear)


def set_view_pixels(csv_path: str, pixels: Dict[str, tuple], clear: Iterable[str] = ()) -> int:
    """Set ``cam_x`` / ``cam_y`` of these ids' rows (``%.2f``) and empty the training view (``cam_filepath``,
    ``cam_x``, ``cam_y``) of the ``clear`` ids; every other cell as it is. Returns the rows changed."""
    rows, eol, low = annotations_mod._read_csv_rows(csv_path)
    idc, fc, xc, yc = low.index("id"), low.index("cam_filepath"), low.index("cam_x"), low.index("cam_y")
    clear = set(clear)
    n = 0
    for r in rows[1:]:
        if len(r) <= max(idc, fc, xc, yc):
            continue
        key = r[idc].strip()
        if key in pixels:
            x, y = pixels[key]
            r[xc], r[yc] = f"{float(x):.2f}", f"{float(y):.2f}"
            n += 1
        elif key in clear:
            r[fc] = r[xc] = r[yc] = ""
            n += 1
    if n:
        annotations_mod._write_csv_rows(csv_path, rows, eol)
    return n


def reproject_views(csv_path: str, cams, tol_px: float = TOL_PX, dry_run: bool = False,
                    keep_out_of_view: bool = False) -> Dict[str, Any]:
    """Move one CSV's training views to the corrected model's pixels and clear those it does not see the point
    in (see the module docstring). Returns :func:`plan`'s result plus ``backup`` (the CSV's backup), ``features``
    (the grouping features moved to ``backup/``) and ``cleared_list`` (the cleared ids' file), None for a dry
    run or when nothing changed."""
    out = plan(csv_path, cams, tol_px)
    if keep_out_of_view:
        out["counts"]["out_of_view_kept"] += out["counts"]["out_of_view"]
        out["counts"]["out_of_view"], out["clear"] = 0, []
    out.update(backup=None, features=None, cleared_list=None)
    if dry_run or not (out["moves"] or out["clear"]):
        return out
    out["backup"] = annotations_mod.backup_annotation_csv(csv_path)
    set_view_pixels(csv_path, out["moves"], out["clear"])
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    bdir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "backup")
    if out["clear"]:
        out["cleared_list"] = os.path.join(bdir, f"{os.path.basename(csv_path)}_cleared_views_{stamp}.txt")
        with open(out["cleared_list"], "w") as fh:
            fh.write("\n".join(out["clear"]) + "\n")
    feats = os.path.splitext(csv_path)[0] + FEATURES_SUFFIX
    if os.path.isfile(feats):
        out["features"] = os.path.join(bdir, f"{os.path.basename(feats)}_{stamp}.bak")
        shutil.move(feats, out["features"])
    return out


def default_csvs(plot_dir: str, base: str) -> List[str]:
    """The annotation CSVs of a plot: ``<plot>/<base>.annos/*.csv``."""
    return sorted(glob.glob(os.path.join(plot_dir, f"{base}.annos", "*.csv")))


def summary(name: str, res: Dict[str, Any]) -> str:
    """One CSV's result in a few lines."""
    c, sh = res["counts"], np.asarray(res["shifts"], float)
    head = (f"{name}: {c['with_view']} training views of 3-D points; {c['moved']} moved, {c['out_of_view']} cleared "
            f"(the point is not in that photo by the corrected model), {c['current']} already current, "
            f"{c['elsewhere']} elsewhere (kept), {c['not_in_plot']} on photos of no camera of this plot"
            + (f", {c['out_of_view_kept']} out of view but kept (left out of training, or not the old model's)"
               if c["out_of_view_kept"] else ""))
    if len(sh):
        head += (f"\n  shift: median {np.median(sh):.1f} px, 90% under {np.percentile(sh, 90):.1f} px, max "
                 f"{sh.max():.1f} px; {int((sh > 56).sum())} over 56 px (a quarter of a 224 px crop), "
                 f"{int((sh > 112).sum())} over 112 px (outside it)")
    return head
