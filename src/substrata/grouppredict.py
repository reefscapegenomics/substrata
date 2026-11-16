"""Predict an annotation set's points from the photos of one chosen camera group.

``substrata predict --camera-group GROUP`` (:func:`predict_from_group`, :func:`write_results`): each point of
the set is classified from the views of that group's cameras only, disabled cameras included (e.g. the AUV
photos aligned into a diver plot, which the plot's cams.xml marks disabled so that they are left out of its
point cloud). The results go to a separate CSV (:func:`output_path`); the set's own CSV (labels, training
views, its predictions) is not touched.

The camera group (:func:`resolve_camera_group`) is a camera type (``assign_camera_types``: one resolution,
focal length within 3%; e.g. ``"2848x2816 f3153"``), a camera's ``group`` (from its photo folder and file name
prefix, e.g. ``"auv"``), ``enabled`` (the cameras ``substrata segment`` uses) or ``all``.

Two ways to choose and combine views (``mode``):

``segment``
    As ``substrata segment`` classifies its query points (:func:`substrata.segmentation.classify_query_points`):
    up to ``settings.SEG_POINT_VIEWS`` views by relevance, the reprojection occlusion test on the plot's point
    cloud, no photo-edge filter; a point-feature classifier fuses the views' dense per-photo maps (mean of the
    probabilities weighted by relevance_min / relevance), a crop classifier uses the best view. So a point's
    prediction is what the cover map built from that group's photos gives at it.

``predict``
    As ``substrata predict`` (:meth:`Annotations.predict`): up to ``settings.MAX_PHOTOS_TO_PREDICT_DEFAULT``
    views by relevance, the stratified occlusion test (else the cloud's), only views whose
    ``settings.CROP_SIZE_DEFAULT`` crop fits in the photo; the relevance-weighted mean of the views'
    probabilities, obstructed views left out unless all are.

Per point the output has ``views_available`` (the group's cameras that have the point in view, before the
occlusion test, the edge filter and the limit of views) and ``pred_n`` (the views used): a point with no view
has an empty ``pred_label``.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import settings

MODES = ("segment", "predict")
OUTPUT_DIR = "predictions"  # in the set's .annos folder (not a set itself: substrata-explore reads .annos/*.csv)
COLUMNS = ("id", "label", "pred_label", "pred_conf", "pred_n", "pred_agree", "pred_top", "views_available",
           "view_filepath", "view_x", "view_y", "camera_group", "view_mode", "pred_model")


def camera_groups(cams) -> Dict[str, Dict[str, Any]]:
    """The camera groups a run can use: ``{name: {"kind", "n", "enabled"}}``, camera types first (most cameras
    first), then the cameras' ``group`` names, then ``enabled`` and ``all``."""
    from .cameras import assign_camera_types

    cl = list(cams.data.values()) if hasattr(cams, "data") else list(cams)
    out: Dict[str, Dict[str, Any]] = {}
    on = lambda c: getattr(c, "enabled", True) is not False  # noqa: E731
    for t in assign_camera_types(cl):
        m = [c for c in cl if getattr(c, "camera_type", None) == t]
        out[t] = dict(kind="type", n=len(m), enabled=sum(map(on, m)))
    groups = sorted({str(getattr(c, "group", "") or "") for c in cl} - {""})
    folders = sorted({g.split("_", 1)[0] for g in groups if "_" in g} - set(groups))
    for g in folders + groups:  # (a photo folder of several file-name groups, e.g. "auv" of "auv_PR")
        if g in out:
            continue
        m = _in_group(cl, g)
        out[g] = dict(kind="group", n=len(m), enabled=sum(map(on, m)))
    out["enabled"] = dict(kind="enabled", n=sum(map(on, cl)), enabled=sum(map(on, cl)))
    out["all"] = dict(kind="all", n=len(cl), enabled=sum(map(on, cl)))
    return out


def _in_group(cl: List[Any], name: str) -> List[Any]:
    """The cameras whose ``group`` is ``name``, or a group of photo folder ``name`` (``name_<prefix>``)."""
    def hit(c):
        g = getattr(c, "group", None) or ""
        return g == name or g.startswith(name + "_")
    return [c for c in cl if hit(c)]


def resolve_camera_group(cams, spec: str) -> Tuple[str, str, List[Any]]:
    """The cameras of camera group ``spec``: ``(name, kind, cameras)``.

    ``spec``: a camera type (``"2848x2816 f3153"``), a camera ``group`` (``"auv_PR"``; or a photo folder's groups,
    ``"auv"`` for ``auv_PR`` and ``auv_X``), ``"enabled"`` (the enabled cameras, as ``substrata segment`` uses them), ``"all"``, or a camera system: a sensor resolution as the types
    name it (``"2848x2816"``: every type of that resolution, whatever its focal length; substrata-explore chooses
    cameras so across plots); ``"type:…"`` / ``"group:…"`` / ``"system:…"`` say which when a name is several.
    Raises ``ValueError`` (listing the groups) for an unknown one."""
    from .cameras import assign_camera_types

    cl = list(cams.data.values()) if hasattr(cams, "data") else list(cams)
    assign_camera_types(cl)
    spec = (spec or "").strip()
    kind, _, name = spec.partition(":") if spec.split(":", 1)[0] in ("type", "group", "system") else ("", "", spec)
    if kind == "system" or (not kind and re.fullmatch(r"\d+x\d+", name)):
        m = [c for c in cl if (getattr(c, "camera_type", None) or "").split(" ")[0] == name]
        if m:
            return name, "system", m
    if not kind and name in ("enabled", "all"):
        m = cl if name == "all" else [c for c in cl if getattr(c, "enabled", True) is not False]
        return name, name, m
    if kind in ("", "type"):
        m = [c for c in cl if getattr(c, "camera_type", None) == name]
        if m:
            return name, "type", m
    if kind in ("", "group"):
        m = _in_group(cl, name)
        if m:
            return name, "group", m
    known = ", ".join(f"{k!r} ({v['kind']}, {v['n']} cameras, {v['enabled']} enabled)"
                      for k, v in camera_groups(cl).items())
    raise ValueError(f"No camera group {spec!r}. Camera groups: {known}")


def _points(anns, ids: Optional[Iterable[str]] = None) -> List[Any]:
    """The annotations with a 3-D point (all, or ``ids``)."""
    keep = None if ids is None else set(ids)
    out = []
    for a in anns.data.values():
        p = getattr(a, "orig_coords", None)
        if (keep is None or a.id in keep) and p is not None and np.isfinite(np.asarray(p, float)).all():
            out.append(a)
    return out


def predict_from_group(
    anns,
    group_cams: List[Any],
    classifier,
    *,
    mode: str = "segment",
    views: Optional[int] = None,
    pcd=None,
    occluder=None,
    world_transform: Optional[np.ndarray] = None,
    crop_size=None,
    clear_of_edge=None,
    ids: Optional[Iterable[str]] = None,
    batch_size: int = 64,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Classify the points of ``anns`` from the views of ``group_cams`` (disabled cameras included).

    Args:
        anns: The annotation set (:class:`substrata.annotations.Annotations`).
        group_cams: The group's cameras (:func:`resolve_camera_group`).
        classifier: Loaded classifier (``classification.get_image_classifier``).
        mode: ``"segment"`` or ``"predict"`` (see the module docstring).
        views: Views per point (default: ``SEG_POINT_VIEWS`` / ``MAX_PHOTOS_TO_PREDICT_DEFAULT``).
        pcd: The plot's point cloud (world frame): segment mode's occlusion test (None: no occlusion test);
            predict mode's when there is no ``occluder``.
        occluder: predict mode: the stratified occlusion test (``stratified.ply_occluder``).
        world_transform: The frame of ``pcd`` (default ``pcd.world_transform``; segment mode).
        crop_size: The classifier's crop size (crop classifiers).
        clear_of_edge: predict mode: crop size ``(w, h)`` a view's crop must fit (default
            ``settings.CROP_SIZE_DEFAULT``).
        ids: Only these points (default: all with a 3-D point).

    Returns:
        ``{id: {"label", "conf", "n", "agree", "top", "available", "view": (path, x, y) or None}}``; ``label``
        None where no view was classified.
    """
    from .cameras import pixel_hits

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    points = _points(anns, ids)
    out: Dict[str, Dict[str, Any]] = {}
    if not points or not group_cams:
        for a in points:
            out[a.id] = dict(label=None, conf=None, n=0, agree=None, top=None, available=0, view=None)
        return out

    if mode == "segment":
        from .annotations import format_pred_top
        from .segmentation import classify_query_points

        if world_transform is None and pcd is not None:
            world_transform = getattr(pcd, "world_transform", None)
        orig = np.array([np.asarray(a.orig_coords, float) for a in points])
        if world_transform is not None:
            W = np.asarray(world_transform, float)
            qc = orig @ W[:3, :3].T + W[:3, 3]
        else:
            qc = orig
        res = classify_query_points(
            qc, list(group_cams), classifier, world_transform=world_transform, pcd=pcd,
            occlusion=pcd is not None, views=views or settings.SEG_POINT_VIEWS,
            crop_size=None if crop_size is None else int(np.atleast_1d(crop_size)[0]),
            batch_size=batch_size, include_disabled=True, verbose=verbose,
        )
        classes = res["classes"]
        for i, a in enumerate(points):
            lab = res["labels"][i] or None
            top = None
            if lab and res["probs"] is not None:
                top = format_pred_top({c: float(p) for c, p in zip(classes, res["probs"][i])})
            elif lab:
                top = format_pred_top({lab: float(res["conf"][i])})
            agree = res["agreement"][i] if res["agreement"] is not None else (1.0 if lab else None)
            view = ((res["cam_paths"][i], float(res["x"][i]), float(res["y"][i]))
                    if lab and res["cam_paths"][i] else None)
            out[a.id] = dict(label=lab, conf=float(res["conf"][i]) if lab else None,
                             n=int(res["n_views"][i]) if lab else 0,
                             agree=None if agree is None or not np.isfinite(agree) else round(float(agree), 2),
                             top=top, available=int(res["available"][i]), view=view)
        return out

    # predict mode: substrata predict's views and consensus, from the group's cameras (pred_agree, pred_top:
    # as Annotation._apply_predictions sets them; only the in-memory annotations, the CSV is not written)
    group = list(group_cams)
    available = {a.id: len(pixel_hits(group, np.asarray(a.orig_coords, float), use_orig_coords=True,
                                       enabled_only=False)) for a in points}
    keep = [a.id for a in points]
    consensus = anns.predict(
        group, classifier,
        crop_size=crop_size if crop_size is not None else settings.PREDICTION_CROP_SIZE_DEFAULT,
        max_cams=views or settings.MAX_PHOTOS_TO_PREDICT_DEFAULT,
        pcd=pcd if occluder is None else None, occluder=occluder, main_type=None,
        clear_of_edge=clear_of_edge if clear_of_edge is not None else settings.CROP_SIZE_DEFAULT,
        per_other_type=0, ids=keep, write_crops=False, batch_size=batch_size, n_jobs=n_jobs,
        enabled_cameras_only=False, progress=verbose,
    )
    for a in points:
        c = consensus.get(a.id)
        view = getattr(a, "training_view", None)  # (select_views: the most relevant unobstructed view)
        if c:
            out[a.id] = dict(label=c["label"], conf=float(c["confidence"]), n=int(c["n"]),
                             agree=getattr(a, "pred_agree", None), top=getattr(a, "pred_top", None),
                             available=available[a.id],
                             view=(view.cam.filepath, float(view.x), float(view.y)) if view is not None else None)
        else:
            out[a.id] = dict(label=None, conf=None, n=0, agree=None, top=None, available=available[a.id],
                             view=None)
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "group"


def output_path(csv_path: str, group: str, mode: str, tag: Optional[str] = None) -> str:
    """``<set's folder>/predictions/<set stem>__<group>__<mode>[__<tag>].csv``."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    name = f"{stem}__{_slug(group)}__{mode}" + (f"__{_slug(tag)}" if tag else "") + ".csv"
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), OUTPUT_DIR, name)


def write_results(out_csv: str, anns, results: Dict[str, Dict[str, Any]], *, group: str, mode: str,
                  model: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write :func:`predict_from_group`'s results (one row per point of the set, in its order; :data:`COLUMNS`)
    and ``<out_csv>.json`` (``meta`` plus counts). Returns the counts."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    fmt = lambda v, f="{:.4f}": "" if v is None else f.format(v)  # noqa: E731
    n = dict(points=0, with_view_available=0, predicted=0, no_view=0)
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for a in anns.data.values():
            r = results.get(a.id)
            if r is None:
                continue
            n["points"] += 1
            n["with_view_available"] += r["available"] > 0
            n["predicted"] += r["label"] is not None
            n["no_view"] += r["label"] is None
            v = r["view"]
            w.writerow([a.id, getattr(a, "label", None) or "", r["label"] or "", fmt(r["conf"]),
                        r["n"], fmt(r["agree"], "{:.2f}"), r["top"] or "", r["available"],
                        v[0] if v else "", fmt(v[1] if v else None, "{:.2f}"), fmt(v[2] if v else None, "{:.2f}"),
                        group, mode, model or ""])
    os.replace(tmp, out_csv)
    from . import __version__

    info = dict(meta or {}, camera_group=group, view_mode=mode, pred_model=model, counts=n,
                substrata=__version__, created=datetime.now().isoformat(timespec="seconds"))
    with open(out_csv + ".json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    return n
