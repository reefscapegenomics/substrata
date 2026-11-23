#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Import camera poses from substrata into a Metashape ``.psx`` project.

Runs under **Metashape's bundled Python** and imports only :mod:`Metashape` plus the standard library (as
``export_project.py``). It is invoked by ``substrata metashape-import-poses``, or directly from the project folder::

    metashape.sh -platform offscreen -r import_camera_poses.py --poses <id>_camsync_poses.json

The poses file is written by ``substrata camsync --match-images``: per camera (by its camera key, as in the
exported ``<id>.meta.json``) its ``label``, ``status`` and, when posed, its ``transform`` in the chunk's own frame
(the frame of Metashape's ``camera.transform``). Each posed camera gets that transform; a camera whose label in the
project differs from the file's is skipped (the file belongs to another project or export). Cameras marked
``estimated`` (no match: from their frame and the rig) or ``unverified`` (the matchers disagreed) get theirs too, and
are disabled, so they can be found and checked (``--posed-only``: not those). Cameras without a pose are left as
they are. The project is saved unless ``--dry-run``.

A camera given a pose this way is not aligned by Metashape: its pose comes from the matches substrata found
(e.g. a close-up against the GoPro frames). Metashape's ``Optimize Cameras`` would refine it only if it had tie
points, which unaligned photos have not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# --------------------------------------------------------------------------- #
# Pure helpers (no Metashape dependency -- unit-testable)
# --------------------------------------------------------------------------- #


MARKED = ("estimated", "unverified")  # statuses that come with a pose not to be trusted as it is


def planned_poses(payload: dict, include_marked: bool = True) -> list:
    """``[(camera key, label, 4x4 transform as a list of rows)]`` of the posed cameras in a poses file's
    payload (and, with ``include_marked``, those marked :data:`MARKED`), in key order."""
    out = []
    ok = ("posed",) + (MARKED if include_marked else ())
    for key, row in sorted((payload.get("cameras") or {}).items(), key=lambda kv: str(kv[0])):
        if row.get("status") not in ok or not row.get("transform"):
            continue
        t = row["transform"]
        if len(t) != 4 or any(len(r) != 4 for r in t):
            raise ValueError(f"Camera {key}: transform is not 4 x 4")
        out.append((str(key), row.get("label"), [[float(v) for v in r] for r in t]))
    return out


def default_paths(cwd: str, psx=None, poses=None) -> tuple:
    """(.psx, poses file): by default ``<folder>.psx`` and ``<folder>_camsync_poses.json`` in ``cwd``."""
    pid = os.path.basename(os.path.normpath(cwd))
    return (psx or os.path.join(cwd, f"{pid}.psx"), poses or os.path.join(cwd, f"{pid}_camsync_poses.json"))


def _select_chunk(doc, chunk_arg):
    chunks = list(doc.chunks or [])
    if not chunks:
        raise RuntimeError("No chunks found in project.")
    if chunk_arg is None:
        return doc.chunk or chunks[0]
    try:
        idx = int(chunk_arg)
        if 0 <= idx < len(chunks):
            return chunks[idx]
        raise RuntimeError(f"Chunk index {idx} out of range (0..{len(chunks) - 1}).")
    except ValueError:
        pass
    for ch in chunks:
        if ch.label == chunk_arg:
            return ch
    raise RuntimeError(f"No chunk labelled {chunk_arg!r}.")


def main(argv=None) -> None:
    args = _parse_args(argv)
    psx, poses_path = default_paths(os.getcwd(), args.psx, args.poses)
    for p in (psx, poses_path):
        if not os.path.isfile(p):
            print(f"Error: {p} does not exist", file=sys.stderr)
            sys.exit(1)
    with open(poses_path) as f:
        payload = json.load(f)
    plan = planned_poses(payload, include_marked=not args.posed_only)
    marked = {str(k) for k, r in (payload.get("cameras") or {}).items() if r.get("status") in MARKED}
    if not plan:
        print(f"No posed cameras in {poses_path}: nothing to import.")
        return

    import Metashape as ms

    doc = ms.Document()
    doc.open(psx, read_only=bool(args.dry_run))
    chunk = _select_chunk(doc, args.chunk)
    by_key = {str(c.key): c for c in chunk.cameras or [] if c is not None}
    n_set, n_marked, skipped = 0, 0, []
    for key, label, t in plan:
        cam = by_key.get(key)
        if cam is None:
            skipped.append(f"{key} ({label}): no such camera")
            continue
        if label and cam.label != label:
            skipped.append(f"{key}: label {cam.label!r} in the project, {label!r} in the file")
            continue
        cam.transform = ms.Matrix(t)
        n_set += 1
        if key in marked:
            cam.enabled = False  # (estimated / unverified: found among the disabled cameras, to check)
            n_marked += 1
    for s in skipped:
        print(f"  skipped {s}")
    print(f"{'Would set' if args.dry_run else 'Set'} the pose of {n_set} of {len(plan)} camera(s) in chunk "
          f"{chunk.label!r}" + (f" ({n_marked} marked estimated / unverified, disabled)" if n_marked else ""))
    if not args.dry_run and n_set:
        doc.save()
        print(f"Saved {psx}")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psx", default=None, help="The Metashape project (default: <folder>.psx in the CWD).")
    parser.add_argument("--poses", default=None,
                        help="The poses file (default: <folder>_camsync_poses.json in the CWD).")
    parser.add_argument("--chunk", default=None, help="Chunk label or 0-based index (default: active/first).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; the project is opened read-only.")
    parser.add_argument("--posed-only", action="store_true",
                        help="Only the posed cameras, not those marked estimated / unverified.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
