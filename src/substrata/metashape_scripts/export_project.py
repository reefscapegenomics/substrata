#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export a substrata project folder from a Metashape ``.psx`` project.

This script runs under **Metashape's bundled Python** and imports only
:mod:`Metashape` plus the standard library -- it must never import
``substrata`` (its heavy dependencies are not installable in Metashape's
interpreter). It is invoked by the ``substrata metashape-export`` CLI command,
but can also be run directly, e.g.::

    metashape.sh -platform offscreen -r export_project.py \\
        --psx /path/to/cur_sna_20m_20200303.psx --output-dir /data/projects

It writes a folder named ``<id>`` (default: the ``.psx`` basename) containing
the files a substrata project expects, all in **raw chunk-local coordinates**
(no CRS/transform applied); substrata computes scale and orientation later:

- ``<id>.cams.xml``     -- ``chunk.exportCameras`` (sensor calibration + cameras)
- ``<id>.meta.json``    -- per-camera path/center/transform (keyed by camera key)
- ``<id>_markers.csv``  -- ``id,x,y,z,label`` from ``chunk.markers``
- ``<id>.ply``          -- point cloud with colour + normals

Adapted from the standalone ``export_substrata_ply.py`` workflow script. The
``import Metashape`` is deferred into the functions that need it so the pure
helpers below can be imported and unit-tested without Metashape installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# --------------------------------------------------------------------------- #
# Pure helpers (no Metashape dependency -- unit-testable)
# --------------------------------------------------------------------------- #


def default_project_id(psx_path: str) -> str:
    """Return the default project id: the ``.psx`` basename without extension."""
    base = os.path.basename(os.path.normpath(psx_path))
    if base.lower().endswith(".psx"):
        base = base[: -len(".psx")]
    return base


def project_layout(output_dir: str, project_id: str) -> dict:
    """Return the on-disk paths substrata expects for a project ``<id>``.

    Args:
        output_dir: Parent directory that will contain the ``<id>`` folder.
        project_id: Project id (folder name and filename stem).

    Returns:
        Mapping with keys ``folder``, ``cams_xml``, ``meta_json``, ``markers``
        and ``ply``.
    """
    folder = os.path.join(output_dir, project_id)
    return {
        "folder": folder,
        "cams_xml": os.path.join(folder, f"{project_id}.cams.xml"),
        "meta_json": os.path.join(folder, f"{project_id}.meta.json"),
        "markers": os.path.join(folder, f"{project_id}_markers.csv"),
        "ply": os.path.join(folder, f"{project_id}.ply"),
    }


def build_meta_dict(
    cameras: dict, crs_authority=None, chunk_transform=None
) -> dict:
    """Assemble the ``.meta.json`` payload from already-extracted camera dicts.

    Args:
        cameras: Mapping of ``str(camera.key)`` to a per-camera dict (with keys
            ``path``, ``center``, ``transform``, ``enabled`` and optional
            ``reference``/``reference_accuracy``/``center_crs``).
        crs_authority: Optional chunk CRS authority string (informational only;
            ignored by substrata).
        chunk_transform: Optional 4x4 chunk transform (informational only).

    Returns:
        Dict with top-level ``crs``, ``chunk_transform`` and ``cameras`` keys.
    """
    return {
        "crs": crs_authority,
        "chunk_transform": chunk_transform,
        "cameras": dict(cameras),
    }


def format_marker_row(key, x, y, z, label) -> str:
    """Format one marker CSV row as ``id,x,y,z,label`` (no trailing newline)."""
    return f"{key},{x},{y},{z},{label}"


MARKERS_HEADER = "id,x,y,z,label"


# --------------------------------------------------------------------------- #
# Metashape-dependent helpers
# --------------------------------------------------------------------------- #


def _mat4_to_list(matrix):
    """Convert a Metashape 4x4 ``Matrix`` to a list of lists (row-major)."""
    return [list(matrix.row(i)) for i in range(matrix.size[1])]


def output_camera_metadata(path, chunk) -> None:
    """Write ``<id>.meta.json`` for the chunk's cameras (raw/local frame).

    Cameras without a resolved ``center`` or ``transform`` are skipped, matching
    the original workflow exporter; substrata tolerates their absence.
    """
    import Metashape as ms  # noqa: F401  (kept for parity / future use)

    cameras: dict = {}
    for cam in chunk.cameras or []:
        if cam is None or cam.center is None or cam.transform is None:
            continue

        center = [cam.center.x, cam.center.y, cam.center.z]
        transform = _mat4_to_list(cam.transform)

        center_crs = None
        if chunk.transform is not None and chunk.crs:
            world = chunk.transform.matrix.mulp(cam.center)
            projected = chunk.crs.project(world)
            if projected:
                center_crs = [projected.x, projected.y, projected.z]

        ref = getattr(cam, "reference", None)
        ref_loc = ref.location if ref else None
        ref_acc = ref.accuracy if ref else None
        reference_xyz = (
            [ref_loc.x, ref_loc.y, ref_loc.z] if ref_loc else None
        )
        reference_acc = (
            [ref_acc.x, ref_acc.y, ref_acc.z] if ref_acc else None
        )

        cam_dict = {
            "path": getattr(getattr(cam, "photo", None), "path", None),
            "center": center,
            "transform": transform,
            "enabled": cam.enabled,
        }
        if reference_xyz is not None:
            cam_dict["reference"] = reference_xyz
        if reference_acc is not None:
            cam_dict["reference_accuracy"] = reference_acc
        if center_crs is not None:
            cam_dict["center_crs"] = center_crs

        cameras[str(cam.key)] = cam_dict

    crs_authority = chunk.crs.authority if chunk.crs else None
    chunk_transform = (
        _mat4_to_list(chunk.transform.matrix) if chunk.transform else None
    )
    payload = build_meta_dict(cameras, crs_authority, chunk_transform)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_markers_to_csv(markers, path) -> None:
    """Write ``<id>_markers.csv`` (``id,x,y,z,label``) in chunk-local coords."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(MARKERS_HEADER + "\n")
        for m in markers or []:
            if not m.position:
                continue
            row = format_marker_row(
                m.key, m.position.x, m.position.y, m.position.z, m.label
            )
            f.write(row + "\n")


def export_cameras(chunk, paths) -> None:
    """Export ``<id>.cams.xml`` and ``<id>.meta.json`` for the chunk."""
    print("Exporting camera positions (.cams.xml)...")
    chunk.exportCameras(paths["cams_xml"])
    print("Exporting camera metadata (.meta.json)...")
    output_camera_metadata(paths["meta_json"], chunk)


def export_markers(chunk, paths) -> None:
    """Export ``<id>_markers.csv`` for the chunk."""
    print("Exporting markers (.csv)...")
    export_markers_to_csv(chunk.markers, paths["markers"])


def export_ply_point_cloud(chunk, paths) -> None:
    """Export ``<id>.ply`` in raw/internal coordinates.

    Temporarily nulls the chunk CRS/transform so the point cloud is written in
    the same raw frame as the cameras and markers, then restores them.
    """
    import Metashape as ms

    print("Exporting point cloud (.ply)...")
    saved_crs = chunk.crs
    saved_transform = chunk.transform

    chunk.crs = None
    chunk.transform = None
    try:
        chunk.exportPointCloud(
            path=paths["ply"],
            source_data=ms.PointCloudData,
            binary=True,
            save_point_normal=True,
            save_point_color=True,
            save_point_classification=False,
            save_point_confidence=True,
            raster_transform=ms.RasterTransformNone,
            colors_rgb_8bit=True,
            format=ms.PointCloudFormatPLY,
            split_in_blocks=False,
        )
    finally:
        chunk.crs = saved_crs
        chunk.transform = saved_transform
        chunk.updateTransform()


def _select_chunk(doc, chunk_arg):
    """Select the chunk to export: by label, by integer index, else active/first."""
    chunks = list(doc.chunks or [])
    if not chunks:
        raise RuntimeError("No chunks found in project.")
    if chunk_arg is None:
        return doc.chunk or chunks[0]
    # Try integer index first, then label match.
    try:
        idx = int(chunk_arg)
        if 0 <= idx < len(chunks):
            return chunks[idx]
        raise RuntimeError(
            f"Chunk index {idx} out of range (0..{len(chunks) - 1})."
        )
    except ValueError:
        pass
    for ch in chunks:
        if ch.label == chunk_arg:
            return ch
    labels = ", ".join(repr(ch.label) for ch in chunks)
    raise RuntimeError(f"No chunk labelled {chunk_arg!r}. Available: {labels}.")


def main(argv=None) -> None:
    """Open a ``.psx`` and export a substrata project folder."""
    # Parse args first so ``--help`` and argument errors work without Metashape.
    args = _parse_args(argv)

    import Metashape as ms

    if not os.path.isfile(args.psx):
        print(f"Error: project file {args.psx} does not exist", file=sys.stderr)
        sys.exit(1)
    if not args.psx.lower().endswith(".psx"):
        print("Error: project file must have a .psx extension", file=sys.stderr)
        sys.exit(1)

    project_id = args.id or default_project_id(args.psx)
    output_dir = args.output_dir or os.getcwd()
    paths = project_layout(output_dir, project_id)

    existing = [
        p
        for k, p in paths.items()
        if k != "folder" and os.path.exists(p)
    ]
    if existing and not args.overwrite:
        joined = "\n  ".join(existing)
        print(
            "Error: output files already exist (pass --overwrite to replace):"
            f"\n  {joined}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(paths["folder"], exist_ok=True)

    print(f"Opening project (read-only): {args.psx}")
    doc = ms.Document()
    try:
        doc.open(args.psx, read_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"Error opening project: {e}", file=sys.stderr)
        sys.exit(1)

    chunk = _select_chunk(doc, args.chunk)
    print(f"Exporting chunk {chunk.label!r} -> {paths['folder']}")

    exports = args.export
    if "cameras" in exports:
        export_cameras(chunk, paths)
    if "markers" in exports:
        export_markers(chunk, paths)
    if "ply" in exports:
        export_ply_point_cloud(chunk, paths)

    print(f"Export completed: {paths['folder']}")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--psx", required=True, help="Path to the Metashape project (.psx)."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent directory for the <id> project folder (default: CWD).",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Project id / folder name (default: .psx basename).",
    )
    parser.add_argument(
        "--chunk",
        default=None,
        help="Chunk label or 0-based index to export (default: active/first).",
    )
    parser.add_argument(
        "--export",
        nargs="+",
        choices=["cameras", "markers", "ply"],
        default=["cameras", "markers", "ply"],
        help="Which outputs to write (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
