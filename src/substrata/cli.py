# Standard Library
import argparse
import ast
from collections import Counter
import os
import re
import subprocess
import sys

# Third-Party Libraries
import yaml
import numpy as np

# Local Modules
from substrata.pointclouds import (
    PointCloud,
    decimate_ply_file,
    ply_head,
    repair_ply_for_open3d,
)
from substrata.initializer import ProjectInitializer
from substrata.annotations import Annotations, Scalebars
from substrata import settings

# ---------------------------- helpers: parents & defaults ----------------------------

## Removed cameras parent; firefish/cams2video now rely on initializer


def _cwd_base():
    cwd = os.getcwd()
    base = os.path.basename(cwd.rstrip(os.sep))
    return base, cwd


def _infer_target_depth(base: str, explicit_depth):
    if explicit_depth is not None:
        return explicit_depth
    m = re.search(r"_(\d+)m_", base)
    if m:
        try:
            return -int(m.group(1))
        except Exception:
            return None
    return None


def _get_output_filepath(init: ProjectInitializer, postfix: str):
    """Get the output filepath from the initializer and the postfix."""
    return os.path.join(init.path or os.getcwd(), f"{init.id}_{postfix}")


def _parse_hex_color(s: str):
    """Parse ``#rrggbb`` (or ``rrggbb``) into a 0-255 ``(r, g, b)`` tuple."""
    h = s.strip().lstrip("#")
    if len(h) != 6:
        raise SystemExit(f"Invalid hex colour {s!r}; expected 6 hex digits.")
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise SystemExit(f"Invalid hex colour {s!r}; expected 6 hex digits.")


def _load_label_colors(path: str):
    """Load a manual label-colour file (``label  #hexcolor`` per line).

    The label is everything up to the last whitespace-separated token, so
    labels may contain spaces. Blank lines and ``#``-only comment lines are
    skipped. A row whose label is ``OTHER`` (case-insensitive) sets the
    catch-all colour for labels not otherwise listed (default ``#999999``).

    Returns:
        Tuple ``(allowed, pil_colors, mpl_colors)`` where ``allowed`` is the
        set of listed labels (excluding ``OTHER``); ``pil_colors`` maps each
        label plus ``"OTHER"`` to a 0-255 ``(r, g, b)`` tuple (for the PIL
        ``OrthoMap`` path); and ``mpl_colors`` maps them to 0-1 tuples (for
        the matplotlib ``OrthoGrid``/animation paths).
    """
    other_rgb = (0x99, 0x99, 0x99)
    pil_colors = {}
    allowed = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            # Skip blanks and full-line comments (a data line is "label #hex",
            # whose label never starts with "#").
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                raise SystemExit(
                    f"Invalid label-colour line {raw!r}; expected 'label #hex'."
                )
            label, hexcode = parts[0].strip(), parts[1]
            rgb = _parse_hex_color(hexcode)
            if label.upper() == "OTHER":
                other_rgb = rgb
                continue
            pil_colors[label] = rgb
            allowed.append(label)
    pil_colors["OTHER"] = other_rgb
    mpl_colors = {
        lbl: tuple(c / 255.0 for c in rgb) for lbl, rgb in pil_colors.items()
    }
    return set(allowed), pil_colors, mpl_colors


def _collapse_labels(anns, allowed):
    """Reassign every annotation whose label is not in *allowed* to ``OTHER``.

    Mutates ``anns`` in place so the grid majority-vote, its legend, the
    animation, and the positions markers all agree. Mirrors ``_resolve_label``
    in ``ortho`` (classifier result preferred, then ``ann.label``).
    """
    for ann in anns.data.values():
        im = getattr(ann, "image_match", None)
        cls = getattr(im, "classification", None)
        if isinstance(cls, dict) and cls.get("label") is not None:
            eff = str(cls["label"])
        else:
            lbl = getattr(ann, "label", None)
            eff = str(lbl) if lbl not in (None, "") else None
        if eff not in allowed:
            ann.label = ann.classification = "OTHER"
            if isinstance(cls, dict):
                cls["label"] = "OTHER"


def _parse_xyz_csv(s: str) -> list[float]:
    """Parse ``x,y,z`` into three floats."""
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise SystemExit(
            f"Expected three comma-separated values for xyz, got {len(parts)} part(s): {s!r}"
        )
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError as e:
        raise SystemExit(f"Invalid xyz values: {s!r} ({e})") from e


def _parse_pose_time_arg(raw: str, fallback_date: str) -> str:
    """Parse a pose-source timestamp string into EXIF format.

    Accepts:
      * ``YYYY:MM:DD HH:MM:SS`` (EXIF form)
      * ``YYYY-MM-DD HH:MM:SS`` (dashed date)
      * ``YYYY-MM-DDTHH:MM:SS`` (ISO-like)
      * ``HH:MM:SS`` (time-only; uses ``fallback_date`` for the date)

    Args:
        raw: User-supplied string.
        fallback_date: Date string (``YYYY:MM:DD`` or ``YYYY-MM-DD``) used
            when ``raw`` is time-only.

    Returns:
        Timestamp formatted as ``"%Y:%m:%d %H:%M:%S"``.

    Raises:
        SystemExit: If ``raw`` cannot be parsed.
    """
    import datetime as _dt

    s = (raw or "").strip()
    if not s:
        raise SystemExit("Empty pose-source timestamp.")

    candidates = (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y:%m:%dT%H:%M:%S",
    )
    for fmt in candidates:
        try:
            return _dt.datetime.strptime(s, fmt).strftime("%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue

    try:
        t = _dt.datetime.strptime(s, "%H:%M:%S").time()
    except ValueError:
        raise SystemExit(
            "Could not parse pose-source timestamp "
            f"{raw!r}.\n  Expected one of: 'YYYY:MM:DD HH:MM:SS', "
            "'YYYY-MM-DD HH:MM:SS', or 'HH:MM:SS' (date taken from suggested)."
        ) from None
    date_norm = fallback_date.replace("-", ":")[:10]
    try:
        d = _dt.datetime.strptime(date_norm, "%Y:%m:%d").date()
    except ValueError:
        raise SystemExit(
            f"Invalid fallback date {fallback_date!r} for time-only "
            "pose-source timestamp."
        ) from None
    return _dt.datetime.combine(d, t).strftime("%Y:%m:%d %H:%M:%S")


def _resolve_dates(cams, flag_value: str | None, label: str, interactive: bool):
    """Resolve a date filter for one side of camsync.

    If ``flag_value`` is set, parse and validate it against the dates present
    in ``cams``. Otherwise, in interactive TTY mode, print a per-date count
    histogram and prompt the user. Returns ``None`` when no filter should be
    applied (single date present, or user accepts "all" at the prompt).

    Args:
        cams: A ``Cameras`` instance with ``cam.datetime`` already populated.
        flag_value: Raw CLI argument value (e.g. ``"2026-04-05,2026-04-06"``)
            or ``None``.
        label: Human label used in prompts/messages (e.g. ``"Pose-source"``).
        interactive: Whether stdin is a TTY (controls prompting).

    Returns:
        list[str] | None: Normalized list of ``YYYY:MM:DD`` to keep, or
        ``None`` to indicate no filter.
    """
    counts = cams.datetime_date_counts()

    if flag_value is not None:
        raw_parts = [p.strip() for p in flag_value.split(",") if p.strip()]
        if not raw_parts:
            raise SystemExit(f"--{label.lower()}-date value is empty.")
        wanted = [p.replace("-", ":")[:10] for p in raw_parts]
        missing = [d for d in wanted if d not in counts]
        if missing:
            avail = ", ".join(f"{d} ({n})" for d, n in counts.items()) or "<none>"
            raise SystemExit(
                f"{label} date(s) not found in EXIF: {missing}. "
                f"Available: {avail}."
            )
        return wanted

    if not interactive or len(counts) <= 1:
        return None

    print(f"\n{label} dates (cam counts):")
    for d, n in counts.items():
        print(f"  {d}  {n:>5d} cams")
    raw = input(
        f"{label} date(s) [comma-separated; Enter for all]: "
    ).strip()
    if raw == "":
        return None
    raw_parts = [p.strip() for p in raw.split(",") if p.strip()]
    wanted = [p.replace("-", ":")[:10] for p in raw_parts]
    missing = [d for d in wanted if d not in counts]
    if missing:
        avail = ", ".join(f"{d} ({n})" for d, n in counts.items())
        raise SystemExit(
            f"{label} date(s) not found in EXIF: {missing}. Available: {avail}."
        )
    return wanted


def _camsync_summary_figure(summary: dict):
    """Return a matplotlib Figure with a text summary of the camsync run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines: list[str] = []
    s = summary

    def add(label: str, value: object) -> None:
        lines.append(f"{label}: {value}")

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.suptitle("Camsync - run summary", fontsize=12, y=0.98)

    def _fmt_cli(val: object) -> str:
        return "(not set)" if val is None else repr(val)

    add("Project path", s.get("project_path", ""))
    add("Working directory", s.get("cwd", ""))
    add("Project id", s.get("project_id", ""))
    lines.append("")
    add("Pose source sensor_id", s.get("pose_source", ""))
    add("Updated target sensor_id", s.get("updated_target", ""))
    lines.append("")
    ox = s.get("offset_xyz")
    if ox is not None:
        oa = np.asarray(ox, dtype=float).ravel()
        add(
            "offset_xyz (pose-local m; applied to sync)",
            f"[{oa[0]:.6g}, {oa[1]:.6g}, {oa[2]:.6g}]",
        )
    else:
        add("offset_xyz (pose-local)", "none (poses copied without xyz offset)")
    lines.append("")
    add("time_offset_sec (applied to target EXIF)", s.get("time_offset_sec", ""))
    add("scale_factor (project / initializer)", s.get("scale_factor", ""))
    lines.append("")
    add("Flag --auto-offsets", s.get("auto_offsets", ""))
    add("Flag --auto-time", s.get("auto_time", ""))
    add("Flag --auto-xyz", s.get("auto_xyz", ""))
    add("Flag --yes", s.get("assume_yes", ""))
    add("Flag --local", s.get("local", ""))
    lines.append("")
    add("spatial_max_dist (m, for --auto-time)", s.get("spatial_max_dist", ""))
    add("min_spatial_pairs (for --auto-time)", s.get("min_spatial_pairs", ""))
    lines.append("")
    add(
        "CLI --time-offset (ignored if --auto-time)",
        _fmt_cli(s.get("cli_time_offset")),
    )
    add("CLI --xyz (ignored if --auto-xyz)", _fmt_cli(s.get("cli_xyz")))
    lines.append("")
    add("PDF intercept_search_radius", s.get("intercept_search_radius", ""))
    add("Point cloud loaded (intercept highlights)", s.get("pcd_loaded", ""))

    text = "\n".join(lines)
    ax.text(
        0.06,
        0.94,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )
    return fig


def _write_camsync_sanity_pdf(
    pose_cams,
    updated_cams,
    output_path: str,
    *,
    time_offset_sec: float,
    pcd=None,
    camsync_summary: dict | None = None,
    max_pairs: int = 5,
    intercept_search_radius: float = 0.01,
) -> None:
    """Write a PDF: first a text summary page, then pose/target image pairs.

    Titles use raw EXIF from each file (``get_datetime_original(None)``). Brackets
    label ``dt`` (seconds) applied for sync on the target side only (pose uses 0 s).
    If
    ``pcd`` is set, draws the same PCD intercept (ray from target camera along its
    viewing axis) highlighted on both images, matching ``sandbox_macros`` notebook
    logic.

    Args:
        pose_cams: Pose-source :class:`~substrata.cameras.Cameras` subset.
        updated_cams: Updated-target :class:`~substrata.cameras.Cameras` subset.
        output_path: Destination ``.pdf`` path.
        time_offset_sec: Seconds applied to updated-target EXIF for time alignment.
        pcd: Optional loaded :class:`~substrata.pointclouds.PointCloud` for intercepts.
        camsync_summary: Optional dict for the first-page run summary (see
            :func:`_camsync_summary_figure`).
        max_pairs: Number of match rows to render (default 5).
        intercept_search_radius: Step radius for :meth:`PointCloud.get_intercept`
            (default 0.01, same as sandbox notebook).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image, ImageDraw

    if pcd is None:
        print(
            "Camsync sanity PDF: no point cloud loaded; "
            "intercept highlights skipped."
        )

    targets = sorted(
        updated_cams.data.values(),
        key=lambda c: (c.datetime is None, str(c.datetime), str(c.cam_id)),
    )
    pairs: list[tuple] = []
    for tcam in targets:
        pcam = pose_cams.get_camera_by_datetime(tcam.datetime)
        if pcam is not None:
            pairs.append((pcam, tcam))
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        print("Camsync sanity PDF skipped: no time-matched pairs with pose cameras.")
        return

    def _exif_title_line(cam, is_target: bool) -> str:
        raw = cam.get_datetime_original(None)
        if raw is None:
            raw = "(no EXIF DateTimeOriginal)"
        k = float(time_offset_sec)
        if is_target:
            bracket = f"(sync dt={k:+.4g} s to target EXIF)"
        else:
            bracket = "(sync dt=0 s)"
        return f"{raw} {bracket}"

    def _pixel_highlight_from_cam(cam, world_pt: np.ndarray):
        px = cam.get_pixel_coords(
            world_pt, required_to_be_in_view=False, use_orig_coords=False
        )
        if px is None or px[0] is None:
            return None
        return (int(px[0]), int(px[1]))

    def _rgb_with_highlight(fp: str, highlight_xy: tuple[int, int] | None):
        if not fp or not os.path.isfile(fp):
            return None
        try:
            image = Image.open(fp).convert("RGB")
        except OSError:
            return None
        if highlight_xy is not None:
            x, y = highlight_xy
            w, h = image.size
            r = max(12, min(50, min(w, h) // 8))
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(255, 0, 0),
            )
        return np.asarray(image)

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10, max(2.5, 2.6 * n)))
    axes = np.atleast_2d(axes)

    for i, (pose_cam, tgt_cam) in enumerate(pairs):
        pose_hl = None
        tgt_hl = None
        if pcd is not None:
            origin = np.asarray(tgt_cam.coords, dtype=float).ravel()[:3]
            icpt = pcd.get_intercept(
                origin,
                intercept_search_radius,
                vector=tgt_cam.vector,
            )
            if icpt is not None:
                world_pt = np.asarray(icpt.coords, dtype=float).ravel()[:3]
                pose_hl = _pixel_highlight_from_cam(pose_cam, world_pt)
                tgt_hl = _pixel_highlight_from_cam(tgt_cam, world_pt)

        for j, (cam, col_title, is_tgt, hl) in enumerate(
            (
                (pose_cam, "source (pose)", False, pose_hl),
                (tgt_cam, "target", True, tgt_hl),
            )
        ):
            ax = axes[i, j]
            fp = cam.filepath
            rgb = _rgb_with_highlight(fp, hl) if fp else None
            if rgb is not None:
                ax.imshow(rgb)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "(missing or unreadable image)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
            ax.set_axis_off()
            short = os.path.basename(fp) if fp else "(no path)"
            dt_line = _exif_title_line(cam, is_tgt)
            ax.set_title(f"{col_title}\n{dt_line}\n{short}", fontsize=7)

    fig.suptitle(
        "Camsync sanity check: first pose vs target pairs (EXIF time match)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    pdf = PdfPages(output_path)
    if camsync_summary is not None:
        fig_sum = _camsync_summary_figure(camsync_summary)
        pdf.savefig(fig_sum)
        plt.close(fig_sum)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)
    print(f"Wrote camsync sanity PDF: {output_path}")


def _print_spatial_time_report(report: dict, scale_factor: float) -> None:
    """Print spatial nearest-neighbor time offset diagnostics."""
    print("\n--- Auto time offset (spatial nearest pose per target) ---")
    print(
        f"scale_factor={scale_factor} (dist_metric_m = dist_stored * scale_factor); "
        "threshold applies to dist_metric_m."
    )
    for i, row in enumerate(report.get("pairs") or [], 1):
        if row.get("skip_reason"):
            print(
                f"  [{i}] target={row['target_id']}  SKIPPED: {row['skip_reason']}"
            )
            print(
                f"      dt_target={row['dt_target']!r}  dt_pose={row['dt_pose']!r}  "
                f"k_sec={row.get('k_sec')}"
            )
            continue
        print(
            f"  [{i}] target={row['target_id']}  pose={row['pose_id']}  "
            f"dist_stored={row['dist_stored']:.6g}  "
            f"dist_metric_m={row['dist_metric_m']:.6g}  inlier={row['inlier']}"
        )
        print(
            f"      dt_target={row['dt_target']!r}  dt_pose={row['dt_pose']!r}  "
            f"k_sec={row.get('k_sec')}"
        )
    st = report.get("stats") or {}
    print(
        f"Inlier pairs with valid EXIF: {report.get('n_inliers', 0)} / "
        f"targets={report.get('n_targets', 0)}"
    )
    if st:
        print(
            f"k_sec stats (inliers): median={st.get('median')}  mean={st.get('mean')}  "
            f"std={st.get('std')}  min={st.get('min')}  max={st.get('max')}"
        )
    print(f"median_k_sec (chosen if ok): {report.get('median_k_sec')}")
    print(f"ok={report.get('ok')}  reason={report.get('reason')!r}")


def _print_xyz_offset_report(report: dict) -> None:
    """Print datetime-matched xyz offset diagnostics."""
    print("\n--- Auto xyz offset (time-matched pairs, pose-local frame) ---")
    for i, row in enumerate(report.get("rows") or [], 1):
        if row.get("error"):
            print(f"  [{i}] target={row['target_id']}  ERROR: {row['error']}")
            continue
        if row.get("skip_reason"):
            print(
                f"  [{i}] target={row['target_id']}  pose={row['pose_id']}  "
                f"SKIPPED: {row['skip_reason']}"
            )
            continue
        print(
            f"  [{i}] target={row['target_id']}  pose={row['pose_id']}  "
            f"delta_world={row['delta_world']}  "
            f"offset_xyz_local={row['offset_xyz_local']}"
        )
    print(f"median_xyz (chosen): {report.get('median_xyz')}")
    print(f"mean_xyz: {report.get('mean_xyz')}")
    print(f"ok={report.get('ok')}  reason={report.get('reason')!r}")


# -------------------------------------- handlers -------------------------------------


def handle_decimate(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    input_path = args.input or init.ply_full_path
    if not input_path:
        raise SystemExit(
            "No input PLY found. Provide --input or ensure initializer finds a PLY in CWD."
        )

    # Default output: initializer's decimated path or <id>_dec50M.ply beside the source
    default_output = init.ply_dec_path or os.path.join(
        init.path or cwd, f"{init.id}_dec50M.ply"
    )
    output_path = args.output or default_output

    color_correction = None
    if getattr(args, "color_calibrate", False):
        if init.color_correction is None:
            raise SystemExit(
                "No color_correction in project YAML. Run `substrata colors -s` first, "
                "or add color_correction to the YAML."
            )
        color_correction = init.color_correction

    decimate_ply_file(
        input_path=input_path,
        output_path=output_path,
        target_points=args.points,
        show_progress=True,
        color_correction=color_correction,
    )


def handle_repair(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    input_path = args.input or init.ply_full_path
    if not input_path:
        raise SystemExit(
            "No input PLY found. Provide --input or ensure initializer finds a PLY in CWD."
        )

    if args.output:
        # Explicit output: leave input untouched.
        out_path = repair_ply_for_open3d(
            input_path=input_path,
            output_path=args.output,
            show_progress=True,
        )
        print(f"Repaired PLY written to: {out_path}")
        return

    # Default: rename the input to <basename>_old.ply and write the repaired
    # PLY back to the original path so downstream tools see no path change.
    base_no_ext, ext = os.path.splitext(input_path)
    backup_path = f"{base_no_ext}_old{ext}"
    if os.path.exists(backup_path):
        raise SystemExit(
            f"Backup target {backup_path} already exists; refusing to overwrite. "
            "Move/delete it or pass --output to write to a separate path."
        )
    os.rename(input_path, backup_path)
    print(f"Backed up original to: {backup_path}")
    try:
        out_path = repair_ply_for_open3d(
            input_path=backup_path,
            output_path=input_path,
            show_progress=True,
        )
    except Exception:
        # Restore the original on failure so the caller isn't left without a PLY.
        if not os.path.exists(input_path):
            os.rename(backup_path, input_path)
        raise
    print(f"Repaired PLY written to: {out_path}")


def handle_head(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    input_path = args.input or init.ply_full_path
    if not input_path:
        raise SystemExit(
            "No input PLY found. Provide --input/--ply or ensure initializer finds a PLY in CWD."
        )

    ply_head(input_path, n=args.num, print_output=True)


def handle_scalebars(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # 1) resolve inputs
    pcd_path = args.input or init.ply_full_path
    if not pcd_path:
        raise SystemExit(
            "No input PLY found. Provide --input/--ply or ensure initializer finds a PLY in CWD."
        )

    markers_path = args.markers or init.markers_filepath
    if not markers_path:
        raise SystemExit(
            "No markers file found. Provide --markers or ensure initializer finds a markers CSV in CWD."
        )

    # 2) load PCD (optionally streaming-decimate on load)
    pcd = PointCloud(pcd_path, max_points=args.points)

    # 3) load markers as annotations
    anns = Annotations()
    anns.get_annotations_from_file(markers_path, header=True)

    # 4) create Scalebars, attach target coords from annotations
    sb = Scalebars(scalebar_data=settings.RGL_SCALEBARS, target_data=anns)

    # 5) save PDF
    output_pdf = args.output_pdf or _get_output_filepath(init, "scalebars.pdf")
    sb.save_pdf(pcd, filepath=output_pdf)

    # 6) optionally persist scale factor to YAML
    if getattr(args, "save_yaml", False):
        scale_factor = sb.calc_scalefactor()
        if scale_factor is not None:
            init.scale_factor = float(scale_factor)
        else:
            print("Warning: failed to compute scale_factor")
        yaml_path = init.yaml_path or os.path.join(
            init.path or os.getcwd(), f"{init.id}.yaml"
        )
        init.save_config_to_yaml(yaml_path)
        print(f"Saved to YAML: {yaml_path}")


def handle_colors(args):
    """Colour calibration QC PDF and optional affine correction in YAML."""
    from substrata.color_calibration import ColorCalibrations
    from substrata.initializer import ProjectInitializer

    _base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    if args.input:
        init.pcd_filepath = args.input

    markers_path = args.markers or init.markers_filepath
    if not markers_path:
        raise SystemExit(
            "No markers file found. Provide --markers or set markers in the project YAML."
        )

    # If -n/--points is provided, always honor it (streaming load with that
    # cap), regardless of whether the PLY came from --input or the YAML.
    # Without -n, defer to the full project initializer; PointCloud's
    # memory-aware cap will still protect against OOM on huge files.
    points_cap = getattr(args, "points", None)

    if points_cap is not None:
        pcd_path = init.ply_filepath
        if not pcd_path:
            raise SystemExit(
                "No input PLY found. Provide --input/--ply or ensure the project YAML "
                "specifies a ply file."
            )
        pcd = PointCloud(pcd_path, max_points=int(points_cap))
        markers = Annotations(markers_path, orig_coords_only=True)
    else:
        init.initialize(apply_color_correction=False)
        if init.pcd is None:
            raise SystemExit(
                "No point cloud loaded. Provide --input/--ply or ensure the project YAML "
                "specifies a ply file."
            )
        if args.markers:
            markers = Annotations(args.markers, orig_coords_only=True)
        else:
            markers = init.markers
        if markers is None:
            raise SystemExit(
                "No markers loaded. Provide --markers or ensure markers exist in the project."
            )
        pcd = init.pcd

    ex_idx = getattr(args, "exclude_indices", None) or None
    ex_names = getattr(args, "exclude_names", None) or None
    cc = ColorCalibrations(
        calibration_data=settings.RGL_COLOR_CALIBRATIONS,
        target_data=markers,
        pcd=pcd,
        exclude_card_indices=ex_idx,
        exclude_names=ex_names,
    )
    output_pdf = args.output_pdf or _get_output_filepath(init, "colorcal.pdf")
    cc.save_pdf(filepath=output_pdf)
    print(f"Saved colour calibration PDF: {output_pdf}")

    if getattr(args, "save_yaml", False):
        if cc.color_correction is None:
            print(
                "Warning: no colour correction computed (no valid ColorChecker patch "
                "samples). Not updating color_correction in YAML.",
                file=sys.stderr,
            )
        else:
            init.color_correction = cc.color_correction
            yaml_path = init.yaml_path or os.path.join(init.path or cwd, f"{init.id}.yaml")
            init.save_config_to_yaml(yaml_path)
            print(f"Saved colour correction to YAML: {yaml_path}")


def handle_views(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize()

    # Save composite views PDF from initialized point cloud
    output_pdf = args.output_pdf or _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)


def handle_orient(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize(apply_transform=False)

    # Run scale_and_orient workflow (skip if --manual flag is set)
    if not getattr(args, "manual", False):
        # Use markers CSV filepath if provided, otherwise use camera depths
        depth_markers_filepath = getattr(args, "markers", None)
        if depth_markers_filepath is not None:
            depth_markers = Annotations(depth_markers_filepath, orig_coords_only=True)
        else:
            depth_markers = None
        init.scale_and_orient(depth_markers=depth_markers)

    # Handle optional manual transform
    if getattr(args, "transform", False) or getattr(args, "manual", False):
        manual_transform = _get_transform_from_user()
        # Apply manual transform to pointcloud
        # (multiplies with existing world_transform)
        init.pcd.apply_transform(manual_transform)
        # Update initializer's world_transform to match pointcloud
        init.world_transform = init.pcd.world_transform
        # Propagate updated world_transform to cameras/markers/annotations
        init.apply_world_transform(skip_pcd=True)

    # Always save values to YAML
    yaml_path = init.yaml_path or os.path.join(init.path or cwd, f"{init.id}.yaml")
    init.save_config_to_yaml(yaml_path)
    print(f"Saved orientation to YAML: {yaml_path}")

    # Also output composite views as done for the "views" command
    output_pdf = _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)

    # Save depth residuals PDF (from markers if used, otherwise from cameras)
    if not getattr(args, "manual", False):
        output_pdf = _get_output_filepath(init, "depth_residuals.pdf")
        if depth_markers is not None:
            # Use annotation depth residuals if depth_markers were used
            depth_markers.save_depth_residuals_pdf(filepath=output_pdf)
        else:
            # Use camera depth residuals (default behavior)
            init.cams.save_depth_residuals_pdf(filepath=output_pdf)


def handle_firefish(args):
    """Handles the 'firefish' command.

    Args:
        args: Parsed command-line arguments.
    """
    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Arguments
    target_depth = _infer_target_depth(base, args.target_depth)
    pdf_output = os.path.join(cwd, f"{base}_firefish.pdf")
    cam_depths_file = args.cam_depths_file or os.path.join(cwd, f"{base}_camdepths.csv")
    depth_and_outlier_threshold = (
        args.depth_outlier_thresh if args.depth_outlier_thresh is not None else None
    )
    if args.input:
        init.pcd_filepath = args.input

    from substrata.firefish import FireFish

    ff = FireFish(args.firefish_file or os.path.join(cwd, f"{base}_firefish.txt"))

    # Initialize project (loads PCD, cameras/markers if available) without transforms
    init.initialize(apply_transform=False)

    # Optionally filter cameras by group name
    if args.cams_group:
        filtered_cams = init.cams.group(args.cams_group)
        print(
            f"Number of cameras considered (after filter by group): {len(filtered_cams.items())}"
        )
    else:
        filtered_cams = init.cams
        print(f"All cameras are being considered: {len(filtered_cams.items())}")

    if len(filtered_cams.items()) == 0:
        raise ValueError(
            f"No cameras available in this group. Available camera groups: {init.cams.group_names}"
        )

    # Calculate scale factor (but do not apply yet - to get accurate camera distances only)
    init.calc_scale_factor()

    if init.scale_factor is None:
        raise ValueError("Scale factor is not set")
    else:
        print(f"Scale factor: {init.scale_factor}")

    # Run up-vector determination (on unscaled/unoriented pointcloud)
    init.up_vector, init.depth_offset, init.depth_per_unit = ff.determine_up_vector(
        filtered_cams,
        target_depth,
        init.pcd,
        distance_scale_factor=init.scale_factor,
        offset=args.offset,
        camdepths_filepath=cam_depths_file,
        pdf_output_filepath=pdf_output,
        depth_and_outlier_threshold=depth_and_outlier_threshold,
    )

    # Optionally persist orientation results to YAML
    if getattr(args, "save_yaml", False):
        # Apply scale and orientation transforms to pointcloud (using its properties
        # for additional centering and orientation)
        init.scale_and_orient(recalculate=False, plot=False)

        # Save values to YAML
        yaml_path = init.yaml_path or os.path.join(
            init.path or os.getcwd(), f"{init.id}.yaml"
        )
        init.save_config_to_yaml(yaml_path)
        print(f"Saved orientation to YAML: {yaml_path}")

        # Save composite views PDF from initialized point cloud
        output_pdf = _get_output_filepath(init, "views.pdf")
        init.pcd.save_pdf(filepath=output_pdf)

        # Save camera depth residuals PDF
        output_pdf = _get_output_filepath(init, "depth_residuals.pdf")
        init.cams.save_depth_residuals_pdf(filepath=output_pdf)


def handle_cams2video(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer
    from substrata import visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input
    # Allow explicit override of annotations path
    if args.annotations_file:
        init.annotations_filepath = args.annotations_file

    # Initialize (loads PCD and cameras if available)
    init.initialize()

    # Resolve annotations (optional)
    anns = None
    anns_path = init.annotations_filepath
    if anns_path:
        anns = Annotations(anns_path, header=True, orig_coords_only=True)
        anns.apply_transform(init.world_transform)
        print(f"Number of annotations: {len(anns)}")
    # Optional subset by camera group name
    cams_for_video = init.cams
    if getattr(args, "cams_group", None):
        try:
            cams_for_video = init.cams.subset_by_group(args.cams_group)
            print(
                f"Filtered cameras by group '{args.cams_group}': {len(cams_for_video)} cameras"
            )
        except Exception as e:
            print(
                f"Warning: Failed to filter cameras by group '{args.cams_group}': {e}"
            )
            print(f"Available camera groups: {init.cams.group_names}")
            print("Using all cameras instead")

    # Validate that we have cameras to work with
    if not cams_for_video or len(cams_for_video) == 0:
        print("Error: No cameras available for video creation")
        return

    print(f"Using {len(cams_for_video)} cameras for video creation")

    # Compute output path and ensure directory
    output_mp4 = args.output_mp4 or _get_output_filepath(init, "cams.mp4")
    # Create video directly to requested output path
    visualizations.create_annotated_video(
        cams_for_video,
        anns,
        output_filename=output_mp4,
        pcd=init.pcd,
        use_label_column=args.use_label_column,
        resize_width=args.resolution,
    )


def handle_intercepts(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer
    from substrata import measurements, visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if getattr(args, "input", None):
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras if available)
    init.initialize()

    # Require a non-identity world transform
    if init.pcd.world_transform_is_identity:
        sys.exit("World transform is not set")

    # Optionally align along slope before proceeding
    if getattr(args, "slope", False):
        slope_normal, slope_elev = init.pcd.apply_along_slope_transform()

    # Create bounding box, subdivide to grid, and visualize
    if getattr(args, "position", None):
        # Parse manual position [x, y] (top-left coordinate)
        try:
            position = ast.literal_eval(args.position)
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError("Position must be a list or tuple of length 2")
            x_top_left, y_top_left = float(position[0]), float(position[1])
        except (ValueError, SyntaxError) as e:
            sys.exit(
                f"Failed to parse --position argument: {e}. Expected format: [x,y]"
            )

        # Construct bbox from top-left position by adding box_length and box_width
        optimal_bbox = [
            [x_top_left, y_top_left],
            [x_top_left + args.box_length, y_top_left + args.box_width],
        ]
    else:
        # Use automatic optimal box position finding
        optimal_bbox = measurements.find_optimal_box_position(
            init.pcd,
            box_length=args.box_length,
            box_width=args.box_width,
            step_size=0.1,
        )
    try:
        bboxes = measurements.subdivide_boxes(optimal_bbox, args.box_size)
    except ValueError as e:
        sys.exit(f"Failed to subdivide boxes: {e}")

    fig = visualizations.show_grid_cells(init.pcd, bboxes)

    # Sample random XY points inside cells and compute intercepts
    random_points = measurements.generate_random_xy_points_within_cells(bboxes, 1, 0)
    intercepts = init.pcd.get_z_intercepts(
        random_points, args.search_radius, always_return=True, id_prefix=init.id
    )

    # Back-compute original coords and get first image matches if cameras are available
    intercepts.get_original_coords(init.pcd.world_transform)

    # Save intercepts to CSV (and YAML only for --slope)
    if getattr(args, "slope", False):
        fig.savefig(_get_output_filepath(init, "slope_bbox.png"))
        csv_path = _get_output_filepath(init, "slope_intercepts.csv")
        intercepts.save(csv_path)
        yaml_path = os.path.splitext(csv_path)[0] + ".yaml"
        payload = {
            "world_transform": init.pcd.world_transform.tolist(),
            "slope_normal": [float(x) for x in slope_normal],
            "slope_elevation_deg": float(slope_elev),
            # Exact generation grid, so intercepts-plot can replay the same cells
            # (rather than re-deriving them from the point extent).
            "grid_bbox": [
                [float(optimal_bbox[0][0]), float(optimal_bbox[0][1])],
                [float(optimal_bbox[1][0]), float(optimal_bbox[1][1])],
            ],
            "grid_cell_size": float(args.box_size),
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(payload, f)
        # additional visualization for sanity check
        a, b, c, d, inliers_idx = measurements.get_best_fit_plane_PCA(init.pcd)
        visualizations.visualize_elevation_angle(
            init.pcd,
            [a, b, c, d],
            point_size=1,
            output_filename=_get_output_filepath(init, "slope_pca.png"),
        )
    else:
        fig.savefig(_get_output_filepath(init, "topdown_bbox.png"))
        csv_path = _get_output_filepath(init, "topdown_intercepts.csv")
        intercepts.save(csv_path)


def handle_intercepts_plot(args):
    from substrata import measurements
    from substrata.ortho import OrthoGrid

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Read the generation YAML (world_transform + optional exact grid). Prefer a
    # custom YAML (e.g. the sibling <id>_slope_intercepts.yaml written by
    # `intercepts --slope`), else fall back to the project's own world_transform.
    transform = None
    grid_bbox = None
    if getattr(args, "yaml", None):
        with open(args.yaml) as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("world_transform") is not None:
            transform = np.array(cfg["world_transform"], dtype=float)
        grid_bbox = cfg.get("grid_bbox")
    if transform is None:
        transform = init.world_transform
    if np.allclose(transform, np.eye(4)):
        print(
            "Warning: orientation transform is identity; the grid may be warped "
            "for slope intercepts (pass --yaml with the generation world_transform)."
        )

    ann_path = args.annotations or init.annotations_filepath
    if not ann_path:
        raise SystemExit("intercepts-plot requires an annotations/intercepts file")

    # Load in the original frame, then apply the orientation exactly once so the
    # grid becomes axis-aligned (transform_coords compounds, so do not re-apply).
    anns = Annotations(ann_path, orig_coords_only=True)
    anns.apply_transform(transform)

    # Optional manual label colours. Any label not listed collapses into a
    # single "OTHER" category (done before building the grid so the grid,
    # animation, and positions plot all share the same labels/colours).
    pil_label_colors = None
    mpl_label_colors = None
    if getattr(args, "label_colors", None):
        allowed, pil_label_colors, mpl_label_colors = _load_label_colors(
            args.label_colors
        )
        _collapse_labels(anns, allowed)

    # Optionally load the project point cloud for the background scatter. Apply
    # the same orientation as the annotations so the points stay aligned with
    # the axis-aligned grid. If it cannot be loaded, warn and continue without.
    pcd = None
    pcd_path = init.ply_filepath
    if pcd_path:
        try:
            pcd = PointCloud(pcd_path, max_points=getattr(args, "points", None))
            pcd.apply_transform(transform)
        except Exception as exc:  # noqa: BLE001 - background scatter is optional
            print(f"Warning: could not load point cloud {pcd_path}: {exc}")
            pcd = None

    # Resolve the reporting area / lattice alignment for the label grid, in
    # order of preference:
    #   1. exact generation grid from the YAML (grid_bbox) -> reporting bbox;
    #   2. --fit-grid: recover the lattice from the intercepts themselves
    #      (point-cloud independent, best for older files lacking grid_bbox);
    #   3. --box-length/--box-width: reconstruct the generation box (a pure
    #      histogram/convolution over the same points), optionally from a manual
    #      top-left --position;
    #   4. otherwise align the grid to the intercepts.
    bbox = None
    intercepts = None
    if grid_bbox is not None:
        bbox = grid_bbox
    elif getattr(args, "fit_grid", False):
        intercepts = anns
    elif getattr(args, "box_length", None) and getattr(args, "box_width", None):
        if pcd is None:
            print(
                "Warning: --box-length/--box-width given but no point cloud "
                "could be loaded; aligning the grid to the intercepts instead."
            )
            intercepts = anns
        else:
            if getattr(args, "points", None):
                print(
                    "Warning: the point cloud was decimated (--points); the "
                    "reconstructed box may shift relative to generation. Omit "
                    "--points for an exact grid reconstruction."
                )
            if getattr(args, "position", None):
                # Manual top-left [x, y] (mirrors `intercepts --position`).
                pos = ast.literal_eval(args.position)
                x_tl, y_tl = float(pos[0]), float(pos[1])
                bbox = [
                    [x_tl, y_tl],
                    [x_tl + args.box_length, y_tl + args.box_width],
                ]
            else:
                bbox = measurements.find_optimal_box_position(
                    pcd,
                    box_length=args.box_length,
                    box_width=args.box_width,
                    step_size=getattr(args, "step_size", 0.1),
                    vis=False,
                )
    else:
        intercepts = anns

    grid = OrthoGrid(
        annotations=anns,
        pcd=pcd,
        value_by="label",
        cell_size=args.grid_size,
        bbox=bbox,
        intercepts=intercepts,
    )
    if grid.info:
        print(
            f"Fitted grid from intercepts: {grid.info.get('nx')}x"
            f"{grid.info.get('ny')} cells, {grid.info.get('empty')} empty / "
            f"{grid.info.get('multi')} multi-occupancy."
        )
    show_pcd = getattr(args, "show_points", True)
    title = getattr(args, "title", None)
    fig = grid.show(
        show_pcd=show_pcd,
        title=title,
        label_colors=mpl_label_colors,
    )
    out = args.output or (os.path.splitext(ann_path)[0] + "_grid.png")
    fig.savefig(out, dpi=100)
    print(f"Saved intercepts plot to {out}")

    # Match the other outputs to the grid figure's width (default 1800 px).
    target_w = int(round(fig.get_size_inches()[0] * fig.dpi))

    # Animated GIF of the grid filling in (same name, .gif extension).
    from substrata.animations import animate_ortho_grid

    gif_out = os.path.splitext(out)[0] + ".gif"
    animate_ortho_grid(
        grid,
        gif_out,
        show_pcd=show_pcd,
        title=title,
        label_colors=mpl_label_colors,
        loop=True,
    )
    print(f"Saved intercepts animation to {gif_out}")

    # Positions plot exactly as annotations.show(pcd, color=True) renders it,
    # named "<stem>_positions.png" and matched to the grid width. Requires the
    # point cloud (the ortho map is rendered from it).
    stem = os.path.splitext(out)[0]
    if stem.endswith("_grid"):
        stem = stem[: -len("_grid")]
    pos_out = stem + "_positions.png"
    if grid.pcd is not None:
        img = grid.annotations.show(
            grid.pcd,
            color=True,
            label_colors=pil_label_colors,
            width=target_w,
        )
        img.save(pos_out)
        print(f"Saved annotation positions to {pos_out}")
    else:
        print(
            "Warning: no point cloud available; skipping the positions plot "
            f"({pos_out})."
        )


def handle_align(args):
    from substrata.pointclouds import PointCloud

    if not args.source or not args.target:
        raise SystemExit("align requires --source and --target PLY paths")

    # Load with optional stream decimation
    src = PointCloud(args.source, max_points=args.points)
    tgt = PointCloud(args.target, max_points=args.points)

    # Compute transform that maps source → target space
    T, metrics = tgt.get_auto_align_transform(src)

    # Print and optionally show
    print("Metrics:", metrics)
    print("Alignment transform (source → target):\n", T)


def _parse_transform_from_input(transform_str: str) -> np.ndarray:
    """Parse a 4x4 transform matrix from user input (YAML or array format).

    Supports:
    - YAML format with world_transform key (3x4 or 4x4)
    - Array format (3x4 or 4x4)
    - Promotes 3x4 matrices to 4x4 by adding [0, 0, 0, 1] bottom row
    - Promotes 3x3 matrices to 4x4 by adding bottom row and right column

    Args:
        transform_str: String containing transform in YAML or array format.

    Returns:
        4x4 numpy array representing the transform matrix.
    """
    transform_str = transform_str.strip()

    # Try to parse as YAML first (check for YAML-like structure)
    if ":" in transform_str or "world_transform" in transform_str.lower():
        try:
            yaml_data = yaml.safe_load(transform_str)
            if isinstance(yaml_data, dict) and "world_transform" in yaml_data:
                transform = np.array(yaml_data["world_transform"], dtype=float)
            elif isinstance(yaml_data, list):
                transform = np.array(yaml_data, dtype=float)
            else:
                raise ValueError("YAML format not recognized")
        except Exception as e:
            raise ValueError(f"Failed to parse YAML format: {e}")
    else:
        # Try to parse as array/list format
        try:
            parsed = ast.literal_eval(transform_str)
            transform = np.array(parsed, dtype=float)
        except Exception as e:
            raise ValueError(
                f"Failed to parse array format: {e}. "
                "Expected format: [[...], [...], [...], [...]] or YAML format"
            )

    return transform


def _read_single_transform_input() -> str:
    """Read a single transform matrix input from stdin.

    Returns:
        String containing the transform matrix in YAML or array format.

    Raises:
        SystemExit: If no transform is provided.
    """
    lines = []
    while True:
        try:
            line = input()
            # If we get an empty line and we already have content, we're done
            if line.strip() == "" and lines:
                break
            # If we get an empty line and no content yet, continue waiting
            if line.strip() == "":
                continue
            lines.append(line)
        except EOFError:
            # Ctrl+D signals end of input
            break

    transform_str = "\n".join(lines)
    if not transform_str.strip():
        raise SystemExit("No transform provided")

    return transform_str


def _get_transform_from_user() -> np.ndarray:
    """Prompt user for transform matrix input, read from stdin, and parse it.

    Supports multiple transforms that are multiplied cumulatively.
    After the first transform is parsed successfully, prompts for additional
    transforms. Each new transform is multiplied with the cumulative result
    using np.dot(new_transform, cumulative_transform).

    Returns:
        4x4 numpy array representing the cumulative transform matrix.

    Raises:
        SystemExit: If no transform is provided or parsing fails.
    """
    print("Please paste the transform matrix (YAML or array format):")
    print("  - YAML format: world_transform")
    print("  - Array format: [[...], [...], [...]] or " "[[...], [...], [...], [...]]")

    # Read and parse first transform
    transform_str = _read_single_transform_input()
    try:
        cumulative_transform = _parse_transform_from_input(transform_str)
        print(f"Parsed transform:\n{cumulative_transform}")
    except Exception as e:
        raise SystemExit(f"Failed to parse transform: {e}")

    # Loop for additional transforms
    while True:
        print(
            "If you want to apply any additional transforms paste the next "
            "one below, otherwise press ENTER"
        )
        try:
            line = input().strip()
            if line == "":
                # Empty line means no more transforms
                break
        except EOFError:
            # Ctrl+D means no more transforms
            break

        # Read the additional transform
        additional_lines = [line]
        while True:
            try:
                next_line = input()
                if next_line.strip() == "":
                    break
                additional_lines.append(next_line)
            except EOFError:
                break

        additional_transform_str = "\n".join(additional_lines)
        if not additional_transform_str.strip():
            break

        try:
            additional_transform = _parse_transform_from_input(additional_transform_str)
            print(f"Parsed additional transform:\n{additional_transform}")
            # Multiply: new_transform @ cumulative_transform
            cumulative_transform = np.dot(additional_transform, cumulative_transform)
            print(f"Cumulative transform:\n{cumulative_transform}")
        except Exception as e:
            raise SystemExit(f"Failed to parse additional transform: {e}")

    return cumulative_transform


def handle_camsync(args):
    """Copy poses from a pose-source sensor to an updated-target sensor via time match."""
    from substrata.cameras import (
        spatial_nearest_time_offset_report,
        xyz_offset_datetime_matches_report,
    )
    from substrata.firefish import get_time_diff_in_secs
    from substrata.initializer import ProjectInitializer

    _base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))
    if not init.cams_meta_json_filepath or not init.cams_xml_filepath:
        raise SystemExit(
            "No cameras meta/XML paths found. Set cams_meta_json and cams_xml in "
            "the project YAML or use default project layout."
        )
    init.initialize()
    if init.cams is None or not init.cams.data:
        raise SystemExit("No cameras loaded.")

    interactive = sys.stdin.isatty()
    pose_id = getattr(args, "pose_source", None)
    tgt_id = getattr(args, "updated_target", None)

    if pose_id is None or tgt_id is None:
        if not interactive:
            raise SystemExit(
                "Specify --pose-source and --updated-target (sensor ids), or run "
                "interactively in a terminal."
            )
        sensor_counts = Counter(
            getattr(cam, "sensor_id", None) for cam in init.cams.data.values()
        )
        print("Sensors (from .cams.xml):")
        for sid in sorted(init.cams.sensors.keys()):
            sens = init.cams.sensors[sid]
            n_cams = int(sensor_counts.get(sid, 0))
            print(
                f"  sensor_id={sid}  label={sens.label!r}  "
                f"resolution={sens.width}x{sens.height}  cameras={n_cams}"
            )
        if pose_id is None:
            pose_id = int(input("Pose source sensor id: ").strip())
        if tgt_id is None:
            tgt_id = int(input("Updated target sensor id: ").strip())

    pose_cams = init.cams.subset_by_sensor(pose_id)
    updated_cams = init.cams.subset_by_sensor(tgt_id)
    if not pose_cams.data:
        raise SystemExit(f"No cameras with sensor_id={pose_id}.")
    if not updated_cams.data:
        raise SystemExit(f"No cameras with sensor_id={tgt_id}.")
    if pose_id == tgt_id:
        raise SystemExit("Pose source and updated target must be different sensors.")

    # Raw EXIF on both subsets (same chunk, comparable centers). Needed early
    # for date-based subsetting below.
    pose_cams.get_datetime_originals()
    updated_cams.get_datetime_originals()

    pose_dates = _resolve_dates(
        pose_cams, getattr(args, "pose_date", None), "Pose-source", interactive
    )
    target_dates = _resolve_dates(
        updated_cams, getattr(args, "target_date", None), "Updated-target", interactive
    )
    if pose_dates is not None:
        pose_cams = pose_cams.subset_by_dates(pose_dates)
        if not pose_cams.data:
            raise SystemExit(
                f"No pose-source cameras left after date filter: {pose_dates}."
            )
        print(
            f"Pose-source filtered to {len(pose_cams.data)} cam(s) on "
            f"{pose_dates}."
        )
    if target_dates is not None:
        updated_cams = updated_cams.subset_by_dates(target_dates)
        if not updated_cams.data:
            raise SystemExit(
                f"No updated-target cameras left after date filter: "
                f"{target_dates}."
            )
        print(
            f"Updated-target filtered to {len(updated_cams.data)} cam(s) on "
            f"{target_dates}."
        )

    auto_time = bool(
        getattr(args, "auto_offsets", False) or getattr(args, "auto_time", False)
    )
    auto_xyz = bool(
        getattr(args, "auto_offsets", False) or getattr(args, "auto_xyz", False)
    )
    assume_yes = bool(getattr(args, "yes", False))
    scale_factor = (
        float(init.scale_factor) if init.scale_factor is not None else 1.0
    )
    spatial_max_m = float(getattr(args, "spatial_max_dist", 0.5))
    min_pairs = int(getattr(args, "min_spatial_pairs", 3))

    cli_time_offset = getattr(args, "time_offset", None)
    cli_pose_time = getattr(args, "pose_time", None)
    if cli_time_offset is not None and cli_pose_time is not None:
        raise SystemExit(
            "--time-offset and --pose-time are mutually exclusive; pass only one."
        )
    if auto_time and (cli_time_offset is not None or cli_pose_time is not None):
        which = "--time-offset" if cli_time_offset is not None else "--pose-time"
        print(
            "Note: --auto-time/--auto-offsets takes precedence; "
            f"ignoring explicit {which}."
        )

    manual_xyz = getattr(args, "xyz", None)

    if auto_xyz and manual_xyz:
        print("Note: --auto-xyz/--auto-offsets takes precedence; ignoring --xyz.")

    spatial_report = None
    if auto_time:
        spatial_report = spatial_nearest_time_offset_report(
            updated_cams,
            pose_cams,
            spatial_max_dist_m=spatial_max_m,
            min_pairs=min_pairs,
            scale_factor=scale_factor,
        )
        _print_spatial_time_report(spatial_report, scale_factor)

    time_offset = None
    if auto_time:
        if spatial_report is not None and spatial_report.get("ok"):
            time_offset = float(spatial_report["median_k_sec"])
            print(f"Using auto time offset k = {time_offset} s (spatial median).")
        else:
            dt_pose, _ = pose_cams.earliest_exif_datetime()
            dt_tgt, _ = updated_cams.earliest_exif_datetime()
            if dt_pose is None or dt_tgt is None:
                raise SystemExit(
                    "Spatial time estimate failed and could not read earliest EXIF; "
                    "set --time-offset or adjust --spatial-max-dist / "
                    "--min-spatial-pairs."
                )
            time_offset = float(get_time_diff_in_secs(dt_pose, dt_tgt))
            print(
                "Spatial auto time failed or too few inliers; falling back to "
                f"earliest-EXIF delta: k = {time_offset} s "
                f"(pose earliest {dt_pose!r}, target earliest {dt_tgt!r})."
            )
    elif cli_time_offset is not None:
        time_offset = float(cli_time_offset)
    else:
        dt_pose, _ = pose_cams.earliest_exif_datetime()
        dt_tgt, _ = updated_cams.earliest_exif_datetime()
        if dt_pose is None or dt_tgt is None:
            raise SystemExit(
                "Could not read EXIF DateTimeOriginal for the earliest image in "
                "one or both subsets; set --time-offset or --pose-time explicitly."
            )
        suggested_pose_ts = dt_pose
        if cli_pose_time is not None:
            chosen_pose_ts = _parse_pose_time_arg(
                cli_pose_time, fallback_date=suggested_pose_ts[:10]
            )
            time_offset = float(get_time_diff_in_secs(chosen_pose_ts, dt_tgt))
            print(
                f"Using --pose-time {cli_pose_time!r} -> {chosen_pose_ts} "
                f"(target earliest {dt_tgt}); time_offset = {time_offset} s."
            )
        elif interactive:
            print(
                f"Earliest updated-target EXIF: {dt_tgt}\n"
                f"Earliest pose-source EXIF:    {dt_pose}\n"
                f"Suggested pose-source timestamp matching earliest target: "
                f"{suggested_pose_ts}"
            )
            raw = input(
                "Enter pose-source timestamp matching earliest target.\n"
                "  Format: 'YYYY:MM:DD HH:MM:SS' or 'HH:MM:SS' "
                "(date from suggested if time-only).\n"
                "  [Enter to use suggested]: "
            ).strip()
            if raw == "":
                chosen_pose_ts = suggested_pose_ts
            else:
                chosen_pose_ts = _parse_pose_time_arg(
                    raw, fallback_date=suggested_pose_ts[:10]
                )
            time_offset = float(get_time_diff_in_secs(chosen_pose_ts, dt_tgt))
            print(
                f"Using pose-source timestamp {chosen_pose_ts} "
                f"(target earliest {dt_tgt}); time_offset = {time_offset} s."
            )
        else:
            chosen_pose_ts = suggested_pose_ts
            time_offset = float(get_time_diff_in_secs(chosen_pose_ts, dt_tgt))
            print(
                f"Using suggested pose-source timestamp {chosen_pose_ts} "
                f"(target earliest {dt_tgt}); time_offset = {time_offset} s."
            )

    updated_cams.get_datetime_originals(offset_secs=time_offset)

    offset_xyz = None
    xyz_report = None
    if auto_xyz:
        xyz_report = xyz_offset_datetime_matches_report(
            updated_cams,
            pose_cams,
            scale_factor=scale_factor,
        )
        _print_xyz_offset_report(xyz_report)
        if not xyz_report.get("ok"):
            raise SystemExit(
                f"Auto xyz failed: {xyz_report.get('reason')}. "
                "Fix time alignment or use manual --xyz."
            )
        offset_xyz = xyz_report["median_xyz"]
    elif manual_xyz and not auto_xyz:
        offset_xyz = _parse_xyz_csv(manual_xyz)

    need_confirm = (auto_time or auto_xyz) and not assume_yes
    if need_confirm:
        if not interactive:
            raise SystemExit(
                "Auto time/xyz mode in non-interactive context requires --yes "
                "(after reviewing output in a log)."
            )
        ans = input("\nProceed with camsync (apply poses and save meta JSON)? [y/N]: ")
        if ans.strip().lower() not in ("y", "yes"):
            raise SystemExit("Aborted.")

    matched_ids, unmatched_ids = (
        updated_cams.get_centers_and_transforms_based_on_timematch(
            pose_cams, offset_xyz=offset_xyz
        )
    )

    intercept_r = 0.01
    camsync_summary = {
        "project_path": init.path,
        "cwd": cwd,
        "project_id": init.id,
        "pose_source": pose_id,
        "updated_target": tgt_id,
        "pose_dates": pose_dates,
        "target_dates": target_dates,
        "offset_xyz": offset_xyz,
        "time_offset_sec": float(time_offset),
        "scale_factor": scale_factor,
        "auto_offsets": getattr(args, "auto_offsets", False),
        "auto_time": auto_time,
        "auto_xyz": auto_xyz,
        "assume_yes": assume_yes,
        "local": getattr(args, "local", False),
        "spatial_max_dist": spatial_max_m,
        "min_spatial_pairs": min_pairs,
        "cli_time_offset": getattr(args, "time_offset", None),
        "cli_pose_time": getattr(args, "pose_time", None),
        "cli_xyz": getattr(args, "xyz", None),
        "intercept_search_radius": intercept_r,
        "pcd_loaded": getattr(init, "pcd", None) is not None,
        "n_matched": len(matched_ids),
        "n_unmatched": len(unmatched_ids),
        "unmatched_cam_ids": unmatched_ids,
    }

    _write_camsync_sanity_pdf(
        pose_cams,
        updated_cams,
        _get_output_filepath(init, "camsync.pdf"),
        time_offset_sec=float(time_offset),
        pcd=getattr(init, "pcd", None),
        camsync_summary=camsync_summary,
        intercept_search_radius=intercept_r,
    )

    matched_set = set(matched_ids)
    n_tgt = len(updated_cams.data)
    n_enabled = 0
    n_disabled = 0
    for cam in updated_cams.data.values():
        if str(cam.cam_id) in matched_set:
            cam.enabled = True
            n_enabled += 1
        else:
            cam.enabled = False
            n_disabled += 1
    print(
        f"Set enabled=True on {n_enabled}/{n_tgt} target sensor camera(s) "
        f"(disabled {n_disabled} unmatched) before save."
    )

    init.cams.save()


def handle_images(args):
    """Handle the image matching CLI command.

    Uses project initializer to load data and optionally apply a transform.
    Generates cropped image match visualizations and saves them to PDF.

    Args:
        args: Arguments from argparse.
    """
    from substrata.initializer import ProjectInitializer
    from substrata import visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input
    # Allow explicit override of annotations path
    if args.annotations:
        init.annotations_filepath = args.annotations

    # Resolve annotations
    if not init.annotations_filepath:
        raise SystemExit(
            "No annotations file found. Provide --annotations or ensure "
            "initializer finds an annotations CSV in CWD."
        )

    # Initialize (loads PCD and cameras if available)
    init.initialize()

    # Handle optional transform
    if getattr(args, "transform", False):
        transform = _get_transform_from_user()

        # Transform orig_coords and use as new orig_coords
        from substrata import geom

        try:
            for annotation in init.annotations:
                old_coords = annotation.orig_coords
                annotation.orig_coords = geom.transform_coords(
                    annotation.orig_coords, transform
                )
                if annotation.extra_coords:
                    raise ValueError(
                        "Extra coords are not supported for image matching"
                    )
                print(
                    f"{annotation.id} orig_coords changed from {old_coords} "
                    f"to {annotation.orig_coords}"
                )
        except Exception as e:
            raise SystemExit(f"Failed to apply transform: {e}")

    # # Apply world_transform from initializer if available
    # if not init.world_transform_is_identity:
    #     init.annotations.apply_transform(init.world_transform)
    # elif init.scale_factor is not None:
    #     init.annotations.apply_transform(geom.Transform.from_scale(init.scale_factor))

    # Validate that we have cameras and annotations
    if not init.cams or len(init.cams) == 0:
        raise SystemExit("No cameras available for image matching")

    if len(init.annotations) == 0:
        raise SystemExit("No annotations found")

    print(f"Number of annotations: {len(init.annotations)}")
    print(f"Number of cameras: {len(init.cams)}")

    # Get first image matches for each annotation
    image_matches = init.annotations.get_first_image_matches(init.cams, pcd=init.pcd)

    if len(image_matches) == 0:
        raise SystemExit("No image matches found for any annotations")

    print(f"Found {len(image_matches)} image matches")

    # Save to PDF
    pdf_output = args.pdf_output or os.path.join(cwd, f"{base}_imagematches.pdf")
    crop_size = args.size if args.size is not None else 1000
    visualizations.save_cropped_image_matches_to_pdf(
        image_matches, pdf_output, crop_w=crop_size, crop_h=crop_size
    )


def handle_transform(args):
    """Handle the transform CLI command.

    Loads annotations from a file, prompts for one or more transforms,
    applies them to orig_coords of each annotation, and sets coords to match.

    Args:
        args: Arguments from argparse.
    """
    from substrata import geom

    # Load annotations from file
    header = not getattr(args, "no_header", False)
    ignore_header = getattr(args, "ignore_header", False)
    anns = Annotations()
    anns.get_annotations_from_file(
        args.input, header=header, ignore_header=ignore_header
    )

    if len(anns) == 0:
        raise SystemExit("No annotations found in input file.")

    print(f"Loaded {len(anns)} annotations from {args.input}")

    # Prompt for transform(s)
    transform = _get_transform_from_user()

    # Apply inverse if requested
    if getattr(args, "inverse", False):
        transform = np.linalg.inv(transform)
        print(f"Applied inverse transform:\n{transform}")

    # Apply transform to each annotation's orig_coords and set coords to match
    for annotation in anns:
        old_orig_coords = annotation.orig_coords.copy()
        annotation.orig_coords = geom.transform_coords(
            annotation.orig_coords, transform
        )
        annotation.coords = annotation.orig_coords
        print(
            f"{annotation.id} orig_coords transformed from {old_orig_coords} "
            f"to {annotation.orig_coords}"
        )

    # Save transformed annotations
    if args.output:
        output_path = args.output
    else:
        # Default output: add _transformed suffix before extension
        base_path, ext = os.path.splitext(args.input)
        output_path = f"{base_path}_transformed{ext}"

    anns.save(output_path, orig_coords_only=True)
    print(f"Saved transformed annotations to {output_path}")


def _find_metashape_executable(explicit: str | None) -> str:
    """Resolve the Metashape executable path.

    Resolution order: an explicit ``--metashape`` value, then the
    ``METASHAPE_EXE`` environment variable, then a probe of common install
    locations.

    Args:
        explicit: Value of ``--metashape`` (or None).

    Returns:
        Path to an executable Metashape binary/launcher.

    Raises:
        SystemExit: If no usable executable can be found.
    """

    def _usable(path: str | None) -> str | None:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

    if explicit is not None:
        resolved = _usable(explicit)
        if resolved is None:
            raise SystemExit(
                f"--metashape path is not an executable file: {explicit!r}"
            )
        return resolved

    env_exe = os.environ.get("METASHAPE_EXE")
    resolved = _usable(env_exe)
    if resolved is not None:
        return resolved

    home = os.path.expanduser("~")
    candidates = [
        "/Applications/MetashapePro.app/Contents/MacOS/MetashapePro",
        os.path.join(home, "tools", "metashape-pro", "metashape.sh"),
        os.path.join(home, "metashape-pro", "metashape.sh"),
        "/opt/metashape-pro/metashape.sh",
        "/usr/local/bin/metashape.sh",
    ]
    for cand in candidates:
        resolved = _usable(cand)
        if resolved is not None:
            return resolved

    raise SystemExit(
        "Could not locate the Metashape executable. Pass --metashape "
        "/path/to/metashape.sh, set the METASHAPE_EXE environment variable, or "
        "install Metashape to a standard location."
    )


def handle_metashape_export(args):
    """Export a substrata project folder from a Metashape ``.psx`` project.

    Shells out to Metashape's bundled Python to run the packaged
    ``metashape_scripts/export_project.py`` (which imports only Metashape +
    stdlib), then writes a starter ``<id>.yaml`` using the project initializer.

    Args:
        args: Parsed CLI arguments.
    """
    from substrata.metashape_scripts import export_project as mse

    # By default the current directory is the project folder and the project is
    # <foldername>.psx inside it; exported files are written alongside it.
    output_dir = os.path.abspath(
        os.path.expanduser(args.output_dir) if args.output_dir else os.getcwd()
    )
    if getattr(args, "psx", None):
        psx = os.path.abspath(os.path.expanduser(args.psx))
        project_id = args.id or mse.default_project_id(psx)
    else:
        project_id = args.id or os.path.basename(os.path.normpath(output_dir))
        psx = os.path.join(output_dir, f"{project_id}.psx")

    if not os.path.isfile(psx):
        raise SystemExit(
            f"Metashape project not found: {psx}\n"
            "Pass --psx, or run inside a folder containing <foldername>.psx."
        )

    exe = _find_metashape_executable(getattr(args, "metashape", None))

    script_path = os.path.join(
        os.path.dirname(__file__), "metashape_scripts", "export_project.py"
    )
    if not os.path.isfile(script_path):
        raise SystemExit(f"Bundled export script missing: {script_path}")

    cmd = [
        exe,
        "-platform",
        "offscreen",
        "-r",
        script_path,
        "--psx",
        psx,
        "--output-dir",
        output_dir,
        "--id",
        project_id,
    ]
    if getattr(args, "chunk", None) is not None:
        cmd += ["--chunk", str(args.chunk)]
    if getattr(args, "overwrite", False):
        cmd += ["--overwrite"]
    metadata_only = bool(getattr(args, "metadata_only", False))
    if metadata_only:
        cmd += ["--metadata-only"]

    print("Running Metashape export:\n  " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise SystemExit(f"Failed to launch Metashape executable {exe!r}: {e}")
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"Metashape export failed (exit code {e.returncode}). "
            "See the output above for details."
        )

    paths = mse.project_layout(output_dir, project_id)

    # Decimate the exported point cloud to <id>_dec50M.ply. Decimation needs
    # open3d (unavailable in Metashape's Python), so it runs here in the conda
    # env. Skipped for --metadata-only or when no full PLY was produced.
    dec_ply_path = os.path.join(output_dir, f"{project_id}_dec50M.ply")
    overwrite = bool(getattr(args, "overwrite", False))
    if not metadata_only and os.path.isfile(paths["ply"]):
        if os.path.isfile(dec_ply_path) and not overwrite:
            print(
                f"Skipping decimation: {dec_ply_path} already exists "
                "(use --overwrite)."
            )
        else:
            print(f"Decimating point cloud -> {dec_ply_path}")
            decimate_ply_file(
                input_path=paths["ply"],
                output_path=dec_ply_path,
                target_points=50_000_000,
                show_progress=True,
            )

    # Finish the folder in the conda env: build a starter YAML pointing at the
    # files just written (scale/orientation stay at defaults until
    # `substrata orient`). Set paths explicitly from the known export layout so
    # it is correct even when --id differs from the folder name. Prefer the
    # decimated PLY (matching ProjectInitializer's default preference).
    init = ProjectInitializer(path=output_dir)
    init.id = project_id
    if os.path.isfile(dec_ply_path):
        init.ply_filepath = dec_ply_path
    elif os.path.isfile(paths["ply"]):
        init.ply_filepath = paths["ply"]
    if os.path.isfile(paths["cams_xml"]):
        init.cams_xml_filepath = paths["cams_xml"]
    if os.path.isfile(paths["meta_json"]):
        init.cams_meta_json_filepath = paths["meta_json"]
    if os.path.isfile(paths["markers"]):
        init.markers_filepath = paths["markers"]
    yaml_path = os.path.join(output_dir, f"{project_id}.yaml")
    init.save_config_to_yaml(yaml_path)
    print(f"Wrote starter project YAML: {yaml_path}")

    print(
        "\nProject files ready in: "
        f"{output_dir}\nNext step (run from this folder):\n"
        "  substrata orient        # compute scale + world_transform"
    )


def handle_train(args):
    """Handle the 'train' command: collate labels, build crops, train/evaluate.

    Pipeline:
      1. Glob annotation CSVs from --csv-path and render the CATAMI label tree;
         the bolded entries are the training labels (confirm with the user).
      2. Verify the unique ``cam_filepath`` directories, falling back to the
         --model-path convention only when a hard-coded directory is missing.
      3. Write a consolidated training annotations CSV (exact-match labels,
         remapped paths, model-prefixed integer ids).
      4. Incrementally sync train/validation/test crops (80/10/10).
      5. Train a fastai classifier (skipped with --test/--validate) and write a
         training_summary.pdf (run settings + per-class crop counts + metrics).
      6. Report stats on validation (default) or test (--test) crops.

    ``--validate`` and ``--test`` both skip steps 1-5 and only re-run the
    evaluation/reporting, on the validation and test crops respectively.

    Args:
        args: Parsed command-line arguments.
    """
    import glob

    from substrata import classification

    cwd = os.getcwd()
    csv_path = args.csv_path or cwd
    model_path = args.model_path or cwd
    output_dir = args.output or cwd
    classes_path = args.classes or os.path.join(cwd, settings.TRAIN_CLASSES_FILE)
    pattern = args.pattern or settings.TRAIN_DEFAULT_PATTERN
    crop_size = args.crop_size or settings.TRAIN_CROP_SIZE
    model_file = args.model or os.path.join(
        output_dir, settings.TRAIN_DEFAULT_MODEL_FILE
    )
    map_path = getattr(args, "label_map", None) or os.path.join(
        output_dir, settings.TRAIN_LABEL_MAP_FILE
    )
    interactive = sys.stdin.isatty()
    assume_yes = bool(getattr(args, "yes", False))

    def _stats_pdf(split_folder):
        return os.path.join(output_dir, f"{split_folder}_stats.pdf")

    # --test / --validate: skip training and only re-run evaluation/reporting
    # on the test or validation crops respectively.
    eval_only = getattr(args, "test", False) or getattr(args, "validate", False)
    if eval_only:
        flag = "--test" if getattr(args, "test", False) else "--validate"
        if getattr(args, "test", False):
            split = settings.TRAIN_CROP_DIRS[2]  # test_crops
        else:
            split = settings.TRAIN_CROP_DIRS[1]  # validation_crops
        if not os.path.isfile(model_file):
            raise SystemExit(f"Model file not found for {flag}: {model_file}")
        if not os.path.isfile(map_path):
            raise SystemExit(
                f"Label map not found for {flag}: {map_path}. Run "
                "`substrata train` (or --prepare-only) first to create it."
            )
        label_map = classification.load_label_map(map_path)
        classification.report_classifier_stats(
            model_file, output_dir, split, label_map, pdf_path=_stats_pdf(split)
        )
        return

    if not os.path.isfile(classes_path):
        raise SystemExit(f"Classes file not found: {classes_path}")

    csv_files = sorted(
        f
        for f in glob.glob(os.path.join(csv_path, pattern))
        if os.path.basename(f) != os.path.basename(classes_path)
    )
    if not csv_files:
        raise SystemExit(
            f"No annotation CSVs match {pattern!r} in {csv_path}."
        )

    # --- Step 1: label tree + seed collapse map ---
    include_classes = getattr(args, "include_classes", None)
    include_set = set(include_classes) if include_classes else None
    lines, training_labels, counts, _unknown, collapse_map = (
        classification.build_label_tree(
            classes_path, csv_files, args.min_count, args.tips_only,
            include_labels=include_set,
            collapse=bool(getattr(args, "collapse", False)),
        )
    )
    print("\n".join(lines))
    if include_set is not None:
        missing = include_set - training_labels
        if missing:
            raise SystemExit(
                "These --include-classes categories are not present in the "
                f"tree: {', '.join(sorted(missing))}. "
                "Check the labels above (the code in brackets) and retry."
            )
    if not training_labels:
        raise SystemExit("No bolded training labels were derived; nothing to do.")

    # Crops are generated for every visible label (independent of selection);
    # selection + collapse live in the editable label map written below.
    visible_labels = {
        code for code, n in counts.items()
        if n >= settings.TRAIN_MIN_VISIBLE_COUNT
    }
    print(
        f"\nThe {len(training_labels)} bolded entries above seed the training "
        f"classes; crops are generated for all {len(visible_labels)} visible "
        "label(s)."
    )
    if not assume_yes:
        if not interactive:
            raise SystemExit(
                "Confirmation required; re-run with --yes in non-interactive use."
            )
        ans = input("Proceed (generate crops and seed the label map)? [y/N]: ")
        if ans.strip().lower() not in ("y", "yes"):
            raise SystemExit("Aborted.")

    # --- Steps 2 & 3: verify paths and write consolidated annotations ---
    # --min_conf keeps empty label_conf rows; --min_conf_strict treats empty
    # as 0. They are mutually exclusive (enforced by argparse).
    min_conf = getattr(args, "min_conf", None)
    conf_strict = getattr(args, "min_conf_strict", None)
    if conf_strict is not None:
        min_conf, conf_is_strict = conf_strict, True
    else:
        conf_is_strict = False
    ann_path = os.path.join(output_dir, settings.TRAIN_ANNOTATIONS_FILE)
    n_written, n_dropped, n_dropped_conf = (
        classification.collate_training_annotations(
            csv_files, pattern, visible_labels, ann_path, model_path,
            prompt=(interactive and not assume_yes),
            min_conf=min_conf, conf_strict=conf_is_strict,
        )
    )
    conf_note = (
        f", {n_dropped_conf} dropped below confidence {min_conf}"
        if min_conf is not None else ""
    )
    print(
        f"\nWrote {n_written} training annotation(s) to {ann_path} "
        f"({n_dropped} dropped for missing camera fields{conf_note})."
    )
    if n_written == 0:
        raise SystemExit("No usable training annotations; aborting.")

    # --- Step 4: sync crops (80/10/10) ---
    stats = classification.sync_crops(
        ann_path, output_dir, crop_size,
        delete_stale=True, prompt=(interactive and not assume_yes),
        n_jobs=args.jobs,
    )
    print(
        f"Crops: {stats['generated']} generated, "
        f"{stats['skipped_existing']} already present, "
        f"{stats['deleted']} deleted, "
        f"{stats['removed_dirs']} empty folder(s) removed, "
        f"{stats['failed']} failed."
    )

    # --- Step 4b: seed/merge the editable label map ---
    # Crops exist for every visible label, but the map only lists labels that
    # are actually selected for training (the collapse-map entries) so it isn't
    # cluttered with blank rows for sub-min_count/excluded labels. A below-
    # min_count child that collapses into a selected parent is still included.
    # By default the map is re-seeded from the current selection flags; pass
    # --keep-map to preserve an existing (hand-edited) map and only append new
    # labels.
    map_labels = {lab for lab in visible_labels if lab in collapse_map}
    label_map = classification.merge_label_map(
        map_path, map_labels, counts, collapse_map,
        reseed=not bool(getattr(args, "keep_map", False)),
    )
    training_classes = sorted(set(label_map.values()))
    print(
        f"\nLabel map: {map_path}\n"
        f"  {len(label_map)} label(s) -> {len(training_classes)} training "
        f"class(es): {', '.join(training_classes)}"
    )

    # --prepare-only: stop here so the map can be hand-tuned before training.
    if getattr(args, "prepare_only", False):
        print(
            f"\nPrepared crops and label map. Edit {map_path} to tune the "
            "selection/collapsing, then re-run `substrata train --keep-map` to "
            "train on the edited map (plain `substrata train` re-seeds it)."
        )
        return

    # --- Step 5: train ---
    learn = classification.train_classifier(
        output_dir, model_file, label_map, arch=args.arch, epochs=args.epochs
    )

    # --- Step 5b: training-run summary PDF (settings + per-class counts) ---
    counts_by_split = classification.count_crops_by_class(output_dir, label_map)
    info = [
        ("Model architecture", args.arch),
        ("Epochs", args.epochs),
        ("Crop size (px)", crop_size),
        ("Input size (px)", settings.TRAIN_IMAGE_SIZE),
        ("Train/val/test split (%)", "/".join(map(str, settings.TRAIN_SPLIT))),
        ("Selection: min_count", args.min_count),
        ("Selection: tips_only", bool(args.tips_only)),
        (
            "Selection: include_classes",
            " ".join(include_classes) if include_classes else "(none)",
        ),
        ("Selection: collapse", bool(getattr(args, "collapse", False))),
        (
            "Selection: min_conf",
            "(none)" if min_conf is None
            else f"{min_conf}{' (strict)' if conf_is_strict else ''}",
        ),
        ("Map mode", "keep-map" if getattr(args, "keep_map", False) else "reseed"),
        ("Annotation pattern", pattern),
        ("CSV files scanned", len(csv_files)),
        ("Training classes", len(training_classes)),
        ("Label map", map_path),
        ("Model output", model_file),
    ]
    summary_pdf = os.path.join(output_dir, settings.TRAIN_SUMMARY_FILE)
    classification.write_training_summary_pdf(
        summary_pdf, info, counts_by_split,
        metrics=classification.final_metrics_from_learner(learn),
    )
    print(f"Wrote training summary: {summary_pdf}")

    # --- Step 6: stats on the validation crops ---
    valid_split = settings.TRAIN_CROP_DIRS[1]
    classification.report_classifier_stats(
        model_file, output_dir, valid_split, label_map,
        pdf_path=_stats_pdf(valid_split),
    )


def main():
    parser = argparse.ArgumentParser(description="Substrata CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # decimate (initializer-driven defaults)
    p_dec = subparsers.add_parser(
        "decimate",
        help=(
            "Decimate a PLY. With no args, uses initializer on CWD, output to <id>_dec50M.ply, target=50,000,000."
        ),
    )

    # No file output or visualization in this simplified command
    p_dec.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_dec.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Optional explicit output PLY path (defaults to <id>_dec50M.ply).",
    )
    p_dec.add_argument(
        "-n",
        "--points",
        dest="points",
        type=int,
        default=50_000_000,
        help="Number of points to keep (default: 50,000,000).",
    )
    p_dec.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )
    p_dec.add_argument(
        "-c",
        "--color_calibrate",
        dest="color_calibrate",
        action="store_true",
        help=(
            "Apply color_correction from the project YAML while writing the output "
            "PLY (requires `substrata colors -s` or equivalent). No second load."
        ),
    )

    # metashape-export (initialize a substrata project folder from a .psx)
    p_meta = subparsers.add_parser(
        "metashape-export",
        help=(
            "Export a substrata project folder (<id>.ply/.cams.xml/.meta.json/"
            "_markers.csv + starter <id>.yaml) from a Metashape .psx by shelling "
            "out to Metashape's bundled Python."
        ),
    )
    p_meta.add_argument(
        "--psx",
        dest="psx",
        type=str,
        default=None,
        help=(
            "Path to the Metashape project file (.psx). Default: "
            "<foldername>.psx in the output directory."
        ),
    )
    p_meta.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Directory to write the project files into (default: CWD). Files are "
            "written directly here, not into an <id> subfolder."
        ),
    )
    p_meta.add_argument(
        "--id",
        dest="id",
        type=str,
        default=None,
        help="Project id / folder name (default: .psx basename).",
    )
    p_meta.add_argument(
        "--chunk",
        dest="chunk",
        type=str,
        default=None,
        help="Chunk label or 0-based index to export (default: active/first).",
    )
    p_meta.add_argument(
        "--metashape",
        dest="metashape",
        type=str,
        default=None,
        help=(
            "Path to the Metashape executable/launcher. Falls back to "
            "$METASHAPE_EXE, then common install locations."
        ),
    )
    p_meta.add_argument(
        "--metadata-only",
        dest="metadata_only",
        action="store_true",
        help=(
            "Skip the point cloud: export cameras + markers only (no .ply, no "
            "decimation)."
        ),
    )
    p_meta.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help=(
            "Overwrite existing outputs instead of skipping them (applies to "
            "exported files and the decimated PLY)."
        ),
    )

    # repair (re-emit a PLY that Open3D can parse, e.g. Metashape exports)
    p_rep = subparsers.add_parser(
        "repair",
        help=(
            "Rewrite a PLY in a strict Open3D-compatible form "
            "(float32 xyz, optional uchar RGB, optional float32 normals); "
            "drops extra vertex properties and non-finite rows. By default "
            "renames the input to <input>_old.ply and writes the repaired "
            "PLY in its place."
        ),
    )
    p_rep.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_rep.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help=(
            "Optional explicit output PLY path. If omitted (default), the input "
            "is renamed to <input>_old.ply and the repaired file is written to "
            "the original input path."
        ),
    )
    p_rep.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # head (PLY preview)
    p_head = subparsers.add_parser(
        "head", help="Show first N vertex rows from a PLY file."
    )
    p_head.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_head.add_argument(
        "-n",
        dest="num",
        type=int,
        default=5,
        help="Number of vertex rows to display (default: 5).",
    )
    p_head.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # scalebars
    p_sb = subparsers.add_parser(
        "scalebars",
        help="Generate scalebar PDF from a point cloud and marker annotations.",
    )
    p_sb.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_sb.add_argument(
        "--markers",
        dest="markers",
        type=str,
        default=None,
        help="Optional explicit markers CSV path (overrides initializer).",
    )
    p_sb.add_argument(
        "--output_pdf",
        dest="output_pdf",
        type=str,
        default=None,
        help="Optional output PDF filepath.",
    )
    p_sb.add_argument(
        "-n",
        "--points",
        dest="points",
        type=int,
        default=50000000,
        help="Optional max points to stream-load PLY (decimation on load).",
    )
    p_sb.add_argument(
        "-s",
        "--save_yaml",
        dest="save_yaml",
        action="store_true",
        help="Save computed scale_factor into a YAML config for this project.",
    )
    p_sb.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # views
    p_views = subparsers.add_parser(
        "views", help="Save composite views PDF for a point cloud."
    )
    p_views.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_views.add_argument(
        "--output_pdf",
        dest="output_pdf",
        type=str,
        default=None,
        help="Output PDF filepath.",
    )
    p_views.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # orient
    p_orient = subparsers.add_parser(
        "orient",
        help="Calculate and apply scale and orientation transforms, save to YAML.",
    )
    p_orient.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_orient.add_argument(
        "--transform",
        dest="transform",
        action="store_true",
        help=(
            "Prompt for manual transform matrix to apply after scale_and_orient "
            "(accepts YAML or array format, supports multiple transforms)."
        ),
    )
    p_orient.add_argument(
        "--manual",
        dest="manual",
        action="store_true",
        help=(
            "Skip the automatic scale_and_orient workflow. "
            "Use with --transform to apply only manual transforms."
        ),
    )
    p_orient.add_argument(
        "--markers",
        dest="markers",
        type=str,
        default=None,
        help="Optional markers CSV filepath to use for up vector calculation.",
    )
    p_orient.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # colors (ColorChecker calibration QC + optional YAML colour correction)
    p_colors = subparsers.add_parser(
        "colors",
        help=(
            "Run ColorCalibrations from project markers, save QC PDF and optionally "
            "store affine colour correction in YAML."
        ),
    )
    p_colors.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help=(
            "Explicit PLY path (overrides YAML). Always loaded in full (never stream-decimated)."
        ),
    )
    p_colors.add_argument(
        "--markers",
        dest="markers",
        type=str,
        default=None,
        help="Optional markers CSV path (overrides initializer / YAML).",
    )
    p_colors.add_argument(
        "--output_pdf",
        dest="output_pdf",
        type=str,
        default=None,
        help="Output PDF path (default: <project_id>_colorcal.pdf in project folder).",
    )
    p_colors.add_argument(
        "-n",
        "--points",
        dest="points",
        type=int,
        default=None,
        help=(
            "Stream-sample at most N vertices from the default YAML PLY only (.ply). "
            "Ignored when --input/--ply is set."
        ),
    )
    p_colors.add_argument(
        "-s",
        "--save_yaml",
        dest="save_yaml",
        action="store_true",
        help="Write affine colour_correction (matrix + offset) into the project YAML.",
    )
    p_colors.add_argument(
        "--exclude-index",
        "--exclude-card",
        dest="exclude_indices",
        action="append",
        type=int,
        default=None,
        metavar="N",
        help=(
            "0-based ColorChecker card index to omit from medians, fit, and PDF card "
            "pages (repeatable). Same order as settings.RGL_COLOR_CALIBRATIONS."
        ),
    )
    p_colors.add_argument(
        "--exclude-name",
        dest="exclude_names",
        action="append",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Exclude a card by its name (optional 5th column per row in "
            "RGL_COLOR_CALIBRATIONS, e.g. top-left). Repeatable."
        ),
    )
    p_colors.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # firefish
    p_ff = subparsers.add_parser(
        "firefish",
        help=(
            "Run FireFish/Cameras alignment: initialize FireFish + Cameras and "
            "determine up vector/output PDF."
        ),
    )
    p_ff.add_argument(
        "--firefish-file",
        dest="firefish_file",
        type=str,
        default=None,
        help=(
            "Path to FireFish file. Default: <cwd_basename>_firefish.txt in current folder."
        ),
    )
    p_ff.add_argument(
        "--target-depth",
        dest="target_depth",
        type=int,
        default=None,
        help=(
            "Optional target depth in meters. Default: extracted from <cwd_basename> pattern _<int>m_."
        ),
    )
    p_ff.add_argument(
        "--cam-depths-file",
        dest="cam_depths_file",
        type=str,
        default=None,
        help=(
            "Optional CSV to store camera depths. Default: <cwd_basename>_camdepths.csv in current folder."
        ),
    )
    p_ff.add_argument(
        "--depth-outlier-threshold",
        dest="depth_outlier_thresh",
        type=int,
        default=settings.FIREFISH_DEPTH_ALTITUDE_OUTLIER_THRESHOLD,
        help=(
            "Optional depth/outlier threshold (meters). Defaults to FireFish.determine_up_vector default."
        ),
    )
    p_ff.add_argument(
        "--cams_group",
        dest="cams_group",
        type=str,
        default=None,
        help=("Optional camera group name to subset (uses Cameras.subset_by_group)."),
    )
    p_ff.add_argument(
        "--offset",
        dest="offset",
        type=int,
        default=None,
        help=(
            "Manually set time offset in seconds; skips automatic offset determination."
        ),
    )
    p_ff.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_ff.add_argument(
        "-s",
        "--save_yaml",
        dest="save_yaml",
        action="store_true",
        help="Save computed up_vector, depth_offset, depth_per_unit into YAML.",
    )
    p_ff.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # cams2video
    p_c2v = subparsers.add_parser(
        "cams2video",
        help=(
            "Create a video from cameras by drawing image matches (initializer-driven)."
        ),
    )
    p_c2v.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_c2v.add_argument(
        "--annotations",
        dest="annotations_file",
        type=str,
        default=None,
        help=("Path to annotations CSV. Uses initializer if omitted."),
    )
    p_c2v.add_argument(
        "-l",
        "--label",
        dest="use_label_column",
        action="store_true",
        help=("Use label column from annotations when drawing matches (default: off)."),
    )
    p_c2v.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        type=int,
        default=None,
        help=("Optional width to resize images when creating frames (pixels)."),
    )
    p_c2v.add_argument(
        "--cams_group",
        dest="cams_group",
        type=str,
        default=None,
        help=("Optional camera group name to subset (uses Cameras.subset_by_group)."),
    )
    p_c2v.add_argument(
        "--output_mp4",
        dest="output_mp4",
        type=str,
        default=None,
        help="Optional output MP4 filepath (default: <id>_cams.mp4).",
    )
    p_c2v.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # intercepts
    p_intercepts = subparsers.add_parser(
        "intercepts",
        help=(
            "Find optimal box, subdivide to grid, sample random points, compute Z-intercepts."
        ),
    )
    p_intercepts.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )

    # align
    p_align = subparsers.add_parser(
        "align",
        help=(
            "Register a source PLY to a target PLY and print transform (optionally save aligned source)."
        ),
    )
    p_align.add_argument(
        "--source", dest="source", type=str, required=True, help="Source PLY path"
    )
    p_align.add_argument(
        "--target", dest="target", type=str, required=True, help="Target PLY path"
    )
    p_align.add_argument(
        "--points",
        dest="points",
        type=int,
        default=5_000_000,
        help="Max points to stream-load",
    )
    # No output or visualization flags
    p_intercepts.add_argument(
        "--box-length",
        dest="box_length",
        type=float,
        default=25.0,
        help="Rectangle length in meters for optimal box search (default: 25).",
    )
    p_intercepts.add_argument(
        "--box-width",
        dest="box_width",
        type=float,
        default=4.0,
        help="Rectangle width in meters for optimal box search (default: 4).",
    )
    p_intercepts.add_argument(
        "--position",
        dest="position",
        type=str,
        default=None,
        help=(
            "Manual top-left coordinate [x,y] for the bounding box. "
            "When provided, skips find_optimal_box_position and uses this position "
            "plus box_length and box_width to define the bounding box. "
            "Format: [x,y] (e.g., '[1.4, 2.6]')."
        ),
    )
    p_intercepts.add_argument(
        "--box-size",
        dest="box_size",
        type=float,
        default=0.2,
        help="Grid cell size in meters for subdividing the optimal box (default: 0.2).",
    )
    p_intercepts.add_argument(
        "--search-radius",
        dest="search_radius",
        type=float,
        default=settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        help="Search radius in meters for Z-intercept lookup (default: 0.005).",
    )
    p_intercepts.add_argument(
        "--slope",
        dest="slope",
        action="store_true",
        help="Apply along-slope transform to pointcloud before processing.",
    )
    p_intercepts.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # intercepts-plot
    p_intercepts_plot = subparsers.add_parser(
        "intercepts-plot",
        help=(
            "Plot a saved intercepts CSV as a grid colored by majority label "
            "(no bounding boxes needed; grid built from a cell size)."
        ),
    )
    p_intercepts_plot.add_argument(
        "--annotations",
        "--intercepts",
        dest="annotations",
        type=str,
        default=None,
        help="Optional intercepts/annotations CSV path (overrides initializer).",
    )
    p_intercepts_plot.add_argument(
        "--yaml",
        "--config",
        dest="yaml",
        type=str,
        default=None,
        help=(
            "Optional project YAML holding the world_transform (orientation) used "
            "to generate the intercepts, e.g. <id>_slope_intercepts.yaml."
        ),
    )
    p_intercepts_plot.add_argument(
        "--grid-size",
        dest="grid_size",
        type=float,
        default=0.2,
        help="Grid cell size in meters (default: 0.2).",
    )
    p_intercepts_plot.add_argument(
        "--fit-grid",
        dest="fit_grid",
        action="store_true",
        help=(
            "Recover the generation grid directly from the intercepts (no point "
            "cloud needed) by fitting the sub-cell origin for the best "
            "one-point-per-cell alignment. Most robust option for older files "
            "without a saved grid_bbox."
        ),
    )
    p_intercepts_plot.add_argument(
        "--box-length",
        dest="box_length",
        type=float,
        default=None,
        help=(
            "Reconstruct the exact generation grid (for older intercepts files "
            "without a saved grid_bbox) by re-running the optimal-box search on "
            "the point cloud with this rectangle length in meters. Requires "
            "--box-width. Must match the value used by `intercepts`."
        ),
    )
    p_intercepts_plot.add_argument(
        "--box-width",
        dest="box_width",
        type=float,
        default=None,
        help=(
            "Rectangle width in meters for grid reconstruction (see "
            "--box-length). Must match the value used by `intercepts`."
        ),
    )
    p_intercepts_plot.add_argument(
        "--position",
        dest="position",
        type=str,
        default=None,
        help=(
            "Manual top-left [x,y] for grid reconstruction (skips the optimal-box "
            "search); mirrors `intercepts --position`. Use only with "
            "--box-length/--box-width."
        ),
    )
    p_intercepts_plot.add_argument(
        "--step-size",
        dest="step_size",
        type=float,
        default=0.1,
        help=(
            "Grid resolution in meters for the optimal-box search during "
            "reconstruction (default: 0.1; match `intercepts`)."
        ),
    )
    p_intercepts_plot.add_argument(
        "--output",
        "-o",
        dest="output",
        type=str,
        default=None,
        help="Optional output PNG path (default: <intercepts_stem>_grid.png).",
    )
    p_intercepts_plot.add_argument(
        "--label-colors",
        "--label-colours",
        dest="label_colors",
        type=str,
        default=None,
        help=(
            "Optional file of manual label colours, one 'label #hexcolor' per "
            "line (e.g. 'Coral  #e6194b'). Labels not listed collapse into a "
            "single 'OTHER' category; add an 'OTHER #hex' row to set its colour "
            "(default #999999). Applies to the grid, the animation, and the "
            "positions plot."
        ),
    )
    p_intercepts_plot.add_argument(
        "--title",
        dest="title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    p_intercepts_plot.add_argument(
        "--hide-points",
        dest="show_points",
        action="store_false",
        default=True,
        help=(
            "Do not scatter the project point cloud as a background behind the "
            "grid cells (the scatter is shown by default when a point cloud is "
            "found)."
        ),
    )
    p_intercepts_plot.add_argument(
        "--points",
        "-n",
        dest="points",
        type=int,
        default=None,
        help="Optional cap on the number of point-cloud points loaded for scatter.",
    )
    p_intercepts_plot.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # images
    p_images = subparsers.add_parser(
        "images",
        help=("Find image matches of annotations and output cropped images to PDF."),
    )
    p_images.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_images.add_argument(
        "--annotations",
        dest="annotations",
        type=str,
        default=None,
        help="Optional explicit annotations CSV path (overrides initializer).",
    )
    p_images.add_argument(
        "--transform",
        dest="transform",
        action="store_true",
        help=(
            "Prompt for 4x4 transform matrix to apply to orig_coords "
            "(accepts YAML or array format)."
        ),
    )
    p_images.add_argument(
        "--pdf-output",
        dest="pdf_output",
        type=str,
        default=None,
        help=(
            "Optional output PDF filepath. "
            "Default: <cwd_basename>_imagematches.pdf in current folder."
        ),
    )
    p_images.add_argument(
        "--size",
        dest="size",
        type=int,
        default=None,
        help=(
            "Crop size in pixels for image matches (sets both crop_w and crop_h). "
            "Default: 1000 (uses function default)."
        ),
    )
    p_images.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # camsync
    p_camsync = subparsers.add_parser(
        "camsync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=(
            "Copy camera centers/transforms from a pose-source sensor to an "
            "updated-target sensor using EXIF time matching; writes meta JSON."
        ),
        epilog=(
            "Per-camera poses are written to the cameras meta JSON only; .cams.xml is "
            "not modified (this tool reads sensor calibration from XML).\n"
            "Use --auto-time / --auto-xyz / --auto-offsets for spatial and median-offset "
            "estimation (see --help on those flags). Auto modes print detailed reports "
            "and prompt before saving unless you pass --yes."
        ),
    )
    p_camsync.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )
    p_camsync.add_argument(
        "--pose-source",
        "-s",
        dest="pose_source",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Sensor id whose centers/transforms are used as the source of truth "
            "(e.g. GoPro). If omitted, lists sensors and prompts (TTY only)."
        ),
    )
    p_camsync.add_argument(
        "--updated-target",
        "-u",
        dest="updated_target",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Sensor id whose cameras are updated (e.g. macro). If omitted, "
            "prompts after --pose-source (TTY only)."
        ),
    )
    p_camsync.add_argument(
        "--pose-date",
        dest="pose_date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD[,...]",
        help=(
            "Filter pose-source cameras to one or more EXIF dates (comma-"
            "separated). If omitted in TTY, prompts when multiple dates exist."
        ),
    )
    p_camsync.add_argument(
        "--target-date",
        dest="target_date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD[,...]",
        help=(
            "Filter updated-target cameras to one or more EXIF dates (comma-"
            "separated). If omitted in TTY, prompts when multiple dates exist."
        ),
    )
    p_camsync.add_argument(
        "--time-offset",
        "-t",
        dest="time_offset",
        type=float,
        default=None,
        help=(
            "Seconds added to updated-target EXIF datetimes before matching. "
            "Mutually exclusive with --pose-time. If both are omitted, "
            "prompts in TTY for a pose-source timestamp (or uses suggested "
            "in non-TTY)."
        ),
    )
    p_camsync.add_argument(
        "--pose-time",
        dest="pose_time",
        type=str,
        default=None,
        metavar="TIMESTAMP",
        help=(
            "Pose-source timestamp matching the earliest updated-target "
            "image. Accepts 'YYYY:MM:DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS', "
            "or 'HH:MM:SS' (date taken from earliest pose-source EXIF). "
            "Mutually exclusive with --time-offset."
        ),
    )
    p_camsync.add_argument(
        "--xyz",
        dest="xyz",
        type=str,
        default=None,
        metavar="X,Y,Z",
        help=(
            "Optional offset in the pose-source camera frame (meters), "
            "comma-separated, e.g. 0,0.12,0. Applied when copying pose."
        ),
    )
    p_camsync.add_argument(
        "--auto-time",
        dest="auto_time",
        action="store_true",
        help=(
            "Estimate time offset from 3D nearest pose per target (same chunk); "
            "verbose report; falls back to earliest-EXIF delta if too few inliers."
        ),
    )
    p_camsync.add_argument(
        "--auto-xyz",
        dest="auto_xyz",
        action="store_true",
        help=(
            "After time alignment, estimate pose-local x,y,z offset from "
            "time-matched camera centers (median; uses scale_factor from YAML)."
        ),
    )
    p_camsync.add_argument(
        "--auto-offsets",
        dest="auto_offsets",
        action="store_true",
        help="Shorthand for --auto-time and --auto-xyz together.",
    )
    p_camsync.add_argument(
        "--spatial-max-dist",
        dest="spatial_max_dist",
        type=float,
        default=0.5,
        metavar="M",
        help=(
            "Max metric distance (m) for spatial NN time pair "
            "(dist_stored * scale_factor). Default: 0.5."
        ),
    )
    p_camsync.add_argument(
        "--min-spatial-pairs",
        dest="min_spatial_pairs",
        type=int,
        default=3,
        metavar="N",
        help="Minimum inlier pairs required for spatial auto time. Default: 3.",
    )
    p_camsync.add_argument(
        "--yes",
        "-y",
        dest="yes",
        action="store_true",
        help="Skip confirmation prompt after auto offset reports (non-interactive).",
    )

    # transform
    p_transform = subparsers.add_parser(
        "transform",
        help=(
            "Transform annotations: load from file, apply transform(s) to orig_coords, "
            "and save result."
        ),
    )
    p_transform.add_argument(
        "input",
        type=str,
        help="Path to input annotations CSV file.",
    )
    p_transform.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help=(
            "Optional output annotations CSV filepath. "
            "Default: <input_basename>_transformed.csv"
        ),
    )
    p_transform.add_argument(
        "--no_header",
        dest="no_header",
        action="store_true",
        help="Set header=False when loading annotations (default: header=True).",
    )
    p_transform.add_argument(
        "--ignore_header",
        dest="ignore_header",
        action="store_true",
        help="Skip the first line of the file and ignore header argument (default: False).",
    )
    p_transform.add_argument(
        "--inverse",
        dest="inverse",
        action="store_true",
        help="Apply the inverse of the cumulative transforms (default: False).",
    )

    # train
    p_train = subparsers.add_parser(
        "train",
        help=(
            "Train a FastAI crop classifier: collate labelled annotations "
            "across CSVs, generate train/val/test crops, train, and report "
            "stats."
        ),
    )
    p_train.add_argument(
        "pattern",
        nargs="?",
        default=settings.TRAIN_DEFAULT_PATTERN,
        help=(
            "Glob pattern of annotation CSVs to collate, matched inside "
            f"--csv-path. Default: {settings.TRAIN_DEFAULT_PATTERN!r}."
        ),
    )
    p_train.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Classes CSV path (default: classes.csv in CWD).",
    )
    p_train.add_argument(
        "--csv-path",
        dest="csv_path",
        type=str,
        default=None,
        help=(
            "Directory the annotation CSVs are matched in (default: CWD)."
        ),
    )
    p_train.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default=None,
        help=(
            "Base directory used only to locate images via the standard path "
            "convention when a CSV's cam_filepath directory is missing "
            "(default: CWD)."
        ),
    )
    p_train.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for crops and the training annotations CSV "
            "(default: CWD)."
        ),
    )
    p_train.add_argument(
        "--min-count",
        dest="min_count",
        type=int,
        default=1,
        help="Only display/train tree items with aggregated count >= this.",
    )
    p_train.add_argument(
        "--tips_only",
        action="store_true",
        help="Only bold tip entries (do not bold heavy parent nodes).",
    )
    p_train.add_argument(
        "--include-classes",
        dest="include_classes",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Explicit list of category labels to train on (the codes shown in "
            "brackets in the tree). Overrides --min-count/--tips_only: exactly "
            "these are bolded and trained. Errors if any is absent from the "
            "tree."
        ),
    )
    p_train.add_argument(
        "--collapse",
        action="store_true",
        help=(
            "Fold non-selected descendants into their nearest selected parent "
            "class when seeding the map. E.g. with --include-classes MAF, the "
            "children MAFG/MAF_T are trained as MAF; without --collapse only "
            "MAF itself is trained and its descendants are excluded."
        ),
    )
    conf_group = p_train.add_mutually_exclusive_group()
    conf_group.add_argument(
        "--min_conf",
        dest="min_conf",
        type=float,
        default=None,
        metavar="CONF",
        help=(
            "Only train on annotations whose label_conf is >= this value. "
            "Annotations with an empty (no value) label_conf are included by "
            "default. Mutually exclusive with --min_conf_strict."
        ),
    )
    conf_group.add_argument(
        "--min_conf_strict",
        dest="min_conf_strict",
        type=float,
        default=None,
        metavar="CONF",
        help=(
            "Like --min_conf, but treats an empty (no value) label_conf as 0, "
            "so such annotations are excluded whenever CONF > 0. Mutually "
            "exclusive with --min_conf."
        ),
    )
    p_train.add_argument(
        "--label-map",
        dest="label_map",
        type=str,
        default=None,
        help=(
            "Editable label->training_class map CSV (selection + collapse). "
            f"Default: <output>/{settings.TRAIN_LABEL_MAP_FILE}. Seeded from "
            "the tree on first run; hand edits are preserved on re-runs."
        ),
    )
    p_train.add_argument(
        "--keep-map",
        dest="keep_map",
        action="store_true",
        help=(
            "Preserve the existing label map (append only newly-seen labels) "
            "instead of re-seeding it from the current --min-count/--tips_only/"
            "--include-classes/--collapse selection. Use this after hand-editing "
            "the map (e.g. between --prepare-only and training)."
        ),
    )
    p_train.add_argument(
        "--prepare-only",
        dest="prepare_only",
        action="store_true",
        help=(
            "Generate crops and write/merge the label map, then stop before "
            "training so the map can be hand-tuned; re-run `substrata train` "
            "to train on the edited map."
        ),
    )
    p_train.add_argument(
        "--crop-size",
        dest="crop_size",
        type=int,
        default=None,
        help=(
            "Square crop width/height in pixels centred on (cam_x, cam_y). "
            f"Default: {settings.TRAIN_CROP_SIZE}."
        ),
    )
    p_train.add_argument(
        "--arch",
        type=str,
        default=settings.TRAIN_DEFAULT_ARCH,
        help=f"torchvision architecture. Default: {settings.TRAIN_DEFAULT_ARCH}.",
    )
    p_train.add_argument(
        "--epochs",
        type=int,
        default=settings.TRAIN_DEFAULT_EPOCHS,
        help=f"Fine-tuning epochs. Default: {settings.TRAIN_DEFAULT_EPOCHS}.",
    )
    p_train.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Exported learner .pkl path (also the load target with --test). "
            f"Default: <output>/{settings.TRAIN_DEFAULT_MODEL_FILE}."
        ),
    )
    p_train.add_argument(
        "--jobs",
        type=int,
        default=settings.TRAIN_CROP_JOBS,
        help=(
            "Parallel workers for crop generation (-1 = all cores). "
            f"Default: {settings.TRAIN_CROP_JOBS}."
        ),
    )
    p_train.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Skip training; load --model and re-run stats on the validation "
            "crops."
        ),
    )
    p_train.add_argument(
        "--test",
        action="store_true",
        help="Skip training; load --model and report stats on the test crops.",
    )
    p_train.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmations (labels, deletions, paths).",
    )

    args = parser.parse_args()

    handlers = {
        "decimate": handle_decimate,
        "metashape-export": handle_metashape_export,
        "repair": handle_repair,
        "head": handle_head,
        "scalebars": handle_scalebars,
        "views": handle_views,
        "orient": handle_orient,
        "colors": handle_colors,
        "firefish": handle_firefish,
        "cams2video": handle_cams2video,
        "intercepts": handle_intercepts,
        "intercepts-plot": handle_intercepts_plot,
        "align": handle_align,
        "images": handle_images,
        "camsync": handle_camsync,
        "transform": handle_transform,
        "train": handle_train,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
