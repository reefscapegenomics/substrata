# Standard Library
import argparse
import ast
from collections import Counter
import json
import os
import re
import shutil
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
from substrata import settings, sidecars

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
    max_pairs: int = settings.CAMSYNC_PDF_PAIRS,
    intercept_search_radius: float = 0.01,
) -> None:
    """Write a PDF: first a text summary page, then pose/target image pairs.

    Pairs are spread over pages of at most ``settings.CAMSYNC_PDF_PAIRS_PER_PAGE``
    rows each.

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
        max_pairs: Number of match rows to render (default
            ``settings.CAMSYNC_PDF_PAIRS``).
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

    pdf = PdfPages(output_path)
    if camsync_summary is not None:
        fig_sum = _camsync_summary_figure(camsync_summary)
        pdf.savefig(fig_sum)
        plt.close(fig_sum)

    n_total = len(pairs)
    per_page = max(1, int(settings.CAMSYNC_PDF_PAIRS_PER_PAGE))
    for start in range(0, n_total, per_page):
        page_pairs = pairs[start : start + per_page]
        n = len(page_pairs)
        fig, axes = plt.subplots(n, 2, figsize=(10, max(2.5, 2.6 * n)))
        axes = np.atleast_2d(axes)

        for i, (pose_cam, tgt_cam) in enumerate(page_pairs):
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

        page_range = f"pairs {start + 1}-{start + n} of {n_total}"
        fig.suptitle(
            "Camsync sanity check: pose vs target pairs (EXIF time match), "
            f"{page_range}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Wrote camsync sanity PDF: {output_path} ({n_total} pairs)")


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
    """Decimate a PLY to a target point count, writing a smaller PLY.

    Resolves the input PLY and output path from the project initializer
    (CWD) when not given explicitly, and can apply the project's stored
    color correction to the written points.

    Args:
        args: Parsed CLI arguments.
    """
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


def handle_stratify(args):
    """Stratify a PLY: reorder it for progressive and regional reads.

    Resolves the input from the project initializer (CWD) when not given and
    reorders it in place (writing ``<id>.strat.npz`` next to it), or writes a
    stratified copy into --out-dir (e.g. a fast local disk).

    Args:
        args: Parsed CLI arguments.
    """
    from substrata.stratified import is_stratified, stratify_ply, stratify_space

    input_path = args.input
    if not input_path:
        from substrata.initializer import ProjectInitializer

        base, cwd = _cwd_base()
        init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))
        input_path = init.ply_full_path
    if not input_path:
        raise SystemExit(
            "No input PLY found. Provide --input or run in a project folder with "
            "<id>.ply."
        )
    output_path = None
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        output_path = os.path.join(args.out_dir, os.path.basename(input_path))
    target = output_path or input_path
    if is_stratified(target) and not args.force:
        raise SystemExit(f"{target} is already stratified (use --force to redo it).")
    work_dir = args.work_dir or os.environ.get("SUBSTRATA_WORK_DIR") or None
    if output_path is None:
        sp = stratify_space(
            input_path, args.tmp_dir, work_dir=work_dir, mode=args.mode
        )
        gb = 1e9
        print(
            f"stratify: {sp['mode']} mode (RAM {sp['mem_need'] / gb:.0f} GB, "
            f"{sp['mem_free'] / gb:.0f} GB available); disk {sp['need'] / gb:.0f} GB "
            f"needed, {sp['free'] / gb:.0f} GB free"
            + (
                f"; work disk {sp['work_need'] / gb:.0f} GB needed, "
                f"{sp['work_free'] / gb:.0f} GB free"
                if "work_need" in sp
                else ""
            )
        )
        if not sp["ok"]:
            raise SystemExit("stratify: not enough disk space.")
        if sp["mode"] == "memory" and sp["mem_need"] > sp["mem_free"]:
            raise SystemExit(
                "stratify: not enough RAM available for --mode memory "
                "(use --mode stream, or free memory first)."
            )
    stratify_ply(
        input_path,
        output_path,
        mode=args.mode,
        cluster_size=args.cluster_size,
        tmp_dir=args.tmp_dir,
        work_dir=work_dir,
        progress=True,
        keep_original=args.keep_original,
    )


def handle_ply_repair(args):
    """Rewrite a PLY into a strict Open3D-compatible form.

    Drops extra vertex properties and non-finite rows (float32 xyz, optional
    uchar RGB and float32 normals). With no --output the input is renamed to
    <input>_old.ply and the repaired file is written back in its place; the
    original is restored if the rewrite fails.

    Args:
        args: Parsed CLI arguments.
    """
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


def handle_explore(argv):
    """Run substrata-explore (the interactive viewer) with these arguments.

    Args:
        argv: Its command-line arguments (after ``explore``).
    """
    try:
        from substrata_explore.cli import main as explore_main
    except ImportError:
        raise SystemExit(
            "substrata explore needs the substrata-explore package: "
            "pip install -e <path to substrata-explore>"
        )
    explore_main(argv)


def handle_head(args):
    """Print the first N vertex rows of a PLY for a quick preview.

    For a stratified PLY (``<id>.strat.npz`` next to it) the index summary
    follows: clusters, rounds and the prefix sizes of its uniform subsamples.
    Resolves the input PLY from the project initializer (CWD) when not given
    explicitly.

    Args:
        args: Parsed CLI arguments.
    """
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
    # a stratified PLY: also its index (clusters, rounds, prefix sizes)
    from substrata.stratified import StratifiedPly, is_stratified

    if is_stratified(input_path):
        print()
        print(StratifiedPly(input_path).summary())
    if init.baseline_id:
        print()
        print(f"baseline: {init.baseline_id} ({_baseline_status(init)})")


def _baseline_status(init) -> str:
    """``fresh``, ``stale`` or why not, for a project with a baseline_id."""
    if init.baseline_transform is None:
        return "no baseline_transform"
    path = init.find_baseline_project()
    if not path:
        return "not found"
    baseline = ProjectInitializer(path=path)
    if init.world_transform_is_stale(baseline):
        return "stale; run `substrata baseline --update`"
    return "fresh"


def handle_scalebars(args):
    """Generate a scalebar QC PDF from a point cloud and marker annotations.

    Loads the PLY (optionally stream-decimated to --points) and the markers
    CSV, builds Scalebars from the RGL target definitions, and writes a PDF.
    With --save_yaml the computed scale factor is persisted to the project
    YAML.

    Args:
        args: Parsed CLI arguments.
    """
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # 1) resolve inputs (the PLY only feeds the figures, so prefer the
    #    decimated <id>_dec50M.ply when present)
    pcd_path = args.input or init.ply_filepath
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
    """Colour calibration QC PDF, and the affine correction in the project YAML.

    The fitted ``color_correction`` (matrix + offset) is written to the project
    YAML by default (``--no-save-yaml`` to skip); loading the project then
    applies it to the point cloud. No PLY is written: a colour-corrected copy
    is an explicit extra step (``substrata decimate -c``).
    """
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
            from substrata.initializer import set_project_yaml_keys

            init.color_correction = cc.color_correction
            yaml_path = init.yaml_path or os.path.join(init.path or cwd, f"{init.id}.yaml")
            if os.path.isfile(yaml_path):
                # Only this key changes: comments and relative paths are kept
                set_project_yaml_keys(
                    yaml_path, {"color_correction": init.color_correction}
                )
            else:
                init.save_config_to_yaml(yaml_path)
            print(f"Saved colour correction to YAML: {yaml_path}")
    else:
        print("Not saving the colour correction (--no-save-yaml)")


def handle_views(args):
    """Save a composite-views PDF for a point cloud.

    Initializes the project (loading the point cloud, and cameras/markers
    when available) and renders multiple perspectives to a PDF.

    Args:
        args: Parsed CLI arguments.
    """
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


def handle_segment(args):
    """Segment the point cloud by classifying camera patches at query points.

    The classified query points are saved as an annotation set
    (``<id>.annos/<id>_seg[_<tag>]_points.csv`` + sidecar), the segmentation's
    source of truth. ``--from-points`` re-reads such a set (e.g. after checking
    labels by hand) and only rewrites the recoloured clouds.
    """
    import numpy as np

    from substrata.initializer import ProjectInitializer
    from substrata import segmentation

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))
    if args.input:
        init.pcd_filepath = args.input
    init.initialize()

    if init.pcd is None:
        raise SystemExit("No point cloud found to segment (provide --input).")
    from_points = getattr(args, "from_points", None)
    names = _seg_output_names(init, cwd, getattr(args, "tag", None))
    if from_points:
        points_csv = os.path.abspath(from_points)
        if not os.path.isfile(points_csv):
            raise SystemExit(f"No segmentation points CSV {points_csv}")
        if getattr(args, "tag", None) is None:  # the clouds keep the set's own tag
            names = _seg_output_names(
                init, cwd, (sidecars.read_annotation_sidecar(points_csv)
                            .get("segmentation") or {}).get("tag")
            )
        seg = segmentation.Segmentation.from_annotations(
            points_csv, world_transform=init.pcd.world_transform
        )
        print(f"Read segmentation: {points_csv} ({seg.n_queries:,} query points)")
    else:
        seg, points_csv = _segment_classify(args, init, names)

    # Propagate to the decimated cloud, print the category breakdown, and write a
    # recoloured copy.
    pts = np.asarray(init.pcd.points)
    codes = seg.propagate(pts, max_radius=args.max_radius)
    print("Categories:")
    for lbl, cnt in sorted(seg.summary(codes).items(), key=lambda kv: -kv[1]):
        print(f"  {lbl}: {cnt:,} ({100 * cnt / len(codes):.1f}%)")

    ply_out = args.output_ply or names["ply_dec"]
    _write_recolored_decimated(init.pcd, seg, codes, ply_out)
    print(f"Saved recoloured decimated cloud: {ply_out}")
    made = {"seg_ply_dec": ply_out, "seg_ply": None}  # None: not (re)made now

    # Optional streaming full-size recolour.
    if getattr(args, "full_ply", None) is not None:
        full_in = args.full_ply or init.ply_full_path
        if not full_in:
            raise SystemExit("--full-ply given but no full-size PLY path found.")
        full_out = names["ply"]
        segmentation.recolor_ply_file(
            full_in, full_out, seg,
            world_transform=init.pcd.world_transform,
            max_radius=args.max_radius,
        )
        print(f"Saved recoloured full-size cloud: {full_out}")
        made["seg_ply"] = full_out

    # The set's sidecar names the clouds made from it (and the colours used)
    params = sidecars.read_annotation_sidecar(points_csv).get("segmentation") or {}
    params = dict(params, colors={**(params.get("colors") or {}), **seg.hex_colors()},
                  max_radius=None if args.max_radius is None else float(args.max_radius))
    sidecars.write_annotation_sidecar(points_csv, {**made, "segmentation": params})


def _seg_output_names(init, cwd, tag=None):
    """Output paths of ``substrata segment`` (version ``tag``, if given).

    ``points``: ``<id>.annos/<id>_seg[_<tag>]_points.csv`` (the source of truth);
    ``ply_dec`` / ``ply``: ``<id>_seg[_<tag>]_dec.ply`` / ``<id>_seg[_<tag>].ply``
    in the plot folder (large, so never in ``.annos``).
    """
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise SystemExit(f"Invalid --tag {tag!r}; use letters, digits, '.', '_' or '-'.")
    stem = f"seg_{tag}" if tag else "seg"
    annos_dir = init.annos_root or os.path.join(cwd, f"{init.id}.annos")
    return {
        "points": os.path.join(annos_dir, f"{init.id}_{stem}_points.csv"),
        "ply_dec": _get_output_filepath(init, f"{stem}_dec.ply"),
        "ply": _get_output_filepath(init, f"{stem}.ply"),
        "masks": _get_output_filepath(init, f"{stem}_masks"),
    }


def _segment_classify(args, init, names):
    """Sample, match and classify query points; save them as the seg points CSV.

    Returns ``(segmentation, points_csv)``.
    """
    from substrata import classification, segmentation

    if init.cams is None:
        raise SystemExit("No cameras found; segmentation needs camera poses + photos.")

    # Remap camera photo paths when the poses reference a different mount than
    # where the photos live locally (same as cams.set_filepath_replace).
    if getattr(args, "photo_path_replace", None):
        init.cams.set_filepath_replace(*args.photo_path_replace)

    classifier_path = getattr(args, "classifier", None) or init.classifier_filepath
    if not classifier_path:
        raise SystemExit(
            "No classifier found. Provide --classifier or set 'classifier' in "
            "the project YAML."
        )
    learn = classification.get_image_classifier(classifier_path)
    crop_size = _classifier_crop_size(
        [args.crop_size] if args.crop_size else None, learn,
        (settings.TRAIN_CROP_SIZE, settings.TRAIN_CROP_SIZE),
    )[0]
    cell_size = args.cell_size or settings.DEFAULT_SEG_CELL_SIZE

    # Manual category colours: --color LABEL=RRGGBB (repeatable).
    label_colors = {}
    for spec in getattr(args, "color", None) or []:
        if "=" not in spec:
            raise SystemExit(f"Invalid --color {spec!r}; expected LABEL=RRGGBB.")
        lbl, hexv = spec.split("=", 1)
        label_colors[lbl] = _parse_hex_color(hexv)

    from substrata import pointfeatures

    point_clf = pointfeatures.is_point_classifier(learn)
    dense = point_clf and not getattr(args, "per_point", False)
    views = (getattr(args, "views", None) or settings.SEG_POINT_VIEWS) if dense else 1
    mask_dir = None
    if getattr(args, "save_masks", None) is not None:
        if not dense:
            raise SystemExit("--save-masks needs a point classifier (dense maps).")
        mask_dir = args.save_masks or names["masks"]
        os.makedirs(mask_dir, exist_ok=True)
    seg = segmentation.segment_point_cloud(
        init.pcd, init.cams, learn,
        world_transform=init.pcd.world_transform,
        cell_size=cell_size,
        sampling=args.sampling,
        occlusion=not args.no_occlusion,
        label_colors=label_colors or None,
        crop_size=crop_size,
        views=views,
        dense=dense,
        mask_dir=mask_dir,
    )
    if mask_dir:
        print(f"Saved per-photo masks: {mask_dir}")

    if args.output_npz:
        seg.save(args.output_npz)
        print(f"Saved segmentation: {args.output_npz} ({seg.n_queries:,} query points)")

    # The classified query points as an annotation set (predictions in pred_*,
    # with the photo + pixel each was classified from): the source of truth
    points_csv = names["points"]
    os.makedirs(os.path.dirname(points_csv), exist_ok=True)
    seg.to_annotations(init.pcd.world_transform, id_prefix=init.id).save(points_csv)
    sidecars.write_annotation_sidecar(points_csv, {
        "type": "segmentation",
        "plot_id": init.id,
        "world_transform": init.world_transform,
        "classifier": classifier_path,
        "segmentation": {
            "tag": getattr(args, "tag", None),
            "cell_size": float(cell_size),
            "sampling": args.sampling,
            "crop_size": int(crop_size),
            "occlusion": not args.no_occlusion,
            "colors": seg.hex_colors(),
            **({"method": "dense" if dense else "per-point", "views": int(views)}
               if point_clf else {}),
            **({"masks": mask_dir} if mask_dir else {}),
        },
    }, update=False)
    print(f"Saved classified query points: {points_csv} ({seg.n_queries:,})")
    return seg, points_csv


def _write_recolored_decimated(pcd, seg, codes, output_path):
    """Write the decimated cloud with segmentation colours to a PLY."""
    import numpy as np
    import open3d as o3d

    new_colors = seg.recolor(np.asarray(pcd.colors), codes)
    o3d_pcd = pcd.o3d_pcd
    o3d_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    o3d.io.write_point_cloud(output_path, o3d_pcd)


def handle_orient(args):
    """Compute and apply scale and orientation, saving results to YAML.

    Runs the scale_and_orient workflow (using marker depths from --markers
    or camera depths otherwise) unless --manual is set, or applies the
    Metashape chunk_transform (as a local ENU frame) with --use_chunk_transform.
    Optionally applies
    an interactively entered manual transform, and propagates the world
    transform to cameras/markers/annotations. Writes the project YAML plus
    composite-views and depth-residuals PDFs.

    Args:
        args: Parsed CLI arguments.
    """
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize(apply_transform=False)

    use_chunk = getattr(args, "use_chunk_transform", False)
    if getattr(args, "north_up", False) and not use_chunk:
        raise SystemExit("--north_up requires --use_chunk_transform")
    if use_chunk and (getattr(args, "manual", False) or args.markers):
        raise SystemExit(
            "--use_chunk_transform cannot be combined with --manual or --markers"
        )

    depth_markers = None
    if use_chunk:
        # Apply the Metashape chunk transform (scale derived from it)
        init.orient_from_chunk_transform(
            north_up=getattr(args, "north_up", False)
        )
    # Run scale_and_orient workflow (skip if --manual flag is set)
    elif not getattr(args, "manual", False):
        # Use markers CSV filepath if provided, otherwise use camera depths
        depth_markers_filepath = getattr(args, "markers", None)
        if depth_markers_filepath is not None:
            depth_markers = Annotations(depth_markers_filepath, orig_coords_only=True)
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

    # Optionally align along slope (same as `intercepts --slope`), persisted
    if getattr(args, "slope", False):
        prev_transform = init.pcd.world_transform.copy()
        _, slope_elev = init.pcd.apply_along_slope_transform()
        print(f"Aligned along slope (elevation {slope_elev:.2f}°)")
        # Propagate only the slope step (other objects already carry the rest)
        slope_transform = init.pcd.world_transform @ np.linalg.inv(prev_transform)
        for attr_name in ("cams", "markers", "annotations"):
            obj = getattr(init, attr_name, None)
            if obj is not None:
                obj.transform_coords(slope_transform)
        init.world_transform = init.pcd.world_transform

    if init.baseline_id:
        print(
            f"Warning: {init.id} inherits its frame from baseline "
            f"{init.baseline_id}; this orientation replaces it and "
            "`substrata baseline --update` would restore the baseline's. Remove "
            "baseline_id/baseline_transform from the YAML to keep this one."
        )
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
        beam_half_angle=args.beam_half_angle,
        distance_percentile=(
            None if args.distance_percentile < 0 else args.distance_percentile
        ),
        sonar_offset=args.sonar_offset,
        recompute_camdists=args.recompute_camdists,
        depth_sensor_offset=args.depth_sensor_offset,
        water_level=args.water_level,
    )
    # Residual analyses on the full container use the same sensor positions
    init.cams.depth_sensor_offset = args.depth_sensor_offset
    init.cams.depth_sensor_offset_scale = init.scale_factor
    if any(args.depth_sensor_offset):
        init.firefish_depth_sensor_offset = list(args.depth_sensor_offset)
    if args.water_level:
        init.firefish_water_level = args.water_level

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
    """Create an annotated fly-through video from the project cameras.

    Initializes the project, optionally loads annotations and subsets the
    cameras by group name, then renders an MP4 walking through the camera
    poses over the point cloud.

    Args:
        args: Parsed CLI arguments.
    """
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
        init.use_annotations_file(args.annotations_file)

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
    """Compute Z-intercepts over a grid and save them to CSV.

    Requires a non-identity world transform. Optionally aligns along the
    slope first, places a bounding box (automatically or from --position),
    subdivides it into cells, samples a random XY point per cell, and
    computes the point-cloud Z-intercepts. Writes a grid image and an
    intercepts CSV with its sidecar YAML (``<csv>.yaml``).

    Args:
        args: Parsed CLI arguments.
    """
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

    # Save the grid image and intercepts CSV, plus the CSV's sidecar YAML
    slope = getattr(args, "slope", False)
    mode = "slope" if slope else "topdown"
    fig.savefig(_get_output_filepath(init, f"{mode}_bbox.png"))
    # Annotation sets live in <id>.annos/ (with their sidecars and crops)
    annos_dir = init.annos_root or os.path.join(os.getcwd(), f"{init.id}.annos")
    os.makedirs(annos_dir, exist_ok=True)
    csv_path = os.path.join(annos_dir, f"{init.id}_{mode}_intercepts.csv")
    intercepts.save(csv_path)

    # Grid frame = grid_transform @ world_transform @ orig, where grid_transform
    # is the slope alignment (identity for topdown)
    grid_transform = init.pcd.world_transform @ np.linalg.inv(init.world_transform)
    (x0, y0), (x1, y1) = [[float(v) for v in xy] for xy in optimal_bbox]
    # bbox corners in orig coords (for display), at the intercepts' median height
    z = float(np.nanmedian([c[2] for c in intercepts.coords])) if intercepts else 0.0
    corners = np.array([[x0, y0, z, 1], [x1, y0, z, 1], [x1, y1, z, 1], [x0, y1, z, 1]])
    bbox_orig = (np.linalg.inv(init.pcd.world_transform) @ corners.T)[:3].T
    grid = {
        "mode": mode,
        "grid_transform": grid_transform if slope else None,
        "grid_bbox": [[x0, y0], [x1, y1]],
        "cell_size": float(args.box_size),
        "box_length": float(args.box_length),
        "box_width": float(args.box_width),
        "search_radius": float(args.search_radius),
        "position": [x0, y0] if getattr(args, "position", None) else "auto",
        "slope_elevation_deg": float(slope_elev) if slope else None,
        "bbox_orig": bbox_orig,
    }
    sidecar_path = sidecars.write_annotation_sidecar(
        csv_path,
        {
            "type": "intercepts",
            "plot_id": init.id,
            "classes": init.classes_filepath,
            "world_transform": init.world_transform,
            "intercepts": grid,
        },
    )
    print(f"Saved intercepts to {csv_path} (settings: {sidecar_path})")

    if slope:
        # additional visualization for sanity check
        a, b, c, d, inliers_idx = measurements.get_best_fit_plane_PCA(init.pcd)
        visualizations.visualize_elevation_angle(
            init.pcd,
            [a, b, c, d],
            point_size=1,
            output_filename=_get_output_filepath(init, "slope_pca.png"),
        )


def handle_intercepts_plot(args):
    """Re-plot a saved intercepts CSV as a 2D grid colored by label.

    Reads the orientation (and optional exact generation grid) from --yaml
    or the project, applies it to the annotations, and builds an OrthoGrid
    aligned via the saved grid_bbox, --fit-grid, or --box-length/--box-width
    reconstruction. Saves a grid PNG, an animated GIF of the grid filling in,
    and (when a point cloud is available) an annotation positions PNG.

    Args:
        args: Parsed CLI arguments.
    """
    from substrata import measurements
    from substrata.ortho import OrthoGrid

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    ann_path = args.annotations or init.annotations_filepath
    if not ann_path:
        raise SystemExit(
            "intercepts-plot requires an intercepts file: pass --annotations "
            "<id>.annos/<id>_<mode>_intercepts.csv"
        )

    # Read the generation settings (orientation + optional exact grid) from --yaml,
    # else from the CSV's sidecar (<csv>.yaml written by `intercepts`), else fall
    # back to the project's own world_transform. Sidecars keep the grid in an
    # `intercepts:` block (grid frame = grid_transform @ world_transform); older
    # <id>_slope_intercepts.yaml files hold the combined world_transform and
    # grid_bbox at the top level.
    if getattr(args, "yaml", None):
        with open(args.yaml) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = sidecars.read_annotation_sidecar(ann_path)
    grid_cfg = cfg.get("intercepts") or cfg
    transform = None
    if cfg.get("world_transform") is not None:
        transform = np.array(cfg["world_transform"], dtype=float)
        if grid_cfg.get("grid_transform") is not None:
            transform = np.array(grid_cfg["grid_transform"], dtype=float) @ transform
    grid_bbox = grid_cfg.get("grid_bbox")
    if transform is None:
        transform = init.world_transform
    if np.allclose(transform, np.eye(4)):
        print(
            "Warning: orientation transform is identity; the grid may be warped "
            "for slope intercepts (pass --yaml with the generation world_transform)."
        )

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
        sweep=getattr(args, "animation_order", "columns"),
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


def _load_align_input(path: str, label: str) -> dict:
    """Resolve an align --source/--target argument (PLY, project YAML or dir).

    A project directory is resolved like other subcommands do (its
    ``<dirname>.yaml`` if present, else filename conventions).

    Returns a dict with ``ply``, ``cams_meta_json``, ``world_transform`` (4x4,
    identity for a bare PLY), ``scale_factor`` and ``init`` (the
    ProjectInitializer, or None for a bare PLY). Nothing heavy is loaded.
    """
    is_dir = os.path.isdir(path)
    if not is_dir and not os.path.isfile(path):
        raise SystemExit(f"align: {label} not found: {path}")
    if is_dir or os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        init = (
            ProjectInitializer(path=path) if is_dir else ProjectInitializer(yaml=path)
        )
        return {
            "ply": init.ply_filepath,
            "cams_meta_json": init.cams_meta_json_filepath,
            "world_transform": np.asarray(init.world_transform, dtype=float),
            "scale_factor": init.scale_factor,
            "init": init,
        }
    return {
        "ply": path,
        "cams_meta_json": None,
        "world_transform": np.eye(4),
        "scale_factor": None,
        "init": None,
    }


def _align_on_cameras(
    src: dict, tgt: dict, with_scaling: bool, max_residual: float | None = None
) -> dict:
    """Fit the raw source → raw target transform from same-named cameras.

    Returns a dict with the 4x4 transform ``T`` and, for reporting, the raw
    matched camera centres ``src_xyz``/``tgt_xyz``, the ``inliers`` mask and
    the matched ``names``.

    Outlier pairs are rejected iteratively (see
    ``geom.fit_correspondences_rejecting_outliers``). ``max_residual`` is in
    metres when the target has a ``scale_factor``, else in target raw units;
    if None, a robust (median/MAD-based) threshold is used.
    """
    from substrata.cameras import Cameras
    from substrata.geom import fit_correspondences_rejecting_outliers

    for label, info in (("source", src), ("target", tgt)):
        if info["init"] is None:
            raise SystemExit(
                f"align --cameras requires a project YAML or directory as {label}"
            )
        meta = info["cams_meta_json"]
        if not meta or not os.path.isfile(meta):
            raise SystemExit(
                f"align --cameras: {label} project has no cameras meta JSON ({meta})"
            )
    src_cams = Cameras(src["cams_meta_json"])
    tgt_cams = Cameras(tgt["cams_meta_json"])
    matches = src_cams.match_by_name(tgt_cams)
    print(f"Matched {len(matches)} camera(s) by name")
    if len(matches) < 3:
        raise SystemExit(
            "align --cameras needs at least 3 cameras with matching names in "
            f"source and target; found {len(matches)}"
        )

    # Fit on the raw (untransformed) camera centres
    p = np.array([np.asarray(s.orig_coords, dtype=float) for _, s, _ in matches])
    q = np.array([np.asarray(t.orig_coords, dtype=float) for _, _, t in matches])
    to_m = tgt["scale_factor"]
    threshold_raw = None
    if max_residual is not None:
        threshold_raw = max_residual / to_m if to_m else max_residual
    try:
        T, inliers = fit_correspondences_rejecting_outliers(
            p, q, with_scaling=with_scaling, max_residual=threshold_raw
        )
    except ValueError as e:
        raise SystemExit(f"align --cameras: {e}")

    # Residuals in target raw units (and metres when the target is scaled)
    residuals = np.linalg.norm(T.apply_to_points(p) - q, axis=1)
    unit = "raw" + (" / m" if to_m else "")

    def _fmt(r):
        return f"{r:.4f}" + (f" / {r * to_m:.4f}" if to_m else "")

    rejected = [(m[0], r) for m, r, ok in zip(matches, residuals, inliers) if not ok]
    print(f"Rejected {len(rejected)} outlier camera(s) (residual, {unit}):")
    for name, r in sorted(rejected, key=lambda x: x[1], reverse=True):
        print(f"  {name}: {_fmt(r)}")
    r_in = residuals[inliers]
    print(f"Fit on {int(inliers.sum())} inlier camera(s), residuals ({unit}):")
    print(f"  RMS:    {_fmt(float(np.sqrt(np.mean(r_in**2))))}")
    print(f"  median: {_fmt(float(np.median(r_in)))}")
    print(f"  max:    {_fmt(float(r_in.max()))}")
    scale = np.cbrt(np.linalg.det(T.mat[:3, :3]))
    print(f"  Scale (source → target raw): {scale:.6f}")
    return {
        "T": T.as_ndarray(),
        "src_xyz": p,
        "tgt_xyz": q,
        "inliers": inliers,
        "names": [m[0] for m in matches],
    }


def handle_align(args):
    """Register a source to a target and print the transform.

    ``--source``/``--target`` are PLY files or project YAMLs. The transform is
    fitted between the raw (untransformed) models, either by point-cloud
    registration (FPFH/RANSAC + ICP) or, with ``--cameras``, from camera centres
    with matching image names (optionally refined by ICP with ``--refine``).

    Two matrices are printed: raw source → raw target, and raw source →
    oriented target (target ``world_transform`` applied on top). The latter can
    be applied directly to source ``orig_coords`` and is what ``--output-yaml``
    stores as the source project's ``world_transform``.

    Args:
        args: Parsed CLI arguments.
    """
    from substrata.pointclouds import PointCloud

    if args.refine and not args.cameras:
        raise SystemExit("align --refine is only valid together with --cameras")
    if args.max_residual is not None and not args.cameras:
        raise SystemExit("align --max-residual is only valid together with --cameras")

    src = _load_align_input(args.source, "source")
    tgt = _load_align_input(args.target, "target")
    if args.output_yaml and src["init"] is None:
        raise SystemExit(
            "align --output-yaml requires a project YAML or directory as --source"
        )
    if args.as_parent and args.as_baseline:
        raise SystemExit("align: use either --as-parent or --as-baseline")
    link = "--as-parent" if args.as_parent else "--as-baseline"
    if (args.as_parent or args.as_baseline) and (
        src["init"] is None or tgt["init"] is None
    ):
        raise SystemExit(
            f"align {link} requires project YAMLs or directories as --source "
            "and --target"
        )

    def _load_pcds():
        for label, info in (("source", src), ("target", tgt)):
            if not info["ply"] or not os.path.isfile(info["ply"]):
                raise SystemExit(f"align: {label} PLY not found ({info['ply']})")
        # Raw clouds (world_transform not applied), optional stream decimation
        return (
            PointCloud(src["ply"], max_points=args.points),
            PointCloud(tgt["ply"], max_points=args.points),
        )

    allow_scaling = not args.no_scaling
    cam_fit = None
    metrics = None
    src_pcd = tgt_pcd = None
    if args.cameras:
        cam_fit = _align_on_cameras(
            src, tgt, with_scaling=allow_scaling, max_residual=args.max_residual
        )
        T_raw = cam_fit["T"]
        if args.refine:
            src_pcd, tgt_pcd = _load_pcds()
            try:
                T_raw, metrics = src_pcd.get_auto_align_transform(
                    tgt_pcd,
                    voxel_size=args.voxel_size,
                    allow_scaling=allow_scaling,
                    init_transform=T_raw,
                )
            except ValueError as e:
                raise SystemExit(f"align --refine: {e}")
            print("ICP refinement metrics:", metrics)
    else:
        src_pcd, tgt_pcd = _load_pcds()
        # Transform that maps source → target space
        T_raw, metrics = src_pcd.get_auto_align_transform(
            tgt_pcd, voxel_size=args.voxel_size, allow_scaling=allow_scaling
        )
        print("Metrics:", metrics)

    T_raw = np.asarray(T_raw, dtype=float)
    W_t = tgt["world_transform"]
    T_final = W_t @ T_raw
    with np.printoptions(precision=8, suppress=True):
        print("Alignment transform (raw source → raw target):\n", T_raw)
        if not np.allclose(W_t, np.eye(4)):
            print(
                "Alignment transform (raw source → oriented target, i.e. target "
                "world_transform applied):\n",
                T_final,
            )

    if args.as_parent:
        _write_align_parent(args, src, tgt, T_raw)
    elif args.as_baseline:
        _write_align_baseline(args, src, tgt, T_raw)
    elif args.output_yaml:
        init = src["init"]
        old = np.asarray(init.world_transform, dtype=float)
        if not np.allclose(old, np.eye(4)):
            print(f"Replacing source world_transform:\n{old}")
        init.world_transform = T_final
        init.save_config_to_yaml(args.output_yaml)
        print(f"Wrote aligned source project YAML: {args.output_yaml}")

    if args.pdf:
        _save_align_qc_pdf(
            args, src, tgt, T_raw, T_final, cam_fit, metrics, src_pcd, tgt_pcd
        )


def _align_link_yaml(args, init, flag: str) -> str:
    """The YAML that align --as-parent/--as-baseline writes its keys to.

    The source project YAML itself, or a copy of it at ``--output-yaml``.
    """
    out = args.output_yaml or init.yaml_path
    if not out:
        raise SystemExit(
            f"align {flag}: the source has no project YAML; pass --output-yaml"
        )
    if args.output_yaml and init.yaml_path and os.path.isfile(init.yaml_path):
        shutil.copyfile(init.yaml_path, out)
    elif not os.path.isfile(out):
        init.save_config_to_yaml(out)
    return out


def _write_align_parent(args, src: dict, tgt: dict, T_raw: np.ndarray) -> None:
    """Store the target as the source project's parent (align --as-parent).

    Writes ``parent_id`` and ``parent_transform`` (raw source → raw target, the
    fitted transform including any scale) to the source project YAML, in place
    or to a copy at ``--output-yaml``. The source ``world_transform`` is left
    untouched; the implied scale ratio between the two world frames is
    reported as a check on both plots' scale factors.
    """
    from substrata.initializer import set_project_yaml_keys

    init, parent = src["init"], tgt["init"]
    init.parent_id = parent.id
    init.parent_transform = np.asarray(T_raw, dtype=float)
    if init.world_transform_is_identity or parent.world_transform_is_identity:
        print("Scale ratio: n/a (source or target has no world_transform)")
    else:
        ratio = init.parent_scale_ratio(parent)
        print(
            f"Scale ratio (parent m per local m): {ratio:.5f} "
            f"({(ratio - 1) * 100:+.2f} %)"
        )
    out = _align_link_yaml(args, init, "--as-parent")
    set_project_yaml_keys(
        out, {"parent_id": parent.id, "parent_transform": init.parent_transform}
    )
    print(f"Wrote parent_id={parent.id} and parent_transform to {out}")


def _write_align_baseline(args, src: dict, tgt: dict, T_raw: np.ndarray) -> None:
    """Store the target as the source project's baseline (align --as-baseline).

    Writes ``baseline_id`` and ``baseline_transform`` (raw source → raw target)
    and the frame inherited from the baseline: ``world_transform`` (=
    baseline ``world_transform @ baseline_transform``) with the scale factor
    and depth model carried over. In place, or to a copy at ``--output-yaml``.
    """
    from substrata.initializer import set_project_yaml_keys

    init, baseline = src["init"], tgt["init"]
    if baseline.world_transform_is_identity:
        print(f"Warning: baseline {baseline.id} has no world_transform yet")
    init.baseline_id = baseline.id
    init.baseline_transform = np.asarray(T_raw, dtype=float)
    keys = init.baseline_world_keys(baseline)
    out = _align_link_yaml(args, init, "--as-baseline")
    set_project_yaml_keys(
        out,
        {"baseline_id": baseline.id, "baseline_transform": init.baseline_transform,
         **keys},
    )
    print(
        f"Wrote baseline_id={baseline.id}, baseline_transform and the inherited "
        f"{', '.join(keys)} to {out}"
    )


def _save_align_qc_pdf(
    args, src, tgt, T_raw, T_final, cam_fit, metrics, src_pcd, tgt_pcd
):
    """Write the align QC PDF (summary, cameras, context, correspondence).

    Everything is drawn in the oriented target frame (metres when the target
    has a scale/orientation). The target is (re)loaded without the --points
    cap so the overlap is dense enough for meaningful point distances.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from substrata import visualizations
    from substrata.geom import Transform
    from substrata.pointclouds import PointCloud, SimplePointCloud

    W_t = tgt["world_transform"]
    oriented = not np.allclose(W_t, np.eye(4))
    units = "m" if oriented and tgt["scale_factor"] else "target units"
    labels = (_align_label(args.source, src), _align_label(args.target, tgt))

    for label, info in (("source", src), ("target", tgt)):
        if not info["ply"] or not os.path.isfile(info["ply"]):
            raise SystemExit(f"align --pdf: {label} PLY not found ({info['ply']})")
    if src_pcd is None:
        src_pcd = PointCloud(src["ply"], max_points=args.points)
    print("Loading target point cloud for the QC report (no --points cap)")
    tgt_pcd = PointCloud(tgt["ply"])

    src_qc = SimplePointCloud(
        Transform(T_final).apply_to_points(np.asarray(src_pcd.points)),
        src_pcd.colors,
    )
    tgt_qc = SimplePointCloud(
        Transform(W_t).apply_to_points(np.asarray(tgt_pcd.points)),
        tgt_pcd.colors,
    )

    figs = []
    summary = [
        "substrata align QC",
        "",
        f"source: {args.source}",
        f"target: {args.target}",
        f"method: {_align_method(args)}",
        f"frame:  {'oriented target' if oriented else 'raw target'} ({units})",
    ]
    if cam_fit is not None:
        inl = cam_fit["inliers"]
        res = np.linalg.norm(
            Transform(T_final).apply_to_points(cam_fit["src_xyz"])
            - Transform(W_t).apply_to_points(cam_fit["tgt_xyz"]),
            axis=1,
        )
        summary += [
            "",
            f"cameras matched: {len(inl)}, kept: {int(inl.sum())}, "
            f"rejected: {int((~inl).sum())}",
            f"camera residual RMS (kept, final transform): "
            f"{np.sqrt(np.mean(res[inl] ** 2)):.4f} {units}",
        ]
    if metrics is not None:
        stage = "ICP refinement" if cam_fit is not None else "registration"
        summary += ["", f"{stage} metrics (target raw units):"] + [
            f"  {k}: {round(v, 4) if isinstance(v, float) else v}"
            for k, v in metrics.items()
        ]
    scale = np.cbrt(np.linalg.det(T_raw[:3, :3]))
    with np.printoptions(precision=6, suppress=True):
        summary += [
            "",
            f"scale (source → target raw): {scale:.6f}",
            "",
            "raw source → raw target:",
            str(T_raw),
        ]
        if oriented:
            summary += ["", "raw source → oriented target:", str(T_final)]
    fig = visualizations.plot_text("\n".join(summary), width=11, height=8.5)
    fig.axes[0].texts[0].set_fontfamily("monospace")
    fig.axes[0].texts[0].set_fontsize(9)
    figs.append(fig)

    if cam_fit is not None:
        figs.append(
            visualizations.plot_camera_correspondence(
                Transform(T_final).apply_to_points(cam_fit["src_xyz"]),
                Transform(W_t).apply_to_points(cam_fit["tgt_xyz"]),
                inliers=cam_fit["inliers"],
                units=units,
                labels=labels,
                title="Matched camera residuals (final transform)",
            )
        )
    figs.append(
        visualizations.plot_pcd_context(
            src_qc, tgt_qc, labels=labels, units=units, title="Location in target"
        )
    )
    fig, stats = visualizations.plot_pcd_correspondence(
        src_qc,
        tgt_qc,
        labels=labels,
        units=units,
        title="Correspondence in the overlap",
        return_stats=True,
    )
    figs.append(fig)
    print("Point-cloud correspondence:", {k: round(v, 4) for k, v in stats.items()})

    with PdfPages(args.pdf) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote align QC PDF: {args.pdf}")


def _align_label(path: str, info: dict) -> str:
    """Short display name for an align input (project id or file name)."""
    init = info.get("init")
    if init is not None and init.id:
        return str(init.id)
    return os.path.splitext(os.path.basename(os.path.normpath(path)))[0]


def _align_method(args) -> str:
    """Human-readable description of the align method used."""
    if args.cameras:
        method = "matching cameras"
        if args.max_residual is not None:
            method += f" (max residual {args.max_residual})"
        else:
            method += " (automatic outlier rejection)"
        if args.refine:
            method += " + ICP refinement"
    else:
        method = "point clouds (FPFH/RANSAC + ICP)"
    if args.no_scaling:
        method += ", rigid"
    return method


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
        max_pairs=max(1, int(args.pdf_pairs)),
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



def _pr_prompt_duplicate(label, name, old_path, candidates):
    """Ask which of several same-named candidates to use.

    Args:
        label (str): What is being resolved, e.g. "Camera" or "YAML key".
        name (str): Identifier shown after the label.
        old_path (str): The value currently stored.
        candidates (list): Absolute paths sharing the same basename.

    Returns:
        tuple: ``(action, path)`` where action is "pick", "skip" or "abort".
    """
    from substrata.pathrepair import path_basename

    print(
        f"\n{label} {name}: {len(candidates)} candidates named "
        f"{path_basename(old_path)!r}"
    )
    print(f"  stored value: {old_path}")
    for i, cand in enumerate(candidates, start=1):
        print(f"  [{i}] {cand}")
    prompt = f"Choose 1-{len(candidates)}, [m]anual path, [s]kip, [a]bort: "
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("a", "abort"):
            return ("abort", None)
        if ans in ("s", "skip"):
            return ("skip", None)
        if ans in ("m", "manual"):
            return _pr_prompt_manual()
        if ans.isdigit() and 1 <= int(ans) <= len(candidates):
            return ("pick", candidates[int(ans) - 1])
        print("  Not a valid choice.")


def _pr_prompt_missing(label, name, old_path):
    """Ask where something is, when the search found nothing.

    Args:
        label (str): What is being resolved, e.g. "Camera" or "YAML key".
        name (str): Identifier shown after the label.
        old_path (str): The value currently stored.

    Returns:
        tuple: ``(action, path)`` where action is "pick", "skip" or "abort".
    """
    from substrata.pathrepair import path_basename

    print(
        f"\n{label} {name}: nothing named {path_basename(old_path)!r} found "
        "under the search root"
    )
    print(f"  stored value: {old_path}")
    return _pr_prompt_manual()


def _pr_prompt_missing_dir(label, old_dir, n):
    """Ask what to do with a folder none of whose images can be found.

    Typically a photo folder deliberately left out of a local copy, so it is
    asked once per folder instead of once per image.

    Args:
        label (str): What the images belong to, e.g. "Camera".
        old_dir (str): The stored folder.
        n (int): Number of images stored in it.

    Returns:
        tuple: ``(action, path)`` where action is "skip", "skip_all", "pick"
        (path is the new folder), "each" or "abort".
    """
    print(
        f"\n{label} folder {old_dir}: none of its {n} image(s) found under the "
        "search root"
    )
    while True:
        ans = input(
            "[s]kip folder, skip [all] missing folders, [p]ath to the folder, "
            "[i]ndividually, [a]bort: "
        ).strip()
        low = ans.lower()
        if low in ("a", "abort"):
            return ("abort", None)
        if low in ("s", "skip") or not ans:
            return ("skip", None)
        if low == "all":
            return ("skip_all", None)
        if low in ("i", "individually"):
            return ("each", None)
        if low in ("p", "path"):
            path = os.path.expanduser(input("Folder path: ").strip())
            if os.path.isdir(path):
                return ("pick", os.path.abspath(path))
            print(f"  Not an existing directory: {path}")
            continue
        print("  Not a valid choice.")


def _pr_prompt_manual():
    """Read a path from the user, re-prompting until it exists.

    A path that does not exist is never accepted: doing so would break the
    guarantee that every planned change is verified before anything is written.

    Returns:
        tuple: ``(action, path)`` where action is "pick", "skip" or "abort".
    """
    while True:
        ans = input("Enter a path, or [s]kip, [a]bort: ").strip()
        if ans.lower() in ("a", "abort"):
            return ("abort", None)
        if ans.lower() in ("s", "skip") or not ans:
            return ("skip", None)
        path = os.path.expanduser(ans)
        if os.path.exists(path):
            return ("pick", os.path.abspath(path))
        print(f"  Not an existing file or directory: {path}")


def _print_yaml_report(yplan, title="Project files"):
    """Print a YAML section (project or sidecar) of the path-repair report."""
    print(f"\n{title} ({os.path.basename(yplan.yaml_path)}):")
    width = max((len(e.key) for e in yplan.entries), default=4)
    for entry in yplan.entries:
        if entry.new is not None:
            print(
                f"  {entry.key:<{width}}  {entry.old}  ->  {entry.new}"
                f"   ({entry.how})"
            )
        elif entry.how in ("missing", "skipped"):
            print(f"  {entry.key:<{width}}  <{entry.how}: {entry.old}>")
        else:
            print(f"  {entry.key:<{width}}  ok")


def _print_cams_report(plan, max_rows, title="Camera images", unit="cameras"):
    """Print an image-path section, grouped by folder mapping."""
    from substrata.pathrepair import path_basename, path_dirname

    def name(cam_id):
        # Annotation image plans are keyed by the stored path itself
        return path_basename(cam_id) if cam_id == plan.stored.get(cam_id) else cam_id

    by_mapping = {}
    individual = []
    for cam_id, new_path in sorted(plan.changes.items()):
        old_dir = path_dirname(plan.stored.get(cam_id, ""))
        new_dir = path_dirname(new_path)
        if plan.dir_map.get(old_dir) == new_dir:
            by_mapping.setdefault((old_dir, new_dir), []).append((cam_id, new_path))
        else:
            individual.append((cam_id, new_path))

    print(f"\n{title} ({len(plan.changes)} to update):")
    for (old_dir, new_dir), rows in sorted(by_mapping.items()):
        print(f"  {old_dir}  ->  {new_dir}   ({len(rows)} {unit})")
        for cam_id, new_path in rows[:max_rows]:
            print(f"      {name(cam_id)}  {path_basename(new_path)}")
        if len(rows) > max_rows:
            print(f"      ... {len(rows) - max_rows} more")
    if individual:
        print("  (individual)")
        for cam_id, new_path in individual[:max_rows]:
            print(f"      {name(cam_id)}  {new_path}")
        if len(individual) > max_rows:
            print(f"      ... {len(individual) - max_rows} more")


def _print_skipped_dirs(plan, max_rows):
    """List the folders skipped as a whole, with their image counts."""
    from substrata.pathrepair import path_dirname

    if not plan.skipped_dirs:
        return
    counts = {}
    for path in plan.skipped.values():
        d = path_dirname(path)
        if d in plan.skipped_dirs:
            counts[d] = counts.get(d, 0) + 1
    print(f"  in {len(plan.skipped_dirs)} skipped folder(s):")
    for d in sorted(counts)[:max_rows]:
        print(f"    {d}  ({counts[d]})")
    if len(counts) > max_rows:
        print(f"    ... {len(counts) - max_rows} more")


def handle_path_repair(args):
    """Check and repair every stale path in a project.

    Phase A repairs the project's own file paths in ``<id>.yaml`` (and the
    path keys of the annotation sidecars, ``<csv>.yaml``); phase B repairs the
    per-camera image paths in ``<id>.meta.json``; phase C repairs the
    ``cam_filepath`` cells of the point CSVs in ``<id>.annos/``. All phases
    are planned in full before any writes, so aborting at any point leaves the
    project untouched.

    Phase A deliberately works on the raw YAML and never constructs a
    :class:`ProjectInitializer`: ``initialize()`` guards the point cloud and the
    cameras on truthiness alone, so a stale path is loaded and crashes with a
    bare ``FileNotFoundError`` before any repair could run.

    Phase B reads and writes :attr:`Camera.orig_filepath` only, never the
    derived :attr:`Camera.filepath`, and freezes the camera selection to a set
    of ids before any mutation, because :attr:`Camera.group` is itself derived
    from the filepath.

    Args:
        args: Parsed command-line arguments.
    """
    from substrata import pathrepair as pr
    from substrata.cameras import Cameras
    from substrata.logging import logger

    find = getattr(args, "find", None)
    replace = getattr(args, "replace", None)
    if replace is not None and find is None:
        raise SystemExit("--replace requires --find.")
    if find is not None and not find:
        raise SystemExit("--find must not be empty.")
    if replace is not None and replace == find:
        raise SystemExit("--find and --replace are identical; nothing to do.")

    do_yaml = not getattr(args, "cams_only", False)
    do_cams = not getattr(args, "yaml_only", False)
    do_anns = not getattr(args, "no_annotations", False)
    interactive = sys.stdin.isatty()
    max_rows = settings.PATHREPAIR_MAX_REPORT_ROWS

    base, cwd = _cwd_base()
    project_dir = os.path.abspath(cwd)
    yaml_path = os.path.join(project_dir, f"{base}.yaml")
    has_yaml = os.path.isfile(yaml_path)

    print(f"Project:   {base}   ({project_dir})")
    # Annotation sets: the point CSVs in <id>.annos/ (id from the YAML)
    project_id = base
    if has_yaml:
        try:
            with open(yaml_path) as f:
                project_id = (yaml.safe_load(f) or {}).get("id") or base
        except (OSError, yaml.YAMLError, AttributeError):
            pass
    ann_csvs = pr.find_annotation_csvs(project_dir, project_id) if do_anns else []

    # One image index (a single walk under --root) for phases B and C
    _index = {}

    def get_index():
        if "index" not in _index:
            root = os.path.abspath(getattr(args, "root", None) or project_dir)
            _index["index"] = pr.build_image_index(root)
        return _index["index"]

    def on_dup(key, old, cands):
        if not interactive:
            raise RuntimeError(
                f"{key!r}: {len(cands)} candidates for {old!r} and no way to ask."
            )
        return _pr_prompt_duplicate("YAML key", key, old, cands)

    def on_miss(key, old):
        if not interactive:
            return ("skip", None)
        return _pr_prompt_missing("YAML key", key, old)

    # ---- Phase A: project files in the YAML -------------------------------
    yplan = None
    if do_yaml and has_yaml:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise SystemExit(f"Project YAML is not a mapping: {yaml_path}")
        try:
            yplan = pr.plan_yaml_repair(yaml_path, config, on_dup, on_miss)
        except pr.PathAborted:
            raise SystemExit("Aborted; nothing written.")
        except RuntimeError as e:
            raise SystemExit(
                f"{e}\n  Re-run in an interactive terminal, or repair the YAML "
                "by hand."
            )
    elif do_yaml:
        print("No project YAML found; project files follow naming conventions.")

    # ---- Phase A2: path keys of the annotation sidecars (<csv>.yaml) ------
    splans = []
    if do_yaml:
        for csv_path in ann_csvs:
            sidecar = sidecars.annotation_sidecar_path(csv_path)
            if not os.path.isfile(sidecar):
                continue
            with open(sidecar, "r") as f:
                sconfig = yaml.safe_load(f) or {}
            if not isinstance(sconfig, dict):
                print(f"Skipping {sidecar}: not a mapping.")
                continue
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            try:
                splan = pr.plan_yaml_repair(
                    sidecar, sconfig, on_dup, on_miss,
                    keys=pr.SIDECAR_YAML_PATH_KEYS, fix_path_key=False,
                    project_id=stem, keep_relative=True,
                )
            except pr.PathAborted:
                raise SystemExit("Aborted; nothing written.")
            except RuntimeError as e:
                raise SystemExit(
                    f"{e}\n  Re-run in an interactive terminal, or repair "
                    f"{os.path.basename(sidecar)} by hand."
                )
            if splan.entries:
                splans.append(splan)

    # ---- Resolve where the cameras live, from the plan (not from disk) ----
    if yplan is not None:
        meta_path = yplan.resolved.get("cams_meta_json")
        xml_path = yplan.resolved.get("cams_xml")
    else:
        meta_path = os.path.join(project_dir, f"{base}.meta.json")
        xml_path = os.path.join(project_dir, f"{base}.cams.xml")
        meta_path = meta_path if os.path.isfile(meta_path) else None
        xml_path = xml_path if os.path.isfile(xml_path) else None

    # ---- Phase B: camera image paths in the meta JSON ---------------------
    cplan = None
    cams = None
    if do_cams:
        if not meta_path or not xml_path:
            print(
                "\nSkipping camera images: the cameras meta JSON / .cams.xml "
                "could not be located."
            )
        else:
            cams = Cameras(meta_path, xml_path)
            if not cams.data:
                print("\nSkipping camera images: no cameras loaded.")
            else:
                cplan = _plan_camera_paths(
                    args, cams, meta_path, project_dir, interactive, pr,
                    get_index=get_index,
                )

    # ---- Phase C: image paths (cam_filepath) in the annotation CSVs -------
    aplan = None
    csv_values = {}
    camera_filters = (
        getattr(args, "cams_group", None) is not None
        or getattr(args, "sensor_id", None) is not None
    )
    if do_cams and ann_csvs and camera_filters:
        print("\nSkipping annotation image paths: camera filters given.")
    elif do_cams and ann_csvs:
        for csv_path in ann_csvs:
            values = set(pr.read_csv_column(csv_path, "cam_filepath"))
            if find is not None:
                values = {v for v in values if find in v}
            if values:
                csv_values[csv_path] = values
        stored = {v: v for values in csv_values.values() for v in values}
        if stored and replace is not None:
            aplan = pr.plan_find_replace(stored, find, replace)
        elif stored:
            try:
                aplan = pr.resolve_image_paths(
                    stored,
                    get_index(),
                    on_duplicate=(
                        (lambda v, old, c: _pr_prompt_duplicate(
                            "Annotation image", pr.path_basename(v), old, c))
                        if interactive else None
                    ),
                    on_missing=(
                        (lambda v, old: _pr_prompt_missing(
                            "Annotation image", pr.path_basename(v), old))
                        if interactive else (lambda v, old: ("skip", None))
                    ),
                    # Folders already mapped for the cameras need no new answer
                    dir_map=cplan.dir_map if cplan is not None else None,
                    on_missing_dir=(
                        (lambda d, n: _pr_prompt_missing_dir(
                            "Annotation image", d, n))
                        if interactive else None
                    ),
                    # ... and folders skipped for the cameras are skipped again
                    skip_dirs=cplan.skipped_dirs if cplan is not None else None,
                )
            except pr.PathAborted:
                raise SystemExit("Aborted; nothing written.")
            except RuntimeError as e:
                raise SystemExit(
                    f"{e}\n  Re-run in an interactive terminal, or use "
                    "--find/--replace to set the paths explicitly."
                )

    # ---- Gates ------------------------------------------------------------
    if cplan is not None:
        if replace is not None and cplan.missing:
            print(f"\n{len(cplan.missing)} resolved image(s) do not exist:")
            for cam_id, path in sorted(cplan.missing.items())[:max_rows]:
                print(f"  {cam_id}  {path}")
            raise SystemExit(
                f"{len(cplan.missing)} resolved image(s) do not exist; "
                "nothing written."
            )
        collisions = cplan.collisions()
        if collisions:
            print(f"\n{len(collisions)} path collision(s):")
            for path, ids in sorted(collisions.items())[:max_rows]:
                print(f"  {path}  <- {', '.join(ids)}")
            raise SystemExit(
                f"Refusing to write: {len(collisions)} path collision(s)."
            )
    # Several stored paths may legitimately name the same image in the CSVs
    # (e.g. different old mounts), so only the --replace gate applies here
    if aplan is not None and replace is not None and aplan.missing:
        print(f"\n{len(aplan.missing)} resolved annotation image(s) do not exist:")
        for old, path in sorted(aplan.missing.items())[:max_rows]:
            print(f"  {old}  ->  {path}")
        raise SystemExit(
            f"{len(aplan.missing)} resolved annotation image(s) do not exist; "
            "nothing written."
        )

    yaml_changes = yplan.changes if yplan is not None else {}
    cam_changes = cplan.changes if cplan is not None else {}
    sidecar_changes = {p.yaml_path: p.changes for p in splans if p.changes}
    ann_changes = aplan.changes if aplan is not None else {}
    n_sidecar_keys = sum(len(c) for c in sidecar_changes.values())
    if not (yaml_changes or cam_changes or sidecar_changes or ann_changes):
        left = [
            (plan, what)
            for plan, what in ((cplan, "camera image"), (aplan, "annotation image"))
            if plan is not None and plan.skipped
        ]
        if not left:
            print("\nNothing to repair; all paths already resolve.")
        for plan, what in left:
            print(
                f"\nNothing to repair; {len(plan.skipped)} {what} path(s) "
                "skipped or not found under the search root."
            )
            _print_skipped_dirs(plan, max_rows)
        return

    # ---- Report -----------------------------------------------------------
    if yplan is not None:
        _print_yaml_report(yplan)
    if cplan is not None and cam_changes:
        _print_cams_report(cplan, max_rows)
        print(f"\nCamera images already valid: {len(cplan.unchanged)}")
        if cplan.skipped:
            print(f"Camera images skipped:       {len(cplan.skipped)}")
        _print_skipped_dirs(cplan, max_rows)
        empty = pr.zero_byte_paths(list(cam_changes.values()))
        if empty:
            print(f"Warning: {len(empty)} matched image(s) are zero bytes.")
    for splan in splans:
        _print_yaml_report(splan, title="Annotation settings")
    if ann_changes:
        _print_cams_report(
            aplan, max_rows, title="Annotation images", unit="images"
        )
        n_csvs = sum(1 for v in csv_values.values() if v & ann_changes.keys())
        print(f"  in {n_csvs} annotation CSV(s)")
        if aplan.skipped:
            print(f"Annotation images not found:  {len(aplan.skipped)}")
        _print_skipped_dirs(aplan, max_rows)

    if getattr(args, "dry_run", False):
        print("\nDry run: nothing written.")
        return

    if not getattr(args, "yes", False):
        if not interactive:
            raise SystemExit(
                "Confirmation required; re-run with --yes in non-interactive use."
            )
        ans = input(
            f"\nProceed? {len(yaml_changes)} YAML key(s), "
            f"{n_sidecar_keys} annotation-settings key(s), "
            f"{len(cam_changes)} camera path(s) and {len(ann_changes)} "
            "annotation image path(s) will be updated. [y/N]: "
        )
        if ans.strip().lower() not in ("y", "yes"):
            raise SystemExit("Aborted.")

    # ---- Write ------------------------------------------------------------
    if yaml_changes:
        with open(yaml_path, "r") as f:
            text = f.read()
        patched = pr.patch_yaml_paths(text, yaml_changes)
        out = pr.atomic_write_text(patched, yaml_path)
        print(f"Updated {len(yaml_changes)} path(s) in {out}")
    if cam_changes:
        cams.set_filepaths(cam_changes)
        n = cams.save_paths(cams_meta_filepath=meta_path,
                            only_cam_ids=set(cam_changes))
        print(f"Updated {n} camera path(s).")
    for sidecar, changes in sidecar_changes.items():
        with open(sidecar, "r") as f:
            text = f.read()
        out = pr.atomic_write_text(pr.patch_yaml_paths(text, changes), sidecar)
        print(f"Updated {len(changes)} path(s) in {out}")
    if ann_changes:
        from substrata.annotations import (
            backup_annotation_csv,
            replace_annotation_csv_values,
        )

        for csv_path, values in csv_values.items():
            mapping = {v: ann_changes[v] for v in values if v in ann_changes}
            if not mapping:
                continue
            backup = backup_annotation_csv(csv_path)
            n = replace_annotation_csv_values(csv_path, "cam_filepath", mapping)
            print(
                f"Updated {n} cam_filepath cell(s) in "
                f"{os.path.basename(csv_path)} (backup: "
                f"{os.path.relpath(backup, project_dir)})"
            )
    logger.info(
        "path-repair: %s YAML key(s), %s sidecar key(s), %s camera path(s), "
        "%s annotation image path(s)",
        len(yaml_changes),
        n_sidecar_keys,
        len(cam_changes),
        len(ann_changes),
    )

    # ---- Post-write verification -----------------------------------------
    if yplan is not None and yplan.unresolved:
        print(f"Warning: {len(yplan.unresolved)} project path(s) still missing:")
        for entry in yplan.unresolved[:max_rows]:
            print(f"  {entry.key}  {entry.old}")
    for splan in splans:
        for entry in splan.unresolved[:max_rows]:
            print(
                f"Warning: {os.path.basename(splan.yaml_path)}: {entry.key} "
                f"still missing: {entry.old}"
            )
    if aplan is not None and (aplan.skipped or aplan.missing):
        n_left = len(aplan.skipped) + len(aplan.missing)
        print(f"Warning: {n_left} annotation image path(s) still missing.")
    if cams is not None:
        still = cams.missing_image_paths(use_orig=True)
        if still:
            print(f"Warning: {len(still)} camera image(s) still missing:")
            for cam_id, path in sorted(still.items())[:max_rows]:
                print(f"  {cam_id}  {path}")
        else:
            print(f"All {len(cams.data)} camera images now resolve on disk.")


def _plan_camera_paths(
    args, cams, meta_path, project_dir, interactive, pr, get_index=None
):
    """Plan the camera-image half of a path repair.

    The camera selection is frozen to a set of ids before anything mutates,
    because :attr:`Camera.group` is derived from the filepath being repaired.

    Args:
        args: Parsed command-line arguments.
        cams: Loaded :class:`Cameras` container.
        meta_path (str): Path to the cameras meta JSON.
        project_dir (str): Absolute project directory.
        interactive (bool): Whether prompting is possible.
        pr: The :mod:`substrata.pathrepair` module.
        get_index (callable, optional): Returns the (shared) image index;
            default builds one under ``--root``.

    Returns:
        PathPlan: The planned camera path changes.
    """
    find = getattr(args, "find", None)
    replace = getattr(args, "replace", None)

    stored = {
        cam_id: cam.orig_filepath
        for cam_id, cam in cams.data.items()
        if cam.orig_filepath
    }

    selected = set(stored)
    filters = []
    if find is not None:
        selected = {cid for cid in selected if find in stored[cid]}
        filters.append(f"--find {find}")
    group = getattr(args, "cams_group", None)
    if group is not None:
        selected = {
            cid
            for cid in selected
            if getattr(cams.data[cid], "group", None) == group
        }
        filters.append(f"--cams-group {group}")
    sensor_id = getattr(args, "sensor_id", None)
    if sensor_id is not None:
        selected = {
            cid
            for cid in selected
            if getattr(cams.data[cid], "sensor_id", None) == sensor_id
        }
        filters.append(f"--sensor-id {sensor_id}")

    if not selected:
        # Built defensively: Cameras.group_names sorts an unfiltered set and
        # raises TypeError when any group is None, which is exactly the case
        # for Windows-authored projects.
        groups = sorted(
            {g for g in (getattr(c, "group", None) for c in cams.data.values()) if g}
        )
        sensors = sorted(
            {
                s
                for s in (getattr(c, "sensor_id", None) for c in cams.data.values())
                if s is not None
            }
        )
        raise SystemExit(
            f"No cameras matched the given filters ({', '.join(filters)}).\n"
            f"  Available groups: {groups if groups else '[]'}\n"
            f"  Available sensor ids: {sensors if sensors else '[]'}"
        )

    sel_paths = {cid: stored[cid] for cid in selected}
    root = os.path.abspath(getattr(args, "root", None) or project_dir)

    if replace is not None:
        return pr.plan_find_replace(sel_paths, find, replace)

    index = get_index() if get_index is not None else pr.build_image_index(root)
    try:
        return pr.resolve_image_paths(
            sel_paths,
            index,
            on_duplicate=(
                (lambda cid, old, c: _pr_prompt_duplicate("Camera", cid, old, c))
                if interactive
                else None
            ),
            on_missing=(
                (lambda cid, old: _pr_prompt_missing("Camera", cid, old))
                if interactive
                else None
            ),
            on_missing_dir=(
                (lambda d, n: _pr_prompt_missing_dir("Camera", d, n))
                if interactive
                else None
            ),
        )
    except pr.PathAborted:
        raise SystemExit("Aborted; nothing written.")
    except RuntimeError as e:
        raise SystemExit(
            f"{e}\n  Re-run in an interactive terminal, or use --find/--replace "
            "to set the paths explicitly."
        )


def _size_arg(values, default):
    """Turn a ``--size`` value (``[N]`` or ``[W, H]``) into ``(w, h)``."""
    if not values:
        return default
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) == 2:
        return tuple(values)
    raise SystemExit("--size takes one value (square) or two values (W H).")


def _classifier_crop_size(values, learn, default):
    """Crop size ``(w, h)``: ``--size``/``--crop-size``, else the classifier's own.

    Classifiers from ``substrata train`` record the crop size they were trained
    on; older ones fall back to ``default`` (with a note).
    """
    from substrata import classification, pointfeatures

    if values:
        return _size_arg(values, default)
    if pointfeatures.is_point_classifier(learn):
        # Point classifiers read the photo around each point themselves; the
        # crop size only affects stored crops / thumbnails and edge margins.
        res = (
            f"{learn.target_mm_px:.3f} mm/px" if learn.target_mm_px
            else f"scale {learn.scale_factor:.3f}"
        )
        cal = ", calibrated" if learn.calibrated else ""
        print(
            f"Point classifier ({learn.backbone}, {res}{cal}); crops of "
            f"{default[0]}x{default[1]} px only for stored crops."
        )
        return default
    stored = classification.classifier_crop_size(learn)
    if stored:
        print(f"Crop size {stored} px (as the classifier was trained)")
        return (stored, stored)
    print(
        f"The classifier records no crop size; using {default[0]}x{default[1]} px "
        "(pass the size it was trained on if that differs)."
    )
    return default


def _load_annotation_project(args, need_annotations=True, use_ply=True):
    """Initialize the project in the CWD for the thumbnails/crops/predict commands.

    Applies ``--annotations``, ``--photo-path-replace`` and ``--local``, skips the
    point cloud unless ``use_ply`` and checks that cameras (and annotations) are
    available.
    """
    from substrata.initializer import ProjectInitializer

    _base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))
    if getattr(args, "annotations", None):
        init.use_annotations_file(args.annotations)
    if not need_annotations:
        init.annotations_filepath = None
    if not use_ply:
        init.ply_filepath = None
    if need_annotations and not init.annotations_filepath:
        raise SystemExit(
"No annotations file given. Provide --annotations or set "
            "`annotations:` in the project YAML (annotation sets live in "
            "<id>.annos/)."
        )
    init.initialize()
    if init.cams is None:
        raise SystemExit("No cameras found (cams_xml / cams_meta_json).")
    if getattr(args, "photo_path_replace", None):
        init.cams.set_filepath_replace(*args.photo_path_replace)
    return init


def _print_status_counts(what, counts, where):
    """Print a one-line summary of crop/thumbnail status counts."""
    summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
    print(f"{what} in {where} - {summary}")


def handle_thumbnails(args):
    """Handle the thumbnails CLI command.

    Writes a downscaled copy of every camera photo to the model's thumbnails
    directory (``thumbnails_path`` or ``<id>.thumbnails``), mirroring the photo
    folders: ``<thumbnails>/<photo folder>/<filename>``.

    Args:
        args: Arguments from argparse.
    """
    init = _load_annotation_project(args, need_annotations=False, use_ply=False)
    root = args.output or init.thumbnails_root
    if root is None:
        raise SystemExit("No thumbnails directory known; pass --output.")
    max_size = _size_arg(args.size, settings.THUMBNAIL_MAX_SIZE)
    counts = init.cams.make_thumbnails(
        root, max_size=max_size, overwrite=args.overwrite, n_jobs=args.jobs
    )
    _print_status_counts("Thumbnails", counts, root)


def handle_crops(args):
    """Handle the crops CLI command.

    Saves crops centred on each annotation's image matches to the annotations'
    crops directory (``cropped_path`` or ``<annotations csv stem>.cropped``):
    ``<crops>/<ann_id>/<photo folder>/<photo stem>_<x>_<y>.jpg``.

    Views are chosen by :meth:`Annotation.select_views`: the main camera type
    first (unobstructed by relevance, then obstructed), plus a few of each other
    camera type. Occlusion is tested on the stratified PLY subsample when the
    plot is stratified, else with the legacy point-cloud test. Each annotation's
    training view (most relevant unobstructed view of the main type) is written
    into the CSV where ``cam_filepath`` is empty (``--no-views``: not at all).

    Args:
        args: Arguments from argparse.
    """
    from substrata import annotations as annotations_mod
    from substrata import cameras as cameras_mod
    from substrata.stratified import ply_occluder

    method = "none" if args.no_ply else args.occlusion
    probe = occluder = None
    if method in ("auto", "subsample"):
        probe = _load_annotation_project(args, use_ply=False)
        occluder = ply_occluder(probe.ply_full_path)
        if occluder is None and method == "subsample":
            raise SystemExit(
                "No stratified full-resolution PLY: run `substrata stratify` "
                "first, or use --occlusion cloud.")
        method = "subsample" if occluder is not None else "cloud"
    init = (probe if probe is not None and method != "cloud"
            else _load_annotation_project(args, use_ply=method == "cloud"))
    if method == "cloud" and init.pcd is None:
        method = "none"
    anns = init.annotations
    root = args.output or anns.crops_root
    crop_size = _size_arg(
        args.size,
        (init.crop_size, init.crop_size) if init.crop_size
        else settings.CROP_SIZE_DEFAULT,
    )
    types = cameras_mod.assign_camera_types(init.cams)
    if args.camera_type == "all":
        main_type = None
    elif args.camera_type:
        if args.camera_type not in types:
            raise SystemExit(f"No camera type {args.camera_type!r}; types: {types}")
        main_type = args.camera_type
    else:
        main_type = anns.main_camera_type(init.cams)
    print(
        f"Occlusion test: {method}. Camera types: {', '.join(types)}; views from "
        f"{main_type or 'all cameras'}.")
    print(f"Saving {crop_size[0]}x{crop_size[1]} crops to {root}")
    counts = anns.save_crops(
        init.cams,
        root=root,
        crop_size=crop_size,
        max_cams=args.max_photos,
        pcd=init.pcd if method == "cloud" else None,
        overwrite=args.overwrite,
        n_jobs=args.jobs,
        occluder=occluder,
        main_type=main_type,
        prune=True,
    )
    _print_status_counts("Crops", counts, root)
    if args.no_views or not init.annotations_filepath:
        return
    views, review = annotations_mod.training_views(anns.data.values())
    csv_path = init.annotations_filepath
    annotations_mod.backup_annotation_csv(csv_path)
    n = annotations_mod.set_training_views(csv_path, views)
    print(
        f"Training views: {n} written into {os.path.basename(csv_path)} (empty "
        f"cam_filepath only; others kept). {len(review)} point(s) have no "
        "unobstructed view of the main camera type and need a look"
        + (f": {', '.join(review[:10])}{' ...' if len(review) > 10 else ''}"
           if review else "") + ".")


def _backup_file(path):
    """Copy ``path`` to ``backup/<name>_<timestamp>.bak`` next to it."""
    import shutil
    from datetime import datetime

    backup_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "backup")
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(backup_dir, f"{os.path.basename(path)}_{timestamp}.bak")
    shutil.copy(path, backup)
    return backup


def handle_import_coralnet(args):
    """Convert CoralNet point exports into substrata annotation CSVs (image-only).

    Each export becomes ``<export stem>_points.csv`` next to it (or ``--output``
    for a single export): ``id``, ``label``, ``cam_filepath`` / ``cam_x`` /
    ``cam_y``, no 3-D position, ``split_key`` = the image and
    ``include_cam_in_training`` = True. That CSV is the annotation set from then
    on (relabel and include/exclude it; train on it without ``--coralnet``); an
    existing one is never overwritten without ``--overwrite``.
    """
    from substrata import classification

    exports = [os.path.abspath(p) for p in args.exports]
    if args.output and len(exports) > 1:
        raise SystemExit("--output needs a single export.")
    for path in exports:
        if not os.path.isfile(path):
            raise SystemExit(f"No such file: {path}")
        if not _is_coralnet_csv(path):
            raise SystemExit(f"{path} is not a CoralNet export (Name, Row, Column).")
    for path in exports:
        out = (os.path.abspath(args.output) if args.output
               else f"{os.path.splitext(path)[0]}_points.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"Skipped {os.path.basename(path)}: {out} exists (--overwrite "
                  "replaces it, losing the edits made to it).")
            continue
        image_dir = args.image_dir or os.path.dirname(path)
        info = classification.convert_coralnet_csv(
            path, classification.index_image_dir(image_dir), out,
            include_column=True)
        print(
            f"{os.path.basename(path)} -> {out}: {info['n_written']} of "
            f"{info['n_points']} points; images {info['n_images_found']}/"
            f"{info['n_images']} found in {info['folder']}"
            + (f" ({info['n_missing']} points on missing images dropped)"
               if info["n_missing"] else ""))


def _is_coralnet_csv(path):
    """Whether a CSV has the CoralNet export columns (Name, Row, Column)."""
    import csv

    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    return {"Name", "Row", "Column"} <= set(header)


def _write_cover_table(anns, out_path):
    """Write per-image percent cover of the predicted labels (plus an ALL row)."""
    import pandas as pd

    df = pd.DataFrame(
        [
            (os.path.basename(ann.cam_filepath), ann.pred_label)
            for ann in anns
            if ann.pred_label is not None
        ],
        columns=["image", "pred_label"],
    )
    if df.empty:
        return None
    cover = pd.crosstab(df["image"], df["pred_label"], normalize="index") * 100
    cover.loc["ALL"] = df["pred_label"].value_counts(normalize=True) * 100
    cover = cover.fillna(0).round(1)
    n_points = df["image"].value_counts()
    cover.insert(0, "n_points", n_points.reindex(cover.index).fillna(len(df)))
    cover["n_points"] = cover["n_points"].astype(int)
    cover.to_csv(out_path)
    return cover


def _print_label_agreement(anns, vocab):
    """Compare predictions with existing labels (e.g. CoralNet annotations)."""
    from sklearn.metrics import classification_report

    pairs = [
        (str(ann.label), ann.pred_label)
        for ann in anns
        if ann.pred_label is not None
        and getattr(ann, "label", None) not in (None, "", "None")
    ]
    if not pairs:
        return
    known = [(y, p) for y, p in pairs if y in vocab]
    n_other = len(pairs) - len(known)
    if n_other:
        other = sorted({y for y, _p in pairs if y not in vocab})
        print(
            f"{n_other} labelled point(s) have labels the classifier does not "
            f"predict ({', '.join(other)}); left out of the comparison."
        )
    if not known:
        return
    y_true, y_pred = zip(*known)
    accuracy = sum(y == p for y, p in known) / len(known)
    print(
        f"\nAgreement with existing labels: {accuracy:.1%} "
        f"({len(known)} point(s))"
    )
    print(classification_report(y_true, y_pred, zero_division=0))


def _predict_images(args):
    """Image-only predict: classify (random) points on plain photos.

    Points come from ``--points`` (a substrata points CSV or a CoralNet
    export) or are sampled on the images in ``--image-dir`` and saved to
    ``<output>/<image dir name>_points.csv``. An existing points file is
    reused (new images get their own points) unless ``--resample``. The
    predictions are written to the points CSV, with a per-image cover table
    next to it.
    """
    from substrata import classification, pointfeatures
    from substrata.annotations import Annotations
    from substrata.cameras import Cameras

    if not args.classifier:
        raise SystemExit("Image-only predict needs --classifier.")
    learn = classification.get_image_classifier(args.classifier)
    crop_size = _classifier_crop_size(
        args.size, learn, settings.PREDICTION_CROP_SIZE_DEFAULT
    )
    margin = max(crop_size) // 2
    output_dir = args.output or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    image_dir = args.image_dir

    def _sample(cams):
        return cams.get_random_image_points(
            args.n_points, margin=margin, stratified=args.stratified,
            seed=args.seed,
        )

    if args.points and args.resample:
        raise SystemExit("--resample only applies to sampled points, not --points.")
    fresh_file = False
    if args.points and _is_coralnet_csv(args.points):
        # CoralNet export: predict at its annotated points (labels kept)
        image_dir = image_dir or os.path.dirname(os.path.abspath(args.points))
        stem = os.path.splitext(os.path.basename(args.points))[0]
        points_path = os.path.join(output_dir, f"{stem}_points.csv")
        if os.path.isfile(points_path):
            print(f"Backup of previous points file: {_backup_file(points_path)}")
        fresh_file = True  # (re)converted below; no backup needed when saving
        info = classification.convert_coralnet_csv(
            args.points, classification.index_image_dir(image_dir), points_path
        )
        print(
            f"CoralNet points: {info['n_written']}/{info['n_points']} on "
            f"{info['n_images_found']}/{info['n_images']} images found in "
            f"{info['folder'] or image_dir}"
        )
        anns = Annotations(points_path)
        Cameras.from_annotation_images(anns)
    elif args.points:
        points_path = args.points
        anns = Annotations(points_path)
        Cameras.from_annotation_images(anns)
        print(f"Loaded {len(anns)} points from {points_path}")
    else:
        if not image_dir:
            raise SystemExit("Pass --image-dir (or --points).")
        name = os.path.basename(os.path.normpath(image_dir))
        points_path = os.path.join(output_dir, f"{name}_points.csv")
        all_cams = Cameras.from_image_dir(image_dir, recursive=True)
        if not all_cams.data:
            raise SystemExit(f"No images found in {image_dir}")
        if os.path.isfile(points_path) and not args.resample:
            anns = Annotations(points_path)
            used = {os.path.abspath(a.cam_filepath) for a in anns if a.cam_filepath}
            Cameras.from_annotation_images(anns)
            print(f"Reusing {len(anns)} points from {points_path}")
            new_cams = Cameras()
            new_cams.data = {
                cam_id: cam for cam_id, cam in all_cams.data.items()
                if cam.filepath not in used
            }
            if new_cams.data:
                print(f"Sampling points on {len(new_cams.data)} new image(s)")
                for ann in _sample(new_cams):
                    anns.append(ann)
        else:
            if os.path.isfile(points_path):
                backup = _backup_file(points_path)
                print(f"Resampling; old points backed up to {backup}")
            anns = _sample(all_cams)

    model = classification.classifier_fingerprint(args.classifier)  # (pred_model)
    if not pointfeatures.is_point_classifier(learn) and (args.gsd_mm or args.domain):
        raise SystemExit("--gsd-mm and --domain apply to point classifiers (.pt).")
    try:
        results = anns.classify_image_matches(
            learn, crop_size=crop_size, batch_size=args.batch_size, n_jobs=args.jobs,
            gsd_mm=args.gsd_mm, domain=args.domain,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    for ann_id, res in results.items():
        if res is not None:
            anns[ann_id].pred_label = res["label"]
            anns[ann_id].pred_conf = round(res["confidence"], 4)
            anns[ann_id].pred_n = 1
            anns[ann_id].pred_model = model
    predicted = sum(1 for res in results.values() if res is not None)
    print(f"Predicted {predicted} of {len(results)} points.")

    if os.path.isfile(points_path) and not fresh_file:
        print(f"Backup of previous points file: {_backup_file(points_path)}")
    anns.save(points_path, orig_coords_only=True)
    print(f"Saved points and predictions to {points_path}")
    cover_path = os.path.splitext(points_path)[0] + "_cover.csv"
    if _write_cover_table(anns, cover_path) is not None:
        print(f"Saved per-image cover (% of points) to {cover_path}")
    vocab = learn.vocab if pointfeatures.is_point_classifier(learn) else learn.dls.vocab
    _print_label_agreement(anns, {str(v) for v in vocab})


def handle_predict(args):
    """Handle the predict CLI command.

    Classifies every image match of every annotation (from the centre of the
    stored crop where it is large enough, otherwise from the source photo),
    combines the per-crop predictions per annotation, writes the consensus to
    the ``pred_label``/``pred_conf``/``pred_n`` columns of the annotations CSV
    (after backing it up to ``backup/``) and each crop's own prediction to its
    EXIF.

    Args:
        args: Arguments from argparse.
    """
    from substrata import classification

    if args.image_dir or args.points:
        return _predict_images(args)

    init = _load_annotation_project(args, use_ply=not args.no_ply)
    classifier_path = args.classifier or init.classifier_filepath
    if not classifier_path:
        raise SystemExit("No classifier: pass --classifier or set it in the YAML.")
    anns = init.annotations
    root = args.crops or anns.crops_root

    learn = classification.get_image_classifier(classifier_path)
    crop_size = _classifier_crop_size(
        args.size, learn, settings.PREDICTION_CROP_SIZE_DEFAULT
    )
    # which classifier: stored with each prediction (pred_model); the points it
    # already predicted are skipped unless --redo
    model = classification.classifier_fingerprint(classifier_path)
    ids = [a.id for a in anns.data.values()
           if args.redo or getattr(a, "pred_model", None) != model]
    skipped = len(anns.data) - len(ids)
    print(f"Classifier {model}"
          + (f"; {skipped} annotation(s) already predicted by it are skipped "
             "(--redo: all)" if skipped else ""))
    consensus = anns.predict(
        init.cams,
        learn,
        crop_size=crop_size,
        max_cams=args.max_photos,
        pcd=init.pcd,
        root=root,
        ids=ids,
        write_crops=not args.no_write_crops,
        method=args.method,
        batch_size=args.batch_size,
        n_jobs=args.jobs,
        model=model,
    )
    predicted = sum(1 for c in consensus.values() if c)
    print(f"Predicted {predicted} of {len(consensus)} annotations ({args.method}).")
    if not consensus:
        return

    # only the prediction columns change: every other column (include_cam_in_training, intercept fields,
    # match links, ...) stays as it is
    from substrata.annotations import set_predictions

    csv_path = init.annotations_filepath
    backup = _backup_file(csv_path)
    set_predictions(csv_path, {
        i: (c["label"], c["confidence"], c["n"], model, anns.data[i].pred_agree) if c else None
        for i, c in consensus.items()})
    print(f"Saved predictions to {csv_path} (backup: {backup})")


def _add_annotation_project_args(parser, annotations=True):
    """Arguments shared by the thumbnails/crops/predict commands."""
    if annotations:
        parser.add_argument(
            "--annotations", dest="annotations", type=str, default=None,
            help="Explicit annotations CSV (overrides initializer).",
        )
    parser.add_argument(
        "--photo-path-replace", dest="photo_path_replace", nargs=2,
        default=None, metavar=("OLD", "NEW"),
        help="Remap camera photo paths (substring OLD -> NEW).",
    )
    parser.add_argument(
        "--jobs", dest="jobs", type=int, default=-1,
        help="Parallel threads (-1 = all cores, 1 = sequential). Default: -1.",
    )
    parser.add_argument(
        "--local", dest="local", action="store_true",
        help="Reset all paths to local (relative to project path).",
    )


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
        init.use_annotations_file(args.annotations)

    # Resolve annotations
    if not init.annotations_filepath:
        raise SystemExit(
"No annotations file given. Provide --annotations or set "
            "`annotations:` in the project YAML (annotation sets live in "
            "<id>.annos/)."
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


def handle_match_annotations(args):
    """Handle the match-annotations CLI command.

    Matches random point annotations to manual (colony) annotations of the
    same object using 3-D proximity + SAM2 masks, writes a copy of the random
    annotation CSV with ``manual_id`` and ``manual_id_conf`` columns (its
    sidecar's ``matched_to`` names the manual CSV), and a PDF to review
    flagged matches.

    The matched CSV can be passed back as ``--random``: rows that already have
    a ``manual_id`` or ``manual_id_conf`` (from a previous run, or added by
    hand) are then kept as they are, and only the remaining rows are matched,
    unless ``--redo-all`` is given. With ``--export-labels`` nothing is
    matched: the matched CSV is written with the manual set's current labels
    joined in.

    Args:
        args: Arguments from argparse.
    """
    from substrata.initializer import ProjectInitializer
    from substrata import segmentation, visualizations
    from substrata.annotations import (
        get_matched_annotation_ids,
        save_annotation_matches_csv,
        save_joined_manual_annotations,
    )

    random_sidecar = sidecars.read_annotation_sidecar(args.random)
    if args.export_labels:
        counts = save_joined_manual_annotations(
            args.random, args.export_labels, manual_filepath=args.manual
        )
        print(f"Joined manual labels saved to {args.export_labels}")
        print("Rows: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if counts.get("missing"):
            print(
                f"Warning: {counts['missing']} manual_id(s) no longer exist in the "
                "manual set"
            )
        return

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))
    manual_path = (
        args.manual or random_sidecar.get("matched_to") or init.annotations_filepath
    )
    if not manual_path:
        raise SystemExit(
            "No manual annotations file given. Provide --manual or set "
            "`annotations:` in the project YAML."
        )
    for path in (args.random, manual_path):
        if not os.path.isfile(path):
            raise SystemExit(f"Annotations file not found: {path}")

    # Only cameras are needed: skip the (large) point cloud and let the two
    # annotation files be loaded explicitly below.
    init.ply_filepath = None
    init.annotations_filepath = None
    init.initialize()
    if not init.cams or len(init.cams) == 0:
        raise SystemExit("No cameras available for annotation matching")
    if getattr(args, "photo_path_replace", None):
        init.cams.set_filepath_replace(*args.photo_path_replace)

    random_anns = Annotations(args.random, orig_coords_only=True)
    manual_anns = Annotations(manual_path, orig_coords_only=True)
    for anns in (random_anns, manual_anns):
        anns.transform_coords(init.world_transform)
    print(f"Random annotations: {len(random_anns)} ({args.random})")
    print(f"Manual annotations: {len(manual_anns)} ({manual_path})")
    skip_ids = set() if args.redo_all else get_matched_annotation_ids(args.random)
    if skip_ids:
        print(
            f"Skipping {len(skip_ids)} already matched random points "
            "(use --redo-all to match all)"
        )

    predictor = segmentation.get_sam2_predictor(sam2_checkpoint=args.sam2_checkpoint)
    records = random_anns.match_annotations(
        manual_anns,
        init.cams,
        sam_predictor=predictor,
        label_prefix=[p.strip() for p in args.label_prefix.split(",")],
        max_dist=args.max_dist,
        max_candidates=args.max_candidates,
        fallback_dist=args.fallback_dist,
        skip_ids=skip_ids,
    )

    stem = os.path.splitext(args.random)[0]
    output = args.output or f"{stem}_matched.csv"
    save_annotation_matches_csv(args.random, records, output)
    # The matched copy keeps the random set's sidecar and names the manual set
    sidecar = {k: v for k, v in random_sidecar.items() if k != "matched_to"}
    sidecars.write_annotation_sidecar(
        output, {**sidecar, "matched_to": os.path.abspath(manual_path)}
    )
    print(f"Matched annotations saved to {output} (manual set: {manual_path})")

    counts = {}
    for rec in records:
        if rec["considered"]:
            key = rec["manual_id_conf"] or "unmatched"
            counts[key] = counts.get(key, 0) + 1
    print("Match confidence: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"Flagged for review: {sum(rec['flagged'] for rec in records)}")

    if not args.no_pdf:
        pdf_output = args.pdf_output or f"{stem}_matches.pdf"
        visualizations.save_annotation_matches_to_pdf(
            records, pdf_output, review_all=args.review_all
        )
        print(f"Review PDF saved to {pdf_output}")


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

    # Transform from a project's parent link, or prompt for transform(s)
    link = args.to_parent or args.from_parent
    if args.to_parent and args.from_parent:
        raise SystemExit("transform: use either --to-parent or --from-parent")
    if link:
        init = (
            ProjectInitializer(path=link)
            if os.path.isdir(link)
            else ProjectInitializer(yaml=link)
        )
        if init.parent_transform is None:
            raise SystemExit(f"transform: {init.id} has no parent_transform")
        transform = np.asarray(init.parent_transform, dtype=float)
        if args.from_parent:
            transform = np.linalg.inv(transform)
        print(
            f"Using parent_transform of {init.id} "
            f"({'to' if args.to_parent else 'from'} parent {init.parent_id})"
        )
    else:
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


def _project_yamls(target: str) -> list[str]:
    """Project YAMLs at ``target``: a YAML, a project dir, or a folder of them.

    A folder is searched up to two levels down (e.g. a site folder holding
    ``<plot>/<timepoint>/<timepoint>.yaml``).
    """
    if os.path.isfile(target):
        return [os.path.abspath(target)]
    dirs, level = [os.path.abspath(target)], [os.path.abspath(target)]
    for _ in range(2):
        level = [
            e.path for d in level for e in sorted(os.scandir(d), key=lambda e: e.name)
            if e.is_dir() and not e.name.startswith(".")
        ]
        dirs += level
    yamls = [os.path.join(d, f"{os.path.basename(d)}.yaml") for d in dirs]
    return [y for y in yamls if os.path.isfile(y)]


def _frame_change(W_old: np.ndarray, W_new: np.ndarray) -> str:
    """Describe the world-frame change ``W_new @ inv(W_old)`` in a few words."""
    D = W_new @ np.linalg.inv(W_old)
    s = float(np.cbrt(np.linalg.det(D[:3, :3])))
    cos = (np.trace(D[:3, :3] / s) - 1) / 2
    angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return (
        f"rotation {angle:.3f} deg, origin shift {np.linalg.norm(D[:3, 3]):.4f}, "
        f"scale x{s:.5f}"
    )


def handle_baseline(args):
    """Check (and refresh) world_transforms inherited from a baseline timepoint.

    A project with ``baseline_id``/``baseline_transform`` caches the baseline's
    frame in its ``world_transform``. This reports per project whether that
    cache still matches the baseline's current ``world_transform``; with
    ``--update`` stale caches (and the scale factor and depth model carried
    over with them) are rewritten in place, baselines before their dependants.
    ``--link`` instead adds the link to YAMLs whose ``world_transform`` was
    built from the baseline before links existed.

    Args:
        args: Parsed CLI arguments.
    """
    from substrata.initializer import set_project_yaml_keys

    targets = args.projects or [os.getcwd()]
    yamls = list(dict.fromkeys(y for t in targets for y in _project_yamls(t)))
    if not yamls:
        raise SystemExit(f"baseline: no project YAMLs found in {', '.join(targets)}")
    projects = {}
    for y in yamls:
        init = ProjectInitializer(yaml=y)
        projects.setdefault(init.id, init)
    search = [os.path.dirname(os.path.dirname(y)) for y in yamls]

    def _load(ref, near):
        if ref in projects:
            return projects[ref]
        path = ref if os.path.exists(ref) else near.find_project(ref, search)
        if not path:
            return None
        init = (
            ProjectInitializer(path=path)
            if os.path.isdir(path)
            else ProjectInitializer(yaml=path)
        )
        return projects.setdefault(init.id, init)

    def _write(init, values):
        if args.update or args.link:
            set_project_yaml_keys(init.yaml_path, values)
            print(f"  updated {init.yaml_path}")

    if args.link:
        # Only the projects named, never every plot found in a folder
        link_yamls = [
            t if os.path.isfile(t)
            else os.path.join(t, f"{os.path.basename(os.path.abspath(t))}.yaml")
            for t in targets
        ]
        for y in link_yamls:
            if not os.path.isfile(y):
                raise SystemExit(f"baseline --link: {y} is not a project YAML")
        print()
        for y in link_yamls:
            init = ProjectInitializer(yaml=y)
            baseline = _load(args.link, init)
            if baseline is None:
                raise SystemExit(f"baseline --link: {args.link} not found")
            if baseline.id == init.id:
                continue
            init.baseline_id = baseline.id
            init.baseline_transform = np.linalg.inv(
                np.asarray(baseline.world_transform, dtype=float)
            ) @ np.asarray(init.world_transform, dtype=float)
            print(f"{init.id}: linked to baseline {baseline.id}")
            _write(
                init,
                {"baseline_id": baseline.id,
                 "baseline_transform": init.baseline_transform,
                 **init.baseline_world_keys(baseline)},
            )
        return

    done, visiting, counts = set(), set(), {}

    def _resolve(init):
        """Bring init's cached frame up to date (in memory), baselines first."""
        if init.id in done or not init.baseline_id:
            return
        if init.id in visiting:
            raise SystemExit(f"baseline: cycle in baseline links at {init.id}")
        visiting.add(init.id)
        status = None
        if init.baseline_transform is None:
            status = "no baseline_transform"
        else:
            baseline = _load(init.baseline_id, init)
            if baseline is None:
                status = f"baseline {init.baseline_id} not found"
        if status:
            print(f"{init.id}: {status}")
            status = "unresolved"
        else:
            _resolve(baseline)
            if init.world_transform_is_stale(baseline):
                change = _frame_change(
                    init.world_transform, init.world_transform_from_baseline(baseline)
                )
                status = "stale"
                print(f"{init.id}: stale vs baseline {baseline.id} ({change})")
                keys = init.baseline_world_keys(baseline)
                init.world_transform = keys["world_transform"]
                for k in ("scale_factor", "up_vector", "depth_offset",
                          "depth_per_unit"):
                    if k in keys:
                        setattr(init, k, keys[k])
                if init.yaml_path in yamls:
                    _write(init, keys)
                    status = "updated" if args.update else status
                else:
                    print(f"  not written: {init.yaml_path} is outside {targets}")
            else:
                status = "fresh"
                print(f"{init.id}: fresh (baseline {baseline.id})")
        counts[status] = counts.get(status, 0) + 1
        visiting.discard(init.id)
        done.add(init.id)

    print()
    for init in list(projects.values()):
        _resolve(init)
    if not counts:
        print("No projects with a baseline_id")
        return
    print("\n" + ", ".join(f"{n} {k}" for k, n in counts.items()))
    if counts.get("stale"):
        print("Run with --update to refresh the stale world_transforms.")
    if counts.get("updated"):
        print(
            "The world_* columns of these plots' annotation CSVs (and their "
            "sidecar world_transform) still follow the old frame; orig_* are "
            "unaffected."
        )


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
      0. With --coralnet, convert CoralNet point exports (image-only) into
         standard annotation CSVs, linking each to its images in --image-dir.
      1. Glob annotation CSVs from --csv-path and render the CATAMI label tree
         (or, without a classes file, a flat list of label counts); the bolded
         entries are the training labels (confirm with the user).
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

    from substrata import classification, pointfeatures

    cwd = os.getcwd()
    csv_path = args.csv_path or cwd
    model_path = args.model_path or cwd
    output_dir = args.output or cwd
    classes_path = args.classes or os.path.join(cwd, settings.TRAIN_CLASSES_FILE)
    coralnet = bool(getattr(args, "coralnet", False))
    pattern = args.pattern or (
        settings.CORALNET_DEFAULT_PATTERN if coralnet
        else settings.TRAIN_DEFAULT_PATTERN
    )
    crop_size = args.crop_size or settings.TRAIN_CROP_SIZE
    # --arch dinov2 / dinov3 (short-hands or torch.hub names; default DINOv2):
    # point-feature classifier; else fastai crops.
    args.arch = pointfeatures.resolve_arch(args.arch)
    point_mode = pointfeatures.is_point_arch(args.arch)
    model_file = args.model or os.path.join(
        output_dir,
        settings.POINT_DEFAULT_MODEL_FILE if point_mode
        else settings.TRAIN_DEFAULT_MODEL_FILE,
    )
    if point_mode and getattr(args, "pretrained_weights", None):
        raise SystemExit(
            "--pretrained-weights applies to the fastai crop classifier; use it "
            "with a torchvision --arch (e.g. resnet50)."
        )
    map_path = getattr(args, "label_map", None) or os.path.join(
        output_dir, settings.TRAIN_LABEL_MAP_FILE
    )
    interactive = sys.stdin.isatty()
    assume_yes = bool(getattr(args, "yes", False))

    def _stats_pdf(split_folder):
        return os.path.join(output_dir, f"{split_folder}_stats.pdf")

    def _crops_pdf(label_map):
        """--pdf: the crops per training class, captioned with their ids."""
        from substrata import visualizations

        ann_csv = os.path.join(output_dir, settings.TRAIN_ANNOTATIONS_FILE)
        if not os.path.isfile(ann_csv):
            raise SystemExit(f"--pdf: training annotations not found: {ann_csv}")
        groups = classification.training_crops_by_class(
            ann_csv, output_dir, label_map, max_per_class=args.pdf_max
        )
        pdf_path = os.path.join(output_dir, settings.TRAIN_CROPS_PDF_FILE)
        n_pages = visualizations.save_training_crops_pdf(
            groups, pdf_path,
            crop_px=classification.stored_crop_size(output_dir, set()),
        )
        print(
            f"Wrote training crops PDF: {pdf_path} ({len(groups)} class(es), "
            f"{n_pages} page(s))"
        )

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
        if pointfeatures.is_point_classifier_path(model_file):
            # Point classifiers carry their label map; points come from the
            # training annotations CSV (same split as training).
            split_name = "test" if getattr(args, "test", False) else "validation"
            ann_csv = os.path.join(output_dir, settings.TRAIN_ANNOTATIONS_FILE)
            if not os.path.isfile(ann_csv):
                raise SystemExit(f"Training annotations not found: {ann_csv}")
            clf = pointfeatures.PointFeatureClassifier.load(model_file)
            try:
                result = pointfeatures.evaluate_point_classifier(
                    clf, ann_csv, output_dir, split_name,
                    batch_size=args.batch_size, n_jobs=args.jobs,
                )
            except ValueError as e:
                raise SystemExit(str(e))
            pointfeatures.report_point_evaluation(
                result, split_name,
                pdf_path=os.path.join(output_dir, f"{split_name}_points_stats.pdf"),
            )
            return
        if not os.path.isfile(map_path):
            raise SystemExit(
                f"Label map not found for {flag}: {map_path}. Run "
                "`substrata train` (or --prepare-only) first to create it."
            )
        label_map = classification.load_label_map(map_path)
        if getattr(args, "pdf", False):
            _crops_pdf(label_map)
        classification.report_classifier_stats(
            model_file, output_dir, split, label_map, pdf_path=_stats_pdf(split)
        )
        return

    # The classes file (CATAMI hierarchy) is optional unless given explicitly
    use_tree = os.path.isfile(classes_path)
    if args.classes and not use_tree:
        raise SystemExit(f"Classes file not found: {classes_path}")
    if not use_tree:
        if args.tips_only or getattr(args, "collapse", False):
            raise SystemExit(
                "--tips_only/--collapse need a classes file (--classes)."
            )
        print(
            f"No classes file ({classes_path}); selecting labels from a flat "
            "list of counts."
        )

    own_outputs = {
        os.path.basename(classes_path),
        settings.TRAIN_ANNOTATIONS_FILE,
        settings.TRAIN_LABEL_MAP_FILE,
    }
    if getattr(args, "sets", None):  # an explicit list (e.g. from substrata-explore), any folders
        with open(args.sets) as f:
            csv_files = [os.path.abspath(ln.strip()) for ln in f if ln.strip() and not ln.startswith("#")]
        missing = [p for p in csv_files if not os.path.isfile(p)]
        if missing:
            raise SystemExit(f"Annotation CSV(s) of {args.sets} not found: {', '.join(missing[:5])}")
    else:
        csv_files = sorted(
            f
            for f in glob.glob(os.path.join(csv_path, pattern))
            if os.path.basename(f) not in own_outputs
        )
    if not csv_files:
        raise SystemExit(
            f"No annotation CSVs match {pattern!r} in {csv_path}."
        )

    # the annotation files read (substrata-curate can edit their labels)
    from substrata import curation as _cur
    _cur.write_training_sources(
        output_dir, csv_files, "coralnet" if coralnet else "annotations")

    # --- Step 0 (--coralnet): convert point exports into standard CSVs ---
    if coralnet:
        image_dir = args.image_dir or csv_path
        points_dir = os.path.join(output_dir, settings.CORALNET_POINTS_DIR)
        os.makedirs(points_dir, exist_ok=True)
        image_index = classification.index_image_dir(image_dir)
        print(f"Converting {len(csv_files)} CoralNet export(s); images: {image_dir}")
        converted = []
        for path in csv_files:
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(points_dir, f"{stem}_points.csv")
            try:
                info = classification.convert_coralnet_csv(
                    path, image_index, out_path
                )
            except ValueError as e:
                raise SystemExit(str(e))
            folder = (
                os.path.relpath(info["folder"], image_dir)
                if info["folder"] else "(no images found)"
            )
            missing = (
                f", {info['n_missing']} on missing images dropped"
                if info["n_missing"] else ""
            )
            print(
                f"  {os.path.basename(path)} -> {folder} "
                f"({info['n_images_found']}/{info['n_images']} images, "
                f"{info['n_written']} points{missing})"
            )
            converted.append(out_path)
        csv_files, pattern = converted, "*_points.csv"

    # --- Curation decisions (`substrata curate`) and pool points ---
    curation_store = None
    if not getattr(args, "no_curation", False):
        from substrata import curation as cur

        try:
            curation_store = cur.store_for_folder(
                output_dir, getattr(args, "curation", None) or [])
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        if len(curation_store):
            done = curation_store.counts()
            what = ", ".join(f"{n} {a}" for a, n in sorted(done.items()))
            extra = "".join(f" + {p}" for p in curation_store.extra)
            print(f"Curation: {len(curation_store)} decision(s) ({what}) from "
                  f"{curation_store.path}{extra}")
            pools = cur.pool_csvs(output_dir)
            if pools and done.get("relabel"):
                csv_files = csv_files + pools
                print(f"  + {len(pools)} pool(s): their points labelled in "
                      "curation join the training")
        else:
            curation_store = None

    # --- Step 1: label tree (or flat label counts) + seed collapse map ---
    include_classes = getattr(args, "include_classes", None)
    include_set = set(include_classes) if include_classes else None
    exclude_set = set(getattr(args, "exclude_classes", None) or ())
    if use_tree:
        lines, training_labels, counts, _unknown, collapse_map = (
            classification.build_label_tree(
                classes_path, csv_files, args.min_count, args.tips_only,
                include_labels=include_set,
                collapse=bool(getattr(args, "collapse", False)),
                curation=curation_store,
            )
        )
    else:
        lines, training_labels, counts, _unknown, collapse_map = (
            classification.build_flat_labels(
                csv_files, args.min_count, include_labels=include_set,
                exclude_labels=exclude_set, curation=curation_store,
            )
        )
    print("\n".join(lines))
    if exclude_set:
        for label in exclude_set:
            counts.pop(label, None)
            collapse_map.pop(label, None)
        training_labels -= exclude_set
        collapse_map = {
            lab: cls for lab, cls in collapse_map.items() if cls not in exclude_set
        }
        print(f"\nExcluded: {', '.join(sorted(exclude_set))}")
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
        f"classes; {'points are kept' if point_mode else 'crops are generated'} "
        f"for all {len(visible_labels)} visible label(s)."
    )
    if not assume_yes:
        if not interactive:
            raise SystemExit(
                "Confirmation required; re-run with --yes in non-interactive use."
            )
        what = "collate the points" if point_mode else "generate crops"
        ans = input(f"Proceed ({what} and seed the label map)? [y/N]: ")
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
    curation_applied = Counter()
    skipped = Counter()
    n_written, n_dropped, n_dropped_conf = (
        classification.collate_training_annotations(
            csv_files, pattern, visible_labels, ann_path, model_path,
            prompt=(interactive and not assume_yes),
            min_conf=min_conf, conf_strict=conf_is_strict,
            curation=curation_store, curation_stats=curation_applied,
            skipped=skipped,
        )
    )
    if skipped["not_in_training"]:
        print(
            f"Left out {skipped['not_in_training']} annotation(s) with "
            f"{settings.TRAIN_INCLUDE_COLUMN} = False."
        )
    if curation_store is not None:
        print(
            "Curation applied: "
            f"{curation_applied['exclude']} excluded, "
            f"{curation_applied['relabel']} relabelled, "
            f"{curation_applied['accept']} accepted."
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
    # The point-feature classifier reads the photos directly; crops are only
    # made for it when --pdf asks for the contact sheet.
    if not point_mode or getattr(args, "pdf", False):
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

    if getattr(args, "pdf", False):
        _crops_pdf(label_map)

    # --prepare-only: stop here so the map can be hand-tuned before training.
    if getattr(args, "prepare_only", False):
        print(
            f"\nPrepared {'training points' if point_mode else 'crops'} and "
            f"label map. Edit {map_path} to tune the "
            "selection/collapsing, then re-run `substrata train --keep-map` to "
            "train on the edited map (plain `substrata train` re-seeds it)."
        )
        return

    selection_info = [
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
        (
            "Selection: exclude_classes",
            " ".join(sorted(exclude_set)) if exclude_set else "(none)",
        ),
        ("Classes file", classes_path if use_tree else "(none; flat labels)"),
        ("Input format", "CoralNet" if coralnet else "substrata"),
        ("Map mode", "keep-map" if getattr(args, "keep_map", False) else "reseed"),
        (
            "Curation",
            "(none)" if curation_store is None else _counts_text(
                curation_applied, "no decision applied"),
        ),
        ("Annotation pattern", pattern),
        ("CSV files scanned", len(csv_files)),
        ("Training classes", len(training_classes)),
        ("Label map", map_path),
        ("Model output", model_file),
    ]

    # --- Step 5 (point features): features, probe, evaluation ---
    if point_mode:
        try:
            clf, run = pointfeatures.train_point_classifier(
                ann_path, label_map, output_dir, model_file, backbone=args.arch,
                feature_mm_px=args.feature_mm_px, feature_scale=args.feature_scale,
                batch_size=args.batch_size, n_jobs=args.jobs,
            )
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(str(e))
        print(f"Saved point classifier: {model_file}")
        if run["n_failed"]:
            print(f"Skipped {run['n_failed']} point(s) on unreadable photos.")
        metrics = {
            f"{split} {k}": round(run["evaluation"][split][k], 4)
            for split in run["evaluation"]
            for k in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        info = [
            ("Classifier", "point features (linear probe on frozen backbone)"),
            ("Backbone", args.arch),
            ("Resolution", run["plan"].description),
            ("Window (model px)", clf.window),
            ("Probe C", clf.meta.get("probe_C")),
            ("Calibration patches", clf.meta.get("n_calibration_patches")),
            ("Train/val/test split (%)", "/".join(map(str, settings.TRAIN_SPLIT))),
        ] + selection_info
        summary_pdf = os.path.join(output_dir, settings.TRAIN_SUMMARY_FILE)
        classification.write_training_summary_pdf(
            summary_pdf, info, run["counts"], crop_dirs=pointfeatures.SPLIT_NAMES,
            metrics=metrics,
        )
        print(f"Wrote training summary: {summary_pdf}")
        if "validation" in run["evaluation"]:
            pointfeatures.report_point_evaluation(
                run["evaluation"]["validation"], "validation",
                pdf_path=os.path.join(output_dir, "validation_points_stats.pdf"),
            )
        return

    # --- Step 5: train ---
    learn = classification.train_classifier(
        output_dir, model_file, label_map, arch=args.arch, epochs=args.epochs,
        pretrained_weights=args.pretrained_weights, crop_size=crop_size,
    )

    # --- Step 5b: training-run summary PDF (settings + per-class counts) ---
    counts_by_split = classification.count_crops_by_class(output_dir, label_map)
    info = [
        ("Model architecture", args.arch),
        ("Pretrained weights", args.pretrained_weights or "ImageNet (torchvision)"),
        ("Epochs", args.epochs),
        ("Crop size (px)", crop_size),
        ("Input size (px)", settings.TRAIN_IMAGE_SIZE),
        ("Train/val/test split (%)", "/".join(map(str, settings.TRAIN_SPLIT))),
    ] + selection_info
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


def handle_calibrate(args):
    """Handle 'calibrate': add another camera system's feature statistics.

    Samples random patches of the given photos (which must show the same area
    as the classifier's training photos), computes their backbone feature
    statistics and writes a copy of the classifier that standardises with them.
    """
    from substrata import pointfeatures

    clf_path = args.classifier or os.path.join(
        os.getcwd(), settings.POINT_DEFAULT_MODEL_FILE
    )
    if not pointfeatures.is_point_classifier_path(clf_path):
        raise SystemExit(f"Not a point classifier (.pt): {clf_path}")
    try:
        clf = pointfeatures.PointFeatureClassifier.load(clf_path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    images = pointfeatures.list_images(args.images)
    if not images:
        raise SystemExit("No photos found for calibration.")
    print(f"Calibrating {os.path.basename(clf_path)} on {len(images)} photo(s)")
    try:
        cal = pointfeatures.calibrate_point_classifier(
            clf, images, gsd_mm=args.gsd_mm, n_images=args.n_images,
            per_image=args.per_image, batch_size=args.batch_size, n_jobs=args.jobs,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    out = args.output or os.path.splitext(clf_path)[0] + "_calibrated.pt"
    cal.meta["calibration"]["images"] = (
        args.images if len(args.images) <= 5 else args.images[:5] + ["..."]
    )
    cal.save(out)
    shift = float(np.mean(np.abs(cal.target_mean - clf.source_mean) / clf.source_std))
    print(
        f"Wrote {out} ({cal.meta['calibration']['n_patches']} patches; mean "
        f"feature shift {shift:.2f} SD from the training photos)."
    )


def _counts_text(counts, empty=""):
    """``"3 accept, 2 exclude"`` from a Counter (``empty`` without counts)."""
    return ", ".join(f"{n} {a}" for a, n in sorted(counts.items())) or empty


def handle_curate_gui(argv):
    """Run the curation grid (substrata-explore's ``curate`` mode).

    Args:
        argv: Its command-line arguments (after ``curate``, without ``--gui``).
    """
    try:
        from substrata_explore.curate import main as curate_main
    except ImportError:
        raise SystemExit(
            "substrata curate --gui needs the substrata-explore package: "
            "pip install -e <path to substrata-explore>. Without it, "
            "`substrata curate --sheets` writes contact sheets and "
            "`--import CSV` reads decisions."
        )
    curate_main(argv)


def _curate_sheets(folder, max_per_sheet):
    """Contact sheets of the review queues (PDF), from curate/points.csv."""
    import tempfile
    from io import BytesIO

    import numpy as np
    from PIL import Image, ImageDraw

    from substrata import curation as cur
    from substrata import visualizations

    out_dir = cur.curation_dir(folder)
    table = os.path.join(out_dir, settings.CURATION_POINTS_FILE)
    if not os.path.isfile(table):
        raise SystemExit(f"{table} not found: run `substrata curate` first.")
    rows = [r for r in cur.read_table(table) if r.get("tile")]
    store = cur.store_for_folder(folder)

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    open_rows = [r for r in rows if store.get(r["key"]) is None and r["cls"]]
    queues = {
        "suspect labels (lowest cross-validated probability of the label)":
            sorted([r for r in open_rows if np.isfinite(f(r["p_label"]))],
                   key=lambda r: f(r["p_label"])),
        "mixed neighbourhoods (lowest share of the class around the point)":
            sorted([r for r in open_rows if np.isfinite(f(r["ring_agree"]))],
                   key=lambda r: f(r["ring_agree"])),
        "unusual points (most novel)":
            sorted([r for r in open_rows if np.isfinite(f(r["novelty"]))],
                   key=lambda r: -f(r["novelty"])),
    }
    tiles = cur.tile_store(folder)
    pdf_path = os.path.join(out_dir, "review_sheets.pdf")
    with tempfile.TemporaryDirectory() as tmp:
        groups = {}
        for title, items in queues.items():
            out = []
            for r in items[:max_per_sheet]:
                dst = os.path.join(tmp, r["tile"])
                tile = tiles.get(r["tile"])
                if tile is None:
                    continue
                if not os.path.isfile(dst):
                    try:
                        with Image.open(BytesIO(tile[0])) as im:
                            im = im.convert("RGB")
                            d = ImageDraw.Draw(im)
                            x, y = f(r["tile_x"]), f(r["tile_y"])
                            if np.isfinite(x) and np.isfinite(y):
                                for c, w in (((0, 0, 0), 3), ((255, 60, 60), 1)):
                                    d.line([(x - 9, y), (x - 3, y)], fill=c, width=w)
                                    d.line([(x + 3, y), (x + 9, y)], fill=c, width=w)
                                    d.line([(x, y - 9), (x, y - 3)], fill=c, width=w)
                                    d.line([(x, y + 3), (x, y + 9)], fill=c, width=w)
                            im.save(dst, quality=90)
                    except OSError:
                        continue
                caption = f"{r['label']}>{r['cv_pred'] or '?'} {f(r['p_label']):.2f}"
                out.append((dst, r["id"], caption, ""))
            groups[title] = (len(items), out)
        n_pages = visualizations.save_training_crops_pdf(groups, pdf_path)
    print(f"Wrote {pdf_path} ({n_pages} page(s)). Captions: id, label>predicted "
          "class and the cross-validated probability of the label. Record "
          "decisions in a CSV (id, action, label, reason) and read it with "
          "`substrata curate --import CSV`.")


def handle_curate(args):
    """Handle 'curate': score training points, mine pools, manage decisions.

    Runs in a classifier folder (``substrata train`` output). Without options
    the training points are scored for review (``curate/points.csv`` + tiles);
    ``--mine`` samples and clusters unlabelled photos into a pool;
    ``--status``, ``--export``, ``--import`` and ``--sheets`` work on the
    decisions; ``--gui`` opens the review grid (substrata-explore).
    """
    from substrata import curation as cur
    from substrata import pointfeatures

    folder = os.path.abspath(args.folder or os.getcwd())
    try:
        store = cur.store_for_folder(folder, args.curation or [])
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    def lookup():
        out = {}
        table = os.path.join(cur.curation_dir(folder), settings.CURATION_POINTS_FILE)
        for path in ([table] if os.path.isfile(table) else []) + cur.pool_csvs(folder):
            for r in cur.read_table(path):
                if r.get("id") and r.get("key"):
                    out[r["id"]] = r
        return out

    if args.status or args.export or args.import_csv:
        if args.import_csv:
            n, problems = cur.import_decisions(store, args.import_csv, lookup())
            for p in problems[:20]:
                print(f"  skipped {p}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")
            print(f"Imported {n} decision(s) into {store.path}")
        if args.export:
            n = store.export_csv(args.export)
            print(f"Exported {n} decision(s) to {args.export}")
        done = store.counts()
        what = ", ".join(f"{n} {a}" for a, n in sorted(done.items()))
        print(f"{store.path}: {len(store)} decision(s)" + (f": {what}" if done else ""))
        reasons = Counter(e.get("reason") or "(none)" for e in store.entries()
                          if e["action"] == "exclude")
        if reasons:
            print("  excluded by reason: " + ", ".join(
                f"{r} {n}" for r, n in reasons.most_common()))
        return
    if args.sheets:
        return _curate_sheets(folder, args.sheets_max)

    clf_path = args.model or os.path.join(folder, settings.POINT_DEFAULT_MODEL_FILE)
    if not pointfeatures.is_point_classifier_path(clf_path):
        raise SystemExit(f"curate needs a point classifier (.pt): {clf_path}")
    try:
        clf = pointfeatures.PointFeatureClassifier.load(clf_path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{e} (run `substrata train` in this folder first).")

    if args.mine or args.image_dir or args.points:
        if not args.name:
            raise SystemExit("A pool needs a --name (e.g. auv_deep).")
        points = info = None
        if args.points:
            if args.mine:
                raise SystemExit("Pass --points or --mine, not both.")
            if not os.path.isfile(args.points):
                raise SystemExit(f"No such points CSV: {args.points}")
            points, info = cur.read_pool_points(args.points, args.image_dir)
            if info.get("coralnet"):
                print(f"CoralNet points: {info['n_points']} on "
                      f"{info['n_images_found']}/{info['n_images']} images found in "
                      f"{info['image_dir']}")
            print(f"Pool points: {info['n_used']} of {info['n_rows']} from "
                  f"{args.points}" + (f" ({info['n_skipped']} without a readable "
                                       "photo or position)" if info["n_skipped"] else ""))
            if not points:
                raise SystemExit("No usable points (are the photos readable? "
                                 "--image-dir for a CoralNet export's photos).")
            images = sorted({q["path"] for q in points})
        else:
            images = pointfeatures.list_images(
                list(args.mine or []) + ([args.image_dir] if args.image_dir else []))
            if not images:
                raise SystemExit("No photos found for the pool.")
        try:
            meta = cur.mine_pool(
                folder, clf, images, args.name, gsd_mm=args.gsd_mm,
                domain=args.domain, n_images=args.mine_images,
                per_image=args.per_image, n_clusters=args.clusters,
                seed=args.seed, resample=args.resample,
                batch_size=args.batch_size, n_jobs=args.jobs,
                points=points, points_info=info,
            )
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(str(e))
        print(f"Pool {meta['name']} (domain {meta['domain']}): {meta['n_points']} "
              f"point(s) in {meta['n_clusters']} cluster(s) -> {meta['csv']}")

        def nov(c):
            return "" if c["novelty"] is None else f" nov {c['novelty']:.2f}"

        print("  largest clusters: " + "; ".join(
            f"#{c['cluster']} n={c['n']} {c['pred']} {c['pred_share']:.0%}{nov(c)}"
            for c in meta["clusters"][:8]))
        nov = [c for c in meta["clusters"] if c["novelty"] is not None]
        if nov:
            top = sorted(nov, key=lambda c: -c["novelty"])[:5]
            print("  most novel clusters: " + "; ".join(
                f"#{c['cluster']} n={c['n']} nov {c['novelty']:.2f} (pred "
                f"{c['pred']})" for c in top))
        if meta.get("n_given_labels"):
            print(f"  {meta['n_given_labels']} point(s) carry a given label: accept "
                  "(or change) it in the review grid to use it.")
        print("Label clusters with `substrata curate --gui`; labelled pool points "
              "join the next `substrata train` in this folder.")
        return

    try:
        summary = cur.score_training_points(
            folder, clf, args.annotations, batch_size=args.batch_size,
            n_jobs=args.jobs,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))
    cv = summary["cv"]
    print(f"Scored {summary['n_points']} training point(s) -> {summary['table']}")
    if cv:
        print(f"  cross-validated ({cv['folds']} folds by photo): accuracy "
              f"{cv['accuracy']:.3f}, macro-F1 {cv['macro_f1']:.3f}")
    print(f"  look clean: {summary['n_clean']} (label probability >= "
          f"{settings.CURATION_CLEAN_P} and neighbourhood >= "
          f"{settings.CURATION_CLEAN_RING})")
    print(f"  suspect labels (probability < {settings.CURATION_SUSPECT_P}): "
          f"{summary['n_suspect']}; mixed neighbourhoods (< "
          f"{settings.CURATION_MIXED_RING}): {summary['n_mixed']}")
    done = store.counts()
    what = ", ".join(f"{n} {a}" for a, n in sorted(done.items()))
    print(f"  decisions so far: {len(store)}" + (f" ({what})" if done else ""))
    print("Review with `substrata curate --gui` (or `--sheets` for PDFs); then "
          "re-run `substrata train` here to apply the decisions.")


def handle_masks(args):
    """Handle 'masks': dense per-photo class masks for plain photos.

    Every photo gets a label mask (palette PNG) and a confidence mask at
    feature-cell resolution from a point classifier; ``masks.json`` indexes
    them (classes, colours, cell size and cover per photo) and
    ``masks_cover.csv`` lists the class fractions per photo. Photos that
    already have masks are skipped (resumable), unless ``--overwrite``.
    """
    import csv

    from substrata import pointfeatures, segmentation

    clf_path = args.classifier or os.path.join(
        os.getcwd(), settings.POINT_DEFAULT_MODEL_FILE
    )
    if not pointfeatures.is_point_classifier_path(clf_path):
        raise SystemExit(f"masks needs a point classifier (.pt): {clf_path}")
    try:
        clf = pointfeatures.PointFeatureClassifier.load(clf_path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    images = pointfeatures.list_images(args.images)
    if not images:
        raise SystemExit("No photos found.")
    if clf.target_mm_px and not args.gsd_mm:
        print(
            f"Note: the classifier works at {clf.target_mm_px:.3f} mm/px but no "
            f"--gsd-mm was given: photos of its training camera use its fixed "
            f"scale {clf.scale_factor:.3f}, others are scaled by their width."
        )
    # ground resolution first; else the training camera's fixed scale, or the
    # width rule for another camera (as predict and curate pools)
    widths = ({} if clf.target_mm_px and args.gsd_mm
              else pointfeatures._image_widths(images))
    photo_scales = [pointfeatures.photo_scale(clf, args.gsd_mm, widths.get(p))
                    for p in images]
    scale = float(np.median(photo_scales))
    out_dir = args.output or os.path.join(os.getcwd(), settings.MASKS_DEFAULT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    overrides = {}
    for spec in getattr(args, "color", None) or []:
        if "=" not in spec:
            raise SystemExit(f"Invalid --color {spec!r}; expected LABEL=RRGGBB.")
        lbl, hexv = spec.split("=", 1)
        overrides[lbl] = _parse_hex_color(hexv)
    colors = segmentation._build_label_colors(list(clf.classes), overrides or None)

    index_path = os.path.join(out_dir, settings.MASKS_INDEX_FILE)
    index = {"classes": list(clf.classes), "photos": {}}
    if os.path.isfile(index_path) and not args.overwrite:
        with open(index_path) as f:
            old = json.load(f)
        if old.get("classes") == list(clf.classes):
            index["photos"] = old.get("photos", {})
    index.update({
        "classifier": os.path.abspath(clf_path),
        "colors": {c: list(v) for c, v in colors.items()},
        "gsd_mm": args.gsd_mm, "scale": scale,
    })
    stems = pointfeatures.unique_stems(images)

    def _done(i):
        label_png = os.path.join(out_dir, f"{stems[i]}_labels.png")
        return images[i] in index["photos"] and os.path.isfile(label_png)

    todo = [i for i in range(len(images)) if args.overwrite or not _done(i)]
    lo, hi = min(photo_scales), max(photo_scales)
    span = f"{lo:.3f}" if hi - lo < 1e-4 else f"{lo:.3f}-{hi:.3f}"
    print(f"Masks for {len(todo)} photo(s) ({len(images) - len(todo)} already "
          f"done) at scale {span} -> {out_dir}")

    def write_index():
        with open(index_path, "w") as f:
            json.dump(index, f, indent=1)

    n_done = 0
    for k, dm in pointfeatures.iter_dense_maps(
        clf, [images[i] for i in todo], [photo_scales[i] for i in todo], n_jobs=args.jobs,
    ):
        i = todo[k]
        if dm is None:
            continue
        files = pointfeatures.save_dense_masks(
            dm, out_dir, stems[i], colors, save_probs=args.probs,
        )
        index["photos"][images[i]] = dict(
            files, cell_px=round(dm.cell_px, 4), scale=round(dm.scale, 5),
            width=dm.width, height=dm.height,
            mean_conf=round(float(dm.confidence().mean()), 4),
            cover={c: round(v, 5) for c, v in dm.cover().items()},
        )
        n_done += 1
        if n_done % 200 == 0:
            write_index()
    write_index()

    cover_path = os.path.join(out_dir, settings.MASKS_COVER_FILE)
    with open(cover_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["photo", "width", "height", "cell_px", "mean_conf"]
        w.writerow(head + list(clf.classes))
        for photo, info in sorted(index["photos"].items()):
            cov = info.get("cover", {})
            meta = [photo] + [info.get(k) for k in head[1:]]
            w.writerow(meta + [cov.get(c, 0.0) for c in clf.classes])
    print(f"Wrote {n_done} new mask pair(s); index {index_path}; cover {cover_path}")


def main():
    """Build the argparse CLI and dispatch to the selected subcommand handler.

    Defines all subcommands (decimate, ply-repair, head, scalebars, views, orient,
    colors, firefish, cams2video, intercepts, intercepts-plot, align, images,
    match-annotations, camsync, path-repair, transform, train, calibrate,
    masks, curate, metashape-export),
    parses the arguments, and calls the matching handle_* function.
    """
    # explore: the interactive viewer (the substrata-explore package), its own arguments
    if len(sys.argv) > 1 and sys.argv[1] == "explore":
        return handle_explore(sys.argv[2:])
    # curate --gui: the review grid (substrata-explore), its own arguments
    if len(sys.argv) > 2 and sys.argv[1] == "curate" and "--gui" in sys.argv[2:]:
        return handle_curate_gui([a for a in sys.argv[2:] if a != "--gui"])

    parser = argparse.ArgumentParser(description="Substrata CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "explore",
        help="Open the interactive viewer for the project in the CWD (needs the "
        "substrata-explore package; `substrata explore -h` for its options).",
    )

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
            "PLY (from `substrata colors`). The output's header is marked, so "
            "loading it does not apply the correction a second time."
        ),
    )

    # stratify (reorder a PLY for progressive / regional reads)
    p_strat = subparsers.add_parser(
        "stratify",
        help=(
            "Stratify a PLY in place (<id>.ply + <id>.strat.npz): its first N points "
            "are then a uniform subsample and regions can be read without a full "
            "scan. With no args, uses initializer on CWD."
        ),
    )
    p_strat.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_strat.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default=None,
        help="Write a stratified copy (same file name) here instead of in place.",
    )
    p_strat.add_argument(
        "--tmp-dir",
        dest="tmp_dir",
        type=str,
        default=None,
        help="Stream mode: folder for the temporary file (as large as the PLY; "
        "default: the work dir, else next to the output).",
    )
    p_strat.add_argument(
        "--mode",
        choices=["auto", "memory", "stream"],
        default="auto",
        help="memory: read the whole PLY into RAM (records + 8 B per point; one "
        "sequential read and write, fastest by far on hard disks); stream: three "
        "passes through a temp file, little RAM; auto (default): memory if it fits "
        "in the RAM available.",
    )
    p_strat.add_argument(
        "--work-dir",
        dest="work_dir",
        type=str,
        default=None,
        help="Write the new PLY (and the temp file) on this disk first, e.g. an SSD "
        "when the PLY is on a hard disk, then copy it next to the PLY "
        "(default: $SUBSTRATA_WORK_DIR).",
    )
    p_strat.add_argument(
        "--cluster-size",
        dest="cluster_size",
        type=int,
        default=2_000_000,
        help="Target points per cluster (default: 2,000,000).",
    )
    p_strat.add_argument(
        "--keep-original",
        dest="keep_original",
        action="store_true",
        help="In place: keep the unstratified PLY as backup_<name>.ply next to it.",
    )
    p_strat.add_argument(
        "--force",
        action="store_true",
        help="Stratify again even if the PLY already is.",
    )
    p_strat.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
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
        "ply-repair",
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
        "head",
        help="Show first N vertex rows from a PLY file (and, for a stratified PLY, "
        "its index summary; for a plot with a baseline_id, whether its "
        "world_transform still follows the baseline).",
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
        "--use_chunk_transform",
        "--use-chunk-transform",
        dest="use_chunk_transform",
        action="store_true",
        help=(
            "Instead of fitting scale/orientation, apply the Metashape "
            "chunk_transform from the cameras meta JSON (converted to a local "
            "East-North-Up frame in metres) and derive the scale factor from it. "
            "The cloud is then rotated about Z so +Y points upslope."
        ),
    )
    p_orient.add_argument(
        "--north_up",
        "--north-up",
        dest="north_up",
        action="store_true",
        help=(
            "With --use_chunk_transform: keep the georeferenced orientation "
            "(+X east, +Y north) instead of rotating +Y upslope."
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
        "--slope",
        dest="slope",
        action="store_true",
        help=(
            "After orienting, align the cloud along its slope (as "
            "`intercepts --slope`) and save that in the world_transform."
        ),
    )
    p_orient.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # segment (point-cloud segmentation via image-match classification)
    p_seg = subparsers.add_parser(
        "segment",
        help=(
            "Segment the point cloud by classifying camera patches at sampled "
            "query points; saves the points as <id>.annos/<id>_seg_points.csv "
            "(the source of truth) and a recoloured cloud."
        ),
    )
    p_seg.add_argument(
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_seg.add_argument(
        "--classifier", dest="classifier", type=str, default=None,
        help="Classifier .pkl (overrides the project YAML 'classifier').",
    )
    p_seg.add_argument(
        "--cell-size", dest="cell_size", type=float, default=None,
        help="Query spacing / voxel size in metres (default 0.05).",
    )
    p_seg.add_argument(
        "--sampling", dest="sampling", choices=["voxel", "xy_grid"], default="voxel",
        help="Query sampling: 'voxel' (3D, default) or 'xy_grid' (top-down).",
    )
    p_seg.add_argument(
        "--crop-size", dest="crop_size", type=int, default=None, metavar="PX",
        help="Patch size in px. Default: the size the classifier was trained on "
             f"(recorded by `substrata train`), else {settings.TRAIN_CROP_SIZE}.",
    )
    p_seg.add_argument(
        "--no-occlusion", dest="no_occlusion", action="store_true",
        help="Skip reprojection occlusion filtering (faster, less accurate).",
    )
    p_seg.add_argument(
        "--views", dest="views", type=int, default=None, metavar="N",
        help="Point classifiers: photos fused per query point (its N best "
             "views by relevance, each classified from that photo's dense map; "
             f"weighted by relevance). Default: {settings.SEG_POINT_VIEWS}.",
    )
    p_seg.add_argument(
        "--save-masks", dest="save_masks", nargs="?", const="", default=None,
        metavar="DIR",
        help="Point classifiers: also save each photo's label/confidence masks "
             "(feature-cell resolution) with a masks.json index. Default folder: "
             "<id>_seg[_<tag>]_masks in the plot folder.",
    )
    p_seg.add_argument(
        "--per-point", dest="per_point", action="store_true",
        help="Point classifiers: classify each query from a window around it "
             "in its best photo instead of dense per-photo maps (no fusion; "
             "slower for dense query grids).",
    )
    p_seg.add_argument(
        "--max-radius", dest="max_radius", type=float, default=None,
        help="Leave points farther than this (m) from any query unlabeled.",
    )
    p_seg.add_argument(
        "--color", dest="color", action="append", default=None,
        metavar="LABEL=RRGGBB",
        help="Manual category colour, e.g. --color CS=ff0000 (repeatable).",
    )
    p_seg.add_argument(
        "--photo-path-replace", dest="photo_path_replace", nargs=2,
        default=None, metavar=("OLD", "NEW"),
        help="Remap camera photo paths (substring OLD -> NEW) when the poses "
             "reference a different mount than where the photos live locally.",
    )
    p_seg.add_argument(
        "--full-ply", dest="full_ply", nargs="?", const="", default=None,
        help="Also stream-recolour the full-size PLY (optional explicit path; "
             "defaults to the project's full PLY).",
    )
    p_seg.add_argument(
        "--output-npz", dest="output_npz", type=str, default=None,
        help="Also save the segmentation as a compact .npz at this path (the "
             "points CSV is the source of truth; no .npz is written by default).",
    )
    p_seg.add_argument(
        "--output-ply", dest="output_ply", type=str, default=None,
        help="Recoloured decimated PLY output path "
             "(default <id>_seg[_<tag>]_dec.ply).",
    )
    p_seg.add_argument(
        "--tag", dest="tag", type=str, default=None, metavar="NAME",
        help="Version name: saves <id>.annos/<id>_seg_<NAME>_points.csv and "
             "<id>_seg_<NAME>[_dec].ply, so several segmentations can coexist.",
    )
    p_seg.add_argument(
        "--from-points", dest="from_points", type=str, default=None,
        metavar="CSV",
        help="Recolour from an existing segmentation points CSV (e.g. after "
             "checking labels by hand: 'label' wins over 'pred_label') instead "
             "of classifying again.",
    )
    p_seg.add_argument(
        "--local", dest="local", action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # colors (ColorChecker calibration QC + optional YAML colour correction)
    p_colors = subparsers.add_parser(
        "colors",
        help=(
            "Run ColorCalibrations from project markers, save a QC PDF and store "
            "the affine colour correction in the project YAML (no PLY is written)."
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
        default=True,
        help=(
            "Write the affine color_correction (matrix + offset) into the project "
            "YAML (the default; kept for compatibility)."
        ),
    )
    p_colors.add_argument(
        "--no-save-yaml",
        "--no_save_yaml",
        dest="save_yaml",
        action="store_false",
        help="Only write the QC PDF; leave the project YAML unchanged.",
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
        "--beam-half-angle",
        dest="beam_half_angle",
        type=float,
        default=settings.FIREFISH_BEAM_HALF_ANGLE,
        help=(
            "Half-angle (degrees) of the cone around the camera view vector used for "
            "camera-derived altitude (camdist). Default: %(default)s (Ping2 sonar, "
            "25 deg full beamwidth)."
        ),
    )
    p_ff.add_argument(
        "--distance-percentile",
        dest="distance_percentile",
        type=float,
        default=settings.FIREFISH_DISTANCE_PERCENTILE,
        help=(
            "Percentile of in-beam distances used as camdist (sonar first-return "
            "proxy). Use -1 for the mean. Default: %(default)s."
        ),
    )
    p_ff.add_argument(
        "--sonar-offset",
        dest="sonar_offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=list(settings.FIREFISH_SONAR_OFFSET_M),
        help=(
            "Sonar position relative to the camera centre in the camera frame "
            "(x: image right, y: image down, z: view direction), in metres. "
            "Default: 0 0 0."
        ),
    )
    p_ff.add_argument(
        "--depth-sensor-offset",
        dest="depth_sensor_offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=list(settings.FIREFISH_DEPTH_SENSOR_OFFSET_M),
        help=(
            "FireFish depth sensor position relative to the camera centre in the "
            "camera frame (x: image right, y: image down, z: view direction), in "
            "metres. Depths are regressed against the sensor positions (lever-arm "
            "correction), e.g. 0 0 -0.15 for a sensor 15 cm behind the camera. "
            "Default: 0 0 0."
        ),
    )
    p_ff.add_argument(
        "--water-level",
        dest="water_level",
        type=float,
        default=0.0,
        help=(
            "Water level relative to mean sea level during the survey (m; e.g. the "
            "tide from a nearby gauge, negative below MSL). FireFish depths are "
            "converted to depth below MSL (depth + water level). Default: 0."
        ),
    )
    p_ff.add_argument(
        "--recompute-camdists",
        dest="recompute_camdists",
        action="store_true",
        help=(
            "Recompute camdist even if the camdepths CSV already exists (datetimes "
            "are kept from the CSV)."
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
            "Register a source (PLY or project YAML) to a target and print the "
            "transform; optionally write an aligned source project YAML."
        ),
    )
    p_align.add_argument(
        "--source",
        dest="source",
        type=str,
        required=True,
        help="Source PLY or project YAML",
    )
    p_align.add_argument(
        "--target",
        dest="target",
        type=str,
        required=True,
        help="Target PLY or project YAML",
    )
    p_align.add_argument(
        "--points",
        dest="points",
        type=int,
        default=5_000_000,
        help="Max points to stream-load",
    )
    p_align.add_argument(
        "--cameras",
        action="store_true",
        help=(
            "Align on centres of cameras with matching image names (any group); "
            "requires project YAMLs"
        ),
    )
    p_align.add_argument(
        "--max-residual",
        dest="max_residual",
        type=float,
        default=None,
        help=(
            "With --cameras: reject cameras whose residual exceeds this (metres "
            "if the target has a scale_factor, else target units); default: "
            "automatic median/MAD threshold"
        ),
    )
    p_align.add_argument(
        "--refine",
        action="store_true",
        help="With --cameras: refine the camera-based fit with point-cloud ICP",
    )
    p_align.add_argument(
        "--no-scaling",
        dest="no_scaling",
        action="store_true",
        help="Estimate a rigid transform only (no uniform scale)",
    )
    p_align.add_argument(
        "--voxel-size",
        dest="voxel_size",
        type=float,
        default=None,
        help="Registration voxel size (default: 2%% of target extent)",
    )
    p_align.add_argument(
        "--pdf",
        dest="pdf",
        type=str,
        default=None,
        help=(
            "Write a QC PDF: summary, camera residuals (--cameras), location of "
            "the source within the target, and point-cloud correspondence"
        ),
    )
    p_align.add_argument(
        "--output-yaml",
        dest="output_yaml",
        type=str,
        default=None,
        help=(
            "Write a copy of the source project YAML with world_transform set "
            "to the raw source → oriented target transform (with --as-parent "
            "or --as-baseline: a copy with the link added instead)"
        ),
    )
    p_align.add_argument(
        "--as-parent",
        dest="as_parent",
        action="store_true",
        help=(
            "Keep the source world_transform and store the target as its parent "
            "instead: parent_id and parent_transform (raw source → raw target) "
            "are written to the source YAML (or to --output-yaml)"
        ),
    )
    p_align.add_argument(
        "--as-baseline",
        dest="as_baseline",
        action="store_true",
        help=(
            "The target is an earlier timepoint whose frame the source inherits: "
            "writes baseline_id, baseline_transform (raw source → raw target) "
            "and world_transform = target world_transform @ baseline_transform "
            "(plus scale factor and depth model) to the source YAML (or to "
            "--output-yaml). See `substrata baseline` to refresh it later"
        ),
    )
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
            "Optional YAML holding the world_transform (orientation) used to "
            "generate the intercepts. Default: the CSV's sidecar <csv>.yaml, else "
            "the project's world_transform."
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
        "--animation-order",
        dest="animation_order",
        choices=("columns", "rows", "scan", "random", "spiral", "categories"),
        default="columns",
        help=(
            "Order in which the animated GIF fills its cells (default: "
            "columns). 'rows' = top-to-bottom raster; 'scan' = columns with a "
            "moving scan line; 'random' = cells pop in in random order; "
            "'spiral' = expand outward from the centre; 'categories' = one "
            "label group at a time, most dominant first (~1s per category)."
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

    # thumbnails
    p_thumbs = subparsers.add_parser(
        "thumbnails",
        help="Write a downscaled copy of every camera photo (model thumbnails).",
    )
    p_thumbs.add_argument(
        "--size", dest="size", type=int, nargs="+", default=None,
        metavar="PX",
        help="Bounding box W H (or one value for a square); aspect ratio is kept. "
             f"Default: {settings.THUMBNAIL_MAX_SIZE[0]} "
             f"{settings.THUMBNAIL_MAX_SIZE[1]}.",
    )
    p_thumbs.add_argument(
        "--output", dest="output", type=str, default=None,
        help="Thumbnails directory. Default: thumbnails_path or <id>.thumbnails.",
    )
    p_thumbs.add_argument(
        "--overwrite", dest="overwrite", action="store_true",
        help="Regenerate existing thumbnails.",
    )
    _add_annotation_project_args(p_thumbs, annotations=False)

    # crops
    p_crops = subparsers.add_parser(
        "crops",
        help="Save crops centred on each annotation's image matches.",
    )
    p_crops.add_argument(
        "--size", dest="size", type=int, nargs="+", default=None,
        metavar="PX",
        help="Crop size W H (or one value for a square). Default: YAML crop_size "
             f"or {settings.CROP_SIZE_DEFAULT[0]} {settings.CROP_SIZE_DEFAULT[1]}.",
    )
    p_crops.add_argument(
        "--max-photos", dest="max_photos", type=int,
        default=settings.MAX_PHOTOS_TO_CROP_DEFAULT,
        help="Views of the main camera type cropped per annotation (the training "
             "view first; other camera types are not cropped: review tools cut "
             f"them on the fly). Default: {settings.MAX_PHOTOS_TO_CROP_DEFAULT}.",
    )
    p_crops.add_argument(
        "--output", dest="output", type=str, default=None,
        help="Crops directory. Default: cropped_path or <annotations stem>.cropped.",
    )
    p_crops.add_argument(
        "--overwrite", dest="overwrite", action="store_true",
        help="Regenerate existing crops.",
    )
    p_crops.add_argument(
        "--no-ply", dest="no_ply", action="store_true",
        help="No occlusion test (same as --occlusion none).",
    )
    p_crops.add_argument(
        "--occlusion", dest="occlusion", default="auto",
        choices=["auto", "subsample", "cloud", "none"],
        help="Occlusion test for choosing views: 'subsample' tests the line of "
             "sight on the stratified full-resolution PLY (needs `substrata "
             "stratify`); 'cloud' is the legacy reprojection test on the "
             "YAML's point cloud (loaded into memory); 'auto' (default): "
             "subsample if the plot is stratified, else cloud.",
    )
    p_crops.add_argument(
        "--camera-type", dest="camera_type", default=None, metavar="TYPE",
        help="Main camera type to choose views from (as printed, e.g. "
             "'9568x6376 f6460'). Default: the type that sees most points best; "
             "'all' uses every camera alike (legacy).",
    )
    p_crops.add_argument(
        "--no-views", dest="no_views", action="store_true",
        help="Do not write training views into the annotations CSV.",
    )
    _add_annotation_project_args(p_crops)

    # import-coralnet
    p_icn = subparsers.add_parser(
        "import-coralnet",
        help="Convert CoralNet point exports into substrata annotation CSVs "
             "(image-only), to edit and train on from then on.",
    )
    p_icn.add_argument("exports", nargs="+", help="CoralNet export CSV(s).")
    p_icn.add_argument(
        "--image-dir", dest="image_dir", default=None,
        help="Directory holding the images, directly or in subfolders (searched "
             "recursively; each export is linked to the subfolder holding most "
             "of its images). Default: the export's folder.",
    )
    p_icn.add_argument(
        "--output", dest="output", default=None,
        help="Output CSV (single export only). Default: <export stem>_points.csv "
             "next to the export.",
    )
    p_icn.add_argument(
        "--overwrite", dest="overwrite", action="store_true",
        help="Replace an existing output CSV (its edits are lost).",
    )

    # predict
    p_predict = subparsers.add_parser(
        "predict",
        help="Predict annotation labels from their crops with a trained classifier.",
    )
    p_predict.add_argument(
        "--classifier", dest="classifier", type=str, default=None,
        help="fastai classifier (.pkl). Default: classifier in the YAML.",
    )
    p_predict.add_argument(
        "--size", dest="size", type=int, nargs="+", default=None,
        metavar="PX",
        help="Classifier input crop size W H (or one value). Default: "
             f"{settings.PREDICTION_CROP_SIZE_DEFAULT[0]} "
             f"{settings.PREDICTION_CROP_SIZE_DEFAULT[1]}.",
    )
    p_predict.add_argument(
        "--max-photos", dest="max_photos", type=int,
        default=settings.MAX_PHOTOS_TO_PREDICT_DEFAULT,
        help="Maximum image matches used per annotation. "
             f"Default: {settings.MAX_PHOTOS_TO_PREDICT_DEFAULT}.",
    )
    p_predict.add_argument(
        "--method", dest="method", choices=["weighted", "first"],
        default=settings.PREDICTION_AGGREGATION,
        help="Combine crops by relevance-weighted mean probability, or use the "
             f"top-ranked crop only. Default: {settings.PREDICTION_AGGREGATION}.",
    )
    p_predict.add_argument(
        "--crops", dest="crops", type=str, default=None,
        help="Crops directory. Default: cropped_path or <annotations stem>.cropped.",
    )
    p_predict.add_argument(
        "--no-write-crops", dest="no_write_crops", action="store_true",
        help="Do not store each crop's own prediction in its EXIF.",
    )
    p_predict.add_argument(
        "--redo", dest="redo", action="store_true",
        help="Predict every annotation again, also those whose stored prediction "
             "(pred_model) is already from this classifier.",
    )
    p_predict.add_argument(
        "--batch-size", dest="batch_size", type=int, default=64,
        help="Inference batch size. Default: 64.",
    )
    p_predict.add_argument(
        "--no-ply", dest="no_ply", action="store_true",
        help="Skip loading the point cloud (faster, but no occlusion filtering).",
    )
    img_group = p_predict.add_argument_group(
        "image-only mode (plain photos, no project)",
        "Classify points on plain images instead of a 3-D project's "
        "annotations. Needs --classifier; --max-photos/--method/--crops/--no-ply "
        "do not apply.",
    )
    img_group.add_argument(
        "--image-dir", dest="image_dir", type=str, default=None,
        help="Folder of images (searched recursively) to sample points on. "
             "Points are saved to <output>/<folder name>_points.csv and reused "
             "on later runs (new images get their own points). With a "
             "CoralNet --points file: where its images are (default: the "
             "CSV's folder).",
    )
    img_group.add_argument(
        "--points", dest="points", type=str, default=None,
        help="Predict at these points instead of sampling: a substrata points "
             "CSV (updated in place) or a CoralNet export (Name, Row, Column, "
             "Label code; predictions are compared with its labels).",
    )
    img_group.add_argument(
        "--n-points", dest="n_points", type=int,
        default=settings.IMAGE_POINTS_DEFAULT,
        help=f"Points per image. Default: {settings.IMAGE_POINTS_DEFAULT}.",
    )
    img_group.add_argument(
        "--stratified", dest="stratified", action="store_true",
        help="Stratified random points (one per grid cell) instead of simple "
             "random points.",
    )
    img_group.add_argument(
        "--seed", dest="seed", type=int, default=None,
        help="Random seed for reproducible points.",
    )
    img_group.add_argument(
        "--gsd-mm", dest="gsd_mm", type=float, default=None, metavar="MM",
        help="Point classifiers: ground resolution of the photos (mm/px), used "
        "first by classifiers with a physical resolution. Without it, photos "
        "of another camera (width more than 25%% off the training photos') are "
        "resampled so the window covers the same share of their width, as "
        "`substrata curate` pools are.",
    )
    img_group.add_argument(
        "--domain", dest="domain", default=None, metavar="NAME",
        help="Point classifiers: standardise the features with this camera "
        "domain's statistics (a `substrata curate` pool whose labelled points "
        "were trained on; default: the training photos').",
    )
    img_group.add_argument(
        "--resample", dest="resample", action="store_true",
        help="Draw new points even if the points file exists (old one is "
             "backed up).",
    )
    img_group.add_argument(
        "--output", dest="output", type=str, default=None,
        help="Folder for the points and cover CSVs. Default: CWD.",
    )
    _add_annotation_project_args(p_predict)

    # match-annotations
    p_match = subparsers.add_parser(
        "match-annotations",
        help=(
            "Match random point annotations to manual annotations of the same "
            "object (3-D proximity + SAM2 masks); write a matched CSV and a "
            "review PDF."
        ),
    )
    p_match.add_argument(
        "--random",
        dest="random",
        type=str,
        required=True,
        help=(
            "Random point annotations CSV (the file that gets matched). A "
            "previously matched CSV can be given to only match the rows without "
            "a manual_id/manual_id_conf."
        ),
    )
    p_match.add_argument(
        "--manual",
        dest="manual",
        type=str,
        default=None,
        help=(
            "Manual annotations CSV. Default: the --random CSV's sidecar "
            "matched_to, else the project's annotations file."
        ),
    )
    p_match.add_argument(
        "--export-labels",
        dest="export_labels",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Don't match: write the --random (matched) CSV with the manual set's "
            "current manual_label, manual_label_conf and manual_status joined in."
        ),
    )
    p_match.add_argument(
        "--label-prefix",
        dest="label_prefix",
        type=str,
        default=settings.MATCH_LABEL_PREFIX,
        help=(
            "Comma-separated label prefix(es) of random points to match. "
            f"Default: {settings.MATCH_LABEL_PREFIX} ('' = all)."
        ),
    )
    p_match.add_argument(
        "--max-dist",
        dest="max_dist",
        type=float,
        default=settings.MATCH_MAX_DISTANCE,
        help=(
            "Max 3-D distance (world units, m) of a candidate manual annotation. "
            f"Default: {settings.MATCH_MAX_DISTANCE}."
        ),
    )
    p_match.add_argument(
        "--max-candidates",
        dest="max_candidates",
        type=int,
        default=settings.MATCH_MAX_CANDIDATES,
        help=(
            "Nearest manual annotations tested per random point. "
            f"Default: {settings.MATCH_MAX_CANDIDATES}."
        ),
    )
    p_match.add_argument(
        "--fallback-dist",
        dest="fallback_dist",
        type=float,
        default=settings.MATCH_FALLBACK_DISTANCE,
        help=(
            "Accept the nearest candidate (confidence 'low') within this "
            "distance when no SAM2 mask agrees. "
            f"Default: {settings.MATCH_FALLBACK_DISTANCE}."
        ),
    )
    p_match.add_argument(
        "--sam2-checkpoint",
        dest="sam2_checkpoint",
        type=str,
        default=None,
        help=f"SAM2 checkpoint. Default: {settings.SAM2_CHECKPOINT}.",
    )
    p_match.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Output CSV. Default: <random>_matched.csv.",
    )
    p_match.add_argument(
        "--pdf-output",
        dest="pdf_output",
        type=str,
        default=None,
        help="Review PDF. Default: <random>_matches.pdf.",
    )
    p_match.add_argument(
        "--review-all",
        dest="review_all",
        action="store_true",
        help="Include every match in the review PDF (default: flagged only).",
    )
    p_match.add_argument(
        "--redo-all",
        dest="redo_all",
        action="store_true",
        help=(
            "Re-match all random points, overwriting existing manual_id/"
            "manual_id_conf values (default: skip rows that already have one)."
        ),
    )
    p_match.add_argument(
        "--no-pdf",
        dest="no_pdf",
        action="store_true",
        help="Skip the review PDF.",
    )
    p_match.add_argument(
        "--photo-path-replace",
        dest="photo_path_replace",
        nargs=2,
        default=None,
        metavar=("OLD", "NEW"),
        help="Remap camera photo paths (substring OLD -> NEW).",
    )
    p_match.add_argument(
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
        "--pdf-pairs",
        dest="pdf_pairs",
        type=int,
        default=settings.CAMSYNC_PDF_PAIRS,
        metavar="N",
        help=(
            "Number of time-matched pose/target image pairs to render in the "
            f"sanity PDF. Default: {settings.CAMSYNC_PDF_PAIRS}."
        ),
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
    p_transform.add_argument(
        "--to-parent",
        dest="to_parent",
        metavar="PROJECT",
        default=None,
        help=(
            "Instead of prompting, use the parent_transform of this project "
            "(dir or YAML): moves its annotations into the parent's orig frame"
        ),
    )
    p_transform.add_argument(
        "--from-parent",
        dest="from_parent",
        metavar="PROJECT",
        default=None,
        help=(
            "Like --to-parent but inverted: moves annotations of the parent "
            "into this project's orig frame"
        ),
    )

    # baseline
    p_baseline = subparsers.add_parser(
        "baseline",
        help=(
            "Check that world_transforms inherited from a baseline timepoint "
            "(baseline_id/baseline_transform) still match the baseline, and "
            "refresh them with --update"
        ),
    )
    p_baseline.add_argument(
        "projects",
        nargs="*",
        metavar="PROJECT",
        help=(
            "Project YAMLs or dirs, or folders of them (searched two levels "
            "down, e.g. a site folder); default: the current directory"
        ),
    )
    p_baseline.add_argument(
        "--update",
        action="store_true",
        help=(
            "Rewrite stale world_transforms (with scale factor and depth model) "
            "in place; baselines are refreshed before their dependants"
        ),
    )
    p_baseline.add_argument(
        "--link",
        metavar="BASELINE",
        default=None,
        help=(
            "Add baseline_id/baseline_transform to the given projects, whose "
            "world_transform was already built from this baseline (dir, YAML or "
            "id); baseline_transform = inv(baseline world_transform) @ "
            "world_transform, so only valid if the baseline was not re-oriented "
            "since"
        ),
    )

    # path-repair
    p_pathrepair = subparsers.add_parser(
        "path-repair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=(
            "Check and repair stale paths in a project: the project files "
            "listed in the YAML, the per-camera image paths and the "
            "annotation sets (sidecar YAMLs, cam_filepath)."
        ),
        epilog=(
            "Phase A repairs the project's own file paths in <id>.yaml, and\n"
            "the path keys (classes, cropped_path, classifier, matched_to) of\n"
            "each annotation sidecar <csv>.yaml; phase B repairs the per-camera\n"
            "image paths in <id>.meta.json; phase C repairs the cam_filepath\n"
            "column of the point CSVs in <id>.annos/ (a backup of each\n"
            "changed CSV goes to backup/). The .cams.xml is never modified (it\n"
            "stores image labels, not paths).\n"
            "\n"
            "Each broken path is resolved by trying, in order: the conventional\n"
            "filename in the project folder, a recursive search by basename,\n"
            "then asking. Repaired YAML entries inside the project folder are\n"
            "written as bare filenames, so the YAML survives the next move.\n"
            "The YAML is patched line by line, preserving comments, key order\n"
            "and any keys substrata does not itself use.\n"
            "\n"
            "All phases are planned in full before any writes, so aborting\n"
            "leaves the project untouched. --find/--replace does a literal\n"
            "substitution on camera paths and refuses to write unless every\n"
            "resulting path exists.\n"
            "\n"
            "Use this instead of --local. That flag rebases every path onto the\n"
            "working directory by basename, unverified and discarding any\n"
            "nested structure; this command changes only paths that are broken\n"
            "and verifies each result.\n"
            "\n"
            "Non-goals: --find is literal, never a regex or glob; a stored .jpg\n"
            "is not matched to an on-disk .png of the same stem."
        ),
    )
    _pr_scope = p_pathrepair.add_mutually_exclusive_group()
    _pr_scope.add_argument(
        "--yaml-only",
        dest="yaml_only",
        action="store_true",
        help=(
            "Repair the project YAML (and annotation sidecars) only; leave "
            "image paths alone."
        ),
    )
    _pr_scope.add_argument(
        "--cams-only",
        dest="cams_only",
        action="store_true",
        help=(
            "Repair image paths only (cameras and annotation cam_filepath); "
            "leave the YAMLs alone."
        ),
    )
    p_pathrepair.add_argument(
        "--no-annotations",
        dest="no_annotations",
        action="store_true",
        help=(
            "Leave the annotation sets alone: the sidecar YAMLs (<csv>.yaml) "
            "and the cam_filepath column of the point CSVs."
        ),
    )
    p_pathrepair.add_argument(
        "--find",
        dest="find",
        type=str,
        default=None,
        metavar="STR",
        help=(
            "Image paths only: consider cameras/annotation images whose "
            "stored path contains STR. With --replace, STR is also the "
            "substring that is replaced."
        ),
    )
    p_pathrepair.add_argument(
        "--replace",
        dest="replace",
        type=str,
        default=None,
        metavar="STR",
        help=(
            "Replace --find with STR in the stored paths (literal, not regex). "
            "Requires --find. Skips the recursive search; every resulting path "
            "must exist or nothing is written."
        ),
    )
    p_pathrepair.add_argument(
        "--cams-group",
        "--cams_group",
        dest="cams_group",
        type=str,
        default=None,
        metavar="NAME",
        help="Only consider cameras in this group (as given by Camera.group).",
    )
    p_pathrepair.add_argument(
        "--sensor-id",
        "--sensor_id",
        dest="sensor_id",
        type=int,
        default=None,
        metavar="ID",
        help="Only consider cameras using this sensor id (from the .cams.xml).",
    )
    p_pathrepair.add_argument(
        "--root",
        dest="root",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory searched recursively for images. Default: current folder.",
    )
    p_pathrepair.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the planned path changes and exit without writing.",
    )
    p_pathrepair.add_argument(
        "--yes",
        "-y",
        dest="yes",
        action="store_true",
        help="Skip the confirmation prompt before writing.",
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
        default=None,
        help=(
            "Glob pattern of annotation CSVs to collate, matched inside "
            f"--csv-path. Default: {settings.TRAIN_DEFAULT_PATTERN!r} "
            f"({settings.CORALNET_DEFAULT_PATTERN!r} with --coralnet)."
        ),
    )
    p_train.add_argument(
        "--classes",
        type=str,
        default=None,
        help=(
            "Classes CSV path with the CATAMI hierarchy (default: classes.csv "
            "in CWD). Optional: without one, labels are selected from a flat "
            "list of counts (no --tips_only/--collapse)."
        ),
    )
    p_train.add_argument(
        "--coralnet",
        action="store_true",
        help=(
            "The CSVs are CoralNet point exports (Name, Row, Column, Label "
            "code) on plain images, without 3-D data. They are converted into "
            f"<output>/{settings.CORALNET_POINTS_DIR}/ and split per image."
        ),
    )
    p_train.add_argument(
        "--image-dir",
        dest="image_dir",
        type=str,
        default=None,
        help=(
            "With --coralnet: directory holding the images, directly or in "
            "subfolders (searched recursively); each CSV is linked to the "
            "subfolder containing most of its images (default: --csv-path)."
        ),
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
        "--exclude-classes",
        dest="exclude_classes",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Labels to leave out entirely (e.g. CoralNet's 'Unc'); no crops "
            "are generated for them and they are never trained."
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
        "--sets", dest="sets", type=str, default=None, metavar="FILE",
        help="A text file listing the annotation CSVs to train on, one path per "
             "line (any folders; instead of PATTERN / --csv-path).",
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
        "--pdf",
        dest="pdf",
        action="store_true",
        help=(
            f"Write {settings.TRAIN_CROPS_PDF_FILE} with the crops per training "
            "class, each captioned with its annotation id, to inspect the labels "
            "(after the crops and label map; also with --prepare-only, --validate "
            "or --test)."
        ),
    )
    p_train.add_argument(
        "--pdf-max",
        dest="pdf_max",
        type=int,
        default=settings.TRAIN_PDF_MAX_PER_CLASS,
        metavar="N",
        help=(
            "Crops per class in the --pdf (a fixed sample when there are more; "
            f"0 = all). Default: {settings.TRAIN_PDF_MAX_PER_CLASS} (one page)."
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
        help=(
            "Point-feature classifier (a linear probe on frozen features read at "
            "each point, no crops) on a DINOv2 or DINOv3 backbone: dinov2 "
            "(= dinov2_vitb14_reg), dinov2-s/-l/-g, dinov3 (= dinov3_vitb16), "
            "dinov3-s/-l/-h, or a torch.hub name. DINOv3 weights are gated: "
            "download them once (see the error message, or the docs). Or a "
            "torchvision architecture (e.g. resnet34, resnet50): fastai crop "
            f"classifier. Default: {settings.TRAIN_DEFAULT_ARCH}."
        ),
    )
    p_feat = p_train.add_mutually_exclusive_group()
    p_feat.add_argument(
        "--feature-mm-px",
        dest="feature_mm_px",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Point features: resample every photo to this ground resolution "
            "(mm per model pixel), from each point's camera distance (project "
            "cameras or a cam_gsd_mm column). Default: automatic - physical "
            "when ground resolutions are known, else a fixed scale."
        ),
    )
    p_feat.add_argument(
        "--feature-scale",
        dest="feature_scale",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Point features: fixed resampling (model px per photo px), e.g. "
            "0.25 for 9568 px photos. Default: automatic (see --feature-mm-px)."
        ),
    )
    p_train.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=settings.POINT_BATCH_SIZE,
        help=(
            "Point features: windows per backbone pass. "
            f"Default: {settings.POINT_BATCH_SIZE}."
        ),
    )
    p_train.add_argument(
        "--pretrained-weights",
        dest="pretrained_weights",
        type=str,
        default=None,
        metavar="CKPT",
        help=(
            "Encoder checkpoint to start from instead of the ImageNet weights, "
            "e.g. BenthicNet's ResNet-50 (use with a matching --arch, such as "
            "resnet50)."
        ),
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
            "Classifier path (also the load target with --validate/--test). "
            f"Default: <output>/{settings.POINT_DEFAULT_MODEL_FILE} for point "
            f"features, <output>/{settings.TRAIN_DEFAULT_MODEL_FILE} for fastai."
        ),
    )
    p_train.add_argument(
        "--jobs",
        type=int,
        default=settings.TRAIN_CROP_JOBS,
        help=(
            "Parallel workers for crop generation / photo decoding (-1 = all "
            f"cores). Default: {settings.TRAIN_CROP_JOBS}."
        ),
    )
    p_train.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Skip training; load --model and re-run stats on the validation "
            "crops (points, for a point classifier)."
        ),
    )
    p_train.add_argument(
        "--test",
        action="store_true",
        help=(
            "Skip training; load --model and report stats on the test crops "
            "(points, for a point classifier)."
        ),
    )
    p_train.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmations (labels, deletions, paths).",
    )
    p_train.add_argument(
        "--curation", action="append", default=None, metavar="JSONL",
        help=(
            f"Another curation store to apply (read-only; repeatable), e.g. the "
            f"{settings.CURATION_FILE} of another classifier folder. The "
            "output folder's own store is always applied (it wins)."
        ),
    )
    p_train.add_argument(
        "--no-curation", dest="no_curation", action="store_true",
        help="Ignore curation decisions (and pool points labelled in them).",
    )

    # calibrate (point classifier -> another camera system)
    p_cal = subparsers.add_parser(
        "calibrate",
        help=(
            "Calibrate a point classifier to another camera system from "
            "unlabelled photos of the same area as its training photos."
        ),
    )
    p_cal.add_argument(
        "images",
        nargs="+",
        help=(
            "Photos of the target camera system covering the training area "
            "(files, folders searched recursively, or glob patterns)."
        ),
    )
    p_cal.add_argument(
        "--classifier",
        type=str,
        default=None,
        help=f"Point classifier (default: {settings.POINT_DEFAULT_MODEL_FILE} "
        "in the CWD).",
    )
    p_cal.add_argument(
        "--gsd-mm",
        dest="gsd_mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Ground resolution of the photos in mm per pixel (camera distance "
            "/ focal length); required for classifiers with a physical "
            "resolution."
        ),
    )
    p_cal.add_argument(
        "--n-images",
        dest="n_images",
        type=int,
        default=settings.POINT_CALIB_IMAGES,
        help=f"Photos sampled. Default: {settings.POINT_CALIB_IMAGES}.",
    )
    p_cal.add_argument(
        "--per-image",
        dest="per_image",
        type=int,
        default=settings.POINT_CALIB_PER_IMAGE,
        help=f"Random patches per photo. Default: {settings.POINT_CALIB_PER_IMAGE}.",
    )
    p_cal.add_argument(
        "--output",
        type=str,
        default=None,
        help="Calibrated classifier path (default: <classifier>_calibrated.pt).",
    )
    p_cal.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=settings.POINT_BATCH_SIZE,
        help=f"Windows per backbone pass. Default: {settings.POINT_BATCH_SIZE}.",
    )
    p_cal.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Photo decoding threads (-1 = all cores). Default: -1.",
    )

    # masks (dense per-photo class masks from a point classifier)
    p_masks = subparsers.add_parser(
        "masks",
        help=(
            "Dense per-photo class masks (label + confidence PNGs) and per-photo "
            "cover from a point classifier, for plain photos (no 3-D model)."
        ),
    )
    p_masks.add_argument(
        "images", nargs="+",
        help="Photos (files, folders searched recursively, or glob patterns).",
    )
    p_masks.add_argument(
        "--classifier", type=str, default=None,
        help=f"Point classifier (default: {settings.POINT_DEFAULT_MODEL_FILE} in "
        "the CWD; use a calibrated one for another camera system).",
    )
    p_masks.add_argument(
        "--gsd-mm", dest="gsd_mm", type=float, default=None, metavar="MM",
        help="Ground resolution of the photos (mm/px); needed for classifiers "
        "with a physical resolution (otherwise their fixed scale is used).",
    )
    p_masks.add_argument(
        "--output", type=str, default=None,
        help=f"Output folder (default: ./{settings.MASKS_DEFAULT_DIR}).",
    )
    p_masks.add_argument(
        "--probs", action="store_true",
        help="Also save all class probabilities per photo (<stem>_probs.npz).",
    )
    p_masks.add_argument(
        "--color", dest="color", action="append", default=None,
        metavar="LABEL=RRGGBB", help="Mask colour of a class (repeatable).",
    )
    p_masks.add_argument(
        "--overwrite", action="store_true",
        help="Redo photos that already have masks.",
    )
    p_masks.add_argument(
        "--jobs", type=int, default=-1,
        help="Photo decoding threads (-1 = all cores). Default: -1.",
    )

    # curate (training-data curation, run in the classifier folder)
    p_cur = subparsers.add_parser(
        "curate",
        help=(
            "Curate point-classifier training data in a classifier folder: "
            "score points for review, mine unlabelled photos into clustered "
            "pools, manage decisions; --gui opens the review grid."
        ),
    )
    p_cur.add_argument(
        "--gui", action="store_true",
        help="Open the review grid in the browser (substrata-explore; "
        "`substrata curate --gui -h` for its options).",
    )
    p_cur.add_argument(
        "--folder", default=None,
        help="Classifier folder (default: the CWD).",
    )
    p_cur.add_argument(
        "--model", default=None,
        help=f"Point classifier (default: {settings.POINT_DEFAULT_MODEL_FILE} "
        "in the folder).",
    )
    p_cur.add_argument(
        "--annotations", default=None,
        help=f"Training annotations (default: {settings.TRAIN_ANNOTATIONS_FILE} "
        "in the folder).",
    )
    p_cur.add_argument(
        "--mine", nargs="+", default=None, metavar="IMAGES",
        help="Sample unlabelled points from these photos (files, folders, glob "
        "patterns, or a text file listing photos) into a clustered pool.",
    )
    p_cur.add_argument(
        "--image-dir", dest="image_dir", default=None, metavar="DIR",
        help="Image-only pool, as `substrata predict --image-dir`: sample points "
        "on the photos under DIR (recursive; same as --mine DIR). With --points: "
        "where a CoralNet export's photos are (default: the CSV's folder).",
    )
    p_cur.add_argument(
        "--points", default=None, metavar="CSV",
        help="Pool of these points instead of sampled ones, as `substrata predict "
        "--points`: a points CSV (cam_filepath, cam_x, cam_y; e.g. predict's "
        "output) or a CoralNet export. Their labels are shown as given labels: "
        "accepting one in the review grid labels the point with it.",
    )
    p_cur.add_argument("--name", default=None, help="Pool name (--mine, --image-dir, --points).")
    p_cur.add_argument(
        "--gsd-mm", dest="gsd_mm", type=float, default=None, metavar="MM",
        help="Ground resolution of the pool's photos (mm/px). Without it the "
        "window covers the share of each photo's width it covers on the "
        "training photos.",
    )
    p_cur.add_argument(
        "--domain", default=None,
        help="Camera domain of the pool (default: its name); pools of one "
        "camera system can share it.",
    )
    p_cur.add_argument(
        "--mine-images", dest="mine_images", type=int,
        default=settings.CURATION_MINE_IMAGES,
        help=f"Photos sampled (default: {settings.CURATION_MINE_IMAGES}).",
    )
    p_cur.add_argument(
        "--per-image", "--n-points", dest="per_image", type=int,
        default=settings.CURATION_MINE_PER_IMAGE,
        help=f"Points per photo (default: {settings.CURATION_MINE_PER_IMAGE}).",
    )
    p_cur.add_argument(
        "--clusters", type=int, default=settings.CURATION_MINE_CLUSTERS,
        help=f"Clusters (default: {settings.CURATION_MINE_CLUSTERS}).",
    )
    p_cur.add_argument(
        "--resample", action="store_true",
        help="Sample the pool again (default: re-cluster the same points).",
    )
    p_cur.add_argument(
        "--seed", type=int, default=0,
        help="Random seed of the pool's sampling (default: 0).",
    )
    p_cur.add_argument(
        "--status", action="store_true", help="Summarise the decisions.",
    )
    p_cur.add_argument(
        "--export", default=None, metavar="CSV",
        help="Write each point's current decision as CSV (no history), by photo.",
    )
    p_cur.add_argument(
        "--import", dest="import_csv", default=None, metavar="CSV",
        help="Append decisions from a CSV (action, label, reason and key, "
        "photo + x + y, or id).",
    )
    p_cur.add_argument(
        "--sheets", action="store_true",
        help="Contact sheets (PDF) of the review queues, without the GUI.",
    )
    p_cur.add_argument(
        "--sheets-max", dest="sheets_max", type=int, default=330,
        help="Points per queue in --sheets (default: 330, two pages).",
    )
    p_cur.add_argument(
        "--curation", action="append", default=None, metavar="JSONL",
        help="Another curation store to read (read-only; repeatable).",
    )
    p_cur.add_argument(
        "--batch-size", dest="batch_size", type=int,
        default=settings.POINT_BATCH_SIZE,
        help=f"Windows per forward pass (default: {settings.POINT_BATCH_SIZE}).",
    )
    p_cur.add_argument(
        "--jobs", type=int, default=-1,
        help="Photo decoding threads (-1 = all cores). Default: -1.",
    )

    args = parser.parse_args()

    handlers = {
        "decimate": handle_decimate,
        "stratify": handle_stratify,
        "metashape-export": handle_metashape_export,
        "ply-repair": handle_ply_repair,
        "head": handle_head,
        "scalebars": handle_scalebars,
        "views": handle_views,
        "orient": handle_orient,
        "segment": handle_segment,
        "colors": handle_colors,
        "firefish": handle_firefish,
        "cams2video": handle_cams2video,
        "intercepts": handle_intercepts,
        "intercepts-plot": handle_intercepts_plot,
        "align": handle_align,
        "images": handle_images,
        "thumbnails": handle_thumbnails,
        "crops": handle_crops,
        "import-coralnet": handle_import_coralnet,
        "predict": handle_predict,
        "match-annotations": handle_match_annotations,
        "camsync": handle_camsync,
        "path-repair": handle_path_repair,
        "transform": handle_transform,
        "baseline": handle_baseline,
        "train": handle_train,
        "calibrate": handle_calibrate,
        "masks": handle_masks,
        "curate": handle_curate,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
