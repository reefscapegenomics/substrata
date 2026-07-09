"""Animated versions of substrata visualizations (matplotlib-based).

The first animation, :func:`animate_ortho_grid`, replays an
:class:`substrata.ortho.OrthoGrid` by gradually filling its cells while the
side-panel summary (label bar chart or value histogram) grows in step, and
writes the result to a GIF or MP4.
"""

from __future__ import annotations

# Standard Library
import os
import warnings
from typing import Optional, Tuple

# Third-Party
import numpy as np

# Local
from substrata.logging import logger


def _reveal_t(nx: int, ny: int) -> np.ndarray:
    """Column-major reveal fraction per lattice cell.

    The sweep runs left→right across columns and, within each column, top→bottom
    (``origin="lower"`` means a high row index ``j`` is the top). Timing is based
    on lattice *position* (not just occupied cells) so the sweep is spatially
    uniform.

    Args:
        nx: Number of grid columns.
        ny: Number of grid rows.

    Returns:
        ``(ny, nx)`` float array in ``[0, 1)``: a cell becomes visible once the
        animation progress ``p`` reaches its value.
    """
    i = np.arange(nx)[None, :]          # column index -> (1, nx)
    j = np.arange(ny)[:, None]          # row index, 0 = bottom -> (ny, 1)
    row_from_top = (ny - 1) - j          # 0 at the top of the column
    order = i * ny + row_from_top        # (ny, nx), column-major, top-first
    # Normalize to (0, 1]: the first cell reveals at p = 1/N (so a p=0 frame is
    # empty) and the last at p = 1 (so the final frame is complete and the fill
    # spans the whole duration).
    return (order + 1).astype(float) / float(max(nx * ny, 1))


def _order_to_reveal_t(order: np.ndarray) -> np.ndarray:
    """Turn an integer reveal *order* (0 = first) into reveal fractions in (0, 1].

    The smallest order reveals at ``1/N`` (so a ``p=0`` frame is empty) and the
    largest at ``1.0`` (the final frame is complete), matching :func:`_reveal_t`.
    """
    n = order.size
    return (order.astype(float) + 1.0) / float(max(n, 1))


def _reveal_t_rows(nx: int, ny: int) -> np.ndarray:
    """Raster reveal: top→bottom, each row left→right (image-scanline order)."""
    i = np.arange(nx)[None, :]
    j = np.arange(ny)[:, None]
    row_from_top = (ny - 1) - j              # 0 at the top
    order = row_from_top * nx + i            # (ny, nx), row-major, top-first
    return _order_to_reveal_t(order)


def _reveal_t_random(nx: int, ny: int, seed: int = 0) -> np.ndarray:
    """Random reveal: cells pop in in a shuffled order (deterministic per seed).

    Reads as many parallel predictions streaming in and completing at different
    times.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(nx * ny).reshape(ny, nx)
    return _order_to_reveal_t(order)


def _reveal_t_spiral(nx: int, ny: int) -> np.ndarray:
    """Radial reveal: expand outward from the lattice centre (angle tie-break).

    Gives the impression of analysis rippling out across the scene.
    """
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    radius = np.hypot(i - cx, j - cy)
    angle = np.arctan2(j - cy, i - cx)       # -pi..pi, rotational tie-break
    key = radius * (2.0 * np.pi + 1.0) + (angle + np.pi)
    order = key.ravel().argsort().argsort().reshape(ny, nx)  # dense ranks
    return _order_to_reveal_t(order)


def _reveal_t_categories(grid, present, report, force_nodata_last: bool = True):
    """Reveal one label group at a time, most dominant → least dominant.

    Dominance is the number of *reported* cells of each label (matching the side
    bar chart). Each category gets an equal share of the fill; within a category
    its cells sweep in column-major (top-first) order so the class "scans in".

    Args:
        grid: The :class:`~substrata.ortho.OrthoGrid` (label mode).
        present: ``(ny, nx)`` bool mask of occupied cells.
        report: ``(ny, nx)`` bool mask of cells inside the reporting area.
        force_nodata_last: Reveal the "No data" (unclassified) group last
            regardless of its size.

    Returns:
        Tuple ``(reveal_t, n_categories)``.
    """
    ny, nx = grid.ny, grid.nx
    cells: dict = {}
    report_counts: dict = {}
    for j in range(ny):
        for i in range(nx):
            if not present[j, i]:
                continue
            lbl = grid.cell_labels[j, i]
            cat = lbl if lbl is not None else "No data"
            cells.setdefault(cat, []).append((j, i))
            if report[j, i]:
                report_counts[cat] = report_counts.get(cat, 0) + 1

    def dominance(cat):
        # Most reported cells first; break ties on total occupancy, then name.
        return (-report_counts.get(cat, 0), -len(cells[cat]), str(cat))

    ordered = sorted(cells.keys(), key=dominance)
    if force_nodata_last and "No data" in ordered:
        ordered = [c for c in ordered if c != "No data"] + ["No data"]

    k = max(1, len(ordered))
    reveal_t = np.ones((ny, nx), dtype=float)
    for rank, cat in enumerate(ordered):
        # Column-major, top-first within the category (like the columns sweep).
        members = sorted(cells[cat], key=lambda ji: (ji[1], (ny - 1) - ji[0]))
        m = len(members)
        for idx, (j, i) in enumerate(members):
            reveal_t[j, i] = (rank + (idx + 1) / m) / k
    return reveal_t, k


def _writer_for(output_path: str, fps: int):
    """Return a matplotlib animation writer chosen by file extension."""
    from matplotlib.animation import PillowWriter, FFMpegWriter

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        return PillowWriter(fps=fps)
    if ext in (".mp4", ".m4v", ".mov"):
        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                f"ffmpeg not found on PATH; required for {ext!r} output "
                "(use .gif instead, or install ffmpeg)"
            )
        return FFMpegWriter(fps=fps)
    raise ValueError(
        f"unsupported output extension {ext!r}; use .gif, .mp4, .m4v, or .mov"
    )


def _set_gif_loop(path: str, loop: bool, fps: int) -> None:
    """Rewrite a GIF to loop forever or (default) play once and stop.

    matplotlib's ``PillowWriter`` always writes an infinitely-looping GIF; this
    re-saves the frames with the requested loop behaviour. When *loop* is False
    no NETSCAPE loop block is written, so the GIF plays once and holds the last
    frame.
    """
    from PIL import Image, ImageSequence

    with Image.open(path) as gif:
        frames, durations = [], []
        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.convert("RGB"))
            durations.append(frame.info.get("duration", int(1000 / fps)))
    if not frames:
        return
    save_kwargs = dict(
        save_all=True, append_images=frames[1:],
        duration=durations, disposal=2,
    )
    if loop:
        save_kwargs["loop"] = 0
    else:
        # Drop any inherited loop flag so no NETSCAPE loop block is written:
        # the GIF then plays once and holds the last frame.
        frames[0].info.pop("loop", None)
    frames[0].save(path, **save_kwargs)


def animate_ortho_grid(
    grid,
    output_path: str,
    duration: Optional[float] = None,
    fps: Optional[int] = None,
    sweep: str = "columns",
    show_pcd: bool = True,
    cmap: Optional[str] = None,
    label_colors: Optional[dict] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 7.5),
    end_hold: float = 0.5,
    loop: bool = False,
    seconds_per_category: Optional[float] = None,
) -> str:
    """Animate an :class:`~substrata.ortho.OrthoGrid` filling in, and save it.

    Shows the whole plot (faded point cloud + empty grid) and then gradually
    fills the grid cells in the order set by *sweep*, with the side panel (label
    bar chart or value histogram) growing as cells appear. Works for label grids
    (``value_by="label"``) and continuous grids (``"z"``/``"count"``/``"density"``).

    Args:
        grid: An :class:`~substrata.ortho.OrthoGrid`.
        output_path: Destination file; ``.gif`` (Pillow) or ``.mp4`` (ffmpeg).
        duration: Total fill time in seconds (default
            ``settings.DEFAULT_ANIM_DURATION``; for ``sweep="categories"`` it
            defaults to ~1s per category instead). The end hold is added on top.
        fps: Frames per second (default ``settings.DEFAULT_ANIM_FPS``).
        sweep: Reveal order — one of:

            * ``"columns"`` (default): left→right, each column top→bottom.
            * ``"rows"``: top→bottom raster (image-scanline order).
            * ``"scan"``: same reveal as ``"columns"`` plus a moving vertical
              scan line, evoking a sensor sweeping the scene live.
            * ``"random"``: cells pop in in a shuffled order, like many
              predictions streaming in and completing at different times.
            * ``"spiral"``: expand radially outward from the centre.
            * ``"categories"`` (label grids only): reveal one label group at a
              time, most dominant → least dominant, "detecting" each class in
              turn. Defaults to ~1s per category (see *seconds_per_category*).
        show_pcd: Draw the whole point cloud (faded) behind the grid and widen
            the view to its full extent (as :meth:`OrthoGrid.show`).
        cmap: Colormap for continuous modes (default viridis).
        label_colors: Optional explicit ``{label: color}`` map (label mode).
        title: Optional left-panel title.
        figsize: Figure size in inches.
        end_hold: Seconds to hold the fully-filled frame at the end.
        loop: If True the GIF loops forever; if False (default) it plays once
            and stops on the final (full) frame. Only affects GIF output.
        seconds_per_category: For ``sweep="categories"`` with no explicit
            *duration*, seconds allotted to each label group (default
            ``settings.DEFAULT_ANIM_SECONDS_PER_CATEGORY``). Ignored otherwise.

    Returns:
        ``output_path``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation

    from substrata import settings

    if fps is None:
        fps = getattr(settings, "DEFAULT_ANIM_FPS", 15)
    valid_sweeps = ("columns", "rows", "scan", "random", "spiral", "categories")
    if sweep not in valid_sweeps:
        raise ValueError(
            f"unsupported sweep {sweep!r}; choose from {valid_sweeps}"
        )
    # Validate the output target up front so a bad path / missing ffmpeg fails
    # before we build the (potentially expensive) figure and animation.
    writer = _writer_for(output_path, fps)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    is_label = grid.value_by == "label"
    ny, nx = grid.ny, grid.nx
    present = grid.present
    report = grid.report_mask
    extent = grid.extent

    # Per-cell reveal fraction in (0, 1], selected by `sweep`. Everything
    # downstream (growing bars, cell fills, the optional scan line) keys off it.
    n_categories = None
    if sweep in ("columns", "scan"):
        reveal_t = _reveal_t(nx, ny)
    elif sweep == "rows":
        reveal_t = _reveal_t_rows(nx, ny)
    elif sweep == "random":
        reveal_t = _reveal_t_random(nx, ny)
    elif sweep == "spiral":
        reveal_t = _reveal_t_spiral(nx, ny)
    else:  # "categories"
        if not is_label:
            raise ValueError(
                "sweep='categories' requires a label grid (value_by='label')"
            )
        reveal_t, n_categories = _reveal_t_categories(grid, present, report)

    # Resolve the fill duration. In 'categories' mode default to ~1s per label
    # group (dominant → least) unless the caller passed an explicit duration.
    if duration is None:
        if sweep == "categories":
            spc = seconds_per_category
            if spc is None:
                spc = getattr(
                    settings, "DEFAULT_ANIM_SECONDS_PER_CATEGORY", 1.0
                )
            duration = max(1, n_categories) * spc
        else:
            duration = getattr(settings, "DEFAULT_ANIM_DURATION", 5.0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.35])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_aspect("equal")
    ax_right = fig.add_subplot(gs[0, 1])

    pcd_extent = grid._draw_context(ax_left) if show_pcd else None

    # A subtle checkerboard placeholder so the empty cell grid is visible from
    # the first frame; each cell then *changes colour* to its assigned value as
    # the sweep reaches it (rather than fading in over the point cloud).
    checker = np.add.outer(np.arange(ny), np.arange(nx)) % 2
    placeholder = np.zeros((ny, nx, 4), dtype=float)
    shade = np.where(checker == 0, 0.90, 0.82)
    placeholder[..., 0] = placeholder[..., 1] = placeholder[..., 2] = shade
    # The whole lattice is an opaque checkerboard (no faded point cloud showing
    # through empty/absent cells), so the grid reads cleanly and only the
    # occupied cells change colour as the sweep reaches them.
    placeholder[..., 3] = 1.0

    # ---- per-mode static setup + per-frame update closure ---------------
    if is_label:
        target_rgb, bars, categories, rep_t, rep_ci = _setup_label(
            grid, ax_left, ax_right, label_colors, mpatches, plt,
            present, report, reveal_t,
        )

        def draw_side(p):
            counts = np.zeros(len(categories))
            sel = rep_t <= p
            if sel.any():
                counts = np.bincount(rep_ci[sel], minlength=len(categories))
            for bar, h in zip(bars, counts):
                bar.set_height(h)

        def cell_rgba(shown):
            rgba = placeholder.copy()
            rgba[shown, :3] = target_rgb[shown]
            rgba[shown, 3] = 1.0
            return rgba
    else:
        target_rgba, bars, edges, rep_t, rep_v, finite = _setup_continuous(
            grid, ax_left, ax_right, cmap, plt, present, report, reveal_t,
        )

        def draw_side(p):
            sel = rep_t <= p
            h = (np.histogram(rep_v[sel], bins=edges)[0]
                 if sel.any() else np.zeros(len(edges) - 1))
            for bar, hh in zip(bars, h):
                bar.set_height(hh)

        def cell_rgba(shown):
            rgba = placeholder.copy()
            m = shown & finite
            rgba[m] = target_rgba[m]
            return rgba

    im = ax_left.imshow(
        np.zeros((ny, nx, 4), dtype=float),
        extent=extent, origin="lower", zorder=1, interpolation="nearest",
    )

    # View the full cloud (union with grid) when the context is drawn.
    ext = list(extent)
    if pcd_extent is not None:
        ext = [min(ext[0], pcd_extent[0]), max(ext[1], pcd_extent[1]),
               min(ext[2], pcd_extent[2]), max(ext[3], pcd_extent[3])]
    ax_left.set_xlim(ext[0], ext[1])
    ax_left.set_ylim(ext[2], ext[3])

    if grid.report_bbox is not None:
        (rx0, ry0), (rx1, ry1) = grid.report_bbox
        ax_left.add_patch(mpatches.Rectangle(
            (rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
            edgecolor="black", linestyle="--", linewidth=1.2, zorder=3,
        ))
    if title is not None:
        ax_left.set_title(title)

    # Optional moving scan line that tracks the (column) reveal frontier and
    # trails the classified cells behind it — a "sensor sweeping live" cue.
    scanline = None
    if sweep == "scan":
        scanline = ax_left.axvline(
            extent[0], color="#00e5ff", linewidth=2.0, alpha=0.9, zorder=4,
        )

    # Match OrthoGrid.show()'s layout: reserve room for the legend below the
    # left panel and make the right (count) panel the same height as the
    # equal-aspect left panel.
    fig.subplots_adjust(bottom=0.2, wspace=0.12)
    try:
        left_pos = ax_left.get_position()
        right_pos = ax_right.get_position()
        ax_right.set_position(
            [right_pos.x0, left_pos.y0, right_pos.width, left_pos.height]
        )
    except Exception:  # pragma: no cover
        pass

    n_frames = max(1, int(round(fps * duration)))
    hold = max(0, int(round(fps * end_hold)))
    total_frames = n_frames + hold

    def update(f):
        # Progress runs 0 -> 1 across the fill frames (frame 0 shows the empty
        # grid, the last fill frame is complete), then holds at 1.
        if f >= n_frames or n_frames == 1:
            p = 1.0
        else:
            p = f / (n_frames - 1)
        im.set_data(cell_rgba(present & (reveal_t <= p)))
        draw_side(p)
        artists = [im, *bars]
        if scanline is not None:
            xf = extent[0] + p * (extent[1] - extent[0])
            scanline.set_xdata([xf, xf])
            scanline.set_visible(p < 1.0)   # hide once the fill completes
            artists.append(scanline)
        return artists

    anim = FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000.0 / fps, blit=False,
    )
    anim.save(output_path, writer=writer)
    plt.close(fig)
    if os.path.splitext(output_path)[1].lower() == ".gif":
        _set_gif_loop(output_path, loop, fps)
    logger.info(
        "animate_ortho_grid: wrote %s (%d frames, %.1fs fill + %.1fs hold @ %dfps)",
        output_path, total_frames, duration, end_hold, fps,
    )
    return output_path


def _setup_label(grid, ax_left, ax_right, label_colors, mpatches, plt,
                 present, report, reveal_t):
    """Static label-mode setup; returns per-cell colours + animatable bars."""
    label_colors, labels_present = grid._resolve_label_colors(label_colors)
    no_data = (0.7, 0.7, 0.7)
    ny, nx = grid.ny, grid.nx

    target_rgb = np.zeros((ny, nx, 3), dtype=float)
    cell_cat = np.empty((ny, nx), dtype=object)
    final_counts: dict = {}
    for j in range(ny):
        for i in range(nx):
            if not present[j, i]:
                continue
            lbl = grid.cell_labels[j, i]
            col = label_colors.get(lbl, no_data) if lbl is not None else no_data
            target_rgb[j, i] = col[:3]
            cat = lbl if lbl is not None else "No data"
            cell_cat[j, i] = cat
            if report[j, i]:
                final_counts[cat] = final_counts.get(cat, 0) + 1

    categories = sorted(k for k in final_counts if k != "No data")
    if "No data" in final_counts:
        categories.append("No data")
    bar_colors = [
        label_colors.get(c, no_data) if c != "No data" else no_data
        for c in categories
    ]
    x_pos = np.arange(len(categories))
    bars = ax_right.bar(x_pos, np.zeros(len(categories)), color=bar_colors)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels([str(c) for c in categories], rotation=90, fontsize=8)
    ax_right.set_ylabel("Count")

    # Match OrthoGrid.show()'s final-frame styling: a log y-scale for wide count
    # spreads, and a count annotation above each (final) bar. Bars start at 0 and
    # grow into this fixed axis during the animation.
    final = [final_counts.get(c, 0) for c in categories]
    positive = [c for c in final if c > 0]
    if positive and max(positive) / min(positive) >= 20:
        # Bars start at 0; guard the all-zero log-autoscale warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax_right.set_yscale("log")
        ax_right.set_ylim(0.7, max(final) * 1.6)
    else:
        top = max(final) if final and max(final) > 0 else 1
        ax_right.set_ylim(0, top * 1.15)

    handles = [
        mpatches.Patch(facecolor=label_colors.get(lbl, no_data),
                       edgecolor="none", label=str(lbl))
        for lbl in labels_present
    ]
    handles.append(mpatches.Patch(facecolor=no_data, edgecolor="none",
                                  label="No data"))
    ax_left.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.16),
                   ncol=min(len(handles), 6), frameon=False)

    cat_index = {c: k for k, c in enumerate(categories)}
    rmask = present & report
    rj, ri = np.where(rmask)
    rep_t = reveal_t[rj, ri]
    rep_ci = np.array([cat_index[cell_cat[j, i]] for j, i in zip(rj, ri)],
                      dtype=int)
    return target_rgb, bars, categories, rep_t, rep_ci


def _setup_continuous(grid, ax_left, ax_right, cmap, plt,
                      present, report, reveal_t):
    """Static continuous-mode setup; returns target RGBA + animatable bars."""
    cmap_obj, norm = grid._continuous_scale(cmap)
    target_rgba = cmap_obj(norm(grid.values))
    finite = ~np.isnan(grid.values)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    ax_left.figure.colorbar(sm, ax=ax_left, fraction=0.046, pad=0.04,
                            label=grid._value_label())

    rep_all = grid._report_values()
    if rep_all.size:
        nb = int(np.clip(np.sqrt(rep_all.size), 5, 30))
        edges = np.histogram_bin_edges(rep_all, bins=nb)
        final_hist = np.histogram(rep_all, bins=edges)[0]
    else:
        edges = np.linspace(0.0, 1.0, 6)
        final_hist = np.zeros(len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bars = ax_right.bar(centers, np.zeros(len(centers)), width=np.diff(edges),
                        color="0.5", edgecolor="0.3", align="center")
    ax_right.set_xlabel(grid._value_label())
    ax_right.set_ylabel("Cell count")
    top = final_hist.max() if final_hist.size and final_hist.max() > 0 else 1
    ax_right.set_ylim(0, top * 1.15)

    rmask = present & report & finite
    rj, ri = np.where(rmask)
    rep_t = reveal_t[rj, ri]
    rep_v = grid.values[rj, ri]
    return target_rgba, bars, edges, rep_t, rep_v, finite
