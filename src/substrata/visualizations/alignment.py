"""Alignment QC figures (``substrata align``).

Where a point cloud sits within another (``plot_pcd_context``), how well the two
coincide in their overlap (``plot_pcd_correspondence``, statistics from
:func:`substrata.measurements.pcd_correspondence`), matched camera residuals
(``plot_camera_correspondence``, :func:`substrata.measurements.camera_residuals`)
and the ``substrata align --pdf`` report (``save_align_qc_pdf``).
"""

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np

# Local Modules
from substrata.visualizations.reports import plot_text


def _pcd_xyz_rgb(pcd):
    """Return ``(points, colors)`` arrays for any supported point cloud type."""
    pts = np.asarray(pcd.points, dtype=float)
    cols = getattr(pcd, "colors", None)
    cols = None if cols is None else np.asarray(cols, dtype=float)
    if cols is None or cols.shape != pts.shape:
        cols = np.full_like(pts, 0.5)
    return pts, cols


def _subsample_idx(n, k, rng):
    """Indices of at most ``k`` of ``n`` items (all of them if ``n <= k``)."""
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def plot_pcd_context(
    src,
    tgt,
    labels=("source", "target"),
    units="m",
    title=None,
    max_output_points=200_000,
    ortho_pixel_width=2000,
    width=12,
    height=11,
):
    """Show where ``src`` sits within the whole of ``tgt``.

    Both clouds must already be in the same frame. Intended for e.g. a focal
    plot aligned onto a large AUV model: a top-down RGB ortho of the target
    with the source footprint outlined, plus XZ and YZ side profiles of the
    target with the source overlaid, to check that the source lies at the
    expected position and z (depth).

    Args:
        src: Source point cloud (PointCloud, SimplePointCloud or Open3D).
        tgt: Target point cloud, same frame as ``src``.
        labels: Names used for ``src`` and ``tgt`` in legends.
        units: Coordinate unit label.
        title: Optional figure title.
        max_output_points: Maximum points per cloud in the profile scatters.
        ortho_pixel_width: Width in pixels of the target ortho raster.
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull

    from types import SimpleNamespace

    from substrata.ortho import OrthoMap

    rng = np.random.default_rng(0)
    s_pts, _ = _pcd_xyz_rgb(src)
    t_pts, t_cols = _pcd_xyz_rgb(tgt)
    s_lab, t_lab = labels

    s_sub = s_pts[_subsample_idx(len(s_pts), max_output_points, rng)]
    t_sub = t_pts[_subsample_idx(len(t_pts), max_output_points, rng)]
    hull = s_sub[ConvexHull(s_sub[:, :2]).vertices, :2]
    hull_closed = np.vstack([hull, hull[:1]])
    centroid = s_pts.mean(axis=0)

    # Target z within the source footprint (convex hull)
    in_fp = Path(hull).contains_points(t_sub[:, :2])
    t_z_fp = t_sub[in_fp, 2]

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])

    # Top-down ortho of the full target, and zoomed on the source footprint
    ortho = OrthoMap(
        SimpleNamespace(points=t_pts, colors=t_cols), pixel_width=ortho_pixel_width
    )
    x0, y0 = ortho.origin
    extent = (
        x0,
        x0 + ortho.width * ortho.resolution,
        y0,
        y0 + ortho.height * ortho.resolution,
    )
    s_lo, s_hi = s_sub.min(axis=0), s_sub.max(axis=0)
    half = 0.75 * max(s_hi[0] - s_lo[0], s_hi[1] - s_lo[1])
    for col, zoom in enumerate((False, True)):
        ax_map = fig.add_subplot(gs[0, col])
        ax_map.imshow(ortho.image, extent=extent, interpolation="nearest")
        ax_map.fill(hull[:, 0], hull[:, 1], color="tab:red", alpha=0.2, lw=0)
        ax_map.plot(hull_closed[:, 0], hull_closed[:, 1], color="tab:red", lw=1.5,
                    label=f"{s_lab} footprint")
        ax_map.plot(*centroid[:2], marker="+", color="tab:red", ms=12, mew=2)
        if zoom:
            ax_map.set_xlim(centroid[0] - half, centroid[0] + half)
            ax_map.set_ylim(centroid[1] - half, centroid[1] + half)
            ax_map.set_title("Zoomed on footprint")
        else:
            ax_map.set_title("Source footprint within target")
            ax_map.legend(loc="upper right", fontsize=8)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel(f"x ({units})")
        ax_map.set_ylabel(f"y ({units})")

    lines = [
        f"{s_lab} centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, "
        f"{centroid[2]:.2f}) {units}",
        f"{s_lab} z range: {s_pts[:, 2].min():.2f} to {s_pts[:, 2].max():.2f}",
    ]
    if len(t_z_fp):
        lines.append(
            f"{t_lab} z in footprint: {t_z_fp.min():.2f} to {t_z_fp.max():.2f}"
        )
    lines.append(
        f"{t_lab} z range (all): {t_pts[:, 2].min():.2f} to {t_pts[:, 2].max():.2f}"
    )
    ax_map.text(
        0.01, 0.01, "\n".join(lines), transform=ax_map.transAxes, fontsize=8,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    # Side profiles of the target in a band through the source footprint
    # (full length along the profile axis), with the source overlaid
    for col, (axis, across, name, other) in enumerate(
        ((0, 1, "x", "y"), (1, 0, "y", "x"))
    ):
        ax = fig.add_subplot(gs[1, col])
        band = (t_pts[:, across] >= s_lo[across]) & (
            t_pts[:, across] <= s_hi[across]
        )
        t_band = t_pts[band]
        t_band = t_band[_subsample_idx(len(t_band), max_output_points, rng)]
        ax.scatter(t_band[:, axis], t_band[:, 2], s=0.2, c="0.6", lw=0,
                   label=t_lab, rasterized=True)
        ax.scatter(s_sub[:, axis], s_sub[:, 2], s=0.2, c="tab:red", lw=0,
                   label=s_lab, rasterized=True)
        ax.set_xlabel(f"{name} ({units})")
        ax.set_ylabel(f"z ({units})")
        ax.set_title(f"{name}–z profile (target within source {other}-range)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8, markerscale=20)
        ax.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pcd_correspondence(
    src,
    tgt,
    labels=("source", "target"),
    units="m",
    title=None,
    max_sample_points=1_000_000,
    max_output_points=200_000,
    margin=0.05,
    width=14,
    height=14,
    return_stats=False,
    result=None,
):
    """Show how well ``src`` coincides with ``tgt`` in their overlap.

    Both clouds must already be in the same frame. The distances (C2C and the
    vertical offset dz) and their statistics are
    :func:`substrata.measurements.pcd_correspondence`'s; this function draws
    them: top-down maps of C2C and dz, their histograms (with the target point
    spacing in the overlap, which bounds the attainable C2C), and two thin
    cross-sections through the overlap centre with both clouds overlaid.

    Args:
        src: Source point cloud (PointCloud, SimplePointCloud or Open3D).
        tgt: Target point cloud, same frame as ``src``.
        labels: Names used for ``src`` and ``tgt`` in legends.
        units: Coordinate unit label; with ``"m"`` the share of points within
            2/5/10 cm is reported.
        title: Optional figure title.
        max_sample_points: Source points used for the distance statistics.
        max_output_points: Maximum points drawn in the map panels.
        margin: Fraction of the source extent added around it when cropping
            the target.
        width: Figure width in inches.
        height: Figure height in inches.
        return_stats: If True, also return the statistics dict.
        result: The :class:`~substrata.measurements.PcdCorrespondence` to draw
            (default: computed here from ``src`` and ``tgt``).

    Returns:
        matplotlib.figure.Figure, or ``(fig, stats)`` if ``return_stats``.
    """
    from substrata import measurements

    if result is None:
        result = measurements.pcd_correspondence(
            src, tgt, units=units, max_sample_points=max_sample_points,
            margin=margin, labels=labels,
        )
    rng = np.random.default_rng(0)  # display subsamples only
    s_pts, _ = _pcd_xyz_rgb(src)
    s_lab, t_lab = labels
    stats = result.stats
    s_sample, c2c, dz = result.sample, result.c2c, result.dz
    t_crop, spacing = result.target_overlap, result.target_spacing
    lo, hi = result.bbox_min, result.bbox_max
    dz_ok = dz[np.isfinite(dz)]

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 0.9])
    draw = _subsample_idx(len(s_sample), max_output_points, rng)
    xy = s_sample[draw, :2]

    # Top-down maps
    ax = fig.add_subplot(gs[0, 0])
    vmax = float(np.percentile(c2c, 95))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c2c[draw], s=0.3, lw=0,
                    cmap="magma_r", vmin=0, vmax=vmax, rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=f"C2C ({units})")
    ax.set_title("C2C distance source → target")
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")

    ax = fig.add_subplot(gs[0, 1])
    # Robust colour/histogram range: walls and edges give a few large |dz|
    dz_lim = (
        abs(stats["dz_median"]) + 3 * stats["dz_iqr"] if len(dz_ok) else 1.0
    ) or 1.0
    dz_draw = dz[draw]
    ok = np.isfinite(dz_draw)
    ax.scatter(xy[~ok, 0], xy[~ok, 1], s=0.3, lw=0, c="0.85", rasterized=True)
    sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=dz_draw[ok], s=0.3, lw=0,
                    cmap="RdBu_r", vmin=-dz_lim, vmax=dz_lim, rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=f"dz ({units})")
    ax.set_title("Vertical offset dz = z(source) − z(target)  [grey: n/a]")
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")

    # Histograms
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(c2c, bins=100, range=(0, float(np.percentile(c2c, 99))), color="0.4")
    ax.axvline(stats["c2c_median"], color="tab:red", lw=1.5)
    ax.axvline(spacing, color="tab:blue", lw=1, ls="--")
    txt = [
        f"median {stats['c2c_median']:.4f}, p90 {stats['c2c_p90']:.4f} {units}",
        f"{t_lab} spacing in overlap {spacing:.4f} {units} (dashed)",
    ]
    if units == "m":
        txt.append(
            "within 2/5/10 cm: "
            + " / ".join(f"{stats[f'c2c_within_{cm}cm']:.0%}" for cm in (2, 5, 10))
        )
    ax.text(0.98, 0.95, "\n".join(txt), transform=ax.transAxes, ha="right",
            va="top", fontsize=8)
    ax.set_xlabel(f"C2C ({units})")
    ax.set_ylabel("points")

    ax = fig.add_subplot(gs[1, 1])
    lim = 2 * dz_lim
    ax.hist(dz_ok, bins=100, range=(-lim, lim), color="0.4")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(stats["dz_median"], color="tab:red", lw=1.5)
    ax.text(0.98, 0.95,
            f"median {stats['dz_median']:+.4f} {units}\nIQR {stats['dz_iqr']:.4f}"
            f"\ncoverage {stats['dz_coverage']:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlabel(f"dz ({units})")
    ax.set_ylabel("points")

    # Thin cross-sections through the overlap centre
    centre = (lo + hi) / 2
    thickness = max(0.005 * float(np.linalg.norm(hi - lo)), 3 * spacing)
    for col, (along, across, name) in enumerate(((0, 1, "x"), (1, 0, "y"))):
        ax = fig.add_subplot(gs[2, col])
        t_sl = t_crop[np.abs(t_crop[:, across] - centre[across]) < thickness / 2]
        s_sl = s_pts[np.abs(s_pts[:, across] - centre[across]) < thickness / 2]
        t_sl = t_sl[_subsample_idx(len(t_sl), max_output_points, rng)]
        s_sl = s_sl[_subsample_idx(len(s_sl), max_output_points, rng)]
        ax.scatter(t_sl[:, along], t_sl[:, 2], s=0.5, lw=0, c="0.3", label=t_lab,
                   rasterized=True)
        ax.scatter(s_sl[:, along], s_sl[:, 2], s=0.5, lw=0, c="tab:red",
                   label=s_lab, rasterized=True)
        other = "y" if name == "x" else "x"
        ax.set_title(
            f"Cross-section along {name} ({other} = {centre[across]:.2f} "
            f"± {thickness / 2:.3f} {units})"
        )
        ax.set_xlabel(f"{name} ({units})")
        ax.set_ylabel(f"z ({units})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8, markerscale=10)
        ax.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return (fig, stats) if return_stats else fig


def plot_camera_correspondence(
    src_xyz,
    tgt_xyz,
    inliers=None,
    units="m",
    labels=("source", "target"),
    title=None,
    width=12,
    height=6,
    residuals=None,
):
    """Plot residuals between matched camera positions in a common frame.

    Args:
        src_xyz: (N, 3) source camera centres, already transformed into the
            target frame.
        tgt_xyz: (N, 3) corresponding target camera centres.
        inliers: Optional boolean mask of pairs used for the fit; the others
            are drawn as rejected outliers.
        units: Coordinate unit label.
        labels: Names used for source and target.
        title: Optional figure title.
        width: Figure width in inches.
        height: Figure height in inches.
        residuals: The :func:`substrata.measurements.camera_residuals` result
            to draw (default: computed here).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    from substrata import measurements

    tgt_xyz = np.asarray(tgt_xyz, dtype=float)
    if residuals is None:
        residuals = measurements.camera_residuals(src_xyz, tgt_xyz, inliers)
    inl = residuals["inliers"]
    r_in = residuals["residuals"][inl]
    rms = residuals["rms"]

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    sc = ax_map.scatter(tgt_xyz[inl, 0], tgt_xyz[inl, 1], c=r_in, s=12,
                        cmap="viridis", vmin=0, vmax=float(r_in.max()))
    fig.colorbar(sc, ax=ax_map, shrink=0.8, label=f"residual ({units})")
    if (~inl).any():
        ax_map.scatter(tgt_xyz[~inl, 0], tgt_xyz[~inl, 1], marker="x", s=20,
                       c="tab:red", label=f"rejected ({int((~inl).sum())})")
        ax_map.legend(loc="best", fontsize=8)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel(f"x ({units})")
    ax_map.set_ylabel(f"y ({units})")
    ax_map.set_title(f"Matched cameras at {labels[1]} positions")

    ax_hist.hist(r_in, bins=40, color="0.4", label=f"inliers ({int(inl.sum())})")
    ax_hist.axvline(rms, color="tab:red", lw=1.5)
    txt = f"inlier RMS {rms:.4f} {units}\nmedian {residuals['median']:.4f} {units}"
    if (~inl).any():
        txt += (
            f"\nrejected: {residuals['rejected_min']:.3f} to "
            f"{residuals['rejected_max']:.3f} {units}"
        )
    ax_hist.text(0.98, 0.95, txt, transform=ax_hist.transAxes, ha="right",
                 va="top", fontsize=8)
    ax_hist.set_xlabel(f"residual ({units})")
    ax_hist.set_ylabel("cameras")
    ax_hist.set_title("Inlier residuals")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_align_qc_pdf(
    pdf_path,
    summary_text,
    src,
    tgt,
    correspondence,
    labels=("source", "target"),
    units="m",
    cameras=None,
):
    """Write the ``substrata align --pdf`` QC report.

    Pages: the run summary (monospace text), the matched-camera residuals
    (:func:`plot_camera_correspondence`, when ``cameras`` is given), the
    source's location in the target (:func:`plot_pcd_context`) and the
    correspondence in the overlap (:func:`plot_pcd_correspondence`). The
    numbers are the given measurement results.

    Args:
        pdf_path: Output PDF path.
        summary_text: Text of the summary page.
        src: Source cloud in the target frame.
        tgt: Target cloud.
        correspondence: :func:`substrata.measurements.pcd_correspondence` of
            ``src`` and ``tgt``.
        labels: Names of source and target.
        units: Coordinate unit label.
        cameras: ``(src_xyz, tgt_xyz, inliers, residuals)``: matched camera
            centres in the target frame and their
            :func:`substrata.measurements.camera_residuals`, or None.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    figs = []
    fig = plot_text(summary_text, width=11, height=8.5)
    fig.axes[0].texts[0].set_fontfamily("monospace")
    fig.axes[0].texts[0].set_fontsize(9)
    figs.append(fig)
    if cameras is not None:
        cam_src, cam_tgt, inliers, residuals = cameras
        figs.append(plot_camera_correspondence(
            cam_src, cam_tgt, inliers=inliers, units=units, labels=labels,
            title="Matched camera residuals (final transform)",
            residuals=residuals,
        ))
    figs.append(plot_pcd_context(
        src, tgt, labels=labels, units=units, title="Location in target"
    ))
    figs.append(plot_pcd_correspondence(
        src, tgt, labels=labels, units=units,
        title="Correspondence in the overlap", result=correspondence,
    ))
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
