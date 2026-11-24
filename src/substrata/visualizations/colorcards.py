"""Colour-calibration figures.

The colour-checker card (plane-normal view and measured-vs-reference grid), the
effect of the affine colour correction and before/after point-cloud scatters,
plus ``color_calibration_figures`` / ``save_color_calibration_pdf`` behind
``ColorCalibrations.show`` / ``save_pdf``. The numbers come from
:mod:`substrata.color_calibration` (``correction_residuals``,
``fit_plane_from_corners``, ``plane_signed_distances``).
"""

# Standard Library
import os
from typing import Any, Dict, List, Optional

# Third-Party Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Local Modules
from substrata import settings, pointclouds


def _render_plane_view_on_ax(
    ax: Any,
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
    br: np.ndarray,
    patch_centers_3d: np.ndarray,
    patch_names: List[str],
    radius: float,
    title: Optional[str] = None,
    point_size: float = 0.3,
    margin_frac: float = 0.15,
) -> None:
    """Render the plane-normal view onto an existing axes object."""
    tl = np.asarray(tl, dtype=float).reshape(3)
    tr = np.asarray(tr, dtype=float).reshape(3)
    bl = np.asarray(bl, dtype=float).reshape(3)
    br = np.asarray(br, dtype=float).reshape(3)
    origin = tl.copy()

    u_axis = tr - tl
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-15)

    v_axis = bl - tl
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-15)

    pts = np.asarray(pcd_points, dtype=float)
    cols = np.asarray(pcd_colors, dtype=float)
    delta = pts - origin.reshape(1, 3)
    proj_u = delta @ u_axis
    proj_v = delta @ v_axis

    corners_2d = np.array(
        [
            [0.0, 0.0],
            [np.dot(tr - origin, u_axis), np.dot(tr - origin, v_axis)],
            [np.dot(bl - origin, u_axis), np.dot(bl - origin, v_axis)],
            [np.dot(br - origin, u_axis), np.dot(br - origin, v_axis)],
        ]
    )
    u_min, v_min = corners_2d.min(axis=0)
    u_max, v_max = corners_2d.max(axis=0)
    mu = (u_max - u_min) * margin_frac
    mv = (v_max - v_min) * margin_frac

    mask = (
        (proj_u >= u_min - mu)
        & (proj_u <= u_max + mu)
        & (proj_v >= v_min - mv)
        & (proj_v <= v_max + mv)
    )
    proj_u = proj_u[mask]
    proj_v = proj_v[mask]
    cols = cols[mask]

    ax.set_aspect("equal")
    ax.scatter(
        proj_u,
        proj_v,
        c=np.clip(cols, 0, 1),
        s=point_size,
        edgecolor="none",
    )

    centres = np.asarray(patch_centers_3d, dtype=float)
    delta_c = centres - origin.reshape(1, 3)
    cu = delta_c @ u_axis
    cv = delta_c @ v_axis

    for i in range(len(cu)):
        circle = plt.Circle(
            (cu[i], cv[i]),
            radius,
            fill=False,
            edgecolor="red",
            linewidth=2.5,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.text(
            cu[i],
            cv[i] - radius * 1.3,
            patch_names[i],
            fontsize=4,
            ha="center",
            va="bottom",
            color="red",
        )

    ax.set_xlim(u_min - mu, u_max + mu)
    ax.set_ylim(v_min - mv, v_max + mv)
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("u  (TL → TR)")
    ax.set_ylabel("v  (TL → BL)")
    ax.set_rasterized(True)


def _render_comparison_grid_on_ax(
    ax: Any,
    patch_results: List[Dict[str, Any]],
    outlier_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    """Render a 6x4 measured-vs-reference colour grid onto an existing axes.

    Each cell is filled with the measured median RGB; a thick border shows
    the reference sRGB. Outlier patches get a red cross overlay.
    """
    n_cols, n_rows = 6, 4
    border_w = 0.4

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)

    inset = border_w / 2
    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        if idx >= len(patch_results):
            continue
        pr = patch_results[idx]
        ref = np.clip(np.asarray(pr["ref_rgb_255"], dtype=float) / 255.0, 0, 1)

        ax.add_patch(
            mpatches.Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=ref,
                edgecolor="none",
            )
        )

        med = pr.get("median_rgb_0_1")
        if med is not None:
            meas = np.clip(np.asarray(med, dtype=float), 0, 1)
            ax.add_patch(
                mpatches.Rectangle(
                    (col - 0.5 + inset, row - 0.5 + inset),
                    1.0 - 2 * inset,
                    1.0 - 2 * inset,
                    facecolor=meas,
                    edgecolor="none",
                )
            )
        else:
            ax.add_patch(
                mpatches.Rectangle(
                    (col - 0.5 + inset, row - 0.5 + inset),
                    1.0 - 2 * inset,
                    1.0 - 2 * inset,
                    facecolor="white",
                    edgecolor="0.4",
                    linewidth=0.5,
                )
            )
            x0 = col - 0.5 + inset
            y0 = row - 0.5 + inset
            s = 1.0 - 2 * inset
            ax.plot(
                [x0, x0 + s],
                [y0, y0 + s],
                color="0.4",
                linewidth=1.2,
                clip_on=True,
            )
            ax.plot(
                [x0 + s, x0],
                [y0, y0 + s],
                color="0.4",
                linewidth=1.2,
                clip_on=True,
            )

        if outlier_mask is not None and idx < len(outlier_mask) and outlier_mask[idx]:
            x0, y0 = col - 0.5, row - 0.5
            ax.plot(
                [x0, x0 + 1],
                [y0, y0 + 1],
                color="red",
                linewidth=2.5,
                clip_on=True,
            )
            ax.plot(
                [x0 + 1, x0],
                [y0, y0 + 1],
                color="red",
                linewidth=2.5,
                clip_on=True,
            )


def plot_colorchecker_card_summary(
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
    br: np.ndarray,
    patch_centers_3d: np.ndarray,
    patch_names: List[str],
    patch_results: List[Dict[str, Any]],
    radius: float,
    outlier_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    point_size: float = 0.3,
    margin_frac: float = 0.15,
    width: int = 16,
    height: int = 5,
) -> Any:
    """Side-by-side plane-normal view (left) and colour comparison grid (right).

    Args:
        pcd_points: (N, 3) point positions.
        pcd_colors: (N, 3) colours in 0-1 range.
        tl, tr, bl, br: 3D corner positions of the marker quad.
        patch_centers_3d: (M, 3) world positions of each patch centre.
        patch_names: Length-M list of patch labels.
        patch_results: Per-patch dicts with ``median_rgb_0_1`` and ``ref_rgb_255``.
        radius: Sampling radius drawn as circles.
        outlier_mask: Boolean array, length ``len(patch_results)``.
            True marks the patch as an outlier (red cross overlay).
        title: Overall figure title.
        point_size: Scatter marker size for point cloud.
        margin_frac: Margin around corner bounds.
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        matplotlib Figure.
    """
    fig, (ax_plane, ax_grid) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    _render_plane_view_on_ax(
        ax_plane,
        pcd_points,
        pcd_colors,
        tl,
        tr,
        bl,
        br,
        patch_centers_3d,
        patch_names,
        radius,
        title="Plane-normal view",
        point_size=point_size,
        margin_frac=margin_frac,
    )

    _render_comparison_grid_on_ax(
        ax_grid,
        patch_results,
        outlier_mask=outlier_mask,
        title="Measured (inner) vs reference (border)",
    )

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def _render_swatch_grid_on_ax(
    ax: Any,
    colors_255: np.ndarray,
    patch_names: List[str],
    outlier_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    """Render a plain 6x4 colour swatch grid onto an existing axes.

    Args:
        colors_255: (24, 3) array of RGB values in 0-255.
        patch_names: Length-24 list of patch labels.
        outlier_mask: Optional boolean array; True draws a red cross.
        title: Axes title.
    """
    n_cols, n_rows = 6, 4
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        if idx >= len(colors_255):
            continue
        raw = np.asarray(colors_255[idx], dtype=float)
        if not np.all(np.isfinite(raw)):
            ax.add_patch(
                mpatches.Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor="white",
                    edgecolor="0.4",
                    linewidth=0.5,
                )
            )
            ax.plot(
                [col - 0.5, col + 0.5],
                [row - 0.5, row + 0.5],
                color="0.4",
                linewidth=1.2,
            )
            ax.plot(
                [col + 0.5, col - 0.5],
                [row - 0.5, row + 0.5],
                color="0.4",
                linewidth=1.2,
            )
            ax.text(
                col,
                row,
                patch_names[idx],
                ha="center",
                va="center",
                fontsize=5,
                color="0.4",
            )
            continue
        c = np.clip(raw / 255.0, 0, 1)
        ax.add_patch(
            mpatches.Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=c,
                edgecolor="grey",
                linewidth=0.3,
            )
        )

        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        txt_col = "white" if lum < 0.45 else "black"
        ax.text(
            col,
            row,
            patch_names[idx],
            ha="center",
            va="center",
            fontsize=5,
            color=txt_col,
        )

        if outlier_mask is not None and idx < len(outlier_mask) and outlier_mask[idx]:
            x0, y0 = col - 0.5, row - 0.5
            ax.plot([x0, x0 + 1], [y0, y0 + 1], color="red", linewidth=2.5)
            ax.plot([x0 + 1, x0], [y0, y0 + 1], color="red", linewidth=2.5)


def plot_color_correction_summary(
    measured_rgb_255: np.ndarray,
    reference_rgb_255: np.ndarray,
    correction: Dict[str, np.ndarray],
    patch_names: List[str],
    outlier_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    width: int = 16,
    height: int = 10,
    residuals: Optional[Dict[str, Any]] = None,
) -> Any:
    """Summary figure showing the effect of the affine colour correction.

    Top row: three 6x4 swatch grids (Measured / Corrected / Reference).
    Bottom row: per-channel scatter plots (R, G, B) with before and after.

    Args:
        measured_rgb_255: (N, 3) measured median RGB in 0-255.
        reference_rgb_255: (N, 3) reference sRGB in 0-255.
        correction: Dict with ``"matrix"`` (3x3) and ``"offset"`` (3,).
        patch_names: Patch label strings.
        outlier_mask: Optional boolean array for the measured grid (patches
            left out of the fit).
        title: Overall figure title.
        width: Figure width in inches.
        height: Figure height in inches.
        residuals: The fit's
            :func:`substrata.color_calibration.correction_residuals` (the
            per-channel RMSE in the panel titles). If None, computed here over
            the patches not in ``outlier_mask``.

    Returns:
        matplotlib Figure.
    """
    from substrata import color_calibration

    if residuals is None:
        fit_mask = np.isfinite(np.asarray(measured_rgb_255, dtype=float)).all(axis=1)
        if outlier_mask is not None:
            fit_mask &= ~np.asarray(outlier_mask, dtype=bool)
        residuals = color_calibration.correction_residuals(
            measured_rgb_255, reference_rgb_255, correction, fit_mask
        )
    matrix = np.asarray(correction["matrix"], dtype=float)
    offset = np.asarray(correction["offset"], dtype=float)
    corrected = np.clip(measured_rgb_255 @ matrix.T + offset, 0, 255)

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)

    ax_meas = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_ref = fig.add_subplot(gs[0, 2])

    _render_swatch_grid_on_ax(
        ax_meas,
        measured_rgb_255,
        patch_names,
        outlier_mask=outlier_mask,
        title="Measured",
    )
    _render_swatch_grid_on_ax(
        ax_corr,
        corrected,
        patch_names,
        title="Corrected",
    )
    _render_swatch_grid_on_ax(
        ax_ref,
        reference_rgb_255,
        patch_names,
        title="Reference",
    )

    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["tab:red", "tab:green", "tab:blue"]
    for ch in range(3):
        ax = fig.add_subplot(gs[1, ch])
        ref_ch = reference_rgb_255[:, ch]
        meas_ch = measured_rgb_255[:, ch]
        corr_ch = corrected[:, ch]

        ax.scatter(
            ref_ch,
            meas_ch,
            s=30,
            alpha=0.4,
            color=channel_colors[ch],
            label="Before",
            edgecolors="none",
        )
        ax.scatter(
            ref_ch,
            corr_ch,
            s=30,
            marker="D",
            color=channel_colors[ch],
            label="After",
            edgecolors="black",
            linewidths=0.3,
        )

        ax.plot([0, 255], [0, 255], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, 260)
        ax.set_ylim(0, 260)
        ax.set_xlabel("Reference", fontsize=8)
        ax.set_ylabel("Measured / Corrected", fontsize=8)
        ax.tick_params(labelsize=7)

        rmse_before = float(residuals["rmse_before_per_channel"][ch])
        rmse_after = float(residuals["rmse_per_channel"][ch])
        ax.set_title(
            f"{channel_names[ch]}  (RMSE {rmse_before:.1f} -> {rmse_after:.1f})",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="upper left")

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_color_before_after(
    pcd: Any,
    correction: Dict[str, np.ndarray],
    max_output_points: int = 500_000,
    point_size: float = 1,
    width: int = 16,
    height: int = 6,
    title: Optional[str] = None,
) -> Any:
    """Side-by-side 2D scatter of the point cloud before and after colour correction.

    A small decimated copy is created so memory stays low.

    Args:
        pcd: PointCloud (substrata or Open3D).
        correction: Dict with ``"matrix"`` (3x3) and ``"offset"`` (3,).
        max_output_points: Number of points in the decimated preview.
        point_size: Scatter marker size.
        width: Figure width in inches.
        height: Figure height in inches.
        title: Optional suptitle.

    Returns:
        matplotlib Figure.
    """
    dec = pointclouds.get_decimated_pcd(pcd, max_output_points)
    pts = dec.points
    colors_before = np.clip(dec.colors, 0, 1)

    matrix = np.asarray(correction["matrix"], dtype=float)
    offset = np.asarray(correction["offset"], dtype=float)
    corrected_255 = np.clip(dec.colors * 255.0 @ matrix.T + offset, 0, 255)
    colors_after = corrected_255 / 255.0

    fig, (ax_before, ax_after) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        sharex=True,
        sharey=True,
    )
    for ax, cols, label in [
        (ax_before, colors_before, "Before correction"),
        (ax_after, colors_after, "After correction"),
    ]:
        ax.set_aspect("equal")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=cols,
            s=point_size,
            edgecolor="none",
        )
        ax.set_title(label, fontsize=10)
        ax.set_rasterized(True)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def color_calibration_figures(
    cc,
    pcd,
    margin_frac=0.15,
    point_size=25,
    plane_thickness=0.01,
    width=16,
    height=5,
):
    """QC figures of a sampled :class:`~substrata.color_calibration.ColorCalibrations`.

    Per valid card: the face-on view of the cloud around the card with the
    sampling circles, next to the measured patch colours framed by the
    reference (:func:`plot_colorchecker_card_summary`). With a fitted
    correction also :func:`plot_color_correction_summary` (titled with the
    fit's ``fit_stats``, panels with its ``fit_residuals``) and
    :func:`plot_color_before_after`. ``ColorCalibrations.show`` samples the
    cloud and fits the correction first when needed.

    Args:
        cc: The ``ColorCalibrations`` (already sampled).
        pcd: PointCloud with colours.
        margin_frac: Extra margin around corners as fraction of card extent.
        point_size: Scatter marker size for the projected points.
        plane_thickness: Only points within this distance (metres) of the card
            plane are shown.
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        list: matplotlib figures.
    """
    from substrata import color_calibration

    sample_r = float(settings.DEFAULT_COLOR_CALIBRATION_RADIUS)
    aggregating = set(cc._aggregation_card_indices or ())

    figs: List[Any] = []
    for i, card in enumerate(cc.data):
        if i in cc._excluded_card_indices:
            continue
        if not card.has_all_corners() or not card.patch_results:
            continue

        corners = np.array([
            card.tl_coords, card.tr_coords,
            card.bl_coords, card.br_coords,
        ])
        centre = corners.mean(axis=0)
        diag = np.max(np.linalg.norm(corners - centre, axis=1))
        fetch_radius = diag * (1.0 + margin_frac)

        sub = pcd.subsample_pointcloud_by_radius(centre, fetch_radius)
        pts = np.asarray(sub.points)
        cols = np.asarray(getattr(sub, "colors", np.ones_like(pts)))
        if pts.shape[0] == 0:
            continue

        normal, origin = color_calibration.fit_plane_from_corners(
            card.tl_coords, card.tr_coords, card.bl_coords,
        )
        signed_dist = color_calibration.plane_signed_distances(pts, origin, normal)
        slab_mask = np.abs(signed_dist) <= plane_thickness
        pts = pts[slab_mask]
        cols = cols[slab_mask]
        if pts.shape[0] == 0:
            continue

        patch_centers = np.array([
            pr["world_xyz"] for pr in card.patch_results
        ])
        patch_names = [pr["name"] for pr in card.patch_results]
        status = "" if i in aggregating else "  [EXCLUDED: missing patches]"
        card_title = (
            f"Card {i + 1}: "
            f"{card.name or card.tl_label + ' … ' + card.br_label}"
            f"{status}"
        )

        om = (
            cc.outlier_mask[i]
            if cc.outlier_mask is not None
            else np.zeros(len(card.patch_results), dtype=bool)
        )
        fig = plot_colorchecker_card_summary(
            pcd_points=pts,
            pcd_colors=cols,
            tl=card.tl_coords,
            tr=card.tr_coords,
            bl=card.bl_coords,
            br=card.br_coords,
            patch_centers_3d=patch_centers,
            patch_names=patch_names,
            patch_results=card.patch_results,
            outlier_mask=om,
            radius=sample_r,
            title=card_title,
            point_size=point_size,
            margin_frac=margin_frac,
            width=width,
            height=height,
        )
        figs.append(fig)

    if cc.color_correction is not None and cc.median_rgb_255_per_patch is not None:
        ref = np.array(
            [[r, g, b] for (_, _, _, r, g, b) in cc.patch_definitions],
            dtype=float,
        )
        names = [name for (name, *_rest) in cc.patch_definitions]
        global_om = None
        if cc.patch_fit_mask is not None:
            global_om = ~np.asarray(cc.patch_fit_mask, dtype=bool)
        title = "Colour Correction Summary"
        if cc.fit_stats:
            fs = cc.fit_stats
            title += (f" (fit RMSE {fs['rmse']:.2f} RGB units, "
                      f"{fs['n_patches_fit']}/{fs['n_patches']} patches, "
                      f"{fs['space']})")
        figs.append(plot_color_correction_summary(
            measured_rgb_255=cc.median_rgb_255_per_patch,
            reference_rgb_255=ref,
            correction=cc.color_correction,
            patch_names=names,
            outlier_mask=global_om,
            title=title,
            residuals=getattr(cc, "fit_residuals", None),
        ))
        figs.append(plot_color_before_after(
            pcd=pcd,
            correction=cc.color_correction,
            title="Point Cloud Before / After Colour Correction",
        ))

    return figs


def save_color_calibration_pdf(cc, pcd, filepath=None):
    """Save ``cc.show(pcd)``'s figures to a PDF with a non-interactive backend.

    Default path: ``<cloud base>_color_calibration.pdf``.
    """
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages

    backend_original = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        if filepath is None:
            base, _ext = os.path.splitext(pcd.filepath)
            filepath = f"{base}_color_calibration.pdf"
        with PdfPages(filepath) as pdf:
            for fig in cc.show(pcd):
                pdf.savefig(fig)
    finally:
        matplotlib.use(backend_original, force=True)
