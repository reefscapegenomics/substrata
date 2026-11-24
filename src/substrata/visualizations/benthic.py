"""Benthic-fraction figures.

Top-down maps of the classified samples around a focal point
(``visualize_benthic_fraction``), the same next to their image-match crops
(``visualize_benthic_image_matches``), ``benthic_fraction_figure`` (the QC image
of :func:`substrata.measurements.calc_benthic_fraction`) and the RGB
distribution of a point subset (``visualize_rgb_stats``,
:func:`substrata.measurements.get_rgb_stats`). The fractions come from
:func:`substrata.measurements.benthic_fraction_from_results`; the crops are
classified with :func:`substrata.classification.classify_image_match`.
"""

# Standard Library
from io import BytesIO

# Third-Party Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# Local Modules
from substrata import pointclouds
from substrata.imagecrops import get_crop_img
from substrata.logging import logger


def _benthic_finish(fig, dpi, output_filename, interactive):
    """Save / show / rasterise a benthic figure (shared tail)."""
    if output_filename is not None:
        fig.savefig(output_filename, dpi=dpi, bbox_inches="tight")
        return fig
    elif interactive:
        plt.show()
        return fig
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _smooth_histogram(counts, sigma_bins):
    """Gaussian-smooth histogram *counts* with a kernel *sigma_bins* wide.

    Uses edge-reflected padding so the curve does not sag at 0 and 1, where a
    channel's density is often highest.  ``sigma_bins <= 0`` returns *counts*
    unchanged.

    Args:
        counts: 1-D array of per-bin counts.
        sigma_bins: Gaussian standard deviation expressed in bins.

    Returns:
        numpy.ndarray: Smoothed counts, same length as *counts*.
    """
    counts = np.asarray(counts, dtype=float)
    if sigma_bins is None or sigma_bins <= 0 or counts.size == 0:
        return counts
    radius = max(1, int(round(3.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / float(sigma_bins)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(counts, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def visualize_rgb_stats(
    pcd,
    output_filename=None,
    width=600,
    height=400,
    dpi=100,
    bins=256,
    smoothing=4.0,
    interactive=False,
    font_size=12,
    label_font_size=14,
):
    """Visualize the RGB colour distribution of a point cloud subset.

    Companion to :func:`substrata.measurements.get_rgb_stats`.  Pass any point
    cloud subset (e.g. the points within a radius of an annotation) and it
    plots one smoothed density curve per channel, each drawn and filled in its
    own colour, with a dashed line marking that channel's median as reported by
    ``get_rgb_stats``.

    Every point contributes to the distribution (no decimation), so the medians
    drawn are exactly those ``get_rgb_stats`` returns for the same subset.

    Args:
        pcd: Point cloud with a ``colors`` attribute of RGB values, either in
            [0, 1] or in [0, 255] (0-255 input is normalised automatically).
        output_filename: Optional path to save the figure (png, jpg, pdf, svg).
            When None the figure is returned as an image array instead.
        width: Figure width in pixels (default 600, matching the other
            visualizations).
        height: Figure height in pixels (default 400).
        dpi: Dots per inch used for the figure and any saved output.
        bins: Number of histogram bins per channel used before smoothing
            (default 256).  More bins plus smoothing gives a finer curve.
        smoothing: Width of the Gaussian smoothing kernel, in bins (default
            4.0).  Pass ``0`` to disable smoothing and plot the raw histogram.
        interactive: When True and *output_filename* is None, show the figure
            instead of returning it as an array.
        font_size: Tick-label font size in pixels (default 12). Together with
            *label_font_size* this matches the text of the plotly 3-D
            measurement plots (e.g. :func:`visualize_roughness`) at the same
            figure size. Pixel sizes are converted to points with *dpi*.
        label_font_size: Axis-label font size in pixels (default 14).

    Returns:
        matplotlib.figure.Figure | numpy.ndarray: The figure when saved or
            shown interactively, otherwise the rendered image as an
            ``(H, W, 3)`` uint8 RGB array.

    Raises:
        ValueError: If the point cloud has no colours.
    """
    from substrata import measurements

    colors = np.asarray(getattr(pcd, "colors", None))
    if colors is None or colors.size == 0:
        raise ValueError("Point cloud has no colours to summarise")
    colors = colors.reshape(-1, 3).astype(float)
    # Accept 0-255 input (as elsewhere in this module) and work in [0, 1].
    if colors.max() > 1.0:
        colors = colors / 255.0

    median_r, median_g, median_b, luminance = measurements.get_rgb_stats(pcd)
    medians = (median_r, median_g, median_b)
    if max(medians) > 1.0:  # get_rgb_stats echoes the input scale
        medians = tuple(m / 255.0 for m in medians)
        luminance = luminance / 255.0

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    edges = np.linspace(0.0, 1.0, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for i, colour in enumerate(("red", "green", "blue")):
        counts, _ = np.histogram(colors[:, i], bins=edges)
        curve = _smooth_histogram(counts.astype(float), smoothing)
        ax.fill_between(centres, curve, color=colour, alpha=0.35, linewidth=0)
        ax.plot(centres, curve, color=colour, linewidth=1.5)
        ax.axvline(medians[i], color=colour, linestyle="--", linewidth=1.2)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    px_to_pt = 72.0 / dpi
    ax.set_xlabel("Channel value", fontsize=label_font_size * px_to_pt)
    ax.set_ylabel("Point count", fontsize=label_font_size * px_to_pt)
    ax.tick_params(labelsize=font_size * px_to_pt)
    fig.tight_layout()

    return _benthic_finish(fig, dpi, output_filename, interactive)


def _draw_annulus_context(ax, center, radius_inner, radius_outer, set_limits=True):
    """Draw the focal star + inner/outer radius circles and set axis limits."""
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if center is None:
        return
    cx, cy = float(center[0]), float(center[1])
    ax.scatter(cx, cy, c="black", marker="*", s=180, zorder=6)
    if radius_inner > 0:
        ax.add_patch(mpatches.Circle(
            (cx, cy), radius_inner, fill=False, color="black", lw=1, ls="--",
            zorder=3,
        ))
    if radius_outer > 0:
        ax.add_patch(mpatches.Circle(
            (cx, cy), radius_outer, fill=False, color="black", lw=1, zorder=3,
        ))
    if set_limits and radius_outer > 0:
        pad = radius_outer * 1.1
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)


def _classified_samples(intercepts, results, target_class):
    """Yield (ann, x, y, label, p_target) for matched+classified samples.

    Returns ``(items, n_unmatched)`` where each item is a tuple
    ``(ann, x, y, label, p)``; ``n_unmatched`` counts samples with no
    image-match/classification.
    """
    items, n_un = [], 0
    for ann in intercepts.data.values():
        res = results.get(ann.id)
        if ann.image_match is None or not res or res.get("label") is None:
            n_un += 1
            continue
        label = str(res["label"])
        probs = res.get("probs") or {}
        p = float(probs.get(target_class, 1.0 if label == target_class else 0.0))
        items.append((ann, float(ann.coords[0]), float(ann.coords[1]), label, p))
    return items, n_un


def _estimate_cell_size(positions, fallback):
    """Median nearest-neighbour spacing of ``positions`` (fallback if <2)."""
    if len(positions) < 2:
        return fallback
    pts = np.asarray(positions, dtype=float)
    if len(pts) > 400:
        idx = np.random.default_rng(0).choice(len(pts), 400, replace=False)
        pts = pts[idx]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    val = float(np.median(d.min(axis=1)))
    return val if val > 0 else fallback


def _interacting_label(fraction_interacting, z_colony):
    """Title fragment reporting the height-weighted interacting fraction.

    Returns an empty string when ``fraction_interacting`` is ``None`` (so the
    title is unchanged for callers that do not pass it). The colony base level
    ``z_colony`` is appended when available.
    """
    if fraction_interacting is None:
        return ""
    base = (
        f", base z={z_colony:.2f} m"
        if z_colony is not None and not np.isnan(z_colony) else ""
    )
    return f"\nfraction_interacting={fraction_interacting:.3f}{base}"


def _draw_benthic_fraction_panel(
    ax, fig, intercepts, results, target_class, center, radius_inner,
    radius_outer, background_pcd, background_colors, weighted, point_size,
    bg_point_size, max_output_points, add_colorbar=True, sample_weights=None,
    max_outline_lw=1.8, result=None,
):
    """Draw the classified-samples panel; return a stats dict for titling.

    When ``sample_weights`` (``{intercept_id: weight in [0, 1]}``) is given, each
    classified ``target_class`` dot gets a **black outline whose thickness scales
    with its height weight** (``lw = weight * max_outline_lw``): a clear outline
    means the sample sits at/above the colony base (counts toward the interaction
    cover) and no visible outline means it is well below it (excluded). With
    ``sample_weights=None`` the panel keeps its previous fixed-outline look.
    """
    if background_pcd is not None:
        bpts = np.asarray(background_pcd.points, dtype=float)
        bcols = None
        if background_colors is not None:
            bcols = np.asarray(background_colors, dtype=float)
            if bcols.shape[0] != bpts.shape[0]:
                bcols = None  # misaligned -> ignore
        if len(bpts) > max_output_points:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(bpts), size=max_output_points, replace=False)
            bpts = bpts[idx]
            bcols = bcols[idx] if bcols is not None else None
        if bcols is not None:
            # Brighten the true colours so the overlaid markers read clearly.
            bright = np.clip(bcols[:, :3] * 0.45 + 0.55, 0.0, 1.0)
            ax.scatter(bpts[:, 0], bpts[:, 1], c=bright, s=bg_point_size,
                       rasterized=True, zorder=0)
        else:
            ax.scatter(bpts[:, 0], bpts[:, 1], c="lightgray", s=bg_point_size,
                       rasterized=True, zorder=0)

    items, n_un = _classified_samples(intercepts, results, target_class)
    cls_x = [x for _a, x, _y, _l, _p in items]
    cls_y = [y for _a, _x, y, _l, _p in items]
    cls_p = [p for _a, _x, _y, _l, p in items]

    def _w(ann):
        """Height weight for a sample, defaulting to 1.0 (full outline)."""
        if not sample_weights:
            return 1.0
        val = sample_weights.get(getattr(ann, "id", None), 1.0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 1.0
        return float(val)

    tgt = [(x, y, _w(a)) for a, x, y, lab, _p in items if lab == target_class]
    oth = [(x, y) for _a, x, y, lab, _p in items if lab != target_class]

    mappable = None
    if weighted:
        if cls_x:
            # Fill = P(target); outline thickness = height weight (when given).
            if sample_weights is not None:
                edge, lw = "black", [_w(a) * max_outline_lw for a, *_ in items]
            else:
                edge, lw = "0.35", 0.3
            mappable = ax.scatter(cls_x, cls_y, c=cls_p, cmap="Reds", vmin=0.0,
                                  vmax=1.0, s=point_size, edgecolors=edge,
                                  linewidths=lw, zorder=4)
            if add_colorbar:
                fig.colorbar(mappable, ax=ax, label=f"P({target_class})",
                             fraction=0.046, pad=0.04)
    else:
        if oth:
            ax.scatter([p[0] for p in oth], [p[1] for p in oth], marker="x",
                       c="black", s=point_size, linewidths=1.0, label="other",
                       zorder=4)
        if tgt:
            # Outline thickness encodes the height weight (clear = at/above base,
            # none = excluded); fixed 0.6 when no weights are supplied.
            tgt_lw = (
                [w * max_outline_lw for _x, _y, w in tgt]
                if sample_weights is not None else 0.6
            )
            ax.scatter([p[0] for p in tgt], [p[1] for p in tgt], marker="o",
                       facecolors="red", edgecolors="black", s=point_size,
                       linewidths=tgt_lw, label=str(target_class), zorder=5)
        if tgt or oth:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    _draw_annulus_context(ax, center, radius_inner, radius_outer)

    # The reported numbers: the measurement's result (calc_benthic_fraction),
    # else benthic_fraction_from_results of these classifications
    if result is None:
        from substrata import measurements

        frac, result = measurements.benthic_fraction_from_results(
            results, target_class, weighted
        )
    else:
        frac = result["fraction"]
    n_tgt, n_cls = int(result["n_target"]), int(result["n_classified"])
    if weighted:
        frac_label = f"weighted fraction={frac:.3f}  (mean P({target_class}))"
    else:
        frac_label = f"fraction={frac:.3f}"
    return {
        "n_target": n_tgt, "n_classified": n_cls, "n_unmatched": n_un,
        "frac": frac, "frac_label": frac_label, "mappable": mappable,
    }


def _draw_image_match_panel(
    ax, intercepts, results, target_class, crop_w, crop_h, weighted, cell_size,
    center, radius_inner, radius_outer, focal_image_match=None, thumb_px=64,
    border_lw=3.0,
):
    """Draw the per-sample crops as a non-overlapping grid; return n_shown.

    Each crop is centred on the matched pixel (same as
    :func:`classification.classify_image_match`) and drawn at its sample XY
    position, sized to ``cell_size`` so neighbouring crops tile without
    overlapping. The border encodes the classification (red = target / gray =
    other, or red-intensity = ``P(target_class)`` when ``weighted`` — using the
    **same ``Reds`` mapping as the dots panel**, so ``P=0`` is white). When a
    ``focal_image_match`` is given its crop fills the inner circle to show the
    colony being measured.
    """
    reds = plt.get_cmap("Reds")
    items, _n_un = _classified_samples(intercepts, results, target_class)
    if cell_size is None:
        cell_size = _estimate_cell_size(
            [(x, y) for _a, x, y, _l, _p in items],
            (radius_outer or 1.0) / 8.0,
        )
    s = cell_size * 0.92

    # Show the colony itself inside the inner circle (clipped to it), at the
    # same display scale (px per metre) as the sample crops.
    if focal_image_match is not None and center is not None and radius_inner > 0:
        cx, cy = float(center[0]), float(center[1])
        try:
            colony_px = int(round(crop_w * (2 * radius_inner) / max(cell_size, 1e-6)))
            colony_px = int(np.clip(colony_px, crop_w, 4000))
            colony = get_crop_img(
                focal_image_match.filepath, focal_image_match.x,
                focal_image_match.y, colony_px, colony_px,
            ).convert("RGB")
            colony.thumbnail((400, 400))
            art = ax.imshow(
                np.asarray(colony),
                extent=[cx - radius_inner, cx + radius_inner,
                        cy - radius_inner, cy + radius_inner],
                aspect="auto", zorder=1, interpolation="nearest",
            )
            art.set_clip_path(mpatches.Circle(
                (cx, cy), radius_inner, transform=ax.transData,
            ))
        except (OSError, ValueError):
            pass

    n_shown = 0
    for ann, x, y, label, p in items:
        im = ann.image_match
        try:
            crop = get_crop_img(im.filepath, im.x, im.y, crop_w, crop_h)
            crop = crop.convert("RGB")
            crop.thumbnail((thumb_px, thumb_px))
            thumb = np.asarray(crop)
        except (OSError, ValueError):
            continue
        # Match the dots panel exactly: Reds(p) (P=0 -> white).
        border = reds(p) if weighted else (
            "red" if label == target_class else "0.5"
        )
        ax.imshow(thumb, extent=[x - s / 2, x + s / 2, y - s / 2, y + s / 2],
                  aspect="auto", zorder=2, interpolation="nearest")
        ax.add_patch(mpatches.Rectangle(
            (x - s / 2, y - s / 2), s, s, fill=False, edgecolor=border,
            lw=border_lw, zorder=5,
        ))
        n_shown += 1

    _draw_annulus_context(ax, center, radius_inner, radius_outer)
    return n_shown


def benthic_fraction_figure(
    pcd,
    result,
    intercepts,
    results,
    cam_list=None,
    sample_weights=None,
    crop_size=None,
    sample_spacing=None,
    focal_annotation=None,
    show_image_matches=False,
    intercept_search_radius=None,
    reprojection_threshold_discard=None,
    reprojection_intercept_radius=None,
):
    """The QC image of :func:`substrata.measurements.calc_benthic_fraction`.

    Draws the classified samples on the local neighbourhood (the points within
    the outer radius, in true colour), titled with the reported ``result``:
    :func:`visualize_benthic_fraction`, or with ``show_image_matches`` the
    side-by-side :func:`visualize_benthic_image_matches` (its inner-circle
    inset is the focal annotation's image match; when it has none, one is
    searched for the surface point under the focal point, as the samples were).

    Args:
        pcd: The world-frame point cloud of the measurement.
        result: The measurement's result dict (``center``, ``radius_inner``,
            ``radius_outer``, ``target_class``, ``weighted``, ``fraction``,
            ``n_target``, ``n_classified``, ``fraction_interacting``,
            ``z_colony``).
        intercepts: The sampled ``Annotations``.
        results: ``{id: classification_dict | None}`` of the samples.
        cam_list: Cameras searched for the focal image match.
        sample_weights: ``{intercept_id: height weight}`` (dot outlines).
        crop_size: Classifier crop size (crops panel).
        sample_spacing: Sample grid spacing (crops panel cell size).
        focal_annotation: The measured annotation (its ``image_match``).
        show_image_matches: Draw the crops next to the dots.
        intercept_search_radius: XY radius of the focal surface point search.
        reprojection_threshold_discard: Occlusion threshold of that match.
        reprojection_intercept_radius: Ray-cast radius of that match.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    pts = np.asarray(pcd.points, dtype=float)
    focal_pt = np.asarray(result["center"], dtype=float)
    focal_xy = focal_pt[:2]
    outer_radius = result["radius_outer"]
    # The local neighbourhood (the "simple_pcd" around the colony) within the
    # outer radius gives the same colony-centred context as visualize_tpi
    # rather than rendering the whole model, in true RGB colour.
    neigh_mask = np.linalg.norm(pts[:, :2] - focal_xy, axis=1) <= outer_radius
    neighborhood = pointclouds.SimplePointCloud(pts[neigh_mask])
    try:
        cols = np.asarray(pcd.colors, dtype=float)
        neigh_cols = cols[neigh_mask] if len(cols) == len(pts) else None
    except Exception:  # noqa: BLE001 - colours are optional context.
        neigh_cols = None

    common = dict(
        center=focal_pt, radius_inner=result["radius_inner"],
        radius_outer=outer_radius, weighted=result["weighted"],
        background_pcd=neighborhood, background_colors=neigh_cols,
        sample_weights=sample_weights,
        fraction_interacting=result["fraction_interacting"],
        z_colony=result["z_colony"], result=result,
    )
    if not show_image_matches:
        return visualize_benthic_fraction(
            intercepts, results, result["target_class"], **common
        )

    # The colony's existing image match (from the source annotation) for the
    # inner-circle inset; only searched for when it isn't already there.
    focal_im = getattr(focal_annotation, "image_match", None)
    if focal_im is None:
        try:
            focal_anns = pcd.get_z_intercepts(
                np.array([focal_xy]), intercept_search_radius
            )
            for fann in focal_anns.data.values():
                fann.get_image_matches(
                    cam_list, max_cams=1, pcd=pcd, use_orig_coords=False,
                    intercept_radius=reprojection_intercept_radius,
                    reprojection_threshold_discard=reprojection_threshold_discard,
                )
            fvals = list(focal_anns.data.values())
            focal_im = fvals[0].image_match if fvals else None
        except Exception as e:  # noqa: BLE001 - colony inset is best-effort.
            logger.debug("focal image match failed: %s", e)
    return visualize_benthic_image_matches(
        intercepts, results, result["target_class"], crop_size,
        focal_image_match=focal_im, cell_size=sample_spacing, **common
    )


def visualize_benthic_fraction(
    intercepts,
    results,
    target_class,
    center=None,
    radius_inner=0.0,
    radius_outer=0.0,
    background_pcd=None,
    background_colors=None,
    weighted=False,
    output_filename=None,
    max_output_points=50000,
    width=700,
    height=600,
    point_size=145,
    bg_point_size=1.5,
    interactive=False,
    sample_weights=None,
    fraction_interacting=None,
    z_colony=None,
    result=None,
):
    """Top-down visualisation of a benthic-fraction sampling (see
    :func:`measurements.calc_benthic_fraction`).

    Mirrors :func:`visualize_tpi`'s colony-centred top-down view: the local
    neighbourhood point cloud (the ``simple_pcd`` around the colony) in its
    **true (brightened) RGB colours** as small context dots, the focal point as
    a star and the inner/outer annulus radii as circles. The classified sample
    points are overlaid: with ``weighted=False`` ``target_class`` as **red
    circles** and others as **black crosses**; with ``weighted=True`` each as a
    circle whose **red intensity encodes** ``P(target_class)`` (with a colour
    bar).

    Args:
        intercepts: ``Annotations`` of the sampled ``InterceptAnnotation`` objects.
        results: ``{id: classification_dict | None}`` from
            :meth:`Annotations.classify_image_matches`.
        target_class: Class label highlighted as the target.
        center: (3,) focal point (star + radius circles drawn when given).
        radius_inner: Inner annulus radius in metres (dashed circle).
        radius_outer: Outer annulus radius in metres (solid circle).
        background_pcd: Local neighbourhood point cloud (colony ``simple_pcd``).
        background_colors: Optional (N, 3) per-point RGB in [0, 1].
        weighted: Colour classified points by ``P(target_class)``.
        output_filename: Optional path to save the figure instead of returning.
        max_output_points: Max background points to render (decimated above).
        width: Figure width in pixels (at 100 dpi).
        height: Figure height in pixels (at 100 dpi).
        point_size: Scatter marker size for the sample points.
        bg_point_size: Scatter marker size for the background cloud points.
        interactive: If True and no output file, display interactively.
        sample_weights: Optional ``{intercept_id: height weight in [0, 1]}``. When
            given, each ``target_class`` dot's black-outline thickness encodes its
            weight (clear outline = sand at/above the colony base, no outline =
            well below it / excluded from ``fraction_interacting``).
        fraction_interacting: Optional height-weighted interacting fraction to
            report in the title (the ``fraction_interacting`` returned by
            :func:`measurements.calc_benthic_fraction`).
        z_colony: Optional colony base level (m) shown alongside it.
        result: The ``calc_benthic_fraction`` result dict (its ``fraction``,
            ``n_target`` and ``n_classified`` are shown). If None,
            :func:`substrata.measurements.benthic_fraction_from_results` of
            ``results``.

    Returns:
        matplotlib.figure.Figure | np.ndarray: Figure when ``interactive=True``
        or ``output_filename`` is set; otherwise an (H, W, 3) uint8 RGB array.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    stats = _draw_benthic_fraction_panel(
        ax, fig, intercepts, results, target_class, center, radius_inner,
        radius_outer, background_pcd, background_colors, weighted, point_size,
        bg_point_size, max_output_points, sample_weights=sample_weights,
        result=result,
    )
    outline_note = (
        "\noutline thickness = height weight (thick = at/above colony base)"
        if sample_weights is not None else ""
    )
    ax.set_title(
        f"Benthic fraction: {target_class}\n"
        f"{stats['frac_label']}{_interacting_label(fraction_interacting, z_colony)}"
        f"  (n_target={stats['n_target']}, "
        f"n_classified={stats['n_classified']}, "
        f"n_unmatched={stats['n_unmatched']})\n"
        f"r_inner={radius_inner:.3f} m  r_outer={radius_outer:.3f} m"
        f"{outline_note}",
        fontsize=9,
    )
    fig.tight_layout()
    return _benthic_finish(fig, dpi, output_filename, interactive)


def visualize_benthic_image_matches(
    intercepts,
    results,
    target_class,
    crop_size,
    center=None,
    radius_inner=0.0,
    radius_outer=0.0,
    weighted=False,
    background_pcd=None,
    background_colors=None,
    focal_image_match=None,
    cell_size=None,
    thumb_px=64,
    border_lw=3.0,
    max_output_points=50000,
    point_size=145,
    bg_point_size=1.5,
    width=1500,
    height=750,
    output_filename=None,
    interactive=False,
    sample_weights=None,
    fraction_interacting=None,
    z_colony=None,
    result=None,
):
    """Side-by-side comparison of the benthic-fraction dots and their crops.

    Left panel: the :func:`visualize_benthic_fraction` view (classified sample
    dots over the colony point cloud). Right panel, **same size and same axis
    extent**: each sample's classifier-input crop (centred on the matched pixel
    ``(image_match.x, image_match.y)`` — the exact crop
    :func:`classification.classify_image_match` uses) placed at the **same XY
    position** as its dot, resized to ``cell_size`` so the crops tile into a
    non-overlapping grid. Each crop has a thick border encoding the
    classification: ``weighted=False`` -> **red** for ``target_class`` / **gray**
    for other; ``weighted=True`` -> border **red intensity** = ``P(target_class)``.

    This makes it easy to compare, point for point, the fraction map against the
    actual image content the classifier judged — telling a classifier error (the
    crop clearly shows the target but is mislabelled) from a method error (the
    crop is not on the sampled feature).

    Args:
        intercepts: ``Annotations`` of the sampled intercepts with
            ``.image_match`` populated.
        results: ``{id: classification_dict | None}`` (carries ``probs``).
        target_class: Target class label.
        crop_size: Classifier crop size in pixels (int or ``(w, h)``).
        center: (3,) focal point.
        radius_inner: Inner annulus radius in metres (dashed circle).
        radius_outer: Outer annulus radius in metres (solid circle).
        weighted: Encode ``P(target_class)`` as the dot/border red intensity.
        background_pcd: Local neighbourhood point cloud for the left panel.
        background_colors: Optional (N, 3) per-point RGB in [0, 1].
        focal_image_match: Optional ImageMatch of the colony itself; its crop
            fills the right panel's inner circle.
        cell_size: Grid cell size (metres) for the crops; defaults to the median
            sample spacing so crops tile without overlap.
        thumb_px: Downscaled crop thumbnail size in pixels.
        border_lw: Crop border line width.
        max_output_points: Max background points to render (decimated above).
        point_size: Left-panel sample marker size.
        bg_point_size: Background cloud point size.
        width: Figure width in pixels (at 100 dpi).
        height: Figure height in pixels (at 100 dpi).
        output_filename: Optional path to save the figure instead of returning.
        interactive: If True and no output file, display interactively.
        sample_weights: Optional ``{intercept_id: height weight in [0, 1]}``;
            scales each left-panel ``target_class`` dot's black-outline thickness
            (clear = sand at/above the colony base, none = excluded).
        fraction_interacting: Optional height-weighted interacting fraction shown
            in the left-panel title.
        z_colony: Optional colony base level (m) shown alongside it.
        result: The ``calc_benthic_fraction`` result dict (its ``fraction``,
            ``n_target`` and ``n_classified`` are shown). If None,
            :func:`substrata.measurements.benthic_fraction_from_results` of
            ``results``.

    Returns:
        matplotlib.figure.Figure | np.ndarray: Figure when ``interactive=True``
        or ``output_filename`` is set; otherwise an (H, W, 3) uint8 RGB array.
    """
    if isinstance(crop_size, int):
        crop_w = crop_h = crop_size
    else:
        crop_w, crop_h = crop_size
    dpi = 100
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(width / dpi, height / dpi), dpi=dpi,
    )
    # Draw without a per-axes colour bar so the two panels stay the same size;
    # a single colour bar spanning both is added afterwards (weighted mode).
    stats = _draw_benthic_fraction_panel(
        ax_l, fig, intercepts, results, target_class, center, radius_inner,
        radius_outer, background_pcd, background_colors, weighted, point_size,
        bg_point_size, max_output_points, add_colorbar=False,
        sample_weights=sample_weights,
        result=result,
    )
    outline_note = (
        "\noutline thickness = height weight (thick = at/above colony base)"
        if sample_weights is not None else ""
    )
    ax_l.set_title(
        f"Classified samples\n{stats['frac_label']}"
        f"{_interacting_label(fraction_interacting, z_colony)}  "
        f"(n_target={stats['n_target']}, n_classified={stats['n_classified']}, "
        f"n_unmatched={stats['n_unmatched']}){outline_note}",
        fontsize=9,
    )
    n_shown = _draw_image_match_panel(
        ax_r, intercepts, results, target_class, crop_w, crop_h, weighted,
        cell_size, center, radius_inner, radius_outer,
        focal_image_match=focal_image_match, thumb_px=thumb_px,
        border_lw=border_lw,
    )
    border_desc = (
        f"border intensity = P({target_class}) (white = 0)" if weighted
        else f"red = {target_class}, gray = other"
    )
    ax_r.set_title(
        f"Image-match crops fed to the classifier ({n_shown} shown)\n"
        f"{border_desc}",
        fontsize=9,
    )
    if weighted and stats.get("mappable") is not None:
        # Steal equally from both axes so they remain the same height.
        fig.colorbar(stats["mappable"], ax=[ax_l, ax_r],
                     label=f"P({target_class})", fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Benthic fraction: {target_class}  "
        f"(r_inner={radius_inner:.3f} m, r_outer={radius_outer:.3f} m)",
        fontsize=11,
    )
    if not (weighted and stats.get("mappable") is not None):
        fig.tight_layout()
    return _benthic_finish(fig, dpi, output_filename, interactive)
