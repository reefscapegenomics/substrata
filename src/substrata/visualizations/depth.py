"""Depth-regression figures.

Actual vs predicted depths and their residuals (``plot_depth_regression``, errors
from :func:`substrata.measurements.regression_errors`), camera residuals in X-Y
and X-Z (``plot_cam_residuals``) and the figures / PDF of a
:class:`substrata.measurements.DepthResidualAnalyzer`.
"""

# Standard Library
import os

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np


def plot_cam_residuals(cams, depths, est_depths, width=10, height=5):
    """Plot camera residuals with two side-by-side views: X–Y and X–Z.

    - Points are colored by depth residual (blue–white–red, symmetric limits).
    - Marker shapes encode Camera.group.
    - Marker fill/shape encodes depth accuracy bins.

    Accuracy encoding (if cam.depth_acc present):
        • Best (lowest bin): filled markers
        • Mid bin: hollow markers (facecolor='none')
        • Worst bin: 'x' markers

    Args:
        cams: Camera collection with data attribute containing cameras
        width (float): Base width per panel (final fig is ~2×width)
        height (float): Figure height in inches
    """
    cams_all = [
        cam
        for cam in cams.data.values()
        if hasattr(cam, "coords") and cam.coords is not None
    ]
    if not cams_all:
        raise ValueError("No cameras with coords found.")

    depths = np.asarray(depths, dtype=float)
    est_depths = np.asarray(est_depths, dtype=float)
    if depths.shape != est_depths.shape:
        raise ValueError("depths and est_depths must have the same shape")
    if depths.ndim != 1:
        raise ValueError("depths and est_depths must be 1-D arrays")
    if depths.size != len(cams_all):
        raise ValueError(
            f"depths/est_depths length ({depths.size}) must match number of cams with coords ({len(cams_all)})"
        )

    residuals_full = depths - est_depths
    # Finite residuals considered available; others treated as missing
    cams_res = []
    residuals = []
    cams_nores = []
    for cam, res in zip(cams_all, residuals_full):
        if np.isfinite(res):
            cams_res.append(cam)
            residuals.append(float(res))
        else:
            cams_nores.append(cam)
    residuals = np.asarray(residuals, dtype=float)

    # Color mapping by residuals (symmetric around 0)
    if cams_res:
        max_abs_res = float(np.max(np.abs(residuals))) if residuals.size else 1.0
        max_abs_res = max(max_abs_res, 1e-9)
        norm = plt.Normalize(vmin=-max_abs_res, vmax=max_abs_res)
    else:
        residuals = np.array([])
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.cm.bwr

    # Determine camera groups (for fill style)
    try:
        group_names = (
            cams.group_names
            if hasattr(cams, "group_names")
            else sorted({getattr(cam, "group", None) for cam in cams_all})
        )
        group_names = [g for g in group_names if g is not None]
    except Exception:
        group_names = []
    group_to_fill = {
        g: (i % 2 == 0) for i, g in enumerate(sorted(group_names))
    }  # True → filled, False → hollow

    # Build numerical accuracy groups (rounded) for cams with residuals
    def _round_acc(val: float) -> float | None:
        return float(np.round(val, 6)) if np.isfinite(val) else None

    acc_values_res = sorted(
        {
            _round_acc(getattr(cam, "depth_acc", np.nan))
            for cam in cams_res
            if np.isfinite(getattr(cam, "depth_acc", np.nan))
        }
    )
    acc_present = len(acc_values_res) > 0
    best_acc_value = acc_values_res[0] if acc_present else None

    # Marker shapes per accuracy value
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*", "h"]
    acc_to_marker = {
        acc: marker_cycle[i % len(marker_cycle)] for i, acc in enumerate(acc_values_res)
    }
    default_marker = "o"

    # Helper to plot into an axis given x and y indices (0=x,1=y,2=z)
    def plot_axis(ax, x_idx: int, y_idx: int):
        # Plot with residuals for cams_res
        if cams_res:
            # Organize by (accuracy_marker, group_fill)
            by_key = {}
            for cam, res in zip(cams_res, residuals):
                g = getattr(cam, "group", None)
                filled = group_to_fill.get(g, True)
                acc_val = _round_acc(getattr(cam, "depth_acc", np.nan))
                marker = acc_to_marker.get(acc_val, default_marker)
                key = (marker, filled)
                arr = by_key.setdefault(key, {"x": [], "y": [], "c": []})
                arr["x"].append(cam.coords[x_idx])
                arr["y"].append(cam.coords[y_idx])
                arr["c"].append(res)

            for (marker, filled), data in by_key.items():
                xs = np.array(data["x"], dtype=float)
                ys = np.array(data["y"], dtype=float)
                cs = np.array(data["c"], dtype=float)
                if filled:
                    ax.scatter(
                        xs,
                        ys,
                        c=cs,
                        cmap=cmap,
                        norm=norm,
                        marker=marker,
                        edgecolor="black",
                        linewidths=0.7,
                    )
                else:
                    ax.scatter(
                        xs,
                        ys,
                        facecolors="none",
                        edgecolors=cmap(norm(cs)),
                        marker=marker,
                        linewidths=1.0,
                    )

        # Cameras without residuals shown in gray
        if cams_nores:
            xs = np.array([cam.coords[x_idx] for cam in cams_nores], dtype=float)
            ys = np.array([cam.coords[y_idx] for cam in cams_nores], dtype=float)
            # Use default marker but gray color; filled to distinguish "no residuals"
            ax.scatter(
                xs,
                ys,
                c="gray",
                marker=default_marker,
                edgecolor="black",
                linewidths=0.5,
            )

        ax.grid(True, alpha=0.3)

    # Build the side-by-side figure
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(width, height), constrained_layout=True
    )

    # Left: X–Y
    plot_axis(ax_left, 0, 1)
    ax_left.set_xlabel("X coordinate")
    ax_left.set_ylabel("Y coordinate")
    ax_left.set_title("Camera residuals (X–Y)")

    # Right: X–Z
    plot_axis(ax_right, 0, 2)
    ax_right.set_xlabel("X coordinate")
    ax_right.set_ylabel("Z coordinate")
    ax_right.set_title("Camera residuals (X–Z)")

    # Shared colorbar (only if residuals exist)
    if cams_res:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_left, ax_right], fraction=0.046, pad=0.04)
        cbar.set_label("Depth residual (m)")

    # Legends: groups and accuracy bins
    from matplotlib.lines import Line2D

    # Group legend (fill style only; use a consistent circle marker)
    group_handles = []
    for g in sorted(group_to_fill.keys()):
        if group_to_fill[g]:
            group_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=str(g),
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=8,
                    linewidth=0,
                )
            )
        else:
            group_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=str(g),
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=8,
                    linewidth=0,
                )
            )

    # Accuracy legend (numerical groups) + no-residuals group
    acc_handles = []
    acc_labels = []
    if acc_present:
        for acc_val in acc_values_res:
            acc_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=acc_to_marker.get(acc_val, default_marker),
                    color="w",
                    label=f"{acc_val}",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=8,
                    linewidth=0,
                )
            )
            acc_labels.append(str(acc_val))
    # Add "no residuals" group
    if cams_nores:
        acc_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="no residuals",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
        )

    # Place legends
    if group_handles:
        ax_left.legend(
            handles=group_handles, title="Groups", loc="upper right", frameon=False
        )
    if acc_handles:
        ax_right.legend(
            handles=acc_handles,
            title="Accuracy (numeric)",
            loc="upper right",
            frameon=False,
        )

    # plt.show()
    return fig


def plot_depth_regression(
    depths, depths_predicted, width=10, height=5, title=None, errors=None
):
    """Plot actual vs predicted depths and the residuals of a depth regression.

    Args:
        depths (np.ndarray): Actual depth values.
        depths_predicted (np.ndarray): Predicted depth values from regression.
        width: Figure width (inches).
        height: Figure height (inches).
        title: Optional prefix of the panel titles.
        errors: The reported error metrics: a
            :class:`~substrata.measurements.DepthRegressionResult` or a dict with
            ``rmse``, ``mae`` and ``r2`` (as
            :func:`substrata.measurements.regression_errors`). If None,
            :func:`~substrata.measurements.regression_errors` of the arrays.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    from substrata import measurements

    depths = np.asarray(depths, dtype=float)
    depths_predicted = np.asarray(depths_predicted, dtype=float)
    depths_residuals = depths - depths_predicted
    num_matches = int(len(depths))

    if errors is None:
        errors = measurements.regression_errors(depths, depths_predicted)
    if not isinstance(errors, dict):
        errors = {k: getattr(errors, k) for k in ("rmse", "mae", "r2")}
    rmse, mae, r2 = (float(errors[k]) for k in ("rmse", "mae", "r2"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))

    # Plot 1: Actual vs Predicted depths
    ax1.scatter(depths, depths_predicted, alpha=0.6, edgecolor="black")
    ax1.plot(
        [depths.min(), depths.max()],
        [depths.min(), depths.max()],
        "r--",
        lw=2,
        label="Perfect fit",
    )
    ax1.set_xlabel("Actual Depth (m)")
    ax1.set_ylabel("Predicted Depth (m)")
    base_title_1 = f"Depth Regression Fit\nR² = {r2:.3f}, RMSE = {rmse:.3f}m"
    ax1.set_title((f"{title} - " if title else "") + base_title_1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals vs Predicted
    ax2.scatter(depths_predicted, depths_residuals, alpha=0.6, edgecolor="black")
    ax2.axhline(y=0, color="r", linestyle="--", lw=2, label="Zero residual")
    ax2.set_xlabel("Predicted Depth (m)")
    ax2.set_ylabel("Residuals (m)")
    base_title_2 = f"Residual Analysis\nMAE = {mae:.3f}m, n = {num_matches}"
    ax2.set_title((f"{title} - " if title else "") + base_title_2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()

    return fig


def depth_residual_figures(analyzer, kind="estimated", width=15, height=5, **kwargs):
    """Depth-regression figures of a ``DepthResidualAnalyzer``.

    (:class:`substrata.measurements.DepthResidualAnalyzer`.)

    Args:
        analyzer: The ``DepthResidualAnalyzer``.
        kind: ``"estimated"``: sensor depths vs the depths estimated from the
            container's depth model
            (the analyzer's ``get_depths_and_estimated_depths``);
            ``"z"``: sensor depths vs the Z coordinates
            (``get_depths_and_z_coords``).
        width: Figure width in inches.
        height: Figure height in inches.
        **kwargs: Passed to the analyzer's getter (accuracy filter).

    Returns:
        tuple: ``(fig1, fig2)``: the regression figure
        (:func:`plot_depth_regression`) and the per-camera residual map
        (:func:`plot_cam_residuals`; for annotations a second regression
        figure).
    """
    if kind == "estimated":
        depths, other, filtered = analyzer.get_depths_and_estimated_depths(**kwargs)
        titles = ("Depth vs Estimated Depth", "Depth Residuals")
    elif kind == "z":
        depths, other, filtered = analyzer.get_depths_and_z_coords(**kwargs)
        titles = ("Depth vs Z-Coordinate", "Z-Coordinate Residuals")
    else:
        raise ValueError(f"kind must be 'estimated' or 'z', got {kind!r}")
    fig1 = plot_depth_regression(
        depths, other, width=width, height=height, title=titles[0]
    )
    if type(analyzer.container).__name__ == "Cameras":
        fig2 = plot_cam_residuals(filtered, depths, other, width=width, height=height)
    else:
        fig2 = plot_depth_regression(
            depths, other, width=width, height=height, title=titles[1]
        )
    return fig1, fig2


def save_depth_residuals_pdf(analyzer, filepath=None, width=15, height=5, **kwargs):
    """Save the four depth-residual figures of an analyzer to a PDF.

    See :func:`depth_residual_figures`. Without ``filepath`` the PDF is named
    after the container's cameras or markers file
    (``<base>_depth_residuals.pdf``).

    Returns:
        str: The filepath written.
    """
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages

    container = analyzer.container
    backend_original = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        if filepath is None:
            base = None
            for attr in ("cams_meta_filepath", "markers_filepath"):
                if getattr(container, attr, None):
                    base = os.path.splitext(getattr(container, attr))[0]
                    break
            if base is None:
                base = type(container).__name__.lower()
            filepath = f"{base}_depth_residuals.pdf"

        figs = list(depth_residual_figures(analyzer, "estimated", width, height,
                                           **kwargs))
        figs += depth_residual_figures(analyzer, "z", width, height, **kwargs)
        with PdfPages(filepath) as pdf:
            for fig in figs:
                pdf.savefig(fig)
        for fig in figs:
            plt.close(fig)
        return filepath
    finally:
        matplotlib.use(backend_original, force=True)
