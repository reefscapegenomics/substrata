# Standard Library
import os
import tempfile
import shutil
from io import BytesIO
import subprocess
from typing import Any, Dict, List, Optional

# Third-Party Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import ffmpeg
from fpdf import FPDF
from mpl_toolkits.mplot3d import Axes3D
from open3d import geometry, io, utility, visualization
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Local Modules
from substrata import settings, segmentation, pointclouds, geom
from substrata.logging import logger

# from open3d.web_visualizer import draw


def capture_geoms_to_file(geoms, output_file):
    vis = visualization.Visualizer()
    vis.create_window(visible=False)
    for geom in geoms:
        vis.add_geometry(geom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_file)
    vis.destroy_window()


def show(geoms, highlight_coords=None, max_output_points=500000):
    """Show PointCloud or SimplePointCloud using plotly for interactive 3D visualization.

    Works in both Jupyter notebooks and VS Code. Optionally highlights specific
    coordinates with red markers.

    Args:
        geoms: PointCloud or SimplePointCloud object to visualize.
        highlight_coords: Optional coordinates to highlight with red markers.
            Can be a single [x, y, z] coordinate or an array of shape (N, 3)
            for multiple coordinates.
        max_output_points: Maximum number of points to display. The point cloud
            will be decimated if it exceeds this limit. Default is at 500,000 points
            based on plotly's performance limits.

    Returns:
        plotly.graph_objects.Figure: The interactive plotly figure.
    """
    import plotly.graph_objects as go

    # Decimate if required (and ensure PointCloud format)
    geoms = pointclouds.get_decimated_pcd(geoms, max_output_points)

    # Check if input is a point cloud
    if isinstance(geoms, pointclouds.SimplePointCloud):
        points = np.asarray(geoms.points)
        colors = getattr(geoms, "colors", None)
        if colors is not None:
            colors = np.asarray(colors)
    elif isinstance(geoms, pointclouds.PointCloud):
        points = np.asarray(geoms.points)
        colors = getattr(geoms, "colors", None)
        if colors is not None:
            colors = np.asarray(geoms.colors)
    elif hasattr(geoms, "o3d_pcd"):
        o3d_pcd = geoms.o3d_pcd
        points = np.asarray(o3d_pcd.points)
        colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else None
    else:
        raise ValueError("geoms must be a PointCloud or SimplePointCloud object")

    # Ensure points are in correct format
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be shape (N, 3), got {points.shape}")

    # Prepare colors for plotly (expects RGB strings like 'rgb(255,0,0)')
    if colors is not None:
        # Normalize colors to 0-255 range if needed
        if colors.max() <= 1.0:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            colors_uint8 = colors.astype(np.uint8)
        # Convert to RGB strings
        color_strings = [
            f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors_uint8
        ]
    else:
        color_strings = None

    # Create scatter3d trace for point cloud
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=color_strings if color_strings else "blue",
            opacity=0.8,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[trace])

    # Add highlighted coordinates if provided
    if highlight_coords is not None:
        coords = np.asarray(highlight_coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        if coords.shape[1] < 3:
            raise ValueError("highlight_coords must contain 3D coordinates [x, y, z]")

        # Add red markers for highlighted coordinates
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="circle",
                ),
                showlegend=False,
            )
        )

    # Calculate ranges for equal aspect ratio (same pixels per meter for all axes)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Use the maximum range to ensure equal scaling
    max_range = max(x_range, y_range, z_range, 1e-9)  # Avoid division by zero

    # Center each axis and use the same range
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    half_range = max_range / 2.0

    # Update layout for better viewing with equal aspect ratio
    # Set camera view similar to matplotlib default (x on right, y going back)
    camera_eye = {
        "x": 1.25,
        "y": -1.25,
        "z": 1.25,
    }
    camera_center = {"x": 0, "y": 0, "z": 0}
    camera_up = {"x": 0, "y": 0, "z": 1}

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[x_center - half_range, x_center + half_range]),
            yaxis=dict(range=[y_center - half_range, y_center + half_range]),
            zaxis=dict(range=[z_center - half_range, z_center + half_range]),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        width=800,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Show the figure (works in both Jupyter and VS Code)
    # Try to detect environment and use appropriate renderer
    try:
        from IPython import get_ipython

        in_jupyter = get_ipython() is not None
    except ImportError:
        in_jupyter = False

    if in_jupyter:
        # In Jupyter: try default renderer first, fallback to browser
        try:
            fig.show()
        except (ValueError, ImportError) as e:
            if "nbformat" in str(e):
                # nbformat not installed - use browser renderer
                import warnings

                warnings.warn(
                    "nbformat>=4.2.0 not installed. Using browser renderer. "
                    "Install with: pip install nbformat>=4.2.0 for inline display."
                )
                fig.show(renderer="browser")
            else:
                raise
    else:
        # Command line: use browser renderer
        fig.show(renderer="browser")


def plot(
    pcd,
    point_size=2,
    width=10,
    height=4,
    max_output_points=50000,
    title=None,
    ax=None,
    highlight_coords=None,
):
    """
    Plot a 3D point cloud, with optional decimation for speed.

    Args:
        pcd: The point cloud object (Open3D format or SimplePointCloud).
        point_size (int): Size of the points in the scatter plot.
        width (int): Width of the figure (if creating a new figure).
        height (int): Height of the figure (if creating a new figure).
        max_output_points (int): Maximum number of points to plot.
        title (str | None): Title for the plot.
        ax (matplotlib.axes.Axes | None): Optional 3D axes to draw into.
        highlight_coords (array-like | None): Optional coordinates to highlight
            with red dots. Can be a single [x, y, z] coordinate or an array
            of shape (N, 3) for multiple coordinates.

    Returns:
        matplotlib.figure.Figure | None: New figure if created, otherwise None.
    """
    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure
    ax.set_box_aspect((width, height, height))
    ax.scatter(
        pcd.points[:, 0],
        pcd.points[:, 1],
        pcd.points[:, 2],
        c=pcd.colors,
        s=point_size,
        edgecolor="none",
    )

    # Highlight the specified coordinates with red dots
    if highlight_coords is not None:
        # Convert single coord to array of coords if needed
        coords = np.array(highlight_coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c="red",
            s=200,  # Size of the highlight dot
            edgecolor="none",
        )

    # Weighted equal scaling: x : y : z = width : height : height
    mins = pcd.points.min(axis=0)
    maxs = pcd.points.max(axis=0)
    center = (mins + maxs) / 2.0
    ranges = maxs - mins

    weights = np.array([width, height, height], dtype=float)
    k = float(np.max(ranges / weights)) if np.all(weights > 0) else 1.0

    half = 0.5 * k * weights
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])

    # Keep the box aspect consistent with the intended physical ratios
    ax.set_box_aspect((width, height, height))
    if title is not None:
        ax.set_title(title)
    ax.set_rasterized(True)
    return fig if created_fig else None


def plot_2d(
    pcd,
    point_size=2,
    width=10,
    height=5,
    highlight_coords=None,
    title=None,
    max_output_points=50000,
):
    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.scatter(
        pcd.points[:, 0],
        pcd.points[:, 1],
        c=pcd.colors,
        s=point_size,
        edgecolor="none",
    )

    # Highlight the specified coordinates with a big red dot
    if highlight_coords is not None:
        # Convert single coord to array of coords if needed
        coords = np.array(highlight_coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="red",
            s=50,  # Size of the highlight dot
            edgecolor="none",
        )

    if title is not None:
        ax.set_title(title)

    ax.set_rasterized(True)
    return fig


def multiplot_2d(
    pcds,
    annotations_list,
    point_size=2,
    width=10,
    height=5,
    title=None,
    max_output_points=50000,
    label_colors=None,
    max_x=None,
):
    """
    Plots 2D point cloud(s) with annotations highlighted in different colors based on their labels.

    Args:
        pcds: Single point cloud object or list of point cloud objects (Open3D format or SimplePointCloud)
        annotations_list: List of Annotations instances to highlight
        point_size (int): Size of the points in the scatter plot
        width (int): Width of the figure
        height (int): Height of the figure
        title (str): Optional title for the plot
        max_output_points (int): Maximum number of points to display
        label_colors (dict): Optional dictionary mapping labels to colors
        max_x (float): Optional maximum x-axis value to display

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Convert single pcd to list if necessary
    if not isinstance(pcds, list):
        pcds = [pcds]

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    # Process each point cloud
    for pcd in pcds:
        # Decimate if required (and ensure PointCloud format)
        pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

        # Plot base point cloud
        ax.scatter(
            pcd.points[:, 0],
            pcd.points[:, 1],
            c=pcd.colors,
            s=point_size,
            edgecolor="none",
        )

    # Plot annotations with different colors based on labels
    if annotations_list is not None:
        # Get unique labels and assign colors
        unique_labels = set()
        for annotations in annotations_list:
            unique_labels.update(ann.label for ann in annotations.data.values())

        # Use provided colors or generate rainbow colors
        if label_colors is None:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            label_colors = dict(zip(unique_labels, colors))

        # Plot each annotation point with its corresponding color
        for annotations in annotations_list:
            for ann in annotations.data.values():
                # Plot white outline first
                ax.scatter(
                    ann.coords[0],
                    ann.coords[1],
                    c="white",
                    s=70,  # Slightly larger than the colored dot
                    edgecolor="none",
                )
                # Plot colored dot on top
                ax.scatter(
                    ann.coords[0],
                    ann.coords[1],
                    c=[label_colors[ann.label]],
                    s=50,  # Size of the annotation dots
                    edgecolor="none",
                    label=f"{ann.label}",
                )

    if title is not None:
        ax.set_title(title)

    # Set x-axis limit if specified
    if max_x is not None:
        ax.set_xlim(right=max_x)

    # Remove y-axis values
    ax.set_yticklabels([])

    ax.set_rasterized(True)
    return fig


def plot_text(text, width=10, height=5):
    """
    Plots a text string as a figure (e.g. for inclusion in a PDF).
    """
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.5, text, ha="left", va="center")
    ax.set_axis_off()  # Hide all axes
    ax.set_rasterized(True)
    return fig


def plot_colorchecker_qc_grid(
    patch_results: List[Dict[str, Any]],
    title: str,
    outlier_mask: Optional[np.ndarray] = None,
) -> Any:
    """6×4 grid of ColorChecker patches with reference vs measured sRGB and outliers.

    Args:
        patch_results: List of dicts with keys ``name``, ``median_rgb_0_1``,
            ``ref_rgb_255``, ``n_points``, ``chart_u``, ``chart_v``, ``marker_u``,
            ``marker_v``.
        title: Figure title (inset / shift text may be appended by caller).
        outlier_mask: Boolean array, length ``len(patch_results)``, True = outlier
            vs other cards.

    Returns:
        matplotlib Figure.
    """
    n = len(patch_results)
    if outlier_mask is None:
        outlier_mask = np.zeros(n, dtype=bool)
    fig, axes = plt.subplots(4, 6, figsize=(14, 9))
    fig.suptitle(title, fontsize=10)
    for idx in range(24):
        row, col = divmod(idx, 6)
        ax = axes[row, col]
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if idx >= n:
            continue
        pr = patch_results[idx]
        ref = np.asarray(pr["ref_rgb_255"], dtype=float) / 255.0
        ref = np.clip(ref, 0.0, 1.0)
        ax.add_patch(
            mpatches.Rectangle(
                (0.0, 0.35),
                1.0,
                0.65,
                facecolor=ref,
                edgecolor="0.3",
                linewidth=0.5,
            )
        )
        med = pr.get("median_rgb_0_1")
        if med is not None:
            m255 = np.asarray(med, dtype=float).reshape(3) * 255.0
            t = (
                f"{pr['name'][:14]}\n"
                f"meas: {m255[0]:.0f},{m255[1]:.0f},{m255[2]:.0f}\n"
                f"ref:  {pr['ref_rgb_255'][0]},"
                f"{pr['ref_rgb_255'][1]},"
                f"{pr['ref_rgb_255'][2]}\n"
                f"n={pr['n_points']}"
            )
        else:
            t = f"{pr['name'][:14]}\n(no color)"
        if outlier_mask[idx]:
            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    0.34,
                    facecolor=(1.0, 0.85, 0.85),
                    edgecolor="red",
                    linewidth=2,
                )
            )
            t += "\nOUTLIER"
        ax.text(
            0.5,
            0.2,
            t,
            ha="center",
            va="center",
            fontsize=5,
            clip_on=True,
        )
    plt.tight_layout()
    fig.set_rasterized(True)
    return fig


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
) -> Any:
    """Summary figure showing the effect of the affine colour correction.

    Top row: three 6x4 swatch grids (Measured / Corrected / Reference).
    Bottom row: per-channel scatter plots (R, G, B) with before and after.

    Args:
        measured_rgb_255: (N, 3) measured median RGB in 0-255.
        reference_rgb_255: (N, 3) reference sRGB in 0-255.
        correction: Dict with ``"matrix"`` (3x3) and ``"offset"`` (3,).
        patch_names: Patch label strings.
        outlier_mask: Optional boolean array for the measured grid.
        title: Overall figure title.
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        matplotlib Figure.
    """
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

        rmse_before = float(np.sqrt(np.mean((meas_ch - ref_ch) ** 2)))
        rmse_after = float(np.sqrt(np.mean((corr_ch - ref_ch) ** 2)))
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


def plot_compare(pcd1, pcd2, point_size=1, max_output_points=50000):
    """
    Plots two 3D point clouds with pcd1 in a blue color scale and pcd2 in a red color scale.

    Args:
        pcd1: First point cloud object with points and colors (Open3D format).
        pcd2: Second point cloud object with points and colors (Open3D format).
        point_size (int): Size of the points in the scatter plot.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Decimate if required (and ensure PointCloud format)
    pcd1 = pointclouds.get_decimated_pcd(pcd1, max_output_points)
    pcd2 = pointclouds.get_decimated_pcd(pcd2, max_output_points)

    colors1 = np.full((pcd1.points.shape[0], 3), [0.0, 0.0, 1.0])  # Blue scale
    colors2 = np.full((pcd2.points.shape[0], 3), [1.0, 0.0, 0.0])  # Red scale

    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((6, 1, 1))

    # Plot the first point cloud
    ax.scatter(
        pcd1.points[:, 0],
        pcd1.points[:, 1],
        pcd1.points[:, 2],
        c=colors1,
        s=point_size,
        edgecolor="none",
    )

    # Plot the second point cloud
    ax.scatter(
        pcd2.points[:, 0],
        pcd2.points[:, 1],
        pcd2.points[:, 2],
        c=colors2,
        s=point_size,
        edgecolor="none",
    )

    ax.set_rasterized(True)

    return fig


def create_vector_geom(vector, length):
    lineset = geometry.LineSet()
    lineset.points = utility.Vector3dVector(
        [np.array([0, 0, 0]), np.array(vector) * float(length)]
    )
    lineset.lines = utility.Vector2iVector([[0, 1]])
    return lineset


def show_coords_as_lines(pcd, points, Jupyter=False):
    # Create orientation z-lines originating from points
    connect_points = []
    for point in points:
        if not point is None:
            connect_points.append(np.array(point))
            connect_points.append(
                np.array([point[0], point[1], point[2] + settings.LEN_ORIENT_LINE])
            )

    connect_lines = [[i, i + 1] for i in range(0, len(connect_points), 2)]

    connecting_lineset = geometry.LineSet()
    connecting_lineset.points = utility.Vector3dVector(connect_points)
    connecting_lineset.lines = utility.Vector2iVector(connect_lines)

    show([pcd.o3d_pcd, connecting_lineset], Jupyter=Jupyter)


def show_grid_points(pcd, grid_indices):
    try:
        pcd.o3d_pcd_tree
    except AttributeError:
        pcd.build_kd_tree()
    # Filter out None values and get the valid closest indices
    grid_point_idx = [idx for idx in grid_indices if idx is not None]
    for point in grid_point_idx:
        [k, idx, _] = pcd.o3d_pcd_tree.search_radius_vector_3d(pcd.points[point], 0.05)
        np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]
    show([pcd.o3d_pcd])


def show_point_values(pcd, annotations, meta_data_col_index=None, size=0.2):
    # Create orientation z-lines originating from points
    sphere_geoms = []
    for annotation in annotations.data.values():
        if meta_data_col_index is not None:
            color_value = min(float(annotation.meta_data[meta_data_col_index]), 1.0)
            color = [color_value, color_value, 0]
        else:
            color = [1, 0, 0]

        sphere = geometry.TriangleMesh.create_sphere(radius=size)
        num_vertices = np.asarray(sphere.vertices).shape[0]
        sphere.vertex_colors = utility.Vector3dVector([color] * num_vertices)

        sphere.compute_vertex_normals()
        sphere.translate(annotation.coords)
        sphere_geoms.append(sphere)

    show([pcd.o3d_pcd, *sphere_geoms])


def show_img_cv2(img_path, highlight_pixels=None):
    """Show image and"""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if highlight_pixels is not None:
        highlight_pixels = np.array(highlight_pixels, dtype=int)
        if highlight_pixels.ndim == 1:
            cv2.circle(
                image,
                (highlight_pixels[0], highlight_pixels[1]),
                radius=50,
                color=(255, 0, 0),
                thickness=-1,
            )
        elif highlight_pixels.ndim > 1:
            for pixel in highlight_pixels:
                cv2.circle(
                    image,
                    (pixel[0], pixel[1]),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,
                )
    plt.imshow(image)
    plt.show()


def show_img(img_path, highlight_pixels=None):
    """
    Load an image using PIL, optionally add highlighted circles, and show it.

    This function ignores the EXIF orientation (unless otherwise handled by PIL)
    and always displays the image in its raw pixel orientation.

    If the path is missing or the file cannot be read, logs a warning and returns
    without raising (execution continues for the caller).

    Args:
        img_path (str): Path to the image file.
        highlight_pixels (None, list, or np.ndarray):
            Pixel coordinates to highlight. If a 1D array/list, it's interpreted
            as a single point [x, y] with a larger circle; if 2D, each sub-list
            represents a point with a smaller circle.
    """
    if not img_path:
        logger.warning("show_img: no image path given; skipping display.")
        return
    if not os.path.isfile(img_path):
        logger.warning("show_img: file not found: %s", img_path)
        return

    try:
        image = Image.open(img_path).convert("RGB")
    except OSError as e:
        logger.warning("show_img: could not open image %s: %s", img_path, e)
        return

    # Prepare drawing context.
    draw = ImageDraw.Draw(image)

    if highlight_pixels is not None:
        # Ensure highlight_pixels is a NumPy array.
        hp = np.array(highlight_pixels, dtype=int)
        if hp.ndim == 1:
            # Draw a large circle (radius 50) for a single point.
            x, y = hp[0], hp[1]
            # PIL's ellipse uses a bounding box: (left, top, right, bottom)
            draw.ellipse((x - 50, y - 50, x + 50, y + 50), fill=(255, 0, 0))
        elif hp.ndim > 1:
            for pixel in hp:
                x, y = pixel[0], pixel[1]
                draw.ellipse((x - 50, y - 50, x + 50, y + 50), fill=(255, 0, 0))

    # Convert the PIL image to a NumPy array and display it.
    plt.imshow(np.array(image))
    plt.axis("off")
    plt.show()


def save_img(img_path, save_path, highlight_pixels=None):
    """
    Read an image from disk, optionally draw highlighted circles on given pixel coordinates,
    and save the full-resolution image to disk.

    Parameters:
        img_path (str): Path to the input image.
        save_path (str): Path where the resulting image will be saved.
        highlight_pixels (None, list, or np.ndarray): Pixel coordinates to highlight.
            - If a 1D array/list, it is assumed to be a single point [x, y].
            - If a 2D array/list, each sub-list represents a point [x, y].
    """
    # Read the full-resolution image in BGR format.
    image = cv2.imread(img_path)
    if image is None:
        raise IOError(f"Could not read image: {img_path}")

    # If highlight_pixels is provided, draw a circle on each.
    if highlight_pixels is not None:
        highlight_pixels = np.array(highlight_pixels, dtype=int)
        if highlight_pixels.ndim == 1:
            cv2.circle(
                image,
                (highlight_pixels[0], highlight_pixels[1]),
                radius=50,
                color=(0, 0, 255),
                thickness=-1,
            )
        elif highlight_pixels.ndim > 1:
            for pixel in highlight_pixels:
                cv2.circle(
                    image,
                    (pixel[0], pixel[1]),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )

    # Save the full-resolution image to disk.
    cv2.imwrite(save_path, image)


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


def plot_positions(
    positions,
    pcd,
    max_output_points=500000,
    width=10,
    height=8,
    point_size=40,
    title=None,
    color=False,
    show_x_z=False,
    show_z=False,
    zoom=None,
):
    """Plot positions (Cameras, Annotations, or Nx3 coords) in X–Y and optionally X–Z views.

    - Background shows a grayscale decimated point cloud if provided.
      If color=True and the point cloud has colors, use those instead.
    - Markers are colored by Z (bwr colormap) if show_z=True, otherwise in red.
    - If items have a `group` attribute, groups are encoded as filled (even index)
      vs hollow (odd index) and a legend is shown.

    Args:
        positions: One of:
            - single Annotation object (will be wrapped in Annotations container)
            - collection with `.data` mapping to objects with `.coords` (e.g., Cameras, Annotations)
            - iterable of objects each with `.coords`
            - numpy array or list-like of shape (N, 3) with XYZ coordinates
        pcd: Point cloud object for background (decimated).
        max_output_points: Max points for background decimation (default 500,000).
        width: Figure width in inches.
        height: Figure height in inches.
        point_size: Marker size for plotted positions.
        title: Optional figure title.
        color: If True, draw background point cloud with its colors when available;
               if False (default), draw background in gray.
        show_x_z: If True, show X–Z plot in addition to X–Y (default False).
        show_z: If True, color markers by Z coordinate with colorbar (default False).
            If False, all markers are colored red.
        zoom: Optional zoom level (only used for single Annotation). If provided,
            limits the view to a zoom x zoom meter area centered on the annotation.
            For example, zoom=1 shows a 1x1m area, zoom=2 shows a 2x2m area.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    # Check if input is a single Annotation (before normalization)
    is_single_annotation = (
        hasattr(positions, "coords")
        and not hasattr(positions, "data")
        and type(positions).__name__ == "Annotation"
    )
    annotation_coords = None
    if is_single_annotation and zoom is not None:
        annotation_coords = np.asarray(positions.coords, dtype=float)

    # Normalize input → coords[N,3], groups[List[Any] | None]
    def _normalize_positions(pos):
        # Case 0: single Annotation object - wrap in Annotations container
        if (
            hasattr(pos, "coords")
            and not hasattr(pos, "data")
            and type(pos).__name__ == "Annotation"
        ):
            from substrata import annotations

            annotations_container = annotations.Annotations()
            annotations_container.data[pos.id] = pos
            pos = annotations_container

        # Case 1: numpy array or list-like of coordinates
        try:
            arr = np.asarray(pos, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                coords = arr[:, :3]
                groups = [None] * len(coords)
                return coords, groups
        except Exception:
            pass

        # Case 2: object with .data mapping to items with .coords
        if hasattr(pos, "data"):
            try:
                items = list(pos.data.values())
                coords = []
                groups = []
                for it in items:
                    if hasattr(it, "coords") and it.coords is not None:
                        c = np.asarray(it.coords, dtype=float)
                        if c.shape[0] >= 3:
                            coords.append(c[:3])
                            groups.append(getattr(it, "group", None))
                if coords:
                    return np.vstack(coords), groups
            except Exception:
                pass

        # Case 3: iterable of objects with .coords
        try:
            coords = []
            groups = []
            for it in list(pos):
                if hasattr(it, "coords") and it.coords is not None:
                    c = np.asarray(it.coords, dtype=float)
                    if c.shape[0] >= 3:
                        coords.append(c[:3])
                        groups.append(getattr(it, "group", None))
            if coords:
                return np.vstack(coords), groups
        except Exception:
            pass

        raise ValueError(
            "positions must be a single Annotation object, Nx3 array-like, "
            "a collection with .data of items each having .coords, or an "
            "iterable of items with .coords."
        )

    coords, groups = _normalize_positions(positions)
    if coords.size == 0:
        raise ValueError("No positions with coords provided.")

    # Prepare group → filled mapping (if any groups are present)
    valid_groups = [g for g in groups if g is not None]
    unique_groups = sorted(set(valid_groups))
    group_to_fill = {g: (i % 2 == 0) for i, g in enumerate(unique_groups)}

    # Prepare color mapping by Z coordinate (blue→red) if show_z is True
    z_vals = np.array(coords[:, 2], dtype=float)
    if show_z:
        if z_vals.size:
            z_min = float(np.nanmin(z_vals))
            z_max = float(np.nanmax(z_vals))
            if not (np.isfinite(z_min) and np.isfinite(z_max)):
                z_min, z_max = -1.0, 1.0
            if z_min == z_max:
                z_min, z_max = z_min - 1e-6, z_max + 1e-6
        else:
            z_min, z_max = -1.0, 1.0
        norm = plt.Normalize(vmin=z_min, vmax=z_max)
        cmap = plt.cm.bwr
    else:
        # Use red color for all markers when show_z is False
        norm = None
        cmap = None

    # Build figure - one or two rows depending on show_x_z
    if show_x_z:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(width, height), constrained_layout=True
        )
    else:
        fig, ax_top = plt.subplots(
            1, 1, figsize=(width, height), constrained_layout=True
        )
        ax_bottom = None

    # Background: decimated PCD and grayscale points for XY and XZ
    try:
        if zoom is None:
            pcd_bg = pointclouds.get_decimated_pcd(pcd, max_output_points)
        else:
            pcd_bg = pcd
        pts = np.asarray(pcd_bg.points)
        if len(pts) > 0:
            # Choose background colors
            if color:
                pcd_cols = np.asarray(pcd_bg.colors)
                use_cols = pcd_cols.shape[0] == pts.shape[0] and pcd_cols.shape[0] > 0
                bg_cols = (
                    pcd_cols
                    if use_cols
                    else np.full((pts.shape[0], 3), 0.7, dtype=float)
                )
            else:
                bg_cols = np.full((pts.shape[0], 3), 0.7, dtype=float)
            # Top: XY background
            ax_top.scatter(
                pts[:, 0], pts[:, 1], s=1, c=bg_cols, alpha=0.4, edgecolor="none"
            )
            # Bottom: XZ background (only if show_x_z is True)
            if show_x_z and ax_bottom is not None:
                ax_bottom.scatter(
                    pts[:, 0], pts[:, 2], s=1, c=bg_cols, alpha=0.4, edgecolor="none"
                )
    except Exception:
        # If background fails, continue with positions only
        pass

    # Helper to plot (x_idx, y_idx)
    def _plot_axis(ax, x_idx: int, y_idx: int, xlabel: str, ylabel: str, ttl: str):
        xs = np.asarray(coords[:, x_idx], dtype=float)
        ys = np.asarray(coords[:, y_idx], dtype=float)
        # Draw hollow vs filled by group
        for x, y, g, z in zip(xs, ys, groups, z_vals):
            if show_z and cmap is not None and norm is not None:
                col = cmap(norm(z))
            else:
                col = "red"
            filled = group_to_fill.get(g, True)
            if filled:
                ax.scatter(
                    x,
                    y,
                    s=point_size,
                    facecolor=col,
                    edgecolor="black",
                    linewidths=0.7,
                    alpha=0.95,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=point_size,
                    facecolors="none",
                    edgecolors=[col],
                    linewidths=1.2,
                    alpha=0.95,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Top: X–Y
    _plot_axis(ax_top, 0, 1, "X coordinate", "Y coordinate", "Positions (X–Y)")
    # Bottom: X–Z (only if show_x_z is True)
    if show_x_z and ax_bottom is not None:
        _plot_axis(ax_bottom, 0, 2, "X coordinate", "Z coordinate", "Positions (X–Z)")

    # Apply zoom limits if zoom is specified and we have a single annotation
    if zoom is not None and is_single_annotation and annotation_coords is not None:
        half_zoom = zoom / 2.0
        # Set X-Y limits
        ax_top.set_xlim(
            annotation_coords[0] - half_zoom, annotation_coords[0] + half_zoom
        )
        ax_top.set_ylim(
            annotation_coords[1] - half_zoom, annotation_coords[1] + half_zoom
        )
        # Set X-Z limits if showing X-Z view
        if show_x_z and ax_bottom is not None:
            ax_bottom.set_xlim(
                annotation_coords[0] - half_zoom, annotation_coords[0] + half_zoom
            )
            ax_bottom.set_ylim(
                annotation_coords[2] - half_zoom, annotation_coords[2] + half_zoom
            )

    # Shared colorbar for Z coordinate (only if show_z is True)
    if show_z and cmap is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if show_x_z and ax_bottom is not None:
            cbar = fig.colorbar(sm, ax=[ax_top, ax_bottom], fraction=0.046, pad=0.04)
        else:
            cbar = fig.colorbar(sm, ax=ax_top, fraction=0.046, pad=0.04)
        cbar.set_label("Z (m)")

    # Group legend (if groups present)
    if unique_groups:
        from matplotlib.lines import Line2D

        handles = []
        for g in unique_groups:
            if group_to_fill[g]:
                handles.append(
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
                handles.append(
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
        ax_top.legend(handles=handles, title="Groups", loc="upper right", frameon=False)

    if title:
        fig.suptitle(title)

    return fig
    # Collect cameras with coords
    cams_all = [
        cam
        for cam in cams.data.values()
        if hasattr(cam, "coords") and cam.coords is not None
    ]
    if not cams_all:
        raise ValueError("No cameras with coords found.")

    # Prepare group → filled mapping
    try:
        group_names = (
            cams.group_names
            if hasattr(cams, "group_names")
            else sorted({getattr(cam, "group", None) for cam in cams_all})
        )
        group_names = [g for g in group_names if g is not None]
    except Exception:
        group_names = []
    group_to_fill = {g: (i % 2 == 0) for i, g in enumerate(sorted(group_names))}

    # Prepare color mapping by Z coordinate (blue→red)
    z_vals = np.array([float(cam.coords[2]) for cam in cams_all], dtype=float)
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    z_min, z_max = (
        (z_min, z_max) if np.isfinite(z_min) and np.isfinite(z_max) else (-1.0, 1.0)
    )
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    cmap = plt.cm.bwr

    # Build a two-row figure (top and bottom)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(width, height), constrained_layout=True
    )

    # Background: decimated PCD and grayscale points for XY and XZ
    try:
        pcd_bg = pointclouds.get_decimated_pcd(pcd, max_output_points)
        pts = np.asarray(pcd_bg.points)
        if len(pts) > 0:
            gray = np.full((pts.shape[0], 3), 0.7, dtype=float)
            # Top: XY background
            ax_top.scatter(
                pts[:, 0], pts[:, 1], s=1, c=gray, alpha=0.4, edgecolor="none"
            )
            # Bottom: XZ background
            ax_bottom.scatter(
                pts[:, 0], pts[:, 2], s=1, c=gray, alpha=0.4, edgecolor="none"
            )
    except Exception:
        # If background fails, continue with cameras only
        pass

    # Helper to plot cameras into an axis given x and y indices (0=x,1=y,2=z),
    # coloring by Z using the bwr colormap. Group determines filled/hollow.
    def _plot_axis(ax, x_idx: int, y_idx: int, title: str, xlabel: str, ylabel: str):
        xs = np.array([float(cam.coords[x_idx]) for cam in cams_all], dtype=float)
        ys = np.array([float(cam.coords[y_idx]) for cam in cams_all], dtype=float)
        colors = [cmap(norm(float(cam.coords[2]))) for cam in cams_all]

        # Draw hollow vs filled by group
        for cam, x, y, color in zip(cams_all, xs, ys, colors):
            filled = group_to_fill.get(getattr(cam, "group", None), True)
            if filled:
                ax.scatter(
                    x,
                    y,
                    s=cam_point_size,
                    facecolor=color,
                    edgecolor="black",
                    linewidths=0.7,
                    alpha=0.95,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=cam_point_size,
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=1.2,
                    alpha=0.95,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Top: X–Y
    _plot_axis(ax_top, 0, 1, "Camera positions (X–Y)", "X coordinate", "Y coordinate")
    # Bottom: X–Z
    _plot_axis(
        ax_bottom, 0, 2, "Camera positions (X–Z)", "X coordinate", "Z coordinate"
    )

    # Legends: groups only (filled vs hollow); add a shared colorbar for Z
    from matplotlib.lines import Line2D

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
    if group_handles:
        ax_top.legend(
            handles=group_handles, title="Groups", loc="upper right", frameon=False
        )

    # Shared colorbar for Z coordinate
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bottom], fraction=0.046, pad=0.04)
    cbar.set_label("Z (m)")

    return fig


def get_crop_img_cv2(img_path, crop_x, crop_y, crop_w, crop_h):
    """Get cropped image"""
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Could not read image: {img_path}")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    left = int(crop_x - crop_w / 2)
    top = int(crop_y - crop_h / 2)
    right = left + int(crop_w)
    bottom = top + int(crop_h)

    h_img, w_img = img.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(w_img, right)
    bottom = min(h_img, bottom)

    # If the requested crop falls completely outside the image, raise an error
    if left >= right or top >= bottom:
        raise ValueError(
            f"Crop area falls outside image bounds: center=({crop_x}, {crop_y}), size=({crop_w}x{crop_h}), image={img_path}"
        )
    cropped_img = img[top:bottom, left:right]
    return cropped_img


def get_crop_img(img_path, crop_x, crop_y, crop_w, crop_h):
    """Get cropped image using PIL"""
    from PIL import Image

    img = Image.open(img_path)

    left = int(crop_x - crop_w / 2)
    top = int(crop_y - crop_h / 2)
    right = left + int(crop_w)
    bottom = top + int(crop_h)

    w_img, h_img = img.size
    left = max(0, left)
    top = max(0, top)
    right = min(w_img, right)
    bottom = min(h_img, bottom)

    # If the requested crop falls completely outside the image, raise an error
    if left >= right or top >= bottom:
        raise ValueError(
            f"Crop area falls outside image bounds: center=({crop_x}, {crop_y}), size=({crop_w}x{crop_h}), image={img_path}"
        )
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def save_to_tmp_file(image):
    """Save image to temporary file"""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_file.close()
    cv2.imwrite(tmp_file.name, image)
    return tmp_file.name


def encode_to_png_buffer(image):
    """Encode image to JPG buffer"""
    # Convert PIL image to numpy array if needed
    if hasattr(image, "size"):  # PIL image
        image = np.array(image)
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise IOError("Could not encode image to JPG.")
    # Convert the NumPy buffer to a BytesIO stream
    return BytesIO(buffer.tobytes())


def save_cropped_image_matches_to_pdf(
    image_matches,
    output_filepath,
    n_cols=3,
    n_rows=4,
    crop_w=1000,
    crop_h=1000,
    single_mask=False,
):
    """
    Save a PDF file from a list of ImageMatches
    """
    pdf = FPDF()
    pdf.set_auto_page_break(False)
    pdf.set_font("Arial", size=8)

    pdf.add_page()
    # Page width and height in FPDF's units (default: mm)
    page_w = pdf.w
    page_h = pdf.h
    margin = 10

    # Compute the usable space (accounting# for margins)
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    # Cell size in the grid
    cell_w = usable_w / n_cols
    cell_h = usable_h / n_rows
    image_h = cell_h * 0.8

    # Track current row and column in the grid
    row_idx = 0
    idx = 0

    for match in image_matches.values():
        # Calculate the (x, y) position for this cell
        x = margin + idx * cell_w
        y = margin + row_idx * cell_h

        if match.masks:
            cropped_img = get_crop_img_from_masks(
                match, crop_w, crop_h, single_mask=single_mask
            )
        else:
            cropped_img = get_crop_img(
                match.cam.filepath, match.x, match.y, crop_w, crop_h
            )
        pdf.image(encode_to_png_buffer(cropped_img), x=x, y=y, w=cell_w, h=image_h)

        # Position the text a bit below the image
        label_x = x + 2  # small offset from left edge
        label_y = y + image_h + 4  # 4 units below bottom of the image
        pdf.text(label_x, label_y, f"{match.annotation.id} {match.annotation.label}")

        # Move to the next column
        idx += 1
        if idx == n_cols:
            idx = 0
            row_idx += 1
            if row_idx == n_rows:
                pdf.add_page()
                row_idx = 0
                idx = 0

    pdf.output(output_filepath)
    print(f"PDF created: {output_filepath}")


def _pil_to_pdf(pdf, pil_img, x, y, max_w, max_h):
    """Place *pil_img* centred inside the cell (x, y, max_w, max_h), preserving aspect ratio."""
    try:
        iw, ih = pil_img.size
        aspect = iw / ih if ih > 0 else 1.0
        cell_aspect = max_w / max_h if max_h > 0 else 1.0
        if aspect > cell_aspect:
            scaled_w = max_w
            scaled_h = max_w / aspect
        else:
            scaled_h = max_h
            scaled_w = max_h * aspect
        img_x = x + (max_w - scaled_w) / 2.0
        img_y = y + (max_h - scaled_h) / 2.0
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        pdf.image(buf, x=img_x, y=img_y, w=scaled_w, h=scaled_h)
    except Exception as e:
        logger.warning("_pil_to_pdf failed: %s", e)
        pdf.rect(x, y, max_w, max_h)


def _draw_placeholder(pdf, x, y, w, h, text=""):
    """Draw a light-grey filled rectangle with an optional centred label."""
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(x, y, w, h, style="F")
    pdf.set_fill_color(0, 0, 0)
    if text:
        pdf.set_xy(x, y + h / 2 - 2)
        pdf.cell(w, 4, text, align="C")


def _ann_pcd_camera_view(ann):
    """Return a PIL Image of ann.simple_pcd viewed from the camera direction, or None."""
    try:
        from substrata.ortho import OrthoMap
        from PIL import ImageFilter

        if ann.simple_pcd is None or ann.image_match is None:
            return None
        cam_vec = ann.image_match.cam.vector
        view = OrthoMap(ann.simple_pcd, up_vector=cam_vec)
        img = view.show(highlights=ann, point_size=15)
        n_pts = max(len(ann.simple_pcd.points), 1)
        raw = int(np.sqrt(view.width * view.height / n_pts) * 2.5)
        filter_size = max(3, raw if raw % 2 == 1 else raw + 1)
        return img.filter(ImageFilter.MinFilter(size=filter_size))
    except Exception as e:
        logger.warning("_ann_pcd_camera_view failed for %s: %s", ann.id, e)
        return None


def _ann_pcd_top_view(ann, radius=None, up_vector=None, rotation=0):
    """Return a PIL Image of ann.simple_pcd as a top-down OrthoMap, or None.

    Args:
        ann: Annotation with a ``simple_pcd`` attribute.
        radius: Sampling radius in metres; when set the highlight circle uses
            ``point_size_metres`` with a red outline and transparent fill.
        up_vector: Up vector for the OrthoMap projection.  Defaults to
            ``[0, 0, 1]`` when ``None``.
        rotation: Clockwise rotation in degrees applied to the output image.
    """
    try:
        from substrata.ortho import OrthoMap
        from PIL import ImageFilter

        if ann.simple_pcd is None:
            return None
        view = OrthoMap(ann.simple_pcd, up_vector=up_vector, rotation=rotation)
        kwargs = dict(highlights=ann, point_size=15)
        if radius is not None:
            kwargs.update(
                point_size_metres=radius * 2,
                point_color=None,
                point_outline=(255, 0, 0),
            )
        img = view.show(**kwargs)
        n_pts = max(len(ann.simple_pcd.points), 1)
        raw = int(np.sqrt(view.width * view.height / n_pts) * 2.5)
        filter_size = max(3, raw if raw % 2 == 1 else raw + 1)
        return img.filter(ImageFilter.MinFilter(size=filter_size))
    except Exception as e:
        logger.warning("_ann_pcd_top_view failed for %s: %s", ann.id, e)
        return None


def plot_measurements(
    annotations,
    measurements: Optional[List[str]] = None,
    cols: int = 4,
    width: int = 16,
    height_per_row: int = 4,
    title: Optional[str] = None,
) -> Any:
    """Box plots of numeric measurements grouped by Annotation.label.

    Args:
        annotations: Annotations container whose items each have a ``label``
            attribute and a ``measurements`` dict.
        measurements: Optional list of measurement keys to include. When None,
            all numeric (scalar) keys are auto-detected (``_image`` keys are
            always excluded).
        cols: Maximum number of subplots per row.
        width: Figure width in inches.
        height_per_row: Height in inches per subplot row.
        title: Optional overall figure title.

    Returns:
        matplotlib Figure.
    """
    import math

    # --- 1. Collect numeric measurement keys ---
    all_keys: set = set()
    for ann in annotations.data.values():
        for k, v in ann.measurements.items():
            if "_image" in k:
                continue
            if np.isscalar(v) and not isinstance(v, (bool, str)):
                all_keys.add(k)

    if measurements is not None:
        measurement_keys = [k for k in measurements if k in all_keys]
    else:
        measurement_keys = sorted(all_keys)

    if not measurement_keys:
        raise ValueError("No numeric measurements found in the provided Annotations.")

    # --- 2. Collect values grouped by label ---
    unique_labels = sorted(
        {
            getattr(ann, "label", None)
            for ann in annotations.data.values()
            if getattr(ann, "label", None) is not None
        }
    )

    data: Dict[str, Dict[str, list]] = {
        key: {label: [] for label in unique_labels} for key in measurement_keys
    }
    for ann in annotations.data.values():
        label = getattr(ann, "label", None)
        if label is None:
            continue
        for key in measurement_keys:
            v = ann.measurements.get(key)
            if v is not None and np.isscalar(v) and not isinstance(v, (bool, str)):
                data[key][label].append(float(v))

    # --- 3. Build subplot grid ---
    n_plots = len(measurement_keys)
    n_cols = min(cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height_per_row * n_rows), squeeze=False
    )

    # Assign a distinct color per label
    cmap_labels = plt.cm.tab10 if len(unique_labels) <= 10 else plt.cm.tab20
    label_colors = {
        label: cmap_labels(i / max(len(unique_labels) - 1, 1))
        for i, label in enumerate(unique_labels)
    }

    rng = np.random.default_rng(0)

    # --- 4. Draw box plots ---
    for idx, key in enumerate(measurement_keys):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        plot_data = [data[key][label] for label in unique_labels]
        bp = ax.boxplot(
            plot_data,
            labels=unique_labels,
            sym="",                 # suppress default fliers; we draw jitter instead
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )

        for i, (label, box) in enumerate(zip(unique_labels, bp["boxes"])):
            color = label_colors[label]
            box.set(facecolor=(*color[:3], 0.25), edgecolor=color)
            bp["whiskers"][2 * i].set_color(color)
            bp["whiskers"][2 * i + 1].set_color(color)
            bp["caps"][2 * i].set_color(color)
            bp["caps"][2 * i + 1].set_color(color)

            vals = plot_data[i]
            n = len(vals)
            if n > 0:
                jitter = rng.uniform(-0.2, 0.2, size=n)
                ax.scatter(
                    np.full(n, i + 1) + jitter,
                    vals,
                    color=color,
                    s=15,
                    alpha=0.6,
                    edgecolors="none",
                    zorder=3,
                )
        tick_labels = [
            f"{label}\nn={len(plot_data[i])}" for i, label in enumerate(unique_labels)
        ]
        ax.set_xticklabels(tick_labels)
        ax.set_title(key, fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45 if len(unique_labels) > 4 else 0)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_rasterized(True)

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def save_measurement_visualizations_to_pdf(
    annotations,
    output_filepath,
    orthomap=None,
    radius: float = None,
):
    """Save per-annotation measurement visualizations to a landscape A4 PDF.

    Each annotation occupies one page with three rows:

    * **Row 1** – OrthoMap overview (left ¾) + annotation metadata text (right ¼).
    * **Row 2** – Full camera image with annotation marked | zoomed crop |
      point-cloud rendered from the camera direction.
    * **Row 3** – Measurement images (one column each) + scalar values column.

    Args:
        annotations: Annotations object whose items will be rendered.
        output_filepath: Destination PDF path.
        orthomap: Optional shared :class:`~substrata.ortho.OrthoMap`.  When
            provided the annotation location is highlighted on the map in row 1.
        radius: Sampling radius in metres.  When provided, the OrthoMap
            highlight circle diameter equals ``radius * 2`` metres and the
            zoomed-crop column 2 is masked to the same radius (greying out
            pixels beyond it).
    """
    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(False)
    pdf.set_font("Arial", size=7)

    margin = 10
    # A4 landscape: 297 × 210 mm  →  usable 277 × 190 mm
    usable_w = 297 - 2 * margin
    usable_h = 210 - 2 * margin
    row_h = usable_h / 3
    label_gap = 3  # mm below an image before the text label
    label_h = 4  # mm for a single text row

    standard_order = [
        "gapF_image",
        "elevation_image",
        "roughness_image",
        "vector_dispersion_image",
    ]

    for ann in annotations.data.values():
        pdf.add_page(orientation="L")
        pdf.set_font("Arial", size=7)
        pdf.set_text_color(0, 0, 0)

        # Use per-annotation radius when no global radius was supplied.
        eff_radius = radius if radius is not None else getattr(ann, "radius", None)

        # ── Row 1: OrthoMap + metadata ────────────────────────────────────
        r1_y = margin
        ortho_w = usable_w * 3 / 4
        text_x = margin + ortho_w
        text_w = usable_w / 4

        if orthomap is not None:
            try:
                ortho_kwargs = dict(highlights=ann, width=1200, height=600)
                if eff_radius is not None:
                    ortho_kwargs.update(
                        point_size_metres=eff_radius * 2,
                        point_color=None,
                        point_outline=(255, 0, 0),
                    )
                ortho_img = orthomap.show(**ortho_kwargs)
                _pil_to_pdf(pdf, ortho_img, margin, r1_y, ortho_w, row_h)
            except Exception as e:
                logger.warning("OrthoMap render failed for %s: %s", ann.id, e)
                _draw_placeholder(pdf, margin, r1_y, ortho_w, row_h, "OrthoMap error")
        else:
            _draw_placeholder(pdf, margin, r1_y, ortho_w, row_h, "No OrthoMap")

        # Metadata text block
        try:
            depth_str = f"{ann.depth_in_m:.4f} m"
        except Exception:
            depth_str = "N/A"

        coords_str = (
            f"[{ann.coords[0]:.4f}, {ann.coords[1]:.4f}, {ann.coords[2]:.4f}]"
            if ann.coords is not None
            else "N/A"
        )
        orig_coords_str = (
            f"[{ann.orig_coords[0]:.4f}, {ann.orig_coords[1]:.4f}, {ann.orig_coords[2]:.4f}]"
            if getattr(ann, "orig_coords", None) is not None
            else "N/A"
        )

        try:
            n_pts = (
                len(ann.simple_pcd.points)
                if getattr(ann, "simple_pcd", None) is not None
                else "N/A"
            )
        except Exception:
            n_pts = "N/A"

        meta_lines = [
            ("ID", str(ann.id) if ann.id else "N/A"),
            ("Label", str(ann.label) if getattr(ann, "label", None) else "N/A"),
            ("Coords", coords_str),
            ("Orig coords", orig_coords_str),
            ("Depth", depth_str),
            ("Subsampled pts", str(n_pts)),
        ]
        line_h = 6
        pdf.set_font("Arial", "B", size=7)
        pdf.set_xy(text_x + 2, r1_y + 3)
        pdf.cell(text_w - 4, line_h, "Annotation info", ln=1)
        pdf.set_font("Arial", size=7)
        for key, val in meta_lines:
            pdf.set_xy(text_x + 2, pdf.get_y())
            pdf.multi_cell(text_w - 4, line_h * 0.8, f"{key}: {val}", border=0)

        # ── Row 2: Camera views ───────────────────────────────────────────
        r2_y = margin + row_h
        col_w = usable_w / 3
        img_h2 = row_h - label_gap - label_h

        ann_has_image_match = (
            getattr(ann, "image_match", None) is not None
            and getattr(ann.image_match, "filepath", None) is not None
        )

        # Col 1: full camera image, oriented + square-cropped, annotation marked
        col1_x = margin
        if ann_has_image_match:
            try:
                highlight_radius = 50
                if eff_radius is not None:
                    try:
                        ppm = ann.image_match.pixels_per_mm
                        if ppm is not None:
                            highlight_radius = max(1, int(eff_radius * 1000.0 * ppm))
                    except Exception:
                        pass
                full_img = ann.image_match.cam.render(
                    highlight_pixels=[ann.image_match.x, ann.image_match.y],
                    orient=True,
                    square=True,
                    highlight_radius=highlight_radius,
                )
                if full_img is not None:
                    _pil_to_pdf(pdf, full_img, col1_x, r2_y, col_w, img_h2)
                else:
                    _draw_placeholder(pdf, col1_x, r2_y, col_w, img_h2, "Image error")
            except Exception as e:
                logger.warning("Full image render failed for %s: %s", ann.id, e)
                _draw_placeholder(pdf, col1_x, r2_y, col_w, img_h2, "Image error")
        else:
            _draw_placeholder(pdf, col1_x, r2_y, col_w, img_h2, "No image match")
        pdf.set_xy(col1_x + 2, r2_y + img_h2 + label_gap)
        pdf.cell(col_w - 4, label_h, "full image", ln=0)

        # Col 2: zoomed crop via ImageMatch.render()
        # If a sampling radius is provided and no circular mask has been set yet,
        # create one so that render() can auto-apply the grey-out.
        col2_x = margin + col_w
        if ann_has_image_match:
            try:
                if eff_radius is not None:
                    existing = getattr(ann.image_match, "mask", None)
                    already_circular = (
                        existing is not None
                        and hasattr(existing, "radius_m")
                        and existing.radius_m is not None
                    )
                    if not already_circular:
                        try:
                            ann.image_match.create_circular_masks([eff_radius])
                        except Exception:
                            pass
                crop_pil = ann.image_match.render(crop_w=1000, crop_h=1000, orient=True)
                _pil_to_pdf(pdf, crop_pil, col2_x, r2_y, col_w, img_h2)
            except Exception as e:
                logger.warning("Zoomed crop failed for %s: %s", ann.id, e)
                _draw_placeholder(pdf, col2_x, r2_y, col_w, img_h2, "Crop error")
        else:
            _draw_placeholder(pdf, col2_x, r2_y, col_w, img_h2, "No image match")
        pdf.set_xy(col2_x + 2, r2_y + img_h2 + label_gap)
        pdf.cell(col_w - 4, label_h, "zoomed crop", ln=0)

        # Col 3: top-down point cloud view
        col3_x = margin + 2 * col_w
        up_vec = getattr(orthomap, "_up_vector", None) if orthomap is not None else None
        pcd_img = _ann_pcd_top_view(ann, radius=eff_radius, up_vector=up_vec, rotation=90)
        if pcd_img is not None:
            _pil_to_pdf(pdf, pcd_img, col3_x, r2_y, col_w, img_h2)
        else:
            _draw_placeholder(pdf, col3_x, r2_y, col_w, img_h2, "No point cloud")
        pdf.set_xy(col3_x + 2, r2_y + img_h2 + label_gap)
        pdf.cell(col_w - 4, label_h, "point cloud view", ln=0)

        # ── Row 3: Measurements ───────────────────────────────────────────
        r3_y = margin + 2 * row_h

        meas = getattr(ann, "measurements", {}) or {}
        image_keys_all = [k for k, v in meas.items() if "_image" in k and v is not None]
        ordered_img_keys = [k for k in standard_order if k in image_keys_all]
        ordered_img_keys += sorted(k for k in image_keys_all if k not in standard_order)

        scalar_meas = {
            k: v
            for k, v in meas.items()
            if "_image" not in k
            and v is not None
            and isinstance(v, (int, float, np.integer, np.floating))
        }
        extra = getattr(ann, "extra_coords", None) or {}

        has_images = bool(ordered_img_keys)
        has_scalars = bool(scalar_meas or extra)

        if has_images or has_scalars:
            n_img_cols = len(ordered_img_keys)
            n_r3_cols = n_img_cols + 1  # +1 for scalar column
            cell_w3 = usable_w / n_r3_cols
            img_h3 = row_h - label_gap - label_h

            for ci, img_key in enumerate(ordered_img_keys):
                cx = margin + ci * cell_w3
                img_val = meas[img_key]
                try:
                    if isinstance(img_val, np.ndarray):
                        arr = (
                            img_val
                            if img_val.dtype == np.uint8
                            else (
                                (np.clip(img_val, 0, 1) * 255).astype(np.uint8)
                                if img_val.max() <= 1.0
                                else img_val.astype(np.uint8)
                            )
                        )
                        pil_v = Image.fromarray(arr)
                    elif isinstance(img_val, Image.Image):
                        pil_v = img_val
                    elif hasattr(img_val, "to_image"):
                        raw = img_val.to_image(format="png", width=600, height=400)
                        pil_v = Image.open(BytesIO(raw)).convert("RGB")
                    else:
                        pil_v = Image.fromarray(np.array(img_val))
                    if img_key == "gapF_image":
                        arr_rgb = np.array(pil_v.convert("RGB"))
                        arr_rgb[np.all(arr_rgb == 0, axis=-1)] = 255
                        pil_v = Image.fromarray(arr_rgb).rotate(90, expand=True)
                    _pil_to_pdf(pdf, pil_v, cx, r3_y, cell_w3, img_h3)
                except Exception as e:
                    logger.warning(
                        "Meas image %s failed for %s: %s", img_key, ann.id, e
                    )
                    _draw_placeholder(pdf, cx, r3_y, cell_w3, img_h3)
                pdf.set_xy(cx + 2, r3_y + img_h3 + label_gap)
                pdf.cell(cell_w3 - 4, label_h, img_key, ln=0)

            # Scalar text column (always last)
            sc_x = margin + n_img_cols * cell_w3
            pdf.set_fill_color(245, 245, 245)
            pdf.rect(sc_x, r3_y, cell_w3, row_h, style="F")
            pdf.set_fill_color(0, 0, 0)
            pdf.set_font("Arial", "B", size=6)
            pdf.set_xy(sc_x + 2, r3_y + 2)
            pdf.cell(cell_w3 - 4, 4, "Measurements", ln=1)
            pdf.set_font("Arial", size=6)
            for k, v in scalar_meas.items():
                if pdf.get_y() > r3_y + row_h - 4:
                    break
                pdf.set_xy(sc_x + 2, pdf.get_y())
                val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
                pdf.multi_cell(cell_w3 - 4, 3.5, f"{k}: {val_str}", border=0)
            if extra:
                pdf.set_font("Arial", "B", size=6)
                pdf.set_xy(sc_x + 2, pdf.get_y())
                pdf.cell(cell_w3 - 4, 4, "Extra coords", ln=1)
                pdf.set_font("Arial", size=6)
                for k, v in extra.items():
                    if pdf.get_y() > r3_y + row_h - 4:
                        break
                    pdf.set_xy(sc_x + 2, pdf.get_y())
                    pdf.multi_cell(cell_w3 - 4, 3.5, f"{k}: {v}", border=0)
            pdf.set_font("Arial", size=7)

    pdf.output(output_filepath)
    logger.info("PDF created: %s", output_filepath)


def get_crop_img_from_masks(  # TODO: needs PIL version
    image_match,
    output_img_w=1000,
    output_img_h=1000,
    pad_ratio=0.0,
    contour_thickness=2,
    annotation_radius=20,
    annotation_color=(0, 255, 0),
    single_mask=False,
):
    """
    Crop an image based on a list of SAM2 masks, draw each mask's contours in
    red, green, and blue (first mask with double thickness), annotate the
    surface area, and mark the annotation point.
    """
    # Load image
    img = cv2.imread(image_match.filepath)
    if img is None:
        raise IOError(f"Could not read image: {image_match.filepath}")
    h_img, w_img = img.shape[:2]

    # If single_mask is True, only the chosen mask is processed
    if single_mask:
        image_match_masks = [image_match.mask]
    else:
        image_match_masks = image_match.masks

    # Compute union bounding box from all masks
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for m in image_match_masks:
        ys, xs = np.where(m.vals)
        if ys.size and xs.size:
            x_mins.append(xs.min())
            x_maxs.append(xs.max())
            y_mins.append(ys.min())
            y_maxs.append(ys.max())
    if not x_mins:
        raise ValueError("No valid mask regions found.")
    x_min, x_max = int(min(x_mins)), int(max(x_maxs))
    y_min, y_max = int(min(y_mins)), int(max(y_maxs))
    bbox_w, bbox_h = x_max - x_min + 1, y_max - y_min + 1

    # Adjust bbox to desired aspect ratio (expanding only)
    bbox_aspect, desired_aspect = bbox_w / bbox_h, output_img_w / output_img_h
    if bbox_aspect < desired_aspect:
        new_w = int(np.ceil(bbox_h * desired_aspect))
        center = (x_min + x_max) // 2
        x_min_adj = center - new_w // 2
        x_max_adj = x_min_adj + new_w - 1
        y_min_adj, y_max_adj = y_min, y_max
    else:
        new_h = int(np.ceil(bbox_w / desired_aspect))
        center = (y_min + y_max) // 2
        y_min_adj = center - new_h // 2
        y_max_adj = y_min_adj + new_h - 1
        x_min_adj, x_max_adj = x_min, x_max

    # Add padding and clamp to image boundaries
    adj_w = x_max_adj - x_min_adj + 1
    adj_h = y_max_adj - y_min_adj + 1
    pad = int(np.round(min(adj_w, adj_h) * pad_ratio))
    x_min_final = np.clip(x_min_adj - pad, 0, w_img - 1)
    y_min_final = np.clip(y_min_adj - pad, 0, h_img - 1)
    x_max_final = np.clip(x_max_adj + pad, 0, w_img - 1)
    y_max_final = np.clip(y_max_adj + pad, 0, h_img - 1)

    # Crop and resize image
    crop = img[y_min_final : y_max_final + 1, x_min_final : x_max_final + 1]
    crop_resized = cv2.resize(
        crop, (output_img_w, output_img_h), interpolation=cv2.INTER_AREA
    )

    # Colors for contours: red, green, blue
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Process each mask: crop, resize, find contours, and draw them
    for idx, m in enumerate(image_match_masks):
        mask_crop = m.vals[y_min_final : y_max_final + 1, x_min_final : x_max_final + 1]
        mask_resized = cv2.resize(
            mask_crop, (output_img_w, output_img_h), interpolation=cv2.INTER_NEAREST
        )
        mask_bin = ((mask_resized > 0).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Chosen mask gets double thickness
        if m == image_match.mask:
            thick = contour_thickness * 5
        else:
            thick = contour_thickness
        color = colors[idx % len(colors)]
        cv2.drawContours(crop_resized, contours, -1, color, thickness=thick)

    # Overlay surface area (in cm2) of chosen mask (default = first mask)
    overlay_text = f"SA: {image_match.mask.area_in_cm2:.4f} cm2"
    font, font_scale, text_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    text_size, _ = cv2.getTextSize(overlay_text, font, font_scale, text_thickness)
    cv2.putText(
        crop_resized,
        overlay_text,
        (10, text_size[1] + 10),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    # Map the annotation point to the crop coordinate system
    crop_orig_w = x_max_final - x_min_final + 1
    crop_orig_h = y_max_final - y_min_final + 1
    scale_x = output_img_w / crop_orig_w
    scale_y = output_img_h / crop_orig_h
    ann_x = int(round((image_match.x - x_min_final) * scale_x))
    ann_y = int(round((image_match.y - y_min_final) * scale_y))
    cv2.circle(
        crop_resized, (ann_x, ann_y), annotation_radius, annotation_color, thickness=-1
    )

    return crop_resized


def show_grid_cells(
    pcd, bboxes, show_colors=False, cell_size=None, sub_divisions=4, show_rep_cell=True
):
    """
    Visualize the grid cells and the main connected component as determined by spread filtering.

    Two visualizations are produced in one figure with two panels:
      - Left panel (10×3): A scatter plot of all x-y points (colored by the point cloud colors)
        with the boundaries of each filtered cell overlaid in red.
      - Right panel (3×3): For one of the filtered cells (the middle one in the list), a plot showing
        its internal subdivisions (drawn in green) overlaid on the points within that cell.
    """
    # If the input is a single bounding box, wrap it in a list.
    if isinstance(bboxes, (list, tuple)):
        # Check if bboxes is a single bounding box by verifying it has two elements and both are list/tuple with length 2.
        if (
            len(bboxes) == 2
            and isinstance(bboxes[0], (list, tuple))
            and isinstance(bboxes[1], (list, tuple))
        ):
            if len(bboxes[0]) == 2 and len(bboxes[1]) == 2:
                bboxes = [bboxes]
                show_rep_cell = False

    # Create a figure with two panels: left panel 10x3, right panel 3x3.
    fig = plt.figure(figsize=(13, 3))  # Total width=13 (10+3) inches, height=3 inches.
    gs = fig.add_gridspec(1, 2, width_ratios=[10, 3])

    # Left panel: All points with filtered grid cell boundaries.
    ax_left = fig.add_subplot(gs[0, 0])
    if show_colors:
        pcd_colors = np.asarray(pcd.colors)
    else:
        pcd_colors = None
    ax_left.scatter(pcd.points[:, 0], pcd.points[:, 1], s=1, c=pcd_colors, alpha=0.5)
    for bbox in bboxes:
        min_corner, max_corner = bbox
        x_vals = [
            min_corner[0],
            max_corner[0],
            max_corner[0],
            min_corner[0],
            min_corner[0],
        ]
        y_vals = [
            min_corner[1],
            min_corner[1],
            max_corner[1],
            max_corner[1],
            min_corner[1],
        ]
        ax_left.plot(x_vals, y_vals, "r-", linewidth=1)
    ax_left.set_xlabel("X")
    ax_left.set_ylabel("Y")
    # Title: include overall box as bottom-left and top-right [[x,y],[x,y]]
    try:
        if bboxes and len(bboxes) > 0:
            # Normalize potential single-box input
            if (
                isinstance(bboxes, (list, tuple))
                and len(bboxes) == 2
                and isinstance(bboxes[0], (list, tuple))
                and isinstance(bboxes[1], (list, tuple))
                and len(bboxes[0]) == 2
                and len(bboxes[1]) == 2
                and not (
                    isinstance(bboxes[0][0], (list, tuple))
                    or isinstance(bboxes[1][0], (list, tuple))
                )
            ):
                boxes_iter = [bboxes]
            else:
                boxes_iter = bboxes

            min_x = min(box[0][0] for box in boxes_iter)
            min_y = min(box[0][1] for box in boxes_iter)
            max_x = max(box[1][0] for box in boxes_iter)
            max_y = max(box[1][1] for box in boxes_iter)
            ax_left.set_title(
                f"Box coordinates: [[{min_x:.1f},{min_y:.1f}],[{max_x:.1f},{max_y:.1f}]]"
            )
        else:
            ax_left.set_title("Overall plot with grid cell boundaries")
    except Exception:
        ax_left.set_title("Overall plot with grid cell boundaries")
    ax_left.axis("equal")

    # Right panel: For one cell, show internal subdivisions.
    if show_rep_cell:
        ax_right = fig.add_subplot(gs[0, 1])
        if bboxes:
            middle_index = len(bboxes) // 2
            min_corner, max_corner = bboxes[middle_index]
            # Filter points within this cell.
            cell_mask = (
                (pcd.points[:, 0] >= min_corner[0])
                & (pcd.points[:, 0] < max_corner[0])
                & (pcd.points[:, 1] >= min_corner[1])
                & (pcd.points[:, 1] < max_corner[1])
            )
            cell_points = pcd.points[cell_mask]
            if pcd_colors is not None:
                cell_colors = pcd_colors[cell_mask]
            else:
                cell_colors = "b"

            ax_right.scatter(
                cell_points[:, 0], cell_points[:, 1], s=1, c=cell_colors, alpha=0.5
            )
            # Draw the cell boundary.
            x_vals = [
                min_corner[0],
                max_corner[0],
                max_corner[0],
                min_corner[0],
                min_corner[0],
            ]
            y_vals = [
                min_corner[1],
                min_corner[1],
                max_corner[1],
                max_corner[1],
                min_corner[1],
            ]
            ax_right.plot(x_vals, y_vals, "r-", linewidth=1)

            # Draw subdivisions.
            if cell_size is None:
                cell_size = max_corner[0] - min_corner[0]

            sub_cell_size = cell_size / sub_divisions
            for m in range(sub_divisions):
                for n in range(sub_divisions):
                    sub_x_min = min_corner[0] + m * sub_cell_size
                    sub_y_min = min_corner[1] + n * sub_cell_size
                    sub_x_max = sub_x_min + sub_cell_size
                    sub_y_max = sub_y_min + sub_cell_size
                    x_vals = [sub_x_min, sub_x_max, sub_x_max, sub_x_min, sub_x_min]
                    y_vals = [sub_y_min, sub_y_min, sub_y_max, sub_y_max, sub_y_min]
                    ax_right.plot(x_vals, y_vals, "g-", linewidth=0.5)

            ax_right.set_xlabel("X")
            ax_right.set_ylabel("Y")
            ax_right.set_title("Representative cell from the middle")
            ax_right.axis("equal")

    plt.tight_layout()
    return plt


def show_classified_grid_cells(
    pcd,
    bboxes,
    annotations,
    show_points=False,
    point_size=1,
    title=None,
    label_colors=None,
    max_output_points=50000,
):
    """
    Show a 2D plot with grid cells colored by majority classification.

    Given a point cloud, a list of grid cell bounding boxes, and an
    `Annotations` object, determine for each cell which annotations fall
    within the cell bounds and color the cell based on the majority
    classification label from `annotation.image_match.classification['label']`.

    - Cells with no classified annotations are colored gray.
    - Grid cell outlines are not drawn; only the filled cell color is shown.

    Args:
        pcd: Point cloud (SimplePointCloud or PointCloud).
        bboxes: List of bounding boxes, where each item is
            (min_corner[x, y], max_corner[x, y]). A single bbox can also be
            passed as ((x_min, y_min), (x_max, y_max)).
        annotations: An `Annotations` instance.
        show_points (bool): If True, scatter the XY points as a background.
        point_size (int): Marker size for background points when shown.
        title (str | None): Optional title for the plot.
        label_colors (dict | None): Optional mapping {label: matplotlib color}.

    Returns:
        matplotlib.pyplot: The pyplot module for further manipulation or display.
    """
    # Normalize bboxes input: accept a single bbox as well
    if (
        isinstance(bboxes, (list, tuple))
        and len(bboxes) == 2
        and isinstance(bboxes[0], (list, tuple))
        and isinstance(bboxes[1], (list, tuple))
        and len(bboxes[0]) == 2
        and len(bboxes[1]) == 2
        and not (
            isinstance(bboxes[0][0], (list, tuple))
            or isinstance(bboxes[1][0], (list, tuple))
        )
    ):
        bboxes = [bboxes]

    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_aspect("equal")
    ax_right = fig.add_subplot(gs[0, 1])

    # Optional background points in grayscale
    if show_points and len(pcd.points) > 0:
        plot_cols = np.full(
            (pcd.points.shape[0], 3), 0.6, dtype=float
        )  # Default to grayscale
        ax_left.scatter(
            pcd.points[:, 0],
            pcd.points[:, 1],
            s=point_size,
            c=plot_cols,
            alpha=0.5,
            edgecolor="none",
        )

    # Build default label color mapping if not provided
    if label_colors is None:
        # Collect labels present in annotations
        labels = []
        for ann in annotations.data.values():
            im = getattr(ann, "image_match", None)
            cls = getattr(im, "classification", None)
            if isinstance(cls, dict) and "label" in cls and cls["label"] is not None:
                labels.append(str(cls["label"]))
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab20", max(1, len(unique_labels)))
        label_colors = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    # Helper to compute majority label for a bbox
    def majority_label_for_bbox(min_corner, max_corner):
        counts = {}
        for ann in annotations.data.values():
            x, y = ann.coords[0], ann.coords[1]
            if (min_corner[0] <= x < max_corner[0]) and (
                min_corner[1] <= y < max_corner[1]
            ):
                im = getattr(ann, "image_match", None)
                cls = getattr(im, "classification", None)
                if isinstance(cls, dict):
                    lbl = cls.get("label", None)
                else:
                    lbl = None
                if lbl is not None:
                    lbl = str(lbl)
                    counts[lbl] = counts.get(lbl, 0) + 1
        if not counts:
            return None
        # Return the label with highest count (break ties deterministically by label name)
        max_count = max(counts.values())
        candidates = sorted([lbl for lbl, c in counts.items() if c == max_count])
        return candidates[0] if candidates else None

    # Draw filled rectangles for each bbox colored by majority label
    labels_present = set()
    label_counts_by_cell = {}
    print(f"Drawing {len(bboxes)} bounding boxes")
    for i, bbox in enumerate(bboxes):
        min_corner, max_corner = bbox
        label = majority_label_for_bbox(min_corner, max_corner)
        if label is None:
            face_color = (0.7, 0.7, 0.7)  # gray for missing
        else:
            face_color = label_colors.get(label, (0.7, 0.7, 0.7))
            labels_present.add(label)
            label_counts_by_cell[label] = label_counts_by_cell.get(label, 0) + 1

        if label is None:
            label_counts_by_cell["No data"] = label_counts_by_cell.get("No data", 0) + 1

        x_vals = [min_corner[0], max_corner[0]]
        y_vals = [min_corner[1], max_corner[1]]
        width = x_vals[1] - x_vals[0]
        height = y_vals[1] - y_vals[0]
        rect = plt.Rectangle(
            (x_vals[0], y_vals[0]),
            width,
            height,
            facecolor=face_color,
            edgecolor="none",
            alpha=0.6,
        )
        ax_left.add_patch(rect)

    # Set axis limits based on bbox data
    if bboxes:
        all_x_coords = []
        all_y_coords = []
        for bbox in bboxes:
            min_corner, max_corner = bbox
            all_x_coords.extend([min_corner[0], max_corner[0]])
            all_y_coords.extend([min_corner[1], max_corner[1]])

        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            print(
                f"Setting axis limits: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]"
            )
            ax_left.set_xlim(x_min, x_max)
            ax_left.set_ylim(y_min, y_max)

    if title is not None:
        ax_left.set_title(title)

    # Build legend: include only labels present plus a gray "No data"
    handles = []
    for lbl in sorted(labels_present):
        handles.append(
            mpatches.Patch(
                facecolor=label_colors.get(lbl, (0.7, 0.7, 0.7)),
                edgecolor="none",
                label=str(lbl),
            )
        )
    handles.append(
        mpatches.Patch(facecolor=(0.7, 0.7, 0.7), edgecolor="none", label="No data")
    )
    if len(handles) > 0:
        ax_left.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(handles), 6),
            frameon=False,
        )

    # Right panel: bar chart with counts per category (including No data)
    if label_counts_by_cell:
        categories = list(
            sorted([k for k in label_counts_by_cell.keys() if k != "No data"])
        )
        counts = [label_counts_by_cell[c] for c in categories]
        # Append No data at the end if present
        if "No data" in label_counts_by_cell:
            categories.append("No data")
            counts.append(label_counts_by_cell["No data"])

        bar_colors = [
            label_colors.get(c, (0.7, 0.7, 0.7)) if c != "No data" else (0.7, 0.7, 0.7)
            for c in categories
        ]
        x_pos = np.arange(len(categories))
        ax_right.bar(x_pos, counts, color=bar_colors)
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
        ax_right.set_ylabel("Count")
        ax_right.set_title("Cells per class")
        ax_right.margins(x=0.05)

    # Tight layout and return pyplot
    plt.tight_layout()

    # Match right plot area height to left plot area height
    try:
        left_pos = ax_left.get_position()
        right_pos = ax_right.get_position()
        ax_right.set_position(
            [right_pos.x0, left_pos.y0, right_pos.width, left_pos.height]
        )
    except Exception:
        pass
    return plt


def show_intercept_point(intercept_point):
    """
    Create a 3D scatter plot visualizing candidate points and key markers.
    Candidate points are colored by their Z value, and the following are
    plotted:
        - The query coordinate as a black square,
        - The computed intercept as a green square.
    A dashed line connects the query and selected point, and a 3D
    cylinder (centered at the query XY) represents the search radius.
    """
    xy_coord = intercept_point.estimated_intercept_coords[0:2]
    search_radius = intercept_point.search_radius
    candidates = np.array(intercept_point.simple_pcd.points)
    intercept = np.array(intercept_point.estimated_intercept_coords)
    selected = np.array(intercept_point.coords)

    # Create figure with two panels: left panel for 3D plot, right panel for 2D plot.
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    # Left panel: 3D scatter plot.
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")

    candidate_x = candidates[:, 0]
    candidate_y = candidates[:, 1]
    candidate_z = candidates[:, 2]

    intercept_z = intercept[2]

    # Determine colors based on z-value differences
    colors = np.where(
        np.abs(candidate_z - intercept_z) <= search_radius,
        "white",
        np.where(candidate_z < intercept_z - search_radius, "blue", "red"),
    )

    # Plot candidate points.
    ax_3d.scatter(
        candidate_x,
        candidate_y,
        candidate_z,
        c=colors,
        s=20,
        alpha=0.7,
        edgecolors="k",
    )

    # For visualization, assume query z equals intercept's z.
    query_z = intercept[2]
    ax_3d.scatter(
        xy_coord[0],
        xy_coord[1],
        query_z,
        c="k",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Plot computed intercept.
    ax_3d.scatter(
        intercept[0],
        intercept[1],
        intercept[2],
        c="g",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Draw dashed line from query to selected point.
    ax_3d.plot(
        [xy_coord[0], selected[0]],
        [xy_coord[1], selected[1]],
        [query_z, selected[2]],
        "k--",
        lw=1,
    )

    # Create a cylinder representing the search radius.
    # Cylinder is centered at (xy_coord[0], xy_coord[1]) and spans z range.
    theta = np.linspace(0, 2 * np.pi, 30)
    z_cyl = np.linspace(candidate_z.min(), candidate_z.max(), 30)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = xy_coord[0] + search_radius * np.cos(theta_grid)
    y_cyl = xy_coord[1] + search_radius * np.sin(theta_grid)

    ax_3d.plot_surface(x_cyl, y_cyl, z_grid, color="gray", alpha=0.2, edgecolor="none")

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Point Intercept: Query, Intercept, Selected")

    # Right panel: 2D scatter plot (XY plane).
    ax_2d = fig.add_subplot(gs[0, 1])

    # Plot candidate points.
    ax_2d.scatter(
        candidate_x,
        candidate_y,
        c=colors,
        s=20,
        alpha=0.7,
        edgecolors="k",
    )

    # Plot the query coordinate.
    ax_2d.scatter(
        xy_coord[0],
        xy_coord[1],
        c="k",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Plot the computed intercept.
    ax_2d.scatter(
        intercept[0],
        intercept[1],
        c="g",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Draw dashed line from query to selected point.
    ax_2d.plot(
        [xy_coord[0], selected[0]],
        [xy_coord[1], selected[1]],
        "k--",
        lw=1,
    )

    # Draw a circle representing the search radius.
    circle = plt.Circle(
        (xy_coord[0], xy_coord[1]), search_radius, color="k", fill=False, ls="--", lw=1
    )
    ax_2d.add_patch(circle)

    ax_2d.set_xlabel("X")
    ax_2d.set_ylabel("Y")
    ax_2d.set_title("2D Point Intercept: XY Plane")
    ax_2d.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()


def plot_2d_ortho(
    pcd,
    resolution=None,
    color_attr="colors",
    figsize=None,
    save_path=None,
    ax=None,
    title=None,
    show=True,
):
    """
    Create and display a top-down orthomosaic (splat) of a point cloud.

    Args:
        pcd: Input point cloud (supports PointCloud, SimplePointCloud, or o3d PointCloud-like).
        resolution (float | None): Ground sampling distance in meters per
            pixel. If None, choose a resolution based on point count and
            plot size.
        color_attr (str): Attribute for color ('colors' or 'intensities').
        figsize (tuple): Figure size for matplotlib (width, height) when creating a new figure.
        save_path (str | None): If provided, save the image to this file.
        ax (matplotlib.axes.Axes | None): Optional axes to draw into; if None, creates a new fig.
        title (str | None): Optional title for the plot.
        show (bool): Whether to call plt.show() when creating a new figure.

    Returns:
        tuple[np.ndarray, matplotlib.figure.Figure | None]: (image array, figure if created else None).
    """
    # Extract points
    pts = np.asarray(pcd.points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Compute bounds
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    extent_x = max(max_x - min_x, 1e-9)
    extent_y = max(max_y - min_y, 1e-9)

    # Choose resolution heuristically if not provided
    if resolution is None:
        n_pts = max(len(pts), 1)
        # Target total pixels ~ n_pts/10, bounded for practicality
        target_pixels = int(np.clip(n_pts / 10.0, 2e5, 2e6))
        aspect = extent_x / extent_y
        width_px = int(max(256, np.sqrt(target_pixels * max(aspect, 1e-6))))
        height_px = int(max(256, target_pixels / max(width_px, 1)))
        resolution = extent_x / width_px
    else:
        width_px = int(np.ceil(extent_x / resolution))
        height_px = int(np.ceil(extent_y / resolution))

    width = width_px
    height = height_px

    # Prepare splat and count buffers
    splat = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width), dtype=int)

    # Fetch colors or default to white
    if hasattr(pcd, color_attr):
        colors = pcd.colors
    else:
        colors = np.ones((len(pts), 3), dtype=np.float64)

    # Rasterize
    for pt, col in zip(pcd.points, pcd.colors):
        ix = int((pt[0] - min_x) / resolution)
        iy = int((pt[1] - min_y) / resolution)
        if 0 <= ix < width and 0 <= iy < height:
            splat[iy, ix] += col
            counts[iy, ix] += 1

    # Normalize and set background (no points) to white
    mask = counts > 0
    splat[mask] /= counts[mask][:, None]
    splat[~mask] = 1.0

    # Convert to uint8
    img = (np.clip(splat, 0, 1) * 255).astype(np.uint8)

    # Set figsize to match pixel size (1 inch = dpi pixels)
    if figsize is None:
        dpi = plt.rcParams.get("figure.dpi", 100)
        figsize = (width / dpi, height / dpi)

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(img, origin="lower")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    if save_path is not None and created_fig:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if created_fig and show:
        plt.show()
    if created_fig and not show:
        plt.close(fig)

    return img, (fig if created_fig else None)


def plot_camera_view_from_pcd(
    pcd,
    image_match,
    resolution=0.005,
    color_attr="colors",
    figsize=None,
    save_path=None,
    show_plot=True,
    crop_to_mask=True,
):
    """
    Create a 2D rasterized view of a point cloud from a camera's perspective.

    This function projects the point cloud onto the camera's image plane and
    creates a rasterized view. If crop_to_mask is True, it crops to the area
    indicated by the ImageMatch's mask.

    Args:
        pcd: Point cloud (SimplePointCloud or Open3D point cloud)
        image_match: ImageMatch object with camera reference and mask information
        resolution (float): Ground sampling distance (m per pixel)
        color_attr (str): Attribute for color ('colors' or 'intensities')
        figsize (tuple): Figure size for matplotlib (width, height)
        save_path (str): Optional path to save the visualization
        show_plot (bool): Whether to display the plot
        crop_to_mask (bool): Whether to crop to the mask area

    Returns:
        tuple: (fig, raster_img) - matplotlib figure and rasterized image array
    """
    # Convert to Open3D point cloud if necessary
    if isinstance(pcd, pointclouds.SimplePointCloud):
        o3d_pcd = pcd.get_o3d_pcd()
    else:
        o3d_pcd = pcd.o3d_pcd

    points = np.asarray(o3d_pcd.points)

    # Get camera from image_match
    cam = image_match.cam

    # Get camera transform matrix
    if hasattr(cam, "transform") and cam.transform is not None:
        transform = cam.transform
    else:
        # Create transform from camera vector and position
        if hasattr(cam, "vector") and cam.vector is not None:
            # Use camera vector to create a simple transform
            # This is a simplified approach - you might want to enhance this
            transform = np.eye(4)
            transform[:3, 3] = cam.coords  # Set translation
            # Note: This doesn't set rotation based on vector - you'd need more complex logic
        else:
            raise ValueError("Camera must have either transform or vector attribute")

    # Transform points to camera coordinate system
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (transform @ points_homogeneous.T).T[:, :3]

    # Project to 2D (simple orthographic projection for now)
    # You might want to implement proper perspective projection here
    points_2d = points_cam[:, :2]  # Just take X,Y coordinates

    # Debug: Check if we have any valid points
    print(f"Total points: {len(points_2d)}")
    print(
        f"Points 2D bounds: X[{points_2d[:, 0].min():.3f}, {points_2d[:, 0].max():.3f}], Y[{points_2d[:, 1].min():.3f}, {points_2d[:, 1].max():.3f}]"
    )

    # Get colors
    if hasattr(o3d_pcd, color_attr):
        colors = np.asarray(getattr(o3d_pcd, color_attr))
        if colors.ndim == 1:
            colors = np.vstack((colors, colors, colors)).T
    else:
        colors = np.ones((points.shape[0], 3), dtype=np.float64)

    # Determine bounds
    min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
    min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()

    print(f"Initial bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")

    # If crop_to_mask is True and image_match has a mask
    if crop_to_mask and hasattr(image_match, "mask") and image_match.mask is not None:
        # Get mask bounds in image coordinates
        mask_vals = image_match.mask.vals
        mask_height, mask_width = mask_vals.shape

        # Convert pixel coordinates to 3D world coordinates using camera projection
        if hasattr(image_match, "x") and hasattr(image_match, "y"):
            # Use camera's pixel_to_point method to get 3D world coordinates
            world_coords, _, _ = cam.pixel_to_point(image_match.x, image_match.y, pcd)

            if world_coords is not None:
                # Use the 3D world coordinates as center
                center_x, center_y = world_coords[0], world_coords[1]

                # Estimate scale based on pixel scale if available
                if hasattr(image_match, "pixel_scale") and image_match.pixel_scale:
                    scale = image_match.pixel_scale
                else:
                    scale = resolution  # Use resolution as fallback

                # Calculate bounds based on mask size
                half_width = (mask_width / 2) * scale
                half_height = (mask_height / 2) * scale

                min_x = center_x - half_width
                max_x = center_x + half_width
                min_y = center_y - half_height
                max_y = center_y + half_height

                print(
                    f"Cropped bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]"
                )
                print(f"Center: ({center_x:.3f}, {center_y:.3f}), Scale: {scale:.6f}")
                print(f"Mask size: {mask_width}x{mask_height} pixels")

    # Compute image size
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # Ensure minimum size
    width = max(width, 100)
    height = max(height, 100)

    # Prepare raster and count buffers
    raster = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width), dtype=int)

    # Rasterize points
    points_in_bounds = 0
    for pt, col in zip(points_2d, colors):
        ix = int((pt[0] - min_x) / resolution)
        iy = int((pt[1] - min_y) / resolution)
        if 0 <= ix < width and 0 <= iy < height:
            raster[iy, ix] += col
            counts[iy, ix] += 1
            points_in_bounds += 1

        print(f"Raster size: {width}x{height}")
    print(f"Points in bounds: {points_in_bounds}/{len(points_2d)}")
    print(f"Non-zero pixels: {np.sum(counts > 0)}")

    # If no points in bounds, fall back to full bounds
    if points_in_bounds == 0:
        print("No points in bounds, using full point cloud bounds")
        min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
        min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))
        width = max(width, 100)
        height = max(height, 100)
        raster = np.zeros((height, width, 3), dtype=np.float64)
        counts = np.zeros((height, width), dtype=int)

        # Re-rasterize with full bounds
        for pt, col in zip(points_2d, colors):
            ix = int((pt[0] - min_x) / resolution)
            iy = int((pt[1] - min_y) / resolution)
            if 0 <= ix < width and 0 <= iy < height:
                raster[iy, ix] += col
                counts[iy, ix] += 1

    # Normalize
    mask = counts > 0
    raster[mask] /= counts[mask][:, None]

    # Convert to uint8
    raster_img = (np.clip(raster, 0, 1) * 255).astype(np.uint8)

    # Set figsize
    if figsize is None:
        dpi = plt.rcParams.get("figure.dpi", 100)
        figsize = (width / dpi, height / dpi)

    # Create visualization
    fig = plt.figure(figsize=figsize)
    plt.imshow(raster_img, extent=[min_x, max_x, min_y, max_y], origin="lower")
    plt.title(
        f"Camera View: {cam.cam_id} - Match at pixel ({image_match.x}, {image_match.y})"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Camera view saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, raster_img


def show_vector_in_pcd(pcd, vector, origin, length, Jupyter=False):
    """
    Visualize a 3D vector as a line from the origin to a point Xm away.

    The function normalizes the input vector, computes the endpoint
    by moving 2 meters along the vector from the origin, creates an
    Open3D LineSet, and calls the show method to render it.

    Args:
        vector (array-like): The 3D direction vector.
        origin (array-like): The starting point of the vector.
        Jupyter (bool): If True, uses Jupyter visualization.
    """
    # Ensure inputs are numpy arrays.
    vec = np.asarray(vector, dtype=float)
    origin = np.asarray(origin, dtype=float)

    # Check for zero-length vector.
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Input vector must be non-zero.")

    # Normalize and compute endpoint Xm away.
    vec = vec / norm
    end_point = origin + length * vec

    # Create a LineSet with two points: origin and end_point.
    line_set = geometry.LineSet()
    points = [origin.tolist(), end_point.tolist()]
    line_set.points = utility.Vector3dVector(points)
    line_set.lines = utility.Vector2iVector([[0, 1]])
    # Color the line (red).
    line_set.colors = utility.Vector3dVector([[1, 0, 0]])

    # Visualize using the provided show method.
    show([pcd.o3d_pcd, line_set])


def draw_image_matches_within_camera(
    image_matches, cam, use_label_column=False, resize_width=None
):
    """
    Open the image, draw an ellipse for each image match, overlay up to the first two masks (if available)
    with smooth contour boundaries (green for the first mask, red for the second mask),
    and add a label with the annotation id to the right of the ellipse.
    Optionally resize the final image to the specified width while preserving the aspect ratio.
    """

    # Open the image and convert to RGB.
    image = Image.open(cam.filepath).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a TrueType font; fall back to the default if it fails.
    # Use a smaller font size for better visibility if the text is not showing up.
    try:
        # Try to load a common cross-platform font; fall back to default if not found
        font = ImageFont.truetype("DejaVuSans.ttf", 100)
    except Exception as e:
        print(f"Error loading DejaVuSans font: {e}")
        font = ImageFont.load_default()

    # Define offsets for the ellipse and text.
    circle_radius = 50
    text_offset_x = circle_radius + 50  # Offset text to the right of the ellipse.
    text_offset_y = -50  # Slightly above the center of the ellipse.

    # If no matches provided, optionally resize and return the base image
    if not image_matches:
        if resize_width is not None:
            orig_width, orig_height = image.size
            new_height = int(orig_height * resize_width / orig_width)
            image = image.resize((resize_width, new_height))
        return image

    # Draw ellipses and labels.
    for match in image_matches:
        x, y = match.x, match.y

        # Draw the ellipse.
        draw.ellipse(
            (
                x - circle_radius,
                y - circle_radius,
                x + circle_radius,
                y + circle_radius,
            ),
            fill=(255, 0, 0),
        )

        # Prepare the label text.
        if use_label_column:
            ann_text = match.annotation.label
        else:
            ann_text = str(match.annotation.id)

        # Draw the label text.
        draw.text(
            (x + text_offset_x, y + text_offset_y),
            ann_text,
            (255, 255, 255),
            font=font,
        )

    # Process masks from match.masks by overlaying smooth contours.
    # Convert the PIL image to a NumPy array.
    np_image = np.array(image)  # (H, W, 3) in RGB

    # Define colors for the first two masks: green and red.
    mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for match in image_matches:
        if match.masks is not None and len(match.masks) > 0:
            for idx in range(min(2, len(match.masks))):
                try:
                    mask_obj = match.masks[idx]
                    # Check format: use .vals if present; otherwise, treat mask_obj as a NumPy array.
                    if hasattr(mask_obj, "vals"):
                        mask_arr = np.array(mask_obj.vals, dtype=np.uint8)
                    else:
                        mask_arr = np.array(mask_obj, dtype=np.uint8)

                    # Ensure the mask is binary.
                    mask_bin = (mask_arr > 0).astype(np.uint8) * 255

                    # Find contours using OpenCV.
                    contours, _ = cv2.findContours(
                        mask_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    # Smooth the contours.
                    smooth_contours = []
                    for cnt in contours:
                        epsilon = 0.01 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        smooth_contours.append(approx)

                    # Draw the smooth contours on the image using the designated color.
                    cv2.drawContours(
                        np_image, smooth_contours, -1, mask_colors[idx], thickness=3
                    )
                except Exception as e:
                    print("Error processing mask for annotation", ann_text, ":", e)

    # Convert the NumPy image (with overlays) back to a PIL image.
    image = Image.fromarray(np_image)

    # Resize the image if a resize_width is provided.
    if resize_width is not None:
        orig_width, orig_height = image.size
        new_height = int(orig_height * resize_width / orig_width)
        image = image.resize((resize_width, new_height))

    return image


def show_image_matches_within_camera(
    image_matches, cam, use_label_column=False, resize_width=None
):
    """Show image matches within the camera image"""
    image = draw_image_matches_within_camera(
        image_matches, cam, use_label_column, resize_width
    )
    plt.imshow(np.array(image))
    plt.show()


def save_image_matches_within_camera(
    image_matches, cam, output_path, use_label_column=False, resize_width=None
):
    """Save image matches within the camera image to a file"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image = draw_image_matches_within_camera(
        image_matches, cam, use_label_column, resize_width
    )
    image.save(os.path.join(output_path, cam.filename))


def create_annotated_video(
    cams,
    annotations,
    output_filename="cams_video.mp4",
    sam_predictor=None,
    pcd=None,
    use_label_column=False,
    resize_width=None,
):
    """
    Create a video from cameras, optionally drawing annotation overlays.

    If no annotations and no overlays are requested, delegates to
    create_video_from_cams for a faster direct ffmpeg path.
    """

    # Create a temporary directory for saving image matches
    temp_image_matches_output = tempfile.mkdtemp(prefix="image_matches_")
    print(f"Temporary image matches output: {temp_image_matches_output}")

    try:
        # Generate image matches for each camera and save them to disk.
        if annotations is not None:
            print(f"Processing {len(cams)} cameras with annotations...")
            for cam in tqdm(
                cams, total=len(cams), desc="Generating annotated frames for video"
            ):
                try:
                    # Check if camera has valid filepath
                    if (
                        not hasattr(cam, "filepath")
                        or not cam.filepath
                        or not os.path.exists(cam.filepath)
                    ):
                        print(
                            f"Warning: Camera {cam.cam_id} has invalid filepath: {getattr(cam, 'filepath', 'None')}"
                        )
                        continue

                    image_matches = cam.get_image_matches(annotations, pcd=pcd)
                    if sam_predictor:
                        for match in image_matches:
                            match.get_sam2_masks(sam_predictor)

                    save_image_matches_within_camera(
                        image_matches,
                        cam,
                        temp_image_matches_output,
                        use_label_column,
                        resize_width,
                    )
                except Exception as e:
                    print(f"Error processing camera {cam.cam_id}: {e}")
                    continue
        else:
            from joblib import Parallel, delayed

            def save_cam_image(cam):
                image_matches = None
                save_image_matches_within_camera(
                    image_matches,
                    cam,
                    temp_image_matches_output,
                    use_label_column,
                    resize_width,
                )

            cams_list = list(cams)
            Parallel(n_jobs=-1)(
                delayed(save_cam_image)(cam)
                for cam in tqdm(
                    cams_list, desc="Generating frames for video (without annotations)"
                )
            )
        # Create a video from the saved images.
        print("Creating video from frames...")

        # Build a concat list to preserve camera order and avoid glob issues
        file_list_path = None
        target_width = resize_width
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as f:
                file_list_path = f.name
                # Determine ordered list of cams used
                ordered_cams = list(cams) if annotations is None else list(cams)
                last_img = None
                frames_found = 0
                for cam in ordered_cams:
                    out_path = os.path.join(temp_image_matches_output, cam.filename)
                    if os.path.isfile(out_path):
                        f.write(f"file '{out_path}'\n")
                        f.write("duration 0.5\n")
                        last_img = out_path
                        frames_found += 1
                    else:
                        print(
                            f"Warning: Frame not found for camera {cam.cam_id}: {out_path}"
                        )

                if frames_found == 0:
                    print("Error: No frames were generated for the video")
                    print(
                        f"Temporary directory contents: {os.listdir(temp_image_matches_output)}"
                    )
                    return

                print(f"Found {frames_found} frames for video creation")
                if last_img is not None:
                    f.write(f"file '{last_img}'\n")

            # Compute safe target width if not provided
            if target_width is None and last_img is not None:
                try:
                    with Image.open(last_img) as im:
                        w, _ = im.size
                        if w > 4096:
                            target_width = 4096
                except Exception:
                    target_width = None

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                file_list_path,
                "-r",
                "2",  # Set output framerate to 2 fps
                "-vsync",
                "cfr",  # Use constant frame rate with -r
            ]
            if target_width is not None:
                cmd += ["-vf", f"scale={int(target_width)}:-2"]
            cmd += [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_filename,
            ]
            subprocess.run(cmd, check=True)
        finally:
            if file_list_path and os.path.exists(file_list_path):
                try:
                    os.remove(file_list_path)
                except Exception:
                    pass
    finally:
        # Clean up the temporary directory after creating the video
        shutil.rmtree(temp_image_matches_output, ignore_errors=True)


def visualize_elevation_angle_legacy(
    pcd,
    plane_coeffs,
    output_filename=None,
    max_output_points=50000,
    width=10,
    height=5,
    point_size=4,
):
    """
    Visualize the fitted plane and the point cloud.

    Args:
        pcd: The point cloud object.
        a, b, c, d: Plane coefficients.
        output_filename: Optional filename to save the visualization.
        max_output_points: Maximum number of points to plot.
        width: Figure width in inches.
        height: Figure height in inches.
        point_size: Scatter marker size for the point cloud.
    """
    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    a, b, c, d = plane_coeffs

    # ------------------------- 3D Visualization --------------------------
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection="3d")

    # 1) Plot the fitted plane FIRST (with a low alpha)
    x_min, x_max = np.min(pcd.points[:, 0]), np.max(pcd.points[:, 0])
    y_min, y_max = np.min(pcd.points[:, 1]), np.max(pcd.points[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = (-a * xx - b * yy - d) / c
    ax.plot_surface(xx, yy, zz, color="red", alpha=0.1)

    # 2) Plot the point cloud next (slightly higher alpha)
    ax.scatter(
        pcd.points[:, 0],
        pcd.points[:, 1],
        pcd.points[:, 2],
        c=pcd.colors,
        s=point_size,
        alpha=0.4,
    )

    # Determine bounding box and origin for arrows
    z_min, z_max = np.min(pcd.points[:, 2]), np.max(pcd.points[:, 2])
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)
    origin = np.array([mid_x, mid_y, mid_z])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    arrow_length = max_range  # Increase to make arrows more visible

    # 3) Plot the vertical arrow (blue)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        0,
        0,
        arrow_length,
        color="blue",
        linewidth=2,
        arrow_length_ratio=0.05,
    )

    # 4) Plot the plane normal arrow (green)
    plane_normal_unit = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        plane_normal_unit[0] * arrow_length,
        plane_normal_unit[1] * arrow_length,
        plane_normal_unit[2] * arrow_length,
        color="green",
        linewidth=2,
        arrow_length_ratio=0.05,
    )

    # 5) Spherical arc between vertical_normal and plane_normal_unit (orange)
    num_arc_points = 40
    arc_points = []
    for i in range(num_arc_points):
        t = i / (num_arc_points - 1)
        direction = geom.slerp(np.array([0, 0, 1]), plane_normal_unit, t)
        arc_points.append(origin + arrow_length * direction)
    arc_points = np.array(arc_points)

    # Plot the arc
    ax.plot3D(
        arc_points[:, 0],
        arc_points[:, 1],
        arc_points[:, 2],
        color="orange",
        linewidth=3,
    )

    # Annotate near the midpoint of the arc
    elevation_angle = np.degrees(
        np.arccos(np.clip(np.dot(plane_normal_unit, [0, 0, 1]), -1.0, 1.0))
    )
    mid_idx = num_arc_points // 2
    ax.text(
        arc_points[mid_idx, 0],
        arc_points[mid_idx, 1],
        arc_points[mid_idx, 2],
        # This calculates the elevation angle between the plane normal (plane_normal_unit)
        # and the vertical direction [0, 0, 1] in degrees, formatted as a string with 1 decimal and a ° symbol.
        f"{elevation_angle:.1f}°",
        color="orange",
        fontsize=10,
    )

    # -------------- Set up the axis labels, title, and limits --------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"Elevation angle: {elevation_angle}°")

    # Make all axes have the same scale
    ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
    ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
    ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)

    if output_filename is not None:
        plt.savefig(output_filename)
    else:
        plt.show()


def visualize_elevation_angle(
    pcd,
    plane_coeffs,
    output_filename=None,
    max_output_points=50000,
    width=600,
    height=400,
    point_size=2,
    interactive=False,
):
    """
    Visualize the fitted plane and the point cloud using plotly for interactive 3D visualization.

    Works in both Jupyter notebooks and VS Code. Can be displayed interactively, saved to file,
    or returned as a static image.

    Args:
        pcd: The point cloud object.
        plane_coeffs: Plane coefficients [a, b, c, d] where ax + by + cz + d = 0.
        output_filename: Optional filename to save the visualization. If provided, saves to file.
            Supports formats: .html, .png, .pdf, .svg, .jpeg. If None and interactive=False,
            returns a static image as bytes.
        max_output_points: Maximum number of points to plot. The point cloud will be decimated
            if it exceeds this limit.
        width: Figure width in pixels (default 800).
        height: Figure height in pixels (default 600).
        point_size: Scatter marker size for the point cloud (default 2).
        interactive: If True and output_filename is None, displays interactively. If False and
            output_filename is None, returns a static image as numpy array (default False).

    Returns:
        plotly.graph_objects.Figure | np.ndarray: The interactive plotly figure if interactive=True
            or output_filename is provided, otherwise returns static image as numpy array
            (H, W, 3), dtype uint8, RGB format (same format as calc_gap_fraction).
    """
    import plotly.graph_objects as go

    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    a, b, c, d = plane_coeffs

    # Extract points and colors
    points = np.asarray(pcd.points)
    colors = getattr(pcd, "colors", None)
    if colors is not None:
        colors = np.asarray(colors)
        # Normalize colors to 0-255 range if needed
        if colors.max() <= 1.0:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            colors_uint8 = colors.astype(np.uint8)
        # Convert to RGB strings for plotly
        color_strings = [
            f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors_uint8
        ]
    else:
        color_strings = "blue"

    # Calculate bounds and origin
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)
    origin = np.array([mid_x, mid_y, mid_z])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    arrow_length = max_range

    # Calculate plane normal and elevation angle
    plane_normal_unit = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    elevation_angle = np.degrees(
        np.arccos(np.clip(np.dot(plane_normal_unit, [0, 0, 1]), -1.0, 1.0))
    )

    # Create figure
    fig = go.Figure()

    # 1) Plot the fitted plane (semi-transparent red surface)
    x_plane = np.linspace(x_min, x_max, 20)
    y_plane = np.linspace(y_min, y_max, 20)
    xx, yy = np.meshgrid(x_plane, y_plane)
    zz = (-a * xx - b * yy - d) / c

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            opacity=0.3,
            name="Plane",
        )
    )

    # 2) Plot the point cloud
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=(
                    color_strings if isinstance(color_strings, list) else color_strings
                ),
                opacity=0.6,
            ),
            showlegend=False,
            name="Point Cloud",
        )
    )

    # 3) Plot the vertical arrow (blue)
    vertical_end = origin + np.array([0, 0, arrow_length])
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], vertical_end[0]],
            y=[origin[1], vertical_end[1]],
            z=[origin[2], vertical_end[2]],
            mode="lines+markers",
            marker=dict(size=8, color="blue"),
            line=dict(color="blue", width=8),
            showlegend=False,
            name="Vertical",
        )
    )
    # Add arrowhead
    fig.add_trace(
        go.Cone(
            x=[vertical_end[0]],
            y=[vertical_end[1]],
            z=[vertical_end[2]],
            u=[0],
            v=[0],
            w=[arrow_length * 0.1],
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False,
            showlegend=False,
        )
    )

    # 4) Plot the plane normal arrow (green)
    normal_end = origin + plane_normal_unit * arrow_length
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], normal_end[0]],
            y=[origin[1], normal_end[1]],
            z=[origin[2], normal_end[2]],
            mode="lines+markers",
            marker=dict(size=8, color="green"),
            line=dict(color="green", width=8),
            showlegend=False,
            name="Plane Normal",
        )
    )
    # Add arrowhead
    fig.add_trace(
        go.Cone(
            x=[normal_end[0]],
            y=[normal_end[1]],
            z=[normal_end[2]],
            u=[plane_normal_unit[0] * arrow_length * 0.1],
            v=[plane_normal_unit[1] * arrow_length * 0.1],
            w=[plane_normal_unit[2] * arrow_length * 0.1],
            colorscale=[[0, "green"], [1, "green"]],
            showscale=False,
            showlegend=False,
        )
    )

    # 5) Plot the spherical arc between vertical and plane normal (orange)
    num_arc_points = 40
    arc_points = []
    for i in range(num_arc_points):
        t = i / (num_arc_points - 1)
        direction = geom.slerp(np.array([0, 0, 1]), plane_normal_unit, t)
        arc_points.append(origin + arrow_length * direction)
    arc_points = np.array(arc_points)

    fig.add_trace(
        go.Scatter3d(
            x=arc_points[:, 0],
            y=arc_points[:, 1],
            z=arc_points[:, 2],
            mode="lines",
            line=dict(color="orange", width=6),
            showlegend=False,
            name="Arc",
        )
    )

    # 6) Add text annotation for elevation angle (using Scatter3d with text mode)
    mid_idx = num_arc_points // 2
    arc_mid = arc_points[mid_idx]

    fig.add_trace(
        go.Scatter3d(
            x=[arc_mid[0]],
            y=[arc_mid[1]],
            z=[arc_mid[2]],
            mode="text",
            text=[f"{elevation_angle:.1f}°"],
            textfont=dict(size=14, color="orange"),
            showlegend=False,
            name="Angle",
        )
    )

    # Update layout with equal aspect ratio (same pixels per meter for all axes)
    # Set camera view similar to matplotlib default (x on right, y going back)
    half_range = max_range / 2.0
    camera_eye = {
        "x": 1.25,
        "y": -1.25,
        "z": 1.25,
    }
    camera_center = {"x": 0, "y": 0, "z": 0}
    camera_up = {"x": 0, "y": 0, "z": 1}

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",  # Ensures equal unit distances across all axes
            xaxis=dict(
                range=[mid_x - half_range, mid_x + half_range],
            ),
            yaxis=dict(
                range=[mid_y - half_range, mid_y + half_range],
            ),
            zaxis=dict(
                range=[mid_z - half_range, mid_z + half_range],
            ),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        title=f"Elevation angle: {elevation_angle:.1f}°",
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Show, save, or return static image
    if output_filename is not None:
        # Determine file format from extension
        ext = os.path.splitext(output_filename)[1].lower()
        if ext == ".html":
            fig.write_html(output_filename)
        elif ext in [".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"]:
            # For static images, use write_image (requires kaleido)
            try:
                fig.write_image(output_filename, width=width, height=height)
            except Exception as e:
                # Fallback to HTML if image export fails
                logger.warning(
                    f"Image export failed ({e}). Saving as HTML instead. "
                    "Install kaleido for image export: pip install kaleido"
                )
                html_filename = os.path.splitext(output_filename)[0] + ".html"
                fig.write_html(html_filename)
        else:
            # Default to HTML
            fig.write_html(output_filename)
        return fig
    elif interactive:
        # Show interactively
        try:
            from IPython import get_ipython

            in_jupyter = get_ipython() is not None
        except ImportError:
            in_jupyter = False

        if in_jupyter:
            try:
                fig.show()
            except (ValueError, ImportError) as e:
                if "nbformat" in str(e):
                    import warnings

                    warnings.warn(
                        "nbformat>=4.2.0 not installed. Using browser renderer. "
                        "Install with: pip install nbformat>=4.2.0 for inline display."
                    )
                    fig.show(renderer="browser")
                else:
                    raise
        else:
            fig.show(renderer="browser")
    else:
        # Return static image as numpy array (same format as calc_gap_fraction)
        try:
            image_bytes = fig.to_image(format="png", width=width, height=height)
            # Convert PNG bytes to numpy array (H, W, 3), dtype uint8, RGB format
            img = Image.open(BytesIO(image_bytes))
            # Convert to RGB if needed (handles RGBA, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_array = np.array(img, dtype=np.uint8)
            return image_array
        except Exception as e:
            logger.warning(
                f"Image export failed ({e}). Returning figure object instead. "
                "Install kaleido for image export: pip install kaleido"
            )
            return fig


def visualize_roughness(
    pcd,
    output_filename=None,
    max_output_points=50000,
    width=600,
    height=400,
    point_size=2,
    interactive=False,
    ra=None,
    rq=None,
):
    """
    Visualize the roughness calculation (Ra and Rq) for a point cloud.

    Shows the point cloud colored by distance from the best-fit plane, with the
    fitted plane surface visible. The Ra and Rq values are displayed in the title.

    Works in both Jupyter notebooks and VS Code. Can be displayed interactively,
    saved to file, or returned as a static image.

    Args:
        pcd: The point cloud object.
        output_filename: Optional filename to save the visualization. If provided,
            saves to file. Supports formats: .html, .png, .pdf, .svg, .jpeg. If None
            and interactive=False, returns a static image as numpy array.
        max_output_points: Maximum number of points to plot. The point cloud will be
            decimated if it exceeds this limit.
        width: Figure width in pixels (default 800).
        height: Figure height in pixels (default 600).
        point_size: Scatter marker size for the point cloud (default 2).
        interactive: If True and output_filename is None, displays interactively.
            If False and output_filename is None, returns a static image as numpy
            array (default False).
        ra: Optional Ra (arithmetical mean roughness) value. If None, will be
            calculated from the point cloud.
        rq: Optional Rq (root mean square roughness) value. If None, will be
            calculated from the point cloud.

    Returns:
        plotly.graph_objects.Figure | np.ndarray: The interactive plotly figure if
            interactive=True or output_filename is provided, otherwise returns static
            image as numpy array (H, W, 3), dtype uint8, RGB format.
    """
    import plotly.graph_objects as go
    from substrata import measurements

    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    # Get plane coefficients (needed for visualization and possibly for ra/rq)
    a, b, c, d = measurements.get_best_fit_plane_PCA(pcd)[:4]

    # Calculate roughness if not provided
    if ra is None or rq is None:
        pts = np.asarray(pcd.points, dtype=float)
        if pts.size == 0:
            raise ValueError("Point cloud has no points")

        normal = np.array([a, b, c], dtype=float)
        denom = np.linalg.norm(normal)
        if denom == 0.0:
            raise ValueError("Best-fit plane has zero-length normal")

        # Perpendicular distances of all points to the plane
        dist = np.abs(pts @ normal + d) / denom

        if ra is None:
            ra = float(dist.mean())
        if rq is None:
            rq = float(np.sqrt((dist**2).mean()))

    # Extract points
    points = np.asarray(pcd.points)

    # Calculate perpendicular distances from points to plane for coloring
    normal = np.array([a, b, c], dtype=float)
    denom = np.linalg.norm(normal)
    if denom == 0.0:
        raise ValueError("Best-fit plane has zero-length normal")

    # Perpendicular distances (signed, for better color mapping)
    dists = (points @ normal + d) / denom

    # Calculate max absolute distance for colorbar scaling
    max_abs_dist = np.max(np.abs(dists)) if len(dists) > 0 else 1.0

    # Calculate bounds and origin
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half_range = max_range / 2.0

    # Create figure
    fig = go.Figure()

    # 1) Plot the fitted plane (semi-transparent gray surface)
    x_plane = np.linspace(x_min, x_max, 20)
    y_plane = np.linspace(y_min, y_max, 20)
    xx, yy = np.meshgrid(x_plane, y_plane)
    zz = (-a * xx - b * yy - d) / c

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale=[[0, "gray"], [1, "gray"]],
            showscale=False,
            opacity=0.3,
            name="Fitted Plane",
        )
    )

    # 2) Plot the point cloud colored by distance from plane
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=dists,  # Use numeric distances for colorbar
                colorscale="RdBu_r",  # Red-Blue reversed colormap
                opacity=0.8,
                colorbar=dict(
                    title="Distance from<br>plane (m)",
                    len=0.5,
                    y=0.5,
                ),
                cmin=-max_abs_dist,
                cmax=max_abs_dist,
            ),
            showlegend=False,
            name="Point Cloud",
        )
    )

    # Update layout with equal aspect ratio
    camera_eye = {
        "x": 1.25,
        "y": -1.25,
        "z": 1.25,
    }
    camera_center = {"x": 0, "y": 0, "z": 0}
    camera_up = {"x": 0, "y": 0, "z": 1}

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(
                range=[mid_x - half_range, mid_x + half_range],
            ),
            yaxis=dict(
                range=[mid_y - half_range, mid_y + half_range],
            ),
            zaxis=dict(
                range=[mid_z - half_range, mid_z + half_range],
            ),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        title=f"Rq (RMS Roughness): {rq:.6f} m, Ra: {ra:.6f} m",
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Show, save, or return static image
    if output_filename is not None:
        # Determine file format from extension
        ext = os.path.splitext(output_filename)[1].lower()
        if ext == ".html":
            fig.write_html(output_filename)
        elif ext in [".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"]:
            # For static images, use write_image (requires kaleido)
            try:
                fig.write_image(output_filename, width=width, height=height)
            except Exception as e:
                # Fallback to HTML if image export fails
                logger.warning(
                    f"Image export failed ({e}). Saving as HTML instead. "
                    "Install kaleido for image export: pip install kaleido"
                )
                html_filename = os.path.splitext(output_filename)[0] + ".html"
                fig.write_html(html_filename)
        else:
            # Default to HTML
            fig.write_html(output_filename)
        return fig
    elif interactive:
        # Show interactively
        try:
            from IPython import get_ipython

            in_jupyter = get_ipython() is not None
        except ImportError:
            in_jupyter = False

        if in_jupyter:
            try:
                fig.show()
            except (ValueError, ImportError) as e:
                if "nbformat" in str(e):
                    import warnings

                    warnings.warn(
                        "nbformat>=4.2.0 not installed. Using browser renderer. "
                        "Install with: pip install nbformat>=4.2.0 for inline display."
                    )
                    fig.show(renderer="browser")
                else:
                    raise
        else:
            fig.show(renderer="browser")
        return fig
    else:
        # Return static image as numpy array (same format as calc_gap_fraction)
        try:
            image_bytes = fig.to_image(format="png", width=width, height=height)
            # Convert PNG bytes to numpy array (H, W, 3), dtype uint8, RGB format
            img = Image.open(BytesIO(image_bytes))
            # Convert to RGB if needed (handles RGBA, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_array = np.array(img, dtype=np.uint8)
            return image_array
        except Exception as e:
            logger.warning(
                f"Image export failed ({e}). Returning figure object instead. "
                "Install kaleido for image export: pip install kaleido"
            )
            return fig


def visualize_tpi(
    pcd,
    tpi_abs,
    tpi_plane,
    output_filename=None,
    max_output_points=50000,
    width=1200,
    height=500,
    point_size=2,
    interactive=False,
    mean_tpi_abs=None,
    mean_tpi_plane=None,
    mean_tri_abs=None,
    mean_tri_plane=None,
    center=None,
    radius_inner=0.0,
    radius_outer=0.0,
    colorscale_max=1.0,
):
    """Visualize TPI (absolute and plane-relative) for a point cloud.

    Renders two side-by-side 2D top-down scatter plots: the left coloured by
    TPI_abs and the right by TPI_plane.  Both use a diverging ``RdBu_r``
    colormap with a fixed symmetric range of ``±colorscale_max`` metres.
    Points with no annulus neighbours (NaN) are shown in light gray.  When
    ``center`` is provided the focal point is marked with a star and the inner
    and outer annulus radii are drawn as circles.

    Args:
        pcd: Point cloud with a ``.points`` attribute.
        tpi_abs: Per-point absolute TPI array (N,), may contain NaN.
        tpi_plane: Per-point plane-relative TPI array (N,), may contain NaN.
        output_filename: Optional path to save the figure (png, jpg, pdf, svg).
        max_output_points: Maximum number of points to render (decimated if
            exceeded).
        width: Figure width in pixels (at 100 dpi).
        height: Figure height in pixels (at 100 dpi).
        point_size: Scatter marker size (matplotlib ``s``).
        interactive: If True and no output file, display interactively.
        mean_tpi_abs: Pre-computed mean TPI_abs for the title (optional).
        mean_tpi_plane: Pre-computed mean TPI_plane for the title (optional).
        mean_tri_abs: Pre-computed TRI_abs for the left-panel title (optional).
        mean_tri_plane: Pre-computed TRI_plane for the right-panel title
            (optional).
        center: (3,) focal point.  If provided, a star marker and radius
            circles are overlaid on each panel.
        radius_inner: Inner annulus radius in metres (drawn as solid circle).
        radius_outer: Outer annulus radius in metres (drawn as dashed circle).
        colorscale_max: Half-range of the fixed symmetric colorscale in metres.

    Returns:
        matplotlib.figure.Figure | np.ndarray: Figure object when
        ``interactive=True`` or ``output_filename`` is set; otherwise an
        (H, W, 3) uint8 RGB array.
    """
    dpi = 100
    pts = np.asarray(pcd.points, dtype=float)
    tpi_abs = np.asarray(tpi_abs, dtype=float)
    tpi_plane = np.asarray(tpi_plane, dtype=float)

    if len(pts) > max_output_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pts), size=max_output_points, replace=False)
        pts = pts[idx]
        tpi_abs = tpi_abs[idx]
        tpi_plane = tpi_plane[idx]

    fig, axes = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi)

    panel_configs = [
        (tpi_abs,   "Z relative to focal point (m)",                mean_tpi_abs,  "TPI_abs",   mean_tri_abs,   "TRI_abs"),
        (tpi_plane, "Z relative to annulus plane at focal point (m)", mean_tpi_plane, "TPI_plane", mean_tri_plane, "TRI_plane"),
    ]

    for ax, (values, label, mean_val, tpi_name, tri_val, tri_name) in zip(axes, panel_configs):
        val_str = f"{mean_val:.4f} m" if mean_val is not None else "N/A"
        tri_str = f"{tri_val:.4f} m" if tri_val is not None else "N/A"
        valid = ~np.isnan(values)

        if np.any(~valid):
            ax.scatter(
                pts[~valid, 0],
                pts[~valid, 1],
                c="lightgray",
                s=point_size,
                rasterized=True,
            )

        if np.any(valid):
            sc = ax.scatter(
                pts[valid, 0],
                pts[valid, 1],
                c=values[valid],
                cmap="RdBu_r",
                vmin=-colorscale_max,
                vmax=colorscale_max,
                s=point_size,
                rasterized=True,
            )
            plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)

        if center is not None:
            cx, cy = float(center[0]), float(center[1])
            # The star marks the focal point and is coloured by the TPI metric
            # (focal vs. neighbourhood), distinct from the per-point colorbar
            # scale of "Z relative to focal point"; label it so its colour is
            # not misread as that scale (on which the focal point would be 0).
            star_label = f"focal point (colour = {tpi_name})"
            if mean_val is not None and np.isfinite(mean_val):
                ax.scatter(
                    cx, cy,
                    c=[mean_val],
                    cmap="RdBu_r",
                    vmin=-colorscale_max,
                    vmax=colorscale_max,
                    marker="*",
                    s=200,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                    label=star_label,
                )
            else:
                ax.scatter(
                    cx, cy, c="black", marker="*", s=200, zorder=5, label=star_label
                )
            if radius_inner > 0:
                ax.add_patch(
                    mpatches.Circle(
                        (cx, cy), radius_inner, fill=False, color="black", lw=1, ls="--"
                    )
                )
            if radius_outer > 0:
                ax.add_patch(
                    mpatches.Circle(
                        (cx, cy), radius_outer, fill=False, color="black", lw=1
                    )
                )
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        radii_str = f"r_inner={radius_inner:.3f} m  r_outer={radius_outer:.3f} m"
        ax.set_title(
            f"{label}\n{tpi_name} (focal point) = {val_str}\n"
            f"{tri_name} (annulus) = {tri_str}\n{radii_str}",
            fontsize=9,
        )

    fig.tight_layout()

    if output_filename is not None:
        ext = os.path.splitext(output_filename)[1].lower()
        if ext == ".html":
            logger.warning(
                "HTML output is not supported for 2D TPI visualization; skipping."
            )
        else:
            fig.savefig(output_filename, dpi=dpi, bbox_inches="tight")
        return fig
    elif interactive:
        plt.show()
        return fig
    else:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)


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
    max_outline_lw=1.8,
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

    n_tgt, n_cls = len(tgt), len(items)
    if weighted:
        frac = (sum(cls_p) / len(cls_p)) if cls_p else float("nan")
        frac_label = f"weighted fraction={frac:.3f}  (mean P({target_class}))"
    else:
        frac = (n_tgt / n_cls) if n_cls else float("nan")
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
    point_size=45,
    bg_point_size=1.5,
    interactive=False,
    sample_weights=None,
    fraction_interacting=None,
    z_colony=None,
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
        intercepts: ``Annotations`` of the sampled ``InterceptAnnotation``s.
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
    point_size=45,
    bg_point_size=1.5,
    width=1500,
    height=750,
    output_filename=None,
    interactive=False,
    sample_weights=None,
    fraction_interacting=None,
    z_colony=None,
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


def visualize_vector_dispersion(
    pcd,
    output_filename=None,
    max_output_points=50000,
    width=600,
    height=400,
    point_size=2,
    interactive=False,
    dispersion=None,
):
    """
    Visualize global vector normal dispersion (Young et al., 2017) on a point cloud.

    Expects a (typically subsampled) point cloud, e.g. points within a radius from
    a given point. Computes the single scalar dispersion via get_vector_dispersion
    and shows each point as a short stick in the direction of its normal, colored
    by deviation from the average (mean) normal: blue = aligned (0°), red = max
    deviation (90°). The heatmap value is the angle in degrees from the mean normal.

    Works in both Jupyter notebooks and VS Code. Can be displayed interactively,
    saved to file, or returned as a static image.

    Args:
        pcd: The point cloud object (must have normals). Typically a subsampled
            cloud within a radius from a point.
        output_filename: Optional filename to save the visualization. If provided,
            saves to file. Supports formats: .html, .png, .pdf, .svg, .jpeg. If None
            and interactive=False, returns a static image as numpy array.
        max_output_points: Maximum number of points to plot. The point cloud will be
            decimated if it exceeds this limit.
        width: Figure width in pixels (default 600).
        height: Figure height in pixels (default 400).
        point_size: Scatter marker size for the point cloud (default 2).
        interactive: If True and output_filename is None, displays interactively.
            If False and output_filename is None, returns a static image as numpy
            array (default False).
        dispersion: Optional precomputed global dispersion scalar. If None, will be
            calculated from the point cloud using get_vector_dispersion.

    Returns:
        plotly.graph_objects.Figure | np.ndarray: The interactive plotly figure if
            interactive=True or output_filename is provided, otherwise returns static
            image as numpy array (H, W, 3), dtype uint8, RGB format.
    """
    import plotly
    import plotly.graph_objects as go
    from substrata import measurements

    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("Point cloud has no points")
    normals = np.asarray(pcd.normals)
    if len(normals) != len(points):
        raise ValueError(
            "Point cloud must have normals for vector dispersion visualization."
        )

    if dispersion is None:
        dispersion, _ = measurements.get_vector_dispersion(pcd)
    dispersion = float(dispersion)

    # Unit normals and resultant (mean) direction (same as in get_vector_dispersion)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_normals = normals / norms
    resultant = np.array(
        [unit_normals[:, 0].sum(), unit_normals[:, 1].sum(), unit_normals[:, 2].sum()]
    )
    r_len = np.linalg.norm(resultant)
    if r_len > 1e-10:
        mean_normal = resultant / r_len
        # Dot product of each normal with mean normal (in [-1, 1])
        alignment = unit_normals @ mean_normal
        # Deviation = angle in degrees from mean normal (0° = aligned, 90° = max)
        deviation_deg = np.degrees(np.arccos(np.clip(np.abs(alignment), 0.0, 1.0)))
    else:
        deviation_deg = np.full(len(points), 90.0)

    # Bounds and layout (same as visualize_roughness)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half_range = max_range / 2.0

    # Stick length as fraction of scene (normals point outward from surface)
    stick_length = max_range * 0.025
    stick_ends = points + stick_length * unit_normals

    # Bin deviation for stick coloring (Plotly line traces need one color per trace)
    n_bins = 10
    bin_edges = np.linspace(0, 90, n_bins + 1)
    fig = go.Figure()

    # Invisible scatter to get a continuous colorbar for deviation (0-90°)
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=0.1,
                color=deviation_deg,
                colorscale="RdBu_r",
                opacity=0,
                cmin=0.0,
                cmax=90.0,
                colorbar=dict(
                    title="Deviation from<br>mean normal (°)",
                    len=0.5,
                    y=0.5,
                ),
            ),
            showlegend=False,
            name="colorbar",
        )
    )

    # One line trace per bin: sticks colored by deviation bin (same RdBu_r scale)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (
            (deviation_deg >= lo) & (deviation_deg <= hi)
            if b == n_bins - 1
            else (deviation_deg >= lo) & (deviation_deg < hi)
        )
        if not np.any(mask):
            continue
        seg_x = []
        seg_y = []
        seg_z = []
        for i in np.where(mask)[0]:
            seg_x.extend([points[i, 0], stick_ends[i, 0], np.nan])
            seg_y.extend([points[i, 1], stick_ends[i, 1], np.nan])
            seg_z.extend([points[i, 2], stick_ends[i, 2], np.nan])
        t = (lo + hi) / 2.0 / 90.0
        bin_color = plotly.colors.sample_colorscale("RdBu_r", [t])[0]
        fig.add_trace(
            go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode="lines",
                line=dict(color=bin_color, width=2),
                showlegend=False,
            )
        )

    # Mean normal arrow (solid green line from centroid)
    origin = np.array([mid_x, mid_y, mid_z])
    arrow_length = max_range * 0.5
    if r_len > 1e-10:
        arrow_end = origin + mean_normal * arrow_length
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], arrow_end[0]],
                y=[origin[1], arrow_end[1]],
                z=[origin[2], arrow_end[2]],
                mode="lines",
                line=dict(color="green", width=6),
                showlegend=False,
                name="Mean Normal",
            )
        )
        fig.add_trace(
            go.Cone(
                x=[arrow_end[0]],
                y=[arrow_end[1]],
                z=[arrow_end[2]],
                u=[mean_normal[0] * arrow_length * 0.1],
                v=[mean_normal[1] * arrow_length * 0.1],
                w=[mean_normal[2] * arrow_length * 0.1],
                colorscale=[[0, "green"], [1, "green"]],
                showscale=False,
                showlegend=False,
            )
        )

    camera_eye = {"x": 1.25, "y": -1.25, "z": 1.25}
    camera_center = {"x": 0, "y": 0, "z": 0}
    camera_up = {"x": 0, "y": 0, "z": 1}
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[mid_x - half_range, mid_x + half_range]),
            yaxis=dict(range=[mid_y - half_range, mid_y + half_range]),
            zaxis=dict(range=[mid_z - half_range, mid_z + half_range]),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        title=f"Vector dispersion: {dispersion:.4f} (blue = aligned, red = max deviation)",
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if output_filename is not None:
        ext = os.path.splitext(output_filename)[1].lower()
        if ext == ".html":
            fig.write_html(output_filename)
        elif ext in [".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"]:
            try:
                fig.write_image(output_filename, width=width, height=height)
            except Exception as e:
                logger.warning(
                    f"Image export failed ({e}). Saving as HTML instead. "
                    "Install kaleido for image export: pip install kaleido"
                )
                html_filename = os.path.splitext(output_filename)[0] + ".html"
                fig.write_html(html_filename)
        else:
            fig.write_html(output_filename)
        return fig
    elif interactive:
        try:
            from IPython import get_ipython

            in_jupyter = get_ipython() is not None
        except ImportError:
            in_jupyter = False

        if in_jupyter:
            try:
                fig.show()
            except (ValueError, ImportError) as e:
                if "nbformat" in str(e):
                    import warnings

                    warnings.warn(
                        "nbformat>=4.2.0 not installed. Using browser renderer. "
                        "Install with: pip install nbformat>=4.2.0 for inline display."
                    )
                    fig.show(renderer="browser")
                else:
                    raise
        else:
            fig.show(renderer="browser")
        return fig
    else:
        try:
            image_bytes = fig.to_image(format="png", width=width, height=height)
            img = Image.open(BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_array = np.array(img, dtype=np.uint8)
            return image_array
        except Exception as e:
            logger.warning(
                f"Image export failed ({e}). Returning figure object instead. "
                "Install kaleido for image export: pip install kaleido"
            )
            return fig


def plot_xy_pca(points, mean, eig_vecs, eig_vals) -> None:
    """Scatter the points and show the first two eigen-vectors."""
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.4)
    plt.plot(mean[0], mean[1], "ro")
    scale = 2.0 * np.sqrt(eig_vals)
    colors = ["r", "g"]
    for i in range(2):
        dx, dy = scale[i] * eig_vecs[:, i]
        plt.arrow(
            mean[0],
            mean[1],
            dx,
            dy,
            width=0.01,
            color=colors[i],
            length_includes_head=True,
        )
    plt.gca().set_aspect("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY PCA")
    plt.show()


def plot_depth_regression(depths, depths_predicted, width=10, height=5, title=None):
    """
    Plot depth regression analysis with actual vs predicted depths and residual analysis.

    Only the arrays of actual depths and predicted depths are required. Residuals and
    evaluation metrics are computed internally.

    Args:
        depths (np.ndarray): Actual depth values.
        depths_predicted (np.ndarray): Predicted depth values from regression.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    depths = np.asarray(depths, dtype=float)
    depths_predicted = np.asarray(depths_predicted, dtype=float)

    # Compute residuals and metrics
    depths_residuals = depths - depths_predicted
    num_matches = int(len(depths))
    mse = float(np.mean((depths - depths_predicted) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(depths - depths_predicted)))
    ss_res = float(np.sum((depths - depths_predicted) ** 2))
    ss_tot = float(np.sum((depths - np.mean(depths)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

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


def plot_views(
    pcd,
    point_size=2,
    width=8,
    height=12,
    max_output_points=50000,
    title=None,
    ortho_resolution=None,
):
    """
    Create a composite figure with:
      - Row 1 (full width): orthoprojection using plot_2d_ortho, titled with pcd.filepath
      - Row 2 (full width): 3D plot using plot(), titled "3D view (N points)"
      - Row 3 (two columns): top-down (XY) and front (XZ) views
      - Row 4 (two columns): side (Y–Z) and side (−Y–Z) views
    """
    # Decimate if required (and ensure PointCloud format)
    ortho_pcd = pcd
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)
    filepath = getattr(pcd, "filepath", None)

    def set_equal_2d(ax, x, y):
        # Keep the same units on X and Y by enforcing equal aspect and
        # using a square view that encloses the data.
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        half = max(xmax - xmin, ymax - ymin) / 2.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal", adjustable="box")

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.2, 0.85, 0.85])

    # Row 1: Orthoprojection (full width)
    ax_ortho = fig.add_subplot(gs[0, :])
    ortho_title = (
        os.path.basename(filepath) if filepath is not None else "Orthoprojection"
    )
    plot_2d_ortho(
        ortho_pcd,
        resolution=ortho_resolution,
        ax=ax_ortho,
        title=ortho_title,
        show=False,
    )

    # Row 2: 3D plot (full width)
    ax_3d = fig.add_subplot(gs[1, :], projection="3d")
    plot_title = f"3D view ({len(pcd.points):,} points)"
    data_mins = pcd.points.min(axis=0)
    data_maxs = pcd.points.max(axis=0)
    data_ranges = np.maximum(data_maxs - data_mins, 1e-9)
    plot(pcd, point_size=point_size, ax=ax_3d, title=plot_title)
    ax_3d.set_box_aspect(tuple(data_ranges))
    try:
        ax_3d.margins(0)
    except Exception:
        pass

    # Row 3: Top-down (XY) and Front (XZ)
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_xy.scatter(
        pcd.points[:, 0],
        pcd.points[:, 1],
        c=pcd.colors,
        s=point_size,
        edgecolor="none",
    )
    ax_xy.set_title("Top-down (X–Y)", pad=6)
    set_equal_2d(ax_xy, pcd.points[:, 0], pcd.points[:, 1])

    ax_xz.scatter(
        pcd.points[:, 0],
        pcd.points[:, 2],
        c=pcd.colors,
        s=point_size,
        edgecolor="none",
    )
    ax_xz.set_title("Front (X–Z)", pad=6)
    set_equal_2d(ax_xz, pcd.points[:, 0], pcd.points[:, 2])

    # Row 4: Side (Y–Z) and Side (−Y–Z)
    ax_yz = fig.add_subplot(gs[3, 0])
    ax_nyz = fig.add_subplot(gs[3, 1])
    ax_yz.scatter(
        pcd.points[:, 1],
        pcd.points[:, 2],
        c=pcd.colors,
        s=point_size,
        edgecolor="none",
    )
    ax_yz.set_title("Side (Y–Z)", pad=6)
    set_equal_2d(ax_yz, pcd.points[:, 1], pcd.points[:, 2])

    ax_nyz.scatter(
        -pcd.points[:, 1],
        pcd.points[:, 2],
        c=pcd.colors,
        s=point_size,
        edgecolor="none",
    )
    ax_nyz.set_title("Side (−Y–Z)", pad=6)
    set_equal_2d(ax_nyz, -pcd.points[:, 1], pcd.points[:, 2])

    if title is not None:
        fig.suptitle(title, y=0.995)

    # Fine-tune layout to minimize whitespace while avoiding overlaps (A4 margins ≈ 0.5 in)
    margin_in = 0.5
    fig.subplots_adjust(
        left=margin_in / width,
        right=1.0 - (margin_in / width),
        top=1.0 - (margin_in / height),
        bottom=margin_in / height,
        wspace=0.10,
        hspace=0.20,
    )

    # Rasterize scatter-heavy 2D axes
    for ax in (ax_xy, ax_xz, ax_yz, ax_nyz):
        ax.set_rasterized(True)
        ax.margins(x=0.04, y=0.04)
        ax.tick_params(pad=2)

    return fig
