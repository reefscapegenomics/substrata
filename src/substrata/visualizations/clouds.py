"""Point-cloud figures: interactive plotly and static matplotlib views.

``show`` (plotly 3-D), ``plot`` (matplotlib 3-D), ``plot_2d`` / ``plot_2d_ortho``
(top-down orthographic images via :class:`substrata.ortho.OrthoMap`),
``plot_compare`` (two clouds overlaid), ``plot_views`` and ``save_views_pdf``
(the composite views behind ``PointCloud.save_pdf``) and ``plot_xy_pca`` (the
eigen-vectors of :func:`substrata.measurements.conduct_xy_pca`). The clouds are
decimated for display with :func:`substrata.pointclouds.get_decimated_pcd`;
nothing is measured here.
"""

# Standard Library
import os

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection="3d")

# Local Modules
from substrata import pointclouds


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
    """Render a top-down ortho image of a point cloud as a matplotlib Figure.

    Rendering goes through :class:`substrata.ortho.OrthoMap`, so the whole
    cloud is rasterized once (colours averaged per pixel) instead of
    scatter-plotting every point. Any ``highlight_coords`` are drawn as red
    markers baked into the image. A Figure is returned so callers that embed
    the result in a matplotlib ``PdfPages`` report (e.g. ``Scalebars``) keep
    working unchanged.

    Args:
        pcd: Point cloud object (substrata or Open3D).
        point_size (int): Retained for backward compatibility; highlight
            markers use a fixed pixel radius.
        width (int): Figure width in inches.
        height (int): Figure height in inches.
        highlight_coords (array-like | None): Coordinate(s) to mark in red.
            A single ``[x, y, z]`` or an ``(N, 3)`` array.
        title (str | None): Optional axes title.
        max_output_points (int): Accepted for backward compatibility; ignored
            (OrthoMap rasterizes memory-safely regardless of point count).

    Returns:
        matplotlib.figure.Figure: Figure showing the ortho image.
    """
    from substrata.ortho import OrthoMap

    ortho_map = OrthoMap(pcd)
    img = ortho_map.show(highlights=highlight_coords, point_size=8)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.imshow(np.asarray(img))
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    ax.set_rasterized(True)
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


def save_views_pdf(
    pcd,
    filepath=None,
    point_size=2,
    width=8,
    height=12,
    max_output_points=50000,
    title=None,
    ortho_resolution=None,
):
    """Save the composite views of a point cloud (:func:`plot_views`) to a PDF.

    Uses a non-interactive backend and closes the figure after saving.
    Default path: ``<cloud base>_views.pdf``.

    Returns:
        str: The filepath written.
    """
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages

    backend_original = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        if filepath is None:
            base, _ = os.path.splitext(getattr(pcd, "filepath", None) or "pointcloud")
            filepath = f"{base}_views.pdf"
        fig = plot_views(
            pcd, point_size=point_size, width=width, height=height,
            max_output_points=max_output_points, title=title,
            ortho_resolution=ortho_resolution,
        )
        with PdfPages(filepath) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        return filepath
    finally:
        matplotlib.use(backend_original, force=True)
