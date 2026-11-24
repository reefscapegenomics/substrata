"""Terrain figures of a point-cloud neighbourhood.

- ``visualize_elevation_angle``: the fitted plane and its slope
  (:func:`substrata.measurements.get_elevation_angle`, ``get_plane_angles``).
- ``visualize_roughness``: distances to the best-fit plane, Ra / Rq
  (:func:`substrata.measurements.calc_roughness`, ``distances_to_plane``,
  ``get_best_fit_plane_pca``).
- ``tpi_figure`` / ``visualize_tpi``: topographic position index, absolute and
  plane-relative (:func:`substrata.measurements.calc_tpi_and_tri`,
  ``annulus_plane``, ``tpi_point_values``).
- ``visualize_vector_dispersion``: normal-vector dispersion
  (:func:`substrata.measurements.get_vector_dispersion`,
  ``normal_deviation_deg``).
"""

# Standard Library
import os
from io import BytesIO

# Third-Party Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# Local Modules
from substrata import pointclouds, geom
from substrata.logging import logger
from substrata.visualizations._common import (
    _compact_colorbar, _resolve_scene_axes, _SCENE_RIGHT_PAD_PX, _symmetric_ticks,
)


def visualize_elevation_angle(
    pcd,
    plane_coeffs,
    output_filename=None,
    max_output_points=50000,
    width=600,
    height=400,
    point_size=2,
    interactive=False,
    xlim=None,
    ylim=None,
    zlim=None,
    elev_deg=None,
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
        xlim: Optional (min, max) X axis range in point-cloud units, for fixing the
            axes across figures (e.g. panels of a manuscript). None (default) centres
            the axis on the data. The aspect ratio follows the requested spans, so one
            metre stays the same on-screen length on every axis. The arrows, angle arc
            and fitted plane scale with the requested window.
        ylim: Optional (min, max) Y axis range. See *xlim*.
        zlim: Optional (min, max) Z axis range. See *xlim*.
        elev_deg: The elevation angle to report (degrees), as returned by
            :func:`substrata.measurements.get_plane_angles`. If None,
            :func:`substrata.measurements.get_elevation_angle` of the plane
            normal.

    Returns:
        plotly.graph_objects.Figure | np.ndarray: The interactive plotly figure if interactive=True
            or output_filename is provided, otherwise returns static image as numpy array
            (H, W, 3), dtype uint8, RGB format (same format as calc_gap_fraction).
    """
    import plotly.graph_objects as go
    from substrata import measurements

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

    # Axis windows: the data extent by default, or the caller's fixed ranges.
    axis_ranges, axis_aspect, max_range = _resolve_scene_axes(
        points, xlim, ylim, zlim
    )
    arrow_length = max_range

    # The plane spans the visible window when that was set explicitly,
    # otherwise just the data (preserving the default appearance).
    plane_x = axis_ranges["x"] if xlim is not None else [x_min, x_max]
    plane_y = axis_ranges["y"] if ylim is not None else [y_min, y_max]

    # The plane normal and the reported elevation angle
    plane_normal_unit = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    if elev_deg is None:
        elev_deg = measurements.get_elevation_angle([a, b, c])
    elevation_angle = float(elev_deg)

    # Create figure
    fig = go.Figure()

    # 1) Plot the fitted plane (semi-transparent red surface)
    x_plane = np.linspace(plane_x[0], plane_x[1], 20)
    y_plane = np.linspace(plane_y[0], plane_y[1], 20)
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
            # Ensures equal unit distances across all axes
            aspectmode="manual",
            aspectratio=dict(**axis_aspect),
            xaxis=dict(range=axis_ranges["x"]),
            yaxis=dict(range=axis_ranges["y"]),
            zaxis=dict(range=axis_ranges["z"]),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        title=f"Elevation angle: {elevation_angle:.1f}°",
        width=width,
        height=height,
        showlegend=False,
        # Reserve the space the other two lose to their colour bars, so the
        # 3-D scene is positioned consistently across the three figures.
        margin=dict(l=0, r=_SCENE_RIGHT_PAD_PX, t=40, b=0),
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
    xlim=None,
    ylim=None,
    zlim=None,
    plane=None,
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
        ra: Ra (arithmetical mean roughness) as reported by
            :func:`substrata.measurements.calc_roughness`. If None (with *rq*),
            that function is called on the full cloud.
        rq: Rq (root mean square roughness); see *ra*.
        xlim: Optional (min, max) X axis range in point-cloud units, for fixing the
            axes across figures (e.g. panels of a manuscript). None (default)
            centres the axis on the data. The aspect ratio follows the requested
            spans, so one metre stays the same on-screen length on every axis. The
            fitted plane spans the requested window.
        ylim: Optional (min, max) Y axis range. See *xlim*.
        zlim: Optional (min, max) Z axis range. See *xlim*.
        plane: The fitted plane ``(a, b, c, d)`` of the full cloud (as used by
            :func:`substrata.measurements.calc_roughness`). If None,
            :func:`substrata.measurements.get_best_fit_plane_pca` of the full
            cloud.

    Returns:
        plotly.graph_objects.Figure | np.ndarray: The interactive plotly figure if
            interactive=True or output_filename is provided, otherwise returns static
            image as numpy array (H, W, 3), dtype uint8, RGB format.
    """
    import plotly.graph_objects as go
    from substrata import measurements

    # The reported numbers come from the measurement on the full cloud; the
    # (decimated) display only shows them.
    if ra is None or rq is None:
        ra, rq, _ = measurements.calc_roughness(pcd, generate_image=False)
    if plane is None:
        plane = measurements.get_best_fit_plane_pca(pcd)[:4]
    a, b, c, d = (float(v) for v in plane[:4])

    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    # Extract points
    points = np.asarray(pcd.points)

    # Signed perpendicular distances to the plane, for the colours
    dists = measurements.distances_to_plane(points, (a, b, c, d))

    # Calculate max absolute distance for colorbar scaling
    max_abs_dist = np.max(np.abs(dists)) if len(dists) > 0 else 1.0

    # Data bounds, used below for the fitted-plane extent.
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    # Axis windows: the data extent by default, or the caller's fixed ranges.
    axis_ranges, axis_aspect, _ = _resolve_scene_axes(
        points, xlim, ylim, zlim
    )

    # The plane spans the visible window when that was set explicitly,
    # otherwise just the data (preserving the default appearance).
    plane_x = axis_ranges["x"] if xlim is not None else [x_min, x_max]
    plane_y = axis_ranges["y"] if ylim is not None else [y_min, y_max]

    # Create figure
    fig = go.Figure()

    # 1) Plot the fitted plane (semi-transparent gray surface)
    x_plane = np.linspace(plane_x[0], plane_x[1], 20)
    y_plane = np.linspace(plane_y[0], plane_y[1], 20)
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
                colorbar=_compact_colorbar(
                    " m", tickvals=_symmetric_ticks(max_abs_dist)
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
            aspectmode="manual",
            aspectratio=dict(**axis_aspect),
            xaxis=dict(range=axis_ranges["x"]),
            yaxis=dict(range=axis_ranges["y"]),
            zaxis=dict(range=axis_ranges["z"]),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        title=f"Rq (RMS Roughness): {rq:.6f} m, Ra: {ra:.6f} m",
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=_SCENE_RIGHT_PAD_PX, t=40, b=0),
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


def tpi_figure(
    pcd,
    center,
    radius_inner,
    radius_outer,
    tpi_abs,
    tpi_plane,
    tri_abs,
    tri_plane,
    plane=None,
    raster_cell_size=None,
    annulus_points=None,
    **kwargs,
):
    """The QC image of :func:`substrata.measurements.calc_tpi_and_tri`.

    Colours the cloud by the per-point values of
    :func:`substrata.measurements.tpi_point_values` (Z and annulus-plane
    distance relative to the focal point) and titles the panels with the
    reported metrics (:func:`visualize_tpi`).

    Args:
        pcd: The point cloud the metrics were computed on.
        center: (3,) focal point (with the focal Z used by the metrics).
        radius_inner: Inner annulus radius.
        radius_outer: Outer annulus radius.
        tpi_abs: Reported TPI_abs.
        tpi_plane: Reported TPI_plane.
        tri_abs: Reported TRI_abs.
        tri_plane: Reported TRI_plane.
        plane: The annulus plane ``(normal, d)``
            (:func:`substrata.measurements.annulus_plane`), or None (fewer
            than 3 annulus points: the plane panel shows "no data").
        raster_cell_size: The metric's raster cell size; with it only the
            rasterised ``annulus_points`` are drawn, each dot sized to its cell.
        annulus_points: (M, 3) rasterised annulus points (with
            ``raster_cell_size``).
        **kwargs: Passed to :func:`visualize_tpi` (e.g. ``output_filename``,
            ``interactive``, ``width``, ``height``).

    Returns:
        The return value of :func:`visualize_tpi` (an RGB array by default).
    """
    from substrata import measurements

    if raster_cell_size is not None and annulus_points is not None:
        # Show only the voxelized annulus points, not the dense background:
        # visualize_tpi's random subsampling (max_output_points) would otherwise
        # discard most voxels (~300k points vs ~2k voxels). The radius circles
        # and star marker give the spatial context.
        display_pts = np.asarray(annulus_points, dtype=float)
        display_pcd = pointclouds.SimplePointCloud(display_pts)
        # Size each dot to its cell: visualize_tpi renders height=500 px at 100
        # dpi over ~2 * radius_outer m, ~65 % of it axes area; matplotlib
        # scatter s is in pt² (1 pt = 100/72 px at 100 dpi).
        px_per_m = (500 * 0.65) / (2 * radius_outer)
        pt_diameter = raster_cell_size * px_per_m * (72 / 100)
        kwargs.setdefault("point_size", max(2, int(pt_diameter**2)))
    else:
        display_pcd = pcd
        display_pts = np.asarray(pcd.points, dtype=float)
    z_rel, plane_rel = measurements.tpi_point_values(display_pts, center, plane)
    kwargs.setdefault("interactive", False)
    return visualize_tpi(
        display_pcd, z_rel, plane_rel,
        mean_tpi_abs=tpi_abs, mean_tpi_plane=tpi_plane,
        mean_tri_abs=tri_abs, mean_tri_plane=tri_plane,
        center=np.asarray(center, dtype=float),
        radius_inner=radius_inner, radius_outer=radius_outer,
        **kwargs,
    )


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
    show_title=True,
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
        show_title: Draw the text above each panel (label, TPI, TRI and radii;
            default True). Set False for a bare plot, e.g. to compare
            side by side with other plots.

    Returns:
        matplotlib.figure.Figure | np.ndarray: Figure object when
        ``interactive=True`` or ``output_filename`` is set; otherwise an
        (H, W, 3) uint8 RGB array.

    Raises:
        ValueError: If ``tpi_abs`` or ``tpi_plane`` is not one value per point
            (e.g. the scalar focal-point TPI stored in
            ``Annotation.measurements``).
    """
    dpi = 100
    pts = np.asarray(pcd.points, dtype=float)
    tpi_abs = np.asarray(tpi_abs, dtype=float)
    tpi_plane = np.asarray(tpi_plane, dtype=float)
    for name, arr in (("tpi_abs", tpi_abs), ("tpi_plane", tpi_plane)):
        if arr.shape != (len(pts),):
            raise ValueError(
                f"{name} must be a per-point array of shape ({len(pts)},), got "
                f"shape {arr.shape}. The scalar focal-point values in "
                "Annotation.measurements cannot be plotted per point; use "
                "measurements['tpi_image'] (rendered by calc_tpi_and_tri) instead."
            )

    if len(pts) > max_output_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pts), size=max_output_points, replace=False)
        pts = pts[idx]
        tpi_abs = tpi_abs[idx]
        tpi_plane = tpi_plane[idx]

    fig, axes = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi)

    panel_configs = [
        (tpi_abs,   "Z relative to focal point (m)",                mean_tpi_abs,  "TPI_abs",   mean_tri_abs,   "TRI_abs"),
        (tpi_plane, "Z relative to annulus plane (m)", mean_tpi_plane, "TPI_plane", mean_tri_plane, "TRI_plane"),
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
            # The star marks the focal point, coloured by the TPI metric (focal
            # vs. neighbourhood) rather than by the per-point colorbar scale.
            # The title reports that metric, so no legend entry is needed.
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
                )
            else:
                ax.scatter(cx, cy, c="black", marker="*", s=200, zorder=5)
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

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        radii_str = f"r_inner={radius_inner:.3f} m  r_outer={radius_outer:.3f} m"
        if show_title:
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


def visualize_vector_dispersion(
    pcd,
    output_filename=None,
    max_output_points=50000,
    width=600,
    height=400,
    point_size=2,
    interactive=False,
    dispersion=None,
    normal_length=None,
    normal_width=2,
    xlim=None,
    ylim=None,
    zlim=None,
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
        dispersion: The dispersion to report, as returned by
            :func:`substrata.measurements.get_vector_dispersion`. If None, that
            function is called on the full cloud.
        normal_length: Optional length of the normal lines, as an absolute distance
            in point-cloud units (metres). None (default) uses 2.5% of the scene
            extent, which varies with the cloud; setting it keeps the lines
            identical across figures.
        normal_width: Line thickness of the normal lines in pixels (default 2).
        xlim: Optional (min, max) X axis range in point-cloud units, for fixing the
            axes across figures (e.g. panels of a manuscript). None (default)
            centres the axis on the data. The aspect ratio follows the requested
            spans, so one metre stays the same on-screen length on every axis. The
            mean-normal arrow scales with the requested window; the normal lines do
            too unless *normal_length* is set.
        ylim: Optional (min, max) Y axis range. See *xlim*.
        zlim: Optional (min, max) Z axis range. See *xlim*.

    Returns:
        plotly.graph_objects.Figure | np.ndarray: The interactive plotly figure if
            interactive=True or output_filename is provided, otherwise returns static
            image as numpy array (H, W, 3), dtype uint8, RGB format.
    """
    import plotly
    import plotly.graph_objects as go
    from substrata import measurements

    if normal_length is not None and float(normal_length) <= 0:
        raise ValueError(f"normal_length must be > 0, got {normal_length}")
    if normal_width <= 0:
        raise ValueError(f"normal_width must be > 0, got {normal_width}")

    full_normals = np.asarray(pcd.normals)
    if len(full_normals) != len(np.asarray(pcd.points)):
        raise ValueError(
            "Point cloud must have normals for vector dispersion visualization."
        )
    # The reported number and the mean direction come from the full cloud
    if dispersion is None:
        dispersion, _ = measurements.get_vector_dispersion(pcd, generate_image=False)
    dispersion = float(dispersion)

    # Decimate if required (and ensure PointCloud format)
    pcd = pointclouds.get_decimated_pcd(pcd, max_output_points)

    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("Point cloud has no points")
    normals = np.asarray(pcd.normals)

    # Angle of each shown normal from the mean normal, 0-180° (directional, like
    # the dispersion metric: a flipped normal counts as 180°, not as aligned)
    deviation_deg, mean_normal = measurements.normal_deviation_deg(
        normals, reference_normals=full_normals
    )
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_normals = normals / norms

    # Bounds and layout (same as visualize_roughness)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)

    # Axis windows: the data extent by default, or the caller's fixed ranges.
    axis_ranges, axis_aspect, max_range = _resolve_scene_axes(
        points, xlim, ylim, zlim
    )

    # Stick length: an absolute distance when given, else 2.5% of the scene
    # (normals point outward from the surface).
    stick_length = (
        max_range * 0.025 if normal_length is None else float(normal_length)
    )
    stick_ends = points + stick_length * unit_normals

    # Bin deviation for stick coloring (Plotly line traces need one color per trace)
    n_bins = 12
    bin_edges = np.linspace(0, 180, n_bins + 1)
    fig = go.Figure()

    # Invisible scatter to get a continuous colorbar for deviation (0-180°)
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
                cmax=180.0,
                colorbar=_compact_colorbar("°", tickvals=(0, 90, 180)),
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
        t = (lo + hi) / 2.0 / 180.0
        bin_color = plotly.colors.sample_colorscale("RdBu_r", [t])[0]
        fig.add_trace(
            go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode="lines",
                line=dict(color=bin_color, width=normal_width),
                showlegend=False,
            )
        )

    # Mean normal arrow (solid green line from centroid)
    origin = np.array([mid_x, mid_y, mid_z])
    arrow_length = max_range * 0.5
    if mean_normal is not None:
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
            aspectmode="manual",
            aspectratio=dict(**axis_aspect),
            xaxis=dict(range=axis_ranges["x"]),
            yaxis=dict(range=axis_ranges["y"]),
            zaxis=dict(range=axis_ranges["z"]),
            camera=dict(eye=camera_eye, center=camera_center, up=camera_up),
        ),
        title=f"Vector dispersion: {dispersion:.4f} (blue = aligned, red = max deviation)",
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=_SCENE_RIGHT_PAD_PX, t=40, b=0),
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
