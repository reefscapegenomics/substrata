"""Gap-fraction figures.

The hemispherical projection of a point's surroundings with its zenith-angle
axes (``render_gap_fraction``, ``render_gap_fraction_axes``). The fractions come
from :func:`substrata.measurements.calc_gap_fraction` (``hemispherical_pixels``,
``hemispherical_disk``, ``gap_fractions``).
"""

# Standard Library
from io import BytesIO

# Third-Party Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from PIL import Image

# Local Modules
from substrata import settings


def render_gap_fraction(
    pcd,
    center,
    resolution,
    connected,
    max_radius=None,
    color_output=True,
    seed_points_px=None,
    fill_color=settings.GAP_FRACTION_FILL_COLOR,
    show_rings=False,
    ring_color=None,
    show_axes=False,
    width=500,
    height=500,
    output_filename=None,
):
    """The image of :func:`substrata.measurements.calc_gap_fraction`.

    The hemispherical projection (:func:`substrata.measurements.hemispherical_pixels`)
    of the points above ``center``: the imaging disk in dark grey, the cover in
    the point colours (nearest point per pixel) or white, and the connected sky
    that ``gapF_fill`` counts (``connected``, from
    :func:`substrata.measurements.gap_fractions`) in ``fill_color``. Rings and
    seed markers are drawn on top. Display only: the gap fractions are computed
    from the binary cover raster before this is drawn.

    Args:
        pcd: Point cloud (``points``, and ``colors`` with ``color_output``).
        center: (3,) viewpoint.
        resolution: Image size in pixels.
        connected: (resolution, resolution) bool mask of the connected sky.
        max_radius: Horizontal radius of the points used (as in the metric).
        color_output: Paint the cover in the point colours (else white).
        seed_points_px: ``(x, y)`` pixel seeds to mark (None: no markers).
        fill_color: RGB 0-255 of the connected sky.
        show_rings: Draw zenith-angle rings at 22.5, 45, 67.5 and 90 degrees.
        ring_color: RGB 0-255 of the rings (default white, black with axes).
        show_axes: Return a figure with zenith-angle axes
            (:func:`render_gap_fraction_axes`).
        width: Figure width in pixels (with ``show_axes``).
        height: Figure height in pixels (with ``show_axes``).
        output_filename: Also write the image to this file.

    Returns:
        (H, W, 3) uint8 RGB array.

    Raises:
        ValueError: For a ``fill_color`` or ``ring_color`` that is not an RGB
            3-tuple in 0-255.
    """
    from substrata import measurements

    fill_color = tuple(int(c) for c in fill_color)
    if len(fill_color) != 3 or not all(0 <= c <= 255 for c in fill_color):
        raise ValueError(
            f"fill_color must be an RGB 3-tuple in 0-255, got {fill_color!r}"
        )
    if ring_color is None:
        ring_color = (0, 0, 0) if show_axes else (255, 255, 255)
    ring_color = tuple(int(c) for c in ring_color)
    if len(ring_color) != 3 or not all(0 <= c <= 255 for c in ring_color):
        raise ValueError(
            f"ring_color must be an RGB 3-tuple in 0-255, got {ring_color!r}"
        )

    radius = resolution // 2
    rows, cols, points_to_keep, norms = measurements.hemispherical_pixels(
        np.asarray(pcd.points), center, resolution, max_radius
    )
    img_mask = measurements.hemispherical_disk(resolution)

    # Imaging disk in dark grey, cover in the point colours (or white), the
    # connected sky in fill_color.
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    image[img_mask] = [80, 80, 80]
    if color_output:
        rgb_colors = (np.asarray(pcd.colors)[points_to_keep] * 255).astype(np.uint8)
        # For each pixel keep the colour of the point closest to the centre. We
        # scatter-assign in order of decreasing distance so the nearest point
        # (smallest norm) is written last and therefore wins per pixel.
        order = np.argsort(-norms, kind="stable")
        image_flat = image.reshape(-1, 3)
        flat_idx = rows * resolution + cols
        image_flat[flat_idx[order]] = rgb_colors[order]
    else:
        image[rows, cols] = [255, 255, 255]
    image[np.asarray(connected, dtype=bool)] = fill_color

    # Zenith-angle rings: the projected radius is (phi / pi) * resolution, so a
    # zenith angle of `deg` sits at (deg / 180) * resolution pixels from the
    # centre - evenly spaced quarters of the imaging area, with 90 deg on its
    # edge. With show_axes they are drawn as dashed vector circles instead.
    if show_rings and not show_axes:
        ring_thickness = max(1, resolution // 400)
        for deg in (22.5, 45.0, 67.5, 90.0):
            ring_px = int(round((deg / 180.0) * resolution))
            cv2.circle(
                image,
                (radius, radius),
                ring_px,
                ring_color,
                thickness=ring_thickness,
                lineType=cv2.LINE_AA,
            )

    # Seed markers: a green cross with a thin black outline stays visible
    # against the dark grey sky, the fill and varied benthic colours.
    if seed_points_px is not None:
        marker_size = max(6, resolution // 25)
        for seed_point in seed_points_px:
            for color, thickness in (((0, 0, 0), 3), ((0, 255, 0), 1)):
                cv2.drawMarker(
                    image,
                    tuple(int(v) for v in seed_point),
                    color=color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                    line_type=cv2.LINE_AA,
                )

    if show_axes:
        image = render_gap_fraction_axes(
            image,
            img_mask,
            width=width,
            height=height,
            rings=(22.5, 45.0, 67.5, 90.0) if show_rings else (),
            ring_color=ring_color,
        )

    if output_filename is not None:
        cv2.imwrite(output_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return image


def render_gap_fraction_axes(
    image,
    img_mask,
    width=500,
    height=500,
    dpi=100,
    rings=(),
    ring_color=(0, 0, 0),
):
    """Render a gap-fraction projection with zenith-angle axes.

    Used by :func:`substrata.measurements.calc_gap_fraction` with
    ``show_axes=True``. Styled like :func:`visualize_tpi` (same dpi, default
    fonts, ``tight_layout`` and tight-bbox export), so axes and font sizes are
    identical when both are plotted at the same size.

    The projection stores the X direction along image rows and Y along
    columns; it is transposed so X runs to the right and Y up, as in the
    top-down TPI plot. Pixels outside the circular imaging area are white.

    Args:
        image: ``(R, R, 3)`` uint8 RGB projection from ``calc_gap_fraction``.
        img_mask: ``(R, R)`` boolean mask of the circular imaging area.
        width: Figure width in pixels.
        height: Figure height in pixels.
        dpi: Dots per inch (100, as in :func:`visualize_tpi`).
        rings: Zenith angles (degrees) at which to draw dashed circles, in the
            style of the inner annulus circle of :func:`visualize_tpi`.
        ring_color: RGB colour (0-255) of those circles.

    Returns:
        numpy.ndarray: ``(H, W, 3)`` uint8 RGB image of the figure.
    """
    shown = np.array(image, dtype=np.uint8, copy=True)
    shown[~np.asarray(img_mask, dtype=bool)] = 255
    shown = shown.transpose(1, 0, 2)  # rows = X -> columns = X (to the right)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(
        shown,
        origin="lower",
        extent=(-90, 90, -90, 90),
        interpolation="nearest",
    )
    for deg in rings:
        ax.add_patch(
            mpatches.Circle(
                (0, 0),
                deg,
                fill=False,
                color=tuple(c / 255.0 for c in ring_color),
                lw=1,
                ls="--",
            )
        )
    ax.set_aspect("equal")
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("X zenith angle (°)")
    ax.set_ylabel("Y zenith angle (°)")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)
