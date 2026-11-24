"""Helpers shared by the figure modules of :mod:`substrata.visualizations`.

Plotly 3-D scene layout (axis ranges, aspect ratio, a compact colour bar with
symmetric ticks, the right-hand padding that keeps scenes in the same place)
used by the terrain figures, and the grey placeholder cell of the PDF reports.
No measurements are computed here.
"""

# Third-Party Libraries
import numpy as np


def _draw_placeholder(pdf, x, y, w, h, text=""):
    """Draw a light-grey filled rectangle with an optional centred label."""
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(x, y, w, h, style="F")
    pdf.set_fill_color(0, 0, 0)
    if text:
        pdf.set_xy(x, y + h / 2 - 2)
        pdf.cell(w, 4, text, align="C")


# Right-hand padding (px) reserved in visualize_elevation_angle,
# visualize_roughness and visualize_vector_dispersion so their 3-D scenes land
# in the same place. Plotly centres a 3-D scene in the space left by the
# margins; this keeps the cubes where they were when the latter two had their
# colour bars (with titles) on the right, which shifted them left by 59 and
# 66 px. Their colour bars now sit in the bottom-left corner instead
# (_compact_colorbar), inside the plot area, so they no longer push the scene.
_SCENE_RIGHT_PAD_PX = 124


def _symmetric_ticks(max_abs):
    """Ticks ``(-v, 0, v)`` with *v* the largest 1-significant-digit value <= *max_abs*.

    E.g. 0.14 -> (-0.1, 0, 0.1) and 0.036 -> (-0.03, 0, 0.03), so a colour bar
    spanning ``[-max_abs, max_abs]`` always labels both ends and zero.
    """
    if not max_abs or not np.isfinite(max_abs) or max_abs <= 0:
        return (0,)
    exponent = np.floor(np.log10(max_abs))
    step = 10.0**exponent
    v = float(np.floor(max_abs / step + 1e-9) * step)  # 0.3 / 0.1 = 2.999…
    v = float(f"{v:.1g}")  # drop floating-point noise (0.30000000000000004)
    return (-v, 0, v)


def _compact_colorbar(tick_suffix, tickvals=None, n_ticks=3):
    """Small, untitled plotly colour bar for the bottom-left corner of 3-D plots.

    Sits under the Z-axis label and left of the X-axis label of the default
    camera view; the unit is appended to the tick labels instead of a title.

    Args:
        tick_suffix: Unit appended to every tick label (e.g. ``" m"``, ``"°"``).
        tickvals: Optional explicit tick values.
        n_ticks: Approximate number of tick labels when *tickvals* is None.

    Returns:
        dict: plotly ``marker.colorbar`` settings.
    """
    if tickvals is not None:  # ticksuffix is not applied to array ticks
        ticks = dict(
            tickmode="array",
            tickvals=list(tickvals),
            ticktext=[f"{v:g}{tick_suffix}" for v in tickvals],
        )
    else:
        ticks = dict(nticks=n_ticks)
    return dict(
        title=dict(text=""),
        ticksuffix=tick_suffix,
        **ticks,
        tickfont=dict(size=11),
        len=0.3,
        lenmode="fraction",
        thickness=10,
        outlinewidth=0,
        x=0.01,
        xanchor="left",
        y=0.02,
        yanchor="bottom",
    )


def _resolve_scene_axes(points, xlim=None, ylim=None, zlim=None):
    """Resolve plotly 3-D scene axis ranges, aspect ratio and scene scale.

    By default each axis is centred on the data and given the same span (the
    largest of the three data extents), which renders as a cube with equal
    units on every axis. Any of *xlim*/*ylim*/*zlim* replaces that axis'
    window with an explicit ``(min, max)``; axes left as None keep the default
    span. The aspect ratio is derived from the resulting spans, so one metre
    stays the same on-screen length on every axis even when the spans differ
    (a 10x10x2 m window renders as a flat slab, not a stretched cube).

    Args:
        points: (N, 3) array of the points being plotted.
        xlim: Optional (min, max) window for the X axis, in point-cloud units.
        ylim: Optional (min, max) window for the Y axis.
        zlim: Optional (min, max) window for the Z axis.

    Returns:
        tuple: ``(ranges, aspect, max_range)`` where *ranges* maps ``"x"``/
            ``"y"``/``"z"`` to a ``[lo, hi]`` list, *aspect* maps the same keys
            to an aspect ratio normalised so the longest axis is ``1.0``, and
            *max_range* is the longest span (used to size arrows and sticks so
            they stay proportional to the requested window).

    Raises:
        ValueError: If a limit is not a two-element ``(min, max)`` with
            ``max > min``.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        mids = np.zeros(3)
        default_half = 0.5
    else:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        mids = 0.5 * (mins + maxs)
        default_half = 0.5 * max(float(np.max(maxs - mins)), 1e-9)

    ranges = {}
    for axis, lim, mid in zip("xyz", (xlim, ylim, zlim), mids):
        if lim is None:
            ranges[axis] = [mid - default_half, mid + default_half]
            continue
        if len(lim) != 2:
            raise ValueError(
                f"{axis}lim must be a (min, max) pair, got {lim!r}"
            )
        lo, hi = float(lim[0]), float(lim[1])
        if not hi > lo:
            raise ValueError(
                f"{axis}lim must have max > min, got {lim!r}"
            )
        ranges[axis] = [lo, hi]

    spans = {axis: ranges[axis][1] - ranges[axis][0] for axis in "xyz"}
    max_range = max(spans.values())
    aspect = {axis: spans[axis] / max_range for axis in "xyz"}
    return ranges, aspect, max_range
