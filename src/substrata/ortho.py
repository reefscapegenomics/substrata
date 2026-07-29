"""Orthographic map module for fast 2D representations of point clouds."""

from __future__ import annotations

# Standard Library
import logging
import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

if TYPE_CHECKING:
    from substrata.annotations import Annotation, Annotations
    from substrata.pointclouds import PointCloud, SimplePointCloud

logger = logging.getLogger(__name__)


def _get_cmap(name: str, n: Optional[int] = None):
    """Return a matplotlib colormap, resampled to *n* colours if given.

    Imported lazily so ``ortho`` has no import-time matplotlib dependency.
    """
    import matplotlib

    try:
        cmap = matplotlib.colormaps[name]
    except (AttributeError, KeyError):  # pragma: no cover - old matplotlib
        import matplotlib.cm as mcm

        cmap = mcm.get_cmap(name)
    if n is not None and hasattr(cmap, "resampled"):
        cmap = cmap.resampled(n)
    return cmap


def _rgb255(rgba) -> Tuple[int, int, int]:
    """Convert a 0-1 RGBA tuple to a 0-255 RGB tuple."""
    return tuple(int(round(float(c) * 255)) for c in rgba[:3])


def _label_color_lut(labels, label_colors=None) -> dict:
    """Map each distinct label to a stable RGB colour.

    Labels are sorted and assigned successive ``tab20`` colours; an explicit
    ``label_colors`` mapping overrides individual entries.  Shared by the
    marker styling and the legend so both agree.
    """
    uniq = sorted({lbl for lbl in labels if lbl is not None})
    cmap = _get_cmap("tab20", max(1, len(uniq)))
    lut = dict(label_colors) if label_colors else {}
    for i, lbl in enumerate(uniq):
        lut.setdefault(lbl, _rgb255(cmap(i)))
    return lut


def _as_per_point(value, n: int) -> list:
    """Broadcast *value* to a length-*n* list of per-point values.

    A single RGB tuple or ``None`` is repeated; a sequence already of
    length *n* is passed through unchanged.
    """
    if value is None:
        return [None] * n
    single_rgb = isinstance(value, tuple) and len(value) == 3
    if single_rgb and all(isinstance(c, (int, float)) for c in value):
        return [value] * n
    seq = list(value)
    return seq if len(seq) == n else [value] * n


class OrthoMap:
    """Pre-rendered 2D orthographic projection of a point cloud.

    The point cloud is rasterized once during initialization and the
    resulting image is kept in memory.  Subsequent calls to ``show``
    overlay dynamic highlights (annotations, coordinate lists, etc.)
    without re-rasterizing.

    Attributes:
        image: Pre-rendered ortho image as a numpy array (H, W, 3, uint8).
        rotation: 3x3 rotation matrix from world to view coordinates.
        origin: 2D offset ``(min_x, min_y)`` in rotated space.
        resolution: Metres per pixel.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    def __init__(
        self,
        pcd: Union[PointCloud, SimplePointCloud],
        up_vector: Optional[Union[List[float], np.ndarray]] = None,
        pixel_width: Optional[int] = None,
        pixel_height: Optional[int] = None,
        rotation: int = 0,
    ) -> None:
        """Initialize an OrthoMap from a point cloud.

        The point cloud is projected orthographically along the given
        *up_vector* direction and rasterized into an in-memory image.

        Args:
            pcd: Point cloud to rasterize.
            up_vector: Direction treated as "up" (camera looks along
                the negative of this vector).  Defaults to
                ``[0, 0, 1]`` for a top-down view along the z-axis.
            pixel_width: Desired image width in pixels.  When only one
                dimension is given the other is computed to preserve
                the aspect ratio.  When both are given they are treated
                as maximum values and the aspect ratio is preserved.
            pixel_height: Desired image height in pixels.  See
                *pixel_width* for interaction rules.
        """
        if up_vector is None:
            up_vector = np.array([0.0, 0.0, 1.0])
        up = np.asarray(up_vector, dtype=np.float64)
        up = up / np.linalg.norm(up)
        self._up_vector = up

        self._rotation = rotation
        self.rotation: np.ndarray = self._rotation_to_z(up)
        points = np.asarray(pcd.points)
        rotated = (self.rotation @ points.T).T
        xs, ys = rotated[:, 0], rotated[:, 1]

        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        extent_x = max(max_x - min_x, 1e-9)
        extent_y = max(max_y - min_y, 1e-9)
        self.origin = np.array([min_x, min_y])
        self.width, self.height, self.resolution = (
            self._compute_dimensions(
                extent_x, extent_y, len(points),
                pixel_width, pixel_height,
            )
        )

        colors = self._get_colors(pcd, len(points))
        self.image: np.ndarray = self._rasterize(
            xs, ys, colors,
            min_x, min_y,
            self.resolution,
            self.width, self.height,
        )

        logger.info(
            "OrthoMap created: %d x %d px  (%.6f m/px, %d points)",
            self.width, self.height, self.resolution, len(points),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise summary of the ortho map."""
        return (
            f"OrthoMap({self.width}x{self.height} px, "
            f"res={self.resolution:.6f} m/px, "
            f"up={self._up_vector.tolist()})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        coords: Union[np.ndarray, List[float]],
    ) -> np.ndarray:
        """Map 3D world coordinate(s) to 2D pixel positions.

        Pixel coordinates follow standard image convention: *x* increases
        to the right, *y* increases downward.

        Args:
            coords: A single 3D coordinate ``(3,)`` or an array of
                coordinates ``(N, 3)``.

        Returns:
            Pixel positions as ``(x, y)`` for a single point or
            ``(N, 2)`` for multiple points.  Values are floats and may
            lie outside the image bounds.
        """
        coords = np.asarray(coords, dtype=np.float64)
        single = coords.ndim == 1
        if single:
            coords = coords[np.newaxis, :]
        rotated = (self.rotation @ coords.T).T
        px = (rotated[:, 0] - self.origin[0]) / self.resolution
        py = (self.height - 1
              - (rotated[:, 1] - self.origin[1]) / self.resolution)
        result = np.column_stack([px, py])
        return result[0] if single else result

    def show(
        self,
        highlights: Optional[
            Union[
                Annotation,
                Annotations,
                List[np.ndarray],
                np.ndarray,
            ]
        ] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        point_size: int = 5,
        point_size_metres: Optional[float] = None,
        point_color: Optional[Tuple[int, int, int]] = (255, 0, 0),
        point_outline: Optional[Tuple[int, int, int]] = (0, 0, 0),
        point_shape: str = "circle",
        background_color: Optional[Tuple[int, int, int]] = None,
        color_by: Optional[str] = None,
        fill_by_group: bool = False,
        label_colors: Optional[dict] = None,
        grayscale: bool = False,
        image_opacity: float = 1.0,
        crop: Optional[Tuple] = None,
    ) -> Image.Image:
        """Return the ortho map as a PIL Image with optional highlights.

        The base image is generated from the pre-rendered raster.  If
        *width* and/or *height* are given the image is resized (aspect
        ratio preserved) before highlights are drawn.

        Args:
            highlights: Locations to highlight.  Accepts an
                ``Annotation``, an ``Annotations``/``Cameras`` container
                (any object with a ``.data`` mapping of items that have
                ``.coords`` and optionally ``.label``/``.group``), a
                numpy array of shape ``(N, 3)``, or a list of 3-element
                coordinate arrays.
            width: Display width in pixels.
            height: Display height in pixels.
            point_size: Radius of highlight markers in display pixels.
                Ignored when *point_size_metres* is set.
            point_size_metres: Diameter of highlight markers in metres.
                When provided, the marker is scaled so that it spans
                exactly this distance in the orthographic projection,
                regardless of display resolution.  Takes precedence over
                *point_size*.
            point_color: RGB fill colour for highlight markers.
                Pass ``None`` for a transparent (hollow) fill.  Used as
                the marker colour unless *color_by* assigns per-point
                colours.
            point_outline: RGB outline colour for filled markers
                (``None`` to disable the outline).
            point_shape: Marker shape — ``"circle"`` or ``"square"``.
            background_color: RGB colour for empty (no-data) pixels.
                Defaults to ``None`` which keeps the original white
                background.
            color_by: Per-point marker colouring.  ``"label"`` assigns a
                distinct ``tab20`` colour per highlight label; ``"z"``
                maps each highlight's Z coordinate through a ``bwr``
                colormap.  ``None`` (default) uses *point_color* for all.
            fill_by_group: When True, highlights are drawn filled or
                hollow by the even/odd index of their ``group`` value.
            label_colors: Optional explicit ``{label: (r, g, b)}`` map,
                overriding the automatic ``tab20`` assignment.
            grayscale: Desaturate the background raster (highlights stay
                coloured).
            image_opacity: Blend the background raster toward the white
                background — ``1.0`` (default) leaves it unchanged, ``0.0``
                fades it out entirely (leaving only highlights). Highlights are
                drawn afterward and are unaffected.
            crop: ``(center, size_metres)`` window to crop to, where
                *center* is a coordinate (used for zoom-to-annotation).

        Returns:
            PIL ``Image`` with the orthographic map and any requested
            highlights rendered.
        """
        if background_color is not None:
            arr = self.image.copy()
            bg_mask = np.all(arr == 255, axis=-1)
            arr[bg_mask] = background_color
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(self.image)

        if grayscale:
            img = img.convert("L").convert("RGB")

        if not 0.0 <= image_opacity <= 1.0:
            raise ValueError(
                f"image_opacity must be in [0, 1], got {image_opacity}"
            )
        if image_opacity < 1.0:
            arr = np.asarray(img).astype(np.float32)
            arr = arr * image_opacity + 255.0 * (1.0 - image_opacity)
            img = Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))

        if width is not None or height is not None:
            img = self._resize(img, width, height)

        if highlights is not None:
            coords_3d, labels, groups = self._extract_highlights(highlights)
            if len(coords_3d) > 0:
                pixels = self.project(coords_3d)
                if pixels.ndim == 1:
                    pixels = pixels[np.newaxis, :]
                out_mask = (
                    (pixels[:, 0] < 0)
                    | (pixels[:, 0] >= self.width)
                    | (pixels[:, 1] < 0)
                    | (pixels[:, 1] >= self.height)
                )
                if out_mask.any():
                    out_coords = (
                        coords_3d[np.newaxis, :]
                        if coords_3d.ndim == 1
                        else coords_3d
                    )
                    logger.warning(
                        "%d highlight(s) outside map bounds:\n%s",
                        int(out_mask.sum()),
                        "\n".join(
                            f"  [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]"
                            for c in out_coords[out_mask]
                        ),
                    )
                fill_colors, outline_colors = self._resolve_marker_style(
                    coords_3d, labels, groups,
                    color_by, label_colors, fill_by_group,
                    point_color, point_outline,
                )
                self._draw_highlights(
                    img, coords_3d,
                    point_size, fill_colors, outline_colors,
                    point_shape, point_size_metres,
                )

        if crop is not None:
            img = self._crop_to(img, crop)

        if self._rotation:
            img = img.rotate(-self._rotation, expand=True)

        return img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_to_z(up: np.ndarray) -> np.ndarray:
        """Build a 3x3 rotation that maps *up* onto the z-axis.

        Uses Rodrigues' rotation formula.

        Args:
            up: Unit direction vector.

        Returns:
            3x3 rotation matrix.
        """
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(up, z_axis):
            return np.eye(3)
        if np.allclose(up, -z_axis):
            return np.diag([1.0, -1.0, -1.0])

        v = np.cross(up, z_axis)
        s = np.linalg.norm(v)
        c = float(np.dot(up, z_axis))
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    @staticmethod
    def _compute_dimensions(
        extent_x: float,
        extent_y: float,
        n_points: int,
        pixel_width: Optional[int],
        pixel_height: Optional[int],
    ) -> Tuple[int, int, float]:
        """Determine image size and resolution.

        Args:
            extent_x: Spatial extent along the horizontal view axis.
            extent_y: Spatial extent along the vertical view axis.
            n_points: Number of points (used for heuristic sizing).
            pixel_width: Requested width or ``None``.
            pixel_height: Requested height or ``None``.

        Returns:
            ``(width, height, resolution)`` tuple.
        """
        aspect = extent_x / extent_y

        if pixel_width is not None and pixel_height is not None:
            if aspect >= pixel_width / pixel_height:
                w = pixel_width
                h = max(1, int(round(pixel_width / aspect)))
            else:
                h = pixel_height
                w = max(1, int(round(pixel_height * aspect)))
        elif pixel_width is not None:
            w = pixel_width
            h = max(1, int(round(pixel_width / aspect)))
        elif pixel_height is not None:
            h = pixel_height
            w = max(1, int(round(pixel_height * aspect)))
        else:
            target = int(np.clip(n_points / 10.0, 2e5, 2e6))
            w = int(max(
                256, np.sqrt(target * max(aspect, 1e-6)),
            ))
            h = int(max(256, target / max(w, 1)))

        resolution = extent_x / max(w, 1)
        return w, h, resolution

    @staticmethod
    def _get_colors(
        pcd: Union[PointCloud, SimplePointCloud],
        n: int,
    ) -> np.ndarray:
        """Return an ``(n, 3)`` colour array for *pcd* (white if absent)."""
        if hasattr(pcd, "colors"):
            colors = np.asarray(pcd.colors)
            if colors.ndim != 2 or colors.shape[0] != n:
                colors = np.ones((n, 3), dtype=np.float64)
        else:
            colors = np.ones((n, 3), dtype=np.float64)
        return colors

    @staticmethod
    def _rasterize(
        xs: np.ndarray,
        ys: np.ndarray,
        colors: np.ndarray,
        min_x: float,
        min_y: float,
        resolution: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Rasterize projected points into an image array.

        Points are processed in chunks so that a progress bar can be
        shown for large point clouds.

        Args:
            xs: Projected x-coordinates.
            ys: Projected y-coordinates.
            colors: Per-point colours ``(N, 3)`` in the 0-1 range.
            min_x: Minimum x in projected space.
            min_y: Minimum y in projected space.
            resolution: Metres per pixel.
            width: Output image width.
            height: Output image height.

        Returns:
            ``uint8`` numpy array of shape ``(height, width, 3)``.
        """
        n = len(xs)

        ix = ((xs - min_x) / resolution).astype(int)
        iy = height - 1 - ((ys - min_y) / resolution).astype(int)
        np.clip(ix, 0, width - 1, out=ix)
        np.clip(iy, 0, height - 1, out=iy)

        colors = np.asarray(colors, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[0] != n:
            colors = np.ones((n, 3), dtype=np.float64)

        splat = np.zeros((height, width, 3), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.int64)
        flat = iy * width + ix

        chunk_size = max(1, n // 100)
        desc = f"Rasterizing ({width}x{height} px)"
        for start in tqdm(range(0, n, chunk_size), desc=desc):
            end = min(start + chunk_size, n)
            chunk_flat = flat[start:end]
            np.add.at(
                splat.reshape(-1, 3), chunk_flat, colors[start:end],
            )
            np.add.at(counts.ravel(), chunk_flat, 1)

        mask = counts > 0
        splat[mask] /= counts[mask, np.newaxis]
        splat[~mask] = 1.0

        return (np.clip(splat, 0, 1) * 255).astype(np.uint8)

    def _draw_highlights(
        self,
        img: Image.Image,
        coords_3d: np.ndarray,
        radius: int,
        fill_colors,
        outline_colors,
        shape: str = "circle",
        radius_metres: Optional[float] = None,
    ) -> None:
        """Draw highlight markers onto *img* (modified in place).

        Args:
            img: PIL Image to draw onto.
            coords_3d: World coordinates ``(N, 3)``.
            radius: Marker radius in display pixels.  Ignored when
                *radius_metres* is set.
            fill_colors: A single RGB fill colour (or ``None`` for hollow)
                applied to every marker, or a length-N sequence of
                per-marker fill colours.
            outline_colors: Outline colour(s), single or per-marker; same
                broadcasting rules as *fill_colors*.
            shape: ``"circle"`` or ``"square"``.
            radius_metres: Marker diameter in metres.  When provided,
                overrides *radius* and scales the marker to span this
                distance in world space.
        """
        scale_x = img.width / self.width
        scale_y = img.height / self.height
        if radius_metres is not None:
            radius = radius_metres * scale_x / (2 * self.resolution)
        pixels = self.project(coords_3d)
        if pixels.ndim == 1:
            pixels = pixels[np.newaxis, :]
        n = len(pixels)
        fills = _as_per_point(fill_colors, n)
        outlines = _as_per_point(outline_colors, n)
        draw = ImageDraw.Draw(img)
        bbox_fn = draw.ellipse if shape == "circle" else draw.rectangle
        for (px, py), fill, outline in zip(pixels, fills, outlines):
            dx = px * scale_x
            dy = py * scale_y
            bbox_fn(
                [dx - radius, dy - radius, dx + radius, dy + radius],
                fill=fill,
                outline=outline,
            )

    @staticmethod
    def _resolve_marker_style(
        coords: np.ndarray,
        labels: list,
        groups: list,
        color_by: Optional[str],
        label_colors: Optional[dict],
        fill_by_group: bool,
        point_color: Optional[Tuple[int, int, int]],
        point_outline: Optional[Tuple[int, int, int]],
    ) -> Tuple[list, list]:
        """Compute per-point fill and outline colours for highlights.

        Filled markers use the resolved colour as fill with *point_outline*
        as the border; hollow markers (group-based) use ``None`` fill and
        the resolved colour as the outline.

        Returns:
            ``(fill_colors, outline_colors)`` lists, length ``len(coords)``.
        """
        n = len(coords)

        if color_by == "label" and any(lbl is not None for lbl in labels):
            lut = _label_color_lut(labels, label_colors)
            base_colors = [lut.get(lbl, point_color) for lbl in labels]
        elif color_by == "z":
            z = coords[:, 2].astype(float)
            zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
            if not (np.isfinite(zmin) and np.isfinite(zmax)) or zmin == zmax:
                zmin, zmax = zmin - 1e-6, zmax + 1e-6
            cmap = _get_cmap("bwr")
            norm = (z - zmin) / (zmax - zmin)
            base_colors = [_rgb255(cmap(float(t))) for t in norm]
        else:
            base_colors = [point_color] * n

        if fill_by_group and any(g is not None for g in groups):
            uniq_g = sorted({g for g in groups if g is not None}, key=str)
            g_fill = {g: (i % 2 == 0) for i, g in enumerate(uniq_g)}
            filled = [g_fill.get(g, True) for g in groups]
        else:
            filled = [True] * n

        fill_colors, outline_colors = [], []
        for col, is_filled in zip(base_colors, filled):
            if is_filled:
                fill_colors.append(col)
                outline_colors.append(point_outline)
            else:
                fill_colors.append(None)
                outline_colors.append(col)
        return fill_colors, outline_colors

    def _crop_to(self, img: Image.Image, crop: Tuple) -> Image.Image:
        """Crop *img* to a metre-space window ``(center, size_metres)``.

        The box is clamped to the image bounds so no padding is added.
        """
        center, size_m = crop
        center = np.asarray(center, dtype=np.float64).reshape(-1)
        if center.shape[0] < 3:
            center = np.array([center[0], center[1], 0.0])
        scale_x = img.width / self.width
        scale_y = img.height / self.height
        cx, cy = self.project(center[:3])
        cx *= scale_x
        cy *= scale_y
        half_w = (size_m / 2.0) / self.resolution * scale_x
        half_h = (size_m / 2.0) / self.resolution * scale_y
        left = max(0, int(round(cx - half_w)))
        top = max(0, int(round(cy - half_h)))
        right = min(img.width, int(round(cx + half_w)))
        bottom = min(img.height, int(round(cy + half_h)))
        if right <= left or bottom <= top:
            return img
        return img.crop((left, top, right, bottom))

    @staticmethod
    def _resize(
        img: Image.Image,
        width: Optional[int],
        height: Optional[int],
    ) -> Image.Image:
        """Resize *img* while preserving aspect ratio.

        Args:
            img: Source image.
            width: Target width (``None`` to derive from *height*).
            height: Target height (``None`` to derive from *width*).

        Returns:
            Resized PIL Image.
        """
        orig_w, orig_h = img.size
        aspect = orig_w / orig_h
        if width is not None and height is not None:
            if aspect >= width / height:
                new_w = width
                new_h = max(1, int(round(width / aspect)))
            else:
                new_h = height
                new_w = max(1, int(round(height * aspect)))
        elif width is not None:
            new_w = width
            new_h = max(1, int(round(width / aspect)))
        else:
            new_h = height
            new_w = max(1, int(round(height * aspect)))
        return img.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _extract_highlights(
        highlights,
    ) -> Tuple[np.ndarray, list, list]:
        """Normalise *highlights* into coords with label/group metadata.

        Accepts a single ``Annotation``, any container with a ``.data``
        mapping of items exposing ``.coords`` (and optionally ``.label``
        / ``.group``) such as ``Annotations`` or ``Cameras``, a numpy
        array of shape ``(N, 3)``, or a list of coordinates.

        Returns:
            ``(coords, labels, groups)`` where *coords* is an ``(N, 3)``
            array and *labels*/*groups* are length-N lists (entries may
            be ``None``).
        """
        # Single Annotation-like object (has .coords but not a container).
        if hasattr(highlights, "coords") and not hasattr(highlights, "data"):
            coords = np.asarray(highlights.coords, dtype=np.float64)
            return (
                coords[np.newaxis, :],
                [getattr(highlights, "label", None)],
                [getattr(highlights, "group", None)],
            )

        # Container with .data mapping to items with .coords.
        if hasattr(highlights, "data") and hasattr(highlights.data, "values"):
            coords, labels, groups = [], [], []
            for item in highlights.data.values():
                c = getattr(item, "coords", None)
                if c is None:
                    continue
                c = np.asarray(c, dtype=np.float64)
                if c.shape[0] >= 3:
                    coords.append(c[:3])
                    labels.append(getattr(item, "label", None))
                    groups.append(getattr(item, "group", None))
            if coords:
                return np.vstack(coords), labels, groups
            return np.empty((0, 3), dtype=np.float64), [], []

        arr = np.asarray(highlights, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        n = len(arr)
        return arr, [None] * n, [None] * n


class _HighlightItem:
    """Minimal highlight carrier consumed by :meth:`OrthoMap._extract_highlights`.

    Holds a shifted 3D ``coords`` plus optional ``label``/``group`` so that
    :class:`OrthoMapGroup` can re-home each plot's annotations into the
    composite layout frame without depending on the original item type.
    """

    __slots__ = ("coords", "label", "group")

    def __init__(self, coords, label=None, group=None):
        self.coords = coords
        self.label = label
        self.group = group


class _MergedHighlights:
    """Container with a ``.data`` mapping, matching the annotation-set duck type."""

    def __init__(self):
        self.data = {}


class OrthoMapGroup(OrthoMap):
    """Composite ortho of several plots laid out in a single frame.

    Unlike a plain :class:`OrthoMap`, the input clouds are **not** assumed to
    share a world frame.  The typical use case is several plots of the same
    location captured at different depths — each carries its own coordinate
    frame, scale, and orientation (already baked into ``pcd.points``).  Each
    plot keeps that scale/orientation; the group only **translates** each plot
    so they line up into one picture.

    Two layout modes (``arrange``):

    * ``"stack"`` (default) — a **depth-ordered vertical stack**.  Plots are
      ordered by mean height (``z``), shallowest (least-negative ``z``) on top
      down to deepest at the bottom, aligned horizontally, and separated by a
      definable vertical gap.  Order can be overridden with *order*.
    * ``None`` — the legacy behaviour: composite every cloud at its true world
      coordinates, assuming the clouds are already co-registered.

    Two per-plot orientations (``orient``):

    * ``"topdown"`` (default) — every plot is projected along the shared
      *up_vector* (a top-down view).
    * ``"slope"`` — each plot is viewed **face-on to its own best-fit plane**.
      This removes the foreshortening a top-down view imposes on steep plots,
      so the imagery reads at a consistent scale.  The orientation reuses the
      same building blocks as :meth:`PointCloud.apply_along_slope_transform`
      (PCA plane normal + :meth:`geom.Transform.from_up_vector`) but is
      computed non-destructively — the input clouds are never modified.  The
      tilt turns about a horizontal axis, so it adds **no in-plane (z-axis)
      rotation**: a near-flat plot is left essentially unchanged and each plot
      keeps its own azimuth.

    All :class:`OrthoMap` machinery (``show``, ``project``, highlight overlays,
    crop, resize) is inherited.  :meth:`show` accepts annotations either as a
    **per-plot list** parallel to *pcds* (each set is transformed with its plot)
    or as a **single combined set** that is auto-split across plots by position.

    Example::

        grp = OrthoMapGroup(
            [p.pcd for p in plots],          # shallow..deep, any order
            pixel_height=6000,
            vertical_spacing=1.0,            # metres between tiles
            names=[p.project_id for p in plots],
        )
        img = grp.show([p.annotations for p in plots])
    """

    def __init__(
        self,
        pcds,
        up_vector: Optional[Union[List[float], np.ndarray]] = None,
        pixel_width: Optional[int] = None,
        pixel_height: Optional[int] = None,
        rotation: int = 0,
        arrange: Optional[str] = "stack",
        order: Optional[List[int]] = None,
        vertical_spacing: float = 1.0,
        align: str = "centroid",
        names: Optional[List[str]] = None,
        orient: str = "topdown",
    ) -> None:
        """Initialize a composite ortho map from several plots.

        Args:
            pcds: Iterable of point clouds, one per plot.
            up_vector: World "up" direction (default z).  Used for the
                top-down projection and to measure each plot's depth for
                ordering.
            pixel_width: Optional target width in pixels.
            pixel_height: Optional target height in pixels.  Recommended over
                *pixel_width* for a tall vertical stack.
            rotation: In-plane rotation in degrees applied on ``show``.
            arrange: ``"stack"`` for a depth-ordered vertical layout
                (default) or ``None`` to composite clouds at their true world
                coordinates (legacy co-registered behaviour).
            order: Optional explicit stacking order as a list of plot indices
                (top to bottom).  ``None`` orders automatically by depth
                (shallow/least-negative on top).  Ignored when *arrange* is
                ``None``.
            vertical_spacing: Gap in metres between stacked plots' bounding
                boxes.  Ignored when *arrange* is ``None``.
            align: Horizontal alignment of the stacked plots — ``"centroid"``
                (default, aligns each plot's mean position), ``"center"``
                (bounding-box midpoint), ``"left"``, or ``"right"``.  Ignored
                when *arrange* is ``None``.
            names: Optional per-plot labels (parallel to *pcds*) drawn on the
                composite by :meth:`show` when ``show_labels`` is enabled.
            orient: ``"topdown"`` (default) projects every plot along
                *up_vector*; ``"slope"`` views each plot face-on to its own
                best-fit plane (down-slope pointing down).  See the class
                docstring.
        """
        pcds = list(pcds)
        if not pcds:
            raise ValueError("OrthoMapGroup requires at least one point cloud")
        if orient not in ("topdown", "slope"):
            raise ValueError(f"Unknown orient mode: {orient!r}")

        if up_vector is None:
            up_vector = np.array([0.0, 0.0, 1.0])
        up = np.asarray(up_vector, dtype=np.float64)
        up = up / np.linalg.norm(up)
        self._up_vector = up
        self._rotation = rotation
        # The composite is assembled directly in the layout plane, so the
        # inherited ``project`` uses an identity view rotation; each plot's own
        # view rotation is applied per plot (below and in ``_to_layout``).
        self.rotation = np.eye(3)
        self._arrange = arrange
        self._orient = orient
        self._names = list(names) if names is not None else None

        r_shared = self._rotation_to_z(up)

        # Project every plot into its own view plane and record, per plot, its
        # view rotation, rotated points/colours, 2D bbox, centroid and depth.
        n = len(pcds)
        rotations: List[Optional[np.ndarray]] = [None] * n
        rot_xs: List[Optional[np.ndarray]] = [None] * n
        rot_ys: List[Optional[np.ndarray]] = [None] * n
        cols: List[Optional[np.ndarray]] = [None] * n
        bboxes: List[Optional[Tuple[float, float, float, float]]] = [None] * n
        centroids: List[Optional[Tuple[float, float]]] = [None] * n
        depth_keys: List[float] = [float("nan")] * n
        total = 0
        for i, pcd in enumerate(pcds):
            points = np.asarray(pcd.points)
            if len(points) == 0:
                continue
            r_i = r_shared if orient == "topdown" else self._slope_rotation(
                points, up, r_shared,
            )
            rotations[i] = r_i
            rotated = (r_i @ points.T).T
            rx, ry = rotated[:, 0], rotated[:, 1]
            rot_xs[i], rot_ys[i] = rx, ry
            cols[i] = self._get_colors(pcd, len(points))
            bboxes[i] = (
                float(rx.min()), float(rx.max()),
                float(ry.min()), float(ry.max()),
            )
            centroids[i] = (float(rx.mean()), float(ry.mean()))
            # Depth for ordering is measured along the shared world up-vector,
            # independent of each plot's own view rotation.
            depth_keys[i] = float((points @ up).mean())
            total += len(points)
        if total == 0:
            raise ValueError("OrthoMapGroup point clouds contain no points")

        # Per-plot 2D offsets in the (per-plot) view plane.
        if arrange == "stack":
            offsets = self._compute_stack_offsets(
                bboxes, centroids, depth_keys, order, vertical_spacing, align,
            )
        elif arrange is None:
            offsets = [(0.0, 0.0)] * n
        else:
            raise ValueError(f"Unknown arrange mode: {arrange!r}")
        self._rotations = rotations
        self._offsets = offsets
        self._plot_bboxes = bboxes  # pre-shift, for auto-splitting annotations

        # Apply offsets and rasterize all plots into one frame.
        xs_all, ys_all, colors_all = [], [], []
        for i in range(n):
            if rot_xs[i] is None:
                continue
            dx, dy = offsets[i]
            xs_all.append(rot_xs[i] + dx)
            ys_all.append(rot_ys[i] + dy)
            colors_all.append(cols[i])
        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)
        colors = np.concatenate(colors_all, axis=0)

        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        extent_x = max(max_x - min_x, 1e-9)
        extent_y = max(max_y - min_y, 1e-9)
        self.origin = np.array([min_x, min_y])
        self.width, self.height, self.resolution = self._compute_dimensions(
            extent_x, extent_y, total, pixel_width, pixel_height,
        )
        self.image = self._rasterize(
            xs, ys, colors,
            min_x, min_y,
            self.resolution,
            self.width, self.height,
        )

        # Layout-space anchor for each tile's label: the composite's left edge
        # (so all labels share one x) at the tile's vertical centre.
        self._label_anchors: List[Optional[np.ndarray]] = [None] * n
        for i in range(n):
            if bboxes[i] is None:
                continue
            _, _, miny, maxy = bboxes[i]
            dy = offsets[i][1]
            cy = 0.5 * (miny + maxy) + dy
            self._label_anchors[i] = np.array([min_x, cy, 0.0])

        logger.info(
            "OrthoMapGroup created: %d x %d px  (%d clouds, %d points, "
            "arrange=%s, orient=%s)",
            self.width, self.height, n, total, arrange, orient,
        )

    def __repr__(self) -> str:
        """Return a concise summary of the composite ortho map."""
        return (
            f"OrthoMapGroup({self.width}x{self.height} px, "
            f"res={self.resolution:.6f} m/px, "
            f"up={self._up_vector.tolist()}, arrange={self._arrange!r}, "
            f"orient={self._orient!r})"
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _compute_stack_offsets(
        self, bboxes, centroids, depth_keys, order, vertical_spacing, align,
    ):
        """Compute per-plot ``(dx, dy)`` offsets for a vertical depth stack.

        Plots are placed top to bottom.  Because :meth:`_rasterize` maps
        larger view-plane ``y`` to the top image row, the first plot in the
        order (shallowest by default) ends up on top.

        Args:
            bboxes: Per-plot view-plane bounding boxes
                ``(minx, maxx, miny, maxy)`` (or ``None`` for empty plots).
            centroids: Per-plot view-plane ``(cx, cy)`` mean positions, used
                for ``align="centroid"``.
            depth_keys: Per-plot mean depth (along the world up-vector), used
                for the default ordering.
            order: Explicit list of plot indices (top to bottom), or ``None``
                to order by descending depth (shallow first).
            vertical_spacing: Gap in metres between consecutive tiles.
            align: ``"centroid"``, ``"center"``, ``"left"`` or ``"right"``.

        Returns:
            List of ``(dx, dy)`` offsets in original plot order; empty plots
            get ``(0.0, 0.0)``.
        """
        n = len(bboxes)
        valid = [i for i in range(n) if bboxes[i] is not None]
        if order is None:
            seq = sorted(valid, key=lambda i: depth_keys[i], reverse=True)
        else:
            seq = [i for i in order if i in valid]

        offsets = [(0.0, 0.0)] * n
        top = 0.0
        for i in seq:
            minx, maxx, miny, maxy = bboxes[i]
            if align == "centroid":
                dx = -centroids[i][0]
            elif align == "center":
                dx = -0.5 * (minx + maxx)
            elif align == "left":
                dx = -minx
            elif align == "right":
                dx = -maxx
            else:
                raise ValueError(f"Unknown align mode: {align!r}")
            dy = top - maxy
            offsets[i] = (dx, dy)
            top = top - (maxy - miny) - vertical_spacing
        return offsets

    def _slope_rotation(self, points, up, fallback):
        """Return a 3x3 view rotation that looks face-on to the plot's slope.

        Reuses the along-slope mechanism of
        :meth:`PointCloud.apply_along_slope_transform`: a PCA best-fit plane
        normal is mapped to +z with :meth:`geom.Transform.from_up_vector` so
        the plot is viewed face-on (removing foreshortening).  The one
        remaining in-plane degree of freedom is then used to keep the plot's
        **top-down heading** — the same world direction points "up" as in the
        top-down view — so the plot is not spun about the view z-axis.  That
        heading reference is a fixed world axis (not the slope gradient), so a
        near-flat plot gets no correction and never spins.  Computed without
        mutating the input; falls back to *fallback* (the shared top-down
        rotation) if the plane fit is degenerate.

        Args:
            points: ``(N, 3)`` world coordinates of the plot.
            up: Unit world up-vector.
            fallback: Shared top-down rotation; also used to derive the
                heading reference and returned if the plane fit is degenerate.

        Returns:
            A 3x3 rotation matrix.
        """
        from substrata import geom  # pure-numpy; safe without heavy deps

        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 3:
            return fallback
        if len(pts) > 50000:  # subsample large clouds for the plane fit
            pts = pts[:: len(pts) // 50000 + 1]

        # PCA best-fit plane normal (smallest-eigenvalue eigenvector), oriented
        # towards up — equivalent to measurements.get_best_fit_plane_PCA.
        cov = np.cov((pts - pts.mean(axis=0)).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, int(np.argmin(evals))]
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return fallback
        normal = normal / norm
        if float(normal @ up) < 0:
            normal = -normal

        m1 = geom.Transform.from_up_vector(normal).mat[:3, :3]  # face-on tilt
        # Heading reference: the world direction that maps to image +y in the
        # top-down view.  Rotate in-plane so it maps to +y here too, keeping
        # the plot's orientation and adding no spurious z-spin.
        world_up_axis = fallback.T @ np.array([0.0, 1.0, 0.0])
        ref = m1 @ world_up_axis
        if np.hypot(ref[0], ref[1]) < 1e-6:  # heading undefined (rare)
            return m1
        beta = float(np.arctan2(ref[1], ref[0]))
        m2 = geom.Transform.from_euler(0.0, 0.0, np.pi / 2 - beta).mat[:3, :3]
        return m2 @ m1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def show(
        self,
        annotations_list=None,
        color_by: Optional[str] = "label",
        show_labels: Optional[bool] = None,
        label_color: Tuple[int, int, int] = (0, 0, 0),
        font_size: Optional[int] = None,
        point_size: Optional[int] = None,
        point_size_metres: Optional[float] = None,
        label_rotation: int = 90,
        legend: bool = False,
        label_colors: Optional[dict] = None,
        **kwargs,
    ) -> Image.Image:
        """Render the composite map with annotations and optional tile labels.

        Marker and label sizes default to a fraction of the composite height
        (so they stay legible on the large rasters this class produces) and can
        be overridden.

        Args:
            annotations_list: Annotations to overlay.  Either a **per-plot
                list** parallel to *pcds* (each set is translated with its
                plot), a **single combined** container (auto-split across plots
                by position), a single ``Annotation``/coordinate array, or
                ``None`` for no overlay.
            color_by: Highlight colouring mode (default ``"label"``).
            show_labels: Draw per-plot *names*.  Defaults to ``True`` when
                *names* were provided and *arrange* is ``"stack"``.  Pass
                ``False`` to hide the per-plot tile labels.
            label_color: RGB colour for the tile labels.
            font_size: Label font size in pixels.  Defaults to ~1.25% of the
                composite height.
            point_size: Marker radius in pixels.  Defaults to ~0.25% of the
                composite height.  Ignored when *point_size_metres* is given.
            point_size_metres: Marker diameter in metres (consistent physical
                size across plots).  Overrides *point_size*.
            label_rotation: Rotation of the tile labels in degrees — ``90``
                (default) draws them vertically in a narrow left gutter, ``0``
                horizontally.
            legend: When True and *color_by* is ``"label"``, append a colour
                legend (swatch + label name) below the composite.
            label_colors: Optional explicit ``{label: (r, g, b)}`` map applied
                to both the markers and the legend.
            **kwargs: Forwarded to :meth:`OrthoMap.show` — notably
                ``image_opacity`` (``0``–``1``) to fade the reef imagery
                toward white so the annotation markers stand out (``1.0``
                unchanged, ``0.0`` imagery hidden), and ``grayscale``.

        Returns:
            PIL ``Image`` of the composite map.
        """
        if point_size_metres is not None:
            kwargs["point_size_metres"] = point_size_metres
        else:
            if point_size is None:
                point_size = max(3, round(self.height * 0.0025))
            kwargs["point_size"] = point_size
        if label_colors is not None:
            kwargs["label_colors"] = label_colors

        highlights = self._build_highlights(annotations_list)
        img = super().show(highlights=highlights, color_by=color_by, **kwargs)

        if font_size is None:
            font_size = max(8, round(self.height * 0.0125))

        if show_labels is None:
            show_labels = self._names is not None and self._arrange == "stack"
        if show_labels and self._names:
            img = self._draw_labels(img, label_color, font_size, label_rotation)

        if legend and color_by == "label":
            pool = []
            if highlights is not None and hasattr(highlights, "data"):
                pool = [getattr(it, "label", None)
                        for it in highlights.data.values()]
            img = self._draw_legend(
                img, pool, label_colors, font_size, label_color,
            )
        return img

    def _draw_labels(self, img, label_color, font_size, label_rotation=90):
        """Draw per-plot names in a left gutter, one per tile.

        A white margin is added to the left of the composite and each name is
        drawn there — right-aligned to a common x and vertically centred on its
        tile — so labels never overlap the imagery and line up with each other.
        *label_rotation* (``90`` by default) draws the text vertically for a
        narrow gutter; ``0`` draws it horizontally.  Returns the (widened)
        image.  Vertical placement honours a ``width``/``height`` resize but
        not ``crop`` or the display ``rotation``.
        """
        pairs = [
            (str(name), anchor)
            for name, anchor in zip(self._names, self._label_anchors)
            if name is not None and anchor is not None
        ]
        if not pairs:
            return img

        font = self._load_font(font_size)
        patches = [
            (self._render_text_patch(name, font, label_color, label_rotation),
             anchor)
            for name, anchor in pairs
        ]
        pad = max(4, round(font_size * 0.4))
        margin = max(p.width for p, _ in patches) + 2 * pad

        canvas = Image.new(
            "RGB", (img.width + margin, img.height), (255, 255, 255),
        )
        canvas.paste(img, (margin, 0))

        scale_y = img.height / self.height
        x_right = margin - pad
        for patch, anchor in patches:
            py = float(self.project(anchor)[1]) * scale_y
            x = int(round(x_right - patch.width))
            y = int(round(py - patch.height / 2))
            canvas.paste(patch, (x, y), patch)
        return canvas

    @staticmethod
    def _render_text_patch(text, font, color, rotation=0):
        """Render *text* to a tight RGBA patch, optionally rotated (degrees)."""
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bb = measure.textbbox((0, 0), text, font=font)
        w, h = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
        patch = Image.new("RGBA", (w + 2, h + 2), (0, 0, 0, 0))
        ImageDraw.Draw(patch).text(
            (1 - bb[0], 1 - bb[1]), text, font=font,
            fill=tuple(color) + (255,),
        )
        if rotation:
            patch = patch.rotate(rotation, expand=True)
        return patch

    def _draw_legend(self, img, labels, label_colors, font_size, text_color):
        """Append a label-colour legend below the composite (returns new img)."""
        uniq = sorted({lbl for lbl in labels if lbl is not None})
        if not uniq:
            return img
        lut = _label_color_lut(labels, label_colors)
        font = self._load_font(font_size)

        pad = max(4, round(font_size * 0.4))
        swatch = font_size
        row_h = swatch + pad
        measure = ImageDraw.Draw(img)
        text_w = max(
            measure.textbbox((0, 0), lbl, font=font)[2] for lbl in uniq
        )
        panel_h = len(uniq) * row_h + 2 * pad
        panel_w = max(img.width, 3 * pad + swatch + text_w)

        canvas = Image.new("RGB", (panel_w, img.height + panel_h),
                           (255, 255, 255))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)

        y = img.height + pad
        for lbl in uniq:
            col = tuple(lut.get(lbl, text_color))
            draw.rectangle([pad, y, pad + swatch, y + swatch],
                           fill=col, outline=(0, 0, 0))
            tx, ty = 2 * pad + swatch, y + swatch / 2
            try:
                draw.text((tx, ty), lbl, fill=text_color, font=font,
                          anchor="lm")
            except (ValueError, TypeError):  # bitmap font: no anchor support
                bb = draw.textbbox((0, 0), lbl, font=font)
                draw.text((tx, ty - (bb[3] - bb[1]) / 2), lbl,
                          fill=text_color, font=font)
            y += row_h
        return canvas

    @staticmethod
    def _load_font(font_size):
        """Load a scalable font at *font_size* px, falling back gracefully.

        Pillow's built-in bitmap default font does not scale, so a large
        ``font_size`` needs either ``load_default(size)`` (Pillow >= 10) or a
        TrueType font; older Pillow falls back to the fixed default.
        """
        from PIL import ImageFont

        try:  # Pillow >= 10 returns a scalable default at the requested size
            return ImageFont.load_default(size=font_size)
        except TypeError:
            pass
        for name in ("DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(name, font_size)
            except Exception:  # pragma: no cover - font not installed
                continue
        return ImageFont.load_default()  # last resort (fixed size)

    # ------------------------------------------------------------------
    # Annotation layout
    # ------------------------------------------------------------------

    def _build_highlights(self, annotations_list):
        """Normalise annotations into layout-shifted highlights.

        Returns a container whose ``.data`` items carry composite-frame
        coordinates, or ``None`` when there is nothing to draw.
        """
        if annotations_list is None:
            return None

        single = hasattr(annotations_list, "data") or hasattr(
            annotations_list, "coords"
        )
        n_plots = len(self._rotations)

        if not single:
            seq = list(annotations_list)
            # A list parallel to the plots (len == n_plots) is treated as
            # per-plot; entries may be ``None`` (a plot with no annotations)
            # and are skipped by :meth:`_merge_per_plot`.
            if len(seq) == n_plots and n_plots > 0:
                return self._merge_per_plot(seq)
            # A bare list of coordinates carries no plot association; auto-split
            # each coordinate across the plots by position.
            return self._autosplit_single(annotations_list)

        return self._autosplit_single(annotations_list)

    def _merge_per_plot(self, containers):
        """Map each plot's annotation set into the composite layout frame."""
        merged = _MergedHighlights()
        idx = 0
        for i, anns in enumerate(containers):
            if anns is None:
                continue
            for coords, label, group in self._iter_coord_label_group(anns):
                merged.data[idx] = _HighlightItem(
                    self._to_layout(coords, i), label, group,
                )
                idx += 1
        return merged if merged.data else None

    def _autosplit_single(self, container):
        """Assign each annotation to a plot by position, then map it."""
        merged = _MergedHighlights()
        idx = 0
        for coords, label, group in self._iter_coord_label_group(container):
            i = self._assign_plot(coords)
            merged.data[idx] = _HighlightItem(
                self._to_layout(coords, i), label, group,
            )
            idx += 1
        return merged if merged.data else None

    def _to_layout(self, coord, i):
        """Map a world *coord* of plot *i* into composite layout coordinates.

        Applies the plot's view rotation and layout offset so the inherited
        (identity-rotation) ``project`` lands it on the right tile.  The z
        component carries world depth so ``color_by="z"`` stays meaningful.
        """
        c = np.asarray(coord, dtype=np.float64)
        r = self._rotations[i] @ c
        dx, dy = self._offsets[i]
        return np.array([r[0] + dx, r[1] + dy, float(c @ self._up_vector)])

    def _assign_plot(self, coord):
        """Index of the plot whose view-plane bbox best matches *coord*."""
        c = np.asarray(coord, dtype=np.float64)
        contained, best, best_d = [], 0, float("inf")
        for i, bb in enumerate(self._plot_bboxes):
            if bb is None:
                continue
            r = self._rotations[i] @ c
            rx, ry = float(r[0]), float(r[1])
            minx, maxx, miny, maxy = bb
            cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
            d = (rx - cx) ** 2 + (ry - cy) ** 2
            if minx <= rx <= maxx and miny <= ry <= maxy:
                contained.append((d, i))
            if d < best_d:
                best_d, best = d, i
        return min(contained)[1] if contained else best

    def _iter_coord_label_group(self, anns):
        """Yield ``(coords, label, group)`` from any annotation-set form."""
        if hasattr(anns, "data") and hasattr(anns.data, "values"):
            for item in anns.data.values():
                c = getattr(item, "coords", None)
                if c is None:
                    continue
                c = np.asarray(c, dtype=np.float64)
                if c.shape[0] >= 3:
                    yield (
                        c[:3], _resolve_label(item),
                        getattr(item, "group", None),
                    )
        elif hasattr(anns, "coords"):
            c = np.asarray(anns.coords, dtype=np.float64)
            if c.shape[0] >= 3:
                yield (
                    c[:3], _resolve_label(anns),
                    getattr(anns, "group", None),
                )
        else:
            arr = np.asarray(anns, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            for c in arr:
                if c.shape[0] >= 3:
                    yield c[:3], None, None


def _resolve_label(ann) -> Optional[str]:
    """Best available classification label for an annotation, or ``None``.

    Prefers the classifier result (``ann.image_match.classification['label']``)
    and falls back to the plain ``ann.label``. Empty strings are treated as
    missing. Duck-typed so ``ortho`` needs no ``annotations`` import.
    """
    im = getattr(ann, "image_match", None)
    cls = getattr(im, "classification", None)
    if isinstance(cls, dict) and cls.get("label") is not None:
        return str(cls["label"])
    lbl = getattr(ann, "label", None)
    return str(lbl) if lbl not in (None, "") else None


def _optional_point_array(pcd, attr: str) -> Optional[np.ndarray]:
    """Return ``pcd.<attr>`` as an ``(N, k)`` array, or ``None`` if absent/empty.

    Point clouds may lack colours or normals; open3d exposes them as
    (possibly empty) buffers. Duck-typed so ``ortho`` needs no point-cloud
    import.
    """
    try:
        arr = np.asarray(getattr(pcd, attr))
    except Exception:  # pragma: no cover - attribute missing or unconvertible
        return None
    return arr if arr.size else None


class OrthoGrid:
    """A metre-cell grid over a point cloud, reduced to a value per cell.

    Bins the cloud (or a set of annotations) into square XY cells of
    ``cell_size`` metres and reduces each cell to a scalar or label:

    - ``value_by="z"``   — height (DEM); ``agg`` in mean/median/max/min.
    - ``value_by="count"`` / ``"density"`` — points per cell / per m².
    - ``value_by="label"`` — majority annotation label per cell.
    - ``value_by="custom"`` — run a :mod:`substrata.measurements` function
      (``measurement``) on each cell's own points and take one output scalar
      (``metric``); e.g. ``measurement=calc_roughness, metric="Ra"``.

    The lattice spans the full extent of the reduced data ("grid everywhere").
    A single ``bbox`` (or a set of ``intercepts`` from which one is derived)
    defines the **reporting area**: the side-panel and any summary aggregate
    only cells whose centre lies inside it, while cells are still rendered
    everywhere.

    Attributes:
        cell_size: Cell side length in metres.
        x0, y0: Lattice origin (min corner) in the projected frame.
        nx, ny: Grid dimensions in cells.
        counts: ``(ny, nx)`` int array of points (or annotations) per cell.
        present: ``(ny, nx)`` bool array, True where the cell holds data.
        values: ``(ny, nx)`` float array (NaN where absent) for continuous
            reductions; ``None`` in label mode.
        cell_labels: ``(ny, nx)`` object array of labels for label mode; else
            ``None``.
        report_mask: ``(ny, nx)`` bool array of cells inside the reporting area.
        report_bbox: ``((x0, y0), (x1, y1))`` reporting area, or ``None``.
        info: Diagnostics dict (populated when built from intercepts).
    """

    # Default output key per recognised measurement for ``value_by="custom"``.
    # Keyed by ``measurement.__name__``; used when *metric* is not given.
    _CUSTOM_METRIC_DEFAULTS = {
        "calc_roughness": "Ra",
        "get_dev_rugosity": "dev_rug",
        "get_vector_dispersion": "vector_disp",
        "get_plane_angles": "theta",
        "get_rgb_stats": "luminance",
        "calc_tpi_and_tri": "tpi_abs",
    }

    def __init__(
        self,
        pcd=None,
        annotations=None,
        value_by: str = "z",
        agg: str = "mean",
        cell_size: Optional[float] = None,
        bbox=None,
        intercepts=None,
        up_vector: Optional[Union[List[float], np.ndarray]] = None,
        label_colors: Optional[dict] = None,
        measurement: Optional[Callable] = None,
        metric: Optional[str] = None,
        neighborhood_scale: float = 1.0,
        min_points: int = 10,
        value_label: Optional[str] = None,
    ) -> None:
        """Build the grid and reduce every cell.

        Args:
            pcd: Point cloud (for ``z``/``count``/``density``/``custom`` and the
                context background).
            annotations: Annotations (required for ``value_by="label"``).
            value_by: ``"z"`` | ``"count"`` | ``"density"`` | ``"label"`` |
                ``"custom"``.
            agg: Aggregation for ``value_by="z"`` — mean/median/max/min.
            cell_size: Cell side in metres (default
                ``settings.DEFAULT_ORTHO_CELL_SIZE``).
            bbox: Reporting-area bbox ``([xmin, ymin], [xmax, ymax])``.
            intercepts: Alternative to *bbox* — an ``Annotations``/array of
                on-grid intercepts from which the reporting bbox and lattice
                alignment are recovered via
                :meth:`_fit_grid_from_intercepts`.
            up_vector: Projection up (default top-down z).
            label_colors: Optional ``{label: rgb}`` map for label mode.
            measurement: For ``value_by="custom"`` — a measurement function from
                :mod:`substrata.measurements` (e.g. ``calc_roughness``). It is
                run on each cell's own points (a ``SimplePointCloud`` built from
                the points that fall in the cell) via
                :meth:`substrata.annotations.Annotation.measure`, and the scalar
                named by *metric* becomes the cell value. Only measurements that
                operate on the neighbourhood cloud are supported (roughness,
                dev-rugosity, vector-dispersion, plane-angles, RGB stats, TPI/
                TRI); camera-/model-based ones (gap fraction, mask surface area,
                benthic fraction) are not.
            metric: Which output key of *measurement* to use as the cell value
                (e.g. ``"Ra"`` or ``"Rq"`` for ``calc_roughness``). Defaults to a
                sensible per-measurement key.
            neighborhood_scale: For ``value_by="custom"`` — widen the per-cell
                sampling window to a square of side
                ``cell_size * neighborhood_scale`` centred on the cell (``1.0``
                = exactly the cell; ``2.0`` = a 2×2-cell square). Must be
                ``>= 1.0``. Larger values borrow neighbouring points for a more
                robust fit on sparse grids.
            min_points: For ``value_by="custom"`` — cells whose sampling window
                holds fewer than this many points are left ``NaN`` instead of
                being measured. Guards plane-fit measurements against
                sparse/near-degenerate edge cells (which otherwise emit numeric
                warnings and nonsense values). Must be ``>= 1``.
            value_label: Optional colorbar/axis label (defaults to *metric* in
                custom mode).
        """
        if bbox is not None and intercepts is not None:
            raise ValueError("Provide either bbox or intercepts, not both")
        if value_by not in ("z", "count", "density", "label", "custom"):
            raise ValueError(f"invalid value_by: {value_by!r}")
        if agg not in ("mean", "median", "max", "min"):
            raise ValueError(f"invalid agg: {agg!r}")

        if value_by == "custom":
            if pcd is None:
                raise ValueError("value_by='custom' requires a point cloud (pcd)")
            if not callable(measurement):
                raise ValueError(
                    "value_by='custom' requires a callable measurement function"
                )
            if neighborhood_scale < 1.0:
                raise ValueError("neighborhood_scale must be >= 1.0")
            if min_points < 1:
                raise ValueError("min_points must be >= 1")
            if metric is None:
                metric = self._CUSTOM_METRIC_DEFAULTS.get(
                    getattr(measurement, "__name__", None)
                )
                if metric is None:
                    raise ValueError(
                        "metric could not be inferred for measurement "
                        f"{getattr(measurement, '__name__', measurement)!r}; "
                        "pass metric explicitly"
                    )

        self.value_by = value_by
        self.agg = agg
        self.pcd = pcd
        self.annotations = annotations
        self.label_colors = label_colors
        self.measurement = measurement
        self.metric = metric
        self.neighborhood_scale = float(neighborhood_scale)
        self.min_points = int(min_points)
        self.value_label = value_label
        self.info: dict = {}

        if cell_size is None:
            try:
                from substrata import settings

                cell_size = getattr(settings, "DEFAULT_ORTHO_CELL_SIZE", 0.1)
            except Exception:  # pragma: no cover - settings always importable
                cell_size = 0.1
        self.cell_size = float(cell_size)
        cs = self.cell_size

        if up_vector is None:
            up_vector = np.array([0.0, 0.0, 1.0])
        up = np.asarray(up_vector, dtype=np.float64)
        up = up / np.linalg.norm(up)
        self._up_vector = up
        self.rotation = OrthoMap._rotation_to_z(up)

        # Projected point coordinates (rotated frame).
        pts_xy = pts_z = None
        self._world_points = self._world_colors = self._world_normals = None
        if pcd is not None and len(np.asarray(pcd.points)):
            world_pts = np.asarray(pcd.points, dtype=float)
            proj = (self.rotation @ world_pts.T).T
            pts_xy, pts_z = proj[:, :2], proj[:, 2]
            if value_by == "custom":
                # Keep the world-frame arrays so per-cell neighbourhoods can be
                # sliced out for the measurement (see _reduce_custom).
                self._world_points = world_pts
                self._world_colors = _optional_point_array(pcd, "colors")
                self._world_normals = _optional_point_array(pcd, "normals")

        # Projected annotation coordinates + labels.
        ann_src = annotations if annotations is not None else intercepts
        ann_xy = None
        ann_labels: list = []
        if ann_src is not None and hasattr(ann_src, "data"):
            items = list(ann_src.data.values())
            if items:
                coords = np.array(
                    [np.asarray(a.coords, dtype=float) for a in items]
                )
                ann_xy = (self.rotation @ coords.T).T[:, :2]
                ann_labels = [_resolve_label(a) for a in items]

        # Reporting area + lattice phase anchor.
        if intercepts is not None:
            xy = self._intercept_xy(intercepts, self.rotation)
            px, py, self.report_bbox, self.info = self._fit_grid_from_intercepts(
                xy, cs
            )
        elif bbox is not None:
            (bx0, by0), (bx1, by1) = self._bbox_corners(bbox)
            px, py = float(bx0), float(by0)
            self.report_bbox = ((float(bx0), float(by0)), (float(bx1), float(by1)))
        else:
            px = py = None
            self.report_bbox = None

        # Lattice extent = extent of the reduced data (pcd for z/count/density,
        # annotations for label), unioned with the reporting bbox.
        ext_xy = pts_xy if value_by != "label" else ann_xy
        if ext_xy is None or len(ext_xy) == 0:
            ext_xy = pts_xy if pts_xy is not None else ann_xy
        if ext_xy is None or len(ext_xy) == 0:
            raise ValueError("OrthoGrid: no point/annotation data to grid")

        xs_min, ys_min = ext_xy.min(axis=0)
        xs_max, ys_max = ext_xy.max(axis=0)
        if self.report_bbox is not None:
            (rx0, ry0), (rx1, ry1) = self.report_bbox
            xs_min, ys_min = min(xs_min, rx0), min(ys_min, ry0)
            xs_max, ys_max = max(xs_max, rx1), max(ys_max, ry1)

        if px is None:
            px = np.floor(xs_min / cs) * cs
            py = np.floor(ys_min / cs) * cs

        i_min = int(np.floor((xs_min - px) / cs))
        j_min = int(np.floor((ys_min - py) / cs))
        i_max = int(np.floor((xs_max - px) / cs))
        j_max = int(np.floor((ys_max - py) / cs))
        self.x0 = px + i_min * cs
        self.y0 = py + j_min * cs
        self.nx = i_max - i_min + 1
        self.ny = j_max - j_min + 1

        # Reduce.
        self.values = None
        self.cell_labels = None
        if value_by == "label":
            self._reduce_label(ann_xy, ann_labels)
        elif value_by == "custom":
            self._reduce_custom(pts_xy, pts_z)
        else:
            self._reduce_continuous(pts_xy, pts_z)
        self._build_report_mask()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_corners(bbox):
        """Return ``((x0, y0), (x1, y1))`` from a bbox in any nested form."""
        (a, b) = bbox
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        return (a[0], a[1]), (b[0], b[1])

    @staticmethod
    def _intercept_xy(intercepts, rotation) -> np.ndarray:
        """Projected XY of an intercept set (``.data`` container or array)."""
        if hasattr(intercepts, "data"):
            coords = np.array(
                [np.asarray(a.coords, dtype=float) for a in intercepts.data.values()]
            )
        else:
            coords = np.asarray(intercepts, dtype=float)
        if coords.size == 0:
            return np.empty((0, 2))
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])
        return (rotation @ coords.T).T[:, :2]

    def _cell_index(self, xy: np.ndarray):
        """Return ``(ix, iy, in_bounds)`` cell indices for projected XY."""
        ix = np.floor((xy[:, 0] - self.x0) / self.cell_size).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / self.cell_size).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, in_bounds

    def _reduce_continuous(self, pts_xy, pts_z) -> None:
        """Fill ``counts``/``present``/``values`` for z/count/density."""
        ny, nx = self.ny, self.nx
        self.counts = np.zeros((ny, nx), dtype=np.int64)
        self.values = np.full((ny, nx), np.nan, dtype=float)
        if pts_xy is None or len(pts_xy) == 0:
            self.present = self.counts > 0
            return
        ix, iy, ok = self._cell_index(pts_xy)
        ix, iy = ix[ok], iy[ok]
        z = pts_z[ok]
        flat = iy * nx + ix
        np.add.at(self.counts.ravel(), flat, 1)
        self.present = self.counts > 0

        if self.value_by == "count":
            self.values[self.present] = self.counts[self.present]
        elif self.value_by == "density":
            area = self.cell_size ** 2
            self.values[self.present] = self.counts[self.present] / area
        else:  # value_by == "z"
            self._reduce_z(flat, z, nx, ny)

    def _reduce_custom(self, pts_xy, pts_z) -> None:
        """Run ``self.measurement`` on each cell's points into ``self.values``.

        Every occupied cell's own points — optionally widened to a square
        window of side ``cell_size * neighborhood_scale`` centred on the cell —
        are wrapped in a ``SimplePointCloud`` and handed to the measurement
        function (via a synthetic ``Annotation``); the scalar named by
        ``self.metric`` becomes the cell value. Cells that fail (too few points,
        etc.) are left ``NaN``.
        """
        ny, nx = self.ny, self.nx
        self.counts = np.zeros((ny, nx), dtype=np.int64)
        self.values = np.full((ny, nx), np.nan, dtype=float)
        if pts_xy is None or len(pts_xy) == 0:
            self.present = self.counts > 0
            return

        ix, iy, ok = self._cell_index(pts_xy)
        idx_all = np.nonzero(ok)[0]  # parent point indices that fall in-grid
        ix, iy = ix[ok], iy[ok]
        flat = iy * nx + ix
        np.add.at(self.counts.ravel(), flat, 1)
        self.present = self.counts > 0

        # Group parent point indices by flat cell id (single sort).
        order = np.argsort(flat, kind="stable")
        flat_sorted, idx_sorted = flat[order], idx_all[order]
        uniq, first, counts = np.unique(
            flat_sorted, return_index=True, return_counts=True
        )
        cell_members = {
            int(u): idx_sorted[f:f + c] for u, f, c in zip(uniq, first, counts)
        }

        cs = self.cell_size
        half = cs * self.neighborhood_scale / 2.0
        r = int(np.ceil(half / cs - 0.5)) if self.neighborhood_scale > 1.0 else 0

        occ = np.argwhere(self.present)
        fails = skipped = 0
        for jy, jx in tqdm(
            occ,
            desc="OrthoGrid custom ({})".format(
                getattr(self.measurement, "__name__", "measurement")
            ),
            disable=len(occ) == 0,
        ):
            jy, jx = int(jy), int(jx)
            cx = self.x0 + (jx + 0.5) * cs
            cy = self.y0 + (jy + 0.5) * cs
            own = cell_members.get(jy * nx + jx)
            if r == 0:
                member_idx = own
            else:
                member_idx = self._window_members(
                    cell_members, jy, jx, r, nx, ny, pts_xy, cx, cy, half
                )
            # Too few points for a stable measurement (e.g. a sparse edge cell
            # that would give a degenerate plane fit): leave NaN, not a failure.
            if member_idx is None or len(member_idx) < self.min_points:
                skipped += 1
                continue
            z_cell = (
                float(np.mean(pts_z[own])) if own is not None and len(own) else 0.0
            )
            center_world = self.rotation.T @ np.array([cx, cy, z_cell])
            try:
                self.values[jy, jx] = self._measure_cell(member_idx, center_world)
            except Exception as exc:  # keep going; a bad cell is just NaN
                fails += 1
                logger.debug("custom cell (%d,%d) failed: %s", jx, jy, exc)
        if skipped:
            logger.info(
                "value_by='custom': %d/%d cells below min_points=%d (left NaN)",
                skipped, len(occ), self.min_points,
            )
        if fails:
            logger.warning(
                "value_by='custom': %d/%d cells failed the measurement",
                fails, len(occ),
            )

    @staticmethod
    def _window_members(cell_members, jy, jx, r, nx, ny, pts_xy, cx, cy, half):
        """Parent point indices inside the square window around cell (jy, jx).

        Gathers candidates from the surrounding ``(2r+1)²`` bin block, then
        applies an exact Chebyshev-distance mask so the window is precisely a
        square of half-side ``half`` centred on ``(cx, cy)``.
        """
        parts = [
            cell_members[by * nx + bx]
            for by in range(max(0, jy - r), min(ny, jy + r + 1))
            for bx in range(max(0, jx - r), min(nx, jx + r + 1))
            if (by * nx + bx) in cell_members
        ]
        if not parts:
            return None
        cand = np.concatenate(parts)
        pxy = pts_xy[cand]
        keep = (np.abs(pxy[:, 0] - cx) <= half) & (np.abs(pxy[:, 1] - cy) <= half)
        return cand[keep]

    def _measure_cell(self, idx, center_world):
        """Measure one cell's neighbourhood, returning the ``self.metric`` scalar.

        Builds a ``SimplePointCloud`` from the parent cloud's ``idx`` points
        (world frame) and runs ``self.measurement`` through a synthetic
        ``Annotation`` so the measurement's output-key mapping is reused.
        Returns ``NaN`` when there are no points or the metric is absent.
        """
        if idx is None or len(idx) == 0:
            return np.nan
        from substrata import pointclouds
        from substrata.annotations import Annotation

        colors = None if self._world_colors is None else self._world_colors[idx]
        normals = None if self._world_normals is None else self._world_normals[idx]
        simple = pointclouds.SimplePointCloud(
            self._world_points[idx], colors, normals
        )
        ann = Annotation(coords=np.asarray(center_world, dtype=float))
        # Half the cell diagonal — a sensible inner radius for location-anchored
        # measurements (e.g. TPI); unused by neighbourhood-only measurements.
        ann.radius = self.cell_size * np.sqrt(2.0) / 2.0
        ann.simple_pcd = simple
        # A marginal cell can still trip expected numeric warnings inside the
        # measurement (degenerate covariance, complex eigvecs); keep them out of
        # the per-cell output. Genuine errors still raise and are handled above.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            ann.measure(self.measurement, generate_image=False)
        val = ann.measurements.get(self.metric)
        return float(val) if val is not None else np.nan

    def _reduce_z(self, flat, z, nx, ny) -> None:
        """Per-cell z aggregation into ``self.values``."""
        vals = np.full(nx * ny, np.nan, dtype=float)
        if self.agg == "mean":
            sums = np.zeros(nx * ny, dtype=float)
            np.add.at(sums, flat, z)
            occ = self.counts.ravel() > 0
            vals[occ] = sums[occ] / self.counts.ravel()[occ]
        elif self.agg == "max":
            acc = np.full(nx * ny, -np.inf, dtype=float)
            np.maximum.at(acc, flat, z)
            vals[np.isfinite(acc)] = acc[np.isfinite(acc)]
        elif self.agg == "min":
            acc = np.full(nx * ny, np.inf, dtype=float)
            np.minimum.at(acc, flat, z)
            vals[np.isfinite(acc)] = acc[np.isfinite(acc)]
        else:  # median
            order = np.argsort(flat, kind="stable")
            sflat, sz = flat[order], z[order]
            uniq, starts, counts = np.unique(
                sflat, return_index=True, return_counts=True
            )
            for u, s, c in zip(uniq, starts, counts):
                vals[u] = float(np.median(sz[s:s + c]))
        self.values = vals.reshape(ny, nx)

    def _reduce_label(self, ann_xy, ann_labels) -> None:
        """Majority-vote label per cell into ``self.cell_labels``."""
        from collections import Counter, defaultdict

        ny, nx = self.ny, self.nx
        self.counts = np.zeros((ny, nx), dtype=np.int64)
        self.cell_labels = np.empty((ny, nx), dtype=object)
        self.cell_labels[:] = None
        if ann_xy is None or len(ann_xy) == 0:
            self.present = self.counts > 0
            return
        ix, iy, ok = self._cell_index(ann_xy)
        cellvotes = defaultdict(Counter)
        for keep, cx, cy, lbl in zip(ok, ix, iy, ann_labels):
            if not keep:
                continue
            self.counts[cy, cx] += 1
            cellvotes[(cy, cx)][lbl] += 1
        for (cy, cx), votes in cellvotes.items():
            valid = {lbl: n for lbl, n in votes.items() if lbl is not None}
            if valid:
                top = max(valid.values())
                self.cell_labels[cy, cx] = sorted(
                    lbl for lbl, n in valid.items() if n == top
                )[0]
        self.present = self.counts > 0

    def _build_report_mask(self) -> None:
        """Cells whose centre lies inside the reporting bbox (all if None)."""
        if self.report_bbox is None:
            self.report_mask = np.ones((self.ny, self.nx), dtype=bool)
            return
        (rx0, ry0), (rx1, ry1) = self.report_bbox
        cs = self.cell_size
        cx = self.x0 + (np.arange(self.nx) + 0.5) * cs
        cy = self.y0 + (np.arange(self.ny) + 0.5) * cs
        mx = (cx >= rx0) & (cx <= rx1)
        my = (cy >= ry0) & (cy <= ry1)
        self.report_mask = np.outer(my, mx)

    @staticmethod
    def _fit_grid_from_intercepts(xy: np.ndarray, cell_size: float):
        """Recover a grid lattice from on-grid intercept XY (no point cloud).

        Ported from ``measurements.get_bboxes_from_intercepts``: scans the
        sub-cell origin offset for the phase that maximises one-point-per-cell
        occupancy, then trims sparse outer rows/columns.

        Args:
            xy: ``(N, 2)`` projected intercept coordinates.
            cell_size: Generation cell side length in metres.

        Returns:
            ``(x0, y0, report_bbox, info)`` — the trimmed lattice origin, the
            reporting bbox ``((x0, y0), (x1, y1))`` covering the fitted grid,
            and an ``info`` diagnostics dict.
        """
        cs = float(cell_size)
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return 0.0, 0.0, None, {}
        x, y = xy[:, 0], xy[:, 1]
        x_lo, y_lo = x.min(), y.min()
        step = cs / 50.0

        def _evaluate(ox, oy):
            x0 = np.floor((x_lo - ox) / cs) * cs + ox
            y0 = np.floor((y_lo - oy) / cs) * cs + oy
            ix = np.floor((x - x0) / cs).astype(np.int64)
            iy = np.floor((y - y0) / cs).astype(np.int64)
            ny = int(iy.max()) + 1
            _, counts = np.unique(ix * (ny + 1) + iy, return_counts=True)
            return int((counts == 1).sum()), (x0, y0, int(ix.max()) + 1, ny)

        best = None
        offsets = np.arange(0.0, cs, step)
        for ox in offsets:
            for oy in offsets:
                singles, meta = _evaluate(ox, oy)
                if best is None or singles > best[0]:
                    best = (singles, meta)
        x0, y0, nx, ny = best[1]

        ix = np.floor((x - x0) / cs).astype(np.int64)
        iy = np.floor((y - y0) / cs).astype(np.int64)
        occ_cells = set(zip(ix.tolist(), iy.tolist()))
        col_occ = np.zeros(nx, dtype=int)
        row_occ = np.zeros(ny, dtype=int)
        for ci, cj in occ_cells:
            col_occ[ci] += 1
            row_occ[cj] += 1

        i_lo, i_hi, j_lo, j_hi = 0, nx - 1, 0, ny - 1
        edge_min_fraction = 0.25
        if occ_cells:
            col_thr = edge_min_fraction * ny
            row_thr = edge_min_fraction * nx
            while i_lo < i_hi and col_occ[i_lo] < col_thr:
                i_lo += 1
            while i_hi > i_lo and col_occ[i_hi] < col_thr:
                i_hi -= 1
            while j_lo < j_hi and row_occ[j_lo] < row_thr:
                j_lo += 1
            while j_hi > j_lo and row_occ[j_hi] < row_thr:
                j_hi -= 1

        gnx, gny = i_hi - i_lo + 1, j_hi - j_lo + 1
        ox0 = x0 + i_lo * cs
        oy0 = y0 + j_lo * cs
        report_bbox = ((float(ox0), float(oy0)),
                       (float(ox0 + gnx * cs), float(oy0 + gny * cs)))

        in_grid = (ix >= i_lo) & (ix <= i_hi) & (iy >= j_lo) & (iy <= j_hi)
        _, counts = np.unique(
            ix[in_grid] * (ny + 1) + iy[in_grid], return_counts=True
        )
        occupied = len(counts)
        info = {
            "origin": (float(ox0), float(oy0)),
            "nx": gnx,
            "ny": gny,
            "n_cells": gnx * gny,
            "empty": gnx * gny - occupied,
            "multi": occupied - int((counts == 1).sum()),
        }
        logger.info(
            "OrthoGrid: fitted %dx%d grid from %d intercepts (%d empty, %d multi)",
            gnx, gny, len(xy), info["empty"], info["multi"],
        )
        return float(ox0), float(oy0), report_bbox, info

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def value_at(self, x: float, y: float) -> Optional[dict]:
        """Return the cell record at world ``(x, y)``, or ``None`` if outside.

        Returns a dict ``{ix, iy, cell_bbox, n_points, value, label,
        in_report}``. ``value`` is the reduced scalar (``None`` in label mode);
        ``label`` is the cell's majority label when annotations were supplied.
        """
        pt = self.rotation @ np.array([float(x), float(y), 0.0])
        ix = int(np.floor((pt[0] - self.x0) / self.cell_size))
        iy = int(np.floor((pt[1] - self.y0) / self.cell_size))
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return None
        cs = self.cell_size
        cell_bbox = (
            [self.x0 + ix * cs, self.y0 + iy * cs],
            [self.x0 + (ix + 1) * cs, self.y0 + (iy + 1) * cs],
        )
        value = None
        if self.values is not None:
            v = self.values[iy, ix]
            value = None if np.isnan(v) else float(v)
        label = None
        if self.cell_labels is not None:
            label = self.cell_labels[iy, ix]
        return {
            "ix": ix,
            "iy": iy,
            "cell_bbox": cell_bbox,
            "n_points": int(self.counts[iy, ix]),
            "value": value,
            "label": label,
            "in_report": bool(self.report_mask[iy, ix]),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _value_label(self) -> str:
        if self.value_by == "custom":
            return self.value_label or self.metric or "value"
        return {
            "z": "z (m)",
            "count": "count",
            "density": "points / m²",
        }.get(self.value_by, self.value_by)

    @property
    def extent(self):
        """imshow extent ``[x0, x1, y0, y1]`` for the lattice."""
        cs = self.cell_size
        return [self.x0, self.x0 + self.nx * cs,
                self.y0, self.y0 + self.ny * cs]

    def _draw_context(self, ax) -> Optional[list]:
        """Draw a faded grayscale OrthoMap of ``pcd`` behind the grid.

        Returns the full point-cloud extent ``[x0, x1, y0, y1]`` that was
        drawn (so the caller can widen the axes to it), or ``None`` when no
        point cloud is available.
        """
        if self.pcd is None:
            return None
        try:
            om = OrthoMap(self.pcd, up_vector=self._up_vector)
        except Exception:  # pragma: no cover - context is best-effort
            return None
        gray = np.asarray(om.image, dtype=float).mean(axis=2) / 255.0
        ext = [om.origin[0], om.origin[0] + om.width * om.resolution,
               om.origin[1], om.origin[1] + om.height * om.resolution]
        ax.imshow(gray, cmap="gray", extent=ext, origin="upper",
                  alpha=0.4, zorder=0, vmin=0.0, vmax=1.0)
        return ext

    def show(
        self,
        cmap: Optional[str] = None,
        show_pcd: bool = True,
        title: Optional[str] = None,
        label_colors: Optional[dict] = None,
        figsize: Tuple[float, float] = (18, 7.5),
        robust: bool = False,
        robust_percentiles: Tuple[float, float] = (2.0, 98.0),
        highlights: Optional[
            Union["Annotation", "Annotations", List[np.ndarray], np.ndarray]
        ] = None,
        color_by: Optional[str] = "auto",
        fill_by_group: bool = False,
        point_size: float = 5,
        point_size_metres: Optional[float] = None,
        point_color: Optional[Tuple[int, int, int]] = (255, 0, 0),
        point_outline: Optional[Tuple[int, int, int]] = (0, 0, 0),
        point_shape: str = "circle",
        highlight_label_colors: Optional[dict] = None,
    ):
        """Render the grid as a matplotlib Figure with a side panel.

        Left: the whole point cloud (faded grayscale) with the grid cells
        overlaid — the colormapped raster (continuous) or label-filled cells;
        the reporting bbox is outlined. Right: a histogram of report-cell
        values (continuous) or a bar chart of per-label cell counts (label
        mode).

        Args:
            cmap: Matplotlib colormap for continuous modes (default viridis).
            show_pcd: Show the entire point cloud (faded grayscale) behind the
                grid and widen the view to its full extent. Set ``False`` to
                show only the grid (clipped to the grid extent).
            title: Optional left-panel title.
            label_colors: Optional ``{label: color}`` map (label mode).
            figsize: Figure size in inches.
            robust: Continuous modes only — clip the colour scale (and histogram
                x-range) to ``robust_percentiles`` of the report cells instead of
                the raw min/max, so a few outlier cells don't flatten the map.
                Default ``False`` (exact min/max).
            robust_percentiles: ``(low, high)`` percentiles used when
                ``robust=True``.
            highlights: Optional annotation points to scatter over the map, in
                every mode. Accepts an ``Annotation``, an ``Annotations`` /
                ``Cameras`` container (any object with a ``.data`` mapping of
                items exposing ``.coords`` and optionally ``.label``/``.group``),
                an ``(N, 3)`` array, or a list of 3-vectors — same as
                :meth:`OrthoMap.show`.
            color_by: Per-point marker colouring. ``"auto"`` (default) colours
                by label when the highlights carry any labels — matching
                ``Annotations.show(pcd, color=True)`` — otherwise falls back to
                *point_color*. ``"label"`` forces a distinct ``tab20`` colour per
                label; ``"z"`` maps height through ``bwr``; ``None`` forces a
                single *point_color*.
            fill_by_group: Draw markers filled/hollow by the even/odd index of
                their ``group``.
            point_size: Marker radius in display points (ignored when
                *point_size_metres* is set).
            point_size_metres: Marker diameter in metres — drawn to scale in the
                map's coordinates (takes precedence over *point_size*).
            point_color: RGB fill for markers (``None`` for hollow), unless
                *color_by* assigns per-point colours.
            point_outline: RGB marker outline (``None`` to disable).
            point_shape: ``"circle"`` or ``"square"``.
            highlight_label_colors: Optional ``{label: (r, g, b)}`` map for
                *color_by="label"* (distinct from *label_colors*, which colours
                the cells in label mode).

        Returns:
            matplotlib.figure.Figure.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.35])
        ax_left = fig.add_subplot(gs[0, 0])
        ax_left.set_aspect("equal")
        ax_right = fig.add_subplot(gs[0, 1])

        pcd_extent = self._draw_context(ax_left) if show_pcd else None

        if self.value_by == "label":
            self._show_label(ax_left, ax_right, label_colors, plt, mpatches)
        else:
            self._show_continuous(
                ax_left, ax_right, cmap, plt, robust, robust_percentiles
            )

        # View the full point cloud (union of its extent and the grid) when the
        # cloud is shown; otherwise clip to the grid extent.
        ext = self.extent
        if pcd_extent is not None:
            ext = [
                min(ext[0], pcd_extent[0]), max(ext[1], pcd_extent[1]),
                min(ext[2], pcd_extent[2]), max(ext[3], pcd_extent[3]),
            ]
        ax_left.set_xlim(ext[0], ext[1])
        ax_left.set_ylim(ext[2], ext[3])

        if self.report_bbox is not None:
            (rx0, ry0), (rx1, ry1) = self.report_bbox
            ax_left.add_patch(
                mpatches.Rectangle(
                    (rx0, ry0), rx1 - rx0, ry1 - ry0,
                    fill=False, edgecolor="black", linestyle="--",
                    linewidth=1.2, zorder=3,
                )
            )

        if highlights is not None:
            self._draw_highlights(
                ax_left, mpatches, highlights,
                color_by=color_by, fill_by_group=fill_by_group,
                point_size=point_size, point_size_metres=point_size_metres,
                point_color=point_color, point_outline=point_outline,
                point_shape=point_shape, label_colors=highlight_label_colors,
            )

        if title is not None:
            ax_left.set_title(title)

        fig.tight_layout(w_pad=0.5)
        try:
            left_pos = ax_left.get_position()
            right_pos = ax_right.get_position()
            ax_right.set_position(
                [right_pos.x0, left_pos.y0, right_pos.width, left_pos.height]
            )
        except Exception:  # pragma: no cover
            pass
        return fig

    def _draw_highlights(
        self, ax, mpatches, highlights, color_by="auto", fill_by_group=False,
        point_size=5, point_size_metres=None,
        point_color=(255, 0, 0), point_outline=(0, 0, 0),
        point_shape="circle", label_colors=None,
    ) -> None:
        """Scatter annotation points over the map (``ax``), styled like OrthoMap.

        Reuses ``OrthoMap._extract_highlights`` / ``_resolve_marker_style`` for
        coords and per-point colours, projects them into the grid frame via
        ``self.rotation``, and draws with matplotlib. ``point_size_metres`` draws
        markers to scale in map coordinates; otherwise ``point_size`` is a
        display-point radius.
        """
        coords, labels, groups = OrthoMap._extract_highlights(highlights)
        if len(coords) == 0:
            return
        # "auto": colour by label when any labels are present (like
        # Annotations.show(color=True)), else use the fixed point_color.
        if color_by == "auto":
            color_by = "label" if any(lb is not None for lb in labels) else None
        xy = (self.rotation @ coords.T).T[:, :2]

        x0, x1, y0, y1 = self.extent
        oob = (
            (xy[:, 0] < x0) | (xy[:, 0] > x1)
            | (xy[:, 1] < y0) | (xy[:, 1] > y1)
        )
        if oob.any():
            logger.warning(
                "%d highlight(s) outside the grid extent", int(oob.sum())
            )

        fills, outlines = OrthoMap._resolve_marker_style(
            coords, labels, groups, color_by, label_colors, fill_by_group,
            point_color, point_outline,
        )

        def _mpl(col):
            return "none" if col is None else tuple(c / 255.0 for c in col[:3])

        fills = [_mpl(c) for c in fills]
        outlines = [_mpl(c) for c in outlines]

        if point_size_metres is not None:
            r = point_size_metres / 2.0
            for (px, py), fc, ec in zip(xy, fills, outlines):
                if point_shape == "square":
                    patch = mpatches.Rectangle(
                        (px - r, py - r), point_size_metres, point_size_metres,
                        facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=4,
                    )
                else:
                    patch = mpatches.Circle(
                        (px, py), r, facecolor=fc, edgecolor=ec,
                        linewidth=1.0, zorder=4,
                    )
                ax.add_patch(patch)
        else:
            marker = "s" if point_shape == "square" else "o"
            ax.scatter(
                xy[:, 0], xy[:, 1], marker=marker, s=(2.0 * point_size) ** 2,
                facecolors=fills, edgecolors=outlines, linewidths=1.0, zorder=4,
            )

    @staticmethod
    def _scale_limits(scale_src, robust, pct):
        """Colour-scale ``(vmin, vmax)`` for the continuous raster.

        ``robust`` clips to the ``pct`` ``(low, high)`` percentiles so a few
        outlier cells don't dominate the scale; otherwise the raw min/max are
        used. Empty input yields ``(0.0, 1.0)``. Equal limits are nudged apart.
        """
        if scale_src.size == 0:
            return 0.0, 1.0
        if robust:
            lo, hi = np.percentile(scale_src, pct)
            vmin, vmax = float(lo), float(hi)
        else:
            vmin, vmax = float(scale_src.min()), float(scale_src.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
        return vmin, vmax

    def _report_values(self) -> np.ndarray:
        """Finite per-cell values inside the reporting area (continuous mode)."""
        return self.values[self.report_mask & ~np.isnan(self.values)]

    def _continuous_scale(self, cmap=None, robust=False,
                          robust_percentiles=(2.0, 98.0)):
        """Return ``(cmap_obj, norm)`` for continuous modes.

        vmin/vmax come from the reporting-area cells (falling back to all finite
        cells) so the colour scale focuses on the reported region; ``robust``
        clips them to ``robust_percentiles`` via :meth:`_scale_limits`. Shared
        by :meth:`_show_continuous` and the animation module.
        """
        import matplotlib.pyplot as plt

        vals = self.values
        finite = vals[~np.isnan(vals)]
        rep = self._report_values()
        scale_src = rep if rep.size else finite
        vmin, vmax = self._scale_limits(scale_src, robust, robust_percentiles)
        return plt.get_cmap(cmap or "viridis"), plt.Normalize(vmin=vmin, vmax=vmax)

    def _resolve_label_colors(self, label_colors=None):
        """Return ``(label_colors, labels_present)`` for label mode.

        Assigns a distinct ``tab20`` colour per present label unless an explicit
        map is supplied. Shared by :meth:`_show_label` and the animation module
        so the two produce identical colours.
        """
        import matplotlib.pyplot as plt

        labels_present = sorted(
            {lbl for lbl in self.cell_labels.ravel() if lbl is not None}
        )
        if label_colors is None:
            label_colors = self.label_colors
        if label_colors is None:
            cmap = plt.get_cmap("tab20")
            label_colors = {
                lbl: cmap(i % 20) for i, lbl in enumerate(labels_present)
            }
        return label_colors, labels_present

    def _show_continuous(
        self, ax_left, ax_right, cmap, plt, robust=False,
        robust_percentiles=(2.0, 98.0),
    ) -> None:
        """Colormapped raster + value histogram."""
        vals = self.values
        cmap_obj, norm = self._continuous_scale(cmap, robust, robust_percentiles)
        rgba = cmap_obj(norm(vals))
        rgba[np.isnan(vals)] = (0.0, 0.0, 0.0, 0.0)
        ax_left.imshow(rgba, extent=self.extent, origin="lower", zorder=1)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        # Horizontal colorbar beneath the map (left-to-right), so the left plot
        # keeps the full column width — matching the label-mode layout.
        ax_left.figure.colorbar(
            sm, ax=ax_left, orientation="horizontal", location="bottom",
            fraction=0.046, pad=0.10, label=self._value_label(),
        )

        rep = self._report_values()
        if rep.size:
            bins = int(np.clip(np.sqrt(rep.size), 5, 30))
            ax_right.hist(rep, bins=bins, color="0.5", edgecolor="0.3")
        if robust and rep.size:
            # Match the histogram window to the (clipped) colour scale.
            ax_right.set_xlim(norm.vmin, norm.vmax)
        ax_right.set_xlabel(self._value_label())
        ax_right.set_ylabel("Cell count")

    def _show_label(self, ax_left, ax_right, label_colors, plt, mpatches) -> None:
        """Label-filled cells + per-label count bar chart."""
        label_colors, labels_present = self._resolve_label_colors(label_colors)

        no_data = (0.7, 0.7, 0.7)
        rgba = np.zeros((self.ny, self.nx, 4), dtype=float)
        counts_by_cat: dict = {}
        for j in range(self.ny):
            for i in range(self.nx):
                if not self.present[j, i]:
                    continue
                lbl = self.cell_labels[j, i]
                col = label_colors.get(lbl, no_data) if lbl is not None else no_data
                rgba[j, i, :3] = col[:3]
                rgba[j, i, 3] = 0.6
                if self.report_mask[j, i]:
                    cat = lbl if lbl is not None else "No data"
                    counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
        ax_left.imshow(rgba, extent=self.extent, origin="lower", zorder=1)

        handles = [
            mpatches.Patch(facecolor=label_colors.get(lbl, no_data),
                           edgecolor="none", label=str(lbl))
            for lbl in labels_present
        ]
        handles.append(
            mpatches.Patch(facecolor=no_data, edgecolor="none", label="No data")
        )
        ax_left.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=min(len(handles), 6), frameon=False,
        )

        self._label_bar_chart(ax_right, counts_by_cat, label_colors, no_data)

    @staticmethod
    def _label_bar_chart(ax_right, counts_by_cat, label_colors, no_data) -> None:
        """Per-label cell-count bar chart (log-y for wide spreads)."""
        if not counts_by_cat:
            return
        categories = sorted(k for k in counts_by_cat if k != "No data")
        counts = [counts_by_cat[c] for c in categories]
        if "No data" in counts_by_cat:
            categories.append("No data")
            counts.append(counts_by_cat["No data"])
        bar_colors = [
            label_colors.get(c, no_data) if c != "No data" else no_data
            for c in categories
        ]
        x_pos = np.arange(len(categories))
        positive = [c for c in counts if c > 0]
        use_log = bool(positive) and (max(positive) / min(positive) >= 20)
        bars = ax_right.bar(x_pos, counts, color=bar_colors)
        if use_log:
            ax_right.set_yscale("log")
            ax_right.set_ylim(bottom=0.7, top=max(counts) * 1.6)
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(
            [str(c) for c in categories], rotation=90, fontsize=8
        )
        for rect, count in zip(bars, counts):
            ax_right.annotate(
                str(int(count)),
                (rect.get_x() + rect.get_width() / 2, count),
                ha="center", va="bottom", fontsize=6,
                xytext=(0, 1), textcoords="offset points",
            )
        ax_right.set_ylabel("Count")
        ax_right.margins(x=0.05)
