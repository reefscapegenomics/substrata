"""Orthographic map module for fast 2D representations of point clouds."""

from __future__ import annotations

# Standard Library
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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

    def _resolve_marker_style(
        self,
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
            uniq = sorted({lbl for lbl in labels if lbl is not None})
            cmap = _get_cmap("tab20", max(1, len(uniq)))
            lut = dict(label_colors) if label_colors else {}
            for i, lbl in enumerate(uniq):
                lut.setdefault(lbl, _rgb255(cmap(i)))
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


class OrthoMapGroup(OrthoMap):
    """Composite orthographic map over several point clouds.

    All clouds are projected into a single shared frame (common origin
    and metres-per-pixel) and rasterized together, so overlapping regions
    blend naturally.  The result behaves like an :class:`OrthoMap` — every
    method (``show``, ``project``, highlight overlays) is inherited — with
    :meth:`show` accepting one or more annotation sets to overlay.
    """

    def __init__(
        self,
        pcds,
        up_vector: Optional[Union[List[float], np.ndarray]] = None,
        pixel_width: Optional[int] = None,
        pixel_height: Optional[int] = None,
        rotation: int = 0,
    ) -> None:
        """Initialize a composite ortho map from several point clouds.

        Args:
            pcds: Iterable of point clouds to composite into one frame.
            up_vector: Projection "up" direction (default top-down z).
            pixel_width: Optional target width in pixels.
            pixel_height: Optional target height in pixels.
            rotation: In-plane rotation in degrees applied on ``show``.
        """
        pcds = list(pcds)
        if not pcds:
            raise ValueError("OrthoMapGroup requires at least one point cloud")

        if up_vector is None:
            up_vector = np.array([0.0, 0.0, 1.0])
        up = np.asarray(up_vector, dtype=np.float64)
        up = up / np.linalg.norm(up)
        self._up_vector = up
        self._rotation = rotation
        self.rotation = self._rotation_to_z(up)

        xs_all, ys_all, colors_all = [], [], []
        total = 0
        for pcd in pcds:
            points = np.asarray(pcd.points)
            if len(points) == 0:
                continue
            rotated = (self.rotation @ points.T).T
            xs_all.append(rotated[:, 0])
            ys_all.append(rotated[:, 1])
            colors_all.append(self._get_colors(pcd, len(points)))
            total += len(points)
        if total == 0:
            raise ValueError("OrthoMapGroup point clouds contain no points")

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
        logger.info(
            "OrthoMapGroup created: %d x %d px  (%d clouds, %d points)",
            self.width, self.height, len(pcds), total,
        )

    def __repr__(self) -> str:
        """Return a concise summary of the composite ortho map."""
        return (
            f"OrthoMapGroup({self.width}x{self.height} px, "
            f"res={self.resolution:.6f} m/px, "
            f"up={self._up_vector.tolist()})"
        )

    def show(
        self,
        annotations_list=None,
        color_by: Optional[str] = "label",
        **kwargs,
    ) -> Image.Image:
        """Render the composite map with per-label-coloured annotations.

        Args:
            annotations_list: A single ``Annotations`` container, an
                iterable of containers (merged), or a coordinate array.
                ``None`` renders the composite with no overlay.
            color_by: Highlight colouring mode (default ``"label"``).
            **kwargs: Forwarded to :meth:`OrthoMap.show`.

        Returns:
            PIL ``Image`` of the composite map.
        """
        highlights = self._merge_annotations(annotations_list)
        return super().show(highlights=highlights, color_by=color_by, **kwargs)

    @staticmethod
    def _merge_annotations(annotations_list):
        """Merge one or more annotation containers into a single set."""
        if annotations_list is None:
            return None
        if hasattr(annotations_list, "data") or hasattr(
            annotations_list, "coords"
        ):
            return annotations_list

        containers = list(annotations_list)
        if not containers:
            return None
        if not all(hasattr(a, "data") for a in containers):
            return annotations_list

        class _MergedHighlights:
            pass

        merged = _MergedHighlights()
        merged.data = {}
        idx = 0
        for anns in containers:
            for item in anns.data.values():
                merged.data[idx] = item
                idx += 1
        return merged


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


class OrthoGrid:
    """A metre-cell grid over a point cloud, reduced to a value per cell.

    Bins the cloud (or a set of annotations) into square XY cells of
    ``cell_size`` metres and reduces each cell to a scalar or label:

    - ``value_by="z"``   — height (DEM); ``agg`` in mean/median/max/min.
    - ``value_by="count"`` / ``"density"`` — points per cell / per m².
    - ``value_by="label"`` — majority annotation label per cell.

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
    ) -> None:
        """Build the grid and reduce every cell.

        Args:
            pcd: Point cloud (for ``z``/``count``/``density`` and the context
                background).
            annotations: Annotations (required for ``value_by="label"``).
            value_by: ``"z"`` | ``"count"`` | ``"density"`` | ``"label"``.
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
        """
        if bbox is not None and intercepts is not None:
            raise ValueError("Provide either bbox or intercepts, not both")
        if value_by not in ("z", "count", "density", "label"):
            raise ValueError(f"invalid value_by: {value_by!r}")
        if agg not in ("mean", "median", "max", "min"):
            raise ValueError(f"invalid agg: {agg!r}")

        self.value_by = value_by
        self.agg = agg
        self.pcd = pcd
        self.annotations = annotations
        self.label_colors = label_colors
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
        if pcd is not None and len(np.asarray(pcd.points)):
            proj = (self.rotation @ np.asarray(pcd.points, dtype=float).T).T
            pts_xy, pts_z = proj[:, :2], proj[:, 2]

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

    def _draw_context(self, ax) -> None:
        """Draw a faded grayscale OrthoMap of ``pcd`` behind the grid."""
        if self.pcd is None:
            return
        try:
            om = OrthoMap(self.pcd, up_vector=self._up_vector)
        except Exception:  # pragma: no cover - context is best-effort
            return
        gray = np.asarray(om.image, dtype=float).mean(axis=2) / 255.0
        ext = [om.origin[0], om.origin[0] + om.width * om.resolution,
               om.origin[1], om.origin[1] + om.height * om.resolution]
        ax.imshow(gray, cmap="gray", extent=ext, origin="upper",
                  alpha=0.4, zorder=0, vmin=0.0, vmax=1.0)

    def show(
        self,
        cmap: Optional[str] = None,
        show_context: bool = True,
        title: Optional[str] = None,
        label_colors: Optional[dict] = None,
        figsize: Tuple[int, int] = (12, 5),
    ):
        """Render the grid as a matplotlib Figure with a side panel.

        Left: the colormapped raster (continuous) or label-filled cells; the
        reporting bbox is outlined. Right: a histogram of report-cell values
        (continuous) or a bar chart of per-label cell counts (label mode).

        Args:
            cmap: Matplotlib colormap for continuous modes (default viridis).
            show_context: Draw a faded grayscale ``pcd`` behind the grid.
            title: Optional left-panel title.
            label_colors: Optional ``{label: color}`` map (label mode).
            figsize: Figure size in inches.

        Returns:
            matplotlib.figure.Figure.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_left = fig.add_subplot(gs[0, 0])
        ax_left.set_aspect("equal")
        ax_right = fig.add_subplot(gs[0, 1])

        if show_context:
            self._draw_context(ax_left)

        if self.value_by == "label":
            self._show_label(ax_left, ax_right, label_colors, plt, mpatches)
        else:
            self._show_continuous(ax_left, ax_right, cmap, plt)

        ext = self.extent
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

        if title is not None:
            ax_left.set_title(title)

        plt.tight_layout()
        try:
            left_pos = ax_left.get_position()
            right_pos = ax_right.get_position()
            ax_right.set_position(
                [right_pos.x0, left_pos.y0, right_pos.width, left_pos.height]
            )
        except Exception:  # pragma: no cover
            pass
        return fig

    def _show_continuous(self, ax_left, ax_right, cmap, plt) -> None:
        """Colormapped raster + value histogram."""
        vals = self.values
        finite = vals[~np.isnan(vals)]
        rep = vals[self.report_mask & ~np.isnan(vals)]
        scale_src = rep if rep.size else finite
        if scale_src.size:
            vmin, vmax = float(scale_src.min()), float(scale_src.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

        cmap_obj = plt.get_cmap(cmap or "viridis")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap_obj(norm(vals))
        rgba[np.isnan(vals)] = (0.0, 0.0, 0.0, 0.0)
        ax_left.imshow(rgba, extent=self.extent, origin="lower", zorder=1)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        ax_left.figure.colorbar(
            sm, ax=ax_left, fraction=0.046, pad=0.04, label=self._value_label()
        )

        if rep.size:
            bins = int(np.clip(np.sqrt(rep.size), 5, 30))
            ax_right.hist(rep, bins=bins, color="0.5", edgecolor="0.3")
        ax_right.set_xlabel(self._value_label())
        ax_right.set_ylabel("Cell count")

    def _show_label(self, ax_left, ax_right, label_colors, plt, mpatches) -> None:
        """Label-filled cells + per-label count bar chart."""
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
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
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
