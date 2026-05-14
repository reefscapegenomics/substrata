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

        self.image: np.ndarray = self._rasterize(
            xs, ys, pcd,
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
    ) -> Image.Image:
        """Return the ortho map as a PIL Image with optional highlights.

        The base image is generated from the pre-rendered raster.  If
        *width* and/or *height* are given the image is resized (aspect
        ratio preserved) before highlights are drawn.

        Args:
            highlights: Locations to highlight.  Accepts an
                ``Annotation``, ``Annotations`` container, a numpy
                array of shape ``(N, 3)``, or a list of 3-element
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
                Pass ``None`` for a transparent (hollow) fill.
            point_outline: RGB outline colour for highlight markers
                (``None`` to disable the outline).
            point_shape: Marker shape — ``"circle"`` or ``"square"``.
            background_color: RGB colour for empty (no-data) pixels.
                Defaults to ``None`` which keeps the original white
                background.

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

        if width is not None or height is not None:
            img = self._resize(img, width, height)

        if highlights is not None:
            coords_3d = self._extract_coords(highlights)
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
                    out_coords = coords_3d[np.newaxis, :] if coords_3d.ndim == 1 else coords_3d
                    logger.warning(
                        "%d highlight(s) outside map bounds:\n%s",
                        int(out_mask.sum()),
                        "\n".join(
                            f"  [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]"
                            for c in out_coords[out_mask]
                        ),
                    )
                self._draw_highlights(
                    img, coords_3d,
                    point_size, point_color, point_outline,
                    point_shape, point_size_metres,
                )

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
    def _rasterize(
        xs: np.ndarray,
        ys: np.ndarray,
        pcd: Union[PointCloud, SimplePointCloud],
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
            pcd: Source point cloud (for colours).
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

        if hasattr(pcd, "colors"):
            colors = np.asarray(pcd.colors)
            if colors.ndim != 2 or colors.shape[0] != n:
                colors = np.ones((n, 3), dtype=np.float64)
        else:
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
        color: Optional[Tuple[int, int, int]],
        outline: Optional[Tuple[int, int, int]],
        shape: str = "circle",
        radius_metres: Optional[float] = None,
    ) -> None:
        """Draw highlight markers onto *img* (modified in place).

        Args:
            img: PIL Image to draw onto.
            coords_3d: World coordinates ``(N, 3)``.
            radius: Marker radius in display pixels.  Ignored when
                *radius_metres* is set.
            color: Fill colour, or ``None`` for transparent fill.
            outline: Outline colour or ``None``.
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
        draw = ImageDraw.Draw(img)
        bbox_fn = draw.ellipse if shape == "circle" else draw.rectangle
        for px, py in pixels:
            dx = px * scale_x
            dy = py * scale_y
            bbox_fn(
                [dx - radius, dy - radius, dx + radius, dy + radius],
                fill=color,
                outline=outline,
            )

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
    def _extract_coords(
        highlights: Union[
            Annotation,
            Annotations,
            List[np.ndarray],
            np.ndarray,
        ],
    ) -> np.ndarray:
        """Normalise *highlights* into an ``(N, 3)`` coordinate array.

        Args:
            highlights: One of the accepted highlight types.

        Returns:
            Numpy array of 3D coordinates with shape ``(N, 3)``.
        """
        from substrata.annotations import Annotation, Annotations

        if isinstance(highlights, Annotations):
            return np.array(
                [a.coords for a in highlights.data.values()]
            )
        if isinstance(highlights, Annotation):
            return highlights.coords[np.newaxis, :]

        arr = np.asarray(highlights, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr
