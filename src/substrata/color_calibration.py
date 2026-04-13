# Standard Library
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-Party Libraries
import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Local Modules
from substrata import settings


# ---------------------------------------------------------------------------
# Geometry / colour helpers (pure numpy, no heavy deps)
# ---------------------------------------------------------------------------


def bilinear_point_3d(
    u: float,
    v: float,
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
    br: np.ndarray,
) -> np.ndarray:
    """Interpolate a point on the bilinear patch defined by four 3D corners.

    Corner order: TL (u=0,v=0), TR (u=1,v=0), BL (u=0,v=1), BR (u=1,v=1).

    Args:
        u: Parameter along top edge TL->TR, in [0, 1].
        v: Parameter along left edge TL->BL, in [0, 1].
        tl: Top-left corner (3,).
        tr: Top-right corner (3,).
        bl: Bottom-left corner (3,).
        br: Bottom-right corner (3,).

    Returns:
        Interpolated 3D point (3,).
    """
    tl = np.asarray(tl, dtype=float).reshape(3)
    tr = np.asarray(tr, dtype=float).reshape(3)
    bl = np.asarray(bl, dtype=float).reshape(3)
    br = np.asarray(br, dtype=float).reshape(3)
    return (
        (1.0 - u) * (1.0 - v) * tl
        + u * (1.0 - v) * tr
        + (1.0 - u) * v * bl
        + u * v * br
    )


def chart_uv_to_marker_quad_uv(
    u: float,
    v: float,
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
) -> Tuple[float, float]:
    """Map nominal chart UV to marker-quad UV.

    The marker quad spans chart UV ``[u_min, u_max] x [v_min, v_max]``.
    This function linearly remaps a point in chart space to ``[0, 1] x [0, 1]``
    on the marker quad so it can be fed to bilinear interpolation.

    Args:
        u: Chart u (nominal patch position).
        v: Chart v (nominal patch position).
        u_min: Chart-u of the TL/BL targets (left edge of marker quad).
        v_min: Chart-v of the TL/TR targets (top edge of marker quad).
        u_max: Chart-u of the TR/BR targets (right edge of marker quad).
        v_max: Chart-v of the BL/BR targets (bottom edge of marker quad).

    Returns:
        (u_prime, v_prime) in marker quad [0, 1] x [0, 1].

    Raises:
        ValueError: If the marker quad has zero extent in either axis.
    """
    du = float(u_max) - float(u_min)
    dv = float(v_max) - float(v_min)
    if abs(du) < 1e-12 or abs(dv) < 1e-12:
        raise ValueError(
            "Marker-quad bounds must span a non-zero range in both u and v."
        )
    up = (float(u) - float(u_min)) / du
    vp = (float(v) - float(v_min)) / dv
    return up, vp


def plane_signed_distances(
    points: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Signed distance from each point to a plane (n must be unit length)."""
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-15)
    return (points - plane_point.reshape(1, 3)) @ n


def fit_plane_from_corners(
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unit_normal, point_on_plane) from three corners."""
    tl = np.asarray(tl, dtype=float).reshape(3)
    tr = np.asarray(tr, dtype=float).reshape(3)
    bl = np.asarray(bl, dtype=float).reshape(3)
    e1 = tr - tl
    e2 = bl - tl
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n)
    if ln < 1e-15:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / ln
    return n, tl


def median_rgb_0_1(colors: np.ndarray) -> Optional[np.ndarray]:
    """Median RGB in 0-1 range; returns None if no valid rows."""
    if colors is None or len(colors) == 0:
        return None
    c = np.asarray(colors, dtype=float)
    if c.ndim != 2 or c.shape[1] < 3:
        return None
    rgb = c[:, :3]
    if rgb.size == 0:
        return None
    return np.nanmedian(rgb, axis=0)


# ---------------------------------------------------------------------------
# Domain classes
# ---------------------------------------------------------------------------


class ColorCalibration:
    """One ColorChecker card defined by four 3D corner targets (marker quad)."""

    def __init__(
        self,
        tl_label: str,
        bl_label: str,
        tr_label: str,
        br_label: str,
    ) -> None:
        self.tl_label = tl_label
        self.bl_label = bl_label
        self.tr_label = tr_label
        self.br_label = br_label
        self.tl_coords: Optional[np.ndarray] = None
        self.bl_coords: Optional[np.ndarray] = None
        self.tr_coords: Optional[np.ndarray] = None
        self.br_coords: Optional[np.ndarray] = None
        self.patch_results: List[Dict[str, Any]] = []

    def has_all_corners(self) -> bool:
        """Return True if all four corner coordinates are set."""
        return (
            self.tl_coords is not None
            and self.bl_coords is not None
            and self.tr_coords is not None
            and self.br_coords is not None
        )

    def world_point_from_marker_uv(self, u: float, v: float) -> np.ndarray:
        """3D point on the bilinear patch for marker-quad UV (u, v)."""
        if not self.has_all_corners():
            raise ValueError("All four corner coordinates must be set.")
        return bilinear_point_3d(
            u,
            v,
            self.tl_coords,
            self.tr_coords,
            self.bl_coords,
            self.br_coords,
        )

    def sample_patches(
        self,
        pcd: Any,
        patch_definitions: Sequence[Tuple[str, float, float, int, int, int]],
        radius: Optional[float] = None,
        plane_epsilon: Optional[float] = None,
        marker_u_min: Optional[float] = None,
        marker_v_min: Optional[float] = None,
        marker_u_max: Optional[float] = None,
        marker_v_max: Optional[float] = None,
    ) -> None:
        """Sample the point cloud at each patch center and store results.

        Args:
            pcd: PointCloud with colors.
            patch_definitions: Sequence of (name, chart_u, chart_v, ref_r, ref_g, ref_b).
            radius: Search radius in point cloud units.
            plane_epsilon: Max distance to fitted plane through corners; None skips filter.
            marker_u_min, marker_v_min, marker_u_max, marker_v_max: Override
                the marker-quad bounds in chart UV (defaults from settings).
        """
        self.patch_results = []
        if not self.has_all_corners():
            return

        r = (
            float(radius)
            if radius is not None
            else float(settings.DEFAULT_COLOR_CALIBRATION_RADIUS)
        )
        eps = (
            float(plane_epsilon)
            if plane_epsilon is not None
            else float(settings.DEFAULT_COLOR_CALIBRATION_PLANE_EPSILON)
        )
        u_lo = (
            float(marker_u_min)
            if marker_u_min is not None
            else float(settings.COLORCHECKER_MARKER_U_MIN)
        )
        v_lo = (
            float(marker_v_min)
            if marker_v_min is not None
            else float(settings.COLORCHECKER_MARKER_V_MIN)
        )
        u_hi = (
            float(marker_u_max)
            if marker_u_max is not None
            else float(settings.COLORCHECKER_MARKER_U_MAX)
        )
        v_hi = (
            float(marker_v_max)
            if marker_v_max is not None
            else float(settings.COLORCHECKER_MARKER_V_MAX)
        )

        n_hat, p0 = fit_plane_from_corners(
            self.tl_coords, self.tr_coords, self.bl_coords
        )

        for name, cu, cv, ref_r, ref_g, ref_b in patch_definitions:
            mu, mv = chart_uv_to_marker_quad_uv(cu, cv, u_lo, v_lo, u_hi, v_hi)
            world = self.world_point_from_marker_uv(mu, mv)
            sub = pcd.subsample_pointcloud_by_radius(world, r)
            pts = np.asarray(sub.points)
            cols = getattr(sub, "colors", None)
            if cols is not None:
                cols = np.asarray(cols)
            n_pts = pts.shape[0]
            median_col: Optional[np.ndarray] = None
            if n_pts > 0 and cols is not None and len(cols) == n_pts:
                if eps is not None and eps > 0 and n_pts > 0:
                    dist = np.abs(plane_signed_distances(pts, p0, n_hat))
                    mask = dist <= eps
                    pts = pts[mask]
                    cols = cols[mask]
                median_col = median_rgb_0_1(cols)
            ref = np.array([ref_r, ref_g, ref_b], dtype=float)
            self.patch_results.append(
                {
                    "name": name,
                    "chart_u": float(cu),
                    "chart_v": float(cv),
                    "marker_u": float(mu),
                    "marker_v": float(mv),
                    "world_xyz": world,
                    "n_points": int(pts.shape[0]),
                    "median_rgb_0_1": median_col,
                    "ref_rgb_255": ref,
                }
            )


class ColorCalibrations:
    """Container for several ColorChecker cards; aggregates samples across children."""

    def __init__(
        self,
        calibration_data: List[Sequence[str]],
        target_data: Optional[Union[Dict, Any]] = None,
        patch_definitions: Optional[Sequence[Tuple[str, float, float, int, int, int]]] = None,
    ) -> None:
        """Build one ColorCalibration per row of four corner labels.

        Args:
            calibration_data: Rows of [tl_label, bl_label, tr_label, br_label].
            target_data: Optional dict label -> coords list, or Annotations instance.
            patch_definitions: Defaults to settings.COLORCHECKER_CLASSIC_PATCHES.
        """
        self.data: List[ColorCalibration] = []
        for row in calibration_data:
            if len(row) != 4:
                raise ValueError(
                    "Each color calibration row must have 4 labels: "
                    "tl, bl, tr, br."
                )
            self.data.append(
                ColorCalibration(str(row[0]), str(row[1]), str(row[2]), str(row[3]))
            )
        self.patch_definitions: Sequence[Tuple[str, float, float, int, int, int]] = (
            patch_definitions
            if patch_definitions is not None
            else settings.COLORCHECKER_CLASSIC_PATCHES
        )
        self.num_cards: Optional[int] = None
        self.median_rgb_255_per_patch: Optional[np.ndarray] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self._last_marker_u_min: Optional[float] = None
        self._last_marker_v_min: Optional[float] = None
        self._last_marker_u_max: Optional[float] = None
        self._last_marker_v_max: Optional[float] = None

        if target_data is not None:
            if hasattr(target_data, "data") and isinstance(target_data.data, dict):
                target_data_dict = {
                    ann.label if hasattr(ann, "label") else key: [ann.coords]
                    for key, ann in target_data.data.items()
                }
                self.store_target_coords(target_data_dict)
            else:
                self.store_target_coords(target_data)

    def store_target_coords(self, target_data: Dict) -> None:
        """Assign 3D coordinates to corner labels from a label -> coords mapping."""
        for target_label, target_coords in target_data.items():
            for card in self.data:
                coords = np.asarray(target_coords[0], dtype=float).reshape(3)
                if target_label == card.tl_label:
                    card.tl_coords = coords
                elif target_label == card.bl_label:
                    card.bl_coords = coords
                elif target_label == card.tr_label:
                    card.tr_coords = coords
                elif target_label == card.br_label:
                    card.br_coords = coords

    def sample_point_cloud(
        self,
        pcd: Any,
        radius: Optional[float] = None,
        plane_epsilon: Optional[float] = None,
        marker_u_min: Optional[float] = None,
        marker_v_min: Optional[float] = None,
        marker_u_max: Optional[float] = None,
        marker_v_max: Optional[float] = None,
    ) -> None:
        """Sample all cards and compute cross-card medians and outlier flags.

        Comparison uses sRGB 0-255 for measured vs reference to match reference table.

        Args:
            pcd: PointCloud with RGB colors.
            radius: Search radius (point cloud units).
            plane_epsilon: Max distance to corner plane; None uses settings default.
            marker_u_min, marker_v_min, marker_u_max, marker_v_max: Override
                the marker-quad bounds in chart UV (defaults from settings).
        """
        u_lo = (
            float(marker_u_min)
            if marker_u_min is not None
            else float(settings.COLORCHECKER_MARKER_U_MIN)
        )
        v_lo = (
            float(marker_v_min)
            if marker_v_min is not None
            else float(settings.COLORCHECKER_MARKER_V_MIN)
        )
        u_hi = (
            float(marker_u_max)
            if marker_u_max is not None
            else float(settings.COLORCHECKER_MARKER_U_MAX)
        )
        v_hi = (
            float(marker_v_max)
            if marker_v_max is not None
            else float(settings.COLORCHECKER_MARKER_V_MAX)
        )
        self._last_marker_u_min = u_lo
        self._last_marker_v_min = v_lo
        self._last_marker_u_max = u_hi
        self._last_marker_v_max = v_hi

        n_patches = len(self.patch_definitions)
        for card in self.data:
            card.sample_patches(
                pcd,
                self.patch_definitions,
                radius=radius,
                plane_epsilon=plane_epsilon,
                marker_u_min=u_lo,
                marker_v_min=v_lo,
                marker_u_max=u_hi,
                marker_v_max=v_hi,
            )

        valid_cards = [
            i
            for i, c in enumerate(self.data)
            if c.has_all_corners()
            and len(c.patch_results) == n_patches
            and all(
                pr["median_rgb_0_1"] is not None for pr in c.patch_results
            )
        ]
        self.num_cards = len(valid_cards)
        if self.num_cards == 0:
            self.median_rgb_255_per_patch = None
            self.outlier_mask = None
            return

        n = n_patches
        stack = np.full((len(valid_cards), n, 3), np.nan, dtype=float)
        for row, ci in enumerate(valid_cards):
            for j in range(n):
                m = self.data[ci].patch_results[j]["median_rgb_0_1"]
                if m is not None:
                    stack[row, j, :] = m * 255.0

        self.median_rgb_255_per_patch = np.nanmedian(stack, axis=0)

        z = float(settings.COLOR_CALIBRATION_OUTLIER_Z)
        out = np.zeros((len(self.data), n), dtype=bool)
        if self.num_cards < 2:
            self.outlier_mask = out
            return

        for j in range(n):
            col = stack[:, j, :]
            med = self.median_rgb_255_per_patch[j]
            abs_dev = np.abs(col - med.reshape(1, 3))
            patch_mad = np.nanmedian(abs_dev, axis=0)
            scale = 1.4826 * (patch_mad + 1e-6)
            for row, ci in enumerate(valid_cards):
                if np.any(np.isnan(col[row])):
                    continue
                if np.any(abs_dev[row] > z * scale):
                    out[ci, j] = True

        self.outlier_mask = out

    def __str__(self) -> str:
        lines = ["ColorCalibrations("]
        lines.append(f"  num_charts={len(self.data)},")
        if self.num_cards is not None:
            lines.append(f"  cards_with_full_samples={self.num_cards},")
        lines.append(")")
        return "\n".join(lines)

    def _qc_summary_text(self) -> str:
        """Human-readable summary for the PDF cover page."""
        lines = ["Color calibration QC", ""]
        lines.append(f"Charts defined: {len(self.data)}")
        lines.append(f"Charts with full patch samples: {self.num_cards}")
        if self._last_marker_u_min is not None:
            lines.append(
                f"marker_quad UV: u=[{self._last_marker_u_min:.4f}, "
                f"{self._last_marker_u_max:.4f}], "
                f"v=[{self._last_marker_v_min:.4f}, {self._last_marker_v_max:.4f}]"
            )
        if self.outlier_mask is not None:
            lines.append(
                f"Outlier flags (card x patch): {int(np.sum(self.outlier_mask))}"
            )
        lines.append("")
        lines.append("Measured RGB vs reference (sRGB 8-bit); outliers use MAD across cards.")
        return "\n".join(lines)

    def _generate_qc_figs(self) -> List[Any]:
        """Build matplotlib figures for QC (one grid per card + summary)."""
        from substrata import visualizations

        figs: List[Any] = []
        title_extra = ""
        if self._last_marker_u_min is not None:
            title_extra = (
                f"marker UV: u=[{self._last_marker_u_min:.4f}, "
                f"{self._last_marker_u_max:.4f}], "
                f"v=[{self._last_marker_v_min:.4f}, {self._last_marker_v_max:.4f}]"
            )
        figs.append(visualizations.plot_text(self._qc_summary_text(), width=10, height=6))
        for i, card in enumerate(self.data):
            if not card.patch_results:
                continue
            om = (
                self.outlier_mask[i]
                if self.outlier_mask is not None
                else np.zeros(len(card.patch_results), dtype=bool)
            )
            fig = visualizations.plot_colorchecker_qc_grid(
                card.patch_results,
                title=f"Card {i + 1}: {card.tl_label} … {card.br_label}\n{title_extra}",
                outlier_mask=om,
            )
            figs.append(fig)
        return figs

    def show(self, pcd: Any) -> List[Any]:
        """Run sampling if needed, then display QC figures.

        Args:
            pcd: Point cloud used for sampling (required if not yet sampled).

        Returns:
            List of matplotlib figures.
        """
        if self.num_cards is None:
            self.sample_point_cloud(pcd)
        figs = self._generate_qc_figs()
        for fig in figs:
            fig.show()
        return figs

    def plot_plane_view(
        self,
        pcd: Any,
        margin_frac: float = 0.15,
        point_size: float = 0.3,
        width: int = 10,
        height: int = 10,
    ) -> List[Any]:
        """Render a face-on 2D view of each card aligned to the card edges.

        The x-axis aligns with the top edge (TL -> TR) and the y-axis with
        the left edge (TL -> BL). The view is clipped to the card corners
        plus a small margin so only the immediate neighbourhood is shown.

        No decimation is applied -- every point within the fetch radius is
        plotted.

        Args:
            pcd: PointCloud with colours.
            margin_frac: Extra margin around corners as fraction of card extent.
            point_size: Scatter marker size for the projected points.
            width: Figure width in inches.
            height: Figure height in inches.

        Returns:
            List of matplotlib figures (one per valid card).
        """
        from substrata import visualizations

        if self.num_cards is None:
            self.sample_point_cloud(pcd)

        sample_r = float(settings.DEFAULT_COLOR_CALIBRATION_RADIUS)

        figs: List[Any] = []
        for i, card in enumerate(self.data):
            if not card.has_all_corners() or not card.patch_results:
                continue

            corners = np.array([
                card.tl_coords, card.tr_coords,
                card.bl_coords, card.br_coords,
            ])
            centre = corners.mean(axis=0)
            diag = np.max(np.linalg.norm(corners - centre, axis=1))
            fetch_radius = diag * (1.0 + margin_frac)

            sub = pcd.subsample_pointcloud_by_radius(centre, fetch_radius)
            pts = np.asarray(sub.points)
            cols = np.asarray(getattr(sub, "colors", np.ones_like(pts)))
            if pts.shape[0] == 0:
                continue

            patch_centers = np.array([
                pr["world_xyz"] for pr in card.patch_results
            ])
            patch_names = [pr["name"] for pr in card.patch_results]

            fig = visualizations.plot_colorchecker_plane_view(
                pcd_points=pts,
                pcd_colors=cols,
                tl=card.tl_coords,
                tr=card.tr_coords,
                bl=card.bl_coords,
                br=card.br_coords,
                patch_centers_3d=patch_centers,
                patch_names=patch_names,
                radius=sample_r,
                title=(
                    f"Card {i + 1}: {card.tl_label} … {card.br_label}  "
                    f"(plane-normal view)"
                ),
                point_size=point_size,
                margin_frac=margin_frac,
                width=width,
                height=height,
            )
            figs.append(fig)
        return figs

    def save_pdf(self, pcd: Any, filepath: Optional[str] = None) -> None:
        """Save QC report to PDF using a non-interactive backend."""
        if self.num_cards is None:
            self.sample_point_cloud(pcd)

        backend_original = matplotlib.get_backend()
        matplotlib.use("Agg", force=True)
        try:
            if filepath is None:
                base, _ext = os.path.splitext(pcd.filepath)
                filepath = f"{base}_color_calibration.pdf"

            pdf = PdfPages(filepath)
            for fig in self._generate_qc_figs():
                pdf.savefig(fig)
            for fig in self.plot_plane_view(pcd):
                pdf.savefig(fig)
            pdf.close()
        finally:
            matplotlib.use(backend_original, force=True)
