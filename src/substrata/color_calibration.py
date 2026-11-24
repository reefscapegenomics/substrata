# Standard Library
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-Party Libraries
import numpy as np

# Local Modules
from substrata import settings

logger = logging.getLogger(__name__)


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


def correction_residuals(
    measured_rgb_255: np.ndarray,
    reference_rgb_255: np.ndarray,
    correction: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Residuals of an affine colour correction against the reference patches.

    ``corrected = measured @ matrix.T + offset`` (unclipped, sRGB 0-255).

    Args:
        measured_rgb_255: (N, 3) measured patch colours (NaN where missing).
        reference_rgb_255: (N, 3) reference colours.
        correction: Dict with ``"matrix"`` (3x3) and ``"offset"`` (3,).
        mask: (N,) bool, the patches the fit used (default: those with a
            finite measurement).

    Returns:
        dict: ``rmse`` (RMS residual after correction over the masked patches
        and channels; the fit RMSE), ``rmse_all`` (over all finite patches),
        ``rmse_before`` (measured vs reference, masked), the per-channel
        ``rmse_per_channel`` and ``rmse_before_per_channel`` (3-element
        arrays, masked) and ``n_patches_fit``.
    """
    measured = np.asarray(measured_rgb_255, dtype=float)
    ref = np.asarray(reference_rgb_255, dtype=float)
    if mask is None:
        mask = np.isfinite(measured).all(axis=1)
    mask = np.asarray(mask, dtype=bool)
    matrix = np.asarray(correction["matrix"], dtype=float)
    offset = np.asarray(correction["offset"], dtype=float)
    after = measured @ matrix.T + offset - ref
    before = measured - ref
    return {
        "rmse": float(np.sqrt(np.mean(after[mask] ** 2))),
        "rmse_all": float(np.sqrt(np.nanmean(after ** 2))),
        "rmse_before": float(np.sqrt(np.mean(before[mask] ** 2))),
        "rmse_per_channel": np.sqrt(np.mean(after[mask] ** 2, axis=0)),
        "rmse_before_per_channel": np.sqrt(np.mean(before[mask] ** 2, axis=0)),
        "n_patches_fit": int(mask.sum()),
    }


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
        name: Optional[str] = None,
    ) -> None:
        self.tl_label = tl_label
        self.bl_label = bl_label
        self.tr_label = tr_label
        self.br_label = br_label
        self.name = name
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
        min_points: Optional[int] = None,
    ) -> None:
        """Sample the point cloud at each patch center and store results.

        Args:
            pcd: PointCloud with colors.
            patch_definitions: Sequence of (name, chart_u, chart_v, ref_r, ref_g, ref_b).
            radius: Search radius in point cloud units (metres in a project's
                world frame); None uses
                ``settings.DEFAULT_COLOR_CALIBRATION_RADIUS``.
            plane_epsilon: Max distance to the plane through the corners (point
                cloud units); None uses
                ``settings.DEFAULT_COLOR_CALIBRATION_PLANE_EPSILON``, 0 skips
                the filter.
            marker_u_min, marker_v_min, marker_u_max, marker_v_max: Override
                the marker-quad bounds in chart UV (defaults from settings).
            min_points: Minimum point count (after radius + plane filter)
                required for a patch to produce a median. Patches below this
                threshold get ``median_rgb_0_1 = None``. Defaults to
                ``settings.DEFAULT_COLOR_CALIBRATION_MIN_POINTS``.
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
        min_pts = (
            int(min_points)
            if min_points is not None
            else int(settings.DEFAULT_COLOR_CALIBRATION_MIN_POINTS)
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
                # Treat very-low-count patches as "no data" so a couple of
                # spurious points outside the chart cannot produce a measured
                # median.
                if pts.shape[0] >= max(1, min_pts):
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


def _resolved_excluded_card_indices(
    cards: List[ColorCalibration],
    exclude_card_indices: Optional[Sequence[int]],
    exclude_names: Optional[Sequence[str]],
) -> frozenset:
    """Return 0-based card indices excluded from aggregation and QC figures."""
    n = len(cards)
    out: set = set()
    if exclude_card_indices:
        for i in exclude_card_indices:
            if i < 0 or i >= n:
                raise ValueError(
                    f"exclude_card_indices: invalid index {i} (have {n} cards)."
                )
            out.add(int(i))
    if exclude_names:
        for nm in exclude_names:
            hits = [i for i, c in enumerate(cards) if c.name == str(nm)]
            if not hits:
                raise ValueError(
                    f"exclude_names: no card with name {nm!r} "
                    "(names come from the optional 5th column per row)."
                )
            if len(hits) > 1:
                raise ValueError(
                    f"exclude_names: ambiguous name {nm!r} ({len(hits)} matches)."
                )
            out.add(hits[0])
    return frozenset(out)


class ColorCalibrations:
    """Container for several ColorChecker cards; aggregates samples across children."""

    def __init__(
        self,
        calibration_data: List[Sequence[str]],
        target_data: Optional[Union[Dict, Any]] = None,
        pcd: Optional[Any] = None,
        patch_definitions: Optional[Sequence[Tuple[str, float, float, int, int, int]]] = None,
        exclude_card_indices: Optional[Sequence[int]] = None,
        exclude_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Build one ColorCalibration per row of four corner labels.

        Args:
            calibration_data: Rows of ``[tl, bl, tr, br]`` or
                ``[tl, bl, tr, br, name]`` where the optional fifth element
                is a human-readable label for the card.
            target_data: Optional dict label -> coords list, or Annotations instance.
            pcd: Optional PointCloud.  When provided (together with
                ``target_data``), ``sample_point_cloud`` and
                ``compute_color_correction`` are called automatically.
            patch_definitions: Defaults to settings.COLORCHECKER_CLASSIC_PATCHES.
            exclude_card_indices: 0-based indices into ``calibration_data`` /
                ``self.data`` to omit from cross-card medians, outlier aggregation,
                affine colour fit, and ``show`` / ``save_pdf`` card figures.
            exclude_names: Card ``name`` values (5th column per row) to exclude;
                mutually combinable with ``exclude_card_indices``.
        """
        self.data: List[ColorCalibration] = []
        for row in calibration_data:
            if len(row) not in (4, 5):
                raise ValueError(
                    "Each color calibration row must have 4 or 5 elements: "
                    "tl, bl, tr, br[, name]."
                )
            name = str(row[4]) if len(row) == 5 else None
            self.data.append(
                ColorCalibration(
                    str(row[0]), str(row[1]), str(row[2]), str(row[3]),
                    name=name,
                )
            )
        self._excluded_card_indices: frozenset = _resolved_excluded_card_indices(
            self.data, exclude_card_indices, exclude_names,
        )
        self._aggregation_card_indices: Optional[Tuple[int, ...]] = None
        self.patch_definitions: Sequence[Tuple[str, float, float, int, int, int]] = (
            patch_definitions
            if patch_definitions is not None
            else settings.COLORCHECKER_CLASSIC_PATCHES
        )
        self.pcd: Optional[Any] = None
        self.num_cards: Optional[int] = None
        self.median_rgb_255_per_patch: Optional[np.ndarray] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self.patch_fit_mask: Optional[np.ndarray] = None
        self.color_correction: Optional[Dict[str, np.ndarray]] = None
        self.fit_stats: Optional[Dict[str, Any]] = None
        #: :func:`correction_residuals` of the last fit (drawn by ``show``)
        self.fit_residuals: Optional[Dict[str, Any]] = None
        self._last_marker_u_min: Optional[float] = None
        self._last_marker_v_min: Optional[float] = None
        self._last_marker_u_max: Optional[float] = None
        self._last_marker_v_max: Optional[float] = None

        if target_data is not None:
            if hasattr(target_data, "data") and isinstance(target_data.data, dict):
                target_data_dict = {
                    (getattr(ann, "label", None) or key): [ann.coords]
                    for key, ann in target_data.data.items()
                }
                self.store_target_coords(target_data_dict)
            else:
                self.store_target_coords(target_data)

        if pcd is not None:
            self.pcd = pcd
            self.sample_point_cloud(pcd)
            if self.median_rgb_255_per_patch is not None:
                self.compute_color_correction()
            else:
                logger.warning(
                    "Colour calibration skipped affine fit: no valid patch samples "
                    "(check marker labels, exclusions, and point cloud colours)."
                )

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
        min_points: Optional[int] = None,
    ) -> None:
        """Sample all cards and compute cross-card medians and outlier flags.

        Comparison uses sRGB 0-255 for measured vs reference to match reference table.

        A card's value of a patch is flagged as an outlier (``outlier_mask``)
        when it lies more than ``settings.COLOR_CALIBRATION_OUTLIER_Z`` robust
        SDs (1.4826 x MAD across cards) from the cross-card median in any
        channel. The per-patch median used for the fit
        (``median_rgb_255_per_patch``) is then taken over the unflagged cards
        only; a patch whose every card is flagged keeps the median of all cards
        but is left out of the fit (``patch_fit_mask``).

        Args:
            pcd: PointCloud with RGB colors.
            radius: Search radius (point cloud units: metres in a project's
                world frame); None uses the settings default.
            plane_epsilon: Max distance to corner plane; None uses the settings
                default, 0 skips the filter.
            marker_u_min, marker_v_min, marker_u_max, marker_v_max: Override
                the marker-quad bounds in chart UV (defaults from settings).
            min_points: Minimum point count (after radius + plane filter)
                required for a patch to produce a median.  Defaults to
                ``settings.DEFAULT_COLOR_CALIBRATION_MIN_POINTS``.
        """
        self.pcd = pcd
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
        for i, card in enumerate(self.data):
            if i in self._excluded_card_indices:
                card.patch_results = []
                continue
            card.sample_patches(
                pcd,
                self.patch_definitions,
                radius=radius,
                plane_epsilon=plane_epsilon,
                marker_u_min=u_lo,
                marker_v_min=v_lo,
                marker_u_max=u_hi,
                marker_v_max=v_hi,
                min_points=min_points,
            )

        valid_cards = [
            i
            for i, c in enumerate(self.data)
            if i not in self._excluded_card_indices
            and c.has_all_corners()
            and len(c.patch_results) == n_patches
            and all(
                pr["median_rgb_0_1"] is not None for pr in c.patch_results
            )
        ]
        self._aggregation_card_indices = tuple(valid_cards)
        self.num_cards = len(valid_cards)
        if self.num_cards == 0:
            self.median_rgb_255_per_patch = None
            self.patch_fit_mask = None
            self.outlier_mask = None
            return

        n = n_patches
        stack = np.full((len(valid_cards), n, 3), np.nan, dtype=float)
        for row, ci in enumerate(valid_cards):
            for j in range(n):
                m = self.data[ci].patch_results[j]["median_rgb_0_1"]
                if m is not None:
                    stack[row, j, :] = m * 255.0

        all_median = np.nanmedian(stack, axis=0)
        self.median_rgb_255_per_patch = all_median
        self.patch_fit_mask = np.isfinite(all_median).all(axis=1)

        z = float(settings.COLOR_CALIBRATION_OUTLIER_Z)
        out = np.zeros((len(self.data), n), dtype=bool)
        if self.num_cards < 2:
            self.outlier_mask = out
            return

        for j in range(n):
            col = stack[:, j, :]
            med = all_median[j]
            abs_dev = np.abs(col - med.reshape(1, 3))
            patch_mad = np.nanmedian(abs_dev, axis=0)
            scale = 1.4826 * (patch_mad + 1e-6)
            for row, ci in enumerate(valid_cards):
                if np.any(np.isnan(col[row])):
                    continue
                if np.any(abs_dev[row] > z * scale):
                    out[ci, j] = True

        self.outlier_mask = out
        # the per-patch median of the cards not flagged for that patch
        flagged = out[np.asarray(valid_cards, dtype=int)]  # (cards, patches)
        clean = stack.copy()
        clean[flagged] = np.nan
        has_clean = np.isfinite(clean).all(axis=2).any(axis=0)
        with warnings.catch_warnings():  # (all-NaN patches: handled below)
            warnings.simplefilter("ignore", RuntimeWarning)
            clean_median = np.nanmedian(clean, axis=0)
        self.median_rgb_255_per_patch = np.where(
            has_clean[:, None], clean_median, all_median)
        self.patch_fit_mask = has_clean & np.isfinite(all_median).all(axis=1)

    def compute_color_correction(self) -> Dict[str, np.ndarray]:
        """Compute a global affine colour correction from measured vs. reference patches.

        Solves ``corrected = M @ measured + b`` via ordinary least-squares over
        the 24 ColorChecker patches, in sRGB 0-255 (gamma-encoded values, not
        linear light). Each patch's measured value is the median over the
        cards not flagged as outliers for it; patches where *every* card is
        flagged are excluded from the fit (``patch_fit_mask``).

        ``sample_point_cloud()`` must have been called first. The fit's
        statistics are kept in ``fit_stats`` (``rmse``: RMS residual over the
        fitted patches and channels, ``rmse_all``: over all patches, in RGB
        units; ``n_patches_fit``, ``n_patches``, ``space``).

        Returns:
            Dict with ``"matrix"`` (3x3 ndarray) and ``"offset"`` (length-3
            ndarray).  Both operate in 0-255 colour space.
        """
        if self.median_rgb_255_per_patch is None:
            raise RuntimeError(
                "No patch data available. Call sample_point_cloud() first."
            )

        ref = np.array(
            [[r, g, b] for (_name, _u, _v, r, g, b) in self.patch_definitions],
            dtype=float,
        )
        measured = self.median_rgb_255_per_patch.copy()
        n_patches = measured.shape[0]

        if ref.shape[0] != n_patches:
            raise ValueError(
                f"Reference has {ref.shape[0]} patches but measured has {n_patches}."
            )

        mask = np.isfinite(measured).all(axis=1)
        fit_mask = getattr(self, "patch_fit_mask", None)
        if fit_mask is not None and len(fit_mask) == n_patches:
            mask &= np.asarray(fit_mask, dtype=bool)
        if mask.sum() < 4:
            logger.warning(
                "Only %d usable patches: the affine colour fit (12 parameters) "
                "is underdetermined.", int(mask.sum()))

        measured_fit = measured[mask]
        ref_fit = ref[mask]

        A = np.hstack([measured_fit, np.ones((measured_fit.shape[0], 1))])
        params, _res, _rank, _sv = np.linalg.lstsq(A, ref_fit, rcond=None)

        matrix = params[:3, :].T
        offset = params[3, :]

        self.color_correction = {
            "matrix": matrix,
            "offset": offset,
        }
        self.fit_residuals = correction_residuals(
            measured, ref, self.color_correction, mask
        )
        rmse = self.fit_residuals["rmse"]
        rmse_all = self.fit_residuals["rmse_all"]

        logger.info(
            "Colour correction fitted on %d/%d patches  (RMSE = %.2f RGB units)",
            int(mask.sum()), n_patches, rmse,
        )
        self.fit_stats = {
            "rmse": rmse, "rmse_all": rmse_all,
            "n_patches_fit": int(mask.sum()), "n_patches": int(n_patches),
            "space": "sRGB 0-255",
        }
        return self.color_correction

    def __str__(self) -> str:
        lines = ["ColorCalibrations("]
        lines.append(f"  num_charts={len(self.data)},")
        if self.num_cards is not None:
            lines.append(f"  cards_with_full_samples={self.num_cards},")
        lines.append(")")
        return "\n".join(lines)

    def show(
        self,
        pcd: Optional[Any] = None,
        margin_frac: float = 0.15,
        point_size: float = 25,
        plane_thickness: float = 0.01,
        resample: bool = False,
        width: int = 16,
        height: int = 5,
    ) -> List[Any]:
        """Show plane-normal view and measured-vs-reference colour grid per card.

        For each valid card the figure contains two side-by-side panels:

        * **Left** — face-on 2D projection of the point cloud with sampling
          circles (same as the old ``plot_plane_view``).
        * **Right** — 6x4 grid of measured average colours with a thick
          border showing the reference sRGB from the patch definitions.

        Args:
            pcd: PointCloud with colours.  Falls back to ``self.pcd`` when
                omitted.
            margin_frac: Extra margin around corners as fraction of card extent.
            point_size: Scatter marker size for the projected points.
            plane_thickness: Only points within this distance (in metres)
                above or below the card plane are shown.  Defaults to 0.025
                (2.5 cm).
            resample: If True, re-run ``sample_point_cloud`` even if results
                are already cached.  Useful after applying colour correction
                to see updated colours.
            width: Figure width in inches.
            height: Figure height in inches.

        Returns:
            List of matplotlib figures (one per valid card).
        """
        from substrata import visualizations

        if pcd is None:
            pcd = self.pcd
        if pcd is None:
            raise ValueError("No point cloud available. Pass pcd or set it at init.")

        if resample:
            self.num_cards = None
            self.color_correction = None
            for card in self.data:
                card.patch_results = []

        if self.num_cards is None:
            self.sample_point_cloud(pcd)
            if self.median_rgb_255_per_patch is not None:
                self.compute_color_correction()
            else:
                self.color_correction = None

        return visualizations.color_calibration_figures(
            self, pcd, margin_frac=margin_frac, point_size=point_size,
            plane_thickness=plane_thickness, width=width, height=height,
        )

    def save_pdf(
        self, pcd: Optional[Any] = None, filepath: Optional[str] = None,
    ) -> None:
        """Save QC report to PDF using a non-interactive backend."""
        if pcd is None:
            pcd = self.pcd
        if pcd is None:
            raise ValueError("No point cloud available. Pass pcd or set it at init.")

        if self.num_cards is None:
            self.sample_point_cloud(pcd)

        from substrata import visualizations  # lazy: figures only

        visualizations.save_color_calibration_pdf(self, pcd, filepath)
