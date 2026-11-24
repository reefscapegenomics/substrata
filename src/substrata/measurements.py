"""Terrain and ecology metrics computed from point clouds.

Plane fits and orientation (:func:`get_best_fit_plane_pca`,
:func:`get_plane_angles`), roughness (:func:`calc_roughness`), topographic
position / ruggedness (:func:`calc_tpi_and_tri`), vector dispersion, colour
statistics, hemispherical gap fraction (:func:`calc_gap_fraction`), benthic
fraction, the depth regression (:func:`fit_depth_regression`) and the sampling
grids used by ``substrata intercepts``.

Conventions: coordinates are in the world frame (metres after scaling, +Z up,
depth negative below the surface). Fitted plane normals are oriented by the
point normals when the cloud has them, otherwise towards +Z.
"""

# Standard Library
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np

# open3d, cv2, scipy and sklearn are imported in the functions that use them, so
# `import substrata.measurements` (e.g. substrata-explore's conduct_pca) stays
# light; figures are drawn by substrata.visualizations, imported lazily too.

# Local Modules
from substrata import settings
from substrata.compat import renamed_kwargs as _renamed_kwargs
from substrata.logging import logger

__all__ = [
    # plane fits, PCA and orientation
    "conduct_pca",
    "conduct_xy_pca",
    "get_best_fit_plane_pca",
    "get_best_fit_plane_ransac",
    "get_plane_angles",
    "plane_orientation",
    "distances_to_plane",
    "slope_normal_from_transform",
    "get_elevation_angle",
    # roughness, TPI/TRI, dispersion, colour
    "get_dev_rugosity",
    "calc_roughness",
    "annulus_coverage",
    "calc_tpi_and_tri",
    "annulus_plane",
    "tpi_point_values",
    "get_vector_dispersion",
    "vector_dispersion_from_normals",
    "normal_deviation_deg",
    "get_rgb_stats",
    # benthic fraction
    "calc_benthic_fraction",
    "benthic_fraction_from_results",
    # gap fraction
    "hemispherical_disk",
    "hemispherical_pixels",
    "hemispherical_cover",
    "gap_fractions",
    "calc_gap_fraction",
    # sampling grids (substrata intercepts)
    "cloud_xy_histogram",
    "find_optimal_box_position",
    "subdivide_boxes",
    "generate_random_xy_points_within_cells",
    "depth_band_sample",
    # surface area (photo mask)
    "get_mask_surface_area",
    # depth regression
    "DepthRegressionResult",
    "normalize_depth_sign",
    "fit_depth_regression",
    "regression_errors",
    "DepthResidualAnalyzer",
    # alignment QC
    "PcdCorrespondence",
    "pcd_correspondence",
    "camera_residuals",
    # deprecated aliases (renamed 2026-11-25)
    "conduct_PCA",
    "conduct_xy_PCA",
    "get_best_fit_plane_PCA",
]

# TPI/TRI annulus coverage threshold (see settings.TPI_MIN_ANNULUS_COVERAGE)
TPI_MIN_ANNULUS_COVERAGE = settings.TPI_MIN_ANNULUS_COVERAGE
TPI_RASTER_CELL_SIZE = settings.TPI_RASTER_CELL_SIZE
TPI_MIN_RASTER_FILL = 0.5  # warn below this fraction of filled raster cells


def _points_array(pcd) -> np.ndarray:
    """The (N, 3) float array of ``pcd.points`` (numpy, open3d or SimplePointCloud)."""
    return np.asarray(pcd.points, dtype=float)


def _normals_array(pcd, n_points: int) -> Optional[np.ndarray]:
    """``pcd.normals`` as an (N, 3) array, or None when missing / not one per point."""
    normals = getattr(pcd, "normals", None)
    if normals is None:
        return None
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape != (n_points, 3):
        return None
    return normals


def conduct_pca(pcd, sort=True):
    """Principal component analysis of a point cloud's XYZ coordinates.

    Eigen-decomposition of the 3 x 3 covariance matrix of ``pcd.points`` with
    ``numpy.linalg.eigh`` (the covariance is symmetric, so eigenvalues and
    eigenvectors are real and the eigenvectors orthonormal).

    Args:
        pcd: Object with ``points`` convertible to an (N, 3) array.
        sort: Sort by decreasing eigenvalue (default). With False the order is
            ``eigh``'s (increasing).

    Returns:
        tuple: ``(eigenvalues, eigenvectors)``; ``eigenvectors[:, i]`` (a
        column) is the axis of ``eigenvalues[i]``. Sorted, column 0 is the
        principal axis and column 2 the normal of the best-fit plane. The sign of
        each eigenvector is arbitrary.
    """
    matrix = np.cov(_points_array(pcd).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    if sort:
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


conduct_PCA = conduct_pca  # deprecated alias (renamed 2026-11-25)


def conduct_xy_pca(pcd, sort=True, visualize=False):
    """Principal component analysis of the XY coordinates only.

    Like :func:`conduct_pca` on the 2 x 2 covariance of X and Y
    (``numpy.linalg.eigh``).

    Args:
        pcd: Object with ``points`` convertible to an (N, 3) array.
        sort: Sort by decreasing eigenvalue (default).
        visualize: Plot the points with the principal axes.

    Returns:
        tuple: ``(eigenvalues, eigenvectors)``; ``eigenvectors[:, i]`` is the
        axis of ``eigenvalues[i]`` (sign arbitrary).
    """
    xy_points = _points_array(pcd)[:, :2]
    mean_xy = np.mean(xy_points, axis=0)
    centered_xy = xy_points - mean_xy  # shape (N, 2)

    cov = np.cov(centered_xy, rowvar=False)  # shape (2, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if sort:
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

    if visualize:
        from substrata import visualizations  # lazy: figures only

        visualizations.plot_xy_pca(
            points=xy_points, mean=mean_xy, eig_vecs=eigenvectors, eig_vals=eigenvalues
        )

    return eigenvalues, eigenvectors


conduct_xy_PCA = conduct_xy_pca  # deprecated alias (renamed 2026-11-25)


def _orient_plane(a, b, c, d, pcd, points, inliers_idx, align_normals):
    """Orient plane coefficients: along the point normals if present, else +Z.

    With ``align_normals`` and one normal per point, the plane normal is flipped
    when it points against the mean normal of the inliers (all points if there
    are no inliers). Without normals (or with ``align_normals=False``) it is
    flipped to ``c >= 0`` (facing +Z), so the result never depends on the
    arbitrary sign of the eigenvector / RANSAC model.
    """
    normals = _normals_array(pcd, len(points)) if align_normals else None
    if normals is not None and len(normals):
        sel = normals[inliers_idx] if len(inliers_idx) else normals
        if np.dot(sel.mean(axis=0), [a, b, c]) < 0:
            return -a, -b, -c, -d
        return a, b, c, d
    if c < 0:
        return -a, -b, -c, -d
    return a, b, c, d


def get_best_fit_plane_pca(
    pcd, inlier_range=settings.DEFAULT_INLIER_RANGE, print_eq=False, align_normals=True
):
    """Least-squares (orthogonal) best-fit plane from PCA.

    The plane passes through the centroid with the smallest-variance principal
    axis as its normal (total least squares). Points within ``inlier_range`` of
    the plane are reported as inliers (they do not change the fit).

    The normal's sign is set by :func:`_orient_plane`: along the mean point
    normal of the inliers when ``align_normals`` and the cloud has one normal
    per point, otherwise towards +Z (``c >= 0``).

    Args:
        pcd: Object with ``points`` (N, 3) and optionally ``normals`` (N, 3).
        inlier_range: Distance from the plane for a point to count as an inlier
            (point-cloud units).
        print_eq: Print the plane equation.
        align_normals: Orient the plane by the point normals when available.

    Returns:
        tuple: ``(a, b, c, d, inliers_idx)`` with ``a x + b y + c z + d = 0``
        and ``(a, b, c)`` a unit normal.

    author: DvH
    """
    points = _points_array(pcd)
    _, eigenvectors = conduct_pca(pcd)
    normal = eigenvectors[:, 2]
    d = -(np.dot(normal, points.mean(axis=0)))

    dist = points @ normal + d
    inliers_idx = np.flatnonzero(np.abs(dist) <= inlier_range)

    a, b, c = (float(v) for v in normal)
    a, b, c, d = _orient_plane(a, b, c, float(d), pcd, points, inliers_idx,
                               align_normals)

    if print_eq:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    return a, b, c, d, inliers_idx


get_best_fit_plane_PCA = get_best_fit_plane_pca  # deprecated alias (renamed 2026-11-25)


def get_best_fit_plane_ransac(
    pcd,
    inlier_range: float = settings.DEFAULT_INLIER_RANGE,
    print_eq: bool = False,
    align_normals: bool = True,
) -> tuple[float, float, float, float, np.ndarray]:
    """Fit a plane to a point cloud with RANSAC (open3d ``segment_plane``).

    The resulting plane follows ``a*x + b*y + c*z + d = 0``. RANSAC is
    stochastic: repeated calls can give slightly different planes. The normal's
    sign is set as in :func:`get_best_fit_plane_pca` (point normals when
    available and ``align_normals``, else +Z).

    Args:
        pcd: A ``PointCloud`` (its ``o3d_pcd`` is used), an open3d point cloud,
            or any object with ``points`` (and optionally ``normals``).
        inlier_range (float): Distance threshold for including a point as an inlier.
        print_eq (bool): If True, prints the plane equation.
        align_normals (bool): If True, adjusts the plane normal so it aligns with
            the general direction of the point cloud normals.

    Returns:
        tuple: (a, b, c, d, inliers_idx), where (a, b, c, d) are the plane
        coefficients and inliers_idx is a NumPy array of indices for points
        within the inlier range.
    """
    import open3d as o3d

    geometry = o3d.geometry
    if isinstance(pcd, geometry.PointCloud):
        o3d_pcd = pcd
    elif isinstance(getattr(pcd, "o3d_pcd", None), geometry.PointCloud):
        o3d_pcd = pcd.o3d_pcd
    else:
        o3d_pcd = geometry.PointCloud(o3d.utility.Vector3dVector(_points_array(pcd)))

    # Use RANSAC to segment the plane.
    plane_model, inliers_idx = o3d_pcd.segment_plane(
        distance_threshold=inlier_range,
        ransac_n=settings.RANSAC_N,
        num_iterations=settings.RANSAC_ITERATIONS,
    )
    a, b, c, d = (float(v) for v in plane_model)
    inliers_idx = np.asarray(inliers_idx, dtype=int)
    a, b, c, d = _orient_plane(a, b, c, d, pcd, _points_array(pcd), inliers_idx,
                               align_normals)

    if print_eq:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    return a, b, c, d, inliers_idx


def distances_to_plane(points, plane) -> np.ndarray:
    """Signed perpendicular distances of points to a plane.

    The plane is ``a x + b y + c z + d = 0``.

    Positive on the side the normal ``(a, b, c)`` points to.

    Args:
        points: (N, 3) array (or an object with ``points``).
        plane: ``(a, b, c, d)``; the normal need not be unit length.

    Raises:
        ValueError: For a zero-length normal.
    """
    pts = np.asarray(getattr(points, "points", points), dtype=float)
    a, b, c, d = (float(v) for v in plane[:4])
    normal = np.array([a, b, c], dtype=float)
    denom = np.linalg.norm(normal)
    if denom == 0.0:
        raise ValueError("plane has a zero-length normal")
    return (pts @ normal + d) / denom


def plane_orientation(normal) -> Tuple[float, float, float, Optional[float]]:
    """Orientation angles (degrees) of a plane normal (see :func:`get_plane_angles`).

    Args:
        normal: Length-3 normal ``(a, b, c)`` (normalised here).

    Returns:
        tuple: ``(theta_deg, psi_deg, elev_deg, az_deg)``: ``theta = atan2(n_x,
        n_z)``, ``psi = atan2(-n_y, hypot(n_x, n_z))`` (``n = R_y(theta) @
        R_x(psi) @ z``), the elevation (angle to +Z, as
        :func:`get_elevation_angle`) and the azimuth of the normal projected on
        XY from +X counter-clockwise (``None`` for a nearly vertical normal).

    Raises:
        ValueError: For a zero-length normal.
    """
    n = np.asarray(normal, dtype=float)[:3]
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("Degenerate plane normal.")
    nx, ny, nz = n / n_norm

    elev_deg = float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))
    theta_deg = float(np.degrees(np.arctan2(nx, nz)))
    psi_deg = float(np.degrees(np.arctan2(-ny, np.hypot(nx, nz))))
    if np.hypot(nx, ny) < 1e-8:
        az_deg = None
    else:
        az_deg = float((np.degrees(np.arctan2(ny, nx)) + 360.0) % 360.0)
    return theta_deg, psi_deg, elev_deg, az_deg


def get_plane_angles(pcd, vis=False, generate_image=True):
    """Calculate orientation angles for the best-fit plane of a point cloud.

    The plane is :func:`get_best_fit_plane_pca`'s; its normal follows the
    point normals when the cloud has them (so it encodes whether the surface
    faces upward or downward), otherwise it faces +Z (elevation 0-90 degrees:
    an overhang cannot be told from the plane alone).

    Args:
        pcd: Point cloud object with ``.points`` (and optionally ``.normals``).
        vis: If True, show an interactive 3-D visualisation of the elevation.
        generate_image: If True (default), also render a static elevation-angle
            QC image (via plotly/Kaleido). Set False to skip that expensive
            render and return ``None`` for the image (e.g. batch runs).

    Returns:
        tuple: A 6-element tuple containing:
            - **theta_deg** (*float*): ``atan2(n_x, n_z)``, the tilt of the
              normal from +Z towards +X in the XZ-plane.
            - **psi_deg** (*float*): ``atan2(-n_y, hypot(n_x, n_z))``.
              Together ``n = R_y(theta) @ R_x(psi) @ [0, 0, 1]`` (right-handed
              rotations about +Y and +X): rotating +Z about +X by ``psi`` and
              then about +Y by ``theta`` gives the normal, and the inverse,
              ``R_x(-psi) @ R_y(-theta)``, brings the normal to +Z.
            - **elev_deg** (*float*): Angle between the plane normal and +Z
              (0 = up, 90 = vertical, 180 = down).
            - **plane** (*list[float]*): Plane coefficients ``[a, b, c, d]``.
            - **az_deg** (*float | None*): Azimuth of the normal projected onto
              XY, measured from +X counter-clockwise (``None`` if the normal is
              nearly vertical).
            - **image** (*np.ndarray*): Static elevation-angle visualisation as
              an (H, W, 3) uint8 RGB array.
    """
    if len(pcd.points) <= 1:
        return 0.0, 0.0, 0.0, [], None, None

    a, b, c, d, _ = get_best_fit_plane_pca(pcd)
    theta_deg, psi_deg, elev_deg, az_deg = plane_orientation([a, b, c])
    plane = [float(a), float(b), float(c), float(d)]

    image = None
    if vis or generate_image:
        from substrata import visualizations  # lazy: figures only

        if vis:
            visualizations.visualize_elevation_angle(
                pcd, plane, interactive=True, elev_deg=elev_deg
            )
        if generate_image:
            image = visualizations.visualize_elevation_angle(
                pcd, plane, interactive=False, elev_deg=elev_deg
            )

    return theta_deg, psi_deg, elev_deg, plane, az_deg, image


def slope_normal_from_transform(transform) -> np.ndarray:
    """Return the slope-plane normal in the source frame of a slope transform.

    A slope-aligning transform (e.g. ``PointCloud.apply_along_slope_transform``,
    or a sidecar's ``grid_transform``) maps the plane normal to +Z; its final
    x-alignment only rotates about Z. For a similarity transform the normal in
    the source frame is therefore the normalized third row of the linear part.
    Pass ``grid_transform @ world_transform`` to get the normal in orig coords.

    Args:
        transform: 4x4 (or 3x3) transform from the source frame to the grid frame.

    Returns:
        Unit normal (length 3) in the source frame.
    """
    n = np.asarray(transform, dtype=float)[2, :3]
    return n / np.linalg.norm(n)


def get_elevation_angle(normal) -> float:
    """
    Return the elevation angle (degrees) of a plane given its normal vector.

    Elevation is the angle between the normal and +Z:
    0° for upward-facing, 90° for vertical, 180° for downward-facing.

    The normal orientation matters: the caller is responsible for ensuring it
    points in the correct direction (e.g. aligned with point cloud normals).

    The input can be any 3-element array-like; it is normalized internally.

    Raises:
        ValueError: If the provided normal is not a length-3 non-zero vector.
    """
    n = np.asarray(normal, dtype=float)
    if n.ndim != 1 or n.size != 3:
        raise ValueError("normal must be a length-3 vector")

    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("normal vector must be non-zero")

    n = n / norm
    elev_rad = np.arccos(np.clip(n[2], -1.0, 1.0))
    return float(np.degrees(elev_rad))


def get_dev_rugosity(pcd):
    """Deviation rugosity: mean absolute distance of the points to their best-fit plane.

    Also called "plane-detrended roughness"; identical to ``Ra`` of
    :func:`calc_roughness` (same plane, same formula), kept for the ``dev_rug``
    measurement key.

    author: DvH
    """
    [a, b, c, d] = get_best_fit_plane_pca(pcd)[0:4]
    dist = abs(
        (np.dot(np.asarray(pcd.points), [a, b, c]) + d)
        / np.sqrt(np.sum(np.square([a, b, c])))
    )
    dev_rugosity = np.sum(dist) / len(pcd.points)
    return dev_rugosity


def calc_roughness(pcd, generate_image=True):
    """
    Compute plane-detrended roughness (Ra, Rq) for a point cloud.

    This function fits a best-fitting plane to the point cloud and then
    measures how much points deviate from that plane, using distances
    perpendicular to the plane.

    Definitions
    -----------
    Ra : arithmetical mean roughness
        The mean of the absolute perpendicular distances from all points
        to the best-fitting plane (``mean(|d_i|)``). This is the classic
        "average height" roughness measure. Same as get_dev_rugosity.

    Rq : root mean square roughness
        The square root of the mean of squared perpendicular distances
        (``sqrt(mean(d_i^2))``). This is more sensitive to larger
        deviations than Ra and is the standard RMS roughness metric used
        in surface metrology.

    Both Ra and Rq are returned in the same absolute units as the point cloud
    coordinates (e.g. meters).

    Parameters
    ----------
    pcd :
        Point cloud object with a .points attribute that can be
        converted to an (N, 3) NumPy array. For example, an Open3D
        PointCloud.

    Returns
    -------
    ra : float
        Arithmetical mean roughness (Ra).
    rq : float
        Root mean square roughness (Rq).
    """
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        raise ValueError("Point cloud has no points")

    # Plane: a*x + b*y + c*z + d = 0
    plane = get_best_fit_plane_pca(pcd)[:4]
    # Perpendicular distances of all points to the plane
    dist = np.abs(distances_to_plane(pts, plane))

    ra = float(dist.mean())
    rq = float(np.sqrt((dist**2).mean()))

    # The QC image is an expensive plotly/Kaleido render; skip it unless asked
    # for (e.g. batch runs via measure_all default to metrics only).
    image = None
    if generate_image:
        from substrata import visualizations  # lazy: figures only

        image = visualizations.visualize_roughness(
            pcd, interactive=False, ra=ra, rq=rq, plane=plane
        )

    return ra, rq, image


def _resolve_outer_radius(
    radius_inner: float, radius_outer: Optional[float],
    annulus_width: Optional[float], default_outer: float,
) -> float:
    """Resolve the annulus outer radius from the mutually-exclusive options.

    Shared by the annulus-based measurements (:func:`calc_tpi_and_tri`,
    :func:`calc_benthic_fraction`). The outer limit is ``radius_inner +
    annulus_width`` when ``annulus_width`` is given, else ``radius_outer``, else
    ``default_outer``.

    Args:
        radius_inner: Inner radius of the annulus in metres.
        radius_outer: Absolute outer radius (mutually exclusive with
            ``annulus_width``).
        annulus_width: Fixed extension beyond ``radius_inner`` (mutually
            exclusive with ``radius_outer``).
        default_outer: Fallback outer radius when neither is supplied.

    Returns:
        The resolved outer radius in metres.

    Raises:
        ValueError: If both ``radius_outer`` and ``annulus_width`` are given.
    """
    if radius_outer is not None and annulus_width is not None:
        raise ValueError("Specify either radius_outer or annulus_width, not both")
    if annulus_width is not None:
        return radius_inner + annulus_width
    return radius_outer if radius_outer is not None else default_outer


def annulus_coverage(
    xy: np.ndarray,
    center_xy: np.ndarray,
    radius_inner: float,
    radius_outer: float,
    n_sectors: int = 24,
    n_rings: int = 4,
) -> float:
    """Fraction of an annulus's area that contains points (0-1).

    The annulus ``radius_inner <= r <= radius_outer`` around ``center_xy`` is
    split into ``n_sectors x n_rings`` polar cells of equal area (ring
    boundaries at ``sqrt(r_in^2 + k / n_rings * (r_out^2 - r_in^2))``); the
    coverage is the fraction of cells holding at least one point. A focal point
    on a straight cloud edge gives about 0.5; inside a cloud with no gaps
    larger than a cell, 1.0. Clouds so sparse that cells are empty by chance
    (fewer than a few points per cell) also read low.

    Args:
        xy: (N, 2) XY coordinates (only those inside the annulus count).
        center_xy: (2,) annulus centre.
        radius_inner: Inner radius (same units as ``xy``).
        radius_outer: Outer radius.
        n_sectors: Number of angular sectors.
        n_rings: Number of equal-area rings.

    Returns:
        The covered fraction of the annulus area.
    """
    d = np.asarray(xy, dtype=float)[:, :2] - np.asarray(center_xy, dtype=float)[:2]
    r2 = np.einsum("ij,ij->i", d, d)
    r_in2, r_out2 = float(radius_inner) ** 2, float(radius_outer) ** 2
    inside = (r2 >= r_in2) & (r2 <= r_out2)
    if not np.any(inside) or r_out2 <= r_in2:
        return 0.0
    ring = np.floor((r2[inside] - r_in2) / (r_out2 - r_in2) * n_rings).astype(int)
    ring = np.clip(ring, 0, n_rings - 1)
    ang = np.arctan2(d[inside, 1], d[inside, 0])
    sector = np.floor((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int)
    sector = np.clip(sector, 0, n_sectors - 1)
    occupied = np.unique(ring * n_sectors + sector)
    return float(len(occupied) / (n_sectors * n_rings))


def annulus_plane(annulus_pts) -> Optional[Tuple[np.ndarray, float]]:
    """Best-fit plane of annulus points, its normal oriented to +Z.

    Total least squares via SVD (the normal is the last right-singular vector of
    the centred points). Used by :func:`calc_tpi_and_tri`.

    Args:
        annulus_pts: (N, 3) points.

    Returns:
        ``(normal, d)`` with ``normal @ p + d = 0`` on the plane (``normal``
        unit length, ``normal[2] >= 0`` so that distances are positive above
        the plane), or None for fewer than 3 points.
    """
    annulus_pts = np.asarray(annulus_pts, dtype=float)
    if len(annulus_pts) < 3:
        return None
    centroid = annulus_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(annulus_pts - centroid, full_matrices=False)
    normal = vt[-1]
    # Orient the normal to +Z so the signed tpi_plane is physically
    # meaningful: positive = crest/elevated, negative = hollow/depressed.
    # SVD's sign is otherwise arbitrary and not reproducible across inputs.
    if normal[2] < 0:
        normal = -normal
    return normal, float(-normal @ centroid)


def tpi_point_values(points, focal_pt, plane=None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point TPI values relative to a focal point (the TPI figure's colours).

    Args:
        points: (N, 3) points.
        focal_pt: (3,) focal point (its Z is the focal height of ``TPI_abs``).
        plane: The annulus plane ``(normal, d)`` of :func:`annulus_plane`, or
            None.

    Returns:
        tuple: ``(z_rel, plane_rel)``: each point's Z minus the focal Z, and
        each point's signed distance from the annulus plane minus the focal
        point's (NaN without a plane).
    """
    points = np.asarray(points, dtype=float)
    focal_pt = np.asarray(focal_pt, dtype=float)
    z_rel = points[:, 2] - focal_pt[2]
    if plane is None:
        return z_rel, np.full(len(points), np.nan)
    normal, d = plane
    return z_rel, points @ normal + d - float(normal @ focal_pt + d)


def calc_tpi_and_tri(
    pcd,
    center: Optional[np.ndarray] = None,
    radius_inner: float = settings.DEFAULT_TPI_RADIUS_INNER,
    radius_outer: Optional[float] = None,
    annulus_width: Optional[float] = None,
    generate_image: bool = True,
    center_z_from_inner: bool = False,
    raster_cell_size: Optional[float] = TPI_RASTER_CELL_SIZE,
    min_annulus_coverage: Optional[float] = TPI_MIN_ANNULUS_COVERAGE,
) -> Tuple[float, float, float, float, float, float, Optional[np.ndarray]]:
    """Compute TPI and TRI for a point cloud (Weiss 2001; Wilson et al. 2007).

    Both metrics share the same annulus neighbourhood in the horizontal (XY)
    plane between ``radius_inner`` and the outer limit; Z is unconstrained
    (vertical cylinder geometry).  The inner exclusion zone prevents the focal
    object (e.g. the coral colony) from biasing the neighbourhood statistics.

    The outer limit is specified as **one** of:

    - ``radius_outer`` — absolute distance from ``center``.
    - ``annulus_width`` — fixed extension beyond ``radius_inner``
      (outer limit = ``radius_inner + annulus_width``).

    If neither is supplied, ``settings.DEFAULT_TPI_RADIUS_OUTER`` is used.

    TPI variants (Weiss 2001):

    - ``TPI_abs``: ``z_center - mean(z_annulus)`` in the same units as the
      point cloud.
    - ``TPI_plane``: Signed perpendicular distance of ``center`` from the
      best-fit plane of the annulus points.  Corrects for sloping habitat.  The
      plane normal is oriented to +Z so the sign is physically meaningful (see
      below).

    Positive TPI values indicate a crest / elevated position; negative indicate
    a hollow / depressed position.

    Area weighting: the annulus (and, with ``center_z_from_inner``, the inner
    disc) is rasterized in XY at ``raster_cell_size`` (default 1 cm,
    ``settings.TPI_RASTER_CELL_SIZE``) and each cell's points are replaced by
    their mean, so every cell of the surface counts once, as on the elevation
    grid TPI is defined on. Photogrammetric clouds are denser where cameras
    were closer: on 83 close-up targets of a 40 m plot the points per 1 cm cell
    varied 5x within an annulus, and weighting by point instead changed
    ``TPI_abs`` by 0.02 m (median; values about 0.19 m), while 5 mm, 1 cm and
    2 cm cells agreed within 2 mm. The cells must not be finer than the
    cloud's point spacing (empty cells would favour the dense parts again; a
    warning is logged when fewer than half of the annulus cells hold points).
    ``raster_cell_size=None`` weights every point equally (the default before
    0.2.0). The cell size is in the cloud's units: metres in the world frame.

    Partial annulus: when the focal point is near the edge of the cloud (or of
    the extracted neighbourhood), part of the annulus is empty and, on a slope,
    the neighbourhood mean comes only from the up- or downslope side. The
    covered fraction of the annulus area is measured with
    :func:`annulus_coverage`; below ``min_annulus_coverage`` all six metrics
    are returned as NaN with a warning. Neighbourhoods should therefore be
    extracted as vertical cylinders (XY radius) at least ``radius_outer`` wide.

    TRI variants (Wilson et al. 2007):

    - ``TRI_abs``: Mean absolute Z-difference between the focal point and all
      annulus points: ``mean(|z_annulus_i - z_center|)``.  Direct point-cloud
      analog of the Wilson et al. bathymetric DEM formula.  Caveat: Wilson's
      formula assumes neighbours are the immediately adjacent DEM cells, where
      the focal-to-neighbour height offset is negligible so only roughness
      remains.  Here the annulus deliberately excludes the focal region and
      spans a wide radius, so on sloping or pedestalled habitat this term also
      absorbs the focal vertical offset — for a perfectly smooth annulus
      offset by height H it returns ``≈ |H| = |TPI_abs|``, reflecting
      topographic *position* as much as terrain *ruggedness*.  Prefer
      ``TRI_plane`` for a slope/offset-free ruggedness measure.
    - ``TRI_plane``: Mean absolute perpendicular distance of annulus points from
      the best-fit plane: ``mean(|d_i|)``.  Slope- and offset-corrected
      ruggedness (residuals about the annulus plane); same formula as Ra
      (arithmetic mean roughness) but restricted to the annulus neighbourhood
      rather than the full point cloud.  Recommended ruggedness metric.

    Args:
        pcd: Point cloud with a ``.points`` attribute convertible to (N, 3).
        center: (3,) focal point.  If ``None``, the XY centroid + mean Z of
            the whole point cloud is used (including the focal object itself,
            e.g. a colony, which lifts the focal Z).
        radius_inner: Inner radius of the annulus in metres.
        radius_outer: Absolute outer radius.  Mutually exclusive with
            ``annulus_width``.
        annulus_width: Fixed extension beyond ``radius_inner``.  Mutually
            exclusive with ``radius_outer``.
        generate_image: If True (default), produce a visualisation image.
        center_z_from_inner: If True, override focal Z with the mean Z of
            points within ``radius_inner`` (the colony footprint), rather than
            using ``center[2]`` directly.  More robust when the provided center
            coordinate sits above or below the actual surface.
        raster_cell_size: XY cell size (metres) for area weighting: each cell
            contributes one representative point (the mean XYZ of its points)
            to every statistic. Default ``TPI_RASTER_CELL_SIZE`` (0.01); ``None``
            weights every point equally.
        min_annulus_coverage: Smallest covered fraction of the annulus area
            (:func:`annulus_coverage`, computed on the points before
            rasterizing) for a result; below it NaN is returned. ``None`` or 0
            disables the check. Default ``TPI_MIN_ANNULUS_COVERAGE``.

    Returns:
        A tuple ``(tpi_abs, std_annulus_z, tpi_plane, std_annulus_plane,
        tri_abs, tri_plane, image)`` where:

        - ``tpi_abs``: Absolute TPI at the focal point.
        - ``std_annulus_z``: Standard deviation of Z within the annulus
          (habitat topographic heterogeneity; population SD, ``ddof=0``).
        - ``tpi_plane``: Plane-relative TPI at the focal point.  ``NaN`` if the
          annulus has fewer than 3 points.
        - ``std_annulus_plane``: Standard deviation of annulus-point distances
          from the best-fit plane.  ``NaN`` if the annulus has fewer than 3
          points.
        - ``tri_abs``: Mean absolute Z-difference from focal point to annulus
          points (Wilson et al. 2007 TRI, adapted for point clouds).
        - ``tri_plane``: Mean absolute perpendicular distance of annulus points
          from the best-fit plane (slope-corrected TRI).  ``NaN`` if the annulus
          has fewer than 3 points.
        - ``image``: Visualisation image as an (H, W, 3) uint8 array, or
          ``None``.

    Raises:
        ValueError: If both ``radius_outer`` and ``annulus_width`` are given,
            the point cloud has fewer than 2 points, or
            ``radius_inner >= outer_radius``.

    Note:
        All six metrics are ``NaN`` when the annulus is empty or covered below
        ``min_annulus_coverage``.
        The three plane-based outputs (``tpi_plane``, ``std_annulus_plane``,
        ``tri_plane``) require at least 3 annulus points for a defined best-fit
        plane; with fewer they are returned as ``NaN`` rather than computed from
        a degenerate plane.  The point-based outputs (``tpi_abs``,
        ``std_annulus_z``, ``tri_abs``) remain valid with a single annulus point.
    """
    outer_radius = _resolve_outer_radius(
        radius_inner, radius_outer, annulus_width, settings.DEFAULT_TPI_RADIUS_OUTER
    )

    pts = np.asarray(pcd.points, dtype=float)
    if len(pts) < 2:
        raise ValueError("Point cloud must have at least 2 points for TPI")
    if radius_inner >= outer_radius:
        raise ValueError("radius_inner must be less than the outer radius")

    if center is None:
        focal_xy = pts[:, :2].mean(axis=0)
        focal_z = pts[:, 2].mean()
    else:
        center = np.asarray(center, dtype=float)
        focal_xy = center[:2]
        focal_z = center[2]

    # Vectorised XY distances from focal point — O(N), no KDTree needed
    dists_xy = np.linalg.norm(pts[:, :2] - focal_xy, axis=1)

    if center_z_from_inner and radius_inner > 0:
        inner_mask = dists_xy < radius_inner
        if np.any(inner_mask):
            inner_pts = pts[inner_mask]
            if raster_cell_size is not None:
                inner_keys = np.floor(inner_pts[:, :2] / raster_cell_size).astype(int)
                _, inner_inv, inner_counts = np.unique(
                    inner_keys, axis=0, return_inverse=True, return_counts=True
                )
                inner_z_sums = np.zeros(len(inner_counts))
                np.add.at(inner_z_sums, inner_inv, inner_pts[:, 2])
                focal_z = float((inner_z_sums / inner_counts).mean())
            else:
                focal_z = float(inner_pts[:, 2].mean())

    annulus_desc = (
        f"annulus_width={annulus_width:.3f} m (outer={outer_radius:.3f} m)"
        if annulus_width is not None
        else f"radius_outer={outer_radius:.3f} m"
    )
    logger.info(
        "calc_tpi_and_tri: center=(%.3f, %.3f, %.3f)  radius_inner=%.3f m  %s  n_pts=%d",
        focal_xy[0],
        focal_xy[1],
        focal_z,
        radius_inner,
        annulus_desc,
        len(pts),
    )

    annulus_mask = (dists_xy >= radius_inner) & (dists_xy <= outer_radius)

    if not np.any(annulus_mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None

    annulus_pts = pts[annulus_mask]

    if min_annulus_coverage:
        coverage = annulus_coverage(
            annulus_pts[:, :2], focal_xy, radius_inner, outer_radius
        )
        if coverage < min_annulus_coverage:
            logger.warning(
                "calc_tpi_and_tri: only %.0f%% of the annulus around (%.3f, %.3f) "
                "has points (minimum %.0f%%): the focal point is near the edge of "
                "the cloud, or the neighbourhood is a sphere on a slope (use "
                "get_point_cloud_by_radius(..., cylinder=True)); returning NaN",
                100 * coverage, focal_xy[0], focal_xy[1], 100 * min_annulus_coverage,
            )
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None

    if raster_cell_size is not None:
        keys = np.floor(annulus_pts[:, :2] / raster_cell_size).astype(int)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        voxel_sums = np.zeros((len(counts), 3))
        np.add.at(voxel_sums, inverse, annulus_pts)
        annulus_pts = voxel_sums / counts[:, np.newaxis]
        # Cells finer than the point spacing stay partly empty, and the dense
        # parts again hold more (filled) cells: area weighting needs most cells filled.
        n_cells = np.pi * (outer_radius ** 2 - radius_inner ** 2) / raster_cell_size ** 2
        if len(counts) < TPI_MIN_RASTER_FILL * n_cells:
            logger.warning(
                "calc_tpi_and_tri: only %.0f%% of the %.3g-unit raster cells in the "
                "annulus hold points: the cloud is sparser than the cells, so denser "
                "parts still weigh more (use a larger raster_cell_size)",
                100 * len(counts) / n_cells, raster_cell_size,
            )

    annulus_mean_z = annulus_pts[:, 2].mean()
    std_annulus_z = float(annulus_pts[:, 2].std())

    tpi_abs = float(focal_z - annulus_mean_z)
    focal_pt = np.array([focal_xy[0], focal_xy[1], focal_z])

    # TRI_abs (Wilson): mean absolute Z-difference; valid with >=1 annulus point
    tri_abs = float(np.abs(annulus_pts[:, 2] - focal_z).mean())

    # Plane-based metrics need >=3 annulus points for a defined best-fit plane;
    # with fewer (e.g. after aggressive raster_cell_size voxelization) they
    # degrade to NaN rather than being computed from a degenerate plane.
    plane = annulus_plane(annulus_pts)
    if plane is not None:
        normal, d = plane
        tpi_plane = float(normal @ focal_pt + d)
        std_annulus_plane = float(np.std(annulus_pts @ normal + d))
        # TRI_plane: mean absolute residual about the annulus plane
        tri_plane = float(np.abs(annulus_pts @ normal + d).mean())
    else:
        tpi_plane = std_annulus_plane = tri_plane = np.nan

    image = None
    if generate_image:
        from substrata import visualizations  # lazy: figures only

        image = visualizations.tpi_figure(
            pcd, focal_pt, radius_inner, outer_radius,
            tpi_abs=tpi_abs, tpi_plane=tpi_plane, tri_abs=tri_abs,
            tri_plane=tri_plane, plane=plane, raster_cell_size=raster_cell_size,
            annulus_points=annulus_pts if raster_cell_size is not None else None,
        )
    return tpi_abs, std_annulus_z, tpi_plane, std_annulus_plane, tri_abs, tri_plane, image


def benthic_fraction_from_results(
    results: Dict[str, Optional[Dict[str, Any]]], target_class: str,
    weight_by_probability: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Aggregate classification results into a benthic fraction.

    Considers only entries with a non-empty ``label`` (matched + classified).
    Unweighted: ``count(label == target_class) / n_classified``. Weighted: the
    mean of ``probs[target_class]`` over those samples (falling back to the hard
    0/1 indicator for any sample lacking a ``probs`` map).

    Args:
        results: ``{id: {"label", "confidence", "probs", ...} | None}`` as
            returned by :meth:`Annotations.classify_image_matches`.
        target_class: Class label whose fraction is measured.
        weight_by_probability: Average ``P(target_class)`` instead of counting.

    Returns:
        Tuple ``(fraction, breakdown)`` where ``breakdown`` has ``n_classified``,
        ``n_target`` (hard count) and ``class_counts`` (label -> count).
        ``fraction`` is ``nan`` when nothing was classified.
    """
    classified = [
        r for r in results.values() if r and (r.get("label") is not None)
    ]
    class_counts = Counter(str(r["label"]) for r in classified)
    n_classified = len(classified)
    n_target = class_counts.get(target_class, 0)
    breakdown = {
        "n_classified": n_classified,
        "n_target": n_target,
        "class_counts": dict(class_counts),
    }
    if n_classified == 0:
        return float("nan"), breakdown

    if weight_by_probability:
        probs_sum = 0.0
        for r in classified:
            probs = r.get("probs") or {}
            if probs:
                probs_sum += float(probs.get(target_class, 0.0))
            else:
                # No probability map: fall back to the hard indicator.
                probs_sum += 1.0 if str(r["label"]) == target_class else 0.0
        fraction = probs_sum / n_classified
    else:
        fraction = n_target / n_classified
    return fraction, breakdown


#: Former name of :func:`benthic_fraction_from_results`.
_benthic_fraction_from_results = benthic_fraction_from_results


def _colony_base_z(
    pts: np.ndarray, focal_xy: np.ndarray, radius_inner: float,
    percentile: float,
) -> float:
    """Colony base level: a low percentile of the inner-footprint Z.

    Estimates where the focal colony meets the substrate from the point-cloud
    points within ``radius_inner`` of the focal XY (the colony footprint), using
    a low percentile (default 10th) of their Z rather than a plane regression —
    a deliberately simple, robust "base" against which surrounding sand height is
    compared in :func:`calc_benthic_fraction`.

    Args:
        pts: (N, 3) point-cloud coordinates (world frame).
        focal_xy: (2,) focal XY centre.
        radius_inner: Inner (exclusion) radius in metres defining the footprint.
        percentile: Percentile of footprint Z to use as the base (e.g. 10).

    Returns:
        The base Z, or ``nan`` if no points fall within ``radius_inner``.
    """
    if radius_inner <= 0:
        return float("nan")
    inner_mask = np.linalg.norm(pts[:, :2] - focal_xy, axis=1) < radius_inner
    if not np.any(inner_mask):
        return float("nan")
    return float(np.percentile(pts[inner_mask, 2], percentile))


def _height_weight(z: float, z_colony: float, falloff_depth: float) -> float:
    """One-sided linear height weight for a sample at height ``z``.

    Full weight (1.0) at or above the colony base ``z_colony``; below it the
    weight ramps **linearly** down to 0 over ``falloff_depth`` metres, so sand
    more than ``falloff_depth`` below the base does not count. A ``falloff_depth``
    of 0 (or less) gives a strict ``z >= z_colony`` step.

    Returns ``nan`` if ``z_colony`` is ``nan`` (undefined base).
    """
    if np.isnan(z_colony):
        return float("nan")
    if z >= z_colony:
        return 1.0
    if falloff_depth <= 0:
        return 0.0
    return float(np.clip(1.0 - (z_colony - z) / falloff_depth, 0.0, 1.0))


def _benthic_interaction_from_results(
    results: Dict[str, Optional[Dict[str, Any]]],
    sample_z: Dict[str, float], target_class: str, z_colony: float,
    falloff_depth: float, weight_by_probability: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Height-weighted "interaction cover" of ``target_class`` over the annulus.

    Like :func:`benthic_fraction_from_results` but each sample's target
    contribution is multiplied by a height weight (:func:`_height_weight`) so
    that only ``target_class`` (e.g. sand) at or above the colony base counts
    fully, with a linear falloff below it. The denominator is the whole annulus
    (``n_classified``), so the result is directly comparable to the plain
    benthic fraction.

    Args:
        results: ``{id: {"label", "probs", ...} | None}`` from
            :meth:`Annotations.classify_image_matches`.
        sample_z: ``{id: surface-intercept Z}`` for the sampled points.
        target_class: Class label whose interacting cover is measured.
        z_colony: Colony base Z (:func:`_colony_base_z`).
        falloff_depth: Depth below the base over which the weight ramps to 0.
        weight_by_probability: Multiply the height weight by ``P(target_class)``
            instead of the hard ``label == target_class`` indicator.

    Returns:
        Tuple ``(fraction_interacting, breakdown)``. ``fraction_interacting``
        is ``Σ w·t / n_classified`` (``nan`` if nothing was classified or
        ``z_colony`` is ``nan``); ``breakdown`` has ``interaction_weight_sum``
        (``Σ w·t``), ``n_classified`` and ``z_colony``.
    """
    classified = [
        (rid, r) for rid, r in results.items()
        if r and (r.get("label") is not None)
    ]
    n_classified = len(classified)
    breakdown = {
        "interaction_weight_sum": 0.0,
        "n_classified": n_classified,
        "z_colony": z_colony,
    }
    if n_classified == 0 or np.isnan(z_colony):
        return float("nan"), breakdown

    weight_sum = 0.0
    for rid, r in classified:
        if weight_by_probability:
            probs = r.get("probs") or {}
            t = (
                float(probs.get(target_class, 0.0)) if probs
                else (1.0 if str(r["label"]) == target_class else 0.0)
            )
        else:
            t = 1.0 if str(r["label"]) == target_class else 0.0
        if t == 0.0:
            continue
        w = _height_weight(float(sample_z.get(rid, z_colony)), z_colony,
                           falloff_depth)
        weight_sum += w * t

    breakdown["interaction_weight_sum"] = weight_sum
    return weight_sum / n_classified, breakdown


def _diagnose_benthic_matching(
    intercepts, cams, cam_list, pcd,
    reprojection_threshold_discard: float, reprojection_intercept_radius: float,
    n_probe: int = 5,
) -> None:
    """Log why :func:`calc_benthic_fraction` failed to match sample points.

    Matching is done in the world frame (the passed ``pcd`` is world-framed).
    Probes a handful of intercepts to distinguish the common failure modes:
    no (enabled) cameras; the cloud is not actually world-transformed (points
    only project with the cameras' *original* poses); a frame/scale mismatch
    (out of view either way); or the occlusion filter discarding in-view views
    (too-strict ``discard`` threshold or a ray-cast that misses the surface).
    Findings are emitted as a single ``logger.warning`` block.
    """
    lines = []
    n_cams = len(cam_list)
    n_enabled = sum(1 for c in cam_list if getattr(c, "enabled", True) is not False)
    lines.append(f"cameras supplied: {n_cams} ({n_enabled} enabled)")
    if n_enabled == 0:
        lines.append("-> no enabled cameras; nothing can match.")
        logger.warning(
            "calc_benthic_fraction matching diagnostics:\n  " + "\n  ".join(lines)
        )
        return

    pcd_wt = np.asarray(getattr(pcd, "world_transform", np.eye(4)), dtype=float)
    cams_wt = np.asarray(getattr(cams, "world_transform", np.eye(4)), dtype=float)
    wt_desc = "identity" if np.allclose(pcd_wt, np.eye(4)) else "non-identity"
    lines.append(f"pcd.world_transform is {wt_desc}")
    if not np.allclose(pcd_wt, cams_wt):
        lines.append("pcd.world_transform != cams.world_transform (frames differ).")

    # Probe a few points: how many cameras see them in the world frame (what we
    # match in) vs the cameras' original poses (a sanity check on the cloud's
    # frame). For intercepts coords == orig_coords (the same world-frame point).
    probe = list(intercepts.data.values())[:n_probe]
    inview_world = inview_orig = 0
    for ann in probe:
        for cam in cam_list:
            if getattr(cam, "enabled", True) is False:
                continue
            if cam.get_pixel_coords(ann.coords)[0] is not None:
                inview_world += 1
            if cam.get_pixel_coords(
                ann.coords, use_orig_coords=True
            )[0] is not None:
                inview_orig += 1
    lines.append(
        f"probe {len(probe)} pts x {n_enabled} cams in-view: "
        f"world poses -> {inview_world}, original poses -> {inview_orig}"
    )

    if inview_world == 0 and inview_orig == 0:
        ic = np.array([a.coords for a in probe], dtype=float)
        cc = np.array(
            [c.coords for c in cam_list if getattr(c, "coords", None) is not None],
            dtype=float,
        )
        lines.append(
            "points project out of view either way -> likely a frame/scale "
            "mismatch between the point cloud and the cameras."
        )
        if len(ic) and len(cc):
            lines.append(f"intercept XYZ min/max: {ic.min(0)} / {ic.max(0)}")
            lines.append(f"camera   XYZ min/max: {cc.min(0)} / {cc.max(0)}")
    elif inview_world == 0 and inview_orig > 0:
        lines.append(
            "points are in-view only with the cameras' ORIGINAL poses -> the "
            "point cloud does not look world-transformed; pass the world-frame "
            "pcd (the same one the cameras share)."
        )
    else:
        # In-view in the world frame, yet nothing matched: the occlusion filter
        # is responsible. Probe whether relaxing the discard threshold helps.
        relaxed, errs = 0, []
        for ann in probe:
            saved, saved_list = ann.image_match, ann.image_matches
            try:
                ms = ann.get_image_matches(
                    cam_list, max_cams=1, pcd=pcd, use_orig_coords=False,
                    intercept_radius=reprojection_intercept_radius,
                    reprojection_threshold_discard=float("inf"),
                )
                if ms:
                    relaxed += 1
                    if ms[0].reprojection_error is not None:
                        errs.append(ms[0].reprojection_error)
            except Exception:  # noqa: BLE001 - diagnostic probe only.
                pass
            finally:
                ann.image_match, ann.image_matches = saved, saved_list
        if relaxed and errs:
            lines.append(
                f"relaxing the occlusion threshold matched {relaxed}/{len(probe)} "
                f"probe pts with reprojection errors ~{np.median(errs):.3f} m -> "
                f"the discard threshold ({reprojection_threshold_discard} m) is "
                "too strict; raise reprojection_threshold_discard."
            )
        elif relaxed:
            lines.append(
                f"relaxing the occlusion threshold matched {relaxed}/{len(probe)} "
                "probe pts -> raise reprojection_threshold_discard."
            )
        else:
            lines.append(
                "in-view points yield no reprojection intercept even with the "
                "threshold relaxed -> the occlusion ray misses the surface; "
                "increase reprojection_intercept_radius (currently "
                f"{reprojection_intercept_radius} m), or check the camera image "
                "files exist."
            )
    logger.warning(
        "calc_benthic_fraction matching diagnostics:\n  " + "\n  ".join(lines)
    )


def calc_benthic_fraction(
    pcd,
    cams,
    classifier: Union[str, Any],
    target_class: str,
    center: Optional[np.ndarray] = None,
    radius_inner: float = settings.DEFAULT_BENTHIC_RADIUS_INNER,
    radius_outer: Optional[float] = None,
    annulus_width: Optional[float] = None,
    sample_spacing: float = settings.DEFAULT_BENTHIC_SAMPLE_SPACING,
    intercept_search_radius: float = settings.DEFAULT_BENTHIC_INTERCEPT_RADIUS,
    crop_size: Optional[int] = settings.TRAIN_CROP_SIZE,
    batch_size: int = 64,
    weight_by_probability: bool = False,
    colony_base_percentile: float = settings.DEFAULT_BENTHIC_BASE_PERCENTILE,
    base_falloff_depth: float = settings.DEFAULT_BENTHIC_BASE_FALLOFF,
    reprojection_threshold_discard: float = (
        settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD
    ),
    reprojection_intercept_radius: float = (
        settings.DEFAULT_INTERCEPT_SEARCH_RADIUS
    ),
    focal_annotation: Optional[Any] = None,
    generate_image: bool = True,
    show_image_matches: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Fraction of the benthos around a focal point that is ``target_class``.

    Mirrors :func:`calc_tpi_and_tri`'s annulus neighbourhood (``radius_inner`` +
    ``radius_outer``/``annulus_width``; the inner radius excludes the focal
    object). A regular XY grid at ``sample_spacing`` is laid over the annulus;
    each cell is turned into a surface point via
    :meth:`PointCloud.get_z_intercepts`, the best camera image that sees it is
    found (:meth:`Annotation.get_image_matches`, in the **world frame** and
    occlusion-filtered via ``pcd``) and classified with the crop classifier
    (:meth:`Annotations.classify_image_matches` -> ``classify_image_match``).
    The fraction of samples classified as ``target_class`` is then returned.
    If no sample matches a camera image, :func:`_diagnose_benthic_matching`
    logs the likely cause (coordinate frame, occlusion threshold, cameras).

    Args:
        pcd: Project ``PointCloud`` in the **world frame** (the same frame as
            ``cams``, as the ``ProjectInitializer`` guarantees). Sample points
            are taken from and matched against this world-frame cloud.
        cams: ``Cameras`` container (or a list of ``Camera`` objects).
        classifier: A loaded classifier or a path to one (``.pkl`` crop classifier or
            ``.pt`` point classifier); a path
            is loaded once up front (not per sample).
        target_class: Class label whose areal fraction is measured.
        center: (3,) focal point. If ``None``, the XY centroid + mean Z of
            ``pcd`` is used.
        radius_inner: Inner radius of the sampled annulus in metres.
        radius_outer: Absolute outer radius (mutually exclusive with
            ``annulus_width``).
        annulus_width: Fixed extension beyond ``radius_inner`` (mutually
            exclusive with ``radius_outer``).
        sample_spacing: XY grid spacing of sample points in metres.
        intercept_search_radius: XY radius used to find each z-intercept.
        crop_size: Square crop size for the classifier (defaults to the training
            crop size).
        batch_size: Inference batch size for the (batched) classification pass.
        weight_by_probability: Average ``P(target_class)`` over samples instead
            of counting hard predictions (see
            :func:`benthic_fraction_from_results`).
        colony_base_percentile: Percentile of the inner-footprint (within
            ``radius_inner``) Z used as the colony **base** level ``z_colony``
            for the height-weighted ``fraction_interacting`` (see
            :func:`_colony_base_z`). Lower means a deeper base.
        base_falloff_depth: Depth (metres) below ``z_colony`` over which the
            per-sample height weight ramps **linearly** from 1 to 0; sand more
            than this far below the base does not count toward
            ``fraction_interacting``. ``0`` gives a strict at/above-base cutoff
            (see :func:`_height_weight`).
        reprojection_threshold_discard: Image matches whose reprojection error
            exceeds this (metres) are treated as occluded and dropped. Raise it
            if good views are being discarded.
        reprojection_intercept_radius: Ray-cast search radius (metres) for the
            occlusion check; enlarge it if the ray misses the surface.
        generate_image: If True, render a top-down visualisation into the
            returned ``"image"``. When ``weight_by_probability`` is set, the
            classified points are coloured by ``P(target_class)`` (red
            intensity) rather than target/other markers.
        focal_annotation: Optional source annotation (e.g. the colony being
            measured). When given, its ``center`` (``coords``) and existing
            ``image_match`` are reused — so ``show_image_matches`` fills the inner
            circle with the colony crop without recomputing an intercept/match.
            ``measure``/``measure_all`` pass the annotation automatically.
        show_image_matches: Make the measurement ``"image"`` a side-by-side
            comparison plot — the classified sample dots (left) next to each
            sample's classifier-input crop placed at the same position, tiled
            into a grid (right) — instead of the dots-only view, for
            troubleshooting misclassification. It is not displayed inline — show
            it with ``annotation.show_measurement_images()``.
        debug: Always run and log the matching diagnostics (otherwise they run
            only when nothing matched).

    Returns:
        Dict with ``fraction`` (``nan`` if nothing was classified),
        ``target_class``, ``weighted``, ``n_samples``, ``n_intercepts``,
        ``n_matched``, ``n_classified``, ``n_target``, ``class_counts``,
        ``center``, ``radius_inner``, ``radius_outer`` and ``image`` (the
        dots view, or the combined comparison plot when ``show_image_matches``).
        Also ``z_colony`` (colony base level), ``fraction_interacting``
        (height-weighted areal cover of ``target_class`` at/above the base,
        ``Σ w·t / n_classified``) and ``interaction_weight_sum`` (its
        numerator). ``fraction_interacting`` is the extra "interacting sand"
        variant of ``fraction``; the plain ``fraction`` is unchanged.

    Raises:
        ValueError: If both ``radius_outer`` and ``annulus_width`` are given,
            the point cloud is empty, or ``radius_inner >= outer_radius``.
    """
    from substrata import classification  # lazy: avoids fastai import at module load

    outer_radius = _resolve_outer_radius(
        radius_inner, radius_outer, annulus_width,
        settings.DEFAULT_BENTHIC_RADIUS_OUTER,
    )

    pts = np.asarray(pcd.points, dtype=float)
    if len(pts) == 0:
        raise ValueError("Point cloud is empty")
    if radius_inner >= outer_radius:
        raise ValueError("radius_inner must be less than the outer radius")

    if center is None and focal_annotation is not None:
        center = getattr(focal_annotation, "coords", None)
    if center is None:
        focal_xy = pts[:, :2].mean(axis=0)
        focal_z = float(pts[:, 2].mean())
    else:
        center = np.asarray(center, dtype=float)
        focal_xy = center[:2]
        focal_z = float(center[2])
    focal_pt = np.array([focal_xy[0], focal_xy[1], focal_z])

    # Regular XY grid over the annulus bounding box, kept to the annulus ring.
    cx, cy = float(focal_xy[0]), float(focal_xy[1])
    n_steps = int(np.floor(outer_radius / sample_spacing))
    offsets = np.arange(-n_steps, n_steps + 1) * sample_spacing
    gx, gy = np.meshgrid(cx + offsets, cy + offsets)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    r = np.linalg.norm(grid - [cx, cy], axis=1)
    xy_coords = grid[(r >= radius_inner) & (r <= outer_radius)]

    annulus_desc = (
        f"annulus_width={annulus_width:.3f} m (outer={outer_radius:.3f} m)"
        if annulus_width is not None
        else f"radius_outer={outer_radius:.3f} m"
    )

    result: Dict[str, Any] = {
        "fraction": float("nan"),
        "target_class": target_class,
        "weighted": weight_by_probability,
        "n_samples": len(xy_coords),
        "n_intercepts": 0,
        "n_matched": 0,
        "n_classified": 0,
        "n_target": 0,
        "class_counts": {},
        "center": focal_pt,
        "radius_inner": radius_inner,
        "radius_outer": outer_radius,
        "z_colony": float("nan"),
        "fraction_interacting": float("nan"),
        "interaction_weight_sum": 0.0,
        "image": None,
    }

    if len(xy_coords) == 0:
        logger.warning(
            "calc_benthic_fraction: no grid samples in the annulus "
            "(radius_inner=%.3f m, %s, spacing=%.3f m).",
            radius_inner, annulus_desc, sample_spacing,
        )
        return result

    # Load the learner once so it is not re-loaded per sample.
    learn = classification.get_image_classifier(classifier) \
        if isinstance(classifier, str) else classifier

    # Sample the surface at each grid XY -> InterceptAnnotations (in the pcd's
    # frame, with its world_transform, as get_image_matches requires).
    intercepts = pcd.get_z_intercepts(xy_coords, intercept_search_radius)
    result["n_intercepts"] = len(intercepts)
    if len(intercepts) == 0:
        logger.info(
            "calc_benthic_fraction: center=(%.3f, %.3f, %.3f)  radius_inner=%.3f m"
            "  %s  n_samples=%d  no surface intercepts.",
            cx, cy, focal_z, radius_inner, annulus_desc, len(xy_coords),
        )
        return result

    cam_list = list(cams.data.values()) if hasattr(cams, "data") else list(cams)
    for ann in intercepts.data.values():
        try:
            # World-frame matching: the intercepts come from the world-frame pcd,
            # so project with the world camera poses (use_orig_coords=False) to
            # keep projection and the occlusion ray-cast in the same frame.
            ann.get_image_matches(
                cam_list, max_cams=1, pcd=pcd, use_orig_coords=False,
                intercept_radius=reprojection_intercept_radius,
                reprojection_threshold_discard=reprojection_threshold_discard,
            )
        except Exception as e:  # noqa: BLE001 - keep going; diagnosed below.
            logger.debug("image match failed for intercept %s: %s", ann.id, e)
    n_matched = sum(
        1 for ann in intercepts.data.values() if ann.image_match is not None
    )
    result["n_matched"] = n_matched

    # Troubleshooting: if nothing (or, in debug, regardless) matched, probe why.
    if n_matched == 0 or debug:
        _diagnose_benthic_matching(
            intercepts, cams, cam_list, pcd,
            reprojection_threshold_discard, reprojection_intercept_radius,
        )
    if n_matched == 0:
        logger.warning(
            "calc_benthic_fraction: 0/%d intercepts matched a camera image; "
            "returning fraction=nan (see diagnostics above).",
            len(intercepts),
        )
        return result

    results = intercepts.classify_image_matches(
        learn, crop_size, batch_size=batch_size
    )
    fraction, breakdown = benthic_fraction_from_results(
        results, target_class, weight_by_probability
    )
    result.update(fraction=fraction, **breakdown)

    # Extra "interacting sand" metric: weight each target sample by its surface
    # height relative to the colony base (p10 of the inner footprint), so only
    # sand at/above the base counts fully (linear falloff below). The plain
    # fraction above is left untouched.
    z_colony = _colony_base_z(pts, focal_xy, radius_inner, colony_base_percentile)
    sample_z = {
        aid: float(ann.coords[2]) for aid, ann in intercepts.data.items()
    }
    fraction_interacting, ibreak = _benthic_interaction_from_results(
        results, sample_z, target_class, z_colony, base_falloff_depth,
        weight_by_probability,
    )
    result["z_colony"] = z_colony
    result["fraction_interacting"] = fraction_interacting
    result["interaction_weight_sum"] = ibreak["interaction_weight_sum"]
    # Per-sample height weights drive the variable dot-outline in the plots.
    sample_weights = {
        aid: _height_weight(z, z_colony, base_falloff_depth)
        for aid, z in sample_z.items()
    }

    logger.info(
        "calc_benthic_fraction: center=(%.3f, %.3f, %.3f)  radius_inner=%.3f m  %s"
        "  n_samples=%d  n_classified=%d  %s%s=%.4f  z_colony=%.3f"
        "  fraction_interacting=%.4f",
        cx, cy, focal_z, radius_inner, annulus_desc, len(xy_coords),
        breakdown["n_classified"],
        "weighted " if weight_by_probability else "",
        f"fraction({target_class})", fraction, z_colony, fraction_interacting,
    )

    if generate_image or show_image_matches:
        from substrata import visualizations  # lazy: figures only

        result["image"] = visualizations.benthic_fraction_figure(
            pcd, result, intercepts, results, cam_list=cam_list,
            sample_weights=sample_weights, crop_size=crop_size,
            sample_spacing=sample_spacing, focal_annotation=focal_annotation,
            show_image_matches=show_image_matches,
            intercept_search_radius=intercept_search_radius,
            reprojection_threshold_discard=reprojection_threshold_discard,
            reprojection_intercept_radius=reprojection_intercept_radius,
        )

    return result


def vector_dispersion_from_normals(normals) -> float:
    """Fisher's spherical dispersion ``(N - R) / (N - 1)`` of (N, 3) normals.

    ``R`` is the length of the sum of the unit normals; 0.0 for fewer than 2
    normals. See :func:`get_vector_dispersion`.
    """
    normals = np.asarray(normals, dtype=float)
    n = len(normals)
    if n > 0:
        unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        resultant = float(np.linalg.norm(unit.sum(axis=0)))
    else:
        resultant = 0.0
    return (n - resultant) / (n - 1) if n > 1 else 0.0


def normal_deviation_deg(normals, reference_normals=None):
    """Angle (degrees, 0-180) of each normal from the mean (resultant) direction.

    Directional, like :func:`vector_dispersion_from_normals`: a flipped normal
    counts as 180 degrees. Zero-length normals are left as they are (angle 90).

    Args:
        normals: (N, 3) normals whose deviation is returned.
        reference_normals: Normals whose resultant defines the mean direction
            (default: ``normals``), e.g. the full cloud when ``normals`` is a
            displayed subsample.

    Returns:
        tuple: ``(deviation_deg, mean_normal)``; ``mean_normal`` is the unit
        resultant, or None when the resultant vanishes (then every deviation
        is 90).
    """
    def _unit(v):
        v = np.asarray(v, dtype=float).reshape(-1, 3)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms

    unit = _unit(normals)
    ref = unit if reference_normals is None else _unit(reference_normals)
    resultant = ref.sum(axis=0)
    r_len = float(np.linalg.norm(resultant))
    if r_len <= 1e-10:
        return np.full(len(unit), 90.0), None
    mean_normal = resultant / r_len
    alignment = unit @ mean_normal
    return np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0))), mean_normal


@_renamed_kwargs(geom="pcd")
def get_vector_dispersion(pcd, generate_image=True):
    """Vector (normal) dispersion of a point cloud or mesh (Young et al. 2017).

    Fisher's spherical dispersion of the unit normals,
    ``(N - R) / (N - 1)`` with ``R`` the length of the resultant (sum of the
    unit normals): 0 when all normals are parallel, 1 for uniformly scattered
    normals, up to 2 for normals in two opposite groups. The statistic is
    *directional*, not axial: the normals must be consistently oriented (e.g.
    all facing outward / up); flipping half of identical normals gives 1.0.

    Args:
        pcd: A ``PointCloud`` / ``SimplePointCloud`` / open3d point cloud (or
            any object with ``points`` and ``normals``), or an open3d
            ``TriangleMesh`` (its triangle normals are used).
        generate_image: Also render the QC image (expensive plotly/Kaleido
            render) for point clouds; ``None`` otherwise.

    Returns:
        tuple: ``(dispersion, image)``; ``dispersion`` is 0.0 for fewer than 2
        normals.

    Raises:
        TypeError: For an unsupported geometry type, or a point cloud without
            one normal per point.
    """
    is_point_cloud = False
    o3d = sys.modules.get("open3d")  # an open3d mesh implies open3d is loaded
    if o3d is not None and isinstance(pcd, o3d.geometry.TriangleMesh):
        if not pcd.has_triangle_normals():
            pcd.compute_triangle_normals()
        normals = np.asarray(pcd.triangle_normals, dtype=float)
    elif hasattr(pcd, "points") and hasattr(pcd, "normals"):
        normals = np.asarray(pcd.normals, dtype=float)
        n_points = len(np.asarray(pcd.points))
        if normals.shape != (n_points, 3):
            raise TypeError(
                f"{type(pcd).__name__} needs one normal per point "
                f"(normals {normals.shape}, {n_points} points)"
            )
        is_point_cloud = True
    else:
        raise TypeError(f"Unsupported geometry type: {type(pcd).__name__}.")

    vector_normal_dispersion = vector_dispersion_from_normals(normals)

    image = None
    if generate_image and is_point_cloud and len(normals) > 0:
        from substrata import visualizations  # lazy: figures only

        image = visualizations.visualize_vector_dispersion(
            pcd, interactive=False, dispersion=vector_normal_dispersion
        )

    return vector_normal_dispersion, image


def get_rgb_stats(pcd):
    """Compute per-channel median colors and a luma of a point cloud.

    Args:
        pcd: A point cloud whose ``colors`` are RGB values in [0, 1].

    Returns:
        tuple: ``(median_red, median_green, median_blue, luminance)``. The
        "luminance" is the Rec. 709 weighted sum of the three *median* channel
        values (not the median of per-point luminance), computed on the stored
        (gamma-encoded) colours, i.e. a luma rather than a radiometric
        luminance.
    """
    median_red = np.median(np.asarray(pcd.colors)[:, 0])
    median_green = np.median(np.asarray(pcd.colors)[:, 1])
    median_blue = np.median(np.asarray(pcd.colors)[:, 2])
    luminance = 0.2126 * median_red + 0.7152 * median_green + 0.0722 * median_blue
    return median_red, median_green, median_blue, luminance


def hemispherical_disk(resolution: int) -> np.ndarray:
    """The imaging disk of a ``resolution`` x ``resolution`` hemispherical image.

    Pixels whose centres lie within ``resolution / 2`` of the image centre: the
    whole upper hemisphere (zenith 0-90 degrees) in the projection of
    :func:`hemispherical_cover`.

    Returns:
        (resolution, resolution) bool array.
    """
    c = np.arange(resolution) + 0.5 - resolution / 2.0
    return (c[:, None] ** 2 + c[None, :] ** 2) <= (resolution / 2.0) ** 2


def hemispherical_pixels(points, center, resolution, max_radius=None):
    """Project the points above ``center`` onto the hemispherical image.

    Equidistant (polar) projection: a point at zenith angle ``phi`` and
    azimuth ``theta`` (from +X towards +Y) lands at ``phi / pi * resolution``
    pixels from the image centre. Image rows follow world +X, columns world +Y.

    Returns:
        tuple: ``(rows, cols, keep, dist)``: pixel indices of the kept points,
        the boolean mask of kept points (above ``center``, and within
        ``max_radius`` horizontally if given) over the input, and the kept
        points' distances from ``center``.
    """
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    keep = rel[:, 2] > 0
    if max_radius is not None:
        keep &= np.hypot(rel[:, 0], rel[:, 1]) <= max_radius
    rel = rel[keep]
    dist = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    phi = np.arccos(rel[:, 2] / dist)
    u = ((phi * np.cos(theta)) + np.pi / 2) / np.pi * resolution
    v = ((phi * np.sin(theta)) + np.pi / 2) / np.pi * resolution
    rows = np.clip(np.floor(u).astype(int), 0, resolution - 1)
    cols = np.clip(np.floor(v).astype(int), 0, resolution - 1)
    return rows, cols, keep, dist


#: Former name of :func:`hemispherical_pixels`.
_hemispherical_pixels = hemispherical_pixels


def hemispherical_cover(
    points: np.ndarray,
    center: np.ndarray,
    resolution: int = 200,
    max_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary canopy-cover raster of the hemisphere above ``center``.

    Every point above ``center`` (``z > center_z``; within ``max_radius``
    horizontally if given) marks the pixel it projects to as covered, in the
    equidistant projection of :func:`hemispherical_pixels` (radius
    proportional to zenith angle; rows follow world +X, columns world +Y).
    Each point covers exactly one pixel, so the cover depends on point density
    and ``resolution``: a sparse cloud or a fine raster leaves pixel holes in
    solid surfaces (read as sky).

    Args:
        points: (N, 3) point coordinates.
        center: (3,) viewpoint (e.g. the annotation).
        resolution: Image size in pixels (the disk spans zenith 0-90 degrees).
        max_radius: Only use points within this horizontal distance.

    Returns:
        tuple: ``(cover, disk)``, two (resolution, resolution) bool arrays: the
        covered pixels (inside the disk) and the imaging disk
        (:func:`hemispherical_disk`).
    """
    rows, cols, _, _ = hemispherical_pixels(points, center, resolution, max_radius)
    disk = hemispherical_disk(resolution)
    cover = np.zeros((resolution, resolution), dtype=bool)
    cover[rows, cols] = True
    return cover & disk, disk


def gap_fractions(
    cover_px: np.ndarray, img_mask: np.ndarray, seed_points_px
) -> Tuple[float, float, np.ndarray]:
    """Raw and connected ("filled") gap fractions of a binary cover raster.

    Sky is ``img_mask & ~cover_px``. ``gapF_raw`` is the sky area over the
    disk area. ``gapF_fill`` counts only the sky connected (4-connectivity:
    pixels sharing an edge) to at least one seed: the union of the connected
    sky components that contain a seed, over the disk area. Sky enclosed by
    cover (pixel holes in a canopy, gaps under an overhang) is then left out.
    Seeds that fall on cover or outside the disk are skipped with a warning.

    Args:
        cover_px: (H, W) bool array, True where covered.
        img_mask: (H, W) bool array of the imaging disk.
        seed_points_px: Iterable of ``(x, y)`` pixel seeds, ``x`` the column and
            ``y`` the row (OpenCV order).

    Returns:
        tuple: ``(gapF_raw, gapF_fill, connected)``; ``connected`` is the
        (H, W) bool mask of the seeded sky.

    Raises:
        ValueError: If the masks differ in shape or the disk is empty.
    """
    cover_px = np.asarray(cover_px, dtype=bool)
    img_mask = np.asarray(img_mask, dtype=bool)
    if cover_px.shape != img_mask.shape:
        raise ValueError(
            f"cover {cover_px.shape} and mask {img_mask.shape} differ in shape"
        )
    area = int(img_mask.sum())
    if area == 0:
        raise ValueError("the imaging disk (img_mask) is empty")
    sky = img_mask & ~cover_px
    gapF_raw = float(sky.sum()) / area

    import cv2

    _, labels = cv2.connectedComponents(sky.astype(np.uint8), connectivity=4)
    h, w = sky.shape
    seeded = set()
    for x, y in seed_points_px:
        x, y = int(x), int(y)
        if not (0 <= x < w and 0 <= y < h):
            logger.warning("gap_fractions: seed (%d, %d) is outside the image; "
                           "skipped", x, y)
            continue
        label = int(labels[y, x])
        if label == 0:
            logger.warning("gap_fractions: seed (%d, %d) is on cover (or outside "
                           "the imaging disk); skipped", x, y)
            continue
        seeded.add(label)
    connected = np.isin(labels, sorted(seeded)) if seeded else np.zeros_like(sky)
    gapF_fill = float(connected.sum()) / area
    return gapF_raw, gapF_fill, connected


@_renamed_kwargs(output_filename="output_path")
def calc_gap_fraction(
    annotation,
    pcd,
    resolution=200,
    color_output=True,
    max_radius=None,
    output_path=None,
    seed_points=None,
    fill_color=settings.GAP_FRACTION_FILL_COLOR,
    show_rings=False,
    ring_color=None,
    show_axes=False,
    width=500,
    height=500,
):
    """Gap fraction from a hemispherical projection of the cloud above a point.

    The points above the annotation are projected onto a hemispherical image
    (:func:`hemispherical_cover`; equidistant projection: the image radius is
    proportional to the zenith angle, rows follow world +X and columns world
    +Y). The metrics are computed on that binary raster by
    :func:`gap_fractions`, independently of the display options:

    - ``gapF_raw``: sky pixels / all pixels of the imaging disk (sky over
      total, not sky over benthos).
    - ``gapF_fill``: the sky connected to the seed points / all pixels (holes
      enclosed by cover do not count).

    Interpretation: pixels are equal steps of zenith angle, not equal solid
    angles, so these are equidistant-image fractions (an opaque canopy over
    zenith 0-45 degrees gives 0.75, whereas the open solid angle fraction would
    be cos 45 = 0.71). Each point covers one pixel, so ``gapF_raw`` rises with
    sparser clouds and finer ``resolution`` (holes in solid surfaces read as
    sky); compare values only at the same resolution and similar point
    density. ``gapF_fill`` is less sensitive to such holes.

    Args:
        annotation: Annotation object with coords attribute.
        pcd: Point cloud object.
        resolution: Image resolution (default 200).
        color_output: Paint the cover in the point colours (default True) or
            white. Display only.
        max_radius: Optional maximum horizontal distance of the points used.
        output_path: Optional filename to save the image to (not saved
            when None).
        seed_points: Optional list of ``(x, y)`` tuples with proportional
            coordinates in ``[0, 1]`` specifying the seeds of the connected sky
            (``x`` horizontal, ``y`` vertical, origin top-left). If ``None``,
            the default seeds ``(0.5, 0.5)`` (image centre: the zenith) and
            ``(0.25, 0.5)`` are used; the latter is at zenith 45 degrees towards
            world -Y (historically called "upslope"; after
            ``align_y_upslope`` +Y is upslope, so it is downslope). Seeds on
            cover are skipped with a warning. When provided, the seed points are
            drawn as markers on the output image (after the gap fraction is
            calculated), which helps to tune their placement.
        fill_color: RGB colour used to paint the connected sky region, as a
            3-tuple in 0-255 (default ``settings.GAP_FRACTION_FILL_COLOR``, an
            ocean blue from the ``RdBu_r`` colour map used by the TPI plot).
            Display only: the gap fractions come from the binary cover raster,
            so the colour may also occur in the benthic imagery.
        show_rings: When True, draw concentric zenith-angle rings at 22.5, 45,
            67.5 and 90 degrees over the output image (default False). In this
            projection the radius scales linearly with the zenith angle, so the
            rings sit at 25%, 50%, 75% and 100% of the image radius, the last
            tracing the horizon at the edge of the imaging area. They are drawn
            after the gap fraction is measured and so never affect the result.
        ring_color: RGB colour for those rings, as a 3-tuple in 0-255
            (default: white on the bare image; black, dashed like the inner
            annulus circle of the TPI plot, when *show_axes* is True).
        show_axes: When True, the returned (and saved) image is a figure of
            the projection with zenith-angle axes in degrees (X to the right,
            Y up) on a white background, styled like
            :func:`substrata.visualizations.visualize_tpi` so axes and fonts
            match at the same figure size (default False: the bare
            projection). Does not affect the gap fractions.
        width: Figure width in pixels when *show_axes* is True (default 500).
        height: Figure height in pixels when *show_axes* is True (default 500).

    Returns:
        tuple: ``(gapF_raw, gapF_fill, image)`` - the raw gap fraction, the
            connected-sky gap fraction, and the hemispherical projection as an
            ``(H, W, 3)`` uint8 RGB array (sky dark grey, cover in the point
            colours or white, connected sky in ``fill_color``).
    """
    radius = resolution // 2
    points = np.asarray(pcd.points)
    cover, img_mask = hemispherical_cover(
        points, annotation.coords, resolution, max_radius
    )

    # Seeds in pixels (x = column, y = row). By default the centre (zenith) and
    # a point at zenith 45 deg towards -Y; otherwise the user's proportional
    # coordinates.
    if seed_points is None:
        seed_points_px = [(radius, radius), (radius // 2, radius)]
    else:
        seed_points_px = [
            (
                int(np.clip(round(sx * resolution), 0, resolution - 1)),
                int(np.clip(round(sy * resolution), 0, resolution - 1)),
            )
            for sx, sy in seed_points
        ]

    # The metrics, from the binary raster only (no display colours involved)
    gapF_raw, gapF_fill, connected = gap_fractions(cover, img_mask, seed_points_px)

    # The image (display only; drawn after the metrics, never read back)
    from substrata import visualizations  # lazy: figures only

    image = visualizations.render_gap_fraction(
        pcd, annotation.coords, resolution, connected,
        max_radius=max_radius, color_output=color_output,
        seed_points_px=seed_points_px if seed_points is not None else None,
        fill_color=fill_color, show_rings=show_rings, ring_color=ring_color,
        show_axes=show_axes, width=width, height=height,
        output_filename=output_path,
    )
    return gapF_raw, gapF_fill, image


def cloud_xy_histogram(points, step_size=0.1):
    """The point counts of a cloud on an xy grid of ``step_size`` cells (what the box search maximises).

    Returns ``(hist, x_edges, y_edges)``: ``hist[i, j]`` counts the points with x in
    ``[x_edges[i], x_edges[i + 1])`` and y in ``[y_edges[j], y_edges[j + 1])``.
    """
    points = np.asarray(points)[:, :2]
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_edges = np.arange(x_min, x_max + step_size, step_size)
    y_edges = np.arange(y_min, y_max + step_size, step_size)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])
    return hist, x_edges, y_edges


def find_optimal_box_position(pcd, box_length, box_width, step_size=0.1, vis=False):
    """Position of the axis-aligned box that contains the most cloud points.

    The cloud's XY points are counted on a grid of ``step_size`` cells
    (:func:`cloud_xy_histogram`) and the grid is convolved with a box of
    ``round(box_length / step_size) x round(box_width / step_size)`` cells; the
    box is anchored on a grid edge. The box dimensions should be multiples of
    ``step_size``: otherwise the search uses the rounded box (logged as a
    warning) while the returned box has the requested size.

    Args:
        pcd: Point cloud object with attribute ``points`` (N x 3 array).
        box_length: The box's length along X.
        box_width: The box's width along Y.
        step_size: Grid resolution of the search (same units).
        vis: Show the box on the cloud (a figure; off by default).

    Returns:
        tuple: ``([x_min, y_min], [x_max, y_max])`` of the box.

    A box larger than the cloud along an axis starts at the cloud's minimum
    along that axis (with a warning).

    Raises:
        ValueError: If the box is smaller than one grid cell.
    """
    # A 2D histogram of the point cloud on a grid of the step size.
    points = np.asarray(pcd.points)[:, :2]
    hist, x_edges, y_edges = cloud_xy_histogram(points, step_size)

    # Map rectangle dimensions to the number of bins.
    box_length_bins = int(np.round(box_length / step_size))
    box_width_bins = int(np.round(box_width / step_size))
    if box_length_bins < 1 or box_width_bins < 1:
        raise ValueError(
            f"box {box_length} x {box_width} is smaller than the grid step {step_size}"
        )
    if box_length_bins > hist.shape[0] or box_width_bins > hist.shape[1]:
        # Larger than the cloud along an axis: pad the grid (empty cells beyond
        # the cloud) so the box starts at the cloud's minimum along that axis.
        logger.warning(
            "find_optimal_box_position: box %g x %g is larger than the cloud's XY "
            "extent (%.2f x %.2f)", box_length, box_width,
            hist.shape[0] * step_size, hist.shape[1] * step_size,
        )
        pad_x = max(0, box_length_bins - hist.shape[0])
        pad_y = max(0, box_width_bins - hist.shape[1])
        hist = np.pad(hist, ((0, pad_x), (0, pad_y)))
        x_edges = x_edges[0] + step_size * np.arange(hist.shape[0] + 1)
        y_edges = y_edges[0] + step_size * np.arange(hist.shape[1] + 1)
    searched = (box_length_bins * step_size, box_width_bins * step_size)
    tol = 1e-6 * step_size
    if not np.allclose(searched, (box_length, box_width), rtol=0, atol=tol):
        logger.warning(
            "find_optimal_box_position: box %g x %g is not a multiple of the step "
            "%g; searched with %g x %g", box_length, box_width, step_size, *searched,
        )

    # Create a kernel corresponding to the rectangle.
    kernel = np.ones((box_length_bins, box_width_bins))

    # Convolve the histogram with the kernel.
    from scipy.signal import convolve2d

    conv_result = convolve2d(hist, kernel, mode="valid")

    # Find the location of the maximum value in the convolved result.
    max_idx = np.unravel_index(np.argmax(conv_result), conv_result.shape)
    best_count = int(conv_result[max_idx])

    # Map the grid indices back to world coordinates.
    best_x = x_edges[max_idx[0]]
    best_y = y_edges[max_idx[1]]
    best_box = ([best_x, best_y], [best_x + box_length, best_y + box_width])

    print(f"Optimal box contains {best_count} out of {len(points)} points")
    print(
        "Box coordinates: ",
        [list(map(float, best_box[0])), list(map(float, best_box[1]))],
    )

    if vis:
        from substrata import visualizations  # lazy: figures only

        visualizations.show_grid_cells(pcd, best_box)

    return best_box


def subdivide_boxes(bboxes, new_cell_size, tol=1e-9):
    """
    Subdivide one or more bounding boxes into smaller grid cells of size new_cell_size.

    Parameters:
      bboxes (tuple or list): A single bounding box in the form ([x_min, y_min], [x_max, y_max])
                              or a list of such bounding boxes.
      new_cell_size (float): Side length of each new square grid cell.
      tol (float): Tolerance for floating-point inaccuracies.

    Returns:
      List of new grid cell bounding boxes, each as ([x_min, y_min], [x_max, y_max]).
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

    new_boxes = []

    for bbox in bboxes:
        # Expect bbox format: ([x_min, y_min], [x_max, y_max])
        min_pt, max_pt = bbox
        x_min, y_min = min_pt
        x_max, y_max = max_pt

        # Calculate the width and height of the bounding box.
        width = round(x_max - x_min, 4)
        height = round(y_max - y_min, 4)
        cs = round(new_cell_size, 4)

        # Compute the number of cells that should fit along each axis.
        ratio_x = width / cs
        ratio_y = height / cs

        # Check that the box dimensions are (nearly) an integer multiple of new_cell_size.
        if not (
            abs(ratio_x - round(ratio_x)) < tol and abs(ratio_y - round(ratio_y)) < tol
        ):
            raise ValueError(
                "Bounding box dimensions must be divisible by new_cell_size within tolerance."
            )

        nx = int(round(ratio_x))
        ny = int(round(ratio_y))

        # Subdivide the bounding box.
        for i in range(nx):
            for j in range(ny):
                new_x_min = x_min + i * cs
                new_y_min = y_min + j * cs
                new_x_max = new_x_min + cs
                new_y_max = new_y_min + cs
                new_boxes.append(([new_x_min, new_y_min], [new_x_max, new_y_max]))

    return new_boxes


def generate_random_xy_points_within_cells(bboxes, points_per_cell, z_value=0):
    """
    Generate random points within specified bounding boxes."""
    random_points = []

    for box in bboxes:
        [xmin, ymin], [xmax, ymax] = box

        x_coords = np.random.uniform(xmin, xmax, points_per_cell)
        y_coords = np.random.uniform(ymin, ymax, points_per_cell)
        z_coords = np.full(points_per_cell, z_value)

        points = np.vstack((x_coords, y_coords, z_coords)).T
        random_points.append(points)

    random_points = np.vstack(random_points)
    return random_points


def get_mask_surface_area(annotation, predictor=None):
    """Calculate and return the surface area of the mask in the image."""
    if annotation.image_match:
        return annotation.image_match.get_mask_surface_area(predictor)
    else:
        return None


@dataclass
class DepthRegressionResult:
    """Result of fitting a linear regression of depth against 3-D position.

    Produced by ``fit_depth_regression``, which models depth as a linear
    function of ``(x, y, z)``. Depths follow the package convention, negative
    below the surface (see :func:`normalize_depth_sign`): ``depths``,
    ``depths_pred``, ``depth_offset`` and the model
    ``depth = depth_offset + up_vector . p`` are all in that convention, so
    ``up_vector`` points up.
    """

    #: Regression coefficient vector: the gradient of depth (negative below the
    #: surface), so it points up (towards shallower water).
    up_vector: np.ndarray
    #: Regression intercept (depth at the origin).
    depth_offset: float
    #: Depth change per unit distance, i.e. the L2 norm of the coefficient vector.
    depth_per_unit: float
    #: Mean squared error of the predicted depths.
    mse: float
    #: Root mean squared error of the predicted depths.
    rmse: float
    #: Mean absolute error of the predicted depths.
    mae: float
    #: Coefficient of determination (R-squared) of the fit.
    r2: float
    #: Predicted depth for each input point.
    depths_pred: np.ndarray
    #: Residuals (observed minus predicted depth).
    depths_res: np.ndarray
    #: The observed depths used for the fit (after the sign normalisation).
    depths: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: True when the given depths were positive below the surface and negated.
    depths_negated: bool = False


def normalize_depth_sign(depths, what: str = "depths") -> Tuple[np.ndarray, bool]:
    """Return depths in the package convention: negative below the surface.

    Depth sensors and CSVs often record depth as a positive number. When the
    median of the finite ``depths`` is positive they are negated (with a
    warning), so that depth increases upwards like the world Z axis.

    Args:
        depths: Array-like of depths in metres.
        what: Name used in the warning.

    Returns:
        tuple: ``(depths, negated)``, a float array and whether it was negated.
    """
    depths = np.asarray(depths, dtype=float)
    finite = depths[np.isfinite(depths)]
    if finite.size and np.median(finite) > 0:
        logger.warning(
            "%s are positive (median %.2f m): negated to the convention "
            "'depth negative below the surface'", what, float(np.median(finite)),
        )
        return -depths, True
    return depths, False


def regression_errors(observed, predicted) -> Dict[str, float]:
    """Error metrics of predictions against observations.

    Args:
        observed: (N,) observed values (e.g. sensor depths).
        predicted: (N,) predicted values.

    Returns:
        dict: ``n``, ``mse`` (mean squared error), ``rmse``, ``mae`` (mean
        absolute error) and ``r2`` (coefficient of determination, scikit-learn's
        ``r2_score``: 1.0 for a perfect fit of constant observations, 0.0 for an
        imperfect one).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(observed, predicted))
    return {
        "n": int(len(observed)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def fit_depth_regression(
    points: np.ndarray, depths: np.ndarray
) -> DepthRegressionResult:
    """Fit the linear depth model ``depth ≈ depth_offset + up_vector · (x, y, z)``.

    Ordinary least squares of depth on the point coordinates (e.g. camera or
    depth-sensor positions in the model frame). Depths are first brought to the
    convention "negative below the surface" (:func:`normalize_depth_sign`;
    positive depths are negated with a warning), so the coefficient vector is
    the depth gradient pointing up: ``Transform.from_up_vector(up_vector)``
    levels the model with +Z up, and ``depth_per_unit = |up_vector|`` is the
    model-to-metre scale along the vertical.

    Args:
        points: (N, 3) positions.
        depths: (N,) depths in metres (either sign convention).

    Returns:
        DepthRegressionResult: model, error metrics (in metres), predictions
        and residuals, all in the negative-below-surface convention.

    Raises:
        ValueError: For mismatched shapes or fewer than 2 points.
    """
    points = np.asarray(points, dtype=float)
    depths = np.asarray(depths, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")
    if depths.ndim != 1 or depths.shape[0] != points.shape[0]:
        raise ValueError("depths must be shape (N,) and align with points")
    if len(points) < 2:
        raise ValueError("Need at least 2 points for regression")

    depths, negated = normalize_depth_sign(depths)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(points, depths)
    coef = model.coef_.astype(float)
    depth_offset = float(model.intercept_)
    depth_per_unit = float(np.linalg.norm(coef))

    depths_pred = model.predict(points).astype(float)
    depths_res = depths - depths_pred
    errors = regression_errors(depths, depths_pred)

    return DepthRegressionResult(
        up_vector=coef,
        depth_offset=depth_offset,
        depth_per_unit=depth_per_unit,
        mse=errors["mse"],
        rmse=errors["rmse"],
        mae=errors["mae"],
        r2=errors["r2"],
        depths_pred=depths_pred,
        depths_res=depths_res,
        depths=depths,
        depths_negated=negated,
    )


class DepthResidualAnalyzer:
    """Analyzes depth residuals for any container with depth data.

    Works with both Cameras and Annotations containers that have:
    - Items with depth_sensor_m and coords/orig_coords attributes
    - Parent container with up_vector and depth_offset attributes

    Sensor depths are reported in the convention of :func:`fit_depth_regression`
    (negative below the surface; :func:`normalize_depth_sign`), so they compare
    with the model's estimates. The filtered containers returned share the
    items with the original container; the items keep their ``parent``.
    """

    def __init__(self, container):
        """Initialize the analyzer with a container.

        Args:
            container: A Cameras or Annotations container instance.
        """
        self.container = container

    def _calculate_estimated_depth(self, item):
        """Calculate estimated depth using parent's up_vector and depth_offset.

        Args:
            item: A Camera or Annotation instance.

        Returns:
            float: Estimated depth in meters, or None if calculation not possible.
        """
        parent = self.container
        if not hasattr(parent, "up_vector") or parent.up_vector is None:
            return None
        if not hasattr(parent, "depth_offset") or parent.depth_offset is None:
            return None

        # For cameras, use orig_coords (as in Camera.depth_in_m property)
        # For annotations, use orig_coords to match the coordinates used in regression
        # (regression uses coords at regression time, but orig_coords preserves those values)
        container_type = type(parent).__name__
        if container_type == "Cameras":
            coords = getattr(item, "orig_coords", None)
            if coords is None:
                coords = getattr(item, "coords", None)
            # Depth sensor mounted away from the camera centre (lever arm): use the
            # sensor position, as in the regression that produced up_vector
            from substrata import cameras

            offset_units = cameras.sensor_offset_units(
                getattr(parent, "depth_sensor_offset", None),
                getattr(parent, "depth_sensor_offset_scale", 1.0),
            )
            if offset_units is not None and coords is not None:
                coords = item.sensor_coords(
                    offset_units, use_orig_coords=getattr(item, "orig_coords", None) is not None
                )
        else:
            # For Annotations, prefer orig_coords (preserves original values used in regression)
            # Fall back to coords if orig_coords not available
            coords = getattr(item, "orig_coords", None)
            if coords is None:
                coords = getattr(item, "coords", None)

        if coords is None:
            return None

        # up_vector may be a geom.Vector (homogeneous 4-vector): use its xyz
        up = np.asarray(getattr(parent.up_vector, "xyz", parent.up_vector), float)[:3]
        coords = np.asarray(coords, dtype=float)[:3]
        return float(parent.depth_offset + float(np.dot(up, coords)))

    @staticmethod
    def _sensor_depths(items):
        """The items' sensor depths, negative below the surface (list of float)."""
        depths, _ = normalize_depth_sign(
            [item.depth_sensor_m for item in items], "sensor depths"
        )
        return [float(d) for d in depths]

    def _filtered_container(self, items):
        """An empty-like container holding ``items`` (their ``parent`` unchanged)."""
        filtered = self.container._empty_like()
        for item in items:
            item_id = getattr(item, "cam_id", None) or getattr(item, "id", None)
            if item_id:
                filtered.data[item_id] = item
        return filtered

    def _filter_items_with_depth(
        self, depth_accuracy_threshold=None, use_accuracy_filter=False
    ):
        """Filter items that have depth data.

        Args:
            depth_accuracy_threshold (float, optional): Threshold for depth accuracy
                filtering (only used for cameras with use_accuracy_filter=True).
            use_accuracy_filter (bool): Whether to apply accuracy threshold filtering.

        Returns:
            list: Filtered list of items with depth data.
        """
        items = []
        for item in self.container.data.values():
            if not hasattr(item, "depth_sensor_m") or item.depth_sensor_m is None:
                continue
            if not hasattr(item, "coords") or item.coords is None:
                continue

            # Apply accuracy filter only if requested and item has depth_acc
            if use_accuracy_filter and depth_accuracy_threshold is not None:
                if hasattr(item, "depth_acc") and item.depth_acc is not None:
                    if item.depth_acc > depth_accuracy_threshold:
                        continue

            items.append(item)
        return items

    def get_depths_and_estimated_depths(
        self, depth_accuracy_threshold=None, use_accuracy_filter=False
    ):
        """Get sensor depths and predicted depths for items.

        Args:
            depth_accuracy_threshold (float, optional): Threshold for depth accuracy
                filtering (only used with use_accuracy_filter=True).
            use_accuracy_filter (bool): Whether to apply accuracy threshold filtering.

        Returns:
            tuple: (depths, est_depths, filtered_container) where:
                - depths: List of sensor depths
                - est_depths: List of estimated depths from regression
                - filtered_container: Container with filtered items
        """
        items = self._filter_items_with_depth(
            depth_accuracy_threshold, use_accuracy_filter
        )
        depths = self._sensor_depths(items)
        est_depths = [self._calculate_estimated_depth(item) for item in items]

        # Filter out None values from est_depths
        valid_indices = [i for i, ed in enumerate(est_depths) if ed is not None]
        depths = [depths[i] for i in valid_indices]
        est_depths = [est_depths[i] for i in valid_indices]
        items = [items[i] for i in valid_indices]

        return depths, est_depths, self._filtered_container(items)

    def get_depths_and_z_coords(
        self, depth_accuracy_threshold=None, use_accuracy_filter=False
    ):
        """Get sensor depths and z-coordinates for items.

        Args:
            depth_accuracy_threshold (float, optional): Threshold for depth accuracy
                filtering (only used with use_accuracy_filter=True).
            use_accuracy_filter (bool): Whether to apply accuracy threshold filtering.

        Returns:
            tuple: (depths, z_coords, filtered_container) where:
                - depths: List of sensor depths
                - z_coords: List of z-coordinates
                - filtered_container: Container with filtered items
        """
        items = self._filter_items_with_depth(
            depth_accuracy_threshold, use_accuracy_filter
        )
        depths = self._sensor_depths(items)
        z_coords = [item.coords[2] for item in items]
        return depths, z_coords, self._filtered_container(items)

    def show_depth_vs_est_depth_residuals(self, width=15, height=5, **kwargs):
        """Figures of sensor vs estimated depths (``(fig1, fig2)``).

        See :func:`substrata.visualizations.depth_residual_figures`.
        """
        from substrata import visualizations  # lazy: figures only

        return visualizations.depth_residual_figures(
            self, "estimated", width, height, **kwargs
        )

    def show_z_vs_depth_residuals(self, width=15, height=5, **kwargs):
        """Figures of sensor depths vs Z coordinates (``(fig1, fig2)``).

        See :func:`substrata.visualizations.depth_residual_figures`.
        """
        from substrata import visualizations  # lazy: figures only

        return visualizations.depth_residual_figures(
            self, "z", width, height, **kwargs
        )

    def save_depth_residuals_pdf(self, filepath=None, width=15, height=5, **kwargs):
        """Save the depth-residual figures as a PDF; returns the filepath.

        See :func:`substrata.visualizations.save_depth_residuals_pdf`.
        """
        from substrata import visualizations  # lazy: figures only

        return visualizations.save_depth_residuals_pdf(
            self, filepath, width, height, **kwargs
        )


def depth_band_sample(
    points: np.ndarray,
    depth_min: float,
    depth_max: float,
    width_step: float,
    depth_step: float,
    seed: Optional[int] = None,
) -> dict:
    """Intercepts over a depth band: one point per cell of a grid across the band's whole width.

    The cloud is in the world frame (z: depth, negative below the surface; x, y
    horizontal, metres). Rows are depth intervals of ``depth_step`` from
    ``depth_min`` to ``depth_max`` (positive metres); columns are intervals of
    ``width_step`` across the slope (the horizontal direction along the depth
    contours, from a plane fitted to the band), over all of the band's width.
    In each cell the cloud point nearest a random position in it (across,
    depth) is taken: like an intercept (the surface at a random position), not
    a random cloud point, which would favour densely sampled parts. Cells
    without cloud are left out. For training data at depths a plot's best box
    leaves out, without searching for a box.

    Args:
        points: (N, 3) cloud points, world frame.
        depth_min, depth_max: The band (metres; their order and sign do not
            matter).
        width_step: Column width across the slope (m).
        depth_step: Row height in depth (m).
        seed: Random seed (the same points again).

    Returns:
        dict: ``points`` (n, 3) the chosen points; ``cells`` (n, 2) their
        [column, row]; ``across`` the across direction (2,) in x, y; ``u0`` the
        across coordinate of column 0's edge; ``width`` the band's width (m);
        ``columns``, ``rows``; ``cells_with_cloud`` (= n); ``band_points`` (cloud
        points in the band); ``cloud_depth`` [min, max] of the whole cloud.
    """
    if width_step <= 0 or depth_step <= 0:
        raise ValueError("width_step and depth_step must be positive")
    pts = np.asarray(points, dtype=float)
    depth = -pts[:, 2]
    lo, hi = sorted((abs(float(depth_min)), abs(float(depth_max))))
    if not hi > lo:
        raise ValueError(f"an empty depth band ({depth_min:g} to {depth_max:g} m)")
    span = [float(np.nanmin(depth)), float(np.nanmax(depth))] if len(pts) else [float("nan")] * 2
    band = (depth >= lo) & (depth < hi)
    if not band.any():
        raise ValueError(f"no cloud between {lo:g} and {hi:g} m depth (the cloud spans "
                         f"{span[0]:.1f} to {span[1]:.1f} m)")
    band_pts, band_depth = pts[band], depth[band]
    rng = np.random.default_rng(seed)
    # the across direction: along the depth contours of a plane z = a x + b y + c fitted to the band
    fit_pts = (band_pts if len(band_pts) <= 200_000
               else band_pts[rng.choice(len(band_pts), 200_000, replace=False)])
    (a, b, _c), *_ = np.linalg.lstsq(
        np.c_[fit_pts[:, 0], fit_pts[:, 1], np.ones(len(fit_pts))], fit_pts[:, 2], rcond=None)
    gradient = float(np.hypot(a, b))
    across = np.array([-b, a]) / gradient if gradient > 1e-9 else np.array([1.0, 0.0])
    across_pos = band_pts[:, :2] @ across
    u0 = float(across_pos.min())
    col = ((across_pos - u0) // width_step).astype(np.int64)
    row = ((band_depth - lo) // depth_step).astype(np.int64)
    ncol = int(col.max()) + 1
    nrow = int(np.ceil((hi - lo) / depth_step - 1e-9))
    key = row * ncol + col
    cells, inv = np.unique(key, return_inverse=True)
    # a random position in each cell; its nearest point in (across, depth)
    cell_across = u0 + (cells % ncol + rng.random(len(cells))) * width_step
    cell_depth = lo + (cells // ncol + rng.random(len(cells))) * depth_step
    dist = (across_pos - cell_across[inv]) ** 2 + (band_depth - cell_depth[inv]) ** 2
    order = np.lexsort((dist, inv))
    first = order[np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]]
    return dict(points=band_pts[first], cells=np.c_[cells % ncol, cells // ncol], across=across,
                u0=u0, width=float(across_pos.max() - u0), columns=ncol, rows=nrow,
                cells_with_cloud=len(cells), band_points=int(band.sum()), cloud_depth=span)


def _subsample_idx(n, k, rng):
    """Indices of at most ``k`` of ``n`` items (all of them if ``n <= k``)."""
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


@dataclass
class PcdCorrespondence:
    """How well a source cloud coincides with a target cloud in their overlap.

    Produced by :func:`pcd_correspondence`; drawn by
    :func:`substrata.visualizations.plot_pcd_correspondence`. Distances are in
    the clouds' units.
    """

    #: (M, 3) source points the distances were computed for (a sample).
    sample: np.ndarray
    #: (M,) C2C: distance of each sample point to the nearest target point.
    c2c: np.ndarray
    #: (M,) signed vertical offset (source z minus the z of a plane fitted to
    #: the 8 nearest target points); NaN on near-vertical surfaces and over
    #: target gaps.
    dz: np.ndarray
    #: (K, 3) target points within the source bounding box (plus the margin).
    target_overlap: np.ndarray
    #: Median nearest-neighbour spacing of the target in the overlap.
    target_spacing: float
    #: Source bounding box (min corner, max corner).
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    #: The summary numbers (see :func:`pcd_correspondence`).
    stats: Dict[str, float] = field(default_factory=dict)


def pcd_correspondence(
    src,
    tgt,
    units: str = "m",
    max_sample_points: int = 1_000_000,
    margin: float = 0.05,
    labels: Tuple[str, str] = ("source", "target"),
    seed: int = 0,
) -> PcdCorrespondence:
    """Distances between two clouds in the same frame, in their overlap.

    The target is cropped to the source bounding box (plus ``margin`` of its
    extent) and, for a random sample of at most ``max_sample_points`` source
    points (``seed``), two distances are computed:

    - C2C: distance to the nearest target point (3-D).
    - dz: signed vertical difference, source z minus the z of a plane fitted
      to the 8 nearest target points (3-D neighbours keep the matching surface
      under overhangs; the plane stops a slope from absorbing the offset). It
      is NaN where the plane is steep (``|n_z| < 0.3``) or the 8th neighbour is
      more than 10 target spacings away (a gap in the target).

    ``stats`` holds ``n_sample``, ``n_target_overlap``, ``target_spacing``
    (median nearest-neighbour distance of up to 200 000 overlap points, the
    bound on the attainable C2C), ``c2c_median``, ``c2c_p90``, ``dz_median``,
    ``dz_iqr`` (of the finite dz), ``dz_coverage`` (finite share of dz) and,
    with ``units="m"``, ``c2c_within_{2,5,10}cm`` (share of the sample).

    Args:
        src: Source point cloud (anything with ``points``).
        tgt: Target point cloud, same frame as ``src``.
        units: Coordinate unit; ``"m"`` adds the within-2/5/10-cm shares.
        max_sample_points: Source points used for the statistics.
        margin: Fraction of the source extent added around it when cropping
            the target.
        labels: Names of ``src`` and ``tgt`` (error message).
        seed: Seed of the source sample.

    Raises:
        ValueError: When no target point lies in the source bounding box.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    s_pts = _points_array(src)
    t_pts = _points_array(tgt)

    lo, hi = s_pts.min(axis=0), s_pts.max(axis=0)
    pad = margin * (hi - lo)
    in_box = np.all((t_pts >= lo - pad) & (t_pts <= hi + pad), axis=1)
    t_crop = t_pts[in_box]
    if len(t_crop) == 0:
        raise ValueError(
            f"No {labels[1]} points overlap the {labels[0]} bounding box"
        )

    s_sample = s_pts[_subsample_idx(len(s_pts), max_sample_points, rng)]
    t_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t_crop))
    s_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s_sample))
    c2c = np.asarray(s_o3d.compute_point_cloud_distance(t_o3d))
    tree = cKDTree(t_crop)
    spacing_idx = _subsample_idx(len(t_crop), 200_000, rng)
    spacing, _ = tree.query(t_crop[spacing_idx], k=2)
    spacing = float(np.median(spacing[:, 1]))

    d_nn, i_nn = tree.query(s_sample, k=8)
    nbrs = t_crop[i_nn]
    nb_mean = nbrs.mean(axis=1)
    rel = nbrs - nb_mean[:, None, :]
    _, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", rel, rel))
    normal = vecs[:, :, 0]
    tilt = np.einsum("ni,ni->n", normal[:, :2], s_sample[:, :2] - nb_mean[:, :2])
    with np.errstate(divide="ignore", invalid="ignore"):
        z_plane = nb_mean[:, 2] - tilt / normal[:, 2]
    dz = s_sample[:, 2] - z_plane
    dz[(np.abs(normal[:, 2]) < 0.3) | (d_nn[:, -1] > 10 * spacing)] = np.nan
    dz_ok = dz[np.isfinite(dz)]

    stats = {
        "n_sample": int(len(s_sample)),
        "n_target_overlap": int(len(t_crop)),
        "target_spacing": spacing,
        "c2c_median": float(np.median(c2c)),
        "c2c_p90": float(np.percentile(c2c, 90)),
        "dz_median": float(np.median(dz_ok)) if len(dz_ok) else float("nan"),
        "dz_iqr": (
            float(np.subtract(*np.percentile(dz_ok, [75, 25])))
            if len(dz_ok)
            else float("nan")
        ),
        "dz_coverage": float(len(dz_ok) / len(dz)),
    }
    if units == "m":
        for cm in (2, 5, 10):
            stats[f"c2c_within_{cm}cm"] = float(np.mean(c2c < cm / 100))

    return PcdCorrespondence(
        sample=s_sample, c2c=c2c, dz=dz, target_overlap=t_crop,
        target_spacing=spacing, bbox_min=lo, bbox_max=hi, stats=stats,
    )


def camera_residuals(src_xyz, tgt_xyz, inliers=None) -> Dict[str, Any]:
    """Residual distances between matched camera positions in a common frame.

    Args:
        src_xyz: (N, 3) source camera centres, already in the target frame.
        tgt_xyz: (N, 3) corresponding target camera centres.
        inliers: Optional boolean mask of the pairs used for the fit (default
            all).

    Returns:
        dict: ``residuals`` (N,), ``inliers`` (N,) bool, ``n_inliers``,
        ``n_rejected``, ``rms`` and ``median`` of the inlier residuals, and
        ``rejected_min`` / ``rejected_max`` (NaN without rejected pairs).
    """
    src_xyz = np.asarray(src_xyz, dtype=float)
    tgt_xyz = np.asarray(tgt_xyz, dtype=float)
    residuals = np.linalg.norm(src_xyz - tgt_xyz, axis=1)
    inl = (
        np.ones(len(residuals), bool) if inliers is None
        else np.asarray(inliers, dtype=bool)
    )
    r_in, r_out = residuals[inl], residuals[~inl]
    nan = float("nan")
    return {
        "residuals": residuals,
        "inliers": inl,
        "n_inliers": int(inl.sum()),
        "n_rejected": int((~inl).sum()),
        "rms": float(np.sqrt(np.mean(r_in**2))) if len(r_in) else nan,
        "median": float(np.median(r_in)) if len(r_in) else nan,
        "rejected_min": float(r_out.min()) if len(r_out) else nan,
        "rejected_max": float(r_out.max()) if len(r_out) else nan,
    }
