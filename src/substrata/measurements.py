# Standard Library
import sys
import random
import copy
from collections import Counter
from typing import Any, Dict, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np
from scipy.spatial import KDTree
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt  # TO REMOVE? (if you no longer need it, you can comment it out or remove it)
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
from open3d import geometry, utility, cpu
import open3d as o3d
from scipy.signal import convolve2d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass

try:  # TO DO
    import alphashape
    from scipy.spatial import distance_matrix
    from alphashape import optimizealpha
    from shapely.geometry import Point
    from shapely.geometry import Polygon, MultiPolygon
    from matplotlib.patches import Polygon as MplPolygon
except ImportError:
    pass

# Local Modules
from substrata import cameras, pointclouds, visualizations, settings, geom
from substrata.logging import logger


def conduct_PCA(pcd, sort=True):
    """
    Calculate eigenvalues/eigenvectors for pointcloud
    """

    matrix = np.cov(pcd.points.T)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def best_fit_plane_normal(pcd) -> geom.Vector:
    """Return a unit normal (Vector) of the PCA best-fit plane."""
    a, b, c, _ = conduct_PCA(pcd)[:4]
    return geom.Vector([a, b, c, 0])  # promote to 4-vector


def conduct_xy_PCA(pcd, sort=True, visualize=False):
    """
    Calculate eigenvalues/eigenvectors for x/y plane only.
    Optionally print and plot the results.
    """
    xy_points = pcd.points[:, :2]
    mean_xy = np.mean(xy_points, axis=0)
    centered_xy = xy_points - mean_xy  # shape (N, 2)

    cov = np.cov(centered_xy, rowvar=False)  # shape (2, 2)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    if sort:
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

    if visualize:
        visualizations.plot_xy_pca(
            points=xy_points, mean=mean_xy, eig_vecs=eigenvectors, eig_vals=eigenvalues
        )

    return eigenvalues, eigenvectors


def get_best_fit_plane_PCA(
    pcd, inlier_range=settings.DEFAULT_INLIER_RANGE, print_eq=False, align_normals=True
):
    """
    Get best fit plane based on PCA

    author: DvH
    """

    _, eigenvectors = conduct_PCA(pcd)
    a, b, c = normal = eigenvectors[:, 2]
    mean = np.mean(np.asarray(pcd.points), axis=0)
    d = -(np.dot(normal, mean))

    if print_eq:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    mask1 = -inlier_range <= np.asarray(pcd.points).dot(normal) + d
    mask2 = np.asarray(pcd.points).dot(normal) + d <= inlier_range
    combined_mask = np.logical_and.reduce([mask1, mask2])
    inliers_idx = np.where(combined_mask == True)[0]

    # If requested, adjust the plane normal based on the point cloud normals.
    if align_normals:
        normals = np.asarray(pcd.normals)
        if normals.size > 0:
            inlier_normals = normals[inliers_idx]
            avg_normal = np.mean(inlier_normals, axis=0)
            # If the average normal points in the opposite direction, flip the plane.
            if np.dot(avg_normal, [a, b, c]) < 0:
                a, b, c, d = -a, -b, -c, -d

    return a, b, c, d, inliers_idx


def get_best_fit_plane_ransac(
    pcd,
    inlier_range: float = settings.DEFAULT_INLIER_RANGE,
    print_eq: bool = False,
    align_normals: bool = True,
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Fit a plane to a point cloud using the RANSAC method.

    If the input pcd is a SimplePointCloud, it is converted to an Open3D point cloud.

    The resulting plane follows the equation: a*x + b*y + c*z + d = 0.

    Args:
        pcd: A point cloud object. Can be a SimplePointCloud or a full
             PointCloud instance.
        inlier_range (float): Distance threshold for including a point as an inlier.
        print_eq (bool): If True, prints the plane equation.
        align_normals (bool): If True, adjusts the plane normal so it aligns with
                              the general direction of the point cloud normals.

    Returns:
        tuple: (a, b, c, d, inliers_idx), where (a, b, c, d) are the plane coefficients
               and inliers_idx is a NumPy array of indices for points within the inlier range.
    """
    # Convert to Open3D point cloud if necessary.
    if isinstance(pcd, pointclouds.SimplePointCloud):
        o3d_pcd = pcd.get_o3d_pcd()
    else:
        o3d_pcd = pcd.o3d_pcd

    # Use RANSAC to segment the plane.
    plane_model, inliers_idx = o3d_pcd.segment_plane(
        distance_threshold=inlier_range,
        ransac_n=settings.RANSAC_N,
        num_iterations=settings.RANSAC_ITERATIONS,
    )
    a, b, c, d = plane_model

    # If requested, adjust the plane normal based on the point cloud normals.
    if align_normals:
        normals = np.asarray(o3d_pcd.normals)
        if normals.size > 0:
            inlier_normals = normals[inliers_idx]
            avg_normal = np.mean(inlier_normals, axis=0)
            # If the average normal points in the opposite direction, flip the plane.
            if np.dot(avg_normal, [a, b, c]) < 0:
                a, b, c, d = -a, -b, -c, -d

    if print_eq:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    return a, b, c, d, inliers_idx


def get_plane_angles(pcd, vis=False, generate_image=True):
    """Calculate orientation angles for the best-fit plane of a point cloud.

    The plane normal is aligned with the point cloud normals by
    ``get_best_fit_plane_PCA``, so its direction encodes whether the surface
    faces upward or downward.

    Args:
        pcd: Point cloud object with ``.points`` (and optionally ``.normals``).
        vis: If True, show an interactive 3-D visualisation of the elevation.
        generate_image: If True (default), also render a static elevation-angle
            QC image (via plotly/Kaleido). Set False to skip that expensive
            render and return ``None`` for the image (e.g. batch runs).

    Returns:
        tuple: A 6-element tuple containing:
            - **theta_deg** (*float*): Rotation about +Y to null the x-component
              of n in the xz-plane (brings n's xz-projection toward +Z).
            - **psi_deg** (*float*): Rotation about +X to then align n with +Z.
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

    a, b, c, d, inliers_idx = get_best_fit_plane_PCA(pcd)

    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("Degenerate plane normal.")
    n = n / n_norm

    nx, ny, nz = n

    elev = np.arccos(np.clip(nz, -1.0, 1.0))
    elev_deg = float(np.degrees(elev))

    # Y-then-X rotations to align n -> +Z
    theta = np.arctan2(nx, nz)
    psi = np.arctan2(-ny, np.hypot(nx, nz))

    theta_deg = float(np.degrees(theta))
    psi_deg = float(np.degrees(psi))

    r_xy = np.hypot(nx, ny)
    if r_xy < 1e-8:
        az_deg = None
    else:
        az = np.arctan2(ny, nx)
        az_deg = float((np.degrees(az) + 360.0) % 360.0)

    if vis:
        visualizations.visualize_elevation_angle(pcd, [a, b, c, d], interactive=True)

    image = None
    if generate_image:
        image = visualizations.visualize_elevation_angle(
            pcd, [a, b, c, d], interactive=False
        )

    return (
        theta_deg,
        psi_deg,
        elev_deg,
        [float(a), float(b), float(c), float(d)],
        az_deg,
        image,
    )


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
    """
    Calculate deviation rugosity for a pointcloud (how much the points vary from the best fitting plane).
    Also called "plane-detrended roughness"

    author: DvH
    """
    [a, b, c, d] = get_best_fit_plane_PCA(pcd)[0:4]
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
    a, b, c, d = get_best_fit_plane_PCA(pcd)[:4]

    normal = np.array([a, b, c], dtype=float)
    denom = np.linalg.norm(normal)
    if denom == 0.0:
        raise ValueError("Best-fit plane has zero-length normal")

    # Perpendicular distances of all points to the plane
    dist = np.abs(pts @ normal + d) / denom

    ra = float(dist.mean())
    rq = float(np.sqrt((dist**2).mean()))

    # The QC image is an expensive plotly/Kaleido render; skip it unless asked
    # for (e.g. batch runs via measure_all default to metrics only).
    image = None
    if generate_image:
        # Pass ra and rq to avoid recalculating in visualize_roughness
        image = visualizations.visualize_roughness(
            pcd, interactive=False, ra=ra, rq=rq
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


def calc_tpi_and_tri(
    pcd,
    center: Optional[np.ndarray] = None,
    radius_inner: float = settings.DEFAULT_TPI_RADIUS_INNER,
    radius_outer: Optional[float] = None,
    annulus_width: Optional[float] = None,
    generate_image: bool = True,
    center_z_from_inner: bool = False,
    raster_cell_size: Optional[float] = None,
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
            the point cloud is used.
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
        raster_cell_size: If set, rasterize the neighbourhood in XY at this cell
            size (metres) before computing any statistics.  Each cell contributes
            one representative point (mean XYZ of all points within it),
            equalising spatial weight regardless of local point density — useful
            for variable-density photogrammetric clouds.  Has no effect when
            ``None`` (default).

    Returns:
        A tuple ``(tpi_abs, std_annulus_z, tpi_plane, std_annulus_plane,
        tri_abs, tri_plane, image)`` where:

        - ``tpi_abs``: Absolute TPI at the focal point.
        - ``std_annulus_z``: Standard deviation of Z within the annulus
          (habitat topographic heterogeneity).
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

    if raster_cell_size is not None:
        keys = np.floor(annulus_pts[:, :2] / raster_cell_size).astype(int)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        voxel_sums = np.zeros((len(counts), 3))
        np.add.at(voxel_sums, inverse, annulus_pts)
        annulus_pts = voxel_sums / counts[:, np.newaxis]

    annulus_mean_z = annulus_pts[:, 2].mean()
    std_annulus_z = float(annulus_pts[:, 2].std())

    tpi_abs = float(focal_z - annulus_mean_z)
    focal_pt = np.array([focal_xy[0], focal_xy[1], focal_z])

    # TRI_abs (Wilson): mean absolute Z-difference; valid with >=1 annulus point
    tri_abs = float(np.abs(annulus_pts[:, 2] - focal_z).mean())

    # Plane-based metrics need >=3 annulus points for a defined best-fit plane;
    # with fewer (e.g. after aggressive raster_cell_size voxelization) they
    # degrade to NaN rather than being computed from a degenerate plane.
    if len(annulus_pts) >= 3:
        # Best-fit plane of annulus via SVD; normal = last right-singular vector
        # (already unit-length, since SVD returns orthonormal rows).
        centroid = annulus_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(annulus_pts - centroid, full_matrices=False)
        normal = vt[-1]
        # Orient the normal to +Z so the signed tpi_plane is physically
        # meaningful: positive = crest/elevated, negative = hollow/depressed.
        # SVD's sign is otherwise arbitrary and not reproducible across inputs.
        if normal[2] < 0:
            normal = -normal
        d = -normal @ centroid
        tpi_plane = float(normal @ focal_pt + d)
        std_annulus_plane = float(np.std(annulus_pts @ normal + d))
        # TRI_plane: mean absolute residual about the annulus plane
        tri_plane = float(np.abs(annulus_pts @ normal + d).mean())
    else:
        normal = None
        tpi_plane = std_annulus_plane = tri_plane = np.nan

    image = None
    if generate_image:
        if raster_cell_size is not None:
            # Show only the voxelized annulus points — not the dense background.
            # Including the background causes visualize_tpi's random subsampling
            # (max_output_points=50000) to discard most voxels, since they are
            # vastly outnumbered by the original cloud (~300k pts vs ~2k voxels).
            # The radius circles and star marker provide sufficient spatial context.
            display_pcd = pointclouds.SimplePointCloud(annulus_pts)
            display_pts = annulus_pts
            # Scale marker size so each voxel dot fills its physical footprint.
            # visualize_tpi renders height=500px at 100dpi over ~2*outer_radius m;
            # ~65 % of the height is axes area after labels/padding.
            # matplotlib scatter s is in pts² (1 pt = 100/72 px at 100 dpi).
            px_per_m = (500 * 0.65) / (2 * outer_radius)
            pt_diameter = raster_cell_size * px_per_m * (72 / 100)
            vis_point_size = max(2, int(pt_diameter**2))
        else:
            display_pcd = pcd
            display_pts = pts
            vis_point_size = None
        vis_tpi_abs = display_pts[:, 2] - focal_z
        if normal is not None:
            vis_tpi_plane = display_pts @ normal + d - tpi_plane
        else:
            # Plane undefined (<3 annulus points): render the plane panel as
            # NaN, which visualize_tpi displays as light-gray "no data" points.
            vis_tpi_plane = np.full(len(display_pts), np.nan)
        vis_kwargs = dict(
            interactive=False,
            mean_tpi_abs=tpi_abs,
            mean_tpi_plane=tpi_plane,
            mean_tri_abs=tri_abs,
            mean_tri_plane=tri_plane,
            center=focal_pt,
            radius_inner=radius_inner,
            radius_outer=outer_radius,
        )
        if vis_point_size is not None:
            vis_kwargs["point_size"] = vis_point_size
        image = visualizations.visualize_tpi(
            display_pcd, vis_tpi_abs, vis_tpi_plane, **vis_kwargs
        )
    return tpi_abs, std_annulus_z, tpi_plane, std_annulus_plane, tri_abs, tri_plane, image


def _benthic_fraction_from_results(
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

    Like :func:`_benthic_fraction_from_results` but each sample's target
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
        classifier: Loaded fastai learner or path to a ``.pkl`` learner; a path
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
            :func:`_benthic_fraction_from_results`).
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

    # Sample the surface at each grid XY -> InterceptAnnotations. Align the
    # container transform with the pcd so the annotation/pcd/cam frames agree
    # (get_image_matches enforces this).
    intercepts = pcd.get_z_intercepts(xy_coords, intercept_search_radius)
    intercepts.world_transform = pcd.world_transform
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
    fraction, breakdown = _benthic_fraction_from_results(
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

    # The local neighbourhood (the "simple_pcd" around the colony) within the
    # outer radius gives the visualisations the same colony-centred context as
    # visualize_tpi rather than rendering the whole model, in true RGB colour.
    neighborhood = neigh_cols = None
    if generate_image or show_image_matches:
        neigh_mask = np.linalg.norm(pts[:, :2] - focal_xy, axis=1) <= outer_radius
        neighborhood = pointclouds.SimplePointCloud(pts[neigh_mask])
        try:
            cols = np.asarray(pcd.colors, dtype=float)
            neigh_cols = cols[neigh_mask] if len(cols) == len(pts) else None
        except Exception:  # noqa: BLE001 - colours are optional context.
            neigh_cols = None

    # The combined fraction-vs-crops comparison plot (when requested) becomes the
    # measurement image; otherwise the standalone fraction view is used.
    if show_image_matches:
        # Use the colony's existing image match (from the source annotation) for
        # the inner-circle inset; only compute one if it isn't already there.
        focal_im = getattr(focal_annotation, "image_match", None)
        if focal_im is None:
            try:
                focal_anns = pcd.get_z_intercepts(
                    np.array([focal_xy]), intercept_search_radius
                )
                focal_anns.world_transform = pcd.world_transform
                for fann in focal_anns.data.values():
                    fann.get_image_matches(
                        cam_list, max_cams=1, pcd=pcd, use_orig_coords=False,
                        intercept_radius=reprojection_intercept_radius,
                        reprojection_threshold_discard=(
                            reprojection_threshold_discard
                        ),
                    )
                fvals = list(focal_anns.data.values())
                focal_im = fvals[0].image_match if fvals else None
            except Exception as e:  # noqa: BLE001 - colony inset is best-effort.
                logger.debug("focal image match failed: %s", e)

        combined = visualizations.visualize_benthic_image_matches(
            intercepts, results, target_class, crop_size,
            center=focal_pt, radius_inner=radius_inner,
            radius_outer=outer_radius, weighted=weight_by_probability,
            background_pcd=neighborhood, background_colors=neigh_cols,
            focal_image_match=focal_im, cell_size=sample_spacing,
            sample_weights=sample_weights,
            fraction_interacting=fraction_interacting, z_colony=z_colony,
        )
        # The combined plot is the single measurement image; show it on demand
        # via annotation.show_measurement_images().
        result["image"] = combined
    elif generate_image:
        result["image"] = visualizations.visualize_benthic_fraction(
            intercepts, results, target_class, center=focal_pt,
            radius_inner=radius_inner, radius_outer=outer_radius,
            background_pcd=neighborhood, background_colors=neigh_cols,
            weighted=weight_by_probability, sample_weights=sample_weights,
            fraction_interacting=fraction_interacting, z_colony=z_colony,
        )

    return result


def get_fractal_dimension(pcd, iterations=10, plot=False):
    """
    get the fractal_dimension of a PCD following Schroeder, 1991 & Yuval, 2023
    """
    lower, upper = pcd.bounding_box
    vox_size = np.max(upper - lower) * 1.0001
    box_sizes = []
    box_counts = []
    for i in range(0, iterations):
        voxel_grid = geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd.o3d_pcd, vox_size, lower, upper
        )
        vox_count = len(voxel_grid.get_voxels())
        box_sizes.append(vox_size)
        box_counts.append(vox_count)
        vox_size = vox_size / 2
    log_box_sizes = np.log(1 / np.array(box_sizes))
    log_counts = np.log(box_counts)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        log_box_sizes, log_counts
    )
    if plot:
        plt.scatter(log_box_sizes, log_counts, label="Data")
        plt.plot(
            log_box_sizes,
            intercept + slope * log_box_sizes,
            "r",
            label=f"Fit: D = {slope:.2f}",
        )
        plt.xlabel("log(1/box size)")
        plt.ylabel("log(count)")
        plt.legend()
        plt.show()
        print(f"Fractal Dimension: {slope:.2f}")
    return slope


def get_vector_dispersion(geom, generate_image=True):
    """
    Function to get the vector normal dispersion of a geometry (either
    PointCloud or Mesh). Adapted from Young et al., 2017.

    Returns the dispersion scalar and a static visualization image (numpy array),
    same pattern as calc_roughness. Only PointCloud-like geometry is visualized;
    for TriangleMesh, the image is None. When ``generate_image`` is False the
    (expensive plotly/Kaleido) image is skipped and returned as None.
    """
    if isinstance(
        geom,
        (pointclouds.SimplePointCloud, pointclouds.PointCloud, geometry.PointCloud),
    ):
        normals = geom.normals
        i = len(geom.points)
    elif isinstance(geom, geometry.TriangleMesh):
        normals = np.asarray(geom.triangle_normals)
        i = len(geom.triangles)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry).__name__}.")
    cos_x = normals[:, 0] / np.sqrt(
        normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2
    )
    cos_y = normals[:, 1] / np.sqrt(
        normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2
    )
    cos_z = normals[:, 2] / np.sqrt(
        normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2
    )
    R1 = np.sqrt(sum(cos_x) ** 2 + sum(cos_y) ** 2 + sum(cos_z) ** 2)
    vector_normal_dispersion = (i - R1) / (i - 1) if i > 1 else 0.0

    image = None
    if (
        generate_image
        and isinstance(
            geom,
            (pointclouds.SimplePointCloud, pointclouds.PointCloud, geometry.PointCloud),
        )
        and i > 0
    ):
        image = visualizations.visualize_vector_dispersion(
            geom, interactive=False, dispersion=vector_normal_dispersion
        )

    return vector_normal_dispersion, image


def get_rgb_stats(pcd):
    """Compute per-channel median colors and overall luminance of a point cloud.

    Args:
        pcd: A point cloud whose ``colors`` are RGB values in [0, 1].

    Returns:
        tuple: ``(median_red, median_green, median_blue, luminance)``, where
        luminance is the Rec. 709 weighted sum of the median channel values.
    """
    median_red = np.median(np.asarray(pcd.colors)[:, 0])
    median_green = np.median(np.asarray(pcd.colors)[:, 1])
    median_blue = np.median(np.asarray(pcd.colors)[:, 2])
    luminance = 0.2126 * median_red + 0.7152 * median_green + 0.0722 * median_blue
    return median_red, median_green, median_blue, luminance


def generate_filled_circle(center, radius, spacing):
    """Generate a grid of points filling a horizontal circle.

    Points are laid out on a regular grid in the x-y plane at the center's z
    height, keeping only those within ``radius`` of the center.

    Args:
        center: Sequence ``(cx, cy, cz)`` giving the circle center coordinates.
        radius (float): Circle radius.
        spacing (float): Grid spacing between candidate points.

    Returns:
        numpy.ndarray: Array of shape ``(N, 3)`` with the points inside the
        circle, all sharing the center's z value.
    """
    points = []
    cx, cy, cz = center  # Center coordinates
    for x in np.arange(cx - radius, cx + radius + spacing, spacing):
        for y in np.arange(cy - radius, cy + radius + spacing, spacing):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                points.append((x, y, cz))
    return np.asarray(points)


def get_canopy_cover_hemisphere(pcd, annotation, radius=0.2, colors=False):
    """

    Author: DvH
    """
    # Project point cloud points onto a hemisphere
    dir_vecs = np.asarray(pcd.points) - annotation.coords
    dir_vec_lengths = np.linalg.norm(dir_vecs, axis=1, keepdims=True)
    vecs_to_sphere = (radius / dir_vec_lengths) * dir_vecs
    sphere_projection = annotation.coords + vecs_to_sphere
    hemisphere = sphere_projection[sphere_projection[:, 2] > annotation.coords[2]]

    # Stereographical projection of hemisphere by projecting
    # points towards the southpole stopping at the equator
    southpole_coords = (
        annotation.coords[0],
        annotation.coords[1],
        annotation.coords[2] - radius,
    )
    equator_z = annotation.coords[2]
    southpole_vecs = southpole_coords - hemisphere
    southpole_vec_lengths = (equator_z - hemisphere[:, 2]) / southpole_vecs[:, 2]
    southpole_vec_lengths = southpole_vec_lengths[:, np.newaxis]
    stereograph_projection = hemisphere + southpole_vecs * southpole_vec_lengths
    hemisphere2d = geometry.PointCloud()
    hemisphere2d.points = utility.Vector3dVector(stereograph_projection)
    if colors:
        sphere_cloud = geometry.PointCloud()
        sphere_cloud.points = utility.Vector3dVector(sphere_projection)
        sphere_cloud.colors = pcd.colors
        hemisphere2d.colors = utility.Vector3dVector(
            np.asarray(sphere_cloud.colors)[
                np.asarray(sphere_cloud.points)[:, 2] > annotation.coords[2]
            ]
        )

    # Calculate % cover with point grid distance by projecting
    # onto 2D circle point grid (pointcloud)
    circle_points = generate_filled_circle(
        annotation.coords, radius, settings.CANOPY_COVER_POINT_SPACING
    )
    uncovered_cloud = geometry.PointCloud()
    uncovered_cloud.points = utility.Vector3dVector(circle_points)
    uncovered_cloud.paint_uniform_color([1, 0, 0])
    dist_pcds = uncovered_cloud.compute_point_cloud_distance(hemisphere2d)
    dist_pcds = np.asarray(dist_pcds)
    uncovered_id = np.where(dist_pcds > settings.CANOPY_COVER_POINT_SPACING)[0]
    uncovered_cloud = uncovered_cloud.select_by_index(uncovered_id)
    CR = (len(circle_points) - len(uncovered_id)) / len(circle_points)

    # Visualize
    visualizations.capture_geoms_to_file(
        [hemisphere2d, uncovered_cloud],
        "{0}/{1}_cover.png".format(settings.OUTPUT_FOLDER, annotation.id),
    )

    return CR


def create_mesh_ball_pivot(pcd):
    """
    Create a mesh from a point cloud through the ball pivot method (geometry.PointCloud)
    """
    avg_point_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    # Create a list of radii to use for the ball pivot method
    radii = [1.5 * avg_point_dist * i for i in range(1, 10)]
    # Create a mesh from the point cloud using the ball pivot method
    return geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, utility.DoubleVector(radii)
    )


def get_largest_cluster(mesh):
    """
    Return the largest cluster from a mesh (geometry.TriangleMesh)
    """
    clust_idx, clust_n_tria, _ = get_cluster_triangles(mesh)
    largest_cluster_idx = clust_n_tria.argmax()
    triangles_to_remove = clust_idx != largest_cluster_idx
    # Create a new mesh retaining only the largest cluster
    import copy

    largest_mesh: cpu.pybind.geometry.TriangleMesh = copy.deepcopy(mesh)
    largest_mesh.remove_triangles_by_mask(triangles_to_remove)
    return largest_mesh


def get_cluster_triangles(mesh):
    """
    Cluster connected triangles, and return the cluster indices, the number
    of triangles in each cluster and the area of each cluster
    """
    clust_idx, clust_n_tria, clust_area = mesh.cluster_connected_triangles()
    return map(np.asarray, (clust_idx, clust_n_tria, clust_area))


def get_3D_area(mesh):
    """
    Calculate the 3D area of a mesh (geometry.TriangleMesh)
    """
    return mesh.get_surface_area()


def get_2D_area(pcd, alpha="optimize", vis=False):
    """
    Calculate the 2D area when rotating point cloud
    to best fit plane with the xy axis (geometry.PointCloud)
    """
    homogeneous_coords = np.hstack([pcd.points, np.ones((pcd.points.shape[0], 1))])
    transform_coords = np.dot(
        homogeneous_coords, transforms.get_rotation_to_xy_plane(pcd)
    )
    points_2D = transform_coords[:, 0:2]
    if alpha == "optimize":
        alpha = optimizealpha(points_2D)

    print("alpha is", alpha)
    hull = alphashape.alphashape(points_2D, alpha)

    # Check that the hull is a valid polygon
    if hull is None or not isinstance(hull, (Polygon, MultiPolygon)):
        raise ValueError(
            "alphashape did not return a valid 2D polygon. Try adjusting alpha."
        )

    if vis:
        fig, ax = plt.subplots()
        ax.scatter(points_2D[:, 0], points_2D[:, 1], s=1)

        def draw_polygon(poly, facecolor="blue", edgecolor="black", transparency=0.5):
            # Exterior
            ext_coords = list(poly.exterior.coords)
            ax.add_patch(
                MplPolygon(
                    ext_coords,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=transparency,
                )
            )

        if isinstance(hull, Polygon):
            draw_polygon(hull)
        elif isinstance(hull, MultiPolygon):
            for poly in hull.geoms:
                draw_polygon(poly)
        plt.axis("equal")
        plt.show()

    return hull.area


def get_3d_rugosity(area_2D, area_3D):
    """
    Get 3D rugosity: ratio of 3D area to 2D area
    """
    try:
        return area_3D / area_2D
    except ZeroDivisionError:
        return None


def calc_gap_fraction(
    annotation,
    pcd,
    resolution=200,
    color_output=True,
    max_radius=None,
    output_filename=None,
    seed_points=None,
):
    """
    Calculate gap fraction based on hemispherical projection:
    ratio of "sky pixels" to "benthic pixels" in the resulting 2D image

    Args:
        annotation: Annotation object with coords attribute.
        pcd: Point cloud object.
        resolution: Image resolution (default 200).
        color_output: Whether to use color output (default True).
        max_radius: Optional maximum radius for filtering points.
        output_filename: Optional filename to save the image. If None, the image
            is not saved to file. If not provided, uses default path based on
            annotation.id and settings.OUTPUT_FOLDER.
        seed_points: Optional list of ``(x, y)`` tuples with proportional
            coordinates in ``[0, 1]`` specifying flood-fill seed locations
            (``x`` horizontal, ``y`` vertical, origin top-left). If ``None``,
            the default seeds ``(0.5, 0.5)`` (image centre) and ``(0.25, 0.5)``
            (an "upslope" point) are used. When provided, the seed points are
            drawn as markers on the output image after the gap fraction is
            calculated, so they do not affect the result and can be used to
            iteratively tune their placement.
    """

    # Translate the point cloud by the negation of the center coordinates
    # and remove points with a negative z value
    trans_points = np.asarray(pcd.points) - annotation.coords
    points_to_keep = trans_points[:, 2] > 0

    # If max_radius is defined, filter points to keep based on xy distance to annotation
    if max_radius is not None:
        xy_distances = np.linalg.norm(trans_points[:, :2], axis=1)
        points_to_keep = points_to_keep & (xy_distances <= max_radius)

    trans_points = trans_points[points_to_keep]

    # Convert to spherical coordinates
    theta = np.arctan2(trans_points[:, 1], trans_points[:, 0])
    phi = np.arccos(trans_points[:, 2] / np.linalg.norm(trans_points, axis=1))

    # Convert from spherical to normalized 2D polar coordinates,
    # scale by image resolution, and convert to pixel coordinates (integers)
    x = (((phi * np.cos(theta)) + np.pi / 2) / np.pi) * resolution
    y = (((phi * np.sin(theta)) + np.pi / 2) / np.pi) * resolution

    cover_pixels = np.stack(
        (
            np.clip(x.astype(int), 0, resolution - 1),
            np.clip(y.astype(int), 0, resolution - 1),
        ),
        axis=-1,
    )

    # Set circular imaging area (in dark gray)
    radius = resolution // 2
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    img_mask = (x - (radius)) ** 2 + (y - (radius)) ** 2 <= radius**2
    image[img_mask] = [80, 80, 80]
    img_area = np.sum(img_mask)

    # Calculate the raw cover
    raw_cover = len(np.unique(cover_pixels, axis=0))
    gapF_raw = (img_area - raw_cover) / img_area

    # Map the points(/colors) to the image pixels. Vectorized so this scales to
    # the full point cloud without a Python-level per-point loop (which made
    # this the runtime bottleneck for large clouds).
    if color_output:
        rgb_colors = (np.asarray(pcd.colors)[points_to_keep] * 255).astype(np.uint8)
        # For each pixel keep the colour of the point closest to the centre. We
        # scatter-assign in order of decreasing distance so the nearest point
        # (smallest norm) is written last and therefore wins per pixel.
        norms = np.linalg.norm(trans_points, axis=1)
        order = np.argsort(-norms, kind="stable")
        image_flat = image.reshape(-1, 3)
        flat_idx = cover_pixels[:, 0] * resolution + cover_pixels[:, 1]
        image_flat[flat_idx[order]] = rgb_colors[order]
    else:
        image[cover_pixels[:, 0], cover_pixels[:, 1]] = [255, 255, 255]

    # Apply floodFill algorithm to calculate center gap fraction.
    # By default, use a centre point and an "upslope" point; otherwise convert
    # the user-supplied proportional coordinates to pixel coordinates.
    if seed_points is None:
        seed_points_px = [(radius, radius), (radius // 2, radius)]
        visualize_seeds = False
    else:
        seed_points_px = [
            (
                int(round(sx * resolution)),
                int(round(sy * resolution)),
            )
            for sx, sy in seed_points
        ]
        visualize_seeds = True

    diff = (1, 1, 1)
    fill_color = (0, 0, 128)
    for seed_point in seed_points_px:
        retval, image, _, _ = cv2.floodFill(
            image, None, seed_point, fill_color, diff, diff
        )

    fill_pixel_count = cv2.countNonZero(cv2.inRange(image, fill_color, fill_color))
    gapF_fill = fill_pixel_count / img_area

    # Draw seed markers AFTER counting so they don't affect gapF_fill.
    # A high-contrast green cross with a thin black outline is used so the
    # markers remain visible against the dark gray sky, navy fill, and varied
    # benthic colours.
    if visualize_seeds:
        marker_size = max(6, resolution // 25)
        for seed_point in seed_points_px:
            cv2.drawMarker(
                image,
                seed_point,
                color=(0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=3,
                line_type=cv2.LINE_AA,
            )
            cv2.drawMarker(
                image,
                seed_point,
                color=(0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

    # Output the image only if output_filename is provided
    if output_filename is not None:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_filename, image_bgr)

    return gapF_raw, gapF_fill, image


def apply_vector_transform_points(points, transform_vector):
    """
    apply transformation so an array of points aligns to the tranform_vector
    TBD: should this go to transforms?
    """
    transform_matrix = transforms.get_up_vector_transform(transform_vector)
    points_hom = np.hstack((points, np.ones((len(points), 1))))
    points_hom = np.dot(points_hom, np.array(transform_matrix))
    points_transformed = points_hom[:, :-1]  # TODO: Make homogenous function?
    return points_transformed


def cast_ray_pointcloud(starting_coord, ray_vector, points, dist_threshold):
    """
    a function that mimics raycasting but with a pointcloud instead of a mesh
    the dist_threshold argument helps ignore points that are not of interest
    author: DVH
    """
    # using dot product calculate distance between the ray and the pointcloud
    t = np.dot(points - starting_coord, ray_vector)
    ray = starting_coord + t[:, np.newaxis] * ray_vector
    closest_index_intersection = np.argmin(np.linalg.norm(ray - points, axis=1))
    closest_point_along_ray = (
        np.dot(ray_vector, (points[closest_index_intersection] - starting_coord))
        / np.linalg.norm(ray_vector)
    ) * ray_vector + starting_coord
    if (
        np.sqrt(
            np.sum((closest_point_along_ray - points[closest_index_intersection]) ** 2)
        )
        <= dist_threshold
    ):
        return closest_index_intersection
    return None


def generate_point_grid(bounding_box, spacing):
    """Generate a regular horizontal grid of points above a bounding box.

    Grid points span the box in x and y at the given spacing and all share a
    single z value set just above the box top (``max_bound[2] + spacing``).

    Args:
        bounding_box: Pair ``[min_bound, max_bound]`` of 3-D corner
            coordinates, e.g. ``[[min_x, min_y, min_z], [max_x, max_y, max_z]]``.
        spacing (float): Spacing between grid points.

    Returns:
        numpy.ndarray: Array of shape ``(N, 3)`` with the flattened grid points.
    """
    min_bound, max_bound = bounding_box
    x_range = np.arange(min_bound[0], max_bound[0], spacing)
    y_range = np.arange(min_bound[1], max_bound[1], spacing)
    z_value = max_bound[2] + spacing
    xx, yy, zz = np.meshgrid(x_range, y_range, z_value, indexing="ij")
    grid_points = np.column_stack(
        (xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), z_value))
    )
    return grid_points


def get_point_intercept_grid_KdTree(pcd, spacing=1, vis=True):
    """
    Get the coordinates of points on the grid that are closest to the point cloud using KDTree.

    bounding_box = [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    """
    bounding_box = pcd.bounding_box
    grid_points = np.asarray(generate_point_grid(bounding_box, spacing))[:, :2]
    point_cloud_2d = np.array(pcd.points)[:, :2]  # Extract only x, y from point cloud
    kd_tree = KDTree(point_cloud_2d)
    distances, closest_indices = kd_tree.query(grid_points)
    closest_points_3d = np.array(pcd.points)[closest_indices]

    if vis:
        # Visualization (optional)
        pcd_copy = copy.deepcopy(pcd)
        visualizations.show_grid_points(pcd_copy, closest_indices)

    return closest_points_3d, closest_indices


def get_point_intercept_grid(pcd, proj_vector=[0, 0, 1], spacing=1, vis=True):
    """
    get the coordinates of matrix point at a certain spacing and from a certain view point.
    standard usage is 1m distance and a top-down view.

    grid z = max_z

    bounding_box = [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    author: DVH
    """
    bounding_box = pcd.bounding_box
    grid_points = generate_point_grid(bounding_box, spacing)

    if proj_vector != [0, 0, 1]:
        grid_points = apply_vector_transform_points(grid_points, proj_vector)

    # Compute intersection points for each grid point in parallel
    ray_vector = proj_vector / np.linalg.norm(proj_vector)
    max_spacing = spacing / 2  # TODO: Make this a setting?
    closest_indices = Parallel(n_jobs=-1)(
        delayed(cast_ray_pointcloud)(grid_point, ray_vector, pcd.points, max_spacing)
        for grid_point in grid_points
    )  # TODO: Centralize Parallel
    grid_point_idx = [value for value in closest_indices if value is not None]
    if vis:
        pcd_copy = copy.deepcopy(pcd)
        visualizations.show_grid_points(pcd_copy, closest_indices)

    return pcd.points[grid_point_idx]


def project_points_onto_plane(points, plane_normal):
    """
    Project 3D points onto a plane defined by its normal vector.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    projected_points = points - np.outer(np.dot(points, plane_normal), plane_normal)
    return projected_points


def generate_transect_points(markers, num_points=50):
    """Return a PointCloud containing points along a transect line
    with num_points defining the number of points per fragment"""
    transect_points = []
    for i in range(len(markers.data) - 1):
        transect_points.append(
            np.linspace(markers.coords[i], markers.coords[i + 1], num_points)
        )
    transect_points_all = np.vstack(transect_points)

    return transect_points_all


def generate_grid_transect(
    pcd, markers, spacing, distance, proj_vector=[0, 0, 1], vis=True
):
    """
    create a sampling matrix design around the transect line marked by cattle tags.
    markers = cattle tags, spacing = spacing between matrix points, distance= distance from transect
    """
    # create a "bounding box" for the markers
    bounding_box = markers.get_bounding_box()  # create a "bounding box" for the markers
    grid_points = generate_point_grid(bounding_box, spacing)

    if proj_vector != [0, 0, 1]:
        grid_points = apply_vector_transform_points(grid_points, proj_vector)
    transect = generate_transect_points(markers)
    # Filter grid points based on distance from original points
    ray_vector = proj_vector / np.linalg.norm(proj_vector)
    closest_indices = Parallel(n_jobs=-1)(
        delayed(cast_ray_pointcloud)(grid_point, ray_vector, pcd.points, spacing)
        for grid_point in grid_points
    )
    closest_indices = [index for index in closest_indices if index is not None]
    transect_sample_points = []
    transect_sample_idx = []
    proj_plane_points = project_points_onto_plane(
        pcd.points[closest_indices], proj_vector
    )  # still remove none from list
    proj_plane_transect = project_points_onto_plane(transect, proj_vector)
    for index, closest_point in enumerate(closest_indices):
        planar_distance = np.min(
            np.sqrt(
                np.sum((proj_plane_points[index] - proj_plane_transect) ** 2, axis=1)
            )
        )
        if planar_distance <= distance:
            transect_sample_points.append(pcd.points[closest_point])
            transect_sample_idx.append(closest_point)
    if vis:
        pcd_copy = copy.deepcopy(pcd)
        visualizations.show_grid_points(pcd_copy, transect_sample_idx)

    return transect_sample_points


def generate_grid_transect_KDtree(pcd, markers, spacing, distance, vis=True):
    """
    create a sampling matrix design around the transect line marked by cattle tags.
    markers = cattle tags, spacing = spacing between matrix points, distance= distance from transect
    """

    bounding_box = pcd.bounding_box
    grid_points = np.asarray(generate_point_grid(bounding_box, spacing))[:, :2]

    point_cloud_2d = np.array(pcd.points)[:, :2]
    kd_tree = KDTree(point_cloud_2d)
    distances, closest_indices = kd_tree.query(grid_points)
    mask = distances <= 0.01
    closest_indices = closest_indices[mask]
    grid_points = pcd.points[closest_indices]

    transect = generate_transect_points(markers, 50)
    transect_2d = np.array(transect)[:, :2]  # Only consider x, y
    kd_tree_transect = KDTree(transect_2d)  # KDTree of the transect line

    # Find closest distances from grid points to transect line
    distances, closest_indices_transect = kd_tree_transect.query(grid_points[:, :2])
    mask = distances <= distance  # Filter points within the distance threshold

    # Filter grid points based on distance to transect
    transect_sample_idx = closest_indices[mask]
    transect_sample_points = pcd.points[transect_sample_idx]
    if vis:
        pcd_copy = copy.deepcopy(pcd)
        if not hasattr(pcd, "o3d_pcd_tree"):
            pcd_copy.build_kd_tree()
        for point in transect_sample_idx:
            [k, idx, _] = pcd_copy.o3d_pcd_tree.search_radius_vector_3d(
                pcd.points[point], 0.1
            )
            np.asarray(pcd_copy.colors)[idx[1:], :] = [1, 0, 0]

        visualizations.show_coords_as_lines(pcd_copy, transect, Jupyter=False)

    return transect_sample_points


def get_distance_to_closest_point(points, point):
    """
    Get the distance to the closest point in a point cloud
    """
    distances = np.linalg.norm(np.asarray(points) - point, axis=1)
    return np.min(distances)


def calc_scale_factor(annotations, scalebars):
    """Calculate scale factor from annotations and scalebars.

    Example::

        scalebars = [['target 5', 'target 6', 0.500],
                     ['target 7', 'target 8', 0.499]]
    """
    scale_factors = []
    for scalebar in scalebars:
        if scalebar[0] in annotations and scalebar[1] in annotations:
            target1 = annotations[scalebar[0]].coords
            target2 = annotations[scalebar[1]].coords
            distance = np.linalg.norm(target1 - target2)
            scale_factor = scalebar[2] / distance
            scale_factors.append(scale_factor)
    return np.mean(scale_factors)


# Function to create 2D bounding boxes from a grid of points in 2D, with user-defined cell size
def create_grid_cells_from_pcd(pcd, cell_size):
    """
    author: Dennis van Hulten, (Reefscape genomics lab @California Academy of Sciences, University of Auckland)
    Create 2D bounding boxes (cells) from a grid of points in 2D based on user-defined cell size.
    :param grid_points: Array of 2D points (x, y).
    :param cell_size: Size of each cell in square meters.
    :return: List of 2D bounding boxes, each defined by its min and max corners.
    """
    pcd_points = np.asarray(pcd.points)[:, :2]

    x_min, y_min = np.min(pcd_points, axis=0)
    x_max, y_max = np.max(pcd_points, axis=0)
    nx = int(np.ceil((x_max - x_min) / cell_size))
    ny = int(np.ceil((y_max - y_min) / cell_size))

    bounding_boxes = []
    for i in range(nx):
        for j in range(ny):
            min_corner = [x_min + i * side_size, y_min + j * side_size]
            max_corner = [x_min + (i + 1) * side_size, y_min + (j + 1) * side_size]
            bounding_boxes.append((min_corner, max_corner))

    return bounding_boxes


def cells_share_edge(bbox1, bbox2, tol=1e-9):
    """
    Check if two bounding boxes share an edge.
    Each bbox is defined as ([x_min, y_min], [x_max, y_max]).
    """
    (x1_min, y1_min), (x1_max, y1_max) = bbox1
    (x2_min, y2_min), (x2_max, y2_max) = bbox2

    # Check for vertical edge sharing.
    if abs(x1_max - x2_min) < tol or abs(x2_max - x1_min) < tol:
        y_overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
        if y_overlap > tol:
            return True

    # Check for horizontal edge sharing.
    if abs(y1_max - y2_min) < tol or abs(y2_max - y1_min) < tol:
        x_overlap = min(x1_max, x2_max) - max(x1_min, x2_min)
        if x_overlap > tol:
            return True

    return False


def create_xy_grid_cells_with_spread_filter(
    pcd,
    cell_size,
    vis=True,
    vis_colors=False,
    sub_divisions=10,
    min_points_sub=1,
    min_proportion=0.5,
    require_adjacent=True,
):
    """Create 2D grid cells from a point cloud, filtered by point spread.

    Each cell is dropped if it does not exhibit a sufficient spread of points
    over the cell. Additionally, optionally filter out cells that do not belong
    to the main connected set, where connectivity is defined by sharing at least
    one edge.

    Also prints statistics:

    1. original grid squares,
    2. after spread filtering,
    3. after connected component filtering,
    4. and the total surface area of (3).
    """
    # Extract x-y coordinates and colors.
    pcd_points = np.asarray(pcd.points)[:, :2]
    pcd_colors = np.asarray(pcd.colors)

    # Overall bounds.
    x_min, y_min = np.min(pcd_points, axis=0)
    x_max, y_max = np.max(pcd_points, axis=0)

    # Number of grid cells.
    nx = int(np.ceil((x_max - x_min) / cell_size))
    ny = int(np.ceil((y_max - y_min) / cell_size))
    total_cells = nx * ny

    # Define the function that processes a single cell.
    def process_cell(idx):
        i = idx // ny
        j = idx % ny

        # Define cell bounds.
        cell_x_min = x_min + i * cell_size
        cell_y_min = y_min + j * cell_size
        cell_x_max = cell_x_min + cell_size
        cell_y_max = cell_y_min + cell_size

        # Extract points within this cell.
        in_cell = (
            (pcd_points[:, 0] >= cell_x_min)
            & (pcd_points[:, 0] < cell_x_max)
            & (pcd_points[:, 1] >= cell_y_min)
            & (pcd_points[:, 1] < cell_y_max)
        )
        cell_points = pcd_points[in_cell]

        # Subdivide the cell.
        sub_cell_size = cell_size / sub_divisions
        subcell_count = 0
        filled_subcells = 0
        for m in range(sub_divisions):
            for n in range(sub_divisions):
                sub_x_min = cell_x_min + m * sub_cell_size
                sub_y_min = cell_y_min + n * sub_cell_size
                sub_x_max = sub_x_min + sub_cell_size
                sub_y_max = sub_y_min + sub_cell_size
                subcell_count += 1

                # Count points in the subcell.
                in_subcell = np.where(
                    (cell_points[:, 0] >= sub_x_min)
                    & (cell_points[:, 0] < sub_x_max)
                    & (cell_points[:, 1] >= sub_y_min)
                    & (cell_points[:, 1] < sub_y_max)
                )[0]
                if len(in_subcell) >= min_points_sub:
                    filled_subcells += 1

        # Return the cell's bounding box if it meets the spread criterion.
        if filled_subcells / subcell_count >= min_proportion:
            return ([cell_x_min, cell_y_min], [cell_x_max, cell_y_max])
        else:
            return None

    print(
        "Creating {0}x{0}m2 grid cells with {1}x{1} subdivision.".format(
            cell_size, sub_divisions
        )
    )
    # Run the main loop in parallel with a progress bar.
    results = Parallel(n_jobs=-1)(
        delayed(process_cell)(idx)
        for idx in tqdm(range(total_cells), desc="Processing grid cells")
    )

    print(
        "Filtering for the largest group of cells that have {0} of subdivisions with at least {1} points.".format(
            min_proportion, min_points_sub
        )
    )
    # Filter out cells that did not meet the criterion.
    filtered_bboxes = [res for res in results if res is not None]

    # Save the initial filtered list.
    initial_bboxes = filtered_bboxes.copy()

    # Further filter cells to only include the largest connected set.
    if require_adjacent and filtered_bboxes:
        n = len(filtered_bboxes)
        # Build connectivity graph.
        graph = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if cells_share_edge(filtered_bboxes[i], filtered_bboxes[j]):
                    graph[i].append(j)
                    graph[j].append(i)

        # Find connected components using DFS.
        seen = set()
        components = []
        for i in range(n):
            if i not in seen:
                comp = []
                stack = [i]
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    comp.append(cur)
                    stack.extend(graph[cur])
                components.append(comp)

        # Select the largest component.
        largest = max(components, key=len)
        filtered_bboxes = [filtered_bboxes[i] for i in largest]

    # Compute statistics.
    orig_count = total_cells
    init_count = len(initial_bboxes)
    final_count = len(filtered_bboxes)
    total_area = final_count * cell_size**2

    print("Original grid squares: {}".format(orig_count))
    print("After initial filtering: {}".format(init_count))
    print("After largest-component filtering: {}".format(final_count))
    print("Total surface area: {:.3f} m²".format(total_area))

    if vis:
        if vis_colors:
            point_colors = pcd.colors
        else:
            point_colors = None
        visualizations.vis_create_xy_grid_cells_with_spread_filter(
            pcd_points, point_colors, filtered_bboxes, cell_size, sub_divisions
        )

    return filtered_bboxes


def find_optimal_box_position(pcd, box_length, box_width, step_size=0.1, vis=True):
    """
    Find the optimal rectangle (box) position from a point cloud using a convolution-based approach.

    Parameters:
      pcd       : Point cloud object with attribute `points` (Nx3 array).
      box_length: float, the rectangle's length (x-dimension).
      box_width : float, the rectangle's width (y-dimension).
      step_size : float, resolution of the grid used for the convolution.
      vis       : bool, whether to visualize the result.

    Returns:
      best_box   : tuple (x_min, y_min, x_max, y_max) of the optimal rectangle's bounding coordinates.
      best_count : int, the maximum point count found within that rectangle.
    """
    # Extract x and y coordinates.
    points = np.asarray(pcd.points)[:, :2]

    # Compute overall bounds.
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Create candidate grid edges using the step size.
    x_edges = np.arange(x_min, x_max + step_size, step_size)
    y_edges = np.arange(y_min, y_max + step_size, step_size)

    # Build a 2D histogram of the point cloud.
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])

    # Map rectangle dimensions to the number of bins.
    box_length_bins = int(np.round(box_length / step_size))
    box_width_bins = int(np.round(box_width / step_size))

    # Create a kernel corresponding to the rectangle.
    kernel = np.ones((box_length_bins, box_width_bins))

    # Convolve the histogram with the kernel.
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


def create_mesh_poisson(pcd, depth):
    """Reconstruct a triangle mesh from a point cloud via Poisson meshing.

    Runs Open3D Poisson surface reconstruction and trims the lowest-density
    vertices (bottom 10% by density) to remove poorly supported geometry.

    Args:
        pcd: A point cloud exposing an ``o3d_pcd`` Open3D point cloud with
            estimated normals.
        depth (int): Poisson reconstruction depth (octree depth); higher values
            yield finer detail.

    Returns:
        open3d.geometry.TriangleMesh: The reconstructed, density-trimmed mesh.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd.o3d_pcd, depth=depth
    )
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def get_random_stratified_points_raycast(
    pcd, cell_size, num_points, vis=True
):  # Should we exclude the meshing in this step?
    """Sample stratified random surface points via top-down raycasting.

    Builds a Poisson mesh of the point cloud, divides its extent into square
    grid cells of ``cell_size``, generates random ray origins above each cell,
    and casts downward (-z) rays onto the mesh. Each hit is snapped back to the
    nearest original point (within 0.01 units), giving stratified samples that
    lie on actual point-cloud vertices.

    Args:
        pcd: A point cloud with ``points`` and an ``o3d_pcd`` used for meshing.
        cell_size (float): Side length of the square stratification cells.
        num_points (int): Number of random rays generated per cell.
        vis (bool): If True, display the point cloud with sampled points
            highlighted. Defaults to True.

    Returns:
        tuple: ``(random_points_coords, random_points_idx)`` where the first is
        an array of the sampled point coordinates and the second is the array of
        their indices into ``pcd.points``.
    """
    mesh = create_mesh_poisson(pcd, depth=9)
    bounding_boxes = create_grid_cells_from_pcd(pcd, cell_size)
    z_max = np.max(pcd.points[:, 2]) + 10  # might be good not to hardcode this

    ray_points = generate_random_points_within_cells(bounding_boxes, num_points, z_max)
    direction_vectors = np.tile([0, 0, -1], (ray_points.shape[0], 1))
    rays = np.hstack((ray_points, direction_vectors))
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    surface = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    surface_id = scene.add_triangles(surface)
    ans = scene.cast_rays(rays)

    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    ray_pcd = o3d.t.geometry.PointCloud(points)
    points = points.numpy()

    kdtree = KDTree(np.asarray(pcd.points))
    distances, indices = kdtree.query(points)
    mask = distances <= 0.01
    random_points_idx = indices[mask]
    random_points_coords = pcd.points[random_points_idx]

    if vis:
        pcd_copy = copy.deepcopy(pcd)
        if not hasattr(pcd, "o3d_pcd_tree"):
            pcd_copy.build_kd_tree()
        for index in indices:
            [k, idx, _] = pcd_copy.o3d_pcd_tree.search_radius_vector_3d(
                pcd_copy.points[index], 0.02
            )
            np.asarray(pcd_copy.colors)[idx[1:], :] = [1, 0, 0]

        visualizations.show([pcd_copy.o3d_pcd])

    print(
        "Returning {0} out of {1} random points".format(
            len(random_points_coords), len(ray_points)
        )
    )
    return random_points_coords, random_points_idx


def get_random_stratified_points_raycast_temp(pcd, ray_points, mesh_depth=9):
    """Raycast pre-supplied ray origins onto a Poisson mesh to sample points.

    Temporary/experimental variant of ``get_random_stratified_points_raycast``.
    Instead of generating ray origins internally from stratification cells, it
    accepts caller-provided ``ray_points`` and exposes the mesh depth, and it
    performs no visualization. Rays are cast downward (-z) onto a Poisson mesh
    of the point cloud, and hits are snapped to the nearest original point
    (within 0.01 units).

    Args:
        pcd: A point cloud with ``points`` and an ``o3d_pcd`` used for meshing.
        ray_points (numpy.ndarray): Array of shape ``(N, 3)`` of ray origin
            coordinates (rays travel in the -z direction).
        mesh_depth (int): Poisson reconstruction depth. Defaults to 9.

    Returns:
        tuple: ``(random_points_coords, random_points_idx)`` where the first is
        an array of the sampled point coordinates and the second is the array of
        their indices into ``pcd.points``.
    """
    print("Creating mesh...")
    mesh = create_mesh_poisson(pcd, depth=mesh_depth)

    direction_vectors = np.tile([0, 0, -1], (ray_points.shape[0], 1))
    rays = np.hstack((ray_points, direction_vectors))
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    surface = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    surface_id = scene.add_triangles(surface)
    ans = scene.cast_rays(rays)

    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    ray_pcd = o3d.t.geometry.PointCloud(points)
    points = points.numpy()

    kdtree = KDTree(np.asarray(pcd.points))
    distances, indices = kdtree.query(points)
    mask = distances <= 0.01
    random_points_idx = indices[mask]
    random_points_coords = pcd.points[random_points_idx]

    return random_points_coords, random_points_idx


def get_random_stratified_points(pcd, cell_size, num_points_per_cell, vis=True):
    """
    Perform stratified random sampling in each 2D bounding box (cell).
    :param grid_points: Array of 2D points (x, y).
    :param bounding_boxes: List of 2D bounding boxes (min and max corners).
    :param num_points_per_cell: Number of random points to sample in each cell.
    :return: List of sampled points.
    """
    pcd_points = np.asarray(pcd.points)[:, :2]
    sampled_points = []
    bounding_boxes = create_grid_cells_from_pcd(pcd, cell_size)

    for min_corner, max_corner in bounding_boxes:
        # Get points within the bounding box (cell)
        points_in_cell_idx = np.where(
            (pcd_points[:, 0] >= min_corner[0])
            & (pcd_points[:, 0] <= max_corner[0])
            & (pcd_points[:, 1] >= min_corner[1])
            & (pcd_points[:, 1] <= max_corner[1])
        )[0]

        points_in_cell = list(pcd_points[points_in_cell_idx])

        # Randomly sample points from this cell
        if len(points_in_cell) > num_points_per_cell:
            sampled_points.extend(random.sample(points_in_cell, num_points_per_cell))
        else:
            # If not enough points, include all available points
            sampled_points.extend(points_in_cell)
    if vis:
        pcd_copy = copy.deepcopy(pcd)
        pcd_points = np.asarray(pcd_copy.points)[:, :2]
        kd_tree = KDTree(pcd_points)
        distance, closest_points = kd_tree.query(sampled_points)
        visualizations.show_grid_points(pcd_copy, closest_points)

    return np.array(sampled_points)


def get_mask_surface_area(annotation, predictor=None):
    """Calculate and return the surface area of the mask in the image."""
    if annotation.image_match:
        return annotation.image_match.get_mask_surface_area(predictor)
    else:
        return None


def get_intercept_points_using_cams(xy_coords, search_radius, pcd, cams, vis=False):
    """
    Find intercept points in a PointCloud but prioritizing points that are visible in the cameras.
    """
    # Ensure the point cloud has a KDTree for XY coordinates.
    if not hasattr(pcd, "o3d_pcd_tree_xy"):
        pcd.build_kd_tree_xy()

    no_points_within_search_radius = 0
    intercept_points = []
    non_intercept_points = []

    for xy_coord in tqdm(xy_coords):
        # Build query: flatten xy_coord by appending z=0.
        query_xy = np.array([xy_coord[0], xy_coord[1], 0.0])

        # Use the prebuilt XY kd-tree (assumed attached as pcd.o3d_pcd_tree_xy).
        [k, idx, _] = pcd.o3d_pcd_tree_xy.search_radius_vector_3d(
            query_xy, search_radius
        )
        if k == 0:
            no_points_within_search_radius += 1
            continue

        # Retrieve candidate points (using original 3D data).
        candidates = pcd.points[idx]

        # Compute the median z value.
        median_z = np.median(candidates[:, 2])
        intercept = np.array([xy_coord[0], xy_coord[1], median_z])

        # Determine the closest candidate by 3D Euclidean distance.
        distances = np.linalg.norm(candidates - intercept, axis=1)
        # Sort candidates by their distances.
        sorted_indices = np.argsort(distances)

        # Loop over each candidate (sorted by distance)
        found_intercept_point = False
        for i in sorted_indices:
            orig_coords = utils.transform_coords(
                candidates[i], np.linalg.inv(pcd.world_transform)
            )
            for cam in cams:
                x, y, depth, relevance = cam.get_pixel_coords(orig_coords)
                if x is not None:
                    obstructions = get_intercept(pcd, cam.coords, candidates[i])
                    if obstructions is None:
                        intercept_points.append(candidates[i])
                        found_intercept_point = True
                        break
            if found_intercept_point:
                break
        if not found_intercept_point:
            max_z = np.max(candidates[:, 2])
            max_z_intercept = np.array([xy_coord[0], xy_coord[1], max_z])
            max_z_distances = np.linalg.norm(candidates - max_z_intercept, axis=1)
            non_intercept_points.append(candidates[np.argmin(max_z_distances)])

    print("Total queried points:", len(xy_coords))
    print("No points within search radius:", no_points_within_search_radius)
    print(
        "No unobstructed image for any of the points:",
        len(xy_coords) - len(intercept_points) - no_points_within_search_radius,
    )
    print("Points with image match:", len(intercept_points))
    print(
        "Proportion of points with image match:", len(intercept_points) / len(xy_coords)
    )

    return intercept_points, non_intercept_points


def slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between unit vectors u and v
    at fraction t in [0,1].
    """
    dot_uv = np.dot(u, v)
    dot_uv = np.clip(dot_uv, -1.0, 1.0)
    angle = np.arccos(dot_uv)

    # If angle ~ 0, vectors are almost identical → no arc.
    if angle < 1e-8:
        return u
    # If angle ~ pi, vectors are nearly opposite → fallback to linear
    # to avoid singularities.
    elif abs(angle - np.pi) < 1e-8:
        return (1.0 - t) * u + t * v

    # Normal slerp
    return (np.sin((1.0 - t) * angle) / np.sin(angle)) * u + (
        np.sin(t * angle) / np.sin(angle)
    ) * v


@dataclass
class DepthRegressionResult:
    """Result of fitting a linear regression of depth against 3-D position.

    Produced by ``fit_depth_regression``, which models depth as a linear
    function of ``(x, y, z)``.
    """

    #: Regression coefficient vector, sign-adjusted so that stepping along it
    #: decreases depth (points "up").
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


def fit_depth_regression(
    points: np.ndarray, depths: np.ndarray
) -> DepthRegressionResult:
    """
    Fit linear regression: depth ≈ intercept + coef · (x, y, z).
    Returns up-vector (coef; sign adjusted so stepping along it decreases depth),
    offset, per-unit depth change, and error metrics with predictions/residuals.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")
    if depths.ndim != 1 or depths.shape[0] != points.shape[0]:
        raise ValueError("depths must be shape (N,) and align with points")
    if len(points) < 2:
        raise ValueError("Need at least 2 points for regression")

    model = LinearRegression()
    model.fit(points, depths)
    coef = model.coef_.astype(float)
    depth_offset = float(model.intercept_)
    depth_per_unit = float(np.linalg.norm(coef))

    # Evaluate sign: flip so that stepping along coef decreases depth
    centroid = points.mean(axis=0)
    depth_centroid = float(model.predict([centroid])[0])
    p_step = centroid + 1.0 * coef
    depth_step = float(model.predict([p_step])[0])
    if depth_step < depth_centroid:
        coef = -coef

    depths_pred = model.predict(points).astype(float)
    depths_res = depths.astype(float) - depths_pred

    mse = float(mean_squared_error(depths, depths_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(depths, depths_pred))
    r2 = float(r2_score(depths, depths_pred))

    return DepthRegressionResult(
        up_vector=coef,
        depth_offset=depth_offset,
        depth_per_unit=depth_per_unit,
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        depths_pred=depths_pred,
        depths_res=depths_res,
    )


class DepthResidualAnalyzer:
    """Analyzes depth residuals for any container with depth data.

    Works with both Cameras and Annotations containers that have:
    - Items with depth_sensor_m and coords/orig_coords attributes
    - Parent container with up_vector and depth_offset attributes
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
        else:
            # For Annotations, prefer orig_coords (preserves original values used in regression)
            # Fall back to coords if orig_coords not available
            coords = getattr(item, "orig_coords", None)
            if coords is None:
                coords = getattr(item, "coords", None)

        if coords is None:
            return None

        return float(parent.depth_offset + float(np.dot(parent.up_vector, coords)))

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
        depths = [item.depth_sensor_m for item in items]
        est_depths = [self._calculate_estimated_depth(item) for item in items]

        # Filter out None values from est_depths
        valid_indices = [i for i, ed in enumerate(est_depths) if ed is not None]
        depths = [depths[i] for i in valid_indices]
        est_depths = [est_depths[i] for i in valid_indices]
        items = [items[i] for i in valid_indices]

        # Create filtered container
        filtered = self.container._empty_like()
        for item in items:
            item_id = getattr(item, "cam_id", None) or getattr(item, "id", None)
            if item_id:
                filtered.data[item_id] = item
                item.parent = filtered

        return depths, est_depths, filtered

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
        depths = [item.depth_sensor_m for item in items]
        z_coords = [item.coords[2] for item in items]

        # Create filtered container
        filtered = self.container._empty_like()
        for item in items:
            item_id = getattr(item, "cam_id", None) or getattr(item, "id", None)
            if item_id:
                filtered.data[item_id] = item
                item.parent = filtered

        return depths, z_coords, filtered

    def show_depth_vs_est_depth_residuals(self, width=15, height=5, **kwargs):
        """Show residuals between predicted and recorded depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            **kwargs: Additional arguments passed to get_depths_and_estimated_depths

        Returns:
            tuple: (fig1, fig2) matplotlib figure objects
        """
        from substrata import visualizations

        depths, est_depths, filtered = self.get_depths_and_estimated_depths(**kwargs)
        fig1 = visualizations.plot_depth_regression(
            depths,
            est_depths,
            width=width,
            height=height,
            title="Depth vs Estimated Depth",
        )

        # Use appropriate visualization function based on container type
        container_type = type(self.container).__name__
        if container_type == "Cameras":
            fig2 = visualizations.plot_cam_residuals(
                filtered, depths, est_depths, width=width, height=height
            )
        else:
            # For annotations, use a simpler residual plot
            # TODO: Create plot_annotation_residuals if needed
            fig2 = visualizations.plot_depth_regression(
                depths,
                est_depths,
                width=width,
                height=height,
                title="Depth Residuals",
            )
        return fig1, fig2

    def show_z_vs_depth_residuals(self, width=15, height=5, **kwargs):
        """Show residuals between z-coordinates and recorded depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            **kwargs: Additional arguments passed to get_depths_and_z_coords

        Returns:
            tuple: (fig1, fig2) matplotlib figure objects
        """
        from substrata import visualizations

        depths, z_coords, filtered = self.get_depths_and_z_coords(**kwargs)
        fig1 = visualizations.plot_depth_regression(
            depths, z_coords, width=width, height=height, title="Depth vs Z-Coordinate"
        )

        # Use appropriate visualization function based on container type
        container_type = type(self.container).__name__
        if container_type == "Cameras":
            fig2 = visualizations.plot_cam_residuals(
                filtered, depths, z_coords, width=width, height=height
            )
        else:
            # For annotations, use a simpler residual plot
            fig2 = visualizations.plot_depth_regression(
                depths,
                z_coords,
                width=width,
                height=height,
                title="Z-Coordinate Residuals",
            )
        return fig1, fig2

    def save_depth_residuals_pdf(self, filepath=None, width=15, height=5, **kwargs):
        """Save depth residuals visualization as a PDF.

        Args:
            filepath (str, optional): Path to save the PDF file. If None, generates
                a default filename.
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            **kwargs: Additional arguments passed to residual methods

        Returns:
            str: The filepath where the PDF was saved.
        """
        import matplotlib
        from matplotlib.backends.backend_pdf import PdfPages
        import os

        backend_original = matplotlib.get_backend()
        matplotlib.use("Agg", force=True)
        try:
            if filepath is None:
                # Generate default filename
                container_type = type(self.container).__name__.lower()
                if (
                    hasattr(self.container, "cams_meta_filepath")
                    and self.container.cams_meta_filepath
                ):
                    base, _ = os.path.splitext(self.container.cams_meta_filepath)
                    filepath = f"{base}_depth_residuals.pdf"
                elif (
                    hasattr(self.container, "markers_filepath")
                    and self.container.markers_filepath
                ):
                    base, _ = os.path.splitext(self.container.markers_filepath)
                    filepath = f"{base}_depth_residuals.pdf"
                else:
                    filepath = f"{container_type}_depth_residuals.pdf"

            # Get the figures
            fig1, fig2 = self.show_depth_vs_est_depth_residuals(
                width=width, height=height, **kwargs
            )
            fig3, fig4 = self.show_z_vs_depth_residuals(
                width=width, height=height, **kwargs
            )

            # Save all figures to PDF
            pdf = PdfPages(filepath)
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            pdf.savefig(fig4)
            pdf.close()

            # Close figures to free memory
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            plt.close(fig4)

            return filepath
        finally:
            matplotlib.use(backend_original, force=True)
