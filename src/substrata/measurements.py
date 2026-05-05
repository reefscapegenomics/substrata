# Standard Library
import sys
import random
import copy

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


def get_plane_angles(pcd, vis=False):
    """Calculate orientation angles for the best-fit plane of a point cloud.

    The plane normal is aligned with the point cloud normals by
    ``get_best_fit_plane_PCA``, so its direction encodes whether the surface
    faces upward or downward.

    Args:
        pcd: Point cloud object with ``.points`` (and optionally ``.normals``).
        vis: If True, show an interactive 3-D visualisation of the elevation.

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


def calc_roughness(pcd):
    """
    Compute plane-detrended roughness (Ra, Rq) for a point cloud.

    This function fits a best-fitting plane to the point cloud and then
    measures how much points deviate from that plane, using distances
    perpendicular to the plane.

    Definitions
    ----------
    Ra : arithmetical mean roughness
        The mean of the absolute perpendicular distances from all points
        to the best-fitting plane (mean(|d_i|)). This is the classic
        "average height" roughness measure. Same as get_dev_rugosity.

    Rq : root mean square roughness
        The square root of the mean of squared perpendicular distances
        (sqrt(mean(d_i^2))). This is more sensitive to larger
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

    # Pass ra and rq to avoid recalculating in visualize_roughness
    image = visualizations.visualize_roughness(pcd, interactive=False, ra=ra, rq=rq)

    return ra, rq, image


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


def get_vector_dispersion(geom):
    """
    Function to get the vector normal dispersion of a geometry (either
    PointCloud or Mesh). Adapted from Young et al., 2017.

    Returns the dispersion scalar and a static visualization image (numpy array),
    same pattern as calc_roughness. Only PointCloud-like geometry is visualized;
    for TriangleMesh, the image is None.
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
        isinstance(
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
    median_red = np.median(np.asarray(pcd.colors)[:, 0])
    median_green = np.median(np.asarray(pcd.colors)[:, 1])
    median_blue = np.median(np.asarray(pcd.colors)[:, 2])
    luminance = 0.2126 * median_red + 0.7152 * median_green + 0.0722 * median_blue
    return median_red, median_green, median_blue, luminance


def generate_filled_circle(center, radius, spacing):
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

    # Set circular imaging area (in red)
    radius = resolution // 2
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    img_mask = (x - (radius)) ** 2 + (y - (radius)) ** 2 <= radius**2
    image[img_mask] = [255, 0, 0]
    img_area = np.sum(img_mask)

    # Calculate the raw cover
    raw_cover = len(np.unique(cover_pixels, axis=0))
    gapF_raw = (img_area - raw_cover) / img_area

    # Map the points(/colors) to the image pixels
    if color_output:
        rgb_colors = (np.asarray(pcd.colors)[points_to_keep] * 255).astype(np.uint8)
        # Calculate the norms to be able to determine closest points
        norms = np.linalg.norm(trans_points, axis=1)
        mapping = -np.ones((resolution, resolution), dtype=int)
        # Iterate over the points
        for i in range(len(trans_points)):
            # Update color if this mapped point is closer to the center
            mapped_id = mapping[cover_pixels[i][0], cover_pixels[i][1]]
            if mapped_id == -1 or norms[i] < norms[mapped_id]:
                mapping[cover_pixels[i][0], cover_pixels[i][1]] = i
                image[cover_pixels[i][0], cover_pixels[i][1]] = rgb_colors[i]
    else:
        for i in range(len(trans_points)):
            image[cover_pixels[i][0], cover_pixels[i][1]] = [255, 255, 255]

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
    fill_color = (0, 0, 255)
    for seed_point in seed_points_px:
        retval, image, _, _ = cv2.floodFill(
            image, None, seed_point, fill_color, diff, diff
        )

    fill_pixel_count = cv2.countNonZero(cv2.inRange(image, fill_color, fill_color))
    gapF_fill = fill_pixel_count / img_area

    # Draw seed markers AFTER counting so they don't affect gapF_fill.
    # A high-contrast green cross with a thin black outline is used so the
    # markers remain visible against the red sky, blue fill, and varied
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
    """ """
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
    """
    Calculate scale factor from annotations and scalebars
    e.g. scalebars = [['target 5', 'target 6', 0.500],
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
    """
    Create 2D grid cells from a point cloud and filter each cell if it does
    not exhibit a sufficient spread of points over the cell. Additionally,
    optionally filter out cells that do not belong to the main connected set,
    where connectivity is defined by sharing at least one edge.
    Also prints statistics:
      (1) original grid squares,
      (2) after spread filtering,
      (3) after connected component filtering,
      and the total surface area of (3).
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
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd.o3d_pcd, depth=depth
    )
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def get_random_stratified_points_raycast(
    pcd, cell_size, num_points, vis=True
):  # Should we exclude the meshing in this step?
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
    up_vector: np.ndarray
    depth_offset: float
    depth_per_unit: float
    mse: float
    rmse: float
    mae: float
    r2: float
    depths_pred: np.ndarray
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
