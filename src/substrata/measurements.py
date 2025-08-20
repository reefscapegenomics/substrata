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
from substrata import cameras, pointclouds, visualizations, settings, geometry


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


def best_fit_plane_normal(pcd) -> geometry.Vector:
    """Return a unit normal (Vector) of the PCA best-fit plane."""
    a, b, c, _ = conduct_PCA(pcd)[:4]
    return geometry.Vector([a, b, c, 0])  # promote to 4-vector


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


def get_plane_angles(pcd, vis=False) -> tuple[float, float, float, list, float | None]:
    """
    Calculate the orientation angles of a best-fit plane from a point cloud
    using PCA.

    The plane is assumed to have an equation: a*x + b*y + c*z + d = 0.
    Angles:
      - theta: angle between the x-z projection of the plane normal
               and the z-axis (rotation about y-axis),
      - psi: angle between the y-z projection of the plane normal
             and the z-axis (rotation about x-axis),
      - elevation: angle between the plane normal and [0, 0, 1].

    author: KP / PB
    """

    if len(pcd.points) <= 1:
        return 0.0, 0.0, 0.0, [], None

    a, b, c, d, inliers_idx = get_best_fit_plane_PCA(
        pcd
    )  # get_best_fit_plane_ransac(pcd)

    if abs(c) < 1e-8:
        raise ValueError(
            "Coefficient c is too close to zero; cannot compute slopes reliably."
        )

    slope_xz = a / c
    slope_yz = b / c
    theta = np.arctan(slope_xz)  # rotation about y-axis
    psi = np.arctan(slope_yz)  # rotation about x-axis

    plane_normal = np.array([a, b, c])
    mag_plane = np.linalg.norm(plane_normal)
    plane_normal_unit = plane_normal / mag_plane

    proj_xy = plane_normal_unit[:2]
    norm_xy = np.linalg.norm(proj_xy)
    if norm_xy < 1e-8:
        azimuth_deg = None  # the surface is horizontal
    else:
        proj_xy_unit = proj_xy / norm_xy
        # Azimuth: angle from (positive y axis) in clockwise direction
        azimuth_rad = np.arctan2(proj_xy_unit[0], proj_xy_unit[1])
        azimuth_deg = (np.degrees(azimuth_rad) + 360) % 360  # Normalise to [0, 360]

    vertical_normal = np.array([0, 0, 1])
    dot_val = np.dot(plane_normal_unit, vertical_normal)
    dot_val = np.clip(dot_val, -1.0, 1.0)
    elevation = np.arccos(dot_val)
    elev_deg = np.degrees(elevation)

    if vis:
        visualizations.visualize_elevation_angle(pcd, [a, b, c, d], elev_deg)

    return (
        float(np.degrees(theta)),
        float(np.degrees(psi)),
        float(elev_deg),
        [a, b, c, d],
        float(azimuth_deg) if azimuth_deg is not None else None,
    )


def get_dev_rugosity(pcd):
    """
    Calculate deviation rugosity for a pointcloud (how much the points vary from the best fitting plane)

    author: DvH
    """
    [a, b, c, d] = get_best_fit_plane_PCA(pcd)[0:4]
    dist = abs(
        (np.dot(np.asarray(pcd.points), [a, b, c]) + d)
        / np.sqrt(np.sum(np.square([a, b, c])))
    )
    dev_rugosity = np.sum(dist) / len(pcd.points)
    return dev_rugosity


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
    PointCloud or Mesh). Adapted from Young et al., 2017
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
    vector_normal_dispersion = (i - R1) / (i - 1)
    return vector_normal_dispersion


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
    annotation, pcd, resolution=200, color_output=True, max_radius=None
):
    """
    Calculate gap fraction based on hemispherical projection:
    ratio of "sky pixels" to "benthic pixels" in the resulting 2D image
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

    # Apply floodFill algorithm to calculate center gap fraction
    # using a center point and upslope point
    # TODO: make this more general
    seed_points = [[radius, radius], [radius // 2, radius]]
    diff = (1, 1, 1)
    fill_color = (0, 0, 255)
    for seed_point in seed_points:
        retval, image, _, _ = cv2.floodFill(
            image, None, seed_point, fill_color, diff, diff
        )

    fill_pixel_count = cv2.countNonZero(cv2.inRange(image, fill_color, fill_color))
    gapF_fill = fill_pixel_count / img_area

    # Output the image
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        "{0}/{1}_gap_fraction.png".format(settings.OUTPUT_FOLDER, annotation.id),
        image_bgr,
    )

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
