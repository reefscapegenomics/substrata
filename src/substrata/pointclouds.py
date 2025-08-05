# Standard Library
import logging
import os
import re
from typing import List, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np
from open3d import geometry, io, utility
from tqdm import tqdm
import numpy.typing as npt
from collections.abc import Iterable

# Local Modules
from unicorn import transforms, annotations, utils, settings, geometry

logger = logging.getLogger(__name__)


class PointCloud:
    """PointCloud decorator class for the Open3D PointCloud object.

    Points/normals/colors can be directly addressed, but if an
    `open3d.geometry.PointCloud` is required (e.g. for visualization)
    then the o3d_pcd attribute should be used.

    Attributes:
        o3d_pcd: Open3D PointCloud object
        world_transform: cumulative transformation matrix
        transforms: list of all applied transformations
        filepath: path to the pointcloud file
        points: point coordinates (refers to o3d_pcd.points)
        normals: point normals (refers to o3d_pcd.normals)
        colors: point colors (refers to o3d_pcd.colors)
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        """Initialize a PointCloud object.

        Args:
            filepath: Optional path to a point cloud file to load.
        """
        self.o3d_pcd = geometry.PointCloud()
        self.name: Optional[str] = None
        self.world_transform: np.ndarray = np.eye(4)  # identity matrix
        self.transforms: List[np.ndarray] = []
        if filepath:
            self.read_point_cloud(filepath)
            self.filepath = filepath
        else:
            self.filepath = None

    @property
    def points(self) -> np.ndarray:
        """Get point coordinates as numpy array.

        Returns:
            Point coordinates as numpy array.
        """
        return np.asarray(self.o3d_pcd.points)

    @property
    def normals(self) -> np.ndarray:
        """Get point normals as numpy array.

        Returns:
            Point normals as numpy array.
        """
        return np.asarray(self.o3d_pcd.normals)

    @property
    def colors(self) -> np.ndarray:
        """Get point colors as numpy array.

        Returns:
            Point colors as numpy array.
        """
        return np.asarray(self.o3d_pcd.colors)

    @property
    def simple_pcd(self) -> "SimplePointCloud":
        """Get a SimplePointCloud representation.

        Returns:
            SimplePointCloud object with current points, colors, normals, and labels.
        """
        labels = np.full((len(np.asarray(self.points)), 1), None, dtype=object)
        return SimplePointCloud(self.points, self.colors, self.normals, labels)

    @property
    def bounding_box(self) -> List[np.ndarray]:
        """Get the axis-aligned bounding box of the point cloud.

        Returns:
            List containing minimum and maximum bounds as numpy arrays.
        """
        bounding_box = self.o3d_pcd.get_axis_aligned_bounding_box()
        return [bounding_box.get_min_bound(), bounding_box.get_max_bound()]

    def read_point_cloud(self, filepath: str) -> None:
        """Read a pointcloud directly from a file.

        Args:
            filepath: Path to the point cloud file to load.
        """
        logger.info("Loading in pointcloud {}...".format(filepath))
        self.filepath = filepath
        self.o3d_pcd = io.read_point_cloud(self.filepath, print_progress=True)
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        # Remove any "_dec" followed by any characters at the end
        # (e.g., _dec7M, _dec50M)
        self.name = re.sub(r"_dec.*$", "", base_name)

    def populate_point_cloud(
        self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray
    ) -> None:
        """Populate the point cloud with points, normals, and colors.

        Args:
            points: Point coordinates as numpy array.
            normals: Point normals as numpy array.
            colors: Point colors as numpy array.
        """
        self.o3d_pcd.points = points
        self.o3d_pcd.normals = normals
        self.o3d_pcd.colors = colors

    def build_kd_tree(self) -> None:
        """Build kd tree for the point cloud."""
        logger.info("Building kdtree for {}...".format(self.filepath))
        self.o3d_pcd_tree = geometry.KDTreeFlann(self.o3d_pcd)

    def build_kd_tree_xy(self) -> None:
        """Build kd tree for the XY coordinates of the point cloud."""
        logger.info("Building kdtree for XY coords of {}...".format(self.filepath))
        pcd_xy = geometry.PointCloud()
        xy_points = np.hstack((self.points[:, :2], np.zeros((self.points.shape[0], 1))))
        pcd_xy.points = utility.Vector3dVector(xy_points)
        self.o3d_pcd_tree_xy = geometry.KDTreeFlann(pcd_xy)

    def apply_transform(self, transform_matrix: npt.NDArray[np.floating]) -> None:
        """Apply a 4x4 homogeneous transform to the point cloud."""
        tm = np.asarray(transform_matrix, dtype=float)
        if tm.shape != (4, 4):
            raise ValueError("`transform_matrix` must have shape (4, 4)")

        self.o3d_pcd.transform(tm)
        logger.debug("Applied transform:\n%s", tm)

        self.transforms.append(tm)
        self.world_transform = tm @ self.world_transform

    def apply_transforms(
        self, transform_matrices: Iterable[npt.NDArray[np.floating]]
    ) -> None:
        """Apply several 4x4 transforms in sequence."""
        for idx, tm in enumerate(transform_matrices):
            logger.debug("Applying transform %d", idx)
            self.apply_transform(tm)

    def apply_orientation_transforms(
        self,
        scale_factor: float,
        up_vector: np.ndarray,
        depth_offset: float,
        z_axis_rotation: bool = False,
    ) -> None:
        """Apply a standardized set of orientations at once.

        Args:
            scale_factor: Scale factor to apply.
            up_vector: Up vector for orientation.
            depth_offset: Depth offset to apply.
            z_axis_rotation: Whether to apply z-axis rotation.
        """
        self.apply_transform(transforms.get_scale_transform(scale_factor))
        self.apply_transform(transforms.get_up_vector_transform(up_vector))
        self.apply_transform(transforms.get_depth_offset_transform(depth_offset))
        self.apply_transform(transforms.get_x_axis_to_eigenvector_transform(self))
        self.apply_transform(transforms.get_y_axis_facing_upslope_transform(self))
        if z_axis_rotation:
            self.apply_transform(transforms.get_rotate_transform("z", np.radians(180)))
        self.apply_transform(transforms.get_positive_xy_transform(self))

    def reduce_pcd_to_points_in_mesh(self, mesh) -> None:
        """Reduce the number of points by sampling points from a target mesh.

        Args:
            mesh: Target mesh to sample points from.
        """
        n_points = len(self.o3d_pcd.points)
        reduced_pcd = mesh.sample_points_uniformly(n_points)
        reduced_pcd = mesh.sample_points_poisson_disk(n_points, pcl=reduced_pcd)
        self.o3d_pcd = reduced_pcd

    def subsample_pointcloud_by_radius(
        self, coords: np.ndarray, radius: float
    ) -> "SimplePointCloud":
        """Subsample the pointcloud by a given radius.

        Args:
            coords: Coordinates to search around.
            radius: Search radius.

        Returns:
            SimplePointCloud with subsampled points.

        Raises:
            RuntimeError: If the pointcloud has <=1 points.
        """
        if not hasattr(self, "o3d_pcd_tree"):
            self.build_kd_tree()

        # Query all points within `radius` of annotation
        [k, idx, _] = self.o3d_pcd_tree.search_radius_vector_3d(coords, radius)

        if len(self.points) <= 1:
            logger.error(
                "Pointcloud for annotation id {} has <=1 points".format(self.id)
            )

        # Return a new pointcloud with the subsampled points
        return SimplePointCloud(
            np.asarray(self.points)[idx[1:], :],
            np.asarray(self.colors)[idx[1:], :],
            np.asarray(self.normals)[idx[1:], :],
            np.asarray(np.repeat("empty", len(np.asarray(self.points)[idx[1:], :]))),
        )

    def get_cam_dist(
        self, cam, beam_angle: float, return_pcd: bool = False
    ) -> Union[float, Tuple[float, geometry.PointCloud]]:
        """Calculate the average distance from the camera to all points within a given beam angle.

        Args:
            cam: Camera object with coords and vector attributes.
            beam_angle: Beam angle in degrees.
            return_pcd: Whether to return a filtered point cloud.

        Returns:
            Average distance, or tuple of (average_distance, filtered_pointcloud) if return_pcd is True.
        """
        cam_coord = cam.coords
        cam_vector = cam.vector / np.linalg.norm(cam.vector)  # Normalize camera vector
        vectors_to_points = self.points - cam_coord
        distances = np.linalg.norm(vectors_to_points, axis=1)
        vectors_to_points_norm = vectors_to_points / distances[:, np.newaxis]

        # Calculate the cosine of the angles between the camera vector and vectors to points
        cos_theta = np.dot(vectors_to_points_norm, cam_vector)
        cos_theta = np.clip(
            cos_theta, -1.0, 1.0
        )  # Ensure values are within valid range
        angles = np.degrees(np.arccos(cos_theta))

        # Mask points that are within the beam angle
        within_beam = angles <= beam_angle

        filtered_distances = distances[within_beam]

        if len(filtered_distances) == 0:
            if return_pcd:
                return 0.0, geometry.PointCloud()
            else:
                return 0.0
        elif return_pcd:
            filtered_points = self.points[within_beam]
            # Create an Open3D PointCloud object for the filtered points
            filtered_pcd = geometry.PointCloud()
            filtered_pcd.points = utility.Vector3dVector(filtered_points)
            # Set the color of the filtered points to red
            red_color = np.array([[1.0, 0.0, 0.0] for _ in range(len(filtered_points))])
            filtered_pcd.colors = utility.Vector3dVector(red_color)
            return np.mean(filtered_distances), filtered_pcd

        return np.mean(filtered_distances)

    def get_z_intercepts(
        self,
        xy_coords: np.ndarray,
        search_radius: float,
        always_return: bool = False,
        store_neighboring_coords: bool = False,
    ) -> annotations.Annotations:
        """Find intercept points for a list of XY coordinates by searching within a specified radius.

        Args:
            xy_coords: Array of XY coordinates to find intercepts for.
            search_radius: Search radius for finding intercept points.
            always_return: Whether to always return a result, extrapolating if necessary.
            store_neighboring_coords: Whether to store neighboring coordinates for visualization.

        Returns:
            Annotations object containing the found intercept points.
        """
        # Iterate through all XY points and find the corresponding 3D intercept points
        intercept_points = annotations.Annotations()
        for idx, xy_point in enumerate(tqdm(xy_coords)):
            intercept_point = self.get_z_intercept(
                xy_point, search_radius, always_return, store_neighboring_coords
            )
            intercept_point.id = idx
            if intercept_point is not None:
                intercept_points.append(intercept_point)

        # Provide a short output summary
        print(
            "Intercept points found: {}/{}".format(
                len(intercept_points), len(xy_coords)
            )
        )
        if always_return:
            extrapolated_count = sum(
                1 for point in intercept_points if point.is_extrapolated
            )
            print("Extrapolated points with no match: {}".format(extrapolated_count))
        return intercept_points

    def get_z_intercept(
        self,
        xy_coord: np.ndarray,
        search_radius: float,
        always_return: bool = False,
        store_neighboring_coords: bool = False,
    ) -> Optional[annotations.InterceptAnnotation]:
        """Find intercepting point in a PointCloud.

        The method finds all points within the search radius (using only XY information),
        computes the median Z value of those points, and uses a 3D nearest neighbor search
        (restricted to the candidate set) to find the closest point to the computed intercept.

        If always_return is True, the function will always return a coordinate. It does
        this by increasing the search radius until a point(s) are found, and then
        returning the coordinate (xy_point[0], xy_point[1], median_z).

        Args:
            xy_coord: XY coordinates to find intercept for.
            search_radius: Search radius for finding intercept points.
            always_return: Whether to always return a result, extrapolating if necessary.
            store_neighboring_coords: Whether to store neighboring coordinates for visualization.

        Returns:
            InterceptAnnotation object if found, None otherwise.
        """
        # Ensure the point cloud has a KDTree for XY coordinates.
        used_search_radius = search_radius
        if not hasattr(self, "o3d_pcd_tree_xy"):
            self.build_kd_tree_xy()

        # Prepare the query point with z=0.``
        query_xy = np.array([xy_coord[0], xy_coord[1], 0.0])

        # Search for neighbors adjusting search_radius if necessary.
        used_search_radius = search_radius
        while True:
            [k, idx, _] = self.o3d_pcd_tree_xy.search_radius_vector_3d(
                query_xy, used_search_radius
            )
            if k == 0:
                if always_return:
                    # Increase the search radius and try again.
                    used_search_radius += search_radius
                    continue
                else:
                    return None
            break  # Found at least one point

        # Retrieve candidate points (using original 3D data).
        candidates = self.points[idx]

        # Compute the median z value.
        median_z = np.median(candidates[:, 2])
        estimated_intercept = np.array([xy_coord[0], xy_coord[1], median_z])

        # Store neighboring coordinates if requested (for visualization).
        if store_neighboring_coords:
            neighboring_coords = candidates
        else:
            neighboring_coords = None

        if used_search_radius != search_radius:
            # Return the estimated intercept if the search radius was increased.
            # So that x/y values are not affected by the increased search radius.
            return annotations.InterceptAnnotation(
                estimated_intercept,
                used_search_radius,
                is_extrapolated=True,
                estimated_intercept_coords=estimated_intercept,
                neighboring_coords=neighboring_coords,
            )

        else:
            # Determine the closest candidate by 3D Euclidean distance.
            # This allows x/y values to deviate within the original search radius.
            distances = np.linalg.norm(candidates - estimated_intercept, axis=1)
            closest_idx = np.argmin(distances)

            return annotations.InterceptAnnotation(
                candidates[closest_idx],
                used_search_radius,
                is_extrapolated=False,
                estimated_intercept_coords=estimated_intercept,
                neighboring_coords=neighboring_coords,
            )

    def get_intercept(
        self,
        origin_coord: np.ndarray,
        search_radius: float,
        target_coord: Optional[np.ndarray] = None,
        vector: Optional[np.ndarray] = None,
        max_dist_from_origin: float = settings.MAX_DIST_FROM_ORIGIN_FOR_INTERCEPT_SEARCH,
        max_search_radius: float = settings.MAX_SEARCH_RADIUS_FOR_INTERCEPT_SEARCH,
        always_return: bool = False,
    ) -> Optional[annotations.InterceptAnnotation]:
        """Find the first point that intersects with a line segment.

        The search is performed by discretizing the line segment and querying the nearest
        neighbor at each step, with an intercept being returned if the distance is less
        than the search radius.

        If always_return is True, the function will keep expanding the search until
        a closest neighbor is found.

        Args:
            origin_coord: Starting coordinate for the line segment.
            search_radius: Search radius for finding intercept points.
            target_coord: End coordinate for the line segment (mutually exclusive with vector).
            vector: Direction vector for the line segment (mutually exclusive with target_coord).
            max_dist_from_origin: Maximum distance from origin for intercept search.
            max_search_radius: Maximum search radius when always_return is True.
            always_return: Whether to always return a result, expanding search if necessary.

        Returns:
            InterceptAnnotation object if found, None otherwise.

        Raises:
            ValueError: If neither target_coord nor vector is provided, or if both are provided.
        """
        # Ensure that either a target_coord or a vector is provided
        if (target_coord is None and vector is None) or (
            target_coord is not None and vector is not None
        ):
            raise ValueError("Specify either a target_coord or vector.")

        # Ensure the pointcloud has a KDTree
        if not hasattr(self, "o3d_pcd_tree"):
            self.build_kd_tree()

        # Calculate endpoint for intercept search
        if target_coord is None:
            unit_vector = vector / np.linalg.norm(vector)
            target_coord = origin_coord + (unit_vector * max_dist_from_origin)

        # Discretize the line segment between start_coord and target_coord
        # using intercept_radius as step distance
        sample_points = utils.sample_points_by_dist(
            origin_coord, target_coord, search_radius
        )
        sample_point_neighbors = []
        search_radius_sq = search_radius**2

        # For each step, query the nearest neighbor to sample_point
        for sample_point in sample_points:
            # k=1 as we only need the nearest neighbor's distance
            k, idxs, dists = self.o3d_pcd_tree.search_knn_vector_3d(sample_point, 1)
            sample_point_neighbors.append([k, idxs, dists])
            if k > 0 and dists[0] <= search_radius_sq:
                return annotations.InterceptAnnotation(
                    self.points[idxs[0]], search_radius
                )

        # If no intercept is found and always_return evaluate an ever increasing
        # search_radius until an intercept is found.
        if always_return:
            used_search_radius = search_radius

            while used_search_radius <= max_search_radius:
                used_search_radius += search_radius
                search_radius_sq = used_search_radius**2

                for k, idxs, dists in sample_point_neighbors:
                    if k > 0 and dists[0] <= search_radius_sq:
                        return annotations.InterceptAnnotation(
                            self.points[idxs[0]], used_search_radius
                        )

        return None  # No intercept found

    def principal_axis(self, plane: str = "xy") -> "Vector":
        """Return the first PCA eigen-vector projected into the chosen plane.

        Args:
            plane: One of {"xy", "xz", "yz"} specifying the plane in which the
                   vector is projected.

        Returns:
            geometry.Vector: Unit-length 4-vector lying in the requested plane.

        Raises:
            ValueError: If *plane* is not "xy", "xz", or "yz".
        """
        from substrata import measurements  # late import to avoid cycle

        eig_vals, eig_vecs = measurements.conduct_PCA(self)

        v = eig_vecs[0]  # dominant 3-D eigen-vector

        plane = plane.lower()
        if plane == "xy":
            v_proj = [v[0], v[1], 0.0]
        elif plane == "xz":
            v_proj = [v[0], 0.0, v[2]]
        elif plane == "yz":
            v_proj = [0.0, v[1], v[2]]
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")

        return geometry.Vector(v_proj)

    def principal_axis_xy_2D(self) -> "Vector":
        """Return the dominant PCA eigen-vector in XY-space.

        The method:

        1.  Runs a 2-D PCA on the cloud’s X/Y coordinates
            (`measurements.conduct_xy_PCA`, which should return
            ``eigenvalues, eigenvectors`` like the 3-D version).
        2.  Takes the first eigen-vector *(v_x, v_y)*, normalises it, and
            returns it as a `geometry.Vector` lying in the XY-plane
            (Z-component set to 0).

        Returns
        -------
        geometry.Vector
            Unit-length 4-vector `[v_x, v_y, 0, 1]` representing the
            dominant horizontal direction.
        """
        from substrata import measurements  # late import to avoid cycle

        eig_vals, eig_vecs = measurements.conduct_xy_PCA(self)
        vx, vy = eig_vecs[:, 0]  # principal 2-D axis
        vec = np.array([vx, vy, 0.0])
        vec /= np.linalg.norm(vec)

        return geometry.Vector(vec)


class SimplePointCloud:
    """Simple point cloud class for storing points, colors and normals."""

    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a SimplePointCloud object.

        Args:
            points: Point coordinates as numpy array.
            colors: Point colors as numpy array.
            normals: Point normals as numpy array.
            labels: Point labels as numpy array.
        """
        self.points = np.asarray(points)
        if colors is not None:
            self.colors = np.asarray(colors)
        if normals is not None:
            self.normals = np.asarray(normals)
        if labels is not None:
            self.labels = np.asarray(labels)

    def get_o3d_pcd(self) -> geometry.PointCloud:
        """Convert to Open3D PointCloud object.

        Returns:
            Open3D PointCloud object with current points, colors, and normals.
        """
        o3d_pcd = geometry.PointCloud()
        o3d_pcd.points = utility.Vector3dVector(self.points)
        o3d_pcd.colors = utility.Vector3dVector(self.colors)
        o3d_pcd.normals = utility.Vector3dVector(self.normals)
        return o3d_pcd
