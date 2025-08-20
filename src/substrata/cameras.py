# Standard Library
import csv
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET

# Third-Party Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Local Modules
from substrata import visualizations, settings

logger = logging.getLogger(__name__)


class Cameras:
    """Container class that holds a collection of Camera objects."""

    def __init__(self, cams_meta_filepath=None, cams_xml_filepath=None):
        """Initialize the Cameras container.

        Depending on the input filepaths, the container is filled from a JSON
        metadata file and/or an XML file containing sensor parameters.

        Args:
            cams_meta_filepath (str, optional): Path to the cameras metadata file.
            cams_xml_filepath (str, optional): Path to the cameras XML file.
        """
        self.data = {}
        self.world_transform = np.eye(4)
        if cams_meta_filepath:
            self.cams_meta_filepath = cams_meta_filepath
            self.get_cams_from_file(cams_meta_filepath)
        if cams_xml_filepath:
            self.cams_xml_filepath = cams_xml_filepath
            self.get_cam_sensor_parameters_from_file(cams_xml_filepath)

    @property
    def coords(self):
        return [camera.coords for camera in self.data.values()]

    def depths(self):
        return [camera.depth for camera in self.data.values()]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        self._iter = iter(self.data.values())
        return self

    def __next__(self):
        return next(self._iter)

    def items(self):
        return self.data.items()

    def append(self, cam):
        if cam.cam_id in self.data:
            raise ValueError(f"Camera with id {cam.cam_id} already exists.")
        else:
            self.data[cam.cam_id] = cam
            self.data[cam.cam_id].parent = self
            # TO DO: any other changes (eg transforms) to be implemented on append?

    def transform_coords(self, transform_matrix):
        """Apply a transformation to all camera coordinates and their transforms.

        Args:
            transform_matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
        """
        for cam_id in self.data:
            self.data[cam_id].transform_coords(transform_matrix)
        self.world_transform = np.dot(np.array(transform_matrix), self.world_transform)

    def get_original_coords(self, transform_matrix):
        """Restore original camera coordinates and transforms.

        Args:
            transform_matrix (np.ndarray): The transformation matrix to reverse.
        """
        for cam_id in self.data:
            self.data[cam_id].reverse_transform_coords(transform_matrix)
        self.world_transform = np.dot(
            np.array(transform_matrix), self.world_transform
        )  # TODO: CHECK!

    def subset(self, length):
        cameras_subset = Cameras()
        for cam_id in list(self.data.keys())[:length]:
            cameras_subset.data[cam_id] = self.data[cam_id]
        return cameras_subset

    def subset_by_filename_prefix(self, prefix):
        """Return a subset of cameras with filenames starting with a prefix.

        Args:
            prefix (str): Prefix to filter camera IDs.

        Returns:
            Annotations: New container with matching annotations.
        """
        cameras_subset = Cameras()
        for cam in self.data.values():
            if cam.filename.startswith(prefix):
                cameras_subset.data[cam.cam_id] = cam
        return cameras_subset

    def get_cams_from_file(self, cams_meta_filepath):
        """Load cameras from a JSON file and store them in the container.

        Args:
            cams_meta_filepath (str): Path to the JSON file with camera metadata.
        """
        with open(cams_meta_filepath, "r") as f:
            data = json.load(f)
        for cam_id, cam_data in data["cameras"].items():
            if cam_data["center"] is not None:
                self.data[cam_id] = Camera(
                    self,
                    cam_id,
                    cam_data["transform"],
                    cam_data["center"],
                    cam_data["path"],
                )

    def get_cam_sensor_parameters_from_file(self, cams_xml_filepath):
        """Get sensor information from a .cams.xml file (Metashape export).

        Converts sensor parameters using information from:
        https://www.agisoft.com/forum/index.php?topic=7523.0

        Args:
            cams_xml_filepath (str): Path to the XML file with sensor parameters.
        """
        tree = ET.parse(cams_xml_filepath)
        root = tree.getroot()
        calibration = root.find('.//calibration[@type="frame"][@class="adjusted"]')
        if calibration is not None:
            self.f = float(calibration.find("f").text)
            self.cx_metashape = float(calibration.find("cx").text)
            self.cy_metashape = float(calibration.find("cy").text)
            b1_element = calibration.find("b1")
            self.b1 = float(b1_element.text) if b1_element is not None else 0.0
            b2_element = calibration.find("b1")
            self.b2 = float(b2_element.text) if b2_element is not None else 0.0
            self.k1 = float(calibration.find("k1").text)
            self.k2 = float(calibration.find("k2").text)
            self.k3 = float(calibration.find("k3").text)
            self.p1 = float(calibration.find("p1").text)
            self.p2 = float(calibration.find("p2").text)
            self.width = int(calibration.find("resolution").attrib["width"])
            self.height = int(calibration.find("resolution").attrib["height"])
            self.fx = self.f + self.b1
            self.fy = self.f
            self.cx = self.width / 2 + self.cx_metashape
            self.cy = self.height / 2 + self.cy_metashape
        else:
            sys.exit("No calibration elements found!")

    def load_camera_attributes(self, input_filepath):
        """Load camera attributes from a CSV file.

        Updates each camera in the container with attributes such as path,
        datetime, distance, and depth information.

        Args:
            input_filepath (str): Path to the CSV file with camera attributes.
        """
        not_found_counter = 0
        with open(input_filepath, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cam_id = row["cam_id"]
                if cam_id in self.data:
                    cam = self.data[cam_id]
                    cam.orig_filepath = row["path"]
                    cam.datetime = row["datetime"] if row["datetime"] else None
                    cam.camdist = row["camdist"] if row["camdist"] else None
                    cam.depth = float(row["depth"]) if row["depth"] else None
                    cam.depth_pred = (
                        float(row["depth_pred"]) if row["depth_pred"] else None
                    )
                    cam.depth_residual = (
                        float(row["depth_res"]) if row["depth_res"] else None
                    )
                else:
                    not_found_counter += 1
        if not_found_counter > 0:
            logger.warning(
                f"File had {not_found_counter} cameras that were not found..."
            )

    def save_camera_attributes(self, output_filepath):
        """Save camera attributes to a CSV file.

        Writes camera ID, path, datetime, distance, depth, predicted depth,
        and depth residual for each camera.

        Args:
            output_filepath (str): Path to the output CSV file.
        """
        with open(output_filepath, "w", newline="") as csvfile:
            fieldnames = [
                "cam_id",
                "path",
                "datetime",
                "camdist",
                "depth",
                "depth_pred",
                "depth_res",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cam in self.data.values():
                writer.writerow(
                    {
                        "cam_id": cam.cam_id,
                        "path": cam.orig_filepath,
                        "datetime": getattr(cam, "datetime", None),
                        "camdist": getattr(cam, "camdist", None),
                        "depth": getattr(cam, "depth", None),
                        "depth_pred": getattr(cam, "depth_pred", None),
                        "depth_res": getattr(cam, "depth_residual", None),
                    }
                )

    def set_filepath_replace(self, find_str, replace_str):
        """Set a find/replace pair for adjusting filepaths.

        Args:
            find_str (str): The string to search for in the filepaths.
            replace_str (str): The string to replace find_str with.
        """
        self.filepath_replace = [find_str, replace_str]

    def set_base_path(self, base_path):
        """Set a replacement base path for adjusting filepaths.

        Args:
            base_path (str): The new base path to use.
        """
        self.filepath_replace = ["", base_path]

    def get_cam_dists(self, pcd, beam_angle):
        """Calculate camera distances to a point cloud.

        For each camera, compute the distance to the given point cloud based on
        the provided beam angle.

        Args:
            pcd: A point cloud object.
            beam_angle (float): The beam angle for distance calculation.
        """
        for cam in tqdm(self.data.values(), desc="Calculating camera distances..."):
            cam.camdist = pcd.get_cam_dist(cam, beam_angle)

    def get_datetime_originals(self):
        """Retrieve DateTimeOriginal metadata from image EXIF for all cameras."""
        for cam in tqdm(
            self.data.values(), desc="Retrieving timestamps from camera files..."
        ):
            cam.datetime = cam.get_datetime_original()

    def get_camera_by_filename(self, filename):
        """Get a camera object by its filename.

        Args:
            filename (str): The filename to search for.

        Returns:
            Camera: The matching Camera object, or None if not found.
        """
        for cam in self.data.values():
            if cam.filename == filename or cam.filename == filename + ".jpg":
                return cam
        return None

    def get_time_delta_between_first_and_last_photo(self):
        """Calculate the time delta between the first and last photos.

        Returns:
            int: The difference in seconds between the first and last camera
                timestamps.
        """
        cams_with_datetime = [
            cam for cam in self.data.values() if hasattr(cam, "datetime")
        ]
        first_cam = cams_with_datetime[0]
        last_cam = cams_with_datetime[-1]
        return int(
            utils.get_unix_time(last_cam.datetime)
            - utils.get_unix_time(first_cam.datetime)
        )

    def get_up_vector_from_camera_depths(self):
        """Compute the up vector using least-squares regression on camera depths.

        Fits a linear regression between the camera 3D points and their depths to
        find the dominant depth direction. Also stores predicted depths and errors.

        Returns:
            np.ndarray: The coefficient vector representing the up vector.
        """
        cams_filtered = [
            cam
            for cam in self.data.values()
            if hasattr(cam, "depth") and hasattr(cam, "coords")
        ]
        cam_ids = [cam.cam_id for cam in cams_filtered]
        points = np.array([cam.coords for cam in cams_filtered])
        depths = np.array([cam.depth for cam in cams_filtered])
        model = LinearRegression()
        model.fit(points, depths)
        depths_predicted = model.predict(points)
        depths_residuals = depths - depths_predicted
        for i, cam_id in enumerate(cam_ids):
            self.data[cam_id].depth_pred = depths_predicted[i]
            self.data[cam_id].depth_residual = depths_residuals[i]
        mse = mean_squared_error(depths, depths_predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(depths, depths_predicted)
        r2 = r2_score(depths, depths_predicted)
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R^2): {r2}")
        return model.coef_


class Camera:
    """Class that holds information about a single camera."""

    def __init__(
        self, parent=None, cam_id=None, camera_transform=None, coords=None, path=None
    ):
        """Initialize a Camera instance.

        Args:
            parent (Cameras): The parent Cameras container.
            cam_id (str): Unique camera identifier.
            camera_transform (list or np.ndarray): The camera transform matrix.
            coords (list or np.ndarray): The camera center coordinates.
            path (str): Path to the camera image.
        """

        self.parent = parent
        self.cam_id = cam_id
        self.camera_transform = self.orig_camera_transform = camera_transform
        self.coords = self.orig_coords = coords
        self.orig_filepath = path
        self.filename = os.path.basename(path)

    @property
    def vector(self):
        """Obtain the camera vector from the transform.

        Returns:
            np.ndarray: A normalized 3D vector.
        """
        transform_matrix = np.array(self.camera_transform).reshape((4, 4))
        camera_vector = transform_matrix[:3, 2]
        camera_vector /= np.linalg.norm(camera_vector)
        return camera_vector

    @property
    def orig_vector(self):
        """Obtain the original camera vector from the original transform.

        Returns:
            np.ndarray: A normalized 3D vector.
        """
        transform_matrix = np.array(self.orig_camera_transform).reshape((4, 4))
        camera_orig_vector = transform_matrix[:3, 2]
        camera_orig_vector /= np.linalg.norm(camera_orig_vector)
        return camera_orig_vector

    @property
    def filepath(self):
        if (
            not hasattr(self.parent, "filepath_replace")
            or not self.parent.filepath_replace
        ):
            return self.orig_filepath
        else:
            return self._get_updated_filepath()

    def transform_coords(self, transform_matrix):
        """Apply a transformation to the camera coordinates and transform.

        Args:
            transform_matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
        """
        self.coords = self.__transform_coords(self.coords, transform_matrix)
        self.camera_transform = np.dot(transform_matrix, self.camera_transform)

    def reverse_transform_coords(self, transform):
        """Restore original coordinates by applying the inverse
        transformation.

        Args:
            transform (np.ndarray): The transformation matrix to invert.
        """
        inverse_transform = np.linalg.inv(transform)
        self.orig_coords = self.__transform_coords(self.coords, inverse_transform)
        self.camera_transform = np.dot(inverse_transform, self.camera_transform)

    def get_pixel_coords(self, coords, use_orig_coords=False):
        """Compute the image pixel coordinates from original 3D coordinates.

        Projects a 3D point and applies lens distortion correction.

        Args:
            coords (list or np.ndarray): The original 3D point.

        Returns:
            tuple: (x, y, depth, relevance metric) or
                   (None, None, None, None) if out of view.
        """
        if use_orig_coords:
            cam_coords = self.orig_coords
            cam_transform = self.orig_camera_transform
        else:
            cam_coords = self.coords
            cam_transform = self.camera_transform

        proj_point = np.dot(np.linalg.inv(cam_transform), np.append(coords, 1))
        x_norm = proj_point[0] / proj_point[2]
        y_norm = proj_point[1] / proj_point[2]
        r2 = x_norm**2 + y_norm**2
        radial = (
            1 + self.parent.k1 * r2 + self.parent.k2 * r2**2 + self.parent.k3 * r2**3
        )
        x_dist = (
            x_norm * radial
            + 2 * self.parent.p1 * x_norm * y_norm
            + self.parent.p2 * (r2 + 2 * x_norm**2)
        )
        y_dist = (
            y_norm * radial
            + self.parent.p1 * (r2 + 2 * y_norm**2)
            + 2 * self.parent.p2 * x_norm * y_norm
        )
        x_img = self.parent.fx * x_dist + self.parent.cx
        y_img = self.parent.fy * y_dist + self.parent.cy
        in_view = (
            r2 * self.parent.fx**2 < 1.01 * self.parent.width**2
            and 0 <= x_img <= self.parent.width
            and 0 <= y_img <= self.parent.height
        )
        if in_view:
            dist_sq = np.sum((cam_coords - coords) ** 2)
            rm = np.abs(
                (np.abs(proj_point[2]) + dist_sq) / (self.parent.fx * self.parent.fy)
            ) * (
                10
                + np.abs(x_img - 0.5 * self.parent.width)
                + np.abs(y_img - 0.5 * self.parent.height)
            )
            return (int(round(x_img)), int(round(y_img)), float(proj_point[2]), rm)
        else:
            return None, None, None, None

    def pixel_to_ray(self, x_img, y_img, use_optimization=False, iterations=20):
        """
        Compute the 3D ray (origin and direction) for a given image pixel.

        This function undistorts the pixel coordinate and converts it to a
        normalized 3D direction vector in the camera coordinate system. The ray
        is then transformed to world coordinates using self.transform, which
        defines the camera's orientation. The camera origin is given by
        self.center.

        Args:
            x_img (float): The x-coordinate in the image.
            y_img (float): The y-coordinate in the image.
            use_optimization (bool): If True, use an optimization routine for
                the undistortion step.
            iterations (int): Number of iterations if not using optimization.

        Returns:
            tuple: (origin, direction) where origin is a 3D point (the camera
            center) and direction is a normalized 3D vector in world coordinates.
        """
        # Retrieve camera intrinsics and distortion parameters.
        cx = self.parent.cx
        cy = self.parent.cy
        fx = self.parent.fx
        fy = self.parent.fy
        k1, k2, k3 = self.parent.k1, self.parent.k2, self.parent.k3
        p1, p2 = self.parent.p1, self.parent.p2

        # Convert pixel coordinates to normalized (distorted) coords.
        x_dist = (x_img - cx) / fx
        y_dist = (y_img - cy) / fy

        if use_optimization:
            # Define error function for distortion inversion.
            def error_func(norm_coords):
                x_norm, y_norm = norm_coords
                r2 = x_norm**2 + y_norm**2
                radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
                delta_x = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
                delta_y = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
                x_est = x_norm * radial + delta_x
                y_est = y_norm * radial + delta_y
                return (x_est - x_dist) ** 2 + (y_est - y_dist) ** 2

            res = minimize(error_func, [x_dist, y_dist])
            x_norm, y_norm = res.x
        else:
            x_norm, y_norm = x_dist, y_dist
            for _ in range(iterations):
                r2 = x_norm**2 + y_norm**2
                radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
                delta_x = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
                delta_y = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
                x_norm = (x_dist - delta_x) / radial
                y_norm = (y_dist - delta_y) / radial

        # Create a direction vector in camera space. Here, we assume the
        # image plane is at unit distance (z = 1).
        vec_cam = np.array([x_norm, y_norm, 1.0])
        vec_cam /= np.linalg.norm(vec_cam)

        # Transform the direction vector from camera to world coordinates.
        # For directions, apply only the rotation part of self.transform.
        R = self.camera_transform[:3, :3]
        vec_world = R.dot(vec_cam)
        vec_world /= np.linalg.norm(vec_world)

        return vec_world

    def pixel_to_point(
        self, x_img, y_img, pcd, search_radius=0.001, reprojection_threshold=None
    ):
        """ """
        back_vector = self.pixel_to_ray(x_img, y_img)
        back_point = pcd.get_intercept(
            self.coords,
            vector=back_vector,
            search_radius=search_radius,
            always_return=True,
        )
        if back_point is None:
            logger.warning(
                f"Back-projection error for {self.cam_id} [{x_img}, {y_img}] failed ..."
            )
            return None

        back_x, back_y, *_ = self.get_pixel_coords(back_point.coords)
        if back_x is None:
            logger.warning(
                f"Back-projection error for {self.cam_id} [{x_img}, {y_img}] when querying 3D intercept {back_point.coords} ..."
            )
            return None

        pixel_distance = np.sqrt((back_x - x_img) ** 2 + (back_y - y_img) ** 2)
        if (
            reprojection_threshold is not None
            and pixel_distance > reprojection_threshold
        ):
            logger.warning(
                f"Back-projection error for {self.cam_id} [{x_img}, {y_img}] resulting in [{back_x}, {back_y}] with distance {pixel_distance:.2f} pixels ..."
            )
            return None

        return back_point.coords, [back_x, back_y], pixel_distance

    def calc_pixel_scale(self, annotations):
        """Calculate the pixel scale based on 3D annotation points.

        Computes distances between 3D annotation points and their corresponding
        2D projection distances.

        Args:
            annotations (Annotations): A collection of annotation points.

        Returns:
            np.ndarray: An array of computed pixel scales.
        """
        point_dists = annotations.get_eucl_distance_matrix()
        pixel_coords = [
            self.get_pixel_coords(ann.orig_coords, use_orig_coords=True)
            for ann in annotations
        ]
        pixel_coords_2d = np.array(pixel_coords)[:, :2]
        pixel_dists = pd.DataFrame(
            squareform(pdist(pixel_coords_2d, metric="euclidean"))
        )
        pixel_scale_matrix = point_dists / pixel_dists
        pixels = pixel_scale_matrix.values[
            np.triu_indices_from(pixel_scale_matrix, k=1)
        ]
        pixels = pixels[~np.isnan(pixels)]
        return pixels

    def get_image_matches(
        self,
        anns,
        pcd=None,
        use_orig_coords=True,
        intercept_radius=settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        reprojection_threshold_uncertain=settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
        reprojection_threshold_discard=settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
    ):
        """
        Obtain image matches for annotation points that are in view.
        """
        image_matches = []
        for ann in anns:
            x, y, depth, relevance = self.get_pixel_coords(
                ann.orig_coords, use_orig_coords=use_orig_coords
            )
            if x is not None:
                # If pixel coordinates are within the camera bounds
                image_match = ImageMatch(self, x, y, depth, relevance, annotation=ann)
                # Classify according to reprojection error if pcd is provided
                if pcd is not None:
                    image_match.get_reprojection_error(pcd, intercept_radius)
                    if image_match.reprojection_error is None:
                        continue  # do not store if no intercept found
                    elif (
                        image_match.reprojection_error > reprojection_threshold_discard
                    ):
                        continue  # do not store if error exceeds threshold
                    elif (
                        image_match.reprojection_error
                        >= reprojection_threshold_uncertain
                    ):
                        image_match.potentially_obstructed = True
                        image_matches.append(image_match)
                    else:
                        image_match.potentially_obstructed = False
                        image_matches.append(image_match)

        # If ImageMatches found, sort by relevance and obstruction
        if len(image_matches) > 0:
            if pcd is not None:
                image_matches = sorted(
                    image_matches, key=lambda x: (x.potentially_obstructed, x.relevance)
                )
            else:
                image_matches = sorted(image_matches, key=lambda x: (x.relevance))

            return image_matches

    def show(self, highlight_pixels=None):
        """
        Display the image match and its attributes.
        """
        visualizations.show_img(self.filepath, highlight_pixels=highlight_pixels)

    def get_datetime_original(self):
        """Retrieve the DateTimeOriginal from the image file EXIF data.

        Returns:
            str or None: The DateTimeOriginal value if found, else None.
        """
        if not os.path.isfile(self.filepath):
            logger.error(f"Image file not found: {self.filepath}")
            return None
        image = Image.open(self.filepath)
        exif_data = image.getexif()
        if exif_data:
            dt_orig = exif_data.get(36867) or exif_data.get(306)
            if dt_orig:
                return dt_orig
            else:
                logger.error(f"No exif DateTimeOriginal for: {self.filepath}")
                return None
        else:
            logger.error(f"No exif data for: {self.filepath}")
            return None

    def has_coords_datetime(self):
        return (
            hasattr(self, "coords")
            and self.coords is not None
            and hasattr(self, "datetime")
            and self.datetime is not None
        )

    def has_coords_datetime_camdist(self):
        return (
            hasattr(self, "coords")
            and self.coords is not None
            and hasattr(self, "datetime")
            and self.datetime is not None
            and hasattr(self, "camdist")
            and self.camdist is not None
        )

    def _get_updated_filepath(self):
        """Update the original filepath based on the parent replacement rules.

        Returns:
            str: The updated filepath.
        """
        if self.parent.filepath_replace[0] and self.parent.filepath_replace[1]:
            updated = self.orig_filepath.replace(
                self.parent.filepath_replace[0], self.parent.filepath_replace[1]
            )
        elif self.parent.filepath_replace[1]:
            updated = self.__replace_base_path(
                self.orig_filepath, self.parent.filepath_replace[1]
            )
        else:
            return self.orig_filepath
        return self.__reformat_filepath_according_to_os(updated)

    @staticmethod
    def __transform_coords(coords, transform):
        hom_coords = np.array([coords[0], coords[1], coords[2], 1], dtype=float)
        return np.array(np.dot(transform, hom_coords)[0:3], dtype=float)

    @staticmethod
    def __replace_base_path(orig_filepath, base_path):
        """Replace the base folder of a filepath.

        Args:
            orig_filepath (str): The original filepath.
            base_path (str): The new base path.

        Returns:
            str: The updated filepath.
        """
        match_string = base_path.rstrip("/").split("/")[-1]
        replace_index = orig_filepath.find(match_string) + len(match_string)
        return base_path.rstrip("/") + orig_filepath[replace_index:]

    @staticmethod
    def __reformat_filepath_according_to_os(filepath):
        """Reformat the filepath to match the current OS.

        Returns:
            str: The reformatted filepath.
        """
        path_parts = re.split(r"[\\/]+", filepath)
        if len(path_parts[0]) == 2 and path_parts[0][1] == ":":
            path_parts[0] += "\\"
        else:
            path_parts[0] += "/"
        return os.path.join(*path_parts)


class ImageMatch:
    """Class that holds information about an image match."""

    def __init__(self, cam, x, y, depth, relevance, annotation=None):
        self.annotation = annotation
        self.cam = cam
        self.filename = cam.filename
        self.filepath = cam.filepath
        self.x = x
        self.y = y
        self.depth = depth
        self.relevance = relevance
        self.pixel_scale = None
        self.pixel_scales = None
        self.masks = None
        self.mask = None  # selected mask for measurements

    def set_image_mask_id(self, mask_id):
        self.mask = self.masks[mask_id]

    def check_if_obstructed(self, pcd, reprojection_threshold, intercept_radius):
        """Check if the image match is obstructed"""
        # Get reprojection error if not already present
        if self.reprojection_error is None:
            self.get_reprojection_error(pcd, intercept_radius)

        # Check if the error is above the threshold
        if self.reprojection_error is None:
            return None
        else:
            if self.reprojection_error > reprojection_threshold:
                self.obstructed = True
            else:
                self.obstructed = False
            return self.obstructed

    def get_reprojection_error(self, pcd, intercept_radius):
        """Check if the image match is obstructed"""
        # Find intercept of camera->pixel vector with pointcloud
        vector = self.cam.pixel_to_ray(self.x, self.y)
        reprojection_intercept = pcd.get_intercept(
            self.cam.coords,
            vector=vector,
            search_radius=intercept_radius,
            always_return=True,
        )
        # Calculate reprojection error if intercept found
        if reprojection_intercept is None:
            print("Warning: no neighboring point found")
            self.reprojection_error = None
            self.reprojection_coords = None
            return None
        else:
            self.reprojection_error = np.linalg.norm(
                self.annotation.coords - reprojection_intercept.coords
            )
            self.reprojection_coords = reprojection_intercept.coords
            return self.reprojection_error

    def calc_pixel_scale_from_crosshair(self, measure_dist=0.01):
        """
        Calculate the pixel scale from 3D crosshair points.

        Args:
            measure_dist (float, optional): Distance for crosshair offset.
        """
        crosshair_anns = self.annotation.get_crosshair_points(
            self.cam.vector, measure_dist
        )
        crosshair_anns.get_original_coords(self.cam.parent.world_transform)
        self.pixel_scales = self.cam.calc_pixel_scale(crosshair_anns)
        self.pixel_scale = np.mean(self.pixel_scales)

    @property
    def pixels_per_mm(self):
        """
        Returns the number of pixels per millimeter based on self.pixel_scale.

        Returns:
            float: Pixels per mm, or None if pixel_scale is not set.
        """
        if self.pixel_scale is not None and self.pixel_scale != 0:
            return 1.0 / (self.pixel_scale * 1000.0)
        return None

    def get_sam2_masks(self, sam_predictor):
        """
        Get the SAM2 masks for the annotation in the image.

        Args:
            sam_predictor: Predictor object for SAM2 segmentation.
        """
        from unicorn import segmentation

        self.masks = segmentation.get_sam2_masks(
            self.filepath, self.x, self.y, sam_predictor
        )
        self.mask = self.masks[0]

    def get_mask_surface_area(self, predictor=None):
        """
        Calculate the surface area of the mask in the image.

        Args:
            predictor: Optional predictor for SAM2.

        Returns:
            float: Surface area in cm^2.
        """
        if not self.pixel_scale:
            self.calc_pixel_scale_from_crosshair()
        if not self.masks and predictor:
            self.get_sam2_masks(predictor)
        if not self.pixel_scale or not self.masks:
            raise ValueError("Scale or masks not available for calculation")
        self.mask.area_in_cm2 = self.mask.area_in_px * (self.pixel_scale**2) * 10000
        return self.mask.area_in_cm2

    def create_rectangular_mask(self, width_m, height_m):
        """
        Create a rectangular mask of specified size around the match point.

        Args:
            width_m (float): Width of the mask in meters
            height_m (float): Height of the mask in meters

        Returns:
            np.ndarray: Binary mask array (True for mask area, False for background)
        """
        if not self.pixel_scale:
            self.calc_pixel_scale_from_crosshair()

        if not self.pixel_scale:
            raise ValueError(
                "Pixel scale not available. Call calc_pixel_scale_from_crosshair() first."
            )

        # Convert meters to pixels using pixel scale (round to minimize discretization error)
        width_px = round(width_m / self.pixel_scale)
        height_px = round(height_m / self.pixel_scale)

        # Get image dimensions
        img = cv2.imread(self.filepath)
        if img is None:
            raise ValueError(f"Cannot load image: {self.filepath}")

        img_height, img_width = img.shape[:2]

        # Calculate rectangle bounds
        x1 = max(0, self.x - width_px // 2)
        y1 = max(0, self.y - height_px // 2)
        x2 = min(img_width, self.x + width_px // 2)
        y2 = min(img_height, self.y + height_px // 2)

        # Create binary mask (use uint8 instead of bool for OpenCV compatibility)
        mask_vals = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_vals[y1:y2, x1:x2] = 255

        # Create a Mask object compatible with segmentation.py Mask class
        class LocalMask:
            def __init__(self, mask_vals, score=1.0, logits=None):
                self.vals = mask_vals
                self.score = score
                self.logits = logits
                self.area_in_px = cv2.countNonZero(mask_vals)
                self.area_in_cm2 = None

        # Store the mask as a Mask object
        self.mask = LocalMask(mask_vals, score=1.0)
        self.mask.area_in_px = cv2.countNonZero(mask_vals)
        self.mask.area_in_cm2 = self.mask.area_in_px * (self.pixel_scale**2) * 10000

        self.masks = np.array([self.mask])

        print(f"Created rectangular mask: {width_m:.2f}m x {height_m:.2f}m")
        print(f"Mask bounds: ({x1}, {y1}) to ({x2}, {y2})")
        print(
            f"Mask area: {self.mask.area_in_px} pixels = {self.mask.area_in_cm2/10000:.4f} m²"
        )

    def show(
        self,
        crop_w=1000,
        crop_h=1000,
        single_mask=False,
    ):
        """
        Display the image match and its attributes.
        """
        print(f"Image match for camera {self.cam.cam_id} at {self.x}, {self.y}")
        print(f"Depth: {self.depth}, Relevance: {self.relevance}")
        if hasattr(self, "potentially_obstructed"):
            print(f"Obstructed: {self.potentially_obstructed}")
            print(f"Reprojection error: {self.reprojection_error}")
            print(f"Reprojection coords: {self.reprojection_coords}")
        else:
            print("No obstruction check performed")
        if self.pixel_scale:
            print(f"Pixel scale: {self.pixel_scale}")
            print(f"Pixels per mm: {self.pixels_per_mm}")
        if self.masks:
            cropped_img = visualizations.get_crop_img_from_masks(
                self, crop_w, crop_h, single_mask=single_mask
            )
        else:
            cropped_img = visualizations.get_crop_img(
                self.cam.filepath, self.x, self.y, crop_w, crop_h
            )
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        plt.imshow(cropped_img)
        plt.show()


class Frame:
    """
    Class representing a single frame from a video.
    """

    def __init__(self, frame_number, timestamp_seconds, image_array, video_source):
        """
        Initialize a Frame object.

        Args:
            frame_number (int): Sequential frame number (0-based)
            timestamp_seconds (float): Timestamp in seconds from video start
            image_array (np.ndarray): The frame image as numpy array
            video_source (Video): Reference to the parent Video object
        """
        self.frame_number = frame_number
        self.timestamp_seconds = timestamp_seconds
        self.image_array = image_array
        self.video_source = video_source

    def __repr__(self):
        return f"Frame(frame_number={self.frame_number}, timestamp={self.timestamp_seconds:.2f}s)"

    def show(self, figsize=(10, 8)):
        """
        Display the frame using matplotlib.

        Args:
            figsize (tuple): Figure size for display
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(self.image_array, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.frame_number} at {self.timestamp_seconds:.2f}s")
        plt.axis("off")
        plt.show()

    def save(self, filepath):
        """
        Save the frame to a file.

        Args:
            filepath (str): Path where to save the frame image
        """
        cv2.imwrite(filepath, self.image_array)
        print(f"Frame saved to: {filepath}")


class Video:
    """
    Class for processing video files and extracting frames at specified intervals.
    """

    def __init__(
        self, video_filepath, frame_interval_seconds=1.0, round_timestamps=True
    ):
        """
        Initialize a Video object.

        Args:
            video_filepath (str): Path to the video file (.mov, .mp4, etc.)
            frame_interval_seconds (float): Interval between extracted frames in seconds
            round_timestamps (bool): If True, round timestamps to nearest frame_interval_seconds
        """
        self.video_filepath = video_filepath
        self.frame_interval_seconds = frame_interval_seconds
        self.round_timestamps = round_timestamps
        self.frames = []
        self.video_info = {}

        # Validate file exists
        if not os.path.exists(video_filepath):
            raise FileNotFoundError(f"Video file not found: {video_filepath}")

        # Extract frames
        self._extract_frames()

    def _extract_frames(self):
        """
        Extract frames from the video at the specified interval.
        """
        cap = cv2.VideoCapture(self.video_filepath)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_filepath}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        self.video_info = {
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration_seconds,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        # Calculate frame interval in frame numbers
        frame_interval = max(1, int(fps * self.frame_interval_seconds))

        print(f"Extracting frames from: {self.video_filepath}")
        print(
            f"Video info: {self.video_info['width']}x{self.video_info['height']}, "
            f"{fps:.2f} fps, {duration_seconds:.2f}s duration"
        )
        print(
            f"Extracting every {frame_interval} frames ({self.frame_interval_seconds}s intervals)"
        )

        frame_number = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at specified interval
            if frame_number % frame_interval == 0:
                # Calculate exact timestamp
                exact_timestamp = frame_number / fps

                # Round timestamp if requested
                if self.round_timestamps:
                    timestamp_seconds = (
                        round(exact_timestamp / self.frame_interval_seconds)
                        * self.frame_interval_seconds
                    )
                else:
                    timestamp_seconds = exact_timestamp

                frame_obj = Frame(frame_number, timestamp_seconds, frame.copy(), self)
                self.frames.append(frame_obj)
                extracted_count += 1

                if extracted_count % 10 == 0:
                    print(f"Extracted {extracted_count} frames...")

            frame_number += 1

        cap.release()
        print(f"Extraction complete: {len(self.frames)} frames extracted")

    def __len__(self):
        """Return the number of extracted frames."""
        return len(self.frames)

    def __getitem__(self, index):
        """Get a frame by index."""
        return self.frames[index]

    def __iter__(self):
        """Iterate over frames."""
        return iter(self.frames)

    def get_frame_by_timestamp(self, timestamp_seconds, tolerance_seconds=0.1):
        """
        Get the frame closest to a specific timestamp.

        Args:
            timestamp_seconds (float): Target timestamp in seconds
            tolerance_seconds (float): Maximum allowed difference from target timestamp

        Returns:
            Frame or None: The closest frame within tolerance, or None if not found
        """
        closest_frame = None
        min_diff = float("inf")

        for frame in self.frames:
            diff = abs(frame.timestamp_seconds - timestamp_seconds)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame

        if min_diff <= tolerance_seconds:
            return closest_frame
        else:
            return None

    def get_frames_in_timerange(self, start_seconds, end_seconds):
        """
        Get all frames within a time range.

        Args:
            start_seconds (float): Start time in seconds
            end_seconds (float): End time in seconds

        Returns:
            list: List of Frame objects within the time range
        """
        return [
            frame
            for frame in self.frames
            if start_seconds <= frame.timestamp_seconds <= end_seconds
        ]

    def get_frame_at_exact_second(self, target_second):
        """
        Get the frame closest to an exact second boundary.

        Args:
            target_second (int): Target second (e.g., 5 for 5.0 seconds)

        Returns:
            Frame or None: The frame closest to the target second
        """
        target_timestamp = float(target_second)
        return self.get_frame_by_timestamp(target_timestamp, tolerance_seconds=0.5)

    def get_frames_at_second_intervals(self, start_second=0, end_second=None):
        """
        Get frames at exact second intervals (0s, 1s, 2s, etc.).

        Args:
            start_second (int): Starting second (default: 0)
            end_second (int): Ending second (default: video duration)

        Returns:
            list: List of Frame objects at second boundaries
        """
        if end_second is None:
            end_second = int(self.video_info["duration_seconds"])

        frames = []
        for second in range(start_second, end_second + 1):
            frame = self.get_frame_at_exact_second(second)
            if frame is not None:
                frames.append(frame)

        return frames

    def save_frames_to_directory(self, output_directory, filename_prefix="frame"):
        """
        Save all extracted frames to a directory.

        Args:
            output_directory (str): Directory to save frames
            filename_prefix (str): Prefix for frame filenames
        """
        os.makedirs(output_directory, exist_ok=True)

        for frame in self.frames:
            filename = f"{filename_prefix}_{frame.frame_number:06d}_{frame.timestamp_seconds:.2f}s.jpg"
            filepath = os.path.join(output_directory, filename)
            frame.save(filepath)

        print(f"Saved {len(self.frames)} frames to: {output_directory}")

    def get_video_info(self):
        """
        Get information about the video.

        Returns:
            dict: Video information including fps, dimensions, duration, etc.
        """
        return self.video_info.copy()

    def show_frame(self, frame_index=0):
        """
        Display a specific frame.

        Args:
            frame_index (int): Index of the frame to display
        """
        if 0 <= frame_index < len(self.frames):
            self.frames[frame_index].show()
        else:
            print(f"Frame index {frame_index} out of range (0-{len(self.frames)-1})")

    def get_frame_statistics(self):
        """
        Get statistics about the extracted frames.

        Returns:
            dict: Statistics about frame extraction
        """
        if not self.frames:
            return {}

        timestamps = [frame.timestamp_seconds for frame in self.frames]
        frame_numbers = [frame.frame_number for frame in self.frames]

        return {
            "total_extracted_frames": len(self.frames),
            "first_frame_time": min(timestamps),
            "last_frame_time": max(timestamps),
            "time_span": max(timestamps) - min(timestamps),
            "average_interval": np.mean(np.diff(timestamps)),
            "frame_interval_seconds": self.frame_interval_seconds,
            "frame_numbers": frame_numbers,
            "timestamps": timestamps,
        }

    def show_timestamp_mapping(self, max_frames=10):
        """
        Show the mapping between frame numbers, exact timestamps, and rounded timestamps.

        Args:
            max_frames (int): Maximum number of frames to display
        """
        print(
            f"Timestamp mapping (showing first {min(max_frames, len(self.frames))} frames):"
        )
        print(
            f"{'Frame':<6} {'Exact Time':<12} {'Rounded Time':<12} {'Difference':<10}"
        )
        print("-" * 45)

        for i, frame in enumerate(self.frames[:max_frames]):
            exact_time = frame.frame_number / self.video_info["fps"]
            rounded_time = frame.timestamp_seconds
            difference = abs(exact_time - rounded_time)

            print(
                f"{frame.frame_number:<6} {exact_time:<12.3f} {rounded_time:<12.3f} {difference:<10.3f}"
            )

        if len(self.frames) > max_frames:
            print(f"... and {len(self.frames) - max_frames} more frames")
