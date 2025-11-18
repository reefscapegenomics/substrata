# Standard Library
import csv
import datetime
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET

# Third-Party Libraries
import cv2
from joblib.externals.cloudpickle.cloudpickle import _property_reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from joblib import Parallel, delayed
import exifread

# Local Modules
from substrata import visualizations, settings, geom, measurements
from substrata.logging import tqdm_joblib

logger = logging.getLogger(__name__)


class Sensor:
    """Class that holds calibration parameters for a specific camera sensor."""

    def __init__(self, sensor_id, label, resolution, calibration_params):
        """Initialize a Sensor instance.

        Args:
            sensor_id (int): Unique sensor identifier.
            label (str): Sensor label/description.
            resolution (dict): Dictionary with 'width' and 'height' keys.
            calibration_params (dict): Dictionary containing calibration parameters.
        """
        self.sensor_id = sensor_id
        self.label = label
        self.width = resolution["width"]
        self.height = resolution["height"]

        # Calibration parameters
        self.f = calibration_params["f"]
        self.cx_metashape = calibration_params["cx"]
        self.cy_metashape = calibration_params["cy"]
        self.b1 = calibration_params.get("b1", 0.0)
        self.b2 = calibration_params.get("b2", 0.0)
        self.k1 = calibration_params["k1"]
        self.k2 = calibration_params["k2"]
        self.k3 = calibration_params["k3"]
        self.p1 = calibration_params["p1"]
        self.p2 = calibration_params["p2"]

        # Derived parameters
        self.fx = self.f + self.b1
        self.fy = self.f
        self.cx = self.width / 2 + self.cx_metashape
        self.cy = self.height / 2 + self.cy_metashape


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
        self.sensors = {}  # Dict mapping sensor_id -> Sensor instance
        self.world_transform = np.eye(4)
        if cams_meta_filepath:
            self.cams_meta_filepath = cams_meta_filepath
            self.get_cams_from_file(cams_meta_filepath)
        if cams_xml_filepath:
            self.cams_xml_filepath = cams_xml_filepath
            self.get_cam_sensor_parameters_from_file(cams_xml_filepath)

    def __str__(self) -> str:
        """Return a concise, human-readable summary of the cameras collection."""
        total = len(self.data)
        sensors_count = len(getattr(self, "sensors", {}) or {})
        wt_is_identity = self.world_transform_is_identity
        groups = self.group_names
        # Count cameras per group
        group_counts = {}
        for cam in self.data.values():
            g = getattr(cam, "group", None)
            if g is None:
                continue
            group_counts[g] = group_counts.get(g, 0) + 1

        # Basic availability stats
        num_with_coords = sum(
            1 for cam in self.data.values() if getattr(cam, "coords", None) is not None
        )
        num_with_datetime = sum(
            1
            for cam in self.data.values()
            if getattr(cam, "datetime", None) is not None
        )
        num_with_depth = sum(
            1
            for cam in self.data.values()
            if getattr(cam, "depth_sensor_m", None) is not None
        )
        num_enabled = sum(
            1 for cam in self.data.values() if getattr(cam, "enabled", True) is True
        )

        # Coordinate bounds (if available)
        try:
            coords_list = [
                cam.coords
                for cam in self.data.values()
                if getattr(cam, "coords", None) is not None
            ]
            if coords_list:
                C = np.vstack(coords_list)
                cmin = C.min(axis=0)
                cmax = C.max(axis=0)
                extent = cmax - cmin
                bb_str = (
                    f"[min=({cmin[0]:.3f}, {cmin[1]:.3f}, {cmin[2]:.3f}), "
                    f"max=({cmax[0]:.3f}, {cmax[1]:.3f}, {cmax[2]:.3f}), "
                    f"extent=({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})]"
                )
            else:
                bb_str = "unavailable"
        except Exception:
            bb_str = "unavailable"

        lines = [
            "Cameras(",
            f"  count={total}, sensors={sensors_count}, enabled={num_enabled},",
            f"  with_coords={num_with_coords}, with_datetime={num_with_datetime}, with_depth_sensor_m={num_with_depth},",
            f"  world_transform={'identity' if wt_is_identity else 'non-identity'},",
            f"  groups={groups if groups else '[]'},",
            f"  groups_counts={group_counts if group_counts else {}},",
            f"  coords_bounds={bb_str}",
            ")",
        ]
        return "\n".join(lines)

    @property
    def coords(self):
        return [camera.coords for camera in self.data.values()]

    # @property
    # def depths(self):
    #     return [camera.depth for camera in self.data.values()]

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

    def __len__(self):
        """Return the number of cameras in the collection."""
        return len(self.data)

    def items(self):
        return self.data.items()

    @property
    def world_transform_is_identity(self) -> bool:
        """Check if the world_transform is the identity matrix."""
        return np.allclose(self.world_transform, np.eye(4))

    @property
    def group_names(self):
        return sorted(
            {cam.group for cam in self.data.values() if hasattr(cam, "group")}
        )

    def show(self, pcd, color=False):
        """
        Show the camera positions in the pointcloud.

        Args:
            pcd (PointCloud): The pointcloud to show the camera positions in.
        """
        visualizations.plot_positions(self, pcd, color=color)

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

    def reset_transform(self):
        """Reset all camera coordinates and transforms to their original state.

        Applies the inverse of the current world_transform to all cameras,
        then sets world_transform to the identity matrix.
        """
        if not self.world_transform_is_identity:
            for cam_id in self.data:
                self.data[cam_id].reverse_transform_coords(self.world_transform)
            self.world_transform = np.eye(4)

    def apply_transform(self, transform_matrix):
        """Alias for transform_coords for compatibility.
        f
                Args:
                    transform_matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
        """
        self.transform_coords(transform_matrix)

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
        cameras_subset = self._empty_like()
        for cam_id in list(self.data.keys())[:length]:
            cameras_subset.data[cam_id] = self.data[cam_id]
            cameras_subset.data[cam_id].parent = cameras_subset
        return cameras_subset

    def subset_by_filename_prefix(self, prefix):
        """Return a subset of cameras with filenames starting with a prefix.

        Args:
            prefix (str): Prefix to filter camera IDs.

        Returns:
            Cameras: New container with matching cameras.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            if cam.filename.startswith(prefix):
                cameras_subset.data[cam.cam_id] = cam
                cam.parent = cameras_subset
        return cameras_subset

    def subset_by_filepath_postfix(self, postfix):
        """Return a subset of cameras whose containing folder ends with a given postfix.

        Args:
            postfix (str): Postfix string to match at the end of the containing folder.

        Returns:
            Cameras: New container with matching cameras.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            folder = os.path.dirname(cam.filepath)
            if folder.endswith(postfix) or folder.endswith(postfix + os.sep):
                cameras_subset.data[cam.cam_id] = cam
                cam.parent = cameras_subset
        return cameras_subset

    def subset_by_group(self, group_name):
        """Return a subset of cameras that belong to a given group.

        Args:
            group_name (str): Name of the group (as given by `Camera.group`).

        Returns:
            Cameras: New container with cameras from the requested group.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            if getattr(cam, "group", None) == group_name:
                cameras_subset.data[cam.cam_id] = cam
                cam.parent = cameras_subset
        return cameras_subset

    def subset_by_sensor(self, sensor_id):
        """Return a subset of cameras that use a specific sensor.

        Args:
            sensor_id (int): The sensor ID to filter by.

        Returns:
            Cameras: New container with cameras using the specified sensor.
        """
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            if getattr(cam, "sensor_id", None) == sensor_id:
                cameras_subset.data[cam.cam_id] = cam
                cam.parent = cameras_subset
        if len(cameras_subset.data) == 0:
            print(f"No cameras found with sensor_id={sensor_id}")
        return cameras_subset

    # Convenience alias to match the requested call-site: Cameras.group("name")
    def group(self, group_name):
        return self.subset_by_group(group_name)

    def _empty_like(self) -> "Cameras":
        """Return an empty Cameras container inheriting this instance's metadata."""
        subset = Cameras()
        for attr, val in self.__dict__.items():
            if attr == "data":
                continue
            try:
                setattr(subset, attr, val)
            except Exception:
                pass
        subset.data = {}
        return subset

    def filter_by_ids(self, cam_ids):
        """Return a new Cameras container containing only the specified camera ids.

        Copies container metadata (shallow) and re-parents included Camera objects
        so downstream code can rely on attributes like world_transform, up_vector, etc.
        """
        subset = self._empty_like()
        for cid in cam_ids:
            cam = self.data.get(cid)
            if cam is not None:
                subset.data[cid] = cam
                cam.parent = subset
        return subset

    def filter_by_cams(self, cams_list):
        """Return a new Cameras container containing only the specified Camera objects.

        Equivalent to calling filter_by_ids([cam.cam_id for cam in cams_list]).
        """
        ids = [getattr(cam, "cam_id", None) for cam in cams_list]
        ids = [cid for cid in ids if cid is not None]
        return self.filter_by_ids(ids)

    def reset_depth_sensor_m(self):
        """Reset the depth_sensor_m attribute for all cameras to None."""
        for cam in self.data.values():
            cam.depth_sensor_m = None

    def set_depth_sensor_m(self, depth_by_cam_id):
        """Set depth_sensor_m for cameras from a mapping of cam_id -> depth (meters)."""
        for cam_id, depth in depth_by_cam_id.items():
            cam = self.data.get(cam_id)
            if cam is not None:
                cam.depth_sensor_m = float(depth)

    def get_cams_from_file(self, cams_meta_filepath):
        """Load cameras from a JSON file and store them in the container.

        Args:
            cams_meta_filepath (str): Path to the JSON file with camera metadata.
        """
        with open(cams_meta_filepath, "r") as f:
            data = json.load(f)
        for cam_id, cam_data in data["cameras"].items():
            if (
                cam_data.get("center") is None
                or cam_data.get("center") == "null"
                or cam_data.get("transform") is None
                or cam_data.get("transform") == "null"
                or cam_data.get("path") is None
                or cam_data.get("path") == "null"
            ):
                continue
            self.data[cam_id] = Camera(
                self,
                cam_id,
                cam_data["transform"],
                cam_data["center"],
                cam_data["path"],
            )
            if "reference" in cam_data:
                self.data[cam_id].reference = cam_data["reference"]
                # Ensure depth is negative (in meters)
                self.data[cam_id].depth_sensor_m = -abs(cam_data["reference"][2])
            if "reference_accuracy" in cam_data:
                self.data[cam_id].reference_acc = cam_data["reference_accuracy"]
                self.data[cam_id].depth_acc = float(cam_data["reference_accuracy"][2])
            if "center_crs" in cam_data:
                self.data[cam_id].center_crs = cam_data["center_crs"]
            if "enabled" in cam_data:
                self.data[cam_id].enabled = bool(cam_data["enabled"])

    def get_cam_sensor_parameters_from_file(self, cams_xml_filepath):
        """Parse XML file and create Sensor objects, then assign to cameras.

        Args:
            cams_xml_filepath (str): Path to the XML file with sensor parameters.
        """
        tree = ET.parse(cams_xml_filepath)
        root = tree.getroot()

        # 1. Parse all sensors and create Sensor objects
        sensors_section = root.find(".//sensors")
        if sensors_section is None:
            sys.exit("No sensors section found in XML file!")

        for sensor_elem in sensors_section.findall("sensor"):
            sensor_id = int(sensor_elem.get("id"))
            label = sensor_elem.get("label", "unknown")

            # Parse resolution
            resolution_elem = sensor_elem.find("resolution")
            if resolution_elem is None:
                logger.warning(
                    f"No resolution found for sensor {sensor_id}, skipping..."
                )
                continue

            resolution = {
                "width": int(resolution_elem.get("width")),
                "height": int(resolution_elem.get("height")),
            }

            # Parse calibration
            calibration = sensor_elem.find(
                './/calibration[@type="frame"][@class="adjusted"]'
            )
            if calibration is not None:
                # Helper function to safely get float values
                def get_float_or_zero(elem, name, default=0.0):
                    found_elem = elem.find(name)
                    if found_elem is not None and found_elem.text is not None:
                        try:
                            return float(found_elem.text)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid {name} value for sensor {sensor_id}, using default {default}"
                            )
                            return default
                    else:
                        logger.warning(
                            f"Missing {name} for sensor {sensor_id}, using default {default}"
                        )
                        return default

                calib_params = {
                    "f": get_float_or_zero(calibration, "f"),
                    "cx": get_float_or_zero(calibration, "cx"),
                    "cy": get_float_or_zero(calibration, "cy"),
                    "k1": get_float_or_zero(calibration, "k1"),
                    "k2": get_float_or_zero(calibration, "k2"),
                    "k3": get_float_or_zero(calibration, "k3"),
                    "p1": get_float_or_zero(calibration, "p1"),
                    "p2": get_float_or_zero(calibration, "p2"),
                }

                # Handle optional b1, b2 parameters
                b1_elem = calibration.find("b1")
                if b1_elem is not None and b1_elem.text is not None:
                    try:
                        calib_params["b1"] = float(b1_elem.text)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid b1 value for sensor {sensor_id}, using default 0.0"
                        )
                        calib_params["b1"] = 0.0
                else:
                    calib_params["b1"] = 0.0

                b2_elem = calibration.find("b2")
                if b2_elem is not None and b2_elem.text is not None:
                    try:
                        calib_params["b2"] = float(b2_elem.text)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid b2 value for sensor {sensor_id}, using default 0.0"
                        )
                        calib_params["b2"] = 0.0
                else:
                    calib_params["b2"] = 0.0

                # Create Sensor instance
                sensor = Sensor(sensor_id, label, resolution, calib_params)
                self.sensors[sensor_id] = sensor
            else:
                logger.warning(
                    f"No calibration found for sensor {sensor_id}, skipping..."
                )

        # 2. Parse cameras and assign sensor references
        cameras_section = root.find(".//cameras")
        if cameras_section is None:
            logger.warning("No cameras section found in XML file!")
            return

        assigned_count = 0
        xml_cam_ids = []
        json_cam_ids = list(self.data.keys())

        for camera_elem in cameras_section.findall(".//camera"):
            cam_id = camera_elem.get("id")
            sensor_id = int(camera_elem.get("sensor_id"))
            xml_cam_ids.append(cam_id)

            # Find matching camera in our data and assign sensor
            static_no_sensor_message_printed = getattr(
                self, "_no_sensor_message_printed", False
            )
            if cam_id in self.data:
                self.data[cam_id].sensor_id = sensor_id
                self.data[cam_id].sensor = self.sensors.get(sensor_id)
                if self.data[cam_id].sensor is not None:
                    assigned_count += 1
                else:
                    if not getattr(self, "_no_sensor_message_printed", False):
                        logger.warning(
                            f"No sensor found for at least one camera (e.g. {cam_id} {sensor_id})"
                        )
                        self._no_sensor_message_printed = True
            else:
                logger.warning(f"Camera {cam_id} from XML not found in loaded cameras")

        if not self.sensors:
            sys.exit("No valid sensors found in XML file!")

    def load_camera_attributes(self, input_filepath):
        """Load camera attributes from a CSV file.

        Updates each camera in the container with attributes such as path,
        datetime, distance, and depth information (stored in depth_sensor_m).

        Args:
            input_filepath (str): Path to the CSV file with camera attributes.
        """
        cam_counter = 0
        not_found_counter = 0
        with open(input_filepath, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cam_id = row.get("cam_id", row.get("id"))
                if cam_id in self.data:
                    cam = self.data[cam_id]
                    cam.orig_filepath = row.get("path", row.get("label"))
                    cam.datetime = row["datetime"] if row["datetime"] else None
                    cam.camdist = row["camdist"] if row["camdist"] else None
                    cam.depth_sensor_m = float(row["depth"]) if row["depth"] else None
                    cam_counter += 1
                else:
                    not_found_counter += 1
        if not_found_counter > 0:
            logger.warning(
                f"File had {not_found_counter} cameras that were not found..."
            )
        if cam_counter == 0:
            logger.warning(f"No cameras found in file {input_filepath}")

    def save_camera_attributes(self, output_filepath):
        """Save camera attributes to a CSV file.

        Writes camera ID, path, datetime, distance, depth, predicted depth,
        and depth residual for each camera.

        Args:
            output_filepath (str): Path to the output CSV file.
        """
        with open(output_filepath, "w", newline="") as csvfile:
            fieldnames = [
                "id",
                "orig_x",
                "orig_y",
                "orig_z",
                "label",
                "datetime",
                "camdist",
                "depth",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cam in self.data.values():
                writer.writerow(
                    {
                        "id": cam.cam_id,
                        "orig_x": cam.orig_coords[0],
                        "orig_y": cam.orig_coords[1],
                        "orig_z": cam.orig_coords[2],
                        "label": cam.orig_filepath,
                        "datetime": getattr(cam, "datetime", None),
                        "camdist": getattr(cam, "camdist", None),
                        # Keep CSV column name 'depth' for compatibility,
                        # but read from Camera.depth_sensor_m
                        "depth": getattr(cam, "depth_sensor_m", None),
                    }
                )

    def get_average_vector(self):
        """Get the average vector of the cameras."""
        vectors = np.array([cam.vector for cam in self.data.values()])
        return vectors.mean(axis=0)

    def get_timematches(self, other_cams):
        """Get the timematches between the cameras.

        Args:
            other_cams (Cameras): The other cameras to match against.
        """
        timematches = []
        for cam in self.data.values():
            other_cam = other_cams.get_camera_by_datetime(cam.datetime)
            if other_cam is not None:
                timematches.append((cam, other_cam))
            else:
                raise ValueError(
                    f"No camera found for camera {cam.cam_id} with datetime {cam.datetime}"
                )
        return timematches

    def get_centers_and_transforms_based_on_timematch(
        self, other_cams, offset_xyz=None
    ):
        """
        Adopt the centers and transform from other cameras based on a timesync.
        Provide an offset_xyz in the camera's local coordinate system to apply to the camera positions.

        Args:
            other_cams (Cameras): The other cameras to match against.
            offset_xyz (array-like): [x, y, z] offsets in the camera's local coordinate system.
        """
        for cam in self.data.values():
            other_cam = other_cams.get_camera_by_datetime(cam.datetime)
            if other_cam is not None:
                if offset_xyz is not None:
                    # Transform offset_xyz from camera coordinate system to world coordinates
                    # Extract rotation matrix from camera transform and orthonormalize it
                    rotation_matrix = np.array(other_cam.camera_transform, dtype=float)[
                        :3, :3
                    ]
                    # Use SVD to obtain the nearest rotation (handles potential scaling/shear)
                    U, _, Vt = np.linalg.svd(rotation_matrix)
                    R = U @ Vt
                    # Ensure a proper rotation with det(R) == +1
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = U @ Vt
                    # Transform the offset to world coordinates using the pure rotation
                    world_offset = R @ np.array(offset_xyz, dtype=float)
                    # Add the transformed offset to the camera position
                    cam.coords = other_cam.coords + world_offset
                    # Also apply the offset to the camera_transform translation
                    cam.camera_transform = other_cam.camera_transform.copy()
                    cam.camera_transform[:3, 3] = (
                        np.array(cam.camera_transform[:3, 3], dtype=float)
                        + world_offset
                    )
                    cam.reverse_transform_coords(other_cam.parent.world_transform)
                else:
                    cam.coords = other_cam.coords
                    cam.camera_transform = other_cam.camera_transform
                    cam.orig_coords = other_cam.orig_coords
                    cam.orig_camera_transform = other_cam.orig_camera_transform
            else:
                raise ValueError(
                    f"No camera found for camera {cam.cam_id} with datetime {cam.datetime}"
                )

    def set_filepath_replace(self, find_str, replace_str):
        """Set a find/replace pair for adjusting filepaths.

        Args:
            find_str (str): The string to search for in the filepaths.
            replace_str (str): The string to replace find_str with.
        """
        self.filepath_replace = [find_str, replace_str]

    def set_filename_prefix(self, filename_prefix):
        """Set a filename prefix for adjusting filepaths.

        Args:
            filename_prefix (str): The filename prefix to use.
        """
        self.filename_prefix = filename_prefix

    def set_base_path(self, base_path):
        """Set a replacement base path for adjusting filepaths.

        Args:
            base_path (str): The new base path to use.
        """
        self.filepath_replace = ["", base_path]

    def get_cam_dists(
        self,
        pcd,
        beam_angle: float,
        n_jobs: int = -1,
        backend: str = "threading",
        scale_factor: float = 1.0,
    ):
        """Calculate camera distances to a point cloud.

        For each camera, compute the distance to the given point cloud based on
        the provided beam angle. If ``n_jobs != 1`` (default -1: all CPUs), this
        will parallelize the computation using joblib with a threading backend
        by default, which avoids pickling the non-serializable point cloud object.

        Args:
            pcd: A point cloud object.
            beam_angle (float): The beam angle for distance calculation.
            n_jobs (int): Number of parallel workers. Use 1 for sequential; -1 for all CPUs.
            backend (str): joblib backend. Use "threading" to avoid pickling ``pcd``.
            scale_factor (float, optional): Value to scale the result distance by. Default is 1.0.
        """
        cams_list = list(self.data.values())
        if n_jobs in (None, 1):
            for cam in tqdm(cams_list, desc="Calculating camera distances..."):
                cam.camdist = pcd.get_cam_dist(cam, beam_angle) * scale_factor
        else:
            with tqdm_joblib(
                tqdm(total=len(cams_list), desc="Calculating camera distances...")
            ):
                dists = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(pcd.get_cam_dist)(cam, beam_angle) for cam in cams_list
                )
            for cam, dist in zip(cams_list, dists):
                cam.camdist = dist * scale_factor

    def get_datetime_originals(self, offset_secs=None):
        """Retrieve DateTimeOriginal metadata from image EXIF for all cameras."""
        for cam in tqdm(
            self.data.values(), desc="Retrieving timestamps from camera files..."
        ):
            cam.datetime = cam.get_datetime_original(offset_secs)

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

    def get_camera_by_filename_partial_match(self, filename):
        """Get a camera object by a partial filename match.

        Args:
            filename (str): The filename to search for.
        """
        for cam in self.data.values():
            if filename in cam.filename:
                return cam
        return None

    def get_camera_by_filepath(self, filepath):
        """Get a camera object by its filepath.

        Args:
            filepath (str): The filepath to search for.
        """
        for cam in self.data.values():
            if cam.filepath == filepath:
                return cam
            elif cam.orig_filepath == filepath:
                return cam
        return None

    def get_camera_by_datetime(self, datetime):
        """Get a camera object by its datetime.

        Args:
            datetime (str): The datetime to search for.
        """
        for cam in self.data.values():
            if cam.datetime == datetime:
                return cam
        return None

    def get_time_delta_between_first_and_last_photo(self):
        """Calculate the time delta between the first and last photos.

        Returns:
            int: The difference in seconds between the first and last camera
                timestamps.
        """
        import substrata.firefish as firefish

        cams_with_datetime = [
            cam for cam in self.data.values() if hasattr(cam, "datetime")
        ]
        first_cam = cams_with_datetime[0]
        last_cam = cams_with_datetime[-1]
        return int(
            firefish.get_unix_time(last_cam.datetime)
            - firefish.get_unix_time(first_cam.datetime)
        )

    def get_up_vector_from_camera_depths(
        self,
        depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
        plot=False,
    ):
        """Compute the up vector using least-squares regression on camera depths.

        Fits a linear regression between the camera 3D points and their sensor depths to
        find the dominant depth direction. Also stores predicted depths and errors.

        Args:
            plot (bool): If True, create a visualization of the regression fit.

        Returns:
            np.ndarray: The coefficient vector representing the up vector.
        """
        # Filter cameras to those with depth_sensor_m and coords, and if they have a depth accuracy threshold
        # ensure that it is below the threshold
        cams_filtered = [
            cam
            for cam in self.data.values()
            if hasattr(cam, "depth_sensor_m")
            and hasattr(cam, "coords")
            and cam.depth_sensor_m is not None
            and cam.coords is not None
            and (
                not hasattr(cam, "depth_acc")
                or cam.depth_acc <= depth_accuracy_threshold
            )
        ]
        print(f"Found {len(cams_filtered)} matching cameras/depths for regression")

        # Conduct regression on the filtered cameras
        cam_ids = [cam.cam_id for cam in cams_filtered]
        points = np.array([cam.coords for cam in cams_filtered])
        depths = np.array([cam.depth_sensor_m for cam in cams_filtered])

        res = measurements.fit_depth_regression(points, depths)

        # 4) Plot the regression fit if requested
        if plot:
            from substrata import visualizations

            visualizations.plot_depth_regression(depths, res.depths_pred)

        # Print summary statistics
        print(
            f"  Up vector: [{res.up_vector[0]}, {res.up_vector[1]}, {res.up_vector[2]}]"
        )
        print(f"  Depth offset: {res.depth_offset:.4f} m")
        print(f"  Depth per unit: {res.depth_per_unit:.4f} m")
        print(f"  Mean squared error: {res.mse:.4f} m²")
        print(f"  Root mean squared error: {res.rmse:.4f} m")
        print(f"  Mean absolute error: {res.mae:.4f} m")
        print(f"  R²: {res.r2:.4f}")
        print(f"  Number of matches: {len(cams_filtered)}")

        # 5) Return the *flipped-if-needed* up vector and error metrics, plus number of matches
        return (
            res.up_vector,
            res.depth_offset,
            res.depth_per_unit,
            res.mse,
            res.rmse,
            res.mae,
            res.r2,
            len(cams_filtered),
        )

    def get_depths_and_estimated_depths(
        self, depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD
    ):
        """
        Get the sensor depths (.depth_sensor_m) and predicted depths (.depth_in_m)
        for the cameras.

        Args:
            depth_accuracy_threshold (float): The depth accuracy threshold.

        Returns:
            tuple: A tuple containing the depths and predicted depths.
        """
        cams_list = [
            cam
            for cam in self.data.values()
            if cam.depth_sensor_m is not None
            and cam.coords is not None
            and (
                not hasattr(cam, "depth_acc")
                or cam.depth_acc <= depth_accuracy_threshold
            )
        ]
        cams_filtered = self.filter_by_cams(cams_list)
        depths = [cam.depth_sensor_m for cam in cams_filtered.data.values()]
        est_depths = [cam.depth_in_m for cam in cams_filtered.data.values()]
        return depths, est_depths, cams_filtered

    def get_depths_and_z_coords(
        self, depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD
    ):
        """
        Get the recorded camera depths (.depth_sensor_m) and current z-coordinates
        (.coords[2]) for the cameras.

        Args:
            depth_accuracy_threshold (float): The depth accuracy threshold.

        Returns:
            tuple: A tuple containing the depths and z-coordinates.
        """
        cams_list = [
            cam
            for cam in self.data.values()
            if hasattr(cam, "depth_sensor_m")
            and cam.depth_sensor_m is not None
            and cam.coords is not None
            and (
                not hasattr(cam, "depth_acc")
                or cam.depth_acc <= depth_accuracy_threshold
            )
        ]
        cams_filtered = self.filter_by_cams(cams_list)
        depths = [cam.depth_sensor_m for cam in cams_filtered.data.values()]
        z_coords = [cam.coords[2] for cam in cams_filtered.data.values()]
        return depths, z_coords, cams_filtered

    def show_depth_vs_est_depth_residuals(self, width=15, height=5):
        """Show residuals between predicted depth_in_m and the original recorded camera depths.

        These residuals are calculated based on recorded camera depths and the predicted depths from the regression model.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            recalculate (bool): If True, recalculate the depth residuals (default: True)
        """
        depths, est_depths, cams_filtered = self.get_depths_and_estimated_depths()
        fig1 = visualizations.plot_depth_regression(
            depths,
            est_depths,
            width=width,
            height=height,
            title="Depth vs Estimated Depth",
        )
        fig2 = visualizations.plot_cam_residuals(
            cams_filtered, depths, est_depths, width=width, height=height
        )
        return fig1, fig2

    def show_z_vs_depth_residuals(self, width=15, height=5):
        """Show residuals between camera z-coordinates and the original recorded camera depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            recalculate (bool): If True, recalculate the depth residuals (default: True)
        """
        depths, z_coords, cams_filtered = self.get_depths_and_z_coords()
        fig1 = visualizations.plot_depth_regression(
            depths, z_coords, width=width, height=height, title="Depth vs Z-Coordinate"
        )
        fig2 = visualizations.plot_cam_residuals(
            cams_filtered, depths, z_coords, width=width, height=height
        )
        return fig1, fig2

    def save_depth_residuals_pdf(
        self, filepath=None, width=15, height=5, recalculate=True
    ):
        """Save camera depth residuals visualization as a PDF.

        These residuals are calculated based on recorded camera depths and the predicted depths from the regression model.

        Args:
            filepath (str, optional): Path to save the PDF file. If None, generates
                a default filename based on the cameras metadata filepath.
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
            recalculate (bool): If True, recalculate the depth residuals (default: True)

        Returns:
            str: The filepath where the PDF was saved.
        """
        import matplotlib
        from matplotlib.backends.backend_pdf import PdfPages

        backend_original = matplotlib.get_backend()
        # Use a non-interactive backend to prevent showing figures
        matplotlib.use("Agg", force=True)
        try:
            if filepath is None:
                # Generate default filename from cameras metadata filepath if available
                if hasattr(self, "cams_meta_filepath") and self.cams_meta_filepath:
                    base, _ = os.path.splitext(self.cams_meta_filepath)
                    filepath = f"{base}_depth_residuals.pdf"
                else:
                    filepath = "depth_residuals.pdf"

            # Get the figures from show_depth_residuals
            fig1, fig2 = self.show_depth_vs_est_depth_residuals(
                width=width, height=height
            )
            fig3, fig4 = self.show_z_vs_depth_residuals(width=width, height=height)

            # Save both figures to PDF
            pdf = PdfPages(filepath)
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            pdf.savefig(fig4)
            pdf.close()

            # Close figures to free memory
            plt.close(fig1)
            plt.close(fig2)

            return filepath
        finally:
            # Restore the original backend
            matplotlib.use(backend_original, force=True)


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
        self.depth_sensor_m = None  # Depth in meters as measured by a sensor
        self.orig_filepath = path
        self.filename = os.path.basename(path)
        self.sensor_id = None  # Will be set during XML parsing
        self.sensor = None  # Will reference the Sensor instance

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
        ) and (
            not hasattr(self.parent, "filename_prefix")
            or not self.parent.filename_prefix
        ):
            return self.orig_filepath
        else:
            return self._get_updated_filepath()

    @property
    def group(self):
        """Derive a grouping label from the camera's filepath.

        Rule:
        - Use the name of the immediate containing folder by default.
        - If that folder name contains a '.', use the text after the first '.'
          as the group name (e.g., "20241008.auv→ "auv")
        - Further split the group by prefixing with the first part of the filename
          if there is an underscore in the filename (e.g., "PR_20250608.jpg" → "PR").
        """
        try:
            folder = os.path.basename(os.path.dirname(self.filepath))
            if not folder:
                return None
            # Handle folder group extraction
            if "." in folder:
                parts = folder.split(".", 1)
                folder_group = parts[1] if len(parts) > 1 and parts[1] else folder
            else:
                folder_group = folder

            # Handle filename prefix extraction
            filename = self.filename
            if "_" in filename:
                filename_prefix = filename.split("_", 1)[0]
            else:
                filename_prefix = os.path.splitext(filename)[0]

            # Determine if there are multiple unique filename prefixes in the parent
            parent = getattr(self, "parent", None)
            unique_prefixes = set()
            if parent is not None and hasattr(parent, "data"):
                for cam in parent.data.values():
                    fname = cam.filename
                    if "_" in fname:
                        prefix = fname.split("_", 1)[0]
                    else:
                        prefix = os.path.splitext(fname)[0]
                    unique_prefixes.add(prefix)
            else:
                # If no parent, fallback to just this camera's prefix
                unique_prefixes.add(filename_prefix)

            if len(unique_prefixes) > 1 and "_" in filename:
                group = f"{folder_group}_{filename_prefix}"
            else:
                group = folder_group
            return group
        except Exception:
            return None

    @property
    def depth_in_m(self) -> float:
        """Depth in meters from the 3D regression (orig frame).

        Uses the regression model stored on the parent:
            depth ≈ depth_offset + up_vector · orig_coords

        Where ``up_vector`` is the 3-D coefficient vector returned by the
        regression (not necessarily unit-length), and ``orig_coords`` are the
        camera centers in the original coordinate frame used for the fit.

        Returns:
            float: Depth value in meters, or None if inputs are missing.
        """
        if self.orig_coords is None:
            print(f"No coordinates found for camera {self.cam_id}")
            return None

        if (
            not hasattr(self.parent, "up_vector")
            or self.parent.up_vector is None
            or not hasattr(self.parent, "depth_offset")
            or self.parent.depth_offset is None
        ):
            print("Cameras 'up_vector' and/or depth_offset not set")
            return None

        return float(
            self.parent.depth_offset
            + float(np.dot(self.parent.up_vector, self.orig_coords))
        )

    def transform_coords(self, transform_matrix):
        """Apply a transformation to the camera coordinates and transform.

        Args:
            transform_matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
        """
        if self.coords is not None:
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
        self.orig_camera_transform = np.dot(
            inverse_transform, self.camera_transform
        )  ### CHECK: changed this from camera_transform to orig_camera_transform

    def get_pixel_coords(
        self, coords, use_orig_coords=False, required_to_be_in_view=True
    ):
        """Compute the image pixel coordinates from original 3D coordinates.

        Projects a 3D point and applies lens distortion correction.

        Args:
            coords (list or np.ndarray): The original 3D point.
            use_orig_coords (bool): If True, use the original coordinates.
            required_to_be_in_view (bool): If True, only return the pixel coordinates if the point is in view.

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

        # Ensure array type and shape for inversion
        cam_transform = np.array(cam_transform, dtype=float).reshape((4, 4))

        proj_point = np.dot(np.linalg.inv(cam_transform), np.append(coords, 1))
        x_norm = proj_point[0] / proj_point[2]
        y_norm = proj_point[1] / proj_point[2]

        r2 = x_norm**2 + y_norm**2
        radial = (
            1 + self.sensor.k1 * r2 + self.sensor.k2 * r2**2 + self.sensor.k3 * r2**3
        )
        x_dist = (
            x_norm * radial
            + 2 * self.sensor.p1 * x_norm * y_norm
            + self.sensor.p2 * (r2 + 2 * x_norm**2)
        )
        y_dist = (
            y_norm * radial
            + self.sensor.p1 * (r2 + 2 * y_norm**2)
            + 2 * self.sensor.p2 * x_norm * y_norm
        )
        x_img = self.sensor.fx * x_dist + self.sensor.cx
        y_img = self.sensor.fy * y_dist + self.sensor.cy
        in_view = (
            r2 * self.sensor.fx**2 < 1.01 * self.sensor.width**2
            and 0 <= x_img <= self.sensor.width
            and 0 <= y_img <= self.sensor.height
        )
        if in_view or not required_to_be_in_view:
            dist_sq = np.sum((cam_coords - coords) ** 2)
            rm = np.abs(
                (np.abs(proj_point[2]) + dist_sq) / (self.sensor.fx * self.sensor.fy)
            ) * (
                10
                + np.abs(x_img - 0.5 * self.sensor.width)
                + np.abs(y_img - 0.5 * self.sensor.height)
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
        cx = self.sensor.cx
        cy = self.sensor.cy
        fx = self.sensor.fx
        fy = self.sensor.fy
        k1, k2, k3 = self.sensor.k1, self.sensor.k2, self.sensor.k3
        p1, p2 = self.sensor.p1, self.sensor.p2

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

        # Check for invalid values
        if np.any(np.isnan(vec_cam)) or np.any(np.isinf(vec_cam)):
            raise ValueError(
                f"Invalid camera vector computed for pixel ({x_img}, {y_img}): {vec_cam}"
            )

        vec_cam_norm = np.linalg.norm(vec_cam)
        if vec_cam_norm == 0:
            raise ValueError(f"Zero-length camera vector for pixel ({x_img}, {y_img})")

        vec_cam /= vec_cam_norm

        # Transform the direction vector from camera to world coordinates.
        # For directions, apply only the rotation part of self.transform.
        transform_matrix = np.array(self.camera_transform, dtype=float).reshape((4, 4))
        R = transform_matrix[:3, :3]
        vec_world = R.dot(vec_cam)

        vec_world_norm = np.linalg.norm(vec_world)
        if vec_world_norm == 0:
            raise ValueError(f"Zero-length world vector for pixel ({x_img}, {y_img})")

        vec_world /= vec_world_norm

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
            # If pixel coordinates are within the camera bounds
            if x is not None:

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
                else:
                    # If no pcd provided: add image_match regardless
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

    def get_datetime_original(self, offset_secs=None):
        """Retrieve the DateTimeOriginal from the image file EXIF data, with optional offset.

        Args:
            offset_secs (float or int, optional): If provided, apply this many seconds to the EXIF datetime.

        Returns:
            str or None: The (possibly offset) DateTimeOriginal value as a string, else None.
        """
        if not os.path.isfile(self.filepath):
            logger.error(f"Image file not found: {self.filepath}")
            return None

        # Try using exifread first (more comprehensive)
        try:
            with open(self.filepath, "rb") as f:
                tags = exifread.process_file(f, details=False)

                if "EXIF DateTimeOriginal" in tags:
                    dt_orig = str(tags["EXIF DateTimeOriginal"])
                    if offset_secs is not None:
                        dt = datetime.datetime.strptime(dt_orig, "%Y:%m:%d %H:%M:%S")
                        dt_offset = dt + datetime.timedelta(seconds=offset_secs)
                        return dt_offset.strftime("%Y:%m:%d %H:%M:%S")
                    else:
                        return dt_orig
                elif "Image DateTime" in tags:
                    dt_orig = str(tags["Image DateTime"])
                    if offset_secs is not None:
                        dt = datetime.datetime.strptime(dt_orig, "%Y:%m:%d %H:%M:%S")
                        dt_offset = dt + datetime.timedelta(seconds=offset_secs)
                        return dt_offset.strftime("%Y:%m:%d %H:%M:%S")
                    else:
                        return dt_orig
        except ImportError:
            pass
        except Exception as e:
            pass

        # Fallback to PIL
        image = Image.open(self.filepath)
        exif_data = image.getexif()
        if exif_data:
            dt_orig = exif_data.get(36867) or exif_data.get(306)
            if dt_orig:
                if offset_secs is not None:
                    dt = datetime.datetime.strptime(dt_orig, "%Y:%m:%d %H:%M:%S")
                    dt_offset = dt + datetime.timedelta(seconds=offset_secs)
                    return dt_offset.strftime("%Y:%m:%d %H:%M:%S")
                else:
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
        updated = self.orig_filepath

        # Apply filepath_replace if set on parent
        if (
            hasattr(self.parent, "filepath_replace")
            and self.parent.filepath_replace
            and len(self.parent.filepath_replace) >= 2
        ):
            find_str, replace_str = (
                self.parent.filepath_replace[0],
                self.parent.filepath_replace[1],
            )
            if find_str and replace_str:
                updated = updated.replace(find_str, replace_str)
            elif replace_str:
                updated = self.__replace_base_path(updated, replace_str)

        # Apply filename_prefix if set on parent
        filename_prefix = getattr(self.parent, "filename_prefix", None)
        if filename_prefix:
            dirname = os.path.dirname(updated)
            basename = os.path.basename(updated)
            updated = os.path.join(dirname, f"{filename_prefix}{basename}")

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
            # print("Warning: no neighboring point found")
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

    def classify(self, classifier, crop_size=None):
        """
        Classify the image match using a FastAI learner.

        Args:
            classifier: Loaded FastAI learner or path to a .pkl learner.
            crop_size: Optional int (square) or (width, height) tuple for center crop.

        Returns:
            dict: Classification result with label, confidence, probabilities, and pred_idx.
        """
        from substrata.classification import classify_image_match

        return classify_image_match(self, classifier, crop_size)

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
        # Print classification information if available
        if hasattr(self, "classification") and self.classification:
            cls = self.classification
            label = cls.get("label")
            conf = cls.get("confidence")
            print(f"Classification: {label}")
            if conf is not None:
                try:
                    print(f"Classification confidence: {float(conf):.3f}")
                except Exception:
                    print(f"Classification confidence: {conf}")
            probs = cls.get("probs")
            if isinstance(probs, dict) and len(probs) > 0:
                try:
                    top_items = sorted(
                        probs.items(), key=lambda kv: kv[1], reverse=True
                    )[:5]
                    print(
                        "Top probabilities: "
                        + ", ".join(
                            [
                                (
                                    f"{k}: {float(v):.3f}"
                                    if isinstance(v, (float, int))
                                    else f"{k}: {v}"
                                )
                                for k, v in top_items
                            ]
                        )
                    )
                except Exception:
                    pass
        if self.masks:
            cropped_img = visualizations.get_crop_img_from_masks(
                self, crop_w, crop_h, single_mask=single_mask
            )
        else:
            cropped_img = visualizations.get_crop_img(
                self.cam.filepath, self.x, self.y, crop_w, crop_h
            )
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(cropped_img)
        # plt.show()

        # Expecting PIL image, display directly
        plt.imshow(cropped_img)

        # Highlight the center pixel in red
        if hasattr(cropped_img, "size"):  # PIL image
            center_x = cropped_img.size[0] // 2
            center_y = cropped_img.size[1] // 2
        else:  # numpy array
            center_y, center_x = cropped_img.shape[:2]
            center_x = center_x // 2
            center_y = center_y // 2

        plt.plot(center_x, center_y, "ro", markersize=8)
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
