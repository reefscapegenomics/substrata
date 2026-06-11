from __future__ import annotations

# Standard Library
import copy
import csv
import datetime
import json
import logging
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

# Third-Party Libraries
import cv2
from joblib.externals.cloudpickle.cloudpickle import _property_reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from joblib import Parallel, delayed
import exifread

# Local Modules
from substrata import visualizations, settings, geom, measurements
from substrata.logging import tqdm_joblib

logger = logging.getLogger(__name__)


def _sync_camera_enabled_to_cams_xml(cameras: Cameras, xml_path: str) -> None:
    """Update ``<camera enabled="...">`` from in-memory :attr:`Camera.enabled`.

    Writes the XML file atomically. Intended for workflows such as ``camsync`` that
    re-enable cameras after pose transfer.

    Args:
        cameras: Loaded cameras whose ``enabled`` flags should be reflected in XML.
        xml_path: Path to the Metashape ``.cams.xml`` export.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError) as e:
        logger.warning("Could not parse %s for enabled sync: %s", xml_path, e)
        return
    root = tree.getroot()
    updated = 0
    for elem in root.findall(".//camera"):
        cid = elem.get("id")
        if cid is None or cid not in cameras.data:
            continue
        cam = cameras.data[cid]
        if not hasattr(cam, "enabled"):
            continue
        val = "true" if bool(cam.enabled) else "false"
        if elem.get("enabled") != val:
            elem.set("enabled", val)
            updated += 1
    if updated == 0:
        return
    xml_abs = os.path.abspath(xml_path)
    out_dir = os.path.dirname(xml_abs) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".xml.tmp", prefix=".cams_", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as xf:
            tree.write(xf, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, xml_abs)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    print(f"Updated {updated} <camera enabled> flags in {xml_abs}")


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

    def subset_by_dates(self, dates: list[str]) -> "Cameras":
        """Return a subset of cameras whose EXIF date matches one of ``dates``.

        Matches the first 10 characters of ``cam.datetime`` (``YYYY:MM:DD``).
        Input dates may use either ``YYYY-MM-DD`` or ``YYYY:MM:DD`` form and
        are normalized internally. Cameras without ``cam.datetime`` set are
        excluded. Assumes :meth:`get_datetime_originals` has already been
        called on this container.

        Args:
            dates: Iterable of date strings (``YYYY-MM-DD`` or ``YYYY:MM:DD``).

        Returns:
            Cameras: New container with cameras whose EXIF date matches.
        """
        wanted = {d.replace("-", ":")[:10] for d in dates}
        cameras_subset = self._empty_like()
        for cam in self.data.values():
            dt = getattr(cam, "datetime", None)
            if dt is None:
                continue
            if str(dt)[:10] in wanted:
                cameras_subset.data[cam.cam_id] = cam
                cam.parent = cameras_subset
        return cameras_subset

    def datetime_date_counts(self) -> dict[str, int]:
        """Count cameras per EXIF date (``YYYY:MM:DD``).

        Assumes :meth:`get_datetime_originals` has already been called.
        Cameras without ``cam.datetime`` set are skipped.

        Returns:
            dict[str, int]: Ordered mapping of ``YYYY:MM:DD`` to camera count,
            sorted ascending by date.
        """
        counts: dict[str, int] = {}
        for cam in self.data.values():
            dt = getattr(cam, "datetime", None)
            if dt is None:
                continue
            day = str(dt)[:10]
            counts[day] = counts.get(day, 0) + 1
        return dict(sorted(counts.items()))

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

        Entries with a valid ``path`` but null ``center``/``transform`` (e.g. disabled
        or unaligned in Metashape) are still loaded using a placeholder pose so they
        can receive poses later (e.g. ``camsync``). Such cameras have
        ``missing_pose_from_meta`` set to True.

        Args:
            cams_meta_filepath (str): Path to the JSON file with camera metadata.
        """
        with open(cams_meta_filepath, "r") as f:
            data = json.load(f)
        n_placeholder = 0
        for cam_id, cam_data in data["cameras"].items():
            if cam_data.get("path") is None or cam_data.get("path") == "null":
                continue
            center = cam_data.get("center")
            transform = cam_data.get("transform")
            pose_ok = (
                center is not None
                and center != "null"
                and transform is not None
                and transform != "null"
            )
            if pose_ok:
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
            else:
                self.data[cam_id] = Camera(
                    self,
                    cam_id,
                    np.eye(4).tolist(),
                    [0.0, 0.0, 0.0],
                    cam_data["path"],
                )
                self.data[cam_id].missing_pose_from_meta = True
                if "enabled" in cam_data:
                    self.data[cam_id].enabled = bool(cam_data["enabled"])
                n_placeholder += 1
        if n_placeholder:
            logger.info(
                "Loaded %s cameras with placeholder pose (null center/transform in meta)",
                n_placeholder,
            )

    def save(
        self,
        cams_meta_filepath: str | None = None,
        cams_xml_filepath: str | None = None,
    ) -> None:
        """Persist camera poses to the meta JSON file.

        Merges ``center`` and ``transform`` for each loaded camera from in-memory
        state (preferring ``orig_coords`` / ``orig_camera_transform`` when set).
        If a camera has an ``enabled`` attribute, it is written to JSON as well.
        Other keys for each camera entry are preserved from the existing file.

        Args:
            cams_meta_filepath: Output path. Defaults to :attr:`cams_meta_filepath`.
            cams_xml_filepath: Optional path to ``.cams.xml``. If provided (or set on
                the container) and the file exists, ``<camera enabled="...">`` is
                updated from each in-memory :attr:`Camera.enabled` (poses are not
                written to XML here).

        Raises:
            ValueError: If no output path is known or the file has no ``cameras`` key.
        """
        xml_path = cams_xml_filepath or getattr(self, "cams_xml_filepath", None)
        out_path = cams_meta_filepath or getattr(self, "cams_meta_filepath", None)
        if not out_path:
            raise ValueError("No cams_meta_filepath; pass cams_meta_filepath=...")

        with open(out_path, "r") as f:
            data = json.load(f)
        if "cameras" not in data:
            raise ValueError(f"Invalid cameras meta JSON (no 'cameras' key): {out_path}")

        cameras_out = copy.deepcopy(data["cameras"])
        for cam_id, cam in self.data.items():
            if cam_id not in cameras_out:
                logger.warning(
                    "Camera %s in memory but not in meta JSON; skipping save for it",
                    cam_id,
                )
                continue
            center = cam.orig_coords if cam.orig_coords is not None else cam.coords
            transform = (
                cam.orig_camera_transform
                if cam.orig_camera_transform is not None
                else cam.camera_transform
            )
            if center is None or transform is None:
                logger.warning("Skipping %s: missing center or transform", cam_id)
                continue
            cameras_out[cam_id]["center"] = np.asarray(center, dtype=float).tolist()
            cameras_out[cam_id]["transform"] = (
                np.asarray(transform, dtype=float).reshape(4, 4).tolist()
            )
            if hasattr(cam, "enabled"):
                cameras_out[cam_id]["enabled"] = bool(cam.enabled)

        data["cameras"] = cameras_out

        out_abs = os.path.abspath(out_path)
        out_dir = os.path.dirname(out_abs) or "."
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp", prefix=".cams_meta_", dir=out_dir
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, out_abs)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        print(f"Saved camera metadata to {out_abs}")

        if xml_path and os.path.isfile(xml_path):
            _sync_camera_enabled_to_cams_xml(self, xml_path)

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
                pass  # Too many warnings for some projects - TODO
                # logger.warning(f"Camera {cam_id} from XML not found in loaded cameras")

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
        """Adopt centers and transforms from ``other_cams`` via exact datetime match.

        Cameras in ``self`` whose ``datetime`` has no exact string match in
        ``other_cams`` are left untouched, tagged with
        ``missing_pose_from_timematch = True``, and reported via a single
        ``logger.warning``. A ``ValueError`` is raised only if zero cameras
        matched (which almost always indicates a wrong ``--time-offset``).

        Args:
            other_cams: The other ``Cameras`` to match against.
            offset_xyz: ``[x, y, z]`` offset in the camera's local frame, applied
                to the matched pose before writing back to ``self``.

        Returns:
            tuple[list[str], list[str]]: ``(matched_cam_ids, unmatched_cam_ids)``.
        """
        n_matched = 0
        matched_ids: list[str] = []
        unmatched: list[tuple[str, str]] = []  # (cam_id, datetime)
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
                if hasattr(cam, "missing_pose_from_meta"):
                    delattr(cam, "missing_pose_from_meta")
                if hasattr(cam, "missing_pose_from_timematch"):
                    delattr(cam, "missing_pose_from_timematch")
                matched_ids.append(str(cam.cam_id))
                n_matched += 1
            else:
                cam.missing_pose_from_timematch = True
                unmatched.append((str(cam.cam_id), str(cam.datetime)))

        unmatched_ids = [cid for cid, _ in unmatched]

        if unmatched:
            from substrata.firefish import get_time_diff_in_secs

            other_dts = sorted(
                str(c.datetime)
                for c in other_cams.data.values()
                if getattr(c, "datetime", None) is not None
            )
            first_id, first_dt = unmatched[0]
            closest_info = ""
            if other_dts:
                try:
                    diffs = [
                        (abs(get_time_diff_in_secs(d, first_dt)), d) for d in other_dts
                    ]
                    _, closest = min(diffs, key=lambda x: x[0])
                    signed = get_time_diff_in_secs(closest, first_dt)
                    closest_info = (
                        f"\n  Closest other_cams datetime to first unmatched: "
                        f"{closest} (off by {signed:+.1f}s)."
                    )
                except Exception:
                    pass
            range_info = (
                f"\n  other_cams datetime range: [{other_dts[0]} .. {other_dts[-1]}] "
                f"(n={len(other_dts)})"
                if other_dts
                else "\n  other_cams have no datetimes set."
            )
            preview = ", ".join(f"{cid}@{dt}" for cid, dt in unmatched[:6])
            more = "" if len(unmatched) <= 6 else f", ... (+{len(unmatched) - 6} more)"
            unmatched_list = f"\n  Unmatched: {preview}{more}"
            summary = (
                f"Time-match: {len(unmatched)}/{len(self.data)} target cameras "
                f"had no exact datetime match in other_cams "
                f"(matched {n_matched}/{len(self.data)})."
                f"{range_info}"
                f"{closest_info}"
                f"{unmatched_list}"
                "\n  Matching is exact on DateTimeOriginal; these cameras may "
                "fall outside the other_cams time range (e.g. source video "
                "cut out early) or indicate a wrong --time-offset."
            )
            if n_matched == 0:
                raise ValueError(
                    "Time-match failed: 0 cameras matched. " + summary
                )
            logger.warning(
                "%s\n  Unmatched cameras were left untouched and tagged with "
                "missing_pose_from_timematch=True.",
                summary,
            )

        return matched_ids, unmatched_ids

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

    def earliest_exif_datetime(
        self, cam_ids: list[str] | None = None
    ) -> tuple[str | None, str | None]:
        """Earliest EXIF DateTimeOriginal among cameras (does not set ``cam.datetime``).

        Args:
            cam_ids: Camera ids to consider; default is all keys in ``self.data``.

        Returns:
            Tuple ``(earliest_datetime_str, cam_id)``, or ``(None, None)`` if none found.
        """
        from substrata import firefish

        ids = cam_ids if cam_ids is not None else list(self.data.keys())
        best_t: float | None = None
        best_dt: str | None = None
        best_id: str | None = None
        for cid in ids:
            cam = self.data.get(cid)
            if cam is None:
                continue
            dt = cam.get_datetime_original(None)
            if dt is None:
                continue
            t = firefish.get_unix_time(dt)
            if best_t is None or t < best_t:
                best_t = t
                best_dt = dt
                best_id = cid
        return (best_dt, best_id)

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

    @property
    def depth_residuals(self):
        """Access depth residual analysis methods."""
        from substrata.measurements import DepthResidualAnalyzer

        return DepthResidualAnalyzer(self)

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
        return self.depth_residuals.get_depths_and_estimated_depths(
            depth_accuracy_threshold=depth_accuracy_threshold,
            use_accuracy_filter=True,
        )

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
        return self.depth_residuals.get_depths_and_z_coords(
            depth_accuracy_threshold=depth_accuracy_threshold,
            use_accuracy_filter=True,
        )

    def show_depth_vs_est_depth_residuals(self, width=15, height=5):
        """Show residuals between predicted depth_in_m and the original recorded camera depths.

        These residuals are calculated based on recorded camera depths and the predicted depths from the regression model.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
        """
        return self.depth_residuals.show_depth_vs_est_depth_residuals(
            width=width,
            height=height,
            depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
            use_accuracy_filter=True,
        )

    def show_z_vs_depth_residuals(self, width=15, height=5):
        """Show residuals between camera z-coordinates and the original recorded camera depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)
        """
        return self.depth_residuals.show_z_vs_depth_residuals(
            width=width,
            height=height,
            depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
            use_accuracy_filter=True,
        )

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
        return self.depth_residuals.save_depth_residuals_pdf(
            filepath=filepath,
            width=width,
            height=height,
            depth_accuracy_threshold=settings.DEFAULT_DEPTH_ACCURACY_THRESHOLD,
            use_accuracy_filter=True,
        )


def orthonormal_rotation_from_camera_transform(
    camera_transform: np.ndarray | list,
) -> np.ndarray:
    """Return a proper 3x3 rotation from a 4x4 camera transform (same logic as camsync)."""
    rotation_matrix = np.asarray(camera_transform, dtype=float).reshape(4, 4)[:3, :3]
    u, _, vt = np.linalg.svd(rotation_matrix)
    r_mat = u @ vt
    if np.linalg.det(r_mat) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        r_mat = u @ vt
    return r_mat


def spatial_nearest_time_offset_report(
    target_cams: Cameras,
    pose_cams: Cameras,
    *,
    spatial_max_dist_m: float,
    min_pairs: int,
    scale_factor: float | None = 1.0,
) -> dict:
    """Estimate EXIF time offset from nearest-neighbor pose cameras in 3D (same chunk).

    Assumes both subsets share one reconstruction frame. ``spatial_max_dist_m`` is
    compared to ``euclidean(center_T, center_P) * scale_factor`` (metric distance in
    meters if coords are model units and ``scale_factor`` converts to meters).

    Args:
        target_cams: Cameras to update (e.g. macro); must have ``cam.coords`` and EXIF
            times available (e.g. via ``get_datetime_originals()``).
        pose_cams: Pose source cameras (e.g. GoPro).
        spatial_max_dist_m: Max metric distance (m) for a pair to count as inlier.
        min_pairs: Minimum number of inlier pairs with valid times required for ``ok``.
        scale_factor: Multiplier from stored coords to meters (1.0 if already metric).

    Returns:
        Dict with ``ok``, ``median_k_sec``, ``pairs`` (per-target rows), ``stats``,
        ``n_inliers``, ``n_targets``, and optional ``reason`` if not ok.
    """
    from substrata import firefish

    sf = float(scale_factor) if scale_factor is not None else 1.0
    pose_list = list(pose_cams.data.values())
    n_targets = len(target_cams.data)
    if not pose_list or n_targets == 0:
        return {
            "ok": False,
            "median_k_sec": None,
            "pairs": [],
            "stats": {},
            "n_inliers": 0,
            "n_targets": n_targets,
            "reason": "empty pose or target subset",
        }

    centers_pose = np.array(
        [np.asarray(c.coords, dtype=float).ravel()[:3] for c in pose_list]
    )
    tree = cKDTree(centers_pose)
    pairs: list[dict] = []
    k_inliers: list[float] = []

    for tcam in target_cams.data.values():
        if getattr(tcam, "missing_pose_from_meta", False):
            dt_t = getattr(tcam, "datetime", None) or tcam.get_datetime_original(
                None
            )
            pairs.append(
                {
                    "target_id": tcam.cam_id,
                    "pose_id": None,
                    "dist_stored": None,
                    "dist_metric_m": None,
                    "inlier": False,
                    "dt_target": dt_t,
                    "dt_pose": None,
                    "k_sec": None,
                    "skip_reason": "missing_pose_in_meta",
                }
            )
            continue
        ct = np.asarray(tcam.coords, dtype=float).ravel()[:3]
        dist_stored, j = tree.query(ct, k=1)
        pcam = pose_list[int(j)]
        dist_metric_m = float(dist_stored) * sf
        inlier = dist_metric_m <= float(spatial_max_dist_m)

        dt_t = getattr(tcam, "datetime", None) or tcam.get_datetime_original(None)
        dt_p = getattr(pcam, "datetime", None) or pcam.get_datetime_original(None)
        k_sec: float | None = None
        if dt_t is not None and dt_p is not None and inlier:
            k_sec = float(
                firefish.get_unix_time(dt_p) - firefish.get_unix_time(dt_t)
            )
            k_inliers.append(k_sec)

        pairs.append(
            {
                "target_id": tcam.cam_id,
                "pose_id": pcam.cam_id,
                "dist_stored": float(dist_stored),
                "dist_metric_m": dist_metric_m,
                "inlier": inlier,
                "dt_target": dt_t,
                "dt_pose": dt_p,
                "k_sec": k_sec,
            }
        )

    stats: dict[str, float | None] = {}
    median_k: float | None = None
    if k_inliers:
        arr = np.array(k_inliers, dtype=float)
        median_k = float(np.median(arr))
        stats = {
            "median": median_k,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    n_inliers = len(k_inliers)
    ok = n_inliers >= int(min_pairs) and median_k is not None
    reason = None
    if not ok:
        reason = (
            f"need >= {min_pairs} inlier pairs with valid EXIF; got {n_inliers}"
        )

    return {
        "ok": ok,
        "median_k_sec": median_k,
        "pairs": pairs,
        "stats": stats,
        "n_inliers": n_inliers,
        "n_targets": n_targets,
        "reason": reason,
    }


def xyz_offset_datetime_matches_report(
    target_cams: Cameras,
    pose_cams: Cameras,
    *,
    scale_factor: float | None = 1.0,
) -> dict:
    """Estimate median pose-local ``[x,y,z]`` offset from time-matched camera pairs.

    For each target camera with ``cam.datetime``, finds the pose camera with the same
    datetime. Computes ``delta_world = (C_pose - C_target) * scale_factor`` (scale
    when coords are in model units), then ``offset_xyz = R^T @ delta_world`` using the
    orthonormalized rotation from the pose camera transform (matches
    ``get_centers_and_transforms_based_on_timematch``).

    Args:
        target_cams: Target subset (datetimes must match pose after time offset).
        pose_cams: Pose source subset.
        scale_factor: Multiply center difference by this (1.0 if coords already meters).

    Returns:
        Dict with ``ok``, ``median_xyz``, ``mean_xyz``, ``rows`` (one dict per target),
        and ``unmatched_ids`` if any target has no pose at same datetime.
    """
    sf = float(scale_factor) if scale_factor is not None else 1.0
    rows: list[dict] = []
    locals_list: list[np.ndarray] = []
    unmatched: list[str] = []

    for tcam in target_cams.data.values():
        pose = pose_cams.get_camera_by_datetime(tcam.datetime)
        if pose is None:
            unmatched.append(str(tcam.cam_id))
            rows.append(
                {
                    "target_id": tcam.cam_id,
                    "pose_id": None,
                    "error": "no pose camera with same datetime",
                }
            )
            continue
        if getattr(tcam, "missing_pose_from_meta", False):
            rows.append(
                {
                    "target_id": tcam.cam_id,
                    "pose_id": pose.cam_id,
                    "skip_reason": (
                        "missing_pose_in_meta (median xyz uses cameras with pose only; "
                        "this camera omitted)"
                    ),
                }
            )
            continue
        ct = np.asarray(tcam.coords, dtype=float).ravel()[:3]
        cp = np.asarray(pose.coords, dtype=float).ravel()[:3]
        delta_w = (cp - ct) * sf
        r_mat = orthonormal_rotation_from_camera_transform(pose.camera_transform)
        off = r_mat.T @ delta_w
        locals_list.append(off)
        rows.append(
            {
                "target_id": tcam.cam_id,
                "pose_id": pose.cam_id,
                "delta_world": delta_w.tolist(),
                "offset_xyz_local": off.tolist(),
            }
        )

    ok = len(unmatched) == 0
    median_xyz: list[float] | None = None
    mean_xyz: list[float] | None = None
    if locals_list:
        arr = np.stack(locals_list, axis=0)
        median_xyz = np.median(arr, axis=0).tolist()
        mean_xyz = np.mean(arr, axis=0).tolist()
    elif ok:
        median_xyz = [0.0, 0.0, 0.0]
        mean_xyz = [0.0, 0.0, 0.0]

    reason = None if ok else f"unmatched target ids: {unmatched}"

    return {
        "ok": ok,
        "median_xyz": median_xyz,
        "mean_xyz": mean_xyz,
        "rows": rows,
        "unmatched_ids": unmatched,
        "reason": reason,
    }


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

    def render(self, highlight_pixels=None, orient=False, square=False,
               highlight_radius: int = 50, highlight_outline_width: int = 10):
        """Return the camera image as a PIL Image with optional highlight markers.

        Args:
            highlight_pixels: A single ``[x, y]`` coordinate or an ``(N, 2)``
                array of pixel coordinates to highlight with red circles.
            orient: If True, rotate the image so that the world X-axis points
                right, making the image orientation roughly consistent with a
                top-down OrthoMap.  The rotation angle is derived from the
                camera_transform: the image x-axis (``T[:3, 0]``) and y-axis
                (``T[:3, 1]``) are used to project the world X direction
                ``[1, 0, 0]`` onto the image plane; ``atan2`` then gives the
                clockwise offset from image-right, which is corrected by a
                counter-clockwise PIL rotation.
            square: If True, crop the (possibly rotated) image to the largest
                possible square centred on the image centre.  When
                ``orient=True`` the square is the largest axis-aligned square
                that fits within the actual image content (no rotation
                whitespace), computed analytically from the original image
                dimensions and rotation angle.  A warning is logged if any
                highlight falls outside the resulting square.
            highlight_radius: Radius in pixels of the circle drawn at each
                highlight position (default 50).
            highlight_outline_width: Stroke width in pixels of the circle
                outline (default 10, matching the selected-mask contour
                thickness used in ``get_crop_img_from_masks``).

        Returns:
            PIL Image, or None if the file is missing or unreadable.
        """
        if not self.filepath or not os.path.isfile(self.filepath):
            logger.warning("Camera.render: file not found: %s", self.filepath)
            return None
        try:
            image = Image.open(self.filepath).convert("RGB")
        except OSError as e:
            logger.warning("Camera.render: could not open %s: %s", self.filepath, e)
            return None

        orig_W, orig_H = image.size

        # Parse highlight pixels into a float (N, 2) array for coordinate tracking.
        hp_draw = None
        is_single = False
        if highlight_pixels is not None:
            hp_draw = np.array(highlight_pixels, dtype=float)
            if hp_draw.ndim == 1:
                hp_draw = hp_draw[np.newaxis, :]
                is_single = True

        # ── Step 1: orientation rotation ─────────────────────────────────
        angle_deg = 0.0
        if orient and self.camera_transform is not None:
            try:
                T = np.array(self.camera_transform, dtype=float).reshape((4, 4))
                cam_right = T[:3, 0] / np.linalg.norm(T[:3, 0])
                cam_down = T[:3, 1] / np.linalg.norm(T[:3, 1])
                world_x = np.array([1.0, 0.0, 0.0])
                angle_deg = float(np.degrees(
                    np.arctan2(np.dot(world_x, cam_down), np.dot(world_x, cam_right))
                ))
            except Exception as e:
                logger.warning("Camera.render: orient failed: %s", e)

        if angle_deg != 0.0:
            image = image.rotate(angle_deg, expand=True)
            rot_W, rot_H = image.size
            # Transform highlight pixels from original to rotated canvas coordinates.
            if hp_draw is not None:
                a_rad = np.radians(angle_deg)
                cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
                dx = hp_draw[:, 0] - orig_W / 2.0
                dy = hp_draw[:, 1] - orig_H / 2.0
                hp_draw = np.column_stack([
                    dx * cos_a + dy * sin_a + rot_W / 2.0,
                    -dx * sin_a + dy * cos_a + rot_H / 2.0,
                ])
        else:
            rot_W, rot_H = orig_W, orig_H

        # ── Step 2: square crop ───────────────────────────────────────────
        crop_offset_x, crop_offset_y = 0, 0
        if square:
            if angle_deg != 0.0:
                # Largest axis-aligned square inscribed in the rotated content
                # (avoids whitespace corner pixels).  Derived from the constraint
                # that all four corners of the square must map back into the
                # original W×H image when un-rotated:
                #   s ≤ min(W, H) / (|cos a| + |sin a|)
                #   s ≤ max(W, H) / ||cos a| - |sin a||
                a_abs = abs(np.radians(angle_deg)) % (np.pi / 2)
                cos_a, sin_a = np.cos(a_abs), np.sin(a_abs)
                diff_cs = abs(cos_a - sin_a)
                s1 = min(orig_W, orig_H) / (cos_a + sin_a)
                s2 = max(orig_W, orig_H) / diff_cs if diff_cs > 1e-8 else float("inf")
                crop_size = int(min(s1, s2))
            else:
                crop_size = min(rot_W, rot_H)

            cx, cy = rot_W / 2.0, rot_H / 2.0

            half = crop_size // 2
            left = int(max(0, min(round(cx - half), rot_W - crop_size)))
            top = int(max(0, min(round(cy - half), rot_H - crop_size)))
            right = left + crop_size
            bottom = top + crop_size

            if hp_draw is not None:
                outside = (
                    (hp_draw[:, 0] < left) | (hp_draw[:, 0] >= right)
                    | (hp_draw[:, 1] < top) | (hp_draw[:, 1] >= bottom)
                )
                if outside.any():
                    logger.warning(
                        "Camera.render: %d highlight(s) fall outside the square crop",
                        int(outside.sum()),
                    )

            image = image.crop((left, top, right, bottom))
            crop_offset_x, crop_offset_y = left, top

        # ── Step 3: draw highlights ───────────────────────────────────────
        if hp_draw is not None:
            draw = ImageDraw.Draw(image)
            for pixel in hp_draw:
                x = int(round(pixel[0])) - crop_offset_x
                y = int(round(pixel[1])) - crop_offset_y
                draw.ellipse((x - highlight_radius, y - highlight_radius,
                              x + highlight_radius, y + highlight_radius),
                             fill=None, outline=(255, 0, 0),
                             width=highlight_outline_width)

        return image

    def show(self, highlight_pixels=None, orient=False, square=False):
        """Display the camera image with optional highlight markers."""
        img = self.render(highlight_pixels=highlight_pixels, orient=orient, square=square)
        if img is not None:
            plt.imshow(np.array(img))
            plt.axis("off")
            plt.show()

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

        # Final fallback: parse "..._YYYYMMDD_HHMMSS" suffix from the filename
        # (e.g. for ffmpeg-extracted video frames that carry no EXIF segment).
        match = re.search(
            r"_(\d{8})_(\d{6})(?=\.[^.]+$|$)", os.path.basename(self.filepath)
        )
        if match:
            try:
                dt = datetime.datetime.strptime(
                    match.group(1) + match.group(2), "%Y%m%d%H%M%S"
                )
                if offset_secs is not None:
                    dt = dt + datetime.timedelta(seconds=offset_secs)
                return dt.strftime("%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

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


class LocalMask:
    """Lightweight mask compatible with the segmentation.py Mask class.

    Must live at module level so that instances (held by annotations) remain
    picklable when sent to joblib worker processes.

    Args:
        mask_vals: Binary mask array (uint8, non-zero inside the mask).
        score: Confidence score of the mask.
        logits: Optional raw logits associated with the mask.
    """

    def __init__(self, mask_vals: np.ndarray, score: float = 1.0, logits=None):
        self.vals = mask_vals
        self.score = score
        self.logits = logits
        self.area_in_px = cv2.countNonZero(mask_vals)
        self.area_in_cm2 = None
        self.radius_m = None  # set for circular masks only


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
        if self.pixel_scale is None:
            self.calc_pixel_scale_from_crosshair()
        return 1.0 / (self.pixel_scale * 1000.0)

    def get_sam2_masks(self, sam_predictor):
        """
        Get the SAM2 masks for the annotation in the image.

        Args:
            sam_predictor: Predictor object for SAM2 segmentation.
        """
        from substrata import segmentation

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

    def create_rectangular_masks(self, mask_sizes):
        """
        Create rectangular masks of specified sizes around the match point.

        Args:
            mask_sizes (list): List of [width_m, height_m] tuples specifying mask
                sizes in meters. Each entry creates a separate mask.

        Returns:
            None: Masks are stored in self.masks, with the first mask also
                assigned to self.mask.
        """
        if not self.pixel_scale:
            self.calc_pixel_scale_from_crosshair()

        if not self.pixel_scale:
            raise ValueError(
                "Pixel scale not available. Call calc_pixel_scale_from_crosshair() first."
            )

        # Validate input
        if not mask_sizes or len(mask_sizes) == 0:
            raise ValueError("mask_sizes must be a non-empty list")

        # Get image dimensions
        img = cv2.imread(self.filepath)
        if img is None:
            raise ValueError(f"Cannot load image: {self.filepath}")

        img_height, img_width = img.shape[:2]

        # Create masks for each size
        mask_objects = []
        for idx, (width_m, height_m) in enumerate(mask_sizes):
            # Convert meters to pixels using pixel scale
            width_px = round(width_m / self.pixel_scale)
            height_px = round(height_m / self.pixel_scale)

            # Calculate rectangle bounds
            x1 = max(0, self.x - width_px // 2)
            y1 = max(0, self.y - height_px // 2)
            x2 = min(img_width, self.x + width_px // 2)
            y2 = min(img_height, self.y + height_px // 2)

            # Create binary mask (use uint8 instead of bool for OpenCV compatibility)
            mask_vals = np.zeros((img_height, img_width), dtype=np.uint8)
            mask_vals[y1:y2, x1:x2] = 255

            # Create Mask object
            mask_obj = LocalMask(mask_vals, score=1.0)
            mask_obj.area_in_px = cv2.countNonZero(mask_vals)
            mask_obj.area_in_cm2 = mask_obj.area_in_px * (self.pixel_scale**2) * 10000

            mask_objects.append(mask_obj)

            print(f"Created rectangular mask {idx+1}: {width_m:.2f}m x {height_m:.2f}m")
            print(f"Mask bounds: ({x1}, {y1}) to ({x2}, {y2})")
            print(
                f"Mask area: {mask_obj.area_in_px} pixels = {mask_obj.area_in_cm2/10000:.4f} m²"
            )

        # Store all masks
        self.masks = np.array(mask_objects)
        # Assign the first mask as the primary mask
        self.mask = mask_objects[0]

    def create_circular_masks(self, radii):
        """Create circular masks of specified radii around the match point.

        Args:
            radii (list): List of radii in meters. Each entry creates a separate mask.

        Returns:
            None: Masks are stored in self.masks, with the first mask also
                assigned to self.mask.
        """
        if not self.pixel_scale:
            self.calc_pixel_scale_from_crosshair()

        if not self.pixel_scale:
            raise ValueError(
                "Pixel scale not available. Call calc_pixel_scale_from_crosshair() first."
            )

        if not radii or len(radii) == 0:
            raise ValueError("radii must be a non-empty list")

        img = cv2.imread(self.filepath)
        if img is None:
            raise ValueError(f"Cannot load image: {self.filepath}")

        img_height, img_width = img.shape[:2]

        if not all(isinstance(r, (int, float)) for r in radii):
            raise ValueError("radii must be a list of scalar values (e.g. [0.1, 0.2])")

        mask_objects = []
        for idx, radius_m in enumerate(radii):
            radius_px = round(radius_m / self.pixel_scale)

            mask_vals = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.circle(mask_vals, (self.x, self.y), radius_px, 255, thickness=-1)

            mask_obj = LocalMask(mask_vals, score=1.0)
            mask_obj.radius_m = radius_m
            mask_obj.area_in_px = cv2.countNonZero(mask_vals)
            mask_obj.area_in_cm2 = mask_obj.area_in_px * (self.pixel_scale**2) * 10000

            mask_objects.append(mask_obj)

            print(f"Created circular mask {idx+1}: radius {radius_m:.4f} m")
            print(f"Center: ({self.x}, {self.y}), radius: {radius_px} px")
            print(
                f"Mask area: {mask_obj.area_in_px} pixels = {mask_obj.area_in_cm2/10000:.4f} m²"
            )

        self.masks = np.array(mask_objects)
        self.mask = mask_objects[0]

    def render(self, crop_w=1000, crop_h=1000, single_mask=False, orient=False,
               circular_image_mask=None):
        """Return the cropped image match as a PIL Image.

        Args:
            crop_w: Crop width in pixels (default 1000).
            crop_h: Crop height in pixels (default 1000).
            single_mask: Whether to use only the primary mask when masks are present.
            orient: If True, rotate so that the world X-axis points right, using
                the camera_transform from :attr:`cam`.
            circular_image_mask: Radius in pixels of a circular mask centred on
                the annotation.  Pixels outside the circle are replaced with a
                grey background.  Provide this only when no circular mask has been
                set via :meth:`create_circular_masks`; when one is present its
                radius is used automatically.

        Returns:
            PIL Image.
        """
        # Detect a circular mask created by create_circular_masks (has radius_m).
        # Such masks use the fixed-center crop path + automatic grey-out rather
        # than the SAM2 contour-overlay path.
        is_circular = (
            getattr(self, "mask", None) is not None
            and hasattr(self.mask, "radius_m")
            and self.mask.radius_m is not None
        )

        angle_deg = 0.0
        if orient and getattr(self.cam, "camera_transform", None) is not None:
            try:
                T = np.array(self.cam.camera_transform, dtype=float).reshape((4, 4))
                cam_right = T[:3, 0] / np.linalg.norm(T[:3, 0])
                cam_down = T[:3, 1] / np.linalg.norm(T[:3, 1])
                world_x = np.array([1.0, 0.0, 0.0])
                angle_deg = float(np.degrees(
                    np.arctan2(np.dot(world_x, cam_down), np.dot(world_x, cam_right))
                ))
            except Exception as e:
                logger.warning("ImageMatch.render: orient failed: %s", e)

        if getattr(self, "masks", None) is not None and len(self.masks) > 0 and not is_circular:
            if angle_deg != 0.0:
                # Inflate the canvas so that after rotating in-place (expand=False)
                # the central crop_w × crop_h region has no black corners.
                # pad_ratio shrinks the mask within the inflated canvas so it still
                # occupies crop_w pixels — exactly matching the final crop size.
                a_abs = abs(np.radians(angle_deg)) % (np.pi / 2)
                cos_a, sin_a = np.cos(a_abs), np.sin(a_abs)
                inflate = int(max(crop_w, crop_h) * (cos_a + sin_a)) + 2
                rotate_pad = (cos_a + sin_a - 1.0) / 2.0
                arr = visualizations.get_crop_img_from_masks(
                    self, inflate, inflate, single_mask=single_mask,
                    pad_ratio=rotate_pad,
                )
                pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                pil_img = pil_img.rotate(angle_deg, expand=False)
                iw, ih = pil_img.size
                left = (iw - crop_w) // 2
                top = (ih - crop_h) // 2
                pil_img = pil_img.crop((left, top, left + crop_w, top + crop_h))
            else:
                arr = visualizations.get_crop_img_from_masks(
                    self, crop_w, crop_h, single_mask=single_mask
                )
                pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        else:
            pil_img = visualizations.get_crop_img(
                self.cam.filepath, self.x, self.y, crop_w, crop_h
            )
            if not isinstance(pil_img, Image.Image):
                pil_img = Image.fromarray(pil_img)
            if angle_deg != 0.0:
                orig_W, orig_H = pil_img.size
                pil_img = pil_img.rotate(angle_deg, expand=True)
                pil_img = self._inscribed_square_crop(pil_img, orig_W, orig_H, angle_deg)
            draw = ImageDraw.Draw(pil_img)
            cx, cy = pil_img.size[0] // 2, pil_img.size[1] // 2
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8],
                         fill=(255, 0, 0), outline=(0, 0, 0))

        # Apply grey-out circle: prefer the stored circular-mask radius, fall back
        # to the explicit circular_image_mask argument (pixels).
        grey_r: int | None = None
        if is_circular and self.pixel_scale is not None:
            grey_r = int(self.mask.radius_m / self.pixel_scale)
        elif circular_image_mask is not None:
            grey_r = int(circular_image_mask)
        if grey_r is not None:
            cx, cy = pil_img.size[0] // 2, pil_img.size[1] // 2
            alpha = Image.new("L", pil_img.size, 0)
            ImageDraw.Draw(alpha).ellipse(
                [cx - grey_r, cy - grey_r, cx + grey_r, cy + grey_r], fill=255
            )
            grey = Image.new("RGB", pil_img.size, (180, 180, 180))
            pil_img = Image.composite(pil_img, grey, alpha)

        return pil_img

    @staticmethod
    def _inscribed_square_crop(
        img: "Image.Image",
        orig_W: int,
        orig_H: int,
        angle_deg: float,
    ) -> "Image.Image":
        """Return the largest whitespace-free square crop of a rotated image."""
        a_abs = abs(np.radians(angle_deg)) % (np.pi / 2)
        cos_a, sin_a = np.cos(a_abs), np.sin(a_abs)
        diff_cs = abs(cos_a - sin_a)
        s = int(min(
            min(orig_W, orig_H) / (cos_a + sin_a),
            max(orig_W, orig_H) / max(diff_cs, 1e-8),
        ))
        rot_W, rot_H = img.size
        left = (rot_W - s) // 2
        top = (rot_H - s) // 2
        return img.crop((left, top, left + s, top + s))

    def show(
        self,
        crop_w=1000,
        crop_h=1000,
        single_mask=False,
        orient=False,
    ):
        """Display the image match and its attributes."""
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
        img = self.render(
            crop_w=crop_w, crop_h=crop_h, single_mask=single_mask,
            orient=orient,
        )

        plt.imshow(np.array(img))
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
