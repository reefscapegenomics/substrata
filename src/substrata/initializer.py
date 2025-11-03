# Standard Library
import os

# Third-Party Libraries
import yaml
import numpy as np

# Local Modules
from substrata import (
    annotations,
    cameras,
    pointclouds,
    geom,
    settings,
)


class ProjectInitializer:
    """
    Factory class that spawns a PointCloud, Cameras, and Annotations instance.

    """

    def __init__(self, yaml=None, path=None):
        # Check that either yaml_filepath or project_path is provided
        if yaml is None and path is None:
            raise ValueError("Either yaml or path must be provided.")

        # Initialize all attributes to None
        self.path = None
        self.yaml_path = None
        self.id = None
        self.ply_filepath = None
        self.ply_dec_path = None
        self.ply_full_path = None
        self.cams_xml_filepath = None
        self.cams_meta_json_filepath = None
        self.markers_filepath = None
        self.annotations_filepath = None
        self.annotations_last_highest_id = None
        self.photos_path = None
        self.cropped_path = None
        self.thumbnail_path = None
        self.classes_filepath = None
        self.world_transform = np.eye(4)
        self.scale_factor = None
        self.up_vector = None
        self.depth_offset = None
        self.depth_per_unit = None

        # If YAML file is provided, read the YAML file to establish configuration,
        # and overwrite any values set by the path.
        if yaml is not None:
            self.init_with_yaml(yaml)
        # If project_path is given, search for files with default naming conventions
        # in the directory to establish configuration.
        else:
            current_folder_name = os.path.basename(os.path.abspath(path))
            yaml_path = os.path.join(path, f"{current_folder_name}.yaml")
            if os.path.isfile(yaml_path):
                self.init_with_yaml(yaml_path)
            else:
                self.init_with_path(path)

    def __str__(self) -> str:
        """
        Returns a summary of all the variables set by the initializer.
        """
        lines = ["ProjectInitializer("]

        # Only show attributes that are set (not None)
        if self.path is not None:
            lines.append(f"  path={self.path},")
        if self.yaml_path is not None:
            lines.append(f"  yaml={self.yaml_path},")
        if self.ply_filepath is not None:
            lines.append(f"  ply_filepath={self.ply_filepath},")
        if self.cams_xml_filepath is not None:
            lines.append(f"  cams_xml_filepath={self.cams_xml_filepath},")
        if self.cams_meta_json_filepath is not None:
            lines.append(f"  cams_meta_json_filepath={self.cams_meta_json_filepath},")
        if self.markers_filepath is not None:
            lines.append(f"  markers_filepath={self.markers_filepath},")
        if self.annotations_filepath is not None:
            lines.append(f"  annotations_filepath={self.annotations_filepath},")
        if self.annotations_last_highest_id is not None:
            lines.append(
                f"  annotations_last_highest_id={self.annotations_last_highest_id},"
            )
        if self.photos_path is not None:
            lines.append(f"  photos_path={self.photos_path},")
        if self.cropped_path is not None:
            lines.append(f"  cropped_path={self.cropped_path},")
        if self.thumbnail_path is not None:
            lines.append(f"  thumbnail_path={self.thumbnail_path},")
        if self.classes_filepath is not None:
            lines.append(f"  classes_filepath={self.classes_filepath},")
        if self.world_transform is not None:
            lines.append(f"  world_transform={self.world_transform},")
        lines.append(f"  scale_factor={self.scale_factor}")

        # Remove trailing comma from last line if present
        if len(lines) > 1 and lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]

        lines.append(")")
        return "\n".join(lines)

    def init_with_yaml(self, filepath):
        """
        Establish configuration by reading the YAML file

        Example YAML file:
        path: "/Users/pbongaerts/Github/unicorn/examples/ton_tof/ton_tof_60m/ton_tof_60m_20241008/"
        id: "ton_tof_60m_20241008"
        ply: "ton_tof_60m_20241008_dec50M.ply"
        cams_xml: "ton_tof_60m_20241008.cams.xml"
        cams_meta_json: "ton_tof_60m_20241008.meta.json"
        annotations: "ton_tof_60m_20241008_DA_PH_ann.csv"
        annotations_last_highest_id: 620
        photos_path: "ton_tof_60m_20241008.photos/"
        cropped_path: "cropped/"
        thumbnails_path: "thumbnails/"
        classes: "classes.csv"
        scale_factor: 1.0
        world_transform:
        - [-3.75205861e-04, -2.11005752e-01, 3.98466962e-02, 1.62435295e+01]
        - [1.75381757e-01, -2.32932058e-02, -1.21696317e-01, 4.42951957e+00]
        - [1.23904908e-01, 3.23315100e-02, 1.72376260e-01, 0.00000000e+00]
        - [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        """
        with open(filepath, "r") as f:
            yaml_config = yaml.safe_load(f)
        self.yaml_path = filepath
        self.path = yaml_config.get("path")
        self.id = yaml_config.get("id")
        # Support both new "ply" key and legacy "pcd" key for backwards compatibility
        ply_file = yaml_config.get("ply") or yaml_config.get("pcd")
        self.ply_filepath = self.__add_path_if_needed(ply_file)
        self.cams_meta_json_filepath = self.__add_path_if_needed(
            yaml_config.get("cams_meta_json")
        )
        self.cams_xml_filepath = self.__add_path_if_needed(yaml_config.get("cams_xml"))
        self.markers_filepath = self.__add_path_if_needed(yaml_config.get("markers"))
        self.annotations_filepath = self.__add_path_if_needed(
            yaml_config.get("annotations")
        )
        self.annotations_last_highest_id = yaml_config.get(
            "annotations_last_highest_id"
        )
        self.classes_filepath = self.__add_path_if_needed(yaml_config.get("classes"))
        self.scale_factor = yaml_config.get("scale_factor", None)
        wt = yaml_config.get("world_transform")
        if wt is not None:
            self.world_transform = np.array(wt, dtype=float)
        self.photos_path = self.__add_path_if_needed(yaml_config.get("photos_path"))
        self.cropped_path = self.__add_path_if_needed(yaml_config.get("cropped_path"))
        self.thumbnail_path = self.__add_path_if_needed(
            yaml_config.get("thumbnails_path")
        )
        # Optional orientation-related fields
        up = yaml_config.get("up_vector")
        if up is not None:
            from substrata.geom import Vector

            self.up_vector = Vector(up)
        d_off = yaml_config.get("depth_offset")
        if d_off is not None:
            self.depth_offset = float(d_off)
        dpu = yaml_config.get("depth_per_unit")
        if dpu is not None:
            self.depth_per_unit = float(dpu)

    @property
    def pcd_filepath(self):
        """Backwards compatibility property for ply_filepath."""
        return self.ply_filepath

    @property
    def world_transform_is_identity(self) -> bool:
        """Check if the world_transform is the identity matrix."""
        return np.allclose(self.world_transform, np.eye(4))

    @pcd_filepath.setter
    def pcd_filepath(self, value):
        """Backwards compatibility setter for ply_filepath."""
        self.ply_filepath = value

    def save_config_to_yaml(self, filepath=None):
        """
        Save the current configuration to a YAML file

        Args:
            filepath (str): Path where the YAML file should be saved
        """
        config = {}
        if self.path is not None:
            config["path"] = self.path
        if self.id is not None:
            config["id"] = self.id
        if self.ply_filepath is not None:
            config["ply"] = os.path.basename(self.ply_filepath)
        if self.cams_xml_filepath is not None:
            config["cams_xml"] = os.path.basename(self.cams_xml_filepath)
        if self.cams_meta_json_filepath is not None:
            config["cams_meta_json"] = os.path.basename(self.cams_meta_json_filepath)
        if self.markers_filepath is not None:
            config["markers"] = os.path.basename(self.markers_filepath)
        if self.annotations_filepath is not None:
            config["annotations"] = os.path.basename(self.annotations_filepath)
        if self.annotations_last_highest_id is not None:
            config["annotations_last_highest_id"] = int(
                self.annotations_last_highest_id
            )
        if self.photos_path is not None:
            config["photos_path"] = os.path.basename(self.photos_path)
        if self.cropped_path is not None:
            config["cropped_path"] = os.path.basename(self.cropped_path)
        if self.thumbnail_path is not None:
            config["thumbnails_path"] = os.path.basename(self.thumbnail_path)
        if self.classes_filepath is not None:
            config["classes"] = os.path.basename(self.classes_filepath)
        if self.scale_factor is not None:
            config["scale_factor"] = float(self.scale_factor)
        if self.world_transform is not None:
            config["world_transform"] = self.world_transform.tolist()
        # Optional orientation-related fields
        if self.up_vector is not None:
            up_val = getattr(self.up_vector, "xyz", self.up_vector)
            if hasattr(up_val, "tolist"):
                up_list = up_val.tolist()
            else:
                up_list = list(up_val)
            config["up_vector"] = [float(v) for v in up_list]
        if self.depth_offset is not None:
            config["depth_offset"] = float(self.depth_offset)
        if self.depth_per_unit is not None:
            config["depth_per_unit"] = float(self.depth_per_unit)

        if filepath is None:
            filepath = self.yaml_path
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def init_with_path(self, filepath):
        """
        Establish configuration by searching for files with default naming conventions
        in the given folder. The default naming conventions are as below, variables are
        only declared if files/paths are found in the folder.
        - ply: <id>_dec50M.ply or <id>.ply
        - cams_xml: <id>.cams.xml
        - cams_meta_json: <id>.meta.json
        - markers: <id>_markers.csv
        - annotations: <id>_annotations.csv
        - photos_path: <id>.photos/
        - cropped_path: <id>.cropped/
        - thumbnails_path: <id>.thumbnails/
        - classes: classes.csv

        Note: scale_factor defaults to 1.0 and world_transform is not set when using path-based initialization.
        """
        self.path = filepath
        self.id = os.path.basename(os.path.normpath(filepath))

        # Follow specific naming conventions from docstring
        # PLY: <id>_dec50M.ply or <id>.ply
        ply_dec_path = os.path.join(filepath, f"{self.id}_dec50M.ply")
        self.ply_dec_path = ply_dec_path if os.path.exists(ply_dec_path) else None

        ply_full_path = os.path.join(filepath, f"{self.id}.ply")
        self.ply_full_path = ply_full_path if os.path.exists(ply_full_path) else None

        if self.ply_dec_path:
            self.ply_filepath = self.ply_dec_path
        elif self.ply_full_path:
            self.ply_filepath = self.ply_full_path
        else:
            self.ply_filepath = None

        # Cameras XML: <id>.cams.xml
        cams_xml_path = os.path.join(filepath, f"{self.id}.cams.xml")
        self.cams_xml_filepath = (
            cams_xml_path if os.path.exists(cams_xml_path) else None
        )

        # Cameras metadata: <id>.meta.json
        cams_meta_path = os.path.join(filepath, f"{self.id}.meta.json")
        self.cams_meta_json_filepath = (
            cams_meta_path if os.path.exists(cams_meta_path) else None
        )

        # Markers: <id>_markers.csv
        markers_path = os.path.join(filepath, f"{self.id}_markers.csv")
        self.markers_filepath = markers_path if os.path.exists(markers_path) else None

        # Annotations: <id>_annotations.csv
        annotations_path = os.path.join(filepath, f"{self.id}_annotations.csv")
        self.annotations_filepath = (
            annotations_path if os.path.exists(annotations_path) else None
        )

        # Photos path: <id>.photos/
        photos_dir = os.path.join(filepath, f"{self.id}.photos")
        self.photos_path = (
            photos_dir
            if os.path.exists(photos_dir) and os.path.isdir(photos_dir)
            else None
        )

        # Cropped path: <id>.cropped/
        cropped_dir = os.path.join(filepath, f"{self.id}.cropped")
        self.cropped_path = (
            cropped_dir
            if os.path.exists(cropped_dir) and os.path.isdir(cropped_dir)
            else None
        )

        # Thumbnails path: <id>.thumbnails/
        thumbnails_dir = os.path.join(filepath, f"{self.id}.thumbnails")
        self.thumbnail_path = (
            thumbnails_dir
            if os.path.exists(thumbnails_dir) and os.path.isdir(thumbnails_dir)
            else None
        )

        # Classes: classes.csv
        classes_path = os.path.join(filepath, "classes.csv")
        self.classes_filepath = classes_path if os.path.exists(classes_path) else None

    def initialize(self, apply_transform=True):
        """
        Instantiate the PointCloud, Cameras, and Annotations objects

        Args:
            no_transform (bool): If True, do not apply world_transform to loaded objects.
        """
        # Create world_transform from scale_factor if it is not set
        # TODO: some inconsistency here, as it then ignores the world_transform from the YAML file
        if (
            apply_transform
            and self.world_transform_is_identity
            and self.scale_factor is not None
        ):
            self.world_transform = geom.Transform.from_scale(self.scale_factor)

        if self.ply_filepath:
            print(f"Loading pointcloud from {self.ply_filepath}")
            self.pcd = pointclouds.PointCloud(self.ply_filepath)
            if apply_transform and not self.world_transform_is_identity:
                self.pcd.apply_transform(self.world_transform)

        if self.cams_meta_json_filepath and self.cams_xml_filepath:
            print(
                f"Loading cameras from {self.cams_meta_json_filepath} and {self.cams_xml_filepath}"
            )
            self.cams = cameras.Cameras(
                self.cams_meta_json_filepath, self.cams_xml_filepath
            )
            if apply_transform and not self.world_transform_is_identity:
                self.cams.transform_coords(self.world_transform)

        if self.markers_filepath:
            print(f"Loading markers from {self.markers_filepath}")
            self.markers = annotations.Annotations(
                self.markers_filepath, orig_coords_only=True
            )
            if apply_transform and not self.world_transform_is_identity:
                self.markers.transform_coords(self.world_transform)

        if self.annotations_filepath:
            print(f"Loading annotations from {self.annotations_filepath}")
            self.annotations = annotations.Annotations(
                self.annotations_filepath, orig_coords_only=True
            )
            if apply_transform and not self.world_transform_is_identity:
                self.annotations.transform_coords(self.world_transform)

    def scale(self):
        """
        Compute project scale factor if missing. Does not apply any transforms.
        """
        if self.scale_factor is None:
            self.scalebars = annotations.Scalebars(settings.RGL_SCALEBARS, self.markers)
            self.scale_factor = self.scalebars.calc_scalefactor()

    def scale_and_orient(self):
        """
        Apply orientation (and scaling) transforms to the point cloud and related assets
        using current values in scale_factor, up_vector, depth_offset, depth_per_unit.
        Computes missing orientation values if absent.
        """
        if self.pcd is None:
            raise ValueError("Pointcloud is not initialized")

        # Ensure required parameters are present
        if (
            (self.up_vector is None)
            or (self.depth_offset is None)
            or (self.depth_per_unit is None)
        ):
            self.up_vector, self.depth_offset, self.depth_per_unit, *_ = (
                self.cams.get_up_vector_from_camera_depths()
            )
        if self.scale_factor is None:
            # If scale not computed, do it now
            self.scale()

        # Apply to pointcloud
        if not self.pcd.world_transform_is_identity:
            print(
                f"Warning: Pointcloud already has a world_transform , this will add to it: {self.pcd.world_transform}"
            )
        self.pcd.apply_orientation_transforms(
            self.scale_factor, self.up_vector, self.depth_offset, self.depth_per_unit
        )
        self.world_transform = self.pcd.world_transform

        # Propagate to cameras/markers/annotations
        if hasattr(self, "cams") and self.cams is not None:
            if not self.cams.world_transform_is_identity:
                print(
                    f"Warning: Camera already has a world_transform, this will add to it: {self.cams.world_transform}"
                )
            self.cams.transform_coords(self.world_transform)
            self.cams.scale_factor = self.scale_factor
            self.cams.depth_offset = self.depth_offset
            self.cams.depth_per_unit = self.depth_per_unit

        if hasattr(self, "markers") and self.markers is not None:
            if not self.markers.world_transform_is_identity:
                print(
                    f"Warning: Markers already have a world_transform, this will add to it: {self.markers.world_transform}"
                )
            self.markers.transform_coords(self.world_transform)
        if hasattr(self, "annotations") and self.annotations is not None:
            if not self.annotations.world_transform_is_identity:
                print(
                    f"Warning: Annotations already have a world_transform, this will add to it: {self.annotations.world_transform}"
                )
            self.annotations.transform_coords(self.world_transform)

    def __add_path_if_needed(self, filename):
        # If filename is an absolute path or contains directories, use it as is.
        if filename is None:
            return None
        elif os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.path.rstrip("/"), filename)

    @staticmethod
    def __is_yaml_file_or_folder_containing_yaml_file(filepath):
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return True
        elif os.path.isdir(filepath):
            yaml_files = [
                f
                for f in os.listdir(filepath)
                if f.lower().endswith(".yaml") or f.lower().endswith(".yml")
            ]
            return bool(yaml_files)
        else:
            return False

    @staticmethod
    def __get_yaml_filepath(filepath):
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return filepath
        elif os.path.isdir(filepath):
            yaml_files = [
                f
                for f in os.listdir(filepath)
                if f.lower().endswith(".yaml") or f.lower().endswith(".yml")
            ]
            return yaml_files[0] if yaml_files else None
        else:
            return None
