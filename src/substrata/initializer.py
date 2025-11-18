# Standard Library
import os

# Third-Party Libraries
import yaml
import numpy as np

# Local Modules
from substrata import geom, settings, annotations
import substrata.annotations as annotations


class ProjectInitializer:
    """
    Factory class that spawns a PointCloud, Cameras, and Annotations instance.

    """

    def __init__(self, yaml=None, path=None):
        # Check that either yaml (path or filepath) or project_path is provided
        if yaml is None and path is None:
            raise ValueError("Either yaml or path must be provided.")
        elif yaml is not None and path is not None:
            raise ValueError("Either yaml or path must be provided, not both.")

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
        self.classifier_filepath = None
        self.world_transform = np.eye(4)
        self.scale_factor = None
        self.up_vector = None
        self.depth_offset = None
        self.depth_per_unit = None

        self.pcd = None
        self.cams = None
        self.markers = None
        self.annotations = None
        self.scalebars = None

        # If YAML file or path to YAML file is provided, use it to initialize the project
        if yaml is not None:
            # YAML file is provided directly
            print(f"Loading project from YAML file: {yaml}")
            self.init_with_yaml(yaml)
        # If path is provided - assess whether a YAML file exists in the same directory
        # otherwise use default naming conventions
        elif path is not None:
            current_folder_name = os.path.basename(os.path.abspath(path))
            yaml_path = os.path.join(path, f"{current_folder_name}.yaml")
            if os.path.isfile(yaml_path):
                print(f"Loading project from YAML file: {yaml_path}")
                self.init_with_yaml(yaml_path)
            else:
                print(
                    f"Loading project using default naming conventions from path: {path}"
                )
                self.init_with_path(path)

    def __str__(self) -> str:
        """
        Returns a summary of all attributes set on the ProjectInitializer instance.

        Returns:
            str: A formatted string summary of non-None attributes.
        """
        attrs = []
        for attr, value in self.__dict__.items():
            if value is not None:
                val_str = f"{value}"
                attrs.append(f"  {attr}={val_str},")
        if attrs:
            attrs[-1] = attrs[-1].rstrip(",")
        lines = ["ProjectInitializer("] + attrs + [")"]
        return "\n".join(lines)

    def init_with_yaml(self, filepath):
        """
        Establish configuration by reading the YAML file

        Example YAML file:
        path: "/Users/pbongaerts/examples/ton_tof/ton_tof_60m/ton_tof_60m_20241008/"
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

        # Filepaths
        self.ply_filepath = self.__add_path_if_needed(ply_file)
        self.cams_meta_json_filepath = self.__add_path_if_needed(
            yaml_config.get("cams_meta_json")
        )
        self.cams_xml_filepath = self.__add_path_if_needed(yaml_config.get("cams_xml"))
        self.markers_filepath = self.__add_path_if_needed(yaml_config.get("markers"))
        self.annotations_filepath = self.__add_path_if_needed(
            yaml_config.get("annotations")
        )
        self.classes_filepath = self.__add_path_if_needed(yaml_config.get("classes"))
        self.classifier_filepath = self.__add_path_if_needed(
            yaml_config.get("classifier")
        )
        self.photos_path = self.__add_path_if_needed(yaml_config.get("photos_path"))
        self.cropped_path = self.__add_path_if_needed(yaml_config.get("cropped_path"))
        self.thumbnail_path = self.__add_path_if_needed(
            yaml_config.get("thumbnails_path")
        )

        # Annotations last highest ID
        self.annotations_last_highest_id = yaml_config.get(
            "annotations_last_highest_id"
        )

        # Optional orientation-related fields

        def _parse_optional_float(value) -> float | None:
            """Parse an optional value to float or return None."""
            return float(value) if value is not None else None

        # Scale factor, orientation-related fields and world transform
        self.scale_factor = _parse_optional_float(yaml_config.get("scale_factor"))

        up_vector = yaml_config.get("up_vector")
        if up_vector is not None:
            self.up_vector = geom.Vector(up_vector)

        self.depth_offset = _parse_optional_float(yaml_config.get("depth_offset"))
        self.depth_per_unit = _parse_optional_float(yaml_config.get("depth_per_unit"))

        world_transform = yaml_config.get("world_transform")
        if world_transform is not None:
            self.world_transform = np.array(world_transform, dtype=float)

    @property
    def pcd_filepath(self) -> str:
        """Backwards compatibility property for ply_filepath."""
        return self.ply_filepath

    @pcd_filepath.setter
    def pcd_filepath(self, value: str) -> None:
        """Backwards compatibility setter for ply_filepath."""
        self.ply_filepath = value

    @property
    def world_transform_is_identity(self) -> bool:
        """Check if the world_transform is the identity matrix."""
        return np.allclose(self.world_transform, np.eye(4))

    def save_config_to_yaml(self, filepath: str | None = None) -> None:
        """
        Save the current configuration to a YAML file.

        Args:
            filepath (str, optional): Path where the YAML file should be saved.
        """
        config = {}
        if self.path is not None:
            config["path"] = self.path
        if self.id is not None:
            config["id"] = self.id
        if self.ply_filepath is not None:
            config["ply"] = self.ply_filepath
        if self.cams_xml_filepath is not None:
            config["cams_xml"] = self.cams_xml_filepath
        if self.cams_meta_json_filepath is not None:
            config["cams_meta_json"] = self.cams_meta_json_filepath
        if self.markers_filepath is not None:
            config["markers"] = self.markers_filepath
        if self.annotations_filepath is not None:
            config["annotations"] = self.annotations_filepath
        if self.annotations_last_highest_id is not None:
            config["annotations_last_highest_id"] = int(
                self.annotations_last_highest_id
            )
        if self.classifier_filepath is not None:
            config["classifier"] = self.classifier_filepath
        if self.photos_path is not None:
            config["photos_path"] = self.photos_path
        if self.cropped_path is not None:
            config["cropped_path"] = self.cropped_path
        if self.thumbnail_path is not None:
            config["thumbnails_path"] = self.thumbnail_path
        if self.classes_filepath is not None:
            config["classes"] = self.classes_filepath
        if self.scale_factor is not None:
            config["scale_factor"] = float(self.scale_factor)
        if self.world_transform is not None:
            # Save as bracketed nested arrays ([[..., ...], [..., ...], ...]) in YAML
            config["world_transform"] = _as_flow_sequence(self.world_transform)
        # Optional orientation-related fields
        if self.up_vector is not None:
            up_val = getattr(self.up_vector, "xyz", self.up_vector)
            if hasattr(up_val, "tolist"):
                up_list = up_val.tolist()
            else:
                up_list = list(up_val)
            # Save vector in bracketed flow style: [x, y, z]
            config["up_vector"] = _as_flow_sequence([float(v) for v in up_list])
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
        Instantiate the PointCloud, Cameras, Markers and Annotations objects

        Note that scale and orientation-related fields are read only, and only
        the world_transform is applied to loaded objects (if apply_transform is True).

        Args:
            apply_transform (bool): If True, apply world_transform to loaded objects.
        """
        from substrata import (
            cameras,
            pointclouds,
        )

        if self.ply_filepath:
            print(f"Loading pointcloud from {self.ply_filepath}")
            self.pcd = pointclouds.PointCloud(self.ply_filepath)

        if self.cams_meta_json_filepath and self.cams_xml_filepath:
            print(
                f"Loading cameras from {self.cams_meta_json_filepath} and {self.cams_xml_filepath}"
            )
            self.cams = cameras.Cameras(
                self.cams_meta_json_filepath, self.cams_xml_filepath
            )
            self.cams.offset = self.depth_offset
            self.cams.per_unit = self.depth_per_unit

        if self.markers_filepath:
            print(f"Loading markers from {self.markers_filepath}")
            self.markers = annotations.Annotations(
                self.markers_filepath, orig_coords_only=True
            )

        if self.annotations_filepath:
            print(f"Loading annotations from {self.annotations_filepath}")
            self.annotations = annotations.Annotations(
                self.annotations_filepath, orig_coords_only=True
            )

        if apply_transform:
            print(
                f"Applying world_transform to loaded objects:\n{self.world_transform}"
            )
            self.apply_world_transform()

    def apply_world_transform(self, skip_pcd=False):
        """
        Apply world_transform to loaded objects.

        """
        transform_targets = [
            ("pcd", "Pointcloud", "apply_transform") if not skip_pcd else None,
            ("cams", "Cameras", "transform_coords"),
            ("markers", "Markers", "transform_coords"),
            ("annotations", "Annotations", "transform_coords"),
        ]

        # Apply world_transform to all loaded objects and warn if they already have a
        # world_transform
        for attr_name, label, method in [t for t in transform_targets if t is not None]:
            obj = getattr(self, attr_name, None)
            if obj is not None:
                if not getattr(obj, "world_transform_is_identity", True):
                    print(
                        f"Warning: {label} already has a world_transform, "
                        f"this will add to it: {obj.world_transform}"
                    )
                getattr(obj, method)(self.world_transform)

    def calc_scale_factor(self):
        """
        Compute project scale factor. Does not apply any transforms.
        """
        self.scalebars = annotations.Scalebars(settings.RGL_SCALEBARS, self.markers)
        self.scale_factor = self.scalebars.calc_scalefactor()
        return self.scale_factor

    def scale_and_orient(self, plot=True, recalculate=True, markers_filepath=None):
        """Calculate scale and orientation transforms and apply them.

        Calculates scale and orientation transforms and applies them using the
        Pointcloud.apply_orientation_transforms() method.

        Note that depth_offset is based on depth_per_unit (vertical scaling factor),
        and so z-values are only very rough approximations of depth.

        Args:
            plot (bool): If True, create visualizations of the regression fit.
            recalculate (bool): If True, recalculate the up vector and scale factor.
            markers_filepath (str, optional): Path to markers CSV file. If provided,
                uses annotation depths instead of camera depths to determine up vector.
        """
        # Ensure at least the poincloud is initialized
        if self.pcd is None:
            raise ValueError("Pointcloud is not initialized")

        if recalculate:
            # Determine up vector, depth offset and depth per unit
            if markers_filepath is not None:
                # Load markers if not already loaded or if different filepath
                if self.markers is None or self.markers_filepath != markers_filepath:
                    print(f"Loading markers from {markers_filepath}")
                    self.markers = annotations.Annotations(
                        markers_filepath, orig_coords_only=True
                    )
                # Use annotation depths to determine up vector
                self.up_vector, self.depth_offset, self.depth_per_unit, *_ = (
                    self.markers.get_up_vector_from_annotation_depths(plot=plot)
                )
            else:
                # Use camera depths to determine up vector (default behavior)
                self.up_vector, self.depth_offset, self.depth_per_unit, *_ = (
                    self.cams.get_up_vector_from_camera_depths(plot=plot)
                )
            # Determine scale factor
            self.calc_scale_factor()

        # Apply scale and orientation transforms to pointcloud
        if not self.pcd.world_transform_is_identity:
            print(
                f"Warning: Pointcloud already has a world_transform , this will add to it: {self.pcd.world_transform}"
            )
        self.pcd.apply_orientation_transforms(
            self.scale_factor, self.up_vector, self.depth_offset, self.depth_per_unit
        )

        # Set camera attributes
        self.cams.up_vector = self.up_vector
        self.cams.depth_offset = self.depth_offset
        self.cams.depth_per_unit = self.depth_per_unit

        # Propagate pointcloud world_transform to cameras/markers/annotations
        self.world_transform = self.pcd.world_transform
        self.apply_world_transform(skip_pcd=True)

        return self.up_vector, self.depth_offset, self.depth_per_unit

    def __add_path_if_needed(self, filename):
        # If filename is an absolute path or contains directories, use it as is.
        if filename is None:
            return None
        elif os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.path.rstrip("/"), filename)


class _YamlFlowList(list):
    """Wrapper list to force YAML flow-style (bracketed) sequences."""


def _represent_flow_list(dumper, data):
    """YAML representer to serialize lists in flow-style."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _ensure_yaml_flow_registered() -> None:
    """Register the YAML representer for flow-style lists once."""
    if not getattr(yaml, "_substrata_flow_registered", False):
        yaml.add_representer(_YamlFlowList, _represent_flow_list)  # type: ignore[arg-type]
        yaml._substrata_flow_registered = True  # type: ignore[attr-defined]


def _as_flow_sequence(value):
    """Convert array-like value to a flow-style YAML sequence (bracketed).

    Preserves nested list structure and ensures both outer and inner lists
    are written using bracketed syntax.
    """
    _ensure_yaml_flow_registered()
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        items = [
            _as_flow_sequence(v) if isinstance(v, (list, tuple, np.ndarray)) else v
            for v in list(value)
        ]
        return _YamlFlowList(items)
    return value
