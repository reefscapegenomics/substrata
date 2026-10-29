# Standard Library
import os

# Third-Party Libraries
import yaml
import numpy as np

# Local Modules
from substrata import geom, settings, annotations, sidecars
import substrata.annotations as annotations


class ProjectInitializer:
    """
    Factory class that spawns a PointCloud, Cameras, and Annotations instance.

    """

    def __init__(self, yaml=None, path=None, local=False, max_points=None):
        # Check that either yaml (path or filepath) or project_path is provided
        if yaml is None and path is None:
            raise ValueError("Either yaml or path must be provided.")
        elif yaml is not None and path is not None:
            raise ValueError("Either yaml or path must be provided, not both.")

        # Runtime-only option (not stored in YAML): cap on points loaded from PLY
        self.max_points = max_points

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
        self.crop_size = None
        self.thumbnail_path = None
        self.classes_filepath = None
        self.classifier_filepath = None
        # Sidecar YAML of the annotations CSV (see apply_annotation_sidecar)
        self.annotation_sidecar = {}
        self.world_transform = np.eye(4)
        self.scale_factor = None
        self.up_vector = None
        self.depth_offset = None
        self.depth_per_unit = None
        self.color_correction = None
        # FireFish depth sensor offset used for the orientation (record only)
        self.firefish_depth_sensor_offset = None
        # Water level (tide) applied to FireFish depths (record only)
        self.firefish_water_level = None

        self.pcd = None
        self.cams = None
        self.markers = None
        self.annotations = None
        self.scalebars = None
        self.depth_markers = None

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

        # Reset paths to local if requested
        if local:
            print(f"Resetting paths to local folder!")
            self.reset_paths_to_local()

        # Per-annotation-set settings from the annotations CSV's sidecar
        self.apply_annotation_sidecar()

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
        firefish_depth_sensor_offset: [0.0, 0.0, -0.152]  # optional, record only
        firefish_water_level: -0.32  # optional, record only

        ``annotations`` names the project's default annotation set. Settings of
        an annotation set (``classes``, ``cropped_path``, ``crop_size``,
        ``classifier``, ``annotations_last_highest_id``) preferably live in that
        CSV's sidecar (``<annotations csv>.yaml``, see :mod:`substrata.sidecars`),
        which overrides these legacy project keys.
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
        self.__set_derived_ply_paths()
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
        crop_size = yaml_config.get("crop_size")
        self.crop_size = int(crop_size) if crop_size is not None else None
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

        self.firefish_water_level = _parse_optional_float(
            yaml_config.get("firefish_water_level")
        )
        ff_offset = yaml_config.get("firefish_depth_sensor_offset")
        if ff_offset is not None:
            self.firefish_depth_sensor_offset = [float(v) for v in ff_offset]

        cc = yaml_config.get("color_correction")
        if cc is not None:
            self.color_correction = {
                "matrix": np.array(cc["matrix"], dtype=float),
                "offset": np.array(cc["offset"], dtype=float),
            }

    @property
    def pcd_filepath(self) -> str:
        """Backwards compatibility property for ply_filepath."""
        return self.ply_filepath

    @pcd_filepath.setter
    def pcd_filepath(self, value: str) -> None:
        """Backwards compatibility setter for ply_filepath."""
        self.ply_filepath = value

    @property
    def ply_stratified(self) -> bool:
        """Whether the full-resolution PLY is stratified.

        True when :attr:`ply_full_path` has a matching ``<id>.strat.npz`` next to
        it (see ``substrata.stratified``; build it with ``substrata stratify``).
        """
        from substrata.stratified import is_stratified

        return is_stratified(self.ply_full_path)

    @property
    def thumbnails_root(self) -> str | None:
        """Thumbnails directory: ``thumbnails_path`` or ``<path>/<id>.thumbnails``.

        Thumbnails belong to the model (unlike crops, which belong to an
        annotations file), so the default sits in the model folder.
        """
        if self.thumbnail_path:
            return self.thumbnail_path
        if self.path is None or self.id is None:
            return None
        return os.path.join(self.path, f"{self.id}{settings.THUMBNAILS_DIR_SUFFIX}")

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
        # Annotation-set keys set by the sidecar stay in the sidecar
        sidecar_keys = set(self.annotation_sidecar or {})
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
        if self.classifier_filepath is not None and "classifier" not in sidecar_keys:
            config["classifier"] = self.classifier_filepath
        if self.photos_path is not None:
            config["photos_path"] = self.photos_path
        if self.cropped_path is not None and "cropped_path" not in sidecar_keys:
            config["cropped_path"] = self.cropped_path
        if self.crop_size is not None and "crop_size" not in sidecar_keys:
            config["crop_size"] = int(self.crop_size)
        if self.thumbnail_path is not None:
            config["thumbnails_path"] = self.thumbnail_path
        if self.classes_filepath is not None and "classes" not in sidecar_keys:
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
        if self.firefish_water_level is not None:
            config["firefish_water_level"] = float(self.firefish_water_level)
        if self.firefish_depth_sensor_offset is not None:
            config["firefish_depth_sensor_offset"] = _as_flow_sequence(
                [float(v) for v in self.firefish_depth_sensor_offset]
            )
        if self.color_correction is not None:
            config["color_correction"] = {
                "matrix": _as_flow_sequence(self.color_correction["matrix"]),
                "offset": _as_flow_sequence(self.color_correction["offset"]),
            }

        if filepath is None:
            filepath = self.yaml_path
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def apply_annotation_sidecar(self) -> None:
        """Take annotation-set settings from the annotations CSV's sidecar YAML.

        The sidecar (``<annotations csv>.yaml``, see :mod:`substrata.sidecars`)
        sets ``classes``, ``cropped_path``, ``crop_size`` and ``classifier`` per
        annotation set, overriding the (legacy) project YAML keys and the folder
        conventions. ``last_highest_id`` only ever raises
        ``annotations_last_highest_id``, so deleted ids are never reused. Called
        on construction; call again after changing ``annotations_filepath``.
        """
        if not self.annotations_filepath:
            return
        sidecar = sidecars.read_annotation_sidecar(self.annotations_filepath)
        self.annotation_sidecar = sidecar
        if not sidecar:
            return
        print(
            "Loading annotation settings from "
            f"{sidecars.annotation_sidecar_path(self.annotations_filepath)}"
        )
        if sidecar.get("classes"):
            self.classes_filepath = sidecar["classes"]
        if sidecar.get("cropped_path"):
            self.cropped_path = sidecar["cropped_path"]
        if sidecar.get("crop_size") is not None:
            self.crop_size = int(sidecar["crop_size"])
        if sidecar.get("classifier"):
            self.classifier_filepath = sidecar["classifier"]
        last_id = sidecar.get("last_highest_id")
        if last_id is not None:
            self.annotations_last_highest_id = max(
                int(last_id), int(self.annotations_last_highest_id or 0)
            )
        world_transform = sidecar.get("world_transform")
        if world_transform is not None and not np.allclose(
            np.array(world_transform, dtype=float), self.world_transform
        ):
            print(
                "Warning: the annotations sidecar world_transform differs from "
                "the project's; the annotations may come from another orientation."
            )

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

    def initialize(self, apply_transform=True, apply_color_correction=True):
        """
        Instantiate the PointCloud, Cameras, Markers and Annotations objects

        Note that scale and orientation-related fields are read only, and only
        the world_transform is applied to loaded objects (if apply_transform is True).

        Args:
            apply_transform (bool): If True, apply world_transform to loaded objects.
            apply_color_correction (bool): If True and a colour correction is
                stored in the YAML config, apply it to the point cloud after
                loading.
        """
        from substrata import (
            cameras,
            pointclouds,
        )

        if self.ply_filepath:
            print(f"Loading pointcloud from {self.ply_filepath}")
            if self.max_points is None:
                self.pcd = pointclouds.PointCloud(self.ply_filepath)
            else:
                self.pcd = pointclouds.PointCloud(
                    self.ply_filepath, max_points=self.max_points
                )

        if self.cams_meta_json_filepath and self.cams_xml_filepath:
            print(
                f"Loading cameras from {self.cams_meta_json_filepath} and {self.cams_xml_filepath}"
            )
            self.cams = cameras.Cameras(
                self.cams_meta_json_filepath, self.cams_xml_filepath
            )

        if self.markers_filepath is not None and os.path.exists(self.markers_filepath):
            print(f"Loading markers from {self.markers_filepath}")
            self.markers = annotations.Annotations(
                self.markers_filepath, orig_coords_only=True
            )
        elif self.markers_filepath is not None and not os.path.exists(
            self.markers_filepath
        ):
            print(f"No markers file found at {self.markers_filepath}")

        if self.annotations_filepath is not None and os.path.exists(
            self.annotations_filepath
        ):
            print(f"Loading annotations from {self.annotations_filepath}")
            self.annotations = annotations.Annotations(
                self.annotations_filepath, orig_coords_only=True
            )
            # An explicit cropped_path wins over the <csv stem>.cropped default
            if self.cropped_path:
                self.annotations.crops_root = self.cropped_path
        elif self.annotations_filepath is not None and not os.path.exists(
            self.annotations_filepath
        ):
            print(f"No annotations file found at {self.annotations_filepath}")

        if apply_transform:
            print(
                f"Applying world_transform to loaded objects:\n{self.world_transform}"
            )
            self.apply_world_transform()

        self.propagate_depth_model()

        if (
            apply_color_correction
            and self.color_correction is not None
            and self.pcd is not None
        ):
            print("Applying colour correction from YAML config")
            self.pcd.apply_color_correction(self.color_correction)

    def propagate_depth_model(self):
        """Pass the depth regression on to the loaded objects.

        Sets ``up_vector``, ``depth_offset`` and ``depth_per_unit`` (on the
        point cloud: ``depth_scale_factor``) on the point cloud, cameras,
        markers and annotations, so ``depth_in_m`` works on them and on objects
        derived from them (e.g. ``pcd.get_z_intercepts``). Fields that are not
        set on the initializer are left untouched on the objects.
        """
        if self.up_vector is None and self.depth_offset is None:
            return
        up_vector = self.up_vector
        if up_vector is not None:
            up_vector = np.asarray(getattr(up_vector, "xyz", up_vector), dtype=float)
        values = {
            "up_vector": up_vector,
            "depth_offset": self.depth_offset,
            "depth_per_unit": self.depth_per_unit,
        }
        for attr_name in ("pcd", "cams", "markers", "annotations"):
            obj = getattr(self, attr_name, None)
            if obj is None:
                continue
            for key, value in values.items():
                if value is None:
                    continue
                if attr_name == "pcd" and key == "depth_per_unit":
                    key = "depth_scale_factor"
                setattr(obj, key, value)

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

    def scale_and_orient(self, plot=True, recalculate=True, depth_markers=None):
        """Calculate scale and orientation transforms and apply them.

        Calculates scale and orientation transforms and applies them using the
        Pointcloud.apply_orientation_transforms() method.

        Note that depth_offset is based on depth_per_unit (vertical scaling factor),
        and so z-values are only very rough approximations of depth.

        Args:
            plot (bool): If True, create visualizations of the regression fit.
            recalculate (bool): If True, recalculate the up vector and scale factor.
            depth_markers (Annotations, optional): Annotations instance with depth data.
                If provided, uses annotation depths instead of camera depths to determine up vector.
        """
        # Ensure at least the poincloud is initialized
        if self.pcd is None:
            raise ValueError("Pointcloud is not initialized")

        if recalculate:
            # Determine up vector, depth offset and depth per unit
            if depth_markers is not None:
                # Use annotation depths to determine up vector
                self.up_vector, self.depth_offset, self.depth_per_unit, *_ = (
                    depth_markers.get_up_vector_from_annotation_depths(plot=plot)
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

        # Propagate pointcloud world_transform to cameras/markers/annotations
        self.world_transform = self.pcd.world_transform
        self.apply_world_transform(skip_pcd=True)

        # Set depth attributes on pcd, cameras, markers and annotations
        self.propagate_depth_model()
        # Set depth_markers attributes if depth_markers were used
        if depth_markers is not None:
            depth_markers.up_vector = self.up_vector
            depth_markers.depth_offset = self.depth_offset
            depth_markers.depth_per_unit = self.depth_per_unit
            depth_markers.apply_transform(self.world_transform)

        return self.up_vector, self.depth_offset, self.depth_per_unit

    def orient_from_chunk_transform(self, north_up=False, shift_to_positive_xy=True):
        """Orient and scale the project using the Metashape chunk transform.

        Alternative to scale_and_orient(): instead of fitting an up vector to
        depths and a scale to scalebars, use the ``chunk_transform`` stored in
        the cameras meta JSON. For georeferenced chunks this maps to geocentric
        (ECEF) metres, so it is composed with an ECEF -> East-North-Up rotation
        (see geom.chunk_transform_to_enu): +X east, +Y north, +Z ellipsoidal
        height. For chunks in a local CRS the transform is applied as-is.

        Unless ``north_up`` is set, the cloud is then rotated about Z only so
        +Y points upslope (shallow end at the top of top-down views, whichever
        way the plot is elongated), with the plot's footprint axes squared to
        X/Y (see geom.Transform.align_y_upslope). Scale and Z (true depth) are
        unaffected.

        The scale factor is the uniform scale contained in the transform. The
        depth fields (up_vector, depth_offset, depth_per_unit) are set to the
        exact linear model implied by the transform (z = up_vector . orig +
        depth_offset), so camera depth residuals can be checked as usual.

        Args:
            north_up (bool): If True, keep the georeferenced orientation (+X
                east, +Y north) instead of aligning +Y upslope.
            shift_to_positive_xy (bool): If True, translate the cloud so its
                minimum X/Y are non-negative (Z is left untouched).

        Returns:
            np.ndarray: The resulting 4x4 world_transform.
        """
        import json

        if self.pcd is None:
            raise ValueError("Pointcloud is not initialized")
        if not self.cams_meta_json_filepath or not os.path.isfile(
            self.cams_meta_json_filepath
        ):
            raise ValueError(
                "No cameras meta JSON found (needed for chunk_transform): "
                f"{self.cams_meta_json_filepath}"
            )
        with open(self.cams_meta_json_filepath, "r") as f:
            chunk_transform = json.load(f).get("chunk_transform")
        if chunk_transform is None:
            raise ValueError(
                f"No chunk_transform in {self.cams_meta_json_filepath} "
                "(was the Metashape chunk georeferenced?)"
            )
        chunk_transform = np.asarray(chunk_transform, dtype=float)

        if not self.pcd.world_transform_is_identity:
            print(
                "Warning: Pointcloud already has a world_transform, this will add "
                f"to it: {self.pcd.world_transform}"
            )

        # Reference point (cloud centre) defines the local ENU origin
        pts = self.pcd.points
        ref_point = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        ref_mapped = geom.transform_coords(ref_point, chunk_transform)
        is_geocentric = 6.0e6 < np.linalg.norm(ref_mapped) < 6.6e6
        if is_geocentric:
            transform, scale, (lat, lon) = geom.chunk_transform_to_enu(
                chunk_transform, ref_point
            )
            print(
                "chunk_transform maps to geocentric (ECEF) coordinates; using "
                f"local ENU frame with origin lat={lat:.7f}, lon={lon:.7f}, h=0"
            )
        else:
            transform = chunk_transform
            scale = float(np.cbrt(np.linalg.det(chunk_transform[:3, :3])))
            print("chunk_transform is not geocentric; applying it as-is")
        print(f"  Scale factor (from chunk_transform): {scale:.6f}")

        self.pcd.apply_transform(transform)
        if not north_up:
            # Z-rotation only: +Y upslope, footprint squared to X/Y
            self.pcd.apply_transform(geom.Transform.align_y_upslope(self.pcd).mat)
        if shift_to_positive_xy:
            self.pcd.apply_transform(geom.Transform.shift_to_positive_xy(self.pcd).mat)
        self.pcd.scale_factor = scale

        # Store results and propagate to cameras/markers/annotations
        self.world_transform = self.pcd.world_transform
        self.scale_factor = scale
        self.up_vector = np.array(self.world_transform[2, :3])
        self.depth_offset = float(self.world_transform[2, 3])
        self.depth_per_unit = float(np.linalg.norm(self.up_vector))
        self.apply_world_transform(skip_pcd=True)
        self.propagate_depth_model()

        return self.world_transform

    def __set_derived_ply_paths(self):
        """Populate ``ply_dec_path`` and ``ply_full_path`` from the conventions.

        ``init_with_path`` sets these while probing the folder, but the YAML
        branch previously left them ``None``. Several subcommands (``decimate``,
        ``ply-repair``, ``head``, ``scalebars``, ``segment --full-ply``) resolve
        their input through :attr:`ply_full_path`, so in any project carrying a
        YAML they failed with "No input PLY found" despite a valid ``ply:`` key.

        Both the decimated and full-resolution conventional names are checked
        against the project directory; whichever exists is recorded. The value
        of :attr:`ply_filepath` itself is left untouched.
        """
        if not self.path or not self.id:
            return
        dec_path = os.path.join(self.path, f"{self.id}_dec50M.ply")
        full_path = os.path.join(self.path, f"{self.id}.ply")
        if self.ply_dec_path is None and os.path.exists(dec_path):
            self.ply_dec_path = dec_path
        if self.ply_full_path is None and os.path.exists(full_path):
            self.ply_full_path = full_path
        # A ply: key pointing somewhere non-conventional still counts as the
        # full-resolution cloud unless it is the decimated file.
        if (
            self.ply_full_path is None
            and self.ply_filepath
            and self.ply_filepath != self.ply_dec_path
            and os.path.exists(self.ply_filepath)
        ):
            self.ply_full_path = self.ply_filepath

    def __add_path_if_needed(self, filename):
        # If filename is an absolute path or contains directories, use it as is.
        if filename is None:
            return None
        elif os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.path.rstrip("/"), filename)

    def reset_paths_to_local(self, path: str | None = None) -> None:
        """Reset all path/filepath variables to a new base path.

        Extracts the filename or directory name from each existing path and
        updates it to be relative to the new base path. If a path is None,
        it remains None.

        Args:
            path (str, optional): New base path. If None, uses self.path.
        """
        if path is None:
            path = os.getcwd()

        # Normalize the new base path
        new_base_path = os.path.normpath(path)

        # List of path attributes to reset
        path_attributes = [
            "ply_filepath",
            "ply_dec_path",
            "ply_full_path",
            "cams_xml_filepath",
            "cams_meta_json_filepath",
            "markers_filepath",
            "annotations_filepath",
            "photos_path",
            "cropped_path",
            "thumbnail_path",
            "classes_filepath",
            "classifier_filepath",
        ]

        # Reset each path attribute
        for attr_name in path_attributes:
            current_path = getattr(self, attr_name, None)
            if current_path is not None:
                # Extract the filename or directory name
                name = os.path.basename(current_path)
                # Join with new base path
                new_path = os.path.join(new_base_path, name)
                setattr(self, attr_name, new_path)

        # Update the base path itself
        self.path = new_base_path

        # Update yaml_path if it exists
        # (keep it in the same directory as the new path)
        if self.yaml_path is not None:
            yaml_filename = os.path.basename(self.yaml_path)
            self.yaml_path = os.path.join(new_base_path, yaml_filename)


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
