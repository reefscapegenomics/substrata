# Standard Library
import csv
import logging
import os
import random
import re

# Third-Party Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from matplotlib.backends.backend_pdf import PdfPages

# Local Modules
from substrata import (
    settings,
    measurements,
    cameras,
    visualizations,
    geometry,
    pointclouds,
)
from substrata.logging import logger


class Annotations:
    """
    Container class that holds a collection of Annotation objects
    """

    def __init__(
        self,
        filepath=None,
        coords=None,
        header=True,
        orig_coords_only=False,
    ):
        self.data = {}
        self.measurements = {}
        self.world_transform = np.eye(4)

        if filepath is not None:
            self.get_annotations_from_file(
                filepath, header=header, orig_coords_only=orig_coords_only
            )
        elif coords is not None:
            self.get_annotations_from_coords(coords)

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
        return len(self.data)

    def items(self):
        return self.data.items()

    @property
    def coords(self):
        return [annotation.coords for annotation in self.data.values()]

    @property
    def classifications(self):
        return [annotation.classification for annotation in self.data.values()]

    @property
    def image_matches(self):
        return {
            annotation.id: annotation.image_match
            for annotation in self.data.values()
            if annotation.image_match is not None
        }

    def append(self, annotation):
        if annotation.id in self.data:
            raise ValueError(f"Annotation with id {annotation.id} already exists.")
        else:
            self.data[annotation.id] = annotation
            self.data[annotation.id].parent = self
            # TO DO: any other changes (eg transforms) to be implemented on append?

    def get_annotations_from_file(
        self, annotations_filepath, header=True, orig_coords_only=False
    ):
        """Read in annotations from a file and store in dict.

        Args:
            annotations_filepath (str): Path to the file.
        """
        annotations_file = open(annotations_filepath, "r")

        for line_no, line in enumerate(annotations_file):
            if line_no == 0 and header:
                self.col_order = self.__determine_col_order(line)
                continue  # skip to next line
            elif line_no == 0 and not header:
                self.col_order = settings.ANN_DEFAULT_COL_ORDER

            (
                id,
                orig_x,
                orig_y,
                orig_z,
                label,
                label_conf,
                world_x,
                world_y,
                world_z,
                other_cols,
            ) = self.__get_annotation_fields(line.rstrip().split(","))
            ann_id = self.__strip_post_fixes(id)
            if ann_id not in self.data:
                # New annotation
                self.data[id] = Annotation([orig_x, orig_y, orig_z], id=id, parent=self)
                self.data[id].line_no = line_no
                self.data[id].label = self.data[id].classification = label
                self.data[id].label_conf = label_conf
                if world_x is not None and orig_coords_only is False:
                    self.data[id].coords = np.asarray(
                        [world_x, world_y, world_z], dtype=float
                    )
                self.data[id].other_cols = other_cols
            else:
                # Additional coordinates for existing annotation
                self.data[ann_id].add_extra_coords(line.rstrip())
        annotations_file.close()

    def get_annotations_from_coords(self, annotations_coords):
        """Use a dict or list of coordinates to fill the annotations class.

        Args:
            annotations_coords (dict, list, or np.ndarray): Coordinate data.
        """
        if isinstance(annotations_coords, dict):
            for i, (key, coords) in enumerate(annotations_coords.items()):
                if coords is not None:
                    self.data[key] = Annotation(coords, id=key, parent=self)
        elif isinstance(annotations_coords, (list, np.ndarray)):
            for i, coords in enumerate(annotations_coords):
                if coords is not None:
                    self.data[i] = Annotation(coords, id=i, parent=self)

    def get_annotations_from_google_worksheet(self, worksheet, header=True):
        """Use a Google worksheet to fill the annotations class.

        Args:
            worksheet: Google worksheet object.
        """
        worksheet_data = worksheet.get_all_values()
        for row_number, row_cols in enumerate(worksheet_data, start=1):
            if row_number == 0 and header:
                self.col_order = self.__determine_col_order(row)
                continue  # skip to next line
            elif row_number == 0 and not header:
                self.col_order = settings.ANN_DEFAULT_COL_ORDER

            (
                id,
                orig_x,
                orig_y,
                orig_z,
                label,
                label_conf,
                world_x,
                world_y,
                world_z,
                other_cols,
            ) = self.__get_annotation_fields(row_cols)
            if ann_id not in self.data:
                # New annotation
                self.data[id] = Annotation([orig_x, orig_y, orig_z], id=id, parent=self)
                self.data[id].line_no = row_number
                self.data[id].label = self.data[id].classification = label
                self.data[id].label_conf = label_conf
                if world_x is not None and orig_coords_only is False:
                    self.data[id].coords = np.asarray(
                        [world_x, world_y, world_z], dtype=float
                    )
                self.data[id].other_cols = other_cols
            else:
                # Additional coordinates for existing annotation
                self.data[id].add_extra_coords(
                    row_cols
                )  # Updated to use row_cols instead of line.rstrip()

    def add_meta_data(self, data_filepath):
        """Add metadata to annotations from a CSV file (requires header).

        Args:
            data_filepath (str): Path to the CSV file.
        """
        data_file = open(data_filepath, "r")
        for line_no, line in enumerate(data_file):
            if line_no == 0:
                self.col_order = self.__determine_col_order(line)
                continue  # skip to next line

            cols = line.rstrip().split(",")
            ann_id = cols[self.col_order["id"]]
            if ann_id in self.data.keys():
                self.data[ann_id].meta_data = {}
                for i in range(0, len(col_headers)):
                    self.data[ann_id].meta_data[col_headers[i]] = cols[i + 1]
            else:
                print(f"No annotation with ID {ann_id} found.")
        data_file.close()

    def get_new_id(self, last_highest_id=None, default_prefix=None):
        """
        Return a identified based on the next available integer
        """
        # Find the highest integer used in existing IDs
        highest_int = 0
        prefix = None
        num_digits = 0

        for ann_id in self.data.keys():
            # Find the last sequence of digits in the ID
            match = re.search(r"(\d+)$", ann_id)
            if match:
                num_str = match.group(1)
                curr_num = int(num_str)

                # Get prefix by removing the number from the end
                curr_prefix = ann_id[: -len(num_str)]

                # Update highest integer if needed
                if curr_num > highest_int:
                    highest_int = curr_num
                    prefix = curr_prefix
                    num_digits = len(
                        num_str
                    )  # Store the number of digits from the highest ID

        # If no IDs found, use default prefix
        if prefix is None:
            if default_prefix is not None:
                prefix = default_prefix
            else:
                prefix = "ann_"
            highest_int = 0
            num_digits = 4  # Default to 4 digits if no existing IDs

        # Check against last_highest_id if provided
        if last_highest_id is not None:
            highest_int = max(highest_int, last_highest_id)

        # Generate new ID with same prefix and next integer
        new_num = highest_int + 1
        new_id = f"{prefix}{new_num:0{num_digits}d}"

        return new_id

    def get_bounding_box(self):
        """Return the min and max values of x, y, z for all points in annotations.

        Returns:
            list: [min_coords, max_coords] for x, y, z.
        """
        xyz_min = np.min(self.coords, axis=0)
        xyz_max = np.max(self.coords, axis=0)
        return [xyz_min, xyz_max]

    def get_eucl_distance_matrix(self):
        """Calculates pairwise Euclidean distances and returns a DataFrame.

        Returns:
            pd.DataFrame: Distance matrix with annotation keys as rows and columns.
        """
        keys = list(self.data.keys())
        coords = np.array([annotation.coords for annotation in self.data.values()])
        distmat = np.sqrt(
            np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1)
        )
        return pd.DataFrame(distmat, index=keys, columns=keys)

    def get_first_image_matches(self, cams, pcd=None):
        """Get the first image match for each annotation."""
        image_matches = {}
        for ann in tqdm(self.data.values(), desc="Getting first image matches"):
            match = ann.get_first_image_match(cams, pcd)
            if match:
                image_matches[ann.id] = match
                # ann.image_matches.append(match) TODO: INCORRECT
        return image_matches

    # def get_pcd(self):
    #     """Get a point cloud of all annotation coordinates."""
    #     pcd = geometry.PointCloud()
    #     pcd.points = utility.Vector3dVector(self.get_all_coords())
    #     return pcd

    def transform_coords(self, transform):
        """Apply a transformation to all annotation coordinates.

        Args:
            transform (np.ndarray or Transform): A 4x4 transformation matrix or a Transform instance.
        """
        # Accept either a 4x4 matrix or a Transform instance
        if hasattr(transform, "matrix"):
            matrix = np.array(transform.matrix)
        else:
            matrix = np.array(transform)
        for ann_id in self.data:
            self.data[ann_id].transform_coords(transform)
        self.world_transform = np.dot(matrix, self.world_transform)

    # Alias for compatibility
    apply_transform = transform_coords

    def get_original_coords(self, transform_matrix):
        """Revert transformed coordinates using the given transformation.

        Args:
            transform_matrix (np.ndarray): The transformation matrix to invert.
        """
        for ann_id in self.data:
            self.data[ann_id].reverse_transform_coords(transform_matrix)
        self.world_transform = np.dot(
            np.array(transform_matrix), self.world_transform
        )  # TODO: CHECK!

    def random_subset(self, length):
        """Return a random subset of annotations.

        Args:
            length (int): Number of annotations to include.

        Returns:
            Annotations: New container with the selected annotations.
        """
        annotations_subset = Annotations()
        random_keys = random.sample(list(self.data.keys()), length)
        for ann_id in random_keys:
            annotations_subset.data[ann_id] = self.data[ann_id]
        return annotations_subset

    def subset(self, length):
        """Return a subset of annotations.

        Args:
            length (int): Number of annotations to include.

        Returns:
            Annotations: New container with the selected annotations.
        """
        annotations_subset = Annotations()
        for ann_id in list(self.data.keys())[:length]:
            annotations_subset.data[ann_id] = self.data[ann_id]
        return annotations_subset

    def subset_by_prefix(self, prefix):
        """Return a subset of annotations with IDs starting with a prefix.

        Args:
            prefix (str): Prefix to filter annotation IDs.

        Returns:
            Annotations: New container with matching annotations.
        """
        annotations_subset = Annotations()
        for ann in self.data.values():
            if ann.id.startswith(prefix):
                annotations_subset.data[ann.id] = ann
        return annotations_subset

    def subset_by_label(self, label_string):
        """Return a subset of annotations with the given label.

        Args:
            label_string (str): Label value to filter annotations.

        Returns:
            Annotations: New container with annotations that match the label.
        """
        annotations_subset = Annotations()
        for ann in self.data.values():
            if ann.label == label_string:
                annotations_subset.data[ann.id] = ann
        return annotations_subset

    def subset_by_range(self, start_idx, end_idx):
        """Return a subset of annotations based on index range.

        Args:
            start_idx (int): Starting index.
            end_idx (int): Ending index.

        Returns:
            Annotations: New container with annotations in the range.
        """
        annotations_subset = Annotations()
        for ann_id in list(self.data.keys())[start_idx:end_idx]:
            annotations_subset.data[ann_id] = self.data[ann_id]
        return annotations_subset

    def get_point_cloud_by_radius(self, source_pcd, radius):
        """Get a point cloud around each annotation within a radius.

        Args:
            source_pcd: Source point cloud.
            radius (float): Radius for subsampling.
        """
        for ann in tqdm(
            self.data.values(), desc="Subsampling pointcloud for each annotation"
        ):
            ann.simple_pcd = source_pcd.subsample_pointcloud_by_radius(
                ann.coords, radius
            )

    def measure_all(self, measurement_func, *args, **kwargs):
        """Conduct measurements for all annotations.

        Args:
            measurement_func (callable): Function to measure an annotation.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        if measurement_func.__name__ in ["get_mask_surface_area"]:
            # Do not parallelize (cannot pickle)
            results = {}
            for ann in tqdm(
                self.data.values(),
                desc="Conducting {} measurement for all annotations...".format(
                    measurement_func
                ),
            ):
                ann_id, output = ann.measure(measurement_func, *args, **kwargs)
                results[ann_id] = output
        else:
            with tqdm_joblib(
                tqdm(
                    desc="Conducting {} measurement for all annotations...".format(
                        measurement_func
                    ),
                    total=len(self.data),
                )
            ):
                results_list = Parallel(n_jobs=-1)(
                    delayed(ann.measure)(measurement_func, *args, **kwargs)
                    for ann in self.data.values()
                )
            results = dict(results_list)

        for id, output in results.items():
            if measurement_func.__name__ == "calc_gap_fraction":
                self.data[id].measurements["gapF_raw"] = output[0]
                self.data[id].measurements["gapF_fill"] = output[1]
            elif measurement_func.__name__ == "get_rgb_stats":
                self.data[id].measurements["median_red"] = output[0]
                self.data[id].measurements["median_green"] = output[1]
                self.data[id].measurements["median_blue"] = output[2]
                self.data[id].measurements["luminance"] = output[3]
            elif measurement_func.__name__ == "get_dev_rugosity":
                self.data[id].measurements["dev_rug"] = output[0]
            elif measurement_func.__name__ == "get_vector_dispersion":
                self.data[id].measurements["vector_disp"] = output[0]
            elif measurement_func.__name__ == "get_mask_surface_area":
                self.data[id].measurements["SA_in_cm2"] = output[0]
            elif measurement_func.__name__ == "get_plane_angles":
                self.data[id].measurements["theta"] = output[0]
                self.data[id].measurements["psi"] = output[1]
                self.data[id].measurements["elevation"] = output[2]

    def save(self, filepath):
        """Save the annotations to a CSV file.

        Args:
            filepath (str): Output file path.
        """
        output_lines = []
        # Header - core columns
        col_headers = [
            "id",
            "orig_x",
            "orig_y",
            "orig_z",
            "label",
            "label_conf",
            "world_x",
            "world_y",
            "world_z",
        ]
        # Header - specific to InterceptAnnotation instances
        first_annotation = next(iter(self.data.values()), None)
        if isinstance(first_annotation, InterceptAnnotation):
            col_headers.extend(
                [
                    "search_radius",
                    "is_extrapolated",
                    "estimated_intercept_world_coords",
                ]
            )

        # Header - metadata and measurements
        col_headers_meta = set()
        for ann in self.data.values():
            col_headers_meta.update(ann.meta_data.keys())
        col_headers.extend(sorted(col_headers_meta))
        col_headers_measure = set()
        for ann in self.data.values():
            col_headers_measure.update(ann.measurements.keys())
        col_headers.extend(sorted(col_headers_measure))
        output_lines.append(col_headers)

        for ann in self.data.values():
            row = []

            # Core columns (id, orig_coords, label, world_coords)
            row.append(ann.id)
            row += [value for value in ann.orig_coords]
            if hasattr(ann, "label"):
                row.append(ann.label)
            else:
                row.append("NA")
            if hasattr(ann, "label_conf"):
                row.append(ann.label_conf)
            else:
                row.append("NA")
            row += [value for value in ann.coords]

            # Columns specific to InterceptAnnotation instances
            if isinstance(ann, InterceptAnnotation):
                row.append(ann.search_radius)
                row.append(ann.is_extrapolated)
                row.append(ann.estimated_intercept_coords)
            # Metadata
            for name in sorted(col_headers_meta):
                if name in ann.meta_data:
                    row.append(ann.meta_data[name])
                else:
                    row.append("NA")

            # Measurements
            for name in sorted(col_headers_measure):
                if name in ann.measurements:
                    row.append(ann.measurements[name])
                else:
                    row.append("NA")
            output_lines.append(row)

        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerows(output_lines)

    def __determine_col_order(self, line):
        """Determine column order from either header or a line of data"""

        def get_col_index(columns, names, mandatory=True):
            # Strip quotation marks from each column value directly
            columns = [col.strip('"') for col in columns]
            for name in names:
                if name in columns:
                    return columns.index(name)
            if mandatory:
                raise ValueError(
                    "Compulsory header columns missing in annotations file: {0} not in {1}".format(
                        ", ".join(names), ", ".join(columns)
                    )
                )
            else:
                return None

        # Try to establish column indexes by assuming header
        cols = line.rstrip().split(",")
        return {
            "id": get_col_index(cols, ["id"]),
            "orig_x": get_col_index(cols, ["x", "orig_x"]),
            "orig_y": get_col_index(cols, ["y", "orig_y"]),
            "orig_z": get_col_index(cols, ["z", "orig_z"]),
            "label": get_col_index(cols, ["label"]),
            "label_conf": get_col_index(cols, ["label_conf"], mandatory=False),
            "world_x": get_col_index(cols, ["world_x"], mandatory=False),
            "world_y": get_col_index(cols, ["world_y"], mandatory=False),
            "world_z": get_col_index(cols, ["world_z"], mandatory=False),
        }

    def __get_annotation_fields(self, cols):
        """Get annotation values"""

        primary_field_indices = set(self.col_order.values())
        other_fields = [
            value for idx, value in enumerate(cols) if idx not in primary_field_indices
        ]

        def get_value(key):
            idx = self.col_order.get(key)
            return cols[idx] if idx is not None and idx < len(cols) else None

        return (
            get_value("id"),
            get_value("orig_x"),
            get_value("orig_y"),
            get_value("orig_z"),
            get_value("label"),
            get_value("label_conf"),
            get_value("world_x"),
            get_value("world_y"),
            get_value("world_z"),
            other_fields,
        )

    @staticmethod
    def __strip_post_fixes(ann_id):
        """Remove postfixes from annotation id."""
        for substring in settings.ANN_ID_POST_FIXES:
            ann_id = ann_id.replace(substring, "")
        return ann_id


class Annotation:
    """Class that holds information about an annotation."""

    def __init__(self, coords, id=None, parent=None):
        self.coords = self.orig_coords = np.asarray(coords, dtype=float)
        self.id = id
        self.parent = parent
        self.image_match = None  # selected image match for measurements
        self.image_matches = []
        self.classification = None
        self.simple_pcd = None
        self.meta_data = {}
        self.measurements = {}
        self.extra_coords = {}
        self.orig_extra_coords = {}

    def add_extra_coords(self, line):
        """Add extra coordinates to the annotation.

        Args:
            line (str): A line with extra coordinate data.
        """
        cols = line.split(",")
        full_id = cols[self.parent.col_order["id"]]
        self.extra_coords[full_id] = np.array(
            [
                cols[self.parent.col_order["orig_x"]],
                cols[self.parent.col_order["orig_y"]],
                cols[self.parent.col_order["orig_z"]],
            ],
            dtype=float,
        )
        self.orig_extra_coords[full_id] = self.extra_coords[full_id]

    def get_radius_from_extra_coords(self):
        """Calculate the radius of the annotation using extra coordinates.

        Returns:
            float: Radius value.
        """
        coords = [value for value in self.extra_coords.values()]
        distmat = np.sqrt(
            np.sum(
                (
                    np.array(coords)[:, np.newaxis, :]
                    - np.array(coords)[np.newaxis, :, :]
                )
                ** 2,
                axis=-1,
            )
        )
        return np.nanmax(distmat) / 2

    def get_radius_from_2D_surface_area(self):
        """Calculate the radius in meters using the 2D surface area in cm².

        Returns:
            float: Radius in meters.
        """
        # Calculate radius in centimeters then convert to meters.
        radius_cm = np.sqrt(self.measurements["SA_in_cm2"] / np.pi)
        return radius_cm / 100

    def get_point_cloud_by_radius(self, source_pcd, radius):
        """Get a point cloud for annotation by sampling a point cloud within a radius.

        Args:
            source_pcd: Source point cloud.
            radius (float): Radius for subsampling.
        """
        self.simple_pcd = source_pcd.subsample_pointcloud_by_radius(self.coords, radius)

    def get_hom_coords(self):
        """Return the annotation coordinates in homogeneous format.

        Returns:
            np.ndarray: [x, y, z, 1]
        """
        return np.array(
            [self.coords[0], self.coords[1], self.coords[2], 1], dtype=float
        )

    def transform_coords(self, transform):
        """Apply a transformation to the annotation coordinates.

        Args:
            transform (np.ndarray): Transformation matrix.
        """
        self.coords = geometry.transform_coords(self.coords, transform)
        for full_id in self.extra_coords:
            self.extra_coords[full_id] = geometry.transform_coords(
                self.extra_coords[full_id], transform
            )

    def reverse_transform_coords(self, transform):
        """Revert the transformation of annotation coordinates.

        Args:
            transform (np.ndarray): Transformation matrix.
        """
        inverse_transform = np.linalg.inv(transform)
        self.orig_coords = geometry.transform_coords(self.coords, inverse_transform)

    def get_image_matches(
        self,
        cams,
        max_cams=None,
        pcd=None,
        use_orig_coords=True,
        intercept_radius=settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        reprojection_threshold_uncertain=settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
        reprojection_threshold_discard=settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
    ):
        """Get all cameras where the annotation is in view."""
        image_matches = []
        for cam in cams:
            # Get pixel coordinates for each camera
            x, y, depth, relevance = cam.get_pixel_coords(
                self.orig_coords, use_orig_coords=use_orig_coords
            )
            if x is not None:
                # If pixel coordinates are within the camera bounds
                image_match = cameras.ImageMatch(
                    cam, x, y, depth, relevance, annotation=self
                )
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
                    print("No pcd provided")
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

            # Return no more than max_cams
            if max_cams and len(image_matches) > max_cams:
                self.image_matches = image_matches[:max_cams]
                self.image_match = image_matches[0]
                return image_matches[:max_cams]
            else:
                self.image_matches = image_matches
                self.image_match = image_matches[0]
                return image_matches

    def get_first_image_match(
        self,
        cams,
        pcd=None,
        use_orig_coords=True,
        intercept_radius=settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        reprojection_threshold_uncertain=settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
        reprojection_threshold_discard=settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
    ):
        """Get the most relevant image match.

        Args:
            cams (list): List of camera objects.
            pcd: Optional point cloud for occlusion filtering.

        Returns:
            ImageMatch or None: Top image match if available.
        """
        image_matches = self.get_image_matches(
            cams,
            1,
            pcd=pcd,
            use_orig_coords=use_orig_coords,
            intercept_radius=intercept_radius,
            reprojection_threshold_uncertain=settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
            reprojection_threshold_discard=settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
        )
        if image_matches:
            return image_matches[0]
        else:
            return None

    def measure(self, measurement_func, *args, **kwargs):
        """Execute a measurement function for this annotation.

        Args:
            measurement_func (callable): Measurement function.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: (annotation id, result)
        """
        if measurement_func.__name__ == "calc_gap_fraction":
            gapF_raw, gapF_fill, _ = measurement_func(self, *args)
            self.measurements["gapF_raw"] = gapF_raw
            self.measurements["gapF_fill"] = gapF_fill
            return self.id, [gapF_raw, gapF_fill]
        elif measurement_func.__name__ == "get_rgb_stats":
            r, g, b, lum = measurement_func(self.simple_pcd)
            self.measurements["median_red"] = r
            self.measurements["median_green"] = g
            self.measurements["median_blue"] = b
            self.measurements["luminance"] = lum
            return self.id, [r, g, b, lum]
        elif measurement_func.__name__ == "get_dev_rugosity":
            dev_rug = measurement_func(self.simple_pcd)
            self.measurements["dev_rug"] = dev_rug
            return self.id, [dev_rug]
        elif measurement_func.__name__ == "get_vector_dispersion":
            vector_disp = measurement_func(self.simple_pcd)
            self.measurements["vector_disp"] = vector_disp
            return self.id, [vector_disp]
        elif measurement_func.__name__ == "get_mask_surface_area":
            SA_in_cm2 = measurement_func(self, *args)
            self.measurements["SA_in_cm2"] = SA_in_cm2
            return self.id, [SA_in_cm2]
        elif measurement_func.__name__ == "get_plane_angles":
            theta, psi, elevation, plane_coeffs = measurement_func(self.simple_pcd)
            self.measurements["theta"] = theta
            self.measurements["psi"] = psi
            self.measurements["elevation"] = elevation
            self.measurements["plane_coeffs"] = plane_coeffs
            return self.id, [theta, psi, elevation, plane_coeffs]
        else:
            logger.error("Measurement not recognized!")
        return self.id, None

    def get_crosshair_points(self, plane_normal, offset_m=0.01):
        """
        Compute four offset 3D points in a plane defined by the normal.

        Args:
            plane_normal (np.ndarray): Normal vector for the plane.
            offset_m (float, optional): Offset in meters.

        Returns:
            Annotations: New annotation container with four offset points.
        """
        n = plane_normal / np.linalg.norm(plane_normal)
        if abs(n[0]) < 0.9:
            a = np.array([1, 0, 0], dtype=float)
        else:
            a = np.array([0, 1, 0], dtype=float)
        u = a - np.dot(a, n) * n
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return Annotations(
            coords=[
                self.coords + offset_m * u,
                self.coords - offset_m * u,
                self.coords + offset_m * v,
                self.coords - offset_m * v,
            ]
        )

    def set_image_mask_id(self, mask_id):
        self.image_match.mask = self.image_match.masks[mask_id]


class InterceptAnnotation(Annotation):
    def __init__(
        self,
        coords,
        search_radius,
        is_extrapolated=False,
        estimated_intercept_coords=None,
        parent=None,
        id=None,
        neighboring_coords=None,
    ):
        # Use intercept_point.coords as the main coordinates.
        super().__init__(coords, id=id, parent=parent)

        self.search_radius = search_radius
        self.is_extrapolated = is_extrapolated
        self.estimated_intercept_coords = estimated_intercept_coords
        if neighboring_coords is not None:
            self.simple_pcd = pointclouds.SimplePointCloud(neighboring_coords)


class Scalebars:
    """
    Container class that holds a collection of Scalebar objects
    """

    def __init__(self, scalebar_data, target_data=None):
        self.data = [
            Scalebar(pred_scalebar[0], pred_scalebar[1], pred_scalebar[2])
            for pred_scalebar in scalebar_data
        ]
        if target_data is not None:
            # If target_data is an Annotations instance, convert to dict of coords
            if hasattr(target_data, "data") and isinstance(target_data.data, dict):
                # Assume keys are labels, values are Annotation objects
                target_data_dict = {
                    ann.label if hasattr(ann, "label") else key: [ann.coords]
                    for key, ann in target_data.data.items()
                }
                self.store_target_coords(target_data_dict)
            else:
                self.store_target_coords(target_data)

    def store_target_coords(self, target_data):
        for target_label, target_coords in target_data.items():
            for scalebar in self.data:
                if target_label == scalebar.target1_label:
                    scalebar.target1_coords = np.asarray(target_coords[0], dtype=float)
                elif target_label == scalebar.target2_label:
                    scalebar.target2_coords = np.asarray(target_coords[0], dtype=float)
        self.calc_scalefactor()

    def calc_scalefactor(self, max_var=0.005):
        scalefactors = []
        for scalebar in self.data:
            scalefactor = scalebar.calc_scalefactor()
            if scalefactor:
                scalefactors.append(scalefactor)

        # Calculate the mean/var/sterr
        if len(scalefactors) > 0:
            self.scalebars = len(scalefactors)
            self.scalefactor = sum(scalefactors) / self.scalebars
            self.var = (
                sum((x - self.scalefactor) ** 2 for x in scalefactors) / self.scalebars
            )
            self.sterr = np.sqrt(self.var) / np.sqrt(self.scalebars)
            # logger.info(
            #     f"Scale factor: {self.scalefactor}, Sterr: {self.sterr}, Var: {self.var}, {self.scalebars} scalebars"
            # )
            if self.var > max_var:
                print(f"WARNING: Scale factor variance is too high: {self.var}")
            return self.scalefactor
        else:
            return None

    def _generate_scalebar_figs(self, pcd):
        """Generate matplotlib figures for each scalebar target."""
        figs = []
        for scalebar in self.data:
            if (
                scalebar.target1_coords is not None
                and scalebar.target2_coords is not None
            ):
                radius = float(scalebar.length) * 5
                target1 = pcd.subsample_pointcloud_by_radius(
                    scalebar.target1_coords, radius
                )
                target2 = pcd.subsample_pointcloud_by_radius(
                    scalebar.target2_coords, radius
                )
                fig1 = visualizations.plot_2d(
                    target1,
                    width=4,
                    height=4,
                    highlight_coords=scalebar.target1_coords,
                    title=scalebar.target1_label + "\n" + str(scalebar.target1_coords),
                )
                fig2 = visualizations.plot_2d(
                    target2,
                    width=4,
                    height=4,
                    highlight_coords=scalebar.target2_coords,
                    title=scalebar.target2_label + "\n" + str(scalebar.target2_coords),
                )
                figs.append(fig1)
                figs.append(fig2)
        return figs

    def show(self, pcd):
        """Visualize the scale bar targets"""
        print(
            f"Number of scalebars: {self.scalebars}\nScale factor: {self.scalefactor:.5f}\nVariance: {self.var:.10f}\nStd Error: {self.sterr:.10f}"
        )
        figs = self._generate_scalebar_figs(pcd)
        # Show the figures interactively
        for fig in figs:
            fig.show()
        return figs

    def save_pdf(self, pcd, filepath=None):
        """Save the scalebar visualization as a PDF (does not display figures)."""
        import matplotlib

        backend_original = matplotlib.get_backend()
        # Use a non-interactive backend to prevent showing figures
        matplotlib.use("Agg", force=True)
        try:
            if filepath is None:
                base, ext = os.path.splitext(pcd.filepath)
                filepath = f"{base}_scalebars.pdf"

            pdf = PdfPages(filepath)
            # Text summary
            fig = visualizations.plot_text(
                f"Number of scalebars: {self.scalebars}\n"
                f"Scale factor: {self.scalefactor:.5f}\n"
                f"Variance: {self.var:.10f}\n"
                f"Std Error: {self.sterr:.10f}"
            )
            pdf.savefig(fig)
            figs = self._generate_scalebar_figs(pcd)
            # Scalebar visualizations
            for fig in figs:
                pdf.savefig(fig)
            pdf.close()
        finally:
            # Restore the original backend
            matplotlib.use(backend_original, force=True)


class Scalebar(object):
    """
    Scalebar
    """

    def __init__(self, target1_label, target2_label, length):
        self.target1_label = target1_label
        self.target2_label = target2_label
        self.length = length
        self.target1_coords = None
        self.target2_coords = None

    def calc_scalefactor(self):
        if self.target1_coords is not None and self.target2_coords is not None:
            x1 = float(self.target1_coords[0])
            y1 = float(self.target1_coords[1])
            z1 = float(self.target1_coords[2])
            x2 = float(self.target2_coords[0])
            y2 = float(self.target2_coords[1])
            z2 = float(self.target2_coords[2])
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            self.scalefactor = self.length / dist
            # logger.info(
            #     f"Scalebar: {self.target1_label} - {self.target2_label}: {dist} m"
            #     f" ({self.length} m)"
            #     f"scalefactor: {self.scalefactor}"
            # )
            return self.scalefactor
        else:
            return None
