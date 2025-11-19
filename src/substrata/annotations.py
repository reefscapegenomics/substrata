# Standard Library

import csv
import logging
import os
import random
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy.typing as npt
    from substrata import cameras, pointclouds

# Third-Party Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# Local Modules
from substrata import (
    settings,
    measurements,
    cameras,
    visualizations,
    geom,
    pointclouds,
)
from substrata.logging import logger


class Annotations:
    """
    Container class that holds a collection of Annotation objects
    """

    # Field configuration: (field_name, [possible_column_names], mandatory)
    _FIELD_CONFIG = [
        ("id", ["id"], True),
        ("orig_x", ["x", "orig_x"], True),
        ("orig_y", ["y", "orig_y"], True),
        ("orig_z", ["z", "orig_z"], True),
        ("label", ["label"], True),
        ("label_conf", ["label_conf"], False),
        ("world_x", ["world_x"], False),
        ("world_y", ["world_y"], False),
        ("world_z", ["world_z"], False),
        ("cam_filepath", ["cam_filepath"], False),
        ("cam_x", ["cam_x"], False),
        ("cam_y", ["cam_y"], False),
        ("depth_sensor_m", ["depth"], False),
    ]

    def __init__(
        self,
        filepath: Optional[str] = None,
        coords: Optional[
            Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]
        ] = None,
        header: bool = True,
        orig_coords_only: bool = False,
        ignore_header: bool = False,
    ) -> None:
        self.data = {}
        self.measurements = {}
        self.world_transform = np.eye(4)

        if filepath is not None:
            self.get_annotations_from_file(
                filepath, header=header, orig_coords_only=orig_coords_only
            )
        elif coords is not None:
            self.get_annotations_from_coords(coords)

    def __getitem__(self, key: str) -> "Annotation":
        return self.data[key]

    def __setitem__(self, key: str, value: "Annotation") -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self) -> "Annotations":
        self._iter = iter(self.data.values())
        return self

    def __next__(self) -> "Annotation":
        return next(self._iter)

    def __len__(self) -> int:
        return len(self.data)

    def items(self) -> Any:
        return self.data.items()

    @property
    def coords(self) -> List[np.ndarray]:
        return [annotation.coords for annotation in self.data.values()]

    @property
    def classifications(self) -> List[Optional[str]]:
        return [annotation.classification for annotation in self.data.values()]

    @property
    def image_matches(self) -> Dict[str, Any]:
        return {
            annotation.id: annotation.image_match
            for annotation in self.data.values()
            if annotation.image_match is not None
        }

    @property
    def world_transform_is_identity(self) -> bool:
        """Check if the world_transform is the identity matrix."""
        return np.allclose(self.world_transform, np.eye(4))

    def show(self, pcd: Any, color=False) -> None:
        """Show annotation positions overlaid on a point cloud.

        Args:
            pcd: Point cloud to draw as background.
        """
        visualizations.plot_positions(self, pcd, color=color)

    def append(self, annotation: "Annotation") -> None:
        if annotation.id in self.data:
            raise ValueError(f"Annotation with id {annotation.id} already exists.")
        else:
            self.data[annotation.id] = annotation
            self.data[annotation.id].parent = self
            # TO DO: any other changes (eg transforms) to be implemented on append?

    def get_annotations_from_file(
        self,
        annotations_filepath: str,
        header: bool = True,
        orig_coords_only: bool = False,
        ignore_header: bool = False,
    ) -> None:
        """Read in annotations from a file and store in dict.

        Args:
            annotations_filepath: Path to the file.
            header: Whether the file has a header row.
            orig_coords_only: Whether to only use original coordinates.
            ignore_header: If True, skip the first line and ignore the header argument.
        """
        annotations_file = open(annotations_filepath, "r")

        # If ignore_header is True, skip first line and treat as no header
        if ignore_header:
            next(annotations_file, None)  # Skip first line
            header = False

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
                cam_filepath,
                cam_x,
                cam_y,
                other_cols,
                depth_sensor_m,
            ) = self.__get_annotation_fields(line.rstrip("\r\n").split(","))
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
                # Load optional camera fields if present
                self.data[id].cam_filepath = (
                    cam_filepath if cam_filepath not in [None, ""] else None
                )
                self.data[id].cam_x = float(cam_x) if cam_x not in [None, ""] else None
                self.data[id].cam_y = float(cam_y) if cam_y not in [None, ""] else None
                self.data[id].depth_sensor_m = (
                    float(depth_sensor_m) if depth_sensor_m not in [None, ""] else None
                )
                self.data[id].other_cols = other_cols
            else:
                # Additional coordinates for existing annotation
                self.data[ann_id].add_extra_coords(line.rstrip("\r\n"))
        annotations_file.close()

    def get_annotations_from_coords(
        self,
        annotations_coords: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray],
    ) -> None:
        """Use a dict or list of coordinates to fill the annotations class.

        Args:
            annotations_coords: Coordinate data as dict, list, or np.ndarray.
        """
        if isinstance(annotations_coords, dict):
            for i, (key, coords) in enumerate(annotations_coords.items()):
                if coords is not None:
                    self.data[key] = Annotation(coords, id=key, parent=self)
        elif isinstance(annotations_coords, (list, np.ndarray)):
            for i, coords in enumerate(annotations_coords):
                if coords is not None:
                    self.data[i] = Annotation(coords, id=i, parent=self)

    def get_annotations_from_google_worksheet(
        self, worksheet: Any, header: bool = True
    ) -> None:
        """Use a Google worksheet to fill the annotations class.

        Args:
            worksheet: Google worksheet object.
            header: Whether the worksheet has a header row.
        """
        worksheet_data = worksheet.get_all_values()
        for row_number, row_cols in enumerate(worksheet_data, start=1):
            if row_number == 1 and header:
                self.col_order = self.__determine_col_order(",".join(row_cols))
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
                cam_filepath,
                cam_x,
                cam_y,
                other_cols,
            ) = self.__get_annotation_fields(row_cols)
            if id not in self.data:
                # New annotation
                self.data[id] = Annotation([orig_x, orig_y, orig_z], id=id, parent=self)
                self.data[id].line_no = row_number
                self.data[id].label = self.data[id].classification = label
                self.data[id].label_conf = label_conf
                if world_x is not None:
                    self.data[id].coords = np.asarray(
                        [world_x, world_y, world_z], dtype=float
                    )
                # Load optional camera fields if present
                self.data[id].cam_filepath = (
                    cam_filepath if cam_filepath not in [None, ""] else None
                )
                self.data[id].cam_x = float(cam_x) if cam_x not in [None, ""] else None
                self.data[id].cam_y = float(cam_y) if cam_y not in [None, ""] else None
                self.data[id].other_cols = other_cols
            else:
                # Additional coordinates for existing annotation
                self.data[id].add_extra_coords(",".join(row_cols))

    def add_meta_data(self, data_filepath: str) -> None:
        """Add metadata to annotations from a CSV file (requires header).

        Args:
            data_filepath: Path to the CSV file.
        """
        data_file = open(data_filepath, "r")
        col_headers = None
        for line_no, line in enumerate(data_file):
            if line_no == 0:
                self.col_order = self.__determine_col_order(line)
                cols = line.rstrip("\r\n").split(",")
                col_headers = [col.strip('"') for col in cols]
                continue  # skip to next line

            cols = line.rstrip("\r\n").split(",")
            ann_id = cols[self.col_order["id"]]
            if ann_id in self.data.keys():
                self.data[ann_id].meta_data = {}
                if col_headers is not None:
                    for i in range(0, len(col_headers)):
                        if i + 1 < len(cols):
                            self.data[ann_id].meta_data[col_headers[i]] = cols[i + 1]
            else:
                print(f"No annotation with ID {ann_id} found.")
        data_file.close()

    def get_new_id(
        self,
        last_highest_id: Optional[int] = None,
        default_prefix: Optional[str] = None,
    ) -> str:
        """Return an identifier based on the next available integer.

        Args:
            last_highest_id: Optional last highest ID to check against.
            default_prefix: Optional prefix to use if no existing IDs found.

        Returns:
            New annotation ID string.
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

    def get_bounding_box(self) -> List[np.ndarray]:
        """Return the min and max values of x, y, z for all points in annotations.

        Returns:
            List containing [min_coords, max_coords] for x, y, z.
        """
        xyz_min = np.min(self.coords, axis=0)
        xyz_max = np.max(self.coords, axis=0)
        return [xyz_min, xyz_max]

    def get_eucl_distance_matrix(self) -> pd.DataFrame:
        """Calculates pairwise Euclidean distances and returns a DataFrame.

        Returns:
            Distance matrix with annotation keys as rows and columns.
        """
        keys = list(self.data.keys())
        coords = np.array([annotation.coords for annotation in self.data.values()])
        distmat = np.sqrt(
            np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1)
        )
        return pd.DataFrame(distmat, index=keys, columns=keys)

    def get_first_image_matches(
        self, cams: List[Any], pcd: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Get the first image match for each annotation.

        Args:
            cams: List of camera objects.
            pcd: Optional point cloud for filtering.

        Returns:
            Dictionary mapping annotation IDs to image matches.
        """
        image_matches = {}
        for ann in tqdm(self.data.values(), desc="Getting first image matches"):
            try:
                match = ann.get_first_image_match(cams, pcd)
                if match:
                    image_matches[ann.id] = match
                    # ann.image_matches.append(match) TODO: INCORRECT
            except Exception as e:
                print(
                    f"Warning: Failed to get image match for annotation {ann.id}: {e}"
                )
                continue
        return image_matches

    def classify_image_matches(
        self,
        classifier: Any,
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        print_summary: bool = False,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Classify all image matches for annotations that have them.

        Args:
            classifier: Loaded FastAI learner or path to a .pkl learner.
            crop_size: Optional int (square) or (width, height) tuple for center crop.
            print_summary: Whether to print classification category counts.

        Returns:
            Mapping of annotation IDs to classification results.
        """
        results = {}
        for ann in tqdm(self.data.values(), desc="Classifying image matches"):
            if ann.image_match is not None:
                try:
                    result = ann.image_match.classify(classifier, crop_size)
                    results[ann.id] = result
                except Exception as e:
                    logger.warning(
                        f"Classification failed for annotation {ann.id}: {e}"
                    )
                    results[ann.id] = None
            else:
                results[ann.id] = None

        # Print summary if requested
        if print_summary:
            self._print_classification_summary(results)

        return results

    def assign_image_match_classification_to_label(self) -> None:
        """Iterate over all annotations and assign the classification to the label."""
        for ann in self.data.values():
            if ann.image_match is not None:
                ann.label = ann.image_match.classification["label"]
                ann.label_conf = -1
            else:
                ann.label = None
                ann.label_conf = None

    def _print_classification_summary(
        self, results: Dict[str, Optional[Dict[str, Any]]]
    ) -> None:
        """Print a summary of classification results.

        Args:
            results: Classification results from classify_image_matches.
        """
        # Count classifications
        label_counts = {}
        total_annotations = len(results)
        successful_classifications = 0

        for ann_id, result in results.items():
            if result is not None and isinstance(result, dict):
                label = result.get("label", "Unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
                successful_classifications += 1

        # Print summary
        print(f"\nClassification Summary:")
        print(f"Total annotations: {total_annotations}")
        print(f"Successfully classified: {successful_classifications}")
        print(
            f"Failed classifications: {total_annotations - successful_classifications}"
        )

        if label_counts:
            print(f"\nClassification categories:")
            for label, count in sorted(label_counts.items()):
                percentage = (count / successful_classifications) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
        print()

    def get_up_vector_from_annotation_depths(
        self,
        plot: bool = False,
    ) -> Tuple[
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
    ]:
        """Compute the up vector using least-squares regression on annotation depths.

        Fits a linear regression between the annotation 3D points and their sensor
        depths to find the dominant depth direction. Also stores predicted depths
        and errors.

        Args:
            plot (bool): If True, create a visualization of the regression fit.

        Returns:
            Tuple containing:
                - up_vector (np.ndarray): The coefficient vector representing the
                  up vector.
                - depth_offset (float): Depth offset from regression.
                - depth_per_unit (float): Depth per unit from regression.
                - mse (float): Mean squared error.
                - rmse (float): Root mean squared error.
                - mae (float): Mean absolute error.
                - r2 (float): R² value.
                - num_matches (int): Number of annotations used.
        """
        # Get annotations with depth_sensor_m and coords
        # (no filtering by accuracy threshold)
        anns_with_depth = [
            ann
            for ann in self.data.values()
            if (
                hasattr(ann, "depth_sensor_m")
                and hasattr(ann, "coords")
                and ann.depth_sensor_m is not None
                and ann.coords is not None
            )
        ]
        print(
            f"Found {len(anns_with_depth)} matching "
            f"annotations/depths for regression"
        )

        # Conduct regression on the annotations
        points = np.array([ann.coords for ann in anns_with_depth])
        depths = np.array([ann.depth_sensor_m for ann in anns_with_depth])

        res = measurements.fit_depth_regression(points, depths)

        # Plot the regression fit if requested
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
        print(f"  Number of matches: {len(anns_with_depth)}")

        # Return the up vector and error metrics, plus number of matches
        return (
            res.up_vector,
            res.depth_offset,
            res.depth_per_unit,
            res.mse,
            res.rmse,
            res.mae,
            res.r2,
            len(anns_with_depth),
        )

    @property
    def depth_residuals(self):
        """Access depth residual analysis methods."""
        from substrata.measurements import DepthResidualAnalyzer

        return DepthResidualAnalyzer(self)

    def get_depths_and_estimated_depths(self):
        """Get sensor depths and predicted depths for annotations.

        Returns:
            tuple: (depths, est_depths, filtered_annotations)
        """
        return self.depth_residuals.get_depths_and_estimated_depths()

    def get_depths_and_z_coords(self):
        """Get sensor depths and z-coordinates for annotations.

        Returns:
            tuple: (depths, z_coords, filtered_annotations)
        """
        return self.depth_residuals.get_depths_and_z_coords()

    def show_depth_vs_est_depth_residuals(self, width=15, height=5):
        """Show residuals between predicted and recorded depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)

        Returns:
            tuple: (fig1, fig2) matplotlib figure objects
        """
        return self.depth_residuals.show_depth_vs_est_depth_residuals(
            width=width, height=height
        )

    def show_z_vs_depth_residuals(self, width=15, height=5):
        """Show residuals between z-coordinates and recorded depths.

        Args:
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)

        Returns:
            tuple: (fig1, fig2) matplotlib figure objects
        """
        return self.depth_residuals.show_z_vs_depth_residuals(
            width=width, height=height
        )

    def save_depth_residuals_pdf(self, filepath=None, width=15, height=5):
        """Save depth residuals visualization as a PDF.

        Args:
            filepath (str, optional): Path to save the PDF file.
            width (float): Figure width in inches (default: 15)
            height (float): Figure height in inches (default: 5)

        Returns:
            str: The filepath where the PDF was saved.
        """
        return self.depth_residuals.save_depth_residuals_pdf(
            filepath=filepath, width=width, height=height
        )

    # def get_pcd(self):
    #     """Get a point cloud of all annotation coordinates."""
    #     pcd = geometry.PointCloud()
    #     pcd.points = utility.Vector3dVector(self.get_all_coords())
    #     return pcd``

    def transform_coords(self, transform: Union[np.ndarray, Any]) -> None:
        """Apply a transformation to all annotation coordinates.

        Args:
            transform: A 4x4 transformation matrix or a Transform instance.
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

    def get_original_coords(self, transform_matrix: np.ndarray) -> None:
        """Revert transformed coordinates using the given transformation.

        Args:
            transform_matrix: The transformation matrix to invert.
        """
        for ann_id in self.data:
            self.data[ann_id].reverse_transform_coords(transform_matrix)
        self.world_transform = np.dot(
            np.array(transform_matrix), self.world_transform
        )  # TODO: CHECK!

    def random_subset(self, length: int) -> "Annotations":
        """Return a random subset of annotations.

        Args:
            length: Number of annotations to include.

        Returns:
            New container with the selected annotations.
        """
        annotations_subset = Annotations()
        random_keys = random.sample(list(self.data.keys()), length)
        for ann_id in random_keys:
            annotations_subset.data[ann_id] = self.data[ann_id]
        return annotations_subset

    def subset(self, length: int) -> "Annotations":
        """Return a subset of annotations.

        Args:
            length: Number of annotations to include.

        Returns:
            New container with the selected annotations.
        """
        annotations_subset = Annotations()
        for ann_id in list(self.data.keys())[:length]:
            annotations_subset.data[ann_id] = self.data[ann_id]
        return annotations_subset

    def subset_by_prefix(self, prefix: str) -> "Annotations":
        """Return a subset of annotations with IDs starting with a prefix.

        Args:
            prefix: Prefix to filter annotation IDs.

        Returns:
            New container with matching annotations.
        """
        annotations_subset = Annotations()
        for ann in self.data.values():
            if ann.id.startswith(prefix):
                annotations_subset.data[ann.id] = ann
        return annotations_subset

    def subset_by_label(
        self, label_string_or_list: Union[str, List[str]]
    ) -> "Annotations":
        """Return a subset of annotations with the given label or labels.

        Args:
            label_string_or_list: Label value(s) to filter annotations.

        Returns:
            New container with annotations that match the label(s).
        """
        if isinstance(label_string_or_list, str):
            label_set = {label_string_or_list}
        else:
            label_set = set(label_string_or_list)
        annotations_subset = Annotations()
        for ann in self.data.values():
            if ann.label in label_set:
                annotations_subset.data[ann.id] = ann
        return annotations_subset

    def subset_by_label_prefix(self, prefix: str) -> "Annotations":
        """Return a subset of annotations where the label contains the given prefix.

        Args:
            prefix: Prefix to search for in annotation labels.

        Returns:
            New container with annotations whose label contains the prefix.
        """
        annotations_subset = Annotations()
        for ann in self.data.values():
            if ann.label is not None and prefix in ann.label:
                annotations_subset.data[ann.id] = ann
        return annotations_subset

    def subset_by_range(self, start_idx: int, end_idx: int) -> "Annotations":
        """Return a subset of annotations based on index range.

        Args:
            start_idx: Starting index.
            end_idx: Ending index.

        Returns:
            New container with annotations in the range.
        """
        annotations_subset = Annotations()
        for ann_id in list(self.data.keys())[start_idx:end_idx]:
            annotations_subset.data[ann_id] = self.data[ann_id]
        return annotations_subset

    def _empty_like(self) -> "Annotations":
        """Return an empty Annotations container inheriting this instance's metadata."""
        subset = Annotations()
        for attr, val in self.__dict__.items():
            if attr == "data":
                continue
            try:
                setattr(subset, attr, val)
            except Exception:
                pass
        subset.data = {}
        return subset

    def get_point_cloud_by_radius(self, source_pcd: Any, radius: float) -> None:
        """Get a point cloud around each annotation within a radius.

        Args:
            source_pcd: Source point cloud.
            radius: Radius for subsampling.
        """
        for ann in tqdm(
            self.data.values(), desc="Subsampling pointcloud for each annotation"
        ):
            ann.simple_pcd = source_pcd.subsample_pointcloud_by_radius(
                ann.coords, radius
            )

    def measure_all(
        self, measurement_func: Callable, *args: Any, **kwargs: Any
    ) -> None:
        """Conduct measurements for all annotations.

        Args:
            measurement_func: Function to measure an annotation.
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

    def save(self, filepath: str, orig_coords_only: bool = False) -> None:
        """Save the annotations to a CSV file.

        Args:
            filepath: Output file path.
            orig_coords_only: If True, only output orig_coords and not world coords.
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
        ]
        if not orig_coords_only:
            col_headers.extend(["world_x", "world_y", "world_z"])
        col_headers.extend(["cam_filepath", "cam_x", "cam_y", "depth"])
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
            if not orig_coords_only:
                row += [value for value in ann.coords]

            # Camera columns (from image_match if available; otherwise any loaded values)
            cam_filepath = None
            cam_x = None
            cam_y = None
            if (
                getattr(ann, "image_match", None) is not None
                and getattr(ann.image_match, "cam", None) is not None
            ):
                cam_filepath = getattr(ann.image_match.cam, "filepath", None)
                cam_x = getattr(ann.image_match, "x", None)
                cam_y = getattr(ann.image_match, "y", None)
            else:
                cam_filepath = getattr(ann, "cam_filepath", None)
                cam_x = getattr(ann, "cam_x", None)
                cam_y = getattr(ann, "cam_y", None)
            row.append(cam_filepath if cam_filepath is not None else "")
            row.append(cam_x if cam_x is not None else "")
            row.append(cam_y if cam_y is not None else "")
            depth_sensor_m = getattr(ann, "depth_sensor_m", None)
            row.append(depth_sensor_m if depth_sensor_m is not None else "")

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

    def __determine_col_order(self, line: str) -> Dict[str, Optional[int]]:
        """Determine column order from either header or a line of data.

        Args:
            line: Header line or data line to parse.

        Returns:
            Dictionary mapping column names to indices.
        """

        def get_col_index(
            columns: List[str], names: List[str], mandatory: bool = True
        ) -> Optional[int]:
            # Strip quotation marks and whitespace from each column value
            columns = [col.strip('"').strip() for col in columns]
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
        # Strip line endings (handle both Unix LF and Windows CRLF)
        cols = line.rstrip("\r\n").split(",")
        return {
            field_name: get_col_index(cols, possible_names, mandatory=mandatory)
            for field_name, possible_names, mandatory in self._FIELD_CONFIG
        }

    def __get_annotation_fields(self, cols: List[str]) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[float],
        List[str],
    ]:
        """Get annotation values.

        Args:
            cols: List of column values.

        Returns:
            Tuple containing id, orig_x, orig_y, orig_z, label, label_conf,
            world_x, world_y, world_z, cam_filepath, cam_x, cam_y, and other_fields.
        """

        primary_field_indices = set(self.col_order.values())

        def get_value(key: str) -> Optional[str]:
            idx = self.col_order.get(key)
            return cols[idx] if idx is not None and idx < len(cols) else None

        # Extract all known fields using the field configuration
        field_values = {}
        for field_name, _, _ in self._FIELD_CONFIG:
            field_values[field_name] = get_value(field_name)

        # Compute other_fields last, after all known fields are extracted
        other_fields = [
            value for idx, value in enumerate(cols) if idx not in primary_field_indices
        ]

        # Return in the expected order (matching the unpacking in get_annotations_from_file)
        return (
            field_values["id"],
            field_values["orig_x"],
            field_values["orig_y"],
            field_values["orig_z"],
            field_values["label"],
            field_values["label_conf"],
            field_values["world_x"],
            field_values["world_y"],
            field_values["world_z"],
            field_values["cam_filepath"],
            field_values["cam_x"],
            field_values["cam_y"],
            other_fields,
            field_values["depth_sensor_m"],
        )

    @staticmethod
    def __strip_post_fixes(ann_id: str) -> str:
        """Remove postfixes from annotation id.

        Args:
            ann_id: Annotation ID string.

        Returns:
            Annotation ID with postfixes removed.
        """
        for substring in settings.ANN_ID_POST_FIXES:
            ann_id = ann_id.replace(substring, "")
        return ann_id


class Annotation:
    """Class that holds information about an annotation."""

    def __init__(
        self,
        coords: Union[List[float], np.ndarray],
        id: Optional[str] = None,
        parent: Optional["Annotations"] = None,
    ) -> None:
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
        # Optional camera fields loaded/saved from CSV
        self.cam_filepath: Optional[str] = None
        self.cam_x: Optional[float] = None
        self.cam_y: Optional[float] = None
        self.depth_sensor_m: Optional[float] = None

    def add_extra_coords(self, line: Union[str, List[str]]) -> None:
        """Add extra coordinates to the annotation.

        Args:
            line: A line with extra coordinate data.
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

    def get_radius_from_extra_coords(self) -> float:
        """Calculate the radius of the annotation using extra coordinates.

        Returns:
            Radius value.
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

    def get_radius_from_2D_surface_area(self) -> float:
        """Calculate the radius in meters using the 2D surface area in cm².

        Returns:
            Radius in meters.
        """
        # Calculate radius in centimeters then convert to meters.
        radius_cm = np.sqrt(self.measurements["SA_in_cm2"] / np.pi)
        return radius_cm / 100

    def get_point_cloud_by_radius(self, source_pcd: Any, radius: float) -> None:
        """Get a point cloud for annotation by sampling a point cloud within a radius.

        Args:
            source_pcd: Source point cloud.
            radius: Radius for subsampling.
        """
        self.simple_pcd = source_pcd.subsample_pointcloud_by_radius(self.coords, radius)

    def get_hom_coords(self) -> np.ndarray:
        """Return the annotation coordinates in homogeneous format.

        Returns:
            Array with [x, y, z, 1].
        """
        return np.array(
            [self.coords[0], self.coords[1], self.coords[2], 1], dtype=float
        )

    def transform_coords(self, transform: np.ndarray) -> None:
        """Apply a transformation to the annotation coordinates.

        Args:
            transform: Transformation matrix.
        """
        self.coords = geom.transform_coords(self.coords, transform)
        for full_id in self.extra_coords:
            self.extra_coords[full_id] = geom.transform_coords(
                self.extra_coords[full_id], transform
            )

    def reverse_transform_coords(self, transform: np.ndarray) -> None:
        """Revert the transformation of annotation coordinates.

        Args:
            transform: Transformation matrix.
        """
        inverse_transform = np.linalg.inv(transform)
        self.orig_coords = geom.transform_coords(self.coords, inverse_transform)

    def get_image_matches(
        self,
        cams: List[Any],
        max_cams: Optional[int] = None,
        pcd: Optional[Any] = None,
        use_orig_coords: bool = True,
        intercept_radius: float = settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        reprojection_threshold_uncertain: float = settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
        reprojection_threshold_discard: float = settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
        enabled_cameras_only: bool = True,
        debug: bool = False,
    ) -> List[Any]:
        """Get all cameras where the annotation is in view.

        Args:
            cams: List of camera objects.
            max_cams: Maximum number of matches to return.
            pcd: Optional point cloud for filtering.
            use_orig_coords: Whether to use original coordinates.
            intercept_radius: Search radius for intercepts.
            reprojection_threshold_uncertain: Threshold for uncertain matches.
            reprojection_threshold_discard: Threshold for discarding matches.
            enabled_cameras_only: Whether to only use enabled cameras.
            debug: Whether to print debug information.

        Returns:
            List of image match objects.
        """
        # If pcd is given, check that the annotations transform matches the pcd transform
        if pcd is not None and self.parent is not None:
            if not np.allclose(self.parent.world_transform, pcd.world_transform):
                raise ValueError(
                    "The annotations transform does not match the pcd transform"
                )
        # Iterate over all cams to find matches
        image_matches = []
        for cam in cams:
            # Skip disabled cameras if enabled_cameras_only is True
            if (
                enabled_cameras_only
                and hasattr(cam, "enabled")
                and cam.enabled is False
            ):
                continue

            # Get pixel coordinates for each camera
            if use_orig_coords:
                coords = self.orig_coords
            else:
                coords = self.coords
            x, y, depth, relevance = cam.get_pixel_coords(
                coords, use_orig_coords=use_orig_coords
            )

            # If pixel coordinates are within the camera bounds
            if x is not None:
                if debug:
                    print(
                        f"Cam {cam.cam_id} has pixel match at: {x}, {y}, {depth}, {relevance}"
                    )

                image_match = cameras.ImageMatch(
                    cam, x, y, depth, relevance, annotation=self
                )
                # Classify according to reprojection error if pcd is provided
                if pcd is not None:
                    image_match.get_reprojection_error(pcd, intercept_radius)
                    if debug:
                        print(f"Reprojection error: {image_match.reprojection_error}")
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
        cams: List[Any],
        pcd: Optional[Any] = None,
        use_orig_coords: bool = True,
        intercept_radius: float = settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        reprojection_threshold_uncertain: float = settings.DEFAULT_REPROJECTION_THRESHOLD_UNCERTAIN,
        reprojection_threshold_discard: float = settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD,
    ) -> Optional[Any]:
        """Get the most relevant image match.

        Args:
            cams: List of camera objects.
            pcd: Optional point cloud for occlusion filtering.
            use_orig_coords: Whether to use original coordinates.
            intercept_radius: Search radius for intercepts.
            reprojection_threshold_uncertain: Threshold for uncertain matches.
            reprojection_threshold_discard: Threshold for discarding matches.

        Returns:
            Top image match if available, None otherwise.
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

    def measure(
        self, measurement_func: Callable, *args: Any, **kwargs: Any
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Execute a measurement function for this annotation.

        Args:
            measurement_func: Measurement function.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing (annotation id, result).
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
            theta, psi, elevation, plane_coeffs, azimuth = measurement_func(
                self.simple_pcd
            )
            self.measurements["theta"] = theta
            self.measurements["psi"] = psi
            self.measurements["elevation"] = elevation
            self.measurements["plane_coeffs"] = plane_coeffs
            self.measurements["azimuth"] = azimuth
            return self.id, [theta, psi, elevation, plane_coeffs, azimuth]
        else:
            logger.error("Measurement not recognized!")
        return self.id, None

    def get_crosshair_points(
        self, plane_normal: np.ndarray, offset_m: float = 0.01
    ) -> "Annotations":
        """Compute four offset 3D points in a plane defined by the normal.

        Args:
            plane_normal: Normal vector for the plane.
            offset_m: Offset in meters.

        Returns:
            New annotation container with four offset points.
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

    def set_image_mask_id(self, mask_id: int) -> None:
        """Set the image mask by ID.

        Args:
            mask_id: Index of the mask to use.
        """
        self.image_match.mask = self.image_match.masks[mask_id]


class InterceptAnnotation(Annotation):
    def __init__(
        self,
        coords: Union[List[float], np.ndarray],
        search_radius: float,
        is_extrapolated: bool = False,
        estimated_intercept_coords: Optional[np.ndarray] = None,
        parent: Optional["Annotations"] = None,
        id: Optional[str] = None,
        neighboring_coords: Optional[np.ndarray] = None,
    ) -> None:
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

    def __init__(
        self,
        scalebar_data: List[Tuple[str, str, float]],
        target_data: Optional[Dict] = None,
    ) -> None:
        self.data = [
            Scalebar(pred_scalebar[0], pred_scalebar[1], pred_scalebar[2])
            for pred_scalebar in scalebar_data
        ]
        # Initialize summary attributes
        self.scalebars = None
        self.scalefactor = None
        self.var = None
        self.sterr = None

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

    def __str__(self) -> str:
        """
        Returns a summary of the scalebars and their calculated metrics.
        """
        lines = ["Scalebars("]

        # Basic information
        lines.append(f"  num_scalebars={len(self.data)},")

        # Show individual scalebar details
        for i, scalebar in enumerate(self.data):
            target1_set = scalebar.target1_coords is not None
            target2_set = scalebar.target2_coords is not None
            scalebar_summary = (
                f"  scalebar_{i+1}='{scalebar.target1_label}'"
                f"-'{scalebar.target2_label}' ({scalebar.length}m)"
            )
            if target1_set and target2_set:
                # Calculate scaled distance using raw_dist and overall scalefactor
                if hasattr(scalebar, "raw_dist") and self.scalefactor is not None:
                    scaled_distance = scalebar.raw_dist * self.scalefactor
                    residual = scaled_distance - scalebar.length
                    scalebar_summary += (
                        f" [dist={scaled_distance:.4f}, residual={residual:+.4f}]"
                    )
                else:
                    scalebar_summary += f" [coords_set]"
            elif target1_set or target2_set:
                scalebar_summary += f" [partial_coords]"
            else:
                scalebar_summary += f" [no_coords]"
            lines.append(scalebar_summary + ",")

        # Summary statistics (if calculated)
        if self.scalefactor is not None:
            lines.append(f"  calculated_scalefactor={self.scalefactor:.6f},")
        if self.var is not None:
            lines.append(f"  variance={self.var:.10f},")
        if self.sterr is not None:
            lines.append(f"  std_error={self.sterr:.10f},")
        if self.scalebars is not None:
            lines.append(f"  valid_scalebars={self.scalebars}")

        # Remove trailing comma from last line if present
        if len(lines) > 1 and lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]

        lines.append(")")
        return "\n".join(lines)

    def store_target_coords(self, target_data: Dict) -> None:
        for target_label, target_coords in target_data.items():
            for scalebar in self.data:
                if target_label == scalebar.target1_label:
                    scalebar.target1_coords = np.asarray(target_coords[0], dtype=float)
                elif target_label == scalebar.target2_label:
                    scalebar.target2_coords = np.asarray(target_coords[0], dtype=float)
        self.calc_scalefactor()

    def calc_scalefactor(self, max_var: float = 0.005) -> Optional[float]:
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

    def _generate_scalebar_figs(self, pcd: Any) -> List[Any]:
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

    def show(self, pcd: Any) -> List[Any]:
        """Visualize the scale bar targets.

        Args:
            pcd: The point cloud object.

        Returns:
            List of matplotlib figure objects.
        """
        if self.scalebars is not None:
            print(
                f"Number of scalebars: {self.scalebars}\n"
                f"Scale factor: {self.scalefactor:.5f}\n"
                f"Variance: {self.var:.10f}\n"
                f"Std Error: {self.sterr:.10f}"
            )
            figs = self._generate_scalebar_figs(pcd)
            # Show the figures interactively
            for fig in figs:
                fig.show()
            return figs
        return []

    def save_pdf(self, pcd: Any, filepath: Optional[str] = None) -> None:
        """Save the scalebar visualization as a PDF (does not display figures).

        Args:
            pcd: The point cloud object.
            filepath: Optional path to save the PDF file.

        Returns:
            None
        """
        if self.scalebars is None:
            return

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
    """Scalebar class for storing scale bar information."""

    def __init__(self, target1_label: str, target2_label: str, length: float) -> None:
        self.target1_label = target1_label
        self.target2_label = target2_label
        self.length = length
        self.target1_coords = None
        self.target2_coords = None

    def calc_scalefactor(self) -> Optional[float]:
        if self.target1_coords is not None and self.target2_coords is not None:
            x1 = float(self.target1_coords[0])
            y1 = float(self.target1_coords[1])
            z1 = float(self.target1_coords[2])
            x2 = float(self.target2_coords[0])
            y2 = float(self.target2_coords[1])
            z2 = float(self.target2_coords[2])
            self.raw_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            self.scalefactor = self.length / self.raw_dist
            # logger.info(
            #     f"Scalebar: {self.target1_label} - {self.target2_label}: {dist} m"
            #     f" ({self.length} m)"
            #     f"scalefactor: {self.scalefactor}"
            # )
            return self.scalefactor
        else:
            return None
