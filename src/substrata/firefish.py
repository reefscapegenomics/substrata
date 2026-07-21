# Standard Library
import logging
import os
from datetime import datetime, timezone
import sys

# Third-Party Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages

# Local Modules
from substrata import settings, visualizations, measurements

logger = logging.getLogger(__name__)


class FireFish:
    """Class that holds FireFish sensor"""

    def __init__(self, data_filepath):
        """Initialize the FireFish class by loading sensor data from CSV.

        Args:
            data_filepath (str): Path to the FireFish sensor CSV file.
        """
        self.data = pd.read_csv(
            data_filepath,
            skiprows=2,
            usecols=range(len(settings.FIREFISH_DEFAULT_COLS)),
        )
        self.data.columns = settings.FIREFISH_DEFAULT_COLS

        # Ensure the 'depth' column values are always negative
        if "depth" in self.data.columns:
            self.data["depth"] = -self.data["depth"].abs()

    def remove_outlier_altitudes(self, threshold):
        """Remove altitudes that are above a certain threshold or below 0"""
        len_before_filter = len(self.data)
        self.data = self.data[
            (self.data["altitude"] < threshold) & (self.data["altitude"] >= 0)
        ]
        len_outliers_removed = len_before_filter - len(self.data)
        print(
            "Removed {} altitude outliers from FireFish data...".format(
                len_outliers_removed
            )
        )

    def plot_cams_vs_firefish(
        self, cams, offset=0, time_range=None, show_overlap_range=False
    ):
        """Plot the camera distances against the FireFish altitudes"""
        # cams_with_datetime = [cam for cam in cams if cam.has_coords_datetime_camdist()]
        cams_with_datetime = [
            cam for cam in cams if cam.has_coords_datetime()
        ]  # TO FIX
        cam_dists = [cam.camdist for cam in cams_with_datetime]
        cam_timestamps = [
            get_unix_time(cam.datetime) + offset for cam in cams_with_datetime
        ]

        firefish_timestamps = self.data["unixtime"].to_numpy()
        firefish_altitudes = self.data["altitude"].to_numpy()
        firefish_depths = self.data["depth"].to_numpy()

        # Initialize the RobustScaler
        scaler1 = RobustScaler()
        scaler2 = RobustScaler()

        # Fit and transform the time series
        norm_firefish_altitudes = scaler1.fit_transform(
            np.array(firefish_altitudes).reshape(-1, 1)
        ).flatten()
        norm_cam_dists = scaler2.fit_transform(
            np.array(cam_dists).reshape(-1, 1)
        ).flatten()

        # Plot the normalized data
        fig, ax1 = plt.subplots()
        ax1.plot(firefish_timestamps, norm_firefish_altitudes, color="tab:red")
        ax1.plot(cam_timestamps, norm_cam_dists, color="tab:blue")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(firefish_timestamps, firefish_depths, color="tab:green")

        if time_range is not None:
            ax1.set_xlim([time_range[0], time_range[1]])
            ax2.set_xlim([time_range[0], time_range[1]])
        elif show_overlap_range:
            ax1.set_xlim([min(cam_timestamps), max(cam_timestamps)])
            ax2.set_xlim([min(cam_timestamps), max(cam_timestamps)])

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        return fig

    def get_offset_of_first_photo_from_firefish_start(self, cams):
        """Determine the offset between the first photo and the first FireFish measurement"""
        cams_with_datetime = [
            cam for cam in cams if cam.has_coords_datetime()
        ]  # TO FIX
        first_camera_time = cams_with_datetime[0].datetime
        first_firefish_time = self.data.iloc[0]["unixtime"]
        return get_time_diff_in_secs(first_firefish_time, first_camera_time)

    def filter_by_depth_range(self, min_depth, max_depth):
        """Filter the DataFrame for values within a certain range."""
        filtered_data = self.data[
            (self.data["depth"] > min_depth) & (self.data["depth"] < max_depth)
        ]

        # Identify fragments by detecting gaps (threshold 60 secs)
        filtered_data = filtered_data.copy()
        filtered_data["fragment_id"] = (filtered_data["unixtime"].diff() > 60).cumsum()

        # Calculate the duration of each fragment and find the longest
        fragment_durations = filtered_data.groupby("fragment_id")["unixtime"].apply(
            lambda x: x.max() - x.min()
        )
        longest_fragment_id = fragment_durations.idxmax()
        print(longest_fragment_id)

        # Filter only the longest fragment
        return filtered_data[filtered_data["fragment_id"] == longest_fragment_id].drop(
            columns="fragment_id"
        )

    def get_last_firefish_time(self):
        """Get the last time in the FireFish data"""
        return int(self.data.iloc[-1]["unixtime"])

    def determine_camera_time_offset_based_on_lowest_depth_regression_error(
        self,
        cams,
        target_depth=None,
        time_range=None,
        time_step=1,
        depth_and_outlier_threshold=settings.FIREFISH_DEPTH_ALTITUDE_OUTLIER_THRESHOLD,
        min_num_cam_matches=settings.FIREFISH_MIN_NUM_CAM_MATCHES,
    ):
        """Determine the time offset for each camera by finding the depth"""
        # Filter the data for outliers and the depth range of interest(e.g. 35-45 m)
        self.remove_outlier_altitudes(depth_and_outlier_threshold)
        if target_depth:
            filtered_data = self.filter_by_depth_range(
                target_depth - depth_and_outlier_threshold,
                target_depth + depth_and_outlier_threshold,
            )
        else:
            filtered_data = self.data

        # Determine starting and ending offset
        if not time_range:
            cams_with_datetime = [
                cam for cam in cams if cam.has_coords_datetime()
            ]  # TO FIX
            first_camera_time = cams_with_datetime[0].datetime
            first_firefish_time = filtered_data.iloc[0]["unixtime"]
            starting_offset = get_time_diff_in_secs(
                first_firefish_time, first_camera_time
            )
            cameras_time_delta = cams.get_time_delta_between_first_and_last_photo()
            last_offset = len(filtered_data) + starting_offset
            time_range = [starting_offset - cameras_time_delta, last_offset]

        # Calculate error statistics for offset range
        error_stats = []

        def get_mae_from_depth_regression(offset):
            # Build arrays for regression without mutating cams
            points, depths = self.get_camera_coords_and_depths_from_firefish(
                cams, offset
            )
            if len(points) < min_num_cam_matches:
                return None
            try:
                res = measurements.fit_depth_regression(points, depths)
            except Exception:
                return None
            if np.isnan(res.mae) or np.isinf(res.mae):
                return None
            return {
                "offset": offset,
                "mae": res.mae,
                "num_matches": len(points),
            }

        error_stats = Parallel(n_jobs=-1)(
            delayed(get_mae_from_depth_regression)(offset)
            for offset in tqdm(
                range(time_range[0], time_range[1], time_step),
                desc="Processing Offsets",
            )
        )

        # Remove None results (failed offsets)
        error_stats = [stat for stat in error_stats if stat is not None]

        if not error_stats:
            logger.error("No valid offsets found for MAE calculation.")
            return None, None

        # Find the lowest MAE value
        offsets = [stat["offset"] for stat in error_stats]
        maes = [stat["mae"] for stat in error_stats]
        lowest_mae = min(maes)
        lowest_mae_offset = offsets[maes.index(lowest_mae)]
        logger.info(
            "Lowest MAE: {0} at offset: {1}".format(lowest_mae, lowest_mae_offset)
        )

        # Plot the matches
        fig = self.__plot_matches(offsets, maes, maes)
        return lowest_mae_offset, fig

    def get_camera_depths_from_firefish(self, cams, offset):
        """
        Build a mapping of cam_id -> depth (meters) from FireFish data at a given offset.
        """
        # Get the cameras with datetime and find matches, but do not mutate cameras here
        cams_with_datetime = [cam for cam in cams if cam.has_coords_datetime()]
        cam_id_to_sensor_depth_m = {}
        for cam in cams_with_datetime:
            cam_time_adjusted = get_unix_time(cam.datetime) + offset
            match_firefish = self.data[self.data["unixtime"] == cam_time_adjusted]
            if not match_firefish.empty:
                cam_id_to_sensor_depth_m[cam.cam_id] = float(
                    match_firefish.iloc[0]["depth"]
                )
        return cam_id_to_sensor_depth_m

    def get_camera_coords_and_depths_from_firefish(self, cams, offset):
        """
        Build numpy arrays of (points, depths) for cameras that have FireFish depth
        at the given offset. Does not mutate cameras.
        """
        mapping = self.get_camera_depths_from_firefish(cams, offset)
        points = []
        depths = []
        for cam_id, depth in mapping.items():
            cam = cams.data.get(cam_id)
            if cam is None or cam.coords is None:
                continue
            points.append(cam.coords)
            depths.append(depth)
        return np.array(points), np.array(depths, dtype=float)

    def determine_up_vector(
        self,
        cams,
        target_depth,
        pcd,
        distance_scale_factor=1.0,
        camdepths_filepath=None,
        pdf_output_filepath=None,
        depth_and_outlier_threshold=settings.FIREFISH_DEPTH_ALTITUDE_OUTLIER_THRESHOLD,
        offset=None,
    ):
        """
        Workflow method that will determine offset, then up vector and then outputs
        visualizations for manual review.

        Distance scale factor is used only for camera distances and plot visualizations.
        It is not used for the up vector determination.
        """
        # Create PDF output file
        if not pdf_output_filepath:
            pdf_output_filepath = "{0}_upvector.pdf".format(pcd.name)
        pdf = PdfPages(pdf_output_filepath)

        # Get datetime stamp from cameras (used for time matching with Firefish data)
        if not camdepths_filepath:
            camdepths_filepath = "{0}_camdepths.csv".format(pcd.name)
        if os.path.exists(camdepths_filepath):
            cams.load_camera_attributes(camdepths_filepath)
        else:
            cams.get_datetime_originals()
            cams.get_cam_dists(pcd, 15, scale_factor=distance_scale_factor)
            cams.save_camera_attributes(camdepths_filepath)

        # Determine camera time offset unless provided manually
        if offset is None:
            offset, fig = (
                self.determine_camera_time_offset_based_on_lowest_depth_regression_error(
                    cams,
                    target_depth=target_depth,
                    depth_and_outlier_threshold=depth_and_outlier_threshold,
                )
            )
            print(
                f"Determined camera time offset for target depth {target_depth}m: {offset}s"
            )
            pdf.savefig(fig)
        else:
            # Remove outliers (as otherwise done in determine_camera_time_offset)
            self.remove_outlier_altitudes(depth_and_outlier_threshold)
            print(f"Using manually provided time offset: {offset}s")

        # Map FireFish depths to cameras for the chosen offset, then apply once
        # and determine the up vector using the applied depths
        cams.reset_depth_sensor_m()
        cam_id_to_sensor_depth_m = self.get_camera_depths_from_firefish(cams, offset)
        cams.set_depth_sensor_m(cam_id_to_sensor_depth_m)
        up_vector, depth_offset, depth_per_unit, *_ = (
            cams.get_up_vector_from_camera_depths()
        )
        # Re-save camera attributes to file (to include determined depth_sensor_m values)
        cams.save_camera_attributes(camdepths_filepath)

        # Visualize altitude matches
        fig = self.plot_cams_vs_firefish(cams, offset)
        pdf.savefig(fig)
        fig = self.plot_cams_vs_firefish(cams, offset, show_overlap_range=True)
        pdf.savefig(fig)

        pdf.close()
        return up_vector, depth_offset, depth_per_unit

    @staticmethod
    def __plot_matches(x_values, series1, series2):
        """Plot the matches between cameras and FireFish data"""
        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.plot(x_values, series1, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = "tab:red"
        ax2.plot(x_values, series2, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        return fig


def get_unix_time(unknown_datetime):
    """Convert a datetime value to a Unix timestamp (seconds since epoch).

    Args:
        unknown_datetime (str or numeric): A datetime value. If a string, it is
            parsed as UTC using settings.CAM_DATETIME_FORMAT; otherwise it is
            assumed to already be a Unix timestamp and returned unchanged.

    Returns:
        float: The Unix timestamp in seconds.
    """
    if isinstance(unknown_datetime, str):
        return (
            datetime.strptime(unknown_datetime, settings.CAM_DATETIME_FORMAT)
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    return unknown_datetime


def get_time_diff_in_secs(datetime1, datetime2):
    """Calculates the time difference in seconds between two datetime values.

    Args:
        datetime1 (str or numeric): The first datetime value. If a string, it should
            conform to settings.CAM_DATETIME_FORMAT.
        datetime2 (str or numeric): The second datetime value. If a string, it should
            conform to settings.CAM_DATETIME_FORMAT.

    Returns:
        int: The difference between datetime1 and datetime2 in seconds.
    """
    datetime1_unix = get_unix_time(datetime1)
    datetime2_unix = get_unix_time(datetime2)
    return int(datetime1_unix - datetime2_unix)
