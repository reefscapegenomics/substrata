# Standard Library
import logging
import os

# Third-Party Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages

# Local Modules
from substrata import settings, visualizations

logger = logging.getLogger(__name__)


class FireFish:
    """Class that holds FireFish sensor"""

    def __init__(self, data_filepath):
        self.data = pd.read_csv(
            data_filepath,
            skiprows=2,
            usecols=range(len(settings.FIREFISH_DEFAULT_COLS)),
        )
        self.data.columns = settings.FIREFISH_DEFAULT_COLS

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
        plt.show()
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

    def determine_camera_time_offset(
        self,
        cams,
        target_depth=None,
        time_range=None,
        time_step=1,
        depth_and_outlier_threshold=3,
    ):
        """Determine the time offset for each camera by finding the depth"""
        # Filter the data for a particular depth range (e.g. 35-45 m)
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

        # for offset in tqdm(range(time_range[0], time_range[1], time_step), desc="Evaluating errors for offsets..."):
        #     model_coef, mse, rmse, mae, r2, cam_depth_residuals = self.get_up_vector_from_camera_depths(cams, offset)
        #     error_stats.append({'offset': offset, 'mae': mae})

        def safe_get_mae(offset, n_treshold=50):
            try:
                result = self.get_up_vector_from_camera_depths(cams, offset)
                mae = result[4]
                num_matches = result[7]
                # If mae is nan, inf, or result is not valid, skip
                if np.isnan(mae) or np.isinf(mae) or num_matches < n_treshold:
                    return None
                return {"offset": offset, "mae": mae}
            except ValueError as e:
                logger.warning(f"Skipping offset {offset} due to error: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error at offset {offset}: {e}")
                return None

        error_stats = Parallel(n_jobs=-1)(
            delayed(safe_get_mae)(offset)
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

    def get_up_vector_from_camera_depths(self, cams, offset):
        """Compute the up vector by doing a least-squared regression between the
        3D points and camera depths to find the vector that is most strongly
        associated with the camera depth measurements. Also returns depth offset
        and the number of matching cameras/depths used in the fit.
        """

        # Gather camera 3D positions (points) and corresponding depth measurements
        cams_with_datetime = [cam for cam in cams if cam.has_coords_datetime()]
        cam_points = []
        cam_depths = []
        cam_ids = []

        for cam in cams_with_datetime:
            cam_time_adjusted = get_unix_time(cam.datetime) + offset
            match = self.data[self.data["unixtime"] == cam_time_adjusted]
            if not match.empty:
                cam_ids.append(cam.cam_id)
                cam_points.append(cam.coords)
                cam_depths.append(match.iloc[0]["depth"])
                # Optionally store that depth back into the camera object
                cam.depth = match.iloc[0]["depth"]

        num_matches = len(cam_points)

        if num_matches < 2:
            # Not enough points to fit a regression
            raise ValueError(
                f"Not enough matching cameras/depths for regression (found {num_matches})"
            )

        points = np.array(cam_points)
        depths = np.array(cam_depths)

        # 1) Fit a linear regression model:  depth ≈ intercept + coef·(x,y,z)
        model = LinearRegression()
        model.fit(points, depths)
        coef = model.coef_  # shape (3,)
        depth_offset = model.intercept_

        # 2) Evaluate sign so that stepping along 'coef' decreases depth:
        centroid = np.mean(points, axis=0)  # single 3D point
        depth_centroid = model.predict([centroid])[0]

        # Take a small step in the direction of 'coef' and predict depth
        step_size = 1.0
        p_step = centroid + step_size * coef
        depth_step = model.predict([p_step])[0]

        # If stepping along coef yields a *larger* depth, flip it
        if depth_step > depth_centroid:
            coef = -coef

        # 3) Store predicted depths and residuals from the (original) linear model
        depths_predicted = model.predict(points)
        depths_residuals = depths - depths_predicted

        mse = mean_squared_error(depths, depths_predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(depths, depths_predicted)
        r2 = r2_score(depths, depths_predicted)

        for idx, cam_id in enumerate(cam_ids):
            cams.data[cam_id].depth_pred = depths_predicted[idx]
            cams.data[cam_id].depth_residual = depths_residuals[idx]

        # Build a dict of residuals if you need them downstream
        cam_depth_residuals = dict(zip(cam_ids, depths_residuals))

        # 4) Return the *flipped-if-needed* up vector and error metrics, plus number of matches
        return coef, depth_offset, mse, rmse, mae, r2, cam_depth_residuals, num_matches

    def determine_up_vector(
        self,
        cams,
        target_depth,
        pcd,
        camdepths_filepath=None,
        pdf_output_filepath=None,
        depth_and_outlier_threshold=3,
    ):
        """
        Workflow method that will determine offset, then up vector and then outputs
        visualizations for manual review.
        """
        # Check if pcd has undergone a transformation (scaling)
        if len(pcd.transforms) == 0:
            logger.warning(
                "The pointcloud has not undergone any transformation and is perhaps not yet scaled (needed for determine_up_vector)"
            )

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
            cams.get_cam_dists(pcd, 15)
            cams.save_camera_attributes(camdepths_filepath)

        # Determine camera time offset
        offset, fig = self.determine_camera_time_offset(
            cams,
            target_depth=target_depth,
            depth_and_outlier_threshold=depth_and_outlier_threshold,
        )
        pdf.savefig(fig)

        # Determine up vector
        (
            up_vector,
            depth_offset,
            mse,
            rmse,
            mae,
            r2,
            cam_depth_residuals,
            num_matches,
        ) = self.get_up_vector_from_camera_depths(cams, offset)
        cams.save_camera_attributes(camdepths_filepath)

        # Visualize altitude matches
        fig = self.plot_cams_vs_firefish(cams, offset)
        pdf.savefig(fig)
        fig = self.plot_cams_vs_firefish(cams, offset, show_overlap_range=True)
        pdf.savefig(fig)

        # Apply transformations to pointcloud
        pcd.apply_orientation_transforms(1, up_vector, depth_offset)

        # # Convert cameras to annotations for visualization
        # cams_reload = cameras.Cameras(cams.cams_meta_filepath, cams.cams_xml_filepath)
        # fig = visualizations.show_cam_residuals(cams_reload, cam_depth_residuals, pcd.world_transform)
        # pdf.savefig(fig)

        # Visualize oriented pointcloud
        fig = visualizations.plot(pcd, width=30, height=8, title=pcd.name)
        pdf.savefig(fig)

        # Output text summary
        text = [
            "Timepoint: {0}\nUp vector: {1}\nDepth offset: {2}\n\nWorld Transform:\n{3}\n".format(
                pcd.name, up_vector, depth_offset, pcd.world_transform
            )
        ]
        fig = visualizations.plot_text("\n".join(text))
        pdf.savefig(fig)
        print(text)

        pdf.close()
        return up_vector, depth_offset, pcd.world_transform

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
        plt.show()
        return fig


def get_unix_time(unknown_datetime):
    """Converts an unknown datetime representation to a Unix timestamp.

    Args:
        unknown_datetime (str or numeric): The datetime to convert. If a string, it
            should match the format specified in settings.CAM_DATETIME_FORMAT. Otherwise,
            it is assumed to already be a Unix timestamp.

    Returns:
        float: The Unix timestamp corresponding to the given datetime.
    """
    if isinstance(unknown_datetime, str):
        return datetime.strptime(
            unknown_datetime, settings.CAM_DATETIME_FORMAT
        ).timestamp()
    else:
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
