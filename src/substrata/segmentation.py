# Standard Library
import os
import math
from ssl import SSLSocket

# Third-Party Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

# Local Modules


class Mask:
    """
    Mask object holding the output of SAM2 prediction
    """

    def __init__(self, mask, score, logits):
        self.vals = mask
        self.score = score
        self.logits = logits
        self.area_in_px = cv2.countNonZero(self.vals)
        self.area_in_cm2 = None


def get_sam2_predictor(
    checkpoint, model_cfg="sam2_hiera_l.yaml", device: str | None = None
):
    build_sam2, SAM2ImagePredictor = _require_sam2()
    try:
        import torch  # lazy

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        dev = device or "cpu"
    sam2 = build_sam2(config=model_cfg, checkpoint=checkpoint, device=dev)
    return SAM2ImagePredictor(sam2)


def show_mask(mask, ax, random_color=False, borders=True):
    """
    Show the mask on the image
    from https://github.com/facebookresearch/sam2
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    Show the annotated points on the image
    from https://github.com/facebookresearch/sam2
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker=".",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker=".",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    """
    Show the annotated box on the image
    from https://github.com/facebookresearch/sam2
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    """
    Show the image masks
    from https://github.com/facebookresearch/sam2
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def get_sam2_masks(
    image_filepath, pixel_x, pixel_y, sam_predictor=None, visualize=False
):
    """
    Get a SAM2 prediction of a single point and return Mask instances.
    """
    if sam_predictor is None:
        # user didn’t pass a predictor → build on demand
        sam_predictor = get_sam2_predictor(checkpoint=None)  # fill your defaults

    image = Image.open(image_filepath)
    image = np.array(image.convert("RGB"))

    sam_predictor.set_image(image)
    masks, scores, logits = sam_predictor.predict(
        point_coords=np.array([[pixel_x, pixel_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    # Sort masks based on scores (descending order)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # Create Mask instances
    mask_objects = [Mask(masks[i], scores[i], logits[i]) for i in range(len(masks))]

    if visualize:
        show_masks(
            image,
            masks,
            scores,
            point_coords=np.array([[pixel_x, pixel_y]]),
            input_labels=np.array([1]),
            borders=True,
        )

    return mask_objects


def sift_match_batched(
    query_frame,
    target_cams,
    max_dim=500,
    target_max_dim=500,
    downscale_interpolation=cv2.INTER_AREA,
    n_jobs=-1,
    batch_size=10,  # Process in batches to avoid memory issues
):
    """
    Batched parallel processing for very large datasets.
    Processes target cameras in batches to manage memory usage.
    Returns a dictionary with camera IDs as keys and number of matches as values.
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    def load_and_resize_gray(filepath, max_dim):
        img = cv2.imread(filepath)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=downscale_interpolation,
            )
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    # Load and resize the query frame
    query_img_gray = cv2.cvtColor(query_frame.image_array, cv2.COLOR_BGR2GRAY)
    h, w = query_img_gray.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        query_img_gray = cv2.resize(
            query_img_gray,
            (int(w * scale), int(h * scale)),
            interpolation=downscale_interpolation,
        )

    # Initialize SIFT
    sift = cv2.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(query_img_gray, None)

    if des_query is None or len(kp_query) == 0:
        print(
            f"No SIFT features found in query frame at {query_frame.timestamp_seconds:.2f}s"
        )
        return

    def process_single_target(target_cam):
        """Process a single target camera"""
        # Initialize SIFT and FLANN inside each worker to avoid pickling issues
        sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        target_img_gray = load_and_resize_gray(target_cam.filepath, target_max_dim)
        if target_img_gray is None:
            return {
                "cam_id": target_cam.cam_id,
                "filepath": target_cam.filepath,
                "matches": 0,
                "message": f"Could not load image for target camera: {target_cam.filepath}",
            }

        kp_target, des_target = sift.detectAndCompute(target_img_gray, None)
        if des_target is None or len(kp_target) == 0:
            return {
                "cam_id": target_cam.cam_id,
                "filepath": target_cam.filepath,
                "matches": 0,
                "message": f"No SIFT features found in target camera image: {target_cam.filepath}",
            }

        # Match descriptors using KNN
        matches = flann.knnMatch(des_query, des_target, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return {
            "cam_id": target_cam.cam_id,
            "filepath": target_cam.filepath,
            "matches": len(good_matches),
            "message": (
                f"Camera {target_cam.cam_id}: {len(good_matches)} good matches with query frame at {query_frame.timestamp_seconds:.2f}s "
                f"(query img size: {query_img_gray.shape[::-1]}, target img size: {target_img_gray.shape[::-1]})"
            ),
        }

    # Convert target_cams dict to list for batching
    target_cam_list = list(target_cams.data.values())
    total_cameras = len(target_cam_list)
    num_batches = math.ceil(total_cameras / batch_size)

    all_results = []

    print(f"Processing {total_cameras} cameras in {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_cameras)
        batch_cameras = target_cam_list[start_idx:end_idx]

        # Process this batch in parallel
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_target)(target_cam) for target_cam in batch_cameras
        )

        all_results.extend(batch_results)

    # Print all results
    # for result in all_results:
    #    print(result['message'])

    # Create dictionary with camera IDs and number of matches
    matches_dict = {result["cam_id"]: result["matches"] for result in all_results}

    # Find the camera with the highest number of matches
    if all_results:
        best_match = max(all_results, key=lambda x: x["matches"])

        print(f"\n{'='*60}")
        print(f"BEST MATCH:")
        print(f"Camera ID: {best_match['cam_id']}")
        print(f"Filepath: {best_match['filepath']}")
        print(f"Number of matches: {best_match['matches']}")
        print(f"{'='*60}")

        return matches_dict
    else:
        print("No results found.")
        return {}


def visualize_sift_matches(
    query_frame,
    target_cam,
    max_dim=800,
    downscale_interpolation=cv2.INTER_AREA,
    use_gpu=False,
    save_path=None,
    show_plot=True,
):
    """
    Visualize SIFT matches between a query frame and target camera side by side.

    Args:
        query_frame: Frame object for the query image
        target_cam: Camera object for the target image
        max_dim: Maximum dimension for resizing images
        downscale_interpolation: Interpolation method for resizing
        use_gpu: Whether to use GPU acceleration
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def load_and_resize_gray(filepath, max_dim):
        img = cv2.imread(filepath)
        if img is None:
            return None, None
        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=downscale_interpolation,
            )
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_gray, img_rgb

    # Load and resize images
    query_img_gray = cv2.cvtColor(query_frame.image_array, cv2.COLOR_BGR2GRAY)
    query_img_rgb = cv2.cvtColor(query_frame.image_array, cv2.COLOR_BGR2RGB)
    h, w = query_img_gray.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        query_img_gray = cv2.resize(
            query_img_gray,
            (int(w * scale), int(h * scale)),
            interpolation=downscale_interpolation,
        )
        query_img_rgb = cv2.resize(
            query_img_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=downscale_interpolation,
        )

    target_img_gray, target_img_rgb = load_and_resize_gray(target_cam.filepath, max_dim)

    if target_img_gray is None:
        print(f"Could not load image for target camera: {target_cam.filepath}")
        return

    # Initialize SIFT
    if use_gpu:
        try:
            sift = cv2.cuda.SIFT_create()
            gpu_available = True
        except:
            print("GPU SIFT not available, falling back to CPU")
            gpu_available = False
            sift = cv2.SIFT_create()
    else:
        sift = cv2.SIFT_create()
        gpu_available = False

    # Detect SIFT features for query image
    if gpu_available:
        gpu_query_img = cv2.cuda_GpuMat()
        gpu_query_img.upload(query_img_gray)
        kp_query, des_query = sift.detectAndCompute(gpu_query_img, None)
        des_query = des_query.download()
    else:
        kp_query, des_query = sift.detectAndCompute(query_img_gray, None)

    if des_query is None or len(kp_query) == 0:
        print(f"No SIFT features found in query camera image: {query_frame.filepath}")
        return

    # Detect SIFT features for target image
    if gpu_available:
        gpu_target_img = cv2.cuda_GpuMat()
        gpu_target_img.upload(target_img_gray)
        kp_target, des_target = sift.detectAndCompute(gpu_target_img, None)
        des_target = des_target.download()
    else:
        kp_target, des_target = sift.detectAndCompute(target_img_gray, None)

    if des_target is None or len(kp_target) == 0:
        print(f"No SIFT features found in target camera image: {target_cam.filepath}")
        return

    # FLANN matcher setup
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(des_query, des_target, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(
        f"Found {len(good_matches)} good matches between query frame at {query_frame.timestamp_seconds:.2f}s and camera {target_cam.cam_id}"
    )

    # Create visualization with proper line drawing
    # Create a single figure with both images side by side
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Create combined image
    h1, w1 = query_img_rgb.shape[:2]
    h2, w2 = target_img_rgb.shape[:2]
    max_h = max(h1, h2)

    # Create combined image
    combined_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = query_img_rgb
    combined_img[:h2, w1 : w1 + w2] = target_img_rgb

    # Display combined image
    ax.imshow(combined_img)
    ax.set_title(
        f"SIFT Feature Matching: Query Frame at {query_frame.timestamp_seconds:.2f}s ↔ Target Camera {target_cam.cam_id}\n"
        f"Matches: {len(good_matches)}/{len(matches)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    # Draw matching lines
    if len(good_matches) > 0:
        # Get matched keypoints
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Draw lines connecting matched points
        for i in range(len(src_pts)):
            # Get coordinates
            x1, y1 = src_pts[i][0]
            x2, y2 = dst_pts[i][0]
            # Adjust x2 coordinate for target image
            x2 += w1

            # Draw connecting line between matched points
            ax.plot([x1, x2], [y1, y2], "b-", linewidth=2, alpha=0.7)

    # Add labels for the two images
    ax.text(
        w1 // 2,
        -20,
        f"Query Frame at {query_frame.timestamp_seconds:.2f}s",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.text(
        w1 + w2 // 2,
        -20,
        f"Target Camera {target_cam.cam_id}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add filepath information
    ax.text(
        10,
        max_h + 20,
        f"Query: Frame {query_frame.frame_number} at {query_frame.timestamp_seconds:.2f}s",
        fontsize=8,
        wrap=True,
    )
    ax.text(
        w1 + 10, max_h + 20, f"Target: {target_cam.filepath}", fontsize=8, wrap=True
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return {
        "query_keypoints": len(kp_query),
        "target_keypoints": len(kp_target),
        "total_matches": len(matches),
        "good_matches": len(good_matches),
        "match_ratio": len(good_matches) / len(matches) if len(matches) > 0 else 0,
    }


def visualize_camera_matches_heatmap(
    cameras,
    matches_dict,
    query_cam_id=None,
    figsize=(10, 3),
    save_path=None,
    show_plot=True,
    top_n_labels=5,
):
    """
    Create a top-down 2D visualization of camera positions with heatmap coloring based on matches.

    Args:
        cameras: Cameras instance containing all camera objects
        matches_dict: Dictionary with camera IDs as keys and number of matches as values
        query_cam_id: Optional camera ID to highlight as the query camera
        figsize: Figure size as (width, height)
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot
        top_n_labels: Number of top matches to label (0 for no labels, default: 5)

    Returns:
        dict: Statistics about the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize
    import numpy as np

    # Extract camera positions (ignoring z-axis)
    camera_positions = {}
    for cam_id, camera in cameras.data.items():
        if hasattr(camera, "coords") and camera.coords is not None:
            # Use x, y coordinates (ignore z)
            camera_positions[cam_id] = (camera.coords[0], camera.coords[1])

    if not camera_positions:
        print("No camera positions found!")
        return {}

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Extract coordinates for plotting
    x_coords = [pos[0] for pos in camera_positions.values()]
    y_coords = [pos[1] for pos in camera_positions.values()]
    cam_ids = list(camera_positions.keys())

    # Get match values for each camera
    match_values = []
    for cam_id in cam_ids:
        match_values.append(matches_dict.get(cam_id, 0))

    # Create color mapping based on match values
    if max(match_values) > 0:
        norm = Normalize(vmin=0, vmax=max(match_values))
        colors = plt.cm.viridis(norm(match_values))
    else:
        # If no matches, use gray for all cameras
        colors = [(0.7, 0.7, 0.7, 1.0)] * len(cam_ids)

    # Plot cameras as scatter points
    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=match_values,
        cmap="viridis",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1,
    )

    # Add colorbar
    if max(match_values) > 0:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Number of Matches", fontsize=12)

    # Highlight query camera if specified
    if query_cam_id and query_cam_id in camera_positions:
        query_pos = camera_positions[query_cam_id]
        ax.scatter(
            query_pos[0],
            query_pos[1],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label=f"Query Camera: {query_cam_id}",
            zorder=5,
        )

    # Add camera ID labels for top N matches
    if top_n_labels > 0:
        # Create list of (cam_id, matches, index) tuples and sort by matches
        camera_matches = [(cam_ids[i], match_values[i], i) for i in range(len(cam_ids))]
        camera_matches.sort(
            key=lambda x: x[1], reverse=True
        )  # Sort by matches descending

        # Label top N cameras with matches
        for i, (cam_id, matches, idx) in enumerate(camera_matches):
            if i >= top_n_labels or matches == 0:
                break
            ax.annotate(
                f"{cam_id} ({matches})",
                (x_coords[idx], y_coords[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )

    # Set labels and title
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    title = f"Camera Positions Heatmap - Matches Distribution"
    if query_cam_id:
        title += f" (Query: {query_cam_id})"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend if query camera is highlighted
    if query_cam_id and query_cam_id in camera_positions:
        ax.legend(loc="upper right")

    # Set equal aspect ratio and grid
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    # Calculate statistics for return value
    total_cameras = len(cameras.data)
    cameras_with_matches = sum(1 for matches in match_values if matches > 0)
    max_matches = max(match_values) if match_values else 0
    avg_matches = np.mean(match_values) if match_values else 0

    # Return statistics
    return {
        "total_cameras": total_cameras,
        "cameras_with_matches": cameras_with_matches,
        "max_matches": max_matches,
        "avg_matches": avg_matches,
        "match_values": match_values,
        "camera_positions": camera_positions,
    }


def visualize_camera_matches_grid_heatmap(
    cameras,
    matches_dict,
    query_cam_id=None,
    grid_size=1.0,
    figsize=(10, 3),
    save_path=None,
    show_plot=True,
    cmap="viridis",
):
    """
    Create a grid-based heatmap visualization of camera matches.

    Divides the area into grid cells (default 1m x 1m) and shows the maximum number
    of matches for cameras in each cell as a heatmap.

    Args:
        cameras: Cameras instance containing all camera objects
        matches_dict: Dictionary with camera IDs as keys and number of matches as values
        query_cam_id: Optional camera ID to highlight as the query camera
        grid_size: Size of grid cells in meters (default: 1.0)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot
        cmap: Colormap to use for the heatmap

    Returns:
        dict: Statistics about the visualization and grid data
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize
    import numpy as np

    # Extract camera positions (ignoring z-axis)
    camera_positions = {}
    for cam_id, camera in cameras.data.items():
        if hasattr(camera, "coords") and camera.coords is not None:
            # Use x, y coordinates (ignore z)
            camera_positions[cam_id] = (camera.coords[0], camera.coords[1])

    if not camera_positions:
        print("No camera positions found!")
        return {}

    # Extract coordinates for analysis
    x_coords = [pos[0] for pos in camera_positions.values()]
    y_coords = [pos[1] for pos in camera_positions.values()]
    cam_ids = list(camera_positions.keys())

    # Get match values for each camera
    match_values = []
    for cam_id in cam_ids:
        match_values.append(matches_dict.get(cam_id, 0))

    # Calculate grid boundaries
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add some padding to the grid
    padding = grid_size * 0.5
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Create grid
    x_edges = np.arange(x_min, x_max + grid_size, grid_size)
    y_edges = np.arange(y_min, y_max + grid_size, grid_size)

    # Initialize grid with zeros
    grid = np.zeros((len(y_edges) - 1, len(x_edges) - 1))

    # Fill grid with maximum match values for each cell
    for i, cam_id in enumerate(cam_ids):
        x, y = x_coords[i], y_coords[i]
        matches = match_values[i]

        # Find grid cell indices
        x_idx = int((x - x_min) // grid_size)
        y_idx = int((y - y_min) // grid_size)

        # Ensure indices are within bounds
        if 0 <= x_idx < len(x_edges) - 1 and 0 <= y_idx < len(y_edges) - 1:
            # Take the maximum value in each cell
            grid[y_idx, x_idx] = max(grid[y_idx, x_idx], matches)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        grid,
        cmap=cmap,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Max Matches per Grid Cell", fontsize=12)

    # Highlight query camera if specified
    if query_cam_id and query_cam_id in camera_positions:
        query_pos = camera_positions[query_cam_id]
        ax.scatter(
            query_pos[0],
            query_pos[1],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label=f"Query Camera: {query_cam_id}",
            zorder=5,
        )

    # Add camera positions as small dots
    ax.scatter(
        x_coords,
        y_coords,
        c="white",
        s=20,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )

    # Add legend if query camera is highlighted
    if query_cam_id and query_cam_id in camera_positions:
        ax.legend(loc="upper right")

    # Set labels and title
    ax.set_xlabel("X Coordinate (m)", fontsize=12)
    ax.set_ylabel("Y Coordinate (m)", fontsize=12)

    title = f"Camera Matches Grid Heatmap (Grid Size: {grid_size}m)"
    if query_cam_id:
        title += f" (Query: {query_cam_id})"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid lines
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    # Set tick spacing to match grid size
    ax.set_xticks(x_edges[:: max(1, len(x_edges) // 10)])  # Show ~10 ticks
    ax.set_yticks(y_edges[:: max(1, len(y_edges) // 10)])  # Show ~10 ticks

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grid heatmap saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    # Calculate statistics
    total_cameras = len(cameras.data)
    cameras_with_matches = sum(1 for matches in match_values if matches > 0)
    max_matches = max(match_values) if match_values else 0
    avg_matches = np.mean(match_values) if match_values else 0
    max_grid_value = np.max(grid)
    avg_grid_value = np.mean(grid[grid > 0]) if np.any(grid > 0) else 0

    # Return statistics and grid data
    return {
        "total_cameras": total_cameras,
        "cameras_with_matches": cameras_with_matches,
        "max_matches": max_matches,
        "avg_matches": avg_matches,
        "match_values": match_values,
        "camera_positions": camera_positions,
        "grid": grid,
        "grid_size": grid_size,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "max_grid_value": max_grid_value,
        "avg_grid_value": avg_grid_value,
        "grid_shape": grid.shape,
    }


import cv2
import numpy as np
import matplotlib.pyplot as plt


def match_frame_feature(frame, orthomosaic_path, ratio_thresh=0.75):
    """
    Locate a frame in an orthomosaic via SIFT and draw its corners.

    Args:
        frame (Frame): Frame with .image_array (BGR).
        orthomosaic_path (str): Path to orthomosaic image file.
        ratio_thresh (float): Lowe's ratio-test threshold.

    Returns:
        result_img (np.ndarray): Mosaic with match polygon.
        corners (np.ndarray): 4x2 array of projected frame corners.
    """
    # Load and grayscale the mosaic
    mosaic = cv2.imread(orthomosaic_path)
    if mosaic is None:
        raise IOError(f"Cannot load image: {orthomosaic_path}")
    gray_m = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.equalizeHist(gray_m)

    # Grayscale and equalize the frame
    gray_p = cv2.cvtColor(frame.image_array, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.equalizeHist(gray_p)

    # Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp_m, des_m = sift.detectAndCompute(gray_m, None)
    kp_p, des_p = sift.detectAndCompute(gray_p, None)

    # Match using FLANN + ratio test
    index_params = {"algorithm": 1, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_p, des_m, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    if len(good) < 4:
        raise ValueError(f"Not enough matches: {len(good)}")

    # Prepare points for homography
    src_pts = np.float32([kp_p[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Project frame corners onto mosaic
    h, w = gray_p.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)
    pts = np.int32(proj).reshape(-1, 2)

    # Draw matched polygon on mosaic copy
    result_img = mosaic.copy()
    cv2.polylines(result_img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

    # Display in Jupyter
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(
        f"SIFT match: Frame {frame.frame_number} @ " f"{frame.timestamp_seconds:.2f}s"
    )
    plt.show()

    return result_img, pts


def _require_sam2():
    """Import SAM2 only when needed."""
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception as e:
        raise ImportError(
            "SAM2 is not installed. Install it only if you use SAM2 features:\n"
            "  pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'\n"
            "or follow the project’s installation instructions."
        ) from e
    return build_sam2, SAM2ImagePredictor
