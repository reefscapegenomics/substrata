# Standard Library
import os
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from ssl import SSLSocket
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

# NOTE: ``torch`` and ``sam2`` are heavy/optional and are imported lazily (inside
# the functions that need them, via ``_require_sam2``) so that importing this
# module — and therefore ``substrata`` as a whole — does not require them. This is
# what lets ``segmentation`` be added to the package's star-imports.

# Local Modules

# Only the point-cloud segmentation API is star-exported into the flat
# ``substrata.*`` namespace (this module is in ``__init__``'s star-import list).
# The SAM2 image helpers below (including a ``Mask`` class that would otherwise
# collide with ``cameras.Mask``) stay accessible via ``substrata.segmentation.*``
# but are deliberately kept out of ``__all__``.
__all__ = [
    "Segmentation",
    "sample_query_points",
    "segment_point_cloud",
    "recolor_ply_file",
]


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
    sam2_checkpoint="/Users/pbongaerts/Github/sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
):
    """
    Initialize SAM2 segmentation options

    from https://github.com/facebookresearch/sam2
    """
    import torch  # lazy: heavy/optional, not required to import the package

    build_sam2, SAM2ImagePredictor = _require_sam2()

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


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
        # user didn't pass a predictor → build on demand
        sam_predictor = get_sam2_predictor(checkpoint=None)

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
    downscale_interpolation=None,
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

    if downscale_interpolation is None:
        downscale_interpolation = cv2.INTER_AREA

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
    query,
    target_cam,
    max_dim=800,
    downscale_interpolation=None,
    use_gpu=False,
    save_path=None,
    show_plot=True,
    query_crop=None,
    target_crop=None,
):
    """
    Visualize SIFT matches between a query frame/camera and target camera side by side.

    Args:
        query: Frame object or Camera object for the query image
        target_cam: Camera object for the target image
        max_dim: Maximum dimension for resizing images
        downscale_interpolation: Interpolation method for resizing
        use_gpu: Whether to use GPU acceleration
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot
        query_crop: Optional array [x, y, width, height] defining crop area for query image
        target_crop: Optional array [x, y, width, height] defining crop area for target image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if downscale_interpolation is None:
        downscale_interpolation = cv2.INTER_AREA

    def load_and_resize_gray_from_array(img_array, max_dim, crop_area=None):
        h, w = img_array.shape[:2]

        # Apply crop if specified
        if crop_area is not None:
            x, y, crop_w, crop_h = crop_area
            # Ensure crop coordinates are within image bounds
            x = max(0, int(x))
            y = max(0, int(y))
            crop_w = min(int(crop_w), w - x)
            crop_h = min(int(crop_h), h - y)

            if crop_w > 0 and crop_h > 0:
                img_array = img_array[y : y + crop_h, x : x + crop_w]
            else:
                print(f"Warning: Invalid crop area {crop_area}, using full image")

        h, w = img_array.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            img_array = cv2.resize(
                img_array,
                (int(w * scale), int(h * scale)),
                interpolation=downscale_interpolation,
            )
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return img_gray, img_rgb

    def load_and_resize_gray_from_file(filepath, max_dim, crop_area=None):
        img = cv2.imread(filepath)
        if img is None:
            return None, None

        h, w = img.shape[:2]

        # Apply crop if specified
        if crop_area is not None:
            x, y, crop_w, crop_h = crop_area
            # Ensure crop coordinates are within image bounds
            x = max(0, int(x))
            y = max(0, int(y))
            crop_w = min(int(crop_w), w - x)
            crop_h = min(int(crop_h), h - y)

            if crop_w > 0 and crop_h > 0:
                img = img[y : y + crop_h, x : x + crop_w]
            else:
                print(f"Warning: Invalid crop area {crop_area}, using full image")

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

    # --- Handle query input: Frame or Camera ---
    # Try to get image array and label info
    if hasattr(query, "image_array") and query.image_array is not None:
        # Frame object or Camera with loaded image
        query_img_gray, query_img_rgb = load_and_resize_gray_from_array(
            query.image_array, max_dim, query_crop
        )
        # Try to get frame_number and timestamp_seconds if present
        query_frame_number = getattr(query, "frame_number", None)
        query_timestamp = getattr(query, "timestamp_seconds", None)
        query_filepath = getattr(query, "filepath", None)
        query_cam_id = getattr(query, "cam_id", None)
        query_label = None
        if query_frame_number is not None and query_timestamp is not None:
            query_label = f"Frame {query_frame_number} at {query_timestamp:.2f}s"
        elif query_cam_id is not None:
            query_label = f"Camera {query_cam_id}"
        else:
            query_label = "Query"
    elif hasattr(query, "filepath"):
        # Camera object with only filepath
        query_img_gray, query_img_rgb = load_and_resize_gray_from_file(
            query.filepath, max_dim, query_crop
        )
        query_filepath = query.filepath
        query_cam_id = getattr(query, "cam_id", None)
        query_label = f"Camera {query_cam_id}" if query_cam_id is not None else "Query"
        query_frame_number = None
        query_timestamp = None
    else:
        print(
            "Query input must be a Frame or Camera object with image_array or filepath."
        )
        return

    if query_img_gray is None:
        print(
            f"Could not load image for query: {getattr(query, 'filepath', 'unknown')}"
        )
        return

    # --- Target image ---
    target_img_gray, target_img_rgb = load_and_resize_gray_from_file(
        target_cam.filepath, max_dim, target_crop
    )

    if target_img_gray is None:
        print(f"Could not load image for target camera: {target_cam.filepath}")
        return

    # Initialize SIFT
    if use_gpu:
        try:
            sift = cv2.cuda.SIFT_create()
            gpu_available = True
        except Exception:
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
        print(f"No SIFT features found in query image: {query_filepath}")
        return

    # Detect SIFT features for target image
    if gpu_available:
        gpu_target_img = cv2.cuda.GpuMat()
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

    # Print match info
    print(
        f"Found {len(good_matches)} good matches between {query_label} and camera {target_cam.cam_id}"
    )

    # Create visualization with proper line drawing
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Create combined image
    h1, w1 = query_img_rgb.shape[:2]
    h2, w2 = target_img_rgb.shape[:2]
    max_h = max(h1, h2)

    combined_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = query_img_rgb
    combined_img[:h2, w1 : w1 + w2] = target_img_rgb

    # Display combined image
    ax.imshow(combined_img)
    ax.set_title(
        f"SIFT Feature Matching: {query_label} ↔ Target Camera {target_cam.cam_id}\n"
        f"Matches: {len(good_matches)}/{len(matches)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    # Draw matching lines
    if len(good_matches) > 0:
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        for i in range(len(src_pts)):
            x1, y1 = src_pts[i][0]
            x2, y2 = dst_pts[i][0]
            x2 += w1
            ax.plot([x1, x2], [y1, y2], "b-", linewidth=2, alpha=0.7)

    # Add labels for the two images
    ax.text(
        w1 // 2,
        -20,
        query_label,
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
    if query_frame_number is not None and query_timestamp is not None:
        ax.text(
            10,
            max_h + 20,
            f"Query: Frame {query_frame_number} at {query_timestamp:.2f}s",
            fontsize=8,
            wrap=True,
        )
    elif query_cam_id is not None:
        ax.text(
            10,
            max_h + 20,
            f"Query: Camera {query_cam_id} ({query_filepath})",
            fontsize=8,
            wrap=True,
        )
    else:
        ax.text(
            10,
            max_h + 20,
            f"Query: {query_filepath}",
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


def estimate_macro_camera_pose_from_gopro(
    gopro_camera,
    macro_camera,
    point_cloud,
    max_dim=800,
    downscale_interpolation=None,
    use_gpu=False,
    min_matches=10,
    reprojection_threshold=2.0,
    confidence=0.99,
    query_crop=None,
    target_crop=None,
    save_visualization=None,
    show_plot=False,
):
    """
    Estimate macro camera pose using SIFT matches with a GoPro camera through PnP solving.

    This method matches SIFT features between a GoPro camera (with known pose) and a macro camera
    (with unknown pose), then uses the 3D-2D correspondences to solve for the macro camera's
    position and orientation using PnP (Perspective-n-Point) algorithm.

    Args:
        gopro_camera: Camera object with known coords and transform (GoPro frame)
        macro_camera: Camera object with unknown pose (macro camera) - must have filepath
        point_cloud: PointCloud object for 3D point correspondences
        max_dim: Maximum dimension for resizing images during SIFT matching
        downscale_interpolation: Interpolation method for resizing
        use_gpu: Whether to use GPU acceleration for SIFT
        min_matches: Minimum number of good matches required for PnP solving
        reprojection_threshold: Threshold for RANSAC reprojection error in pixels
        confidence: Confidence level for RANSAC
        query_crop: Optional array [x, y, width, height] for cropping GoPro image
        target_crop: Optional array [x, y, width, height] for cropping macro image
        save_visualization: Optional path to save SIFT matching visualization
        show_plot: Whether to display the SIFT matching visualization

    Returns:
        dict: Results containing:
            - success (bool): Whether pose estimation was successful
            - coords (np.ndarray): Estimated camera coordinates [x, y, z]
            - transform (np.ndarray): Estimated 4x4 camera transform matrix
            - rvec (np.ndarray): Rotation vector from PnP
            - tvec (np.ndarray): Translation vector from PnP
            - inliers (int): Number of inlier matches used
            - reprojection_error (float): Mean reprojection error
            - match_stats (dict): SIFT matching statistics
    """
    import cv2
    import numpy as np
    from substrata.logging import logger

    if downscale_interpolation is None:
        downscale_interpolation = cv2.INTER_AREA

    # Validate inputs
    if not hasattr(gopro_camera, "coords") or gopro_camera.coords is None:
        raise ValueError("GoPro camera must have known coordinates")
    if (
        not hasattr(gopro_camera, "camera_transform")
        or gopro_camera.camera_transform is None
    ):
        raise ValueError("GoPro camera must have known transform")
    if not hasattr(macro_camera, "filepath") or macro_camera.filepath is None:
        raise ValueError("Macro camera must have a valid filepath")

    logger.info(f"Estimating macro camera pose using GoPro {gopro_camera.cam_id}")

    # Step 1: Perform SIFT matching between GoPro and macro camera
    match_results = visualize_sift_matches(
        query=gopro_camera,
        target_cam=macro_camera,
        max_dim=max_dim,
        downscale_interpolation=downscale_interpolation,
        use_gpu=use_gpu,
        save_path=save_visualization,
        show_plot=show_plot,
        query_crop=query_crop,
        target_crop=target_crop,
    )

    if match_results is None:
        return {"success": False, "error": "SIFT matching failed", "match_stats": {}}

    # Step 2: Get detailed SIFT matching results for 3D-2D correspondences
    sift_results = _perform_detailed_sift_matching(
        gopro_camera,
        macro_camera,
        point_cloud,
        max_dim,
        downscale_interpolation,
        use_gpu,
        query_crop,
        target_crop,
    )

    if sift_results is None or len(sift_results["good_matches"]) < min_matches:
        return {
            "success": False,
            "error": f"Insufficient matches: {len(sift_results['good_matches']) if sift_results else 0} < {min_matches}",
            "match_stats": match_results,
        }

    # Step 3: Extract 3D-2D correspondences
    object_points = sift_results["object_points"]  # 3D points in world coordinates
    image_points = sift_results["image_points"]  # 2D points in macro camera image

    logger.info(f"Using {len(object_points)} 3D-2D correspondences for PnP solving")

    # Step 4: Get macro camera intrinsic parameters
    if not hasattr(macro_camera, "sensor") or macro_camera.sensor is None:
        raise ValueError("Macro camera must have sensor calibration parameters")

    sensor = macro_camera.sensor
    camera_matrix = np.array(
        [[sensor.fx, 0, sensor.cx], [0, sensor.fy, sensor.cy], [0, 0, 1]],
        dtype=np.float32,
    )

    dist_coeffs = np.array(
        [sensor.k1, sensor.k2, sensor.p1, sensor.p2, sensor.k3], dtype=np.float32
    )

    # Step 5: Solve PnP problem
    try:
        # Validate input data
        logger.info(f"Object points shape: {object_points.shape}")
        logger.info(f"Image points shape: {image_points.shape}")
        logger.info(f"Camera matrix:\n{camera_matrix}")
        logger.info(f"Distortion coefficients: {dist_coeffs}")

        # Check for valid data
        if len(object_points) < 4:
            return {
                "success": False,
                "error": f"Insufficient object points: {len(object_points)} < 4",
                "match_stats": match_results,
            }

        if len(image_points) < 4:
            return {
                "success": False,
                "error": f"Insufficient image points: {len(image_points)} < 4",
                "match_stats": match_results,
            }

        # Check for NaN or infinite values
        if np.any(np.isnan(object_points)) or np.any(np.isinf(object_points)):
            return {
                "success": False,
                "error": "Object points contain NaN or infinite values",
                "match_stats": match_results,
            }

        if np.any(np.isnan(image_points)) or np.any(np.isinf(image_points)):
            return {
                "success": False,
                "error": "Image points contain NaN or infinite values",
                "match_stats": match_results,
            }

        # Try different PnP methods
        pnp_methods = [
            (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
            (cv2.SOLVEPNP_EPNP, "EPNP"),
            (cv2.SOLVEPNP_P3P, "P3P"),
            (cv2.SOLVEPNP_DLS, "DLS"),
            (cv2.SOLVEPNP_UPNP, "UPNP"),
        ]

        success = False
        rvec, tvec, inliers = None, None, None
        used_method = None

        for method_flag, method_name in pnp_methods:
            try:
                logger.info(f"Trying PnP method: {method_name}")
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    reprojectionError=reprojection_threshold,
                    confidence=confidence,
                    flags=method_flag,
                )

                if success:
                    used_method = method_name
                    logger.info(f"PnP succeeded with method: {method_name}")
                    break
                else:
                    logger.info(f"PnP failed with method: {method_name}")

            except Exception as e:
                logger.warning(
                    f"PnP method {method_name} failed with exception: {str(e)}"
                )
                continue

        if not success:
            # Try without RANSAC as fallback
            logger.info("Trying PnP without RANSAC as fallback")
            try:
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=object_points,
                    imagePoints=image_points,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                inliers = np.arange(len(object_points))  # All points are inliers
                used_method = "ITERATIVE (no RANSAC)"
                logger.info("PnP succeeded without RANSAC")
            except Exception as e:
                logger.error(f"PnP without RANSAC also failed: {str(e)}")
                return {
                    "success": False,
                    "error": f"All PnP methods failed. Last error: {str(e)}",
                    "match_stats": match_results,
                }

        if not success:
            return {
                "success": False,
                "error": "PnP solving failed with all methods",
                "match_stats": match_results,
            }

        # Step 6: Convert rotation vector to rotation matrix and create transform
        rmat, _ = cv2.Rodrigues(rvec)

        # Create 4x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = rmat
        transform[:3, 3] = tvec.flatten()

        # Step 7: Calculate reprojection error
        projected_points, _ = cv2.projectPoints(
            object_points[inliers], rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        actual_points = image_points[inliers]

        reprojection_error = np.mean(
            np.linalg.norm(projected_points - actual_points, axis=1)
        )

        # Step 8: Extract camera coordinates (position)
        coords = tvec.flatten()

        logger.info(f"Successfully estimated macro camera pose:")
        logger.info(f"  Method used: {used_method}")
        logger.info(
            f"  Coordinates: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]"
        )
        logger.info(f"  Inliers: {len(inliers)}/{len(object_points)}")
        logger.info(f"  Reprojection error: {reprojection_error:.2f} pixels")

        return {
            "success": True,
            "coords": coords,
            "transform": transform,
            "rvec": rvec.flatten(),
            "tvec": tvec.flatten(),
            "inliers": len(inliers),
            "reprojection_error": reprojection_error,
            "match_stats": match_results,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
        }

    except Exception as e:
        logger.error(f"PnP solving failed with error: {str(e)}")
        return {
            "success": False,
            "error": f"PnP solving failed: {str(e)}",
            "match_stats": match_results,
        }


def _perform_detailed_sift_matching(
    gopro_camera,
    macro_camera,
    point_cloud,
    max_dim,
    downscale_interpolation,
    use_gpu,
    query_crop,
    target_crop,
):
    """
    Perform detailed SIFT matching to get 3D-2D correspondences for PnP solving.

    Returns:
        dict: Contains 'good_matches', 'object_points', 'image_points', and keypoint info
    """
    import cv2
    import numpy as np

    def load_and_resize_gray_from_array(img_array, max_dim, crop_area=None):
        h, w = img_array.shape[:2]

        # Apply crop if specified
        if crop_area is not None:
            x, y, crop_w, crop_h = crop_area
            x = max(0, int(x))
            y = max(0, int(y))
            crop_w = min(int(crop_w), w - x)
            crop_h = min(int(crop_h), h - y)

            if crop_w > 0 and crop_h > 0:
                img_array = img_array[y : y + crop_h, x : x + crop_w]

        h, w = img_array.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            img_array = cv2.resize(
                img_array,
                (int(w * scale), int(h * scale)),
                interpolation=downscale_interpolation,
            )
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY), scale

    def load_and_resize_gray_from_file(filepath, max_dim, crop_area=None):
        img = cv2.imread(filepath)
        if img is None:
            return None, None

        h, w = img.shape[:2]

        # Apply crop if specified
        if crop_area is not None:
            x, y, crop_w, crop_h = crop_area
            x = max(0, int(x))
            y = max(0, int(y))
            crop_w = min(int(crop_w), w - x)
            crop_h = min(int(crop_h), h - y)

            if crop_w > 0 and crop_h > 0:
                img = img[y : y + crop_h, x : x + crop_w]

        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=downscale_interpolation,
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scale

    # Load and resize images
    if hasattr(gopro_camera, "image_array") and gopro_camera.image_array is not None:
        gopro_gray, gopro_scale = load_and_resize_gray_from_array(
            gopro_camera.image_array, max_dim, query_crop
        )
    elif hasattr(gopro_camera, "filepath"):
        gopro_gray, gopro_scale = load_and_resize_gray_from_file(
            gopro_camera.filepath, max_dim, query_crop
        )
    else:
        return None

    macro_gray, macro_scale = load_and_resize_gray_from_file(
        macro_camera.filepath, max_dim, target_crop
    )

    if gopro_gray is None or macro_gray is None:
        return None

    # Initialize SIFT
    if use_gpu:
        try:
            sift = cv2.cuda.SIFT_create()
            gpu_available = True
        except Exception:
            gpu_available = False
            sift = cv2.SIFT_create()
    else:
        sift = cv2.SIFT_create()
        gpu_available = False

    # Detect SIFT features
    if gpu_available:
        gpu_gopro = cv2.cuda_GpuMat()
        gpu_gopro.upload(gopro_gray)
        kp_gopro, des_gopro = sift.detectAndCompute(gpu_gopro, None)
        des_gopro = des_gopro.download()

        gpu_macro = cv2.cuda.GpuMat()
        gpu_macro.upload(macro_gray)
        kp_macro, des_macro = sift.detectAndCompute(gpu_macro, None)
        des_macro = des_macro.download()
    else:
        kp_gopro, des_gopro = sift.detectAndCompute(gopro_gray, None)
        kp_macro, des_macro = sift.detectAndCompute(macro_gray, None)

    if des_gopro is None or des_macro is None:
        return None

    # Match features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_gopro, des_macro, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None

    # Get matched keypoints
    gopro_pts = np.float32([kp_gopro[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 2
    )
    macro_pts = np.float32([kp_macro[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 2
    )

    # Scale keypoints back to original image coordinates
    gopro_pts /= gopro_scale
    macro_pts /= macro_scale

    # Apply crop offset if specified
    if query_crop is not None:
        gopro_pts[:, 0] += query_crop[0]
        gopro_pts[:, 1] += query_crop[1]
    if target_crop is not None:
        macro_pts[:, 0] += target_crop[0]
        macro_pts[:, 1] += target_crop[1]

    # Get 3D points corresponding to GoPro keypoints
    object_points = []
    image_points = []

    for i, (gopro_pt, macro_pt) in enumerate(zip(gopro_pts, macro_pts)):
        # Project GoPro keypoint to 3D world coordinates
        try:
            world_coords, _, _ = gopro_camera.pixel_to_point(
                int(gopro_pt[0]), int(gopro_pt[1]), point_cloud
            )
            if world_coords is not None:
                object_points.append(world_coords)
                image_points.append(macro_pt)
        except Exception:
            continue

    if len(object_points) < 4:
        return None

    return {
        "good_matches": good_matches,
        "object_points": np.array(object_points, dtype=np.float32),
        "image_points": np.array(image_points, dtype=np.float32),
        "gopro_keypoints": kp_gopro,
        "macro_keypoints": kp_macro,
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


# ===========================================================================
# Point-cloud segmentation via image-match classification
# ===========================================================================
#
# Classify/segment a whole point cloud by reusing the trained per-annotation
# crop classifier: sample a grid of query points across the cloud, project each
# to its best camera photo, cut a 224 px patch, classify it, then propagate the
# labels to every cloud point via nearest neighbour. Unlike the SAM2 code above
# this is a 3-D point-cloud operation; its only heavy dependency (fastai) is
# imported lazily, so this module stays importable without torch/sam2.

# PLY storage type -> numpy dtype (for the streaming full-size recolor).
_PLY_NP_DTYPE = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _build_label_colors(
    codebook: List[str], overrides: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> Dict[str, Tuple[int, int, int]]:
    """Build a ``{label: (r, g, b)}`` 0-255 map (tab20 defaults + manual overrides).

    Mirrors the ``label_colors``/``tab20`` convention used across ortho and
    annotations rendering. ``overrides`` (e.g. ``{"CS": (255, 0, 0)}``) win over
    the auto colours; labels not in ``overrides`` get a stable tab20 colour.
    """
    import matplotlib.pyplot as plt  # already a top-level import; kept explicit

    cmap = plt.get_cmap("tab20")
    colors = {
        lbl: tuple(int(255 * c) for c in cmap(i % 20)[:3])
        for i, lbl in enumerate(codebook)
    }
    if overrides:
        colors.update({k: tuple(v) for k, v in overrides.items() if k in codebook})
    return colors


@dataclass
class Segmentation:
    """A point-cloud segmentation result — standalone, not attached to a cloud.

    The source of truth is the small set of classified *query points*
    (``query_coords`` + ``query_labels``); per-point labels for a full cloud are
    a derived cache produced on demand by :meth:`propagate`. Persisted compactly
    to ``.npz`` so it costs zero memory when not loaded.

    Attributes:
        query_coords: ``(Nq, 3)`` float32 world-frame coordinates of classified
            query points (kept only where a label was assigned).
        query_labels: ``(Nq,)`` object array of category label strings.
        query_conf: ``(Nq,)`` float classifier confidences.
        codebook: ordered unique labels; the integer code of a label is its index.
        label_colors: ``{label: (r, g, b)}`` 0-255 display colours.
        point_codes: optional ``(N,)`` int cache of per-point label codes for a
            specific cloud (``-1`` = unlabeled). Derived; not persisted by default.
    """

    query_coords: np.ndarray
    query_labels: np.ndarray
    query_conf: np.ndarray
    codebook: List[str]
    label_colors: Dict[str, Tuple[int, int, int]]
    point_codes: Optional[np.ndarray] = None
    _tree: Any = field(default=None, repr=False, compare=False)

    @property
    def n_queries(self) -> int:
        return len(self.query_coords)

    @classmethod
    def from_query_labels(
        cls,
        query_coords: np.ndarray,
        labels: np.ndarray,
        conf: np.ndarray,
        label_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> "Segmentation":
        """Build a Segmentation, keeping only successfully-classified queries."""
        matched = np.array([bool(l) for l in labels])
        if not matched.any():
            raise ValueError(
                "No query points were classified. Nothing to segment — most "
                "likely the camera photo filepaths do not resolve on disk "
                "(check cams.set_filepath_replace(old, new))."
            )
        qc = np.asarray(query_coords)[matched].astype(np.float32)
        ql = np.asarray(labels, dtype=object)[matched]
        qcf = np.asarray(conf, dtype=float)[matched]
        codebook = sorted(set(ql.tolist()))
        colors = _build_label_colors(codebook, label_colors)
        return cls(qc, ql, qcf, codebook, colors)

    def _query_tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.query_coords)
        return self._tree

    def propagate(
        self, points: np.ndarray, max_radius: Optional[float] = None
    ) -> np.ndarray:
        """Assign each point the label of its nearest classified query.

        Args:
            points: ``(N, 3)`` world-frame coordinates.
            max_radius: if given, points farther than this from any query are
                left unlabeled (``-1``); default ``None`` labels every point
                (gap-free).

        Returns:
            ``(N,)`` int16 array of label codes (``-1`` = unlabeled).
        """
        if self.n_queries == 0:
            raise ValueError("Segmentation has no query points to propagate.")
        code_of_query = np.array(
            [self.codebook.index(l) for l in self.query_labels], dtype=np.int16
        )
        dist, idx = self._query_tree().query(points, k=1, workers=-1)
        codes = code_of_query[idx].astype(np.int16)
        if max_radius is not None:
            codes[dist > max_radius] = -1
        return codes

    def recolor(
        self,
        orig_colors: np.ndarray,
        point_codes: Optional[np.ndarray] = None,
        value_floor: float = 0.3,
        unlabeled: str = "gray",
    ) -> np.ndarray:
        """Blend each point's category colour with its original luminance.

        The category base colour is modulated by the point's own luminance
        (floored at ``value_floor`` so the tint stays visible in dark areas), so
        surface relief survives the recolouring. Returns ``(N, 3)`` 0-1 RGB.

        Args:
            orig_colors: ``(N, 3)`` original colours (0-1 or 0-255).
            point_codes: per-point label codes; defaults to ``self.point_codes``.
            value_floor: minimum luminance scaling for labeled points.
            unlabeled: ``"gray"`` (dim grayscale) or ``"keep"`` (original colour).
        """
        if point_codes is None:
            point_codes = self.point_codes
        if point_codes is None:
            raise ValueError("No point_codes given; call propagate() first.")
        oc = np.asarray(orig_colors, dtype=float)
        if oc.size and oc.max() > 1.0:
            oc = oc / 255.0
        lum = oc @ np.array([0.299, 0.587, 0.114])
        v = value_floor + (1.0 - value_floor) * lum
        palette = (
            np.array([self.label_colors[l] for l in self.codebook], dtype=float)
            / 255.0
        )
        out = np.zeros_like(oc)
        labeled = point_codes >= 0
        out[labeled] = palette[point_codes[labeled]] * v[labeled, None]
        if unlabeled == "gray":
            out[~labeled] = (lum[~labeled] * 0.6)[:, None]
        else:  # "keep": original colour
            out[~labeled] = oc[~labeled]
        return np.clip(out, 0.0, 1.0)

    def summary(self, point_codes: Optional[np.ndarray] = None) -> Dict[str, int]:
        """Return {label: count}. Uses ``point_codes`` if given, else query labels."""
        if point_codes is not None:
            uniq, cnt = np.unique(point_codes[point_codes >= 0], return_counts=True)
            return {self.codebook[c]: int(n) for c, n in zip(uniq, cnt)}
        uniq, cnt = np.unique(self.query_labels, return_counts=True)
        return {str(u): int(n) for u, n in zip(uniq, cnt)}

    def save(self, path: str, include_point_codes: bool = False) -> str:
        """Persist to a compact ``.npz`` (query points are tiny)."""
        payload = dict(
            query_coords=self.query_coords,
            query_labels=np.array(self.query_labels, dtype=object),
            query_conf=self.query_conf,
            codebook=np.array(self.codebook, dtype=object),
            color_values=np.array(
                [self.label_colors[l] for l in self.codebook], dtype=np.int16
            ),
        )
        if include_point_codes and self.point_codes is not None:
            payload["point_codes"] = self.point_codes
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str) -> "Segmentation":
        d = np.load(path, allow_pickle=True)
        codebook = d["codebook"].tolist()
        colors = {
            lbl: tuple(int(c) for c in d["color_values"][i])
            for i, lbl in enumerate(codebook)
        }
        return cls(
            d["query_coords"], d["query_labels"], d["query_conf"], codebook, colors,
            point_codes=d["point_codes"] if "point_codes" in d.files else None,
        )


def sample_query_points(
    pcd, cell_size: float, method: str = "voxel", rep: str = "highest"
) -> np.ndarray:
    """Sample one representative query point per cell across the cloud.

    Args:
        pcd: a :class:`~substrata.pointclouds.PointCloud` (uses ``pcd.points`` in
            the world frame, and ``pcd.o3d_pcd`` for voxel downsampling).
        cell_size: cell / voxel side in metres (== target spatial resolution).
        method: ``"voxel"`` (3-D voxel downsample — covers vertical faces and
            overhangs, the default) or ``"xy_grid"`` (one point per top-down XY
            cell — faster, but leaves gaps at oblique viewing angles).
        rep: for ``xy_grid``, ``"highest"`` (max-z surface point, best for camera
            matching) or ``"centroid"``.

    Returns:
        ``(Nq, 3)`` array of query coordinates (world frame).
    """
    pts = np.asarray(pcd.points)
    if method == "voxel":
        down = pcd.o3d_pcd.voxel_down_sample(cell_size)
        return np.asarray(down.points)
    if method != "xy_grid":
        raise ValueError(f"unknown sampling method: {method!r}")

    ix = np.floor(pts[:, 0] / cell_size).astype(np.int64)
    iy = np.floor(pts[:, 1] / cell_size).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    key = ix * (iy.max() + 1) + iy  # unique per XY cell

    if rep == "highest":
        order = np.lexsort((pts[:, 2], key))
        k_sorted = key[order]
        last = np.ones(len(k_sorted), dtype=bool)
        last[:-1] = k_sorted[1:] != k_sorted[:-1]
        return pts[order[last]]

    uniq, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, pts)
    counts = np.bincount(inv, minlength=len(uniq))[:, None]
    return sums / counts


def _match_points_to_cameras(
    query_coords: np.ndarray,
    cams,
    world_transform: Optional[np.ndarray] = None,
    pcd=None,
    occlusion: bool = True,
    intercept_radius: Optional[float] = None,
    discard_threshold: Optional[float] = None,
    verbose: bool = True,
):
    """Find each query point's best camera (fast, vectorized, occlusion-aware).

    For every camera, project *all* query points at once
    (:meth:`~substrata.cameras.Camera.project_points`) and keep the running
    best camera per query by the ``relevance`` metric — O(cameras) numpy passes,
    no per-query Python loop. When ``occlusion`` is on, run the reprojection
    ray-march (``pcd.get_intercept``) once per query against its *chosen*
    camera and drop queries whose line of sight is blocked. This is the
    production replacement for ``Annotations.get_first_image_matches``, which
    looped every camera per query and ran the ray-march per candidate (~15 min
    for 40 k queries).

    Returns:
        ``(best_cam, best_x, best_y, best_depth, cam_list)`` where ``best_cam``
        is an ``(Nq,)`` int index into ``cam_list`` (``-1`` = unmatched),
        ``best_x``/``best_y`` are ``(Nq,)`` pixel coordinates, and ``best_depth``
        is the ``(Nq,)`` camera-to-point distance (world metres) of the chosen
        camera — used to report the physical crop footprint.
    """
    from substrata import settings

    if intercept_radius is None:
        intercept_radius = settings.DEFAULT_INTERCEPT_SEARCH_RADIUS
    if discard_threshold is None:
        discard_threshold = settings.DEFAULT_REPROJECTION_THRESHOLD_DISCARD

    query_coords = np.asarray(query_coords, dtype=float)
    n = len(query_coords)

    # Camera projection uses each point's *original* (pre-world-transform)
    # coords; occlusion uses the world-frame coords + world pcd. Derive the
    # original-frame coords once.
    if world_transform is not None and not np.allclose(world_transform, np.eye(4)):
        inv = np.linalg.inv(np.asarray(world_transform, dtype=float))
        homog = np.hstack([query_coords, np.ones((n, 1))])
        orig = (homog @ inv.T)[:, :3]
    else:
        orig = query_coords

    cam_list = list(cams)
    best_rel = np.full(n, np.inf)
    best_cam = np.full(n, -1, dtype=int)
    best_x = np.zeros(n)
    best_y = np.zeros(n)
    best_depth = np.zeros(n)

    iterator = (
        tqdm(cam_list, desc="Projecting cameras", unit="cam") if verbose else cam_list
    )
    for ci, cam in enumerate(iterator):
        if getattr(cam, "enabled", True) is False or cam.sensor is None:
            continue
        x, y, depth, rel, in_view = cam.project_points(orig, use_orig_coords=True)
        better = in_view & (depth > 0) & (rel < best_rel)
        if not better.any():
            continue
        best_rel = np.where(better, rel, best_rel)
        best_cam = np.where(better, ci, best_cam)
        best_x = np.where(better, x, best_x)
        best_y = np.where(better, y, best_y)
        best_depth = np.where(better, depth, best_depth)

    if occlusion and pcd is not None:
        pcd.build_kd_tree()
        matched_idx = np.nonzero(best_cam >= 0)[0]
        it = (
            tqdm(matched_idx, desc="Occlusion check", unit="pt")
            if verbose else matched_idx
        )
        for qi in it:
            cam = cam_list[best_cam[qi]]
            try:
                vector = cam.pixel_to_ray(float(best_x[qi]), float(best_y[qi]))
                intercept = pcd.get_intercept(
                    cam.coords, search_radius=intercept_radius,
                    vector=vector, always_return=True,
                )
            except Exception:  # noqa: BLE001 - drop the match on any ray failure
                best_cam[qi] = -1
                continue
            if intercept is None:
                best_cam[qi] = -1
                continue
            err = float(np.linalg.norm(query_coords[qi] - intercept.coords))
            if err > discard_threshold:
                best_cam[qi] = -1  # line of sight blocked by nearer surface

    return best_cam, best_x, best_y, best_depth, cam_list


def _classify_crops(
    query_coords: np.ndarray,
    best_cam: np.ndarray,
    best_x: np.ndarray,
    best_y: np.ndarray,
    cam_list,
    learn,
    crop_size: int = 224,
    batch_size: int = 64,
    flush_batch: int = 16384,
    verbose: bool = True,
):
    """Decode each photo once, cut its crops, and classify them in flushed batches.

    Groups matched queries by source photo so every full-res JPEG is decoded a
    single time (the dominant cost) and slices its patches from the in-memory
    array. Crops are classified and **freed** in bounded batches of
    ``flush_batch`` rather than accumulating every patch for the whole cloud
    before a single forward pass — that unbounded list was ~150 KB × millions of
    queries and blew the 125 GB budget at fine ``cell_size``. Peak crop memory is
    now ~``flush_batch`` patches (+ one decoded photo), independent of cloud size.
    Returns ``(labels, conf)`` aligned with ``query_coords``
    (unmatched/failed → ``""`` / ``nan``).
    """
    from substrata import classification

    n = len(query_coords)
    labels = np.full(n, "", dtype=object)
    conf = np.full(n, np.nan)

    by_file: Dict[str, list] = defaultdict(list)
    for qi in np.nonzero(best_cam >= 0)[0]:
        cam = cam_list[best_cam[qi]]
        by_file[cam.filepath].append((int(qi), best_x[qi], best_y[qi]))
    if not by_file:
        return labels, conf

    example_fp = next(iter(by_file))
    if verbose:
        print(
            f"  {len(by_file)} unique photos; example: {example_fp} "
            f"(exists={os.path.isfile(example_fp) if example_fp else 'n/a'})"
        )

    vocab = getattr(getattr(learn, "dls", None), "vocab", None)
    crops: list = []
    crop_qids: list = []
    n_open_fail = n_oob = n_classified = 0
    half = crop_size // 2

    def _flush():
        """Classify the buffered crops, write results, and free the buffer."""
        nonlocal crops, crop_qids, n_classified
        if not crops:
            return
        dl = learn.dls.test_dl(crops, bs=batch_size)
        # Silence fastai's per-call progress bar (this runs many times); no-op
        # for any learner without ``no_bar`` (e.g. test doubles).
        cm = learn.no_bar() if hasattr(learn, "no_bar") else nullcontext()
        with cm:
            probs, _ = learn.get_preds(dl=dl)
        for qi, p in zip(crop_qids, probs.tolist()):
            res = classification._result_from_probs(p, vocab)
            labels[qi] = res["label"]
            conf[qi] = res["confidence"]
        n_classified += len(crops)
        crops, crop_qids = [], []

    it = (
        tqdm(by_file.items(), desc="Decoding photos", unit="photo")
        if verbose else by_file.items()
    )
    for fp, group in it:
        try:
            arr = np.asarray(Image.open(fp).convert("RGB"))
        except (OSError, ValueError):
            n_open_fail += 1
            continue
        h, w = arr.shape[:2]
        for qi, x, y in group:
            left, top = int(x - half), int(y - half)
            lft, tp = max(0, left), max(0, top)
            rgt, bot = min(w, left + crop_size), min(h, top + crop_size)
            if lft >= rgt or tp >= bot:
                n_oob += 1
                continue
            crops.append(Image.fromarray(arr[tp:bot, lft:rgt]))
            crop_qids.append(qi)
            if len(crops) >= flush_batch:
                _flush()  # bound peak memory: classify + free mid-photo
    _flush()  # remainder

    if verbose:
        if n_open_fail:
            print(
                f"  WARNING: {n_open_fail}/{len(by_file)} photos failed to open "
                "— check camera filepaths / cams.set_filepath_replace(...)"
            )
        print(f"  classified {n_classified} crops ({n_oob} dropped as out-of-bounds)")
    return labels, conf


def _report_crop_footprint(
    best_cam, best_depth, cam_list, crop_size, cell_size, depth_scale=1.0
):
    """Print the physical size a classifier crop covers on the substrate.

    A crop is a fixed ``crop_size``-px cut from the full-res photo, so on the
    substrate it spans ``crop_size * depth / fx`` (``fx`` = focal length in px,
    ``depth`` = camera-to-point distance). ``best_depth`` is in the *original*
    (pre-world-transform) frame, so ``depth_scale`` (world metres per original
    unit) converts it to metres. This is the real resolution floor: sampling
    much finer than the footprint just re-classifies heavily overlapping crops.
    Uses the depths already computed during matching.
    """
    matched = best_cam >= 0
    if not matched.any():
        return
    fxs = [
        c.sensor.fx for c in cam_list
        if getattr(c, "sensor", None) is not None
        and getattr(c.sensor, "fx", None)
    ]
    if not fxs:
        return
    fx = float(np.median(fxs))
    depths = np.abs(best_depth[matched].astype(float)) * depth_scale
    fp = crop_size * depths / fx  # metres
    lo, med, hi = np.percentile(fp, [5, 50, 95])
    print(
        f"[footprint] crop ~= {med * 100:.1f} cm on substrate "
        f"(5-95 pct {lo * 100:.1f}-{hi * 100:.1f} cm; "
        f"crop {crop_size}px, fx {fx:.0f}px, depth {np.median(depths):.2f} m)"
    )
    if cell_size < 0.5 * med:
        print(
            f"  NOTE: cell_size {cell_size * 100:.1f} cm is well below the crop "
            f"footprint — adjacent queries classify overlapping patches "
            f"(oversampling; little extra detail)."
        )


def segment_point_cloud(
    pcd,
    cams,
    classifier,
    *,
    world_transform: Optional[np.ndarray] = None,
    cell_size: Optional[float] = None,
    sampling: str = "voxel",
    occlusion: bool = True,
    label_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    crop_size: Optional[int] = None,
    batch_size: int = 64,
    flush_batch: int = 16384,
    verbose: bool = True,
) -> Segmentation:
    """Segment a point cloud by classifying camera patches at sampled query points.

    Samples query points across ``pcd``, matches each to its best camera photo
    (fast vectorized, occlusion-aware), classifies the 224 px patches with the
    trained crop classifier, and returns a :class:`Segmentation`. Propagate its
    labels to a cloud with :meth:`Segmentation.propagate`, recolour with
    :meth:`Segmentation.recolor`, or recolour a full-size PLY with
    :func:`recolor_ply_file`.

    Args:
        pcd: decimated :class:`~substrata.pointclouds.PointCloud` (world frame).
        cams: project :class:`~substrata.cameras.Cameras`.
        classifier: a loaded fastai learner or a path to a ``.pkl`` (loaded via
            :func:`substrata.classification.get_image_classifier`).
        world_transform: project world transform; defaults to
            ``pcd.world_transform``.
        cell_size: query spacing / voxel size in metres
            (default ``settings.DEFAULT_SEG_CELL_SIZE``).
        sampling: ``"voxel"`` (default) or ``"xy_grid"`` (see
            :func:`sample_query_points`).
        occlusion: run reprojection occlusion filtering (drops points seen
            through a nearer surface).
        label_colors: optional ``{label: (r, g, b)}`` overrides (0-255).
        crop_size: patch size in px (default ``settings.TRAIN_CROP_SIZE``).
        batch_size: inference batch size.
        flush_batch: max crops held in memory before a forward pass frees them
            (bounds peak RAM regardless of ``cell_size`` / cloud size).

    Returns:
        A :class:`Segmentation` (classified query points + codebook + colours).
    """
    from substrata import classification, settings

    if cell_size is None:
        cell_size = settings.DEFAULT_SEG_CELL_SIZE
    if crop_size is None:
        crop_size = settings.TRAIN_CROP_SIZE
    if world_transform is None:
        world_transform = getattr(pcd, "world_transform", None)

    learn = (
        classification.get_image_classifier(classifier)
        if isinstance(classifier, str)
        else classifier
    )

    qc = sample_query_points(pcd, cell_size, method=sampling)
    if verbose:
        print(
            f"[sample] {len(qc):,} query points "
            f"({sampling}, {cell_size * 100:.0f} cm)"
        )

    best_cam, best_x, best_y, best_depth, cam_list = _match_points_to_cameras(
        qc, cams, world_transform=world_transform,
        pcd=(pcd if occlusion else None), occlusion=occlusion, verbose=verbose,
    )
    if verbose:
        print(f"[match] {int((best_cam >= 0).sum()):,}/{len(qc):,} queries matched")
        # best_depth is in the original frame; convert to world metres via the
        # world_transform's linear scale (cbrt of its 3x3 determinant).
        depth_scale = 1.0
        if world_transform is not None:
            lin = np.asarray(world_transform, dtype=float)[:3, :3]
            depth_scale = float(np.cbrt(abs(np.linalg.det(lin)))) or 1.0
        _report_crop_footprint(
            best_cam, best_depth, cam_list, crop_size, cell_size, depth_scale
        )

    labels, conf = _classify_crops(
        qc, best_cam, best_x, best_y, cam_list, learn,
        crop_size=crop_size, batch_size=batch_size, flush_batch=flush_batch,
        verbose=verbose,
    )
    if verbose:
        print(f"[classify] {int((labels != '').sum()):,} classified")

    return Segmentation.from_query_labels(qc, labels, conf, label_colors=label_colors)


def recolor_ply_file(
    input_ply: str,
    output_ply: str,
    segmentation: Segmentation,
    *,
    world_transform: Optional[np.ndarray] = None,
    max_radius: Optional[float] = None,
    value_floor: float = 0.3,
    unlabeled: str = "keep",
    chunk_bytes: int = 64 << 20,
    verbose: bool = True,
) -> str:
    """Stream a full-size PLY and rewrite its colours from a segmentation.

    Reads ``input_ply`` in chunks (never loading it fully), propagates the
    segmentation's labels to each chunk's points, blends the category colour
    with the point's original luminance, and writes the recoloured cloud to
    ``output_ply`` — type-preserving (uchar 0-255 or float 0-1). Reuses the PLY
    header/streaming helpers from :mod:`substrata.pointclouds`.

    Args:
        input_ply: path to the full-size PLY (raw, original frame).
        output_ply: destination path.
        segmentation: a :class:`Segmentation` (query points are in world frame).
        world_transform: transform mapping the PLY's raw coords into the world
            frame the segmentation lives in (default: the pcd world transform you
            used when segmenting). If ``None``/identity, coords are used as-is.
        max_radius: propagation cap (``None`` = label every point).
        value_floor / unlabeled: passed to :meth:`Segmentation.recolor`
            (``unlabeled="keep"`` retains original colour for unlabeled points).
        chunk_bytes: streaming chunk size.
    """
    from substrata import pointclouds as pc

    wt = None
    if world_transform is not None and not np.allclose(world_transform, np.eye(4)):
        wt = np.asarray(world_transform, dtype=float)

    with open(input_ply, "rb") as fin:
        fmt, endian, n_vertices, vprops, rec_size, _ = pc._parse_ply_header(fin)
        layout = pc._ply_red_green_blue_layout(vprops)
        if layout is None:
            raise ValueError("Input PLY has no red/green/blue properties to recolour.")
        red_type = layout[3]

        dtype = np.dtype([(name, endian + _PLY_NP_DTYPE[t]) for t, name in vprops])
        recs_per_chunk = max(1, chunk_bytes // rec_size)

        with open(output_ply, "wb") as fout:
            fout.write(_seg_output_header(fmt, vprops, n_vertices))
            remaining = n_vertices
            bar = (
                tqdm(total=n_vertices, desc="Recolouring PLY", unit="pt")
                if verbose else None
            )
            while remaining > 0:
                k = min(recs_per_chunk, remaining)
                buf = fin.read(k * rec_size)
                if len(buf) < k * rec_size:
                    raise ValueError("Unexpected EOF while reading PLY vertex data.")
                recs = np.frombuffer(buf, dtype=dtype, count=k).copy()

                xyz = np.column_stack(
                    [recs["x"], recs["y"], recs["z"]]
                ).astype(float)
                if wt is not None:
                    xyz = (np.hstack([xyz, np.ones((k, 1))]) @ wt.T)[:, :3]

                codes = segmentation.propagate(xyz, max_radius=max_radius)
                orig_rgb = np.column_stack(
                    [recs["red"], recs["green"], recs["blue"]]
                ).astype(float)
                new_rgb = segmentation.recolor(
                    orig_rgb, codes, value_floor=value_floor, unlabeled=unlabeled,
                )  # 0-1

                if red_type in ("uchar", "uint8"):
                    vals = np.round(new_rgb * 255.0).astype(recs["red"].dtype)
                else:  # float/double stored 0-1
                    vals = new_rgb.astype(recs["red"].dtype)
                recs["red"] = vals[:, 0]
                recs["green"] = vals[:, 1]
                recs["blue"] = vals[:, 2]

                fout.write(recs.tobytes())
                remaining -= k
                if bar is not None:
                    bar.update(k)
            if bar is not None:
                bar.close()
    return output_ply


def _seg_output_header(fmt: str, vprops, vertex_count: int) -> bytes:
    """Clean minimal PLY header for the recoloured output (all props preserved)."""
    lines = [b"ply\n", f"format {fmt} 1.0\n".encode("ascii")]
    lines.append(b"comment recoloured by substrata segmentation\n")
    lines.append(f"element vertex {vertex_count}\n".encode("ascii"))
    for t, name in vprops:
        lines.append(f"property {t} {name}\n".encode("ascii"))
    lines.append(b"end_header\n")
    return b"".join(lines)
