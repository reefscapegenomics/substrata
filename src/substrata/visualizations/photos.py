"""Photo figures: images, crops and overlays.

A photo with highlighted pixels (``show_img``), crops outlined with their SAM2
mask contours (``get_crop_img_from_masks``), image matches drawn into a camera
image and annotated videos built from those frames (``create_annotated_video``).
The crop geometry comes from :mod:`substrata.imagecrops` and the matches from
:mod:`substrata.annotations` / :mod:`substrata.cameras`; nothing is measured here.
"""

# Standard Library
import os
import tempfile
import shutil
import subprocess

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Local Modules
from substrata.fileio import imread as _imread
from substrata.logging import logger


def show_img(img_path, highlight_pixels=None):
    """
    Load an image using PIL, optionally add highlighted circles, and show it.

    This function ignores the EXIF orientation (unless otherwise handled by PIL)
    and always displays the image in its raw pixel orientation.

    If the path is missing or the file cannot be read, logs a warning and returns
    without raising (execution continues for the caller).

    Args:
        img_path (str): Path to the image file.
        highlight_pixels (None, list, or np.ndarray):
            Pixel coordinates to highlight. If a 1D array/list, it's interpreted
            as a single point [x, y] with a larger circle; if 2D, each sub-list
            represents a point with a smaller circle.
    """
    if not img_path:
        logger.warning("show_img: no image path given; skipping display.")
        return
    if not os.path.isfile(img_path):
        logger.warning("show_img: file not found: %s", img_path)
        return

    try:
        image = Image.open(img_path).convert("RGB")
    except OSError as e:
        logger.warning("show_img: could not open image %s: %s", img_path, e)
        return

    # Prepare drawing context.
    draw = ImageDraw.Draw(image)

    if highlight_pixels is not None:
        # Ensure highlight_pixels is a NumPy array.
        hp = np.array(highlight_pixels, dtype=int)
        if hp.ndim == 1:
            # Draw a large circle (radius 50) for a single point.
            x, y = hp[0], hp[1]
            # PIL's ellipse uses a bounding box: (left, top, right, bottom)
            draw.ellipse((x - 50, y - 50, x + 50, y + 50), fill=(255, 0, 0))
        elif hp.ndim > 1:
            for pixel in hp:
                x, y = pixel[0], pixel[1]
                draw.ellipse((x - 50, y - 50, x + 50, y + 50), fill=(255, 0, 0))

    # Convert the PIL image to a NumPy array and display it.
    plt.imshow(np.array(image))
    plt.axis("off")
    plt.show()


def _radius_line_endpoint(center_xy, contour_xy, r_px):
    """End point of a radius line drawn from ``center_xy``.

    The line points towards the contour point whose distance from the centre
    is closest to ``r_px``, but always has length ``r_px``.

    Args:
        center_xy: ``(x, y)`` start of the line (the annotation point).
        contour_xy: ``(N, 2)`` contour points.
        r_px: Line length (radius) in pixels.

    Returns:
        ``(x, y)`` end point as a float array, or None if there is no usable
        contour point.
    """
    center = np.asarray(center_xy, dtype=float)
    pts = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    if not len(pts) or not r_px or r_px <= 0:
        return None
    dist = np.linalg.norm(pts - center, axis=1)
    usable = dist > 0
    if not usable.any():
        return None
    pts, dist = pts[usable], dist[usable]
    k = int(np.argmin(np.abs(dist - r_px)))
    return center + (pts[k] - center) / dist[k] * r_px


def get_crop_img_from_masks(  # TODO: needs PIL version
    image_match,
    output_img_w=1000,
    output_img_h=1000,
    pad_ratio=0.0,
    contour_thickness=2,
    annotation_radius=20,
    annotation_color=(0, 255, 0),
    single_mask=False,
    show_text=True,
    show_radius=False,
    show_point=True,
):
    """
    Crop an image based on a list of SAM2 masks, draw each mask's contours in
    red, green, and blue (first mask with double thickness), annotate the
    surface area, and mark the annotation point.

    Args:
        image_match: ImageMatch with ``masks`` and a chosen ``mask``.
        output_img_w: Output width in pixels.
        output_img_h: Output height in pixels.
        pad_ratio: Padding around the mask bounding box (fraction).
        contour_thickness: Contour line thickness in pixels.
        annotation_radius: Radius of the annotation dot in pixels.
        annotation_color: BGR colour of the annotation dot (and radius line).
        single_mask: Only draw the chosen mask.
        show_text: Overlay the surface area of the chosen mask (only when its
            ``area_in_cm2`` is known); includes the radius when
            ``show_radius`` is on.
        show_radius: Draw a line from the annotation point with the length of
            the chosen mask's equivalent-circle radius (``sqrt(area / pi)``),
            pointing towards the mask edge point whose distance from the
            annotation point is closest to that radius.
        show_point: Draw the annotation point (default True). The radius line
            still starts there when this is off.

    Returns:
        BGR image (numpy array).
    """
    # Load image
    img = _imread(image_match.filepath)
    if img is None:
        raise IOError(f"Could not read image: {image_match.filepath}")
    h_img, w_img = img.shape[:2]

    # If single_mask is True, only the chosen mask is processed
    if single_mask:
        image_match_masks = [image_match.mask]
    else:
        image_match_masks = image_match.masks

    # Compute union bounding box from all masks
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
    for m in image_match_masks:
        ys, xs = np.where(m.vals)
        if ys.size and xs.size:
            x_mins.append(xs.min())
            x_maxs.append(xs.max())
            y_mins.append(ys.min())
            y_maxs.append(ys.max())
    if not x_mins:
        raise ValueError("No valid mask regions found.")
    x_min, x_max = int(min(x_mins)), int(max(x_maxs))
    y_min, y_max = int(min(y_mins)), int(max(y_maxs))
    bbox_w, bbox_h = x_max - x_min + 1, y_max - y_min + 1

    # Adjust bbox to desired aspect ratio (expanding only)
    bbox_aspect, desired_aspect = bbox_w / bbox_h, output_img_w / output_img_h
    if bbox_aspect < desired_aspect:
        new_w = int(np.ceil(bbox_h * desired_aspect))
        center = (x_min + x_max) // 2
        x_min_adj = center - new_w // 2
        x_max_adj = x_min_adj + new_w - 1
        y_min_adj, y_max_adj = y_min, y_max
    else:
        new_h = int(np.ceil(bbox_w / desired_aspect))
        center = (y_min + y_max) // 2
        y_min_adj = center - new_h // 2
        y_max_adj = y_min_adj + new_h - 1
        x_min_adj, x_max_adj = x_min, x_max

    # Add padding and clamp to image boundaries
    adj_w = x_max_adj - x_min_adj + 1
    adj_h = y_max_adj - y_min_adj + 1
    pad = int(np.round(min(adj_w, adj_h) * pad_ratio))
    x_min_final = np.clip(x_min_adj - pad, 0, w_img - 1)
    y_min_final = np.clip(y_min_adj - pad, 0, h_img - 1)
    x_max_final = np.clip(x_max_adj + pad, 0, w_img - 1)
    y_max_final = np.clip(y_max_adj + pad, 0, h_img - 1)

    # Crop and resize image
    crop = img[y_min_final : y_max_final + 1, x_min_final : x_max_final + 1]
    crop_resized = cv2.resize(
        crop, (output_img_w, output_img_h), interpolation=cv2.INTER_AREA
    )

    # Colors for contours: red, green, blue
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Process each mask: crop, resize, find contours, and draw them
    for idx, m in enumerate(image_match_masks):
        mask_crop = m.vals[y_min_final : y_max_final + 1, x_min_final : x_max_final + 1]
        mask_resized = cv2.resize(
            mask_crop, (output_img_w, output_img_h), interpolation=cv2.INTER_NEAREST
        )
        mask_bin = ((mask_resized > 0).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Chosen mask gets double thickness
        if m == image_match.mask:
            thick = contour_thickness * 5
        else:
            thick = contour_thickness
        color = colors[idx % len(colors)]
        cv2.drawContours(crop_resized, contours, -1, color, thickness=thick)

    # Overlay surface area (in cm2) of chosen mask (default = first mask)
    area_cm2 = getattr(image_match.mask, "area_in_cm2", None)
    if show_text and area_cm2 is not None:
        overlay_text = f"SA: {area_cm2:.4f} cm2"
        if show_radius:
            overlay_text += f"  r: {np.sqrt(area_cm2 / np.pi):.2f} cm"
        font, font_scale, text_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        text_size, _ = cv2.getTextSize(
            overlay_text, font, font_scale, text_thickness
        )
        cv2.putText(
            crop_resized,
            overlay_text,
            (10, text_size[1] + 10),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    # Map the annotation point to the crop coordinate system
    crop_orig_w = x_max_final - x_min_final + 1
    crop_orig_h = y_max_final - y_min_final + 1
    scale_x = output_img_w / crop_orig_w
    scale_y = output_img_h / crop_orig_h
    ann_x = int(round((image_match.x - x_min_final) * scale_x))
    ann_y = int(round((image_match.y - y_min_final) * scale_y))

    # Equivalent-circle radius of the chosen mask, from its (pixel) area
    if show_radius:
        mask = image_match.mask
        mask_scale = getattr(mask, "scale", 1.0)
        r_px = np.sqrt(mask.area_in_px / np.pi) * mask_scale
        contours, _ = cv2.findContours(
            (np.asarray(mask.vals) > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if contours:
            contour_xy = np.vstack([c.reshape(-1, 2) for c in contours]) * mask_scale
            end = _radius_line_endpoint(
                (image_match.x, image_match.y), contour_xy, r_px
            )
            if end is not None:
                end_x = int(round((end[0] - x_min_final) * scale_x))
                end_y = int(round((end[1] - y_min_final) * scale_y))
                cv2.line(
                    crop_resized,
                    (ann_x, ann_y),
                    (end_x, end_y),
                    annotation_color,
                    thickness=contour_thickness * 2,
                    lineType=cv2.LINE_AA,
                )
    if show_point:
        cv2.circle(
            crop_resized,
            (ann_x, ann_y),
            annotation_radius,
            annotation_color,
            thickness=-1,
        )

    return crop_resized


def draw_image_matches_within_camera(
    image_matches, cam, use_label_column=False, resize_width=None
):
    """
    Open the image, draw an ellipse for each image match, overlay up to the first two masks (if available)
    with smooth contour boundaries (green for the first mask, red for the second mask),
    and add a label with the annotation id to the right of the ellipse.
    Optionally resize the final image to the specified width while preserving the aspect ratio.
    """

    # Open the image and convert to RGB.
    image = Image.open(cam.filepath).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a TrueType font; fall back to the default if it fails.
    # Use a smaller font size for better visibility if the text is not showing up.
    try:
        # Try to load a common cross-platform font; fall back to default if not found
        font = ImageFont.truetype("DejaVuSans.ttf", 100)
    except Exception as e:
        print(f"Error loading DejaVuSans font: {e}")
        font = ImageFont.load_default()

    # Define offsets for the ellipse and text.
    circle_radius = 50
    text_offset_x = circle_radius + 50  # Offset text to the right of the ellipse.
    text_offset_y = -50  # Slightly above the center of the ellipse.

    # If no matches provided, optionally resize and return the base image
    if not image_matches:
        if resize_width is not None:
            orig_width, orig_height = image.size
            new_height = int(orig_height * resize_width / orig_width)
            image = image.resize((resize_width, new_height))
        return image

    # Draw ellipses and labels.
    for match in image_matches:
        x, y = match.x, match.y

        # Draw the ellipse.
        draw.ellipse(
            (
                x - circle_radius,
                y - circle_radius,
                x + circle_radius,
                y + circle_radius,
            ),
            fill=(255, 0, 0),
        )

        # Prepare the label text.
        if use_label_column:
            ann_text = match.annotation.label
        else:
            ann_text = str(match.annotation.id)

        # Draw the label text.
        draw.text(
            (x + text_offset_x, y + text_offset_y),
            ann_text,
            (255, 255, 255),
            font=font,
        )

    # Process masks from match.masks by overlaying smooth contours.
    # Convert the PIL image to a NumPy array.
    np_image = np.array(image)  # (H, W, 3) in RGB

    # Define colors for the first two masks: green and red.
    mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for match in image_matches:
        if match.masks is not None and len(match.masks) > 0:
            for idx in range(min(2, len(match.masks))):
                try:
                    mask_obj = match.masks[idx]
                    # Check format: use .vals if present; otherwise, treat mask_obj as a NumPy array.
                    if hasattr(mask_obj, "vals"):
                        mask_arr = np.array(mask_obj.vals, dtype=np.uint8)
                    else:
                        mask_arr = np.array(mask_obj, dtype=np.uint8)

                    # Ensure the mask is binary.
                    mask_bin = (mask_arr > 0).astype(np.uint8) * 255

                    # Find contours using OpenCV.
                    contours, _ = cv2.findContours(
                        mask_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    # Smooth the contours.
                    smooth_contours = []
                    for cnt in contours:
                        epsilon = 0.01 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        smooth_contours.append(approx)

                    # Draw the smooth contours on the image using the designated color.
                    cv2.drawContours(
                        np_image, smooth_contours, -1, mask_colors[idx], thickness=3
                    )
                except Exception as e:
                    print("Error processing mask for annotation", ann_text, ":", e)

    # Convert the NumPy image (with overlays) back to a PIL image.
    image = Image.fromarray(np_image)

    # Resize the image if a resize_width is provided.
    if resize_width is not None:
        orig_width, orig_height = image.size
        new_height = int(orig_height * resize_width / orig_width)
        image = image.resize((resize_width, new_height))

    return image


def save_image_matches_within_camera(
    image_matches, cam, output_path, use_label_column=False, resize_width=None
):
    """Save image matches within the camera image to a file"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image = draw_image_matches_within_camera(
        image_matches, cam, use_label_column, resize_width
    )
    image.save(os.path.join(output_path, cam.filename))


def create_annotated_video(
    cams,
    annotations,
    output_filename="cams_video.mp4",
    sam_predictor=None,
    pcd=None,
    use_label_column=False,
    resize_width=None,
):
    """
    Create a video from cameras, optionally drawing annotation overlays.

    If no annotations and no overlays are requested, delegates to
    create_video_from_cams for a faster direct ffmpeg path.
    """

    # Create a temporary directory for saving image matches
    temp_image_matches_output = tempfile.mkdtemp(prefix="image_matches_")
    print(f"Temporary image matches output: {temp_image_matches_output}")

    try:
        # Generate image matches for each camera and save them to disk.
        if annotations is not None:
            print(f"Processing {len(cams)} cameras with annotations...")
            for cam in tqdm(
                cams, total=len(cams), desc="Generating annotated frames for video"
            ):
                try:
                    # Check if camera has valid filepath
                    if (
                        not hasattr(cam, "filepath")
                        or not cam.filepath
                        or not os.path.exists(cam.filepath)
                    ):
                        print(
                            f"Warning: Camera {cam.cam_id} has invalid filepath: {getattr(cam, 'filepath', 'None')}"
                        )
                        continue

                    image_matches = cam.get_image_matches(annotations, pcd=pcd)
                    if sam_predictor:
                        for match in image_matches:
                            match.get_sam2_masks(sam_predictor)

                    save_image_matches_within_camera(
                        image_matches,
                        cam,
                        temp_image_matches_output,
                        use_label_column,
                        resize_width,
                    )
                except Exception as e:
                    print(f"Error processing camera {cam.cam_id}: {e}")
                    continue
        else:
            from joblib import Parallel, delayed

            def save_cam_image(cam):
                image_matches = None
                save_image_matches_within_camera(
                    image_matches,
                    cam,
                    temp_image_matches_output,
                    use_label_column,
                    resize_width,
                )

            cams_list = list(cams)
            Parallel(n_jobs=-1)(
                delayed(save_cam_image)(cam)
                for cam in tqdm(
                    cams_list, desc="Generating frames for video (without annotations)"
                )
            )
        # Create a video from the saved images.
        print("Creating video from frames...")

        # Build a concat list to preserve camera order and avoid glob issues
        file_list_path = None
        target_width = resize_width
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as f:
                file_list_path = f.name
                # Determine ordered list of cams used
                ordered_cams = list(cams) if annotations is None else list(cams)
                last_img = None
                frames_found = 0
                for cam in ordered_cams:
                    out_path = os.path.join(temp_image_matches_output, cam.filename)
                    if os.path.isfile(out_path):
                        f.write(f"file '{out_path}'\n")
                        f.write("duration 0.5\n")
                        last_img = out_path
                        frames_found += 1
                    else:
                        print(
                            f"Warning: Frame not found for camera {cam.cam_id}: {out_path}"
                        )

                if frames_found == 0:
                    print("Error: No frames were generated for the video")
                    print(
                        f"Temporary directory contents: {os.listdir(temp_image_matches_output)}"
                    )
                    return

                print(f"Found {frames_found} frames for video creation")
                if last_img is not None:
                    f.write(f"file '{last_img}'\n")

            # Compute safe target width if not provided
            if target_width is None and last_img is not None:
                try:
                    with Image.open(last_img) as im:
                        w, _ = im.size
                        if w > 4096:
                            target_width = 4096
                except Exception:
                    target_width = None

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                file_list_path,
                "-r",
                "2",  # Set output framerate to 2 fps
                "-vsync",
                "cfr",  # Use constant frame rate with -r
            ]
            if target_width is not None:
                cmd += ["-vf", f"scale={int(target_width)}:-2"]
            cmd += [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_filename,
            ]
            subprocess.run(cmd, check=True)
        finally:
            if file_list_path and os.path.exists(file_list_path):
                try:
                    os.remove(file_list_path)
                except Exception:
                    pass
    finally:
        # Clean up the temporary directory after creating the video
        shutil.rmtree(temp_image_matches_output, ignore_errors=True)
