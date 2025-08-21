# Standard Library
import os
import tempfile
from io import BytesIO

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import ffmpeg
from fpdf import FPDF
from mpl_toolkits.mplot3d import Axes3D
from open3d import geometry, io, utility, visualization
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Local Modules
from substrata import settings, segmentation, pointclouds
from substrata.logging import logger

# from open3d.web_visualizer import draw


def capture_geoms_to_file(geoms, output_file):
    vis = visualization.Visualizer()
    vis.create_window(visible=False)
    for geom in geoms:
        vis.add_geometry(geom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_file)
    vis.destroy_window()


def show(geoms, Jupyter=False):
    if Jupyter:
        draw(geoms)
    else:
        vis = visualization.Visualizer()
        vis.create_window()

        for geom in geoms:
            vis.add_geometry(geom)
        coordinate_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=3, origin=[0, 0, 0]
        )
        vis.add_geometry(coordinate_frame)

        vis.run()


def plot(
    pcd,
    point_size=2,
    width=10,
    height=4,
    max_output_points=50000,
    title=None,
    ax=None,
):
    """
    Plot a 3D point cloud, with optional decimation for speed.

    Args:
        pcd: The point cloud object (Open3D format or SimplePointCloud).
        point_size (int): Size of the points in the scatter plot.
        width (int): Width of the figure (if creating a new figure).
        height (int): Height of the figure (if creating a new figure).
        max_output_points (int): Maximum number of points to plot.
        title (str | None): Title for the plot.
        ax (matplotlib.axes.Axes | None): Optional 3D axes to draw into.

    Returns:
        matplotlib.figure.Figure | None: New figure if created, otherwise None.
    """
    # Convert to Open3D point cloud if necessary.
    if isinstance(pcd, pointclouds.SimplePointCloud):
        o3d_pcd = pcd.get_o3d_pcd()
    else:
        o3d_pcd = pcd.o3d_pcd

    points = np.asarray(o3d_pcd.points)
    colors = np.asarray(o3d_pcd.colors)

    # Decimate the point cloud
    if len(points) > max_output_points:
        logger.info(
            f"Decimating point cloud from {len(points)} to {max_output_points} points"
        )
        decimation_factor = len(points) // max_output_points
        indices = np.random.choice(
            len(points), size=len(points) // decimation_factor, replace=False
        )
        points = points[indices]
        colors = colors[indices]

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure
    ax.set_box_aspect((width, height, height))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=point_size,
        edgecolor="none",
    )
    # Weighted equal scaling: x : y : z = width : height : height
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    ranges = maxs - mins

    weights = np.array([width, height, height], dtype=float)
    k = float(np.max(ranges / weights)) if np.all(weights > 0) else 1.0

    half = 0.5 * k * weights
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])

    # Keep the box aspect consistent with the intended physical ratios
    ax.set_box_aspect((width, height, height))
    if title is not None:
        ax.set_title(title)
    ax.set_rasterized(True)
    return fig if created_fig else None


def plot_2d(
    pcd,
    point_size=2,
    width=10,
    height=5,
    highlight_coords=None,
    title=None,
    max_output_points=50000,
):
    # Convert to Open3D point cloud if necessary.
    if isinstance(pcd, pointclouds.SimplePointCloud):
        o3d_pcd = pcd.get_o3d_pcd()
    else:
        o3d_pcd = pcd.o3d_pcd

    points = np.asarray(o3d_pcd.points)
    colors = np.asarray(o3d_pcd.colors)

    # Decimate the point cloud
    if len(points) > max_output_points:
        logger.info(
            f"Decimating point cloud from {len(points)} to {max_output_points} points"
        )
        decimation_factor = len(points) // max_output_points
        indices = np.random.choice(
            len(points), size=len(points) // decimation_factor, replace=False
        )
        points = points[indices]
        colors = colors[indices]

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=point_size,
        edgecolor="none",
    )

    # Highlight the specified coordinates with a big red dot
    if highlight_coords is not None:
        # Convert single coord to array of coords if needed
        coords = np.array(highlight_coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="red",
            s=50,  # Size of the highlight dot
            edgecolor="none",
        )

    if title is not None:
        ax.set_title(title)

    ax.set_rasterized(True)
    return fig


def multiplot_2d(
    pcds,
    annotations_list,
    point_size=2,
    width=10,
    height=5,
    title=None,
    max_output_points=50000,
    label_colors=None,
    max_x=None,
):
    """
    Plots 2D point cloud(s) with annotations highlighted in different colors based on their labels.

    Args:
        pcds: Single point cloud object or list of point cloud objects (Open3D format or SimplePointCloud)
        annotations_list: List of Annotations instances to highlight
        point_size (int): Size of the points in the scatter plot
        width (int): Width of the figure
        height (int): Height of the figure
        title (str): Optional title for the plot
        max_output_points (int): Maximum number of points to display
        label_colors (dict): Optional dictionary mapping labels to colors
        max_x (float): Optional maximum x-axis value to display

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Convert single pcd to list if necessary
    if not isinstance(pcds, list):
        pcds = [pcds]

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    # Process each point cloud
    for pcd in pcds:
        # Convert to Open3D point cloud if necessary
        if isinstance(pcd, pointclouds.SimplePointCloud):
            o3d_pcd = pcd.get_o3d_pcd()
        else:
            o3d_pcd = pcd.o3d_pcd

        points = np.asarray(o3d_pcd.points)
        colors = np.asarray(o3d_pcd.colors)

        # Decimate the point cloud
        if len(points) > max_output_points:
            logger.info(
                f"Decimating point cloud from {len(points)} to {max_output_points} points"
            )
            decimation_factor = len(points) // max_output_points
            indices = np.random.choice(
                len(points), size=len(points) // decimation_factor, replace=False
            )
            points = points[indices]
            colors = colors[indices]

        # Plot base point cloud
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            s=point_size,
            edgecolor="none",
        )

    # Plot annotations with different colors based on labels
    if annotations_list is not None:
        # Get unique labels and assign colors
        unique_labels = set()
        for annotations in annotations_list:
            unique_labels.update(ann.label for ann in annotations.data.values())

        # Use provided colors or generate rainbow colors
        if label_colors is None:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            label_colors = dict(zip(unique_labels, colors))

        # Plot each annotation point with its corresponding color
        for annotations in annotations_list:
            for ann in annotations.data.values():
                # Plot white outline first
                ax.scatter(
                    ann.coords[0],
                    ann.coords[1],
                    c="white",
                    s=70,  # Slightly larger than the colored dot
                    edgecolor="none",
                )
                # Plot colored dot on top
                ax.scatter(
                    ann.coords[0],
                    ann.coords[1],
                    c=[label_colors[ann.label]],
                    s=50,  # Size of the annotation dots
                    edgecolor="none",
                    label=f"{ann.label}",
                )

    if title is not None:
        ax.set_title(title)

    # Set x-axis limit if specified
    if max_x is not None:
        ax.set_xlim(right=max_x)

    # Remove y-axis values
    ax.set_yticklabels([])

    ax.set_rasterized(True)
    return fig


def plot_text(text, width=10, height=5):
    """
    Plots a text string as a figure (e.g. for inclusion in a PDF).
    """
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.5, text, ha="left", va="center")
    ax.set_axis_off()  # Hide all axes
    ax.set_rasterized(True)
    return fig


def plot_compare(pcd1, pcd2, point_size=1):
    """
    Plots two 3D point clouds with pcd1 in a blue color scale and pcd2 in a red color scale.

    Args:
        pcd1: First point cloud object with points and colors (Open3D format).
        pcd2: Second point cloud object with points and colors (Open3D format).
        point_size (int): Size of the points in the scatter plot.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    points1 = np.asarray(pcd1.o3d_pcd.points)
    colors1 = np.full((points1.shape[0], 3), [0.0, 0.0, 1.0])  # Blue scale

    points2 = np.asarray(pcd2.o3d_pcd.points)
    colors2 = np.full((points2.shape[0], 3), [1.0, 0.0, 0.0])  # Red scale

    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((6, 1, 1))

    # Plot the first point cloud
    ax.scatter(
        points1[:, 0],
        points1[:, 1],
        points1[:, 2],
        c=colors1,
        s=point_size,
        edgecolor="none",
    )

    # Plot the second point cloud
    ax.scatter(
        points2[:, 0],
        points2[:, 1],
        points2[:, 2],
        c=colors2,
        s=point_size,
        edgecolor="none",
    )

    ax.set_rasterized(True)

    return fig


def create_vector_geom(vector, length):
    lineset = geometry.LineSet()
    lineset.points = utility.Vector3dVector(
        [np.array([0, 0, 0]), np.array(vector) * float(length)]
    )
    lineset.lines = utility.Vector2iVector([[0, 1]])
    return lineset


def show_coords_as_lines(pcd, points, Jupyter=False):
    # Create orientation z-lines originating from points
    connect_points = []
    for point in points:
        if not point is None:
            connect_points.append(np.array(point))
            connect_points.append(
                np.array([point[0], point[1], point[2] + settings.LEN_ORIENT_LINE])
            )

    connect_lines = [[i, i + 1] for i in range(0, len(connect_points), 2)]

    connecting_lineset = geometry.LineSet()
    connecting_lineset.points = utility.Vector3dVector(connect_points)
    connecting_lineset.lines = utility.Vector2iVector(connect_lines)

    show([pcd.o3d_pcd, connecting_lineset], Jupyter=Jupyter)


def show_grid_points(pcd, grid_indices):
    try:
        pcd.o3d_pcd_tree
    except AttributeError:
        pcd.build_kd_tree()
    # Filter out None values and get the valid closest indices
    grid_point_idx = [idx for idx in grid_indices if idx is not None]
    for point in grid_point_idx:
        [k, idx, _] = pcd.o3d_pcd_tree.search_radius_vector_3d(pcd.points[point], 0.05)
        np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]
    show([pcd.o3d_pcd])


def show_point_values(pcd, annotations, meta_data_col_index=None, size=0.2):
    # Create orientation z-lines originating from points
    sphere_geoms = []
    for annotation in annotations.data.values():
        if meta_data_col_index is not None:
            color_value = min(float(annotation.meta_data[meta_data_col_index]), 1.0)
            color = [color_value, color_value, 0]
        else:
            color = [1, 0, 0]

        sphere = geometry.TriangleMesh.create_sphere(radius=size)
        num_vertices = np.asarray(sphere.vertices).shape[0]
        sphere.vertex_colors = utility.Vector3dVector([color] * num_vertices)

        sphere.compute_vertex_normals()
        sphere.translate(annotation.coords)
        sphere_geoms.append(sphere)

    show([pcd.o3d_pcd, *sphere_geoms])


def show_img_cv2(img_path, highlight_pixels=None):
    """Show image and"""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if highlight_pixels is not None:
        highlight_pixels = np.array(highlight_pixels, dtype=int)
        if highlight_pixels.ndim == 1:
            cv2.circle(
                image,
                (highlight_pixels[0], highlight_pixels[1]),
                radius=50,
                color=(255, 0, 0),
                thickness=-1,
            )
        elif highlight_pixels.ndim > 1:
            for pixel in highlight_pixels:
                cv2.circle(
                    image,
                    (pixel[0], pixel[1]),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,
                )
    plt.imshow(image)
    plt.show()


def show_img(img_path, highlight_pixels=None):
    """
    Load an image using PIL, optionally add highlighted circles, and show it.

    This function ignores the EXIF orientation (unless otherwise handled by PIL)
    and always displays the image in its raw pixel orientation.

    Args:
        img_path (str): Path to the image file.
        highlight_pixels (None, list, or np.ndarray):
            Pixel coordinates to highlight. If a 1D array/list, it's interpreted
            as a single point [x, y] with a larger circle; if 2D, each sub-list
            represents a point with a smaller circle.
    """
    # Open the image and convert to RGB.
    image = Image.open(img_path).convert("RGB")

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


def save_img(img_path, save_path, highlight_pixels=None):
    """
    Read an image from disk, optionally draw highlighted circles on given pixel coordinates,
    and save the full-resolution image to disk.

    Parameters:
        img_path (str): Path to the input image.
        save_path (str): Path where the resulting image will be saved.
        highlight_pixels (None, list, or np.ndarray): Pixel coordinates to highlight.
            - If a 1D array/list, it is assumed to be a single point [x, y].
            - If a 2D array/list, each sub-list represents a point [x, y].
    """
    # Read the full-resolution image in BGR format.
    image = cv2.imread(img_path)
    if image is None:
        raise IOError(f"Could not read image: {img_path}")

    # If highlight_pixels is provided, draw a circle on each.
    if highlight_pixels is not None:
        highlight_pixels = np.array(highlight_pixels, dtype=int)
        if highlight_pixels.ndim == 1:
            cv2.circle(
                image,
                (highlight_pixels[0], highlight_pixels[1]),
                radius=50,
                color=(0, 0, 255),
                thickness=-1,
            )
        elif highlight_pixels.ndim > 1:
            for pixel in highlight_pixels:
                cv2.circle(
                    image,
                    (pixel[0], pixel[1]),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )

    # Save the full-resolution image to disk.
    cv2.imwrite(save_path, image)


def show_cam_residuals(cams, cam_depth_residuals, transform=np.eye(4)):
    """Show camera residuals"""
    # Filter cameras based on cam_id being in cam_depth_residuals keys
    cams_with_depths_and_coords = [
        cam for cam in cams.data.values() if cam.cam_id in cam_depth_residuals
    ]
    points = np.array([cam.coords for cam in cams_with_depths_and_coords])

    # Apply transformation
    points_homogeneous = np.c_[points, np.ones(points.shape[0])]
    points_homogeneous = points_homogeneous.T
    transformed_points_homogeneous = np.dot(transform, points_homogeneous)
    transformed_points = transformed_points_homogeneous[:3].T

    # Get residuals from cam_depth_residuals
    residuals = np.array(
        [cam_depth_residuals[cam.cam_id] for cam in cams_with_depths_and_coords]
    )

    # Normalize the residuals (z-values) for color mapping
    max_abs_residual = max(abs(residuals))
    norm = plt.Normalize(vmin=-max_abs_residual, vmax=max_abs_residual)
    cmap = plt.cm.bwr  # Blue-White-Red colormap

    # Plot
    fig = plt.figure()
    plt.scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        c=residuals,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
    )
    plt.colorbar(label="Residual (z-value)")
    plt.title("Residuals Plot (as color)")
    plt.show()
    return fig

    ## Plot
    # x_values = np.arange(len(residuals))
    # plt.scatter(x_values, residuals, alpha=0.5)
    # plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Horizontal line at y = 0
    # plt.title('Residuals Plot (as value)')
    # plt.show()


def get_crop_img(img_path, crop_x, crop_y, crop_w, crop_h):
    """Get cropped image"""
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Could not read image: {img_path}")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    left = int(crop_x - crop_w / 2)
    top = int(crop_y - crop_h / 2)
    right = left + int(crop_w)
    bottom = top + int(crop_h)

    h_img, w_img = img.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(w_img, right)
    bottom = min(h_img, bottom)

    cropped_img = img[top:bottom, left:right]
    return cropped_img


def save_to_tmp_file(image):
    """Save image to temporary file"""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_file.close()
    cv2.imwrite(tmp_file.name, image)
    return tmp_file.name


def encode_to_png_buffer(image):
    """Encode image to JPG buffer"""
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise IOError("Could not encode image to JPG.")
    # Convert the NumPy buffer to a BytesIO stream
    return BytesIO(buffer.tobytes())


def save_cropped_image_matches_to_pdf(
    image_matches,
    output_filepath,
    n_cols=3,
    n_rows=4,
    crop_w=1000,
    crop_h=1000,
    single_mask=False,
):
    """
    Save a PDF file from a list of ImageMatches
    """
    pdf = FPDF()
    pdf.set_auto_page_break(False)
    pdf.set_font("Arial", size=8)

    pdf.add_page()
    # Page width and height in FPDF's units (default: mm)
    page_w = pdf.w
    page_h = pdf.h
    margin = 10

    # Compute the usable space (accounting# for margins)
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    # Cell size in the grid
    cell_w = usable_w / n_cols
    cell_h = usable_h / n_rows
    image_h = cell_h * 0.8

    # Track current row and column in the grid
    row_idx = 0
    idx = 0

    for match in image_matches.values():
        # Calculate the (x, y) position for this cell
        x = margin + idx * cell_w
        y = margin + row_idx * cell_h

        if match.masks:
            cropped_img = get_crop_img_from_masks(
                match, crop_w, crop_h, single_mask=single_mask
            )
        else:
            cropped_img = get_crop_img(
                match.cam.filepath, match.x, match.y, crop_w, crop_h
            )
        pdf.image(encode_to_png_buffer(cropped_img), x=x, y=y, w=cell_w, h=image_h)

        # Position the text a bit below the image
        label_x = x + 2  # small offset from left edge
        label_y = y + image_h + 4  # 4 units below bottom of the image
        pdf.text(label_x, label_y, f"{match.annotation.id} {match.annotation.label}")

        # Move to the next column
        idx += 1
        if idx == n_cols:
            idx = 0
            row_idx += 1
            if row_idx == n_rows:
                pdf.add_page()
                row_idx = 0
                idx = 0

    pdf.output(output_filepath)
    print(f"PDF created: {output_filepath}")


def get_crop_img_from_masks(
    image_match,
    output_img_w=1000,
    output_img_h=1000,
    pad_ratio=0.0,
    contour_thickness=2,
    annotation_radius=20,
    annotation_color=(0, 255, 0),
    single_mask=False,
):
    """
    Crop an image based on a list of SAM2 masks, draw each mask's contours in
    red, green, and blue (first mask with double thickness), annotate the
    surface area, and mark the annotation point.
    """
    # Load image
    img = cv2.imread(image_match.filepath)
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
    overlay_text = f"SA: {image_match.mask.area_in_cm2:.4f} cm2"
    font, font_scale, text_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    text_size, _ = cv2.getTextSize(overlay_text, font, font_scale, text_thickness)
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
    cv2.circle(
        crop_resized, (ann_x, ann_y), annotation_radius, annotation_color, thickness=-1
    )

    return crop_resized


def show_grid_cells(
    pcd, bboxes, show_colors=False, cell_size=None, sub_divisions=4, show_rep_cell=True
):
    """
    Visualize the grid cells and the main connected component as determined by spread filtering.

    Two visualizations are produced in one figure with two panels:
      - Left panel (10×3): A scatter plot of all x-y points (colored by the point cloud colors)
        with the boundaries of each filtered cell overlaid in red.
      - Right panel (3×3): For one of the filtered cells (the middle one in the list), a plot showing
        its internal subdivisions (drawn in green) overlaid on the points within that cell.
    """
    # If the input is a single bounding box, wrap it in a list.
    if isinstance(bboxes, (list, tuple)):
        # Check if bboxes is a single bounding box by verifying it has two elements and both are list/tuple with length 2.
        if (
            len(bboxes) == 2
            and isinstance(bboxes[0], (list, tuple))
            and isinstance(bboxes[1], (list, tuple))
        ):
            if len(bboxes[0]) == 2 and len(bboxes[1]) == 2:
                bboxes = [bboxes]
                show_rep_cell = False

    # Create a figure with two panels: left panel 10x3, right panel 3x3.
    fig = plt.figure(figsize=(13, 3))  # Total width=13 (10+3) inches, height=3 inches.
    gs = fig.add_gridspec(1, 2, width_ratios=[10, 3])

    # Left panel: All points with filtered grid cell boundaries.
    ax_left = fig.add_subplot(gs[0, 0])
    if show_colors:
        pcd_colors = np.asarray(pcd.colors)
    else:
        pcd_colors = None
    ax_left.scatter(pcd.points[:, 0], pcd.points[:, 1], s=1, c=pcd_colors, alpha=0.5)
    for bbox in bboxes:
        min_corner, max_corner = bbox
        x_vals = [
            min_corner[0],
            max_corner[0],
            max_corner[0],
            min_corner[0],
            min_corner[0],
        ]
        y_vals = [
            min_corner[1],
            min_corner[1],
            max_corner[1],
            max_corner[1],
            min_corner[1],
        ]
        ax_left.plot(x_vals, y_vals, "r-", linewidth=1)
    ax_left.set_xlabel("X")
    ax_left.set_ylabel("Y")
    ax_left.set_title("Overall plot with grid cell boundaries")
    ax_left.axis("equal")

    # Right panel: For one cell, show internal subdivisions.
    if show_rep_cell:
        ax_right = fig.add_subplot(gs[0, 1])
        if bboxes:
            middle_index = len(bboxes) // 2
            min_corner, max_corner = bboxes[middle_index]
            # Filter points within this cell.
            cell_mask = (
                (pcd.points[:, 0] >= min_corner[0])
                & (pcd.points[:, 0] < max_corner[0])
                & (pcd.points[:, 1] >= min_corner[1])
                & (pcd.points[:, 1] < max_corner[1])
            )
            cell_points = pcd.points[cell_mask]
            if pcd_colors is not None:
                cell_colors = pcd_colors[cell_mask]
            else:
                cell_colors = "b"

            ax_right.scatter(
                cell_points[:, 0], cell_points[:, 1], s=1, c=cell_colors, alpha=0.5
            )
            # Draw the cell boundary.
            x_vals = [
                min_corner[0],
                max_corner[0],
                max_corner[0],
                min_corner[0],
                min_corner[0],
            ]
            y_vals = [
                min_corner[1],
                min_corner[1],
                max_corner[1],
                max_corner[1],
                min_corner[1],
            ]
            ax_right.plot(x_vals, y_vals, "r-", linewidth=1)

            # Draw subdivisions.
            if cell_size is None:
                cell_size = max_corner[0] - min_corner[0]

            sub_cell_size = cell_size / sub_divisions
            for m in range(sub_divisions):
                for n in range(sub_divisions):
                    sub_x_min = min_corner[0] + m * sub_cell_size
                    sub_y_min = min_corner[1] + n * sub_cell_size
                    sub_x_max = sub_x_min + sub_cell_size
                    sub_y_max = sub_y_min + sub_cell_size
                    x_vals = [sub_x_min, sub_x_max, sub_x_max, sub_x_min, sub_x_min]
                    y_vals = [sub_y_min, sub_y_min, sub_y_max, sub_y_max, sub_y_min]
                    ax_right.plot(x_vals, y_vals, "g-", linewidth=0.5)

            ax_right.set_xlabel("X")
            ax_right.set_ylabel("Y")
            ax_right.set_title("Representative cell from the middle")
            ax_right.axis("equal")

    plt.tight_layout()
    return plt


def show_intercept_point(intercept_point):
    """
    Create a 3D scatter plot visualizing candidate points and key markers.
    Candidate points are colored by their Z value, and the following are
    plotted:
        - The query coordinate as a black square,
        - The computed intercept as a green square.
    A dashed line connects the query and selected point, and a 3D
    cylinder (centered at the query XY) represents the search radius.
    """
    xy_coord = intercept_point.estimated_coords[0:2]
    search_radius = intercept_point.search_radius
    candidates = np.array(intercept_point.simple_pcd.points)
    intercept = np.array(intercept_point.estimated_coords)
    selected = np.array(intercept_point.coords)

    # Create figure with two panels: left panel for 3D plot, right panel for 2D plot.
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    # Left panel: 3D scatter plot.
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")

    candidate_x = candidates[:, 0]
    candidate_y = candidates[:, 1]
    candidate_z = candidates[:, 2]

    intercept_z = intercept[2]

    # Determine colors based on z-value differences
    colors = np.where(
        np.abs(candidate_z - intercept_z) <= search_radius,
        "white",
        np.where(candidate_z < intercept_z - search_radius, "blue", "red"),
    )

    # Plot candidate points.
    ax_3d.scatter(
        candidate_x,
        candidate_y,
        candidate_z,
        c=colors,
        s=20,
        alpha=0.7,
        edgecolors="k",
    )

    # For visualization, assume query z equals intercept's z.
    query_z = intercept[2]
    ax_3d.scatter(
        xy_coord[0],
        xy_coord[1],
        query_z,
        c="k",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Plot computed intercept.
    ax_3d.scatter(
        intercept[0],
        intercept[1],
        intercept[2],
        c="g",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Draw dashed line from query to selected point.
    ax_3d.plot(
        [xy_coord[0], selected[0]],
        [xy_coord[1], selected[1]],
        [query_z, selected[2]],
        "k--",
        lw=1,
    )

    # Create a cylinder representing the search radius.
    # Cylinder is centered at (xy_coord[0], xy_coord[1]) and spans z range.
    theta = np.linspace(0, 2 * np.pi, 30)
    z_cyl = np.linspace(candidate_z.min(), candidate_z.max(), 30)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = xy_coord[0] + search_radius * np.cos(theta_grid)
    y_cyl = xy_coord[1] + search_radius * np.sin(theta_grid)

    ax_3d.plot_surface(x_cyl, y_cyl, z_grid, color="gray", alpha=0.2, edgecolor="none")

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Point Intercept: Query, Intercept, Selected")

    # Right panel: 2D scatter plot (XY plane).
    ax_2d = fig.add_subplot(gs[0, 1])

    # Plot candidate points.
    ax_2d.scatter(
        candidate_x,
        candidate_y,
        c=colors,
        s=20,
        alpha=0.7,
        edgecolors="k",
    )

    # Plot the query coordinate.
    ax_2d.scatter(
        xy_coord[0],
        xy_coord[1],
        c="k",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Plot the computed intercept.
    ax_2d.scatter(
        intercept[0],
        intercept[1],
        c="g",
        marker="s",
        s=120,
        edgecolors="k",
    )

    # Draw dashed line from query to selected point.
    ax_2d.plot(
        [xy_coord[0], selected[0]],
        [xy_coord[1], selected[1]],
        "k--",
        lw=1,
    )

    # Draw a circle representing the search radius.
    circle = plt.Circle(
        (xy_coord[0], xy_coord[1]), search_radius, color="k", fill=False, ls="--", lw=1
    )
    ax_2d.add_patch(circle)

    ax_2d.set_xlabel("X")
    ax_2d.set_ylabel("Y")
    ax_2d.set_title("2D Point Intercept: XY Plane")
    ax_2d.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()


def plot_2d_ortho(
    pcd,
    resolution=None,
    color_attr="colors",
    figsize=None,
    save_path=None,
    ax=None,
    title=None,
    show=True,
):
    """
    Create and display a top-down orthomosaic (splat) of a point cloud.

    Args:
        pcd: Input point cloud (supports PointCloud, SimplePointCloud, or o3d PointCloud-like).
        resolution (float | None): Ground sampling distance in meters per
            pixel. If None, choose a resolution based on point count and
            plot size.
        color_attr (str): Attribute for color ('colors' or 'intensities').
        figsize (tuple): Figure size for matplotlib (width, height) when creating a new figure.
        save_path (str | None): If provided, save the image to this file.
        ax (matplotlib.axes.Axes | None): Optional axes to draw into; if None, creates a new fig.
        title (str | None): Optional title for the plot.
        show (bool): Whether to call plt.show() when creating a new figure.

    Returns:
        tuple[np.ndarray, matplotlib.figure.Figure | None]: (image array, figure if created else None).
    """
    # Extract points
    pts = np.asarray(pcd.points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Compute bounds
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    extent_x = max(max_x - min_x, 1e-9)
    extent_y = max(max_y - min_y, 1e-9)

    # Choose resolution heuristically if not provided
    print(f"Resolution: {resolution}")
    if resolution is None:
        n_pts = max(len(pts), 1)
        # Target total pixels ~ n_pts/10, bounded for practicality
        target_pixels = int(np.clip(n_pts / 10.0, 2e5, 2e6))
        aspect = extent_x / extent_y
        width_px = int(max(256, np.sqrt(target_pixels * max(aspect, 1e-6))))
        height_px = int(max(256, target_pixels / max(width_px, 1)))
        resolution = extent_x / width_px
    else:
        width_px = int(np.ceil(extent_x / resolution))
        height_px = int(np.ceil(extent_y / resolution))

    width = width_px
    height = height_px

    # Prepare splat and count buffers
    splat = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width), dtype=int)

    # Fetch colors or default to white
    if hasattr(pcd, color_attr):
        cols = np.asarray(getattr(pcd, color_attr))
        if cols.ndim == 1:
            cols = np.vstack((cols, cols, cols)).T
    else:
        cols = np.ones((pts.shape[0], 3), dtype=np.float64)

    # Rasterize
    for pt, col in zip(pts, cols):
        ix = int((pt[0] - min_x) / resolution)
        iy = int((pt[1] - min_y) / resolution)
        if 0 <= ix < width and 0 <= iy < height:
            splat[iy, ix] += col
            counts[iy, ix] += 1

    # Normalize and set background (no points) to white
    mask = counts > 0
    splat[mask] /= counts[mask][:, None]
    splat[~mask] = 1.0

    # Convert to uint8
    img = (np.clip(splat, 0, 1) * 255).astype(np.uint8)

    # Set figsize to match pixel size (1 inch = dpi pixels)
    if figsize is None:
        dpi = plt.rcParams.get("figure.dpi", 100)
        figsize = (width / dpi, height / dpi)

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(img)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    if save_path is not None and created_fig:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if created_fig and show:
        plt.show()
    if created_fig and not show:
        plt.close(fig)

    return img, (fig if created_fig else None)


def plot_camera_view_from_pcd(
    pcd,
    image_match,
    resolution=0.005,
    color_attr="colors",
    figsize=None,
    save_path=None,
    show_plot=True,
    crop_to_mask=True,
):
    """
    Create a 2D rasterized view of a point cloud from a camera's perspective.

    This function projects the point cloud onto the camera's image plane and
    creates a rasterized view. If crop_to_mask is True, it crops to the area
    indicated by the ImageMatch's mask.

    Args:
        pcd: Point cloud (SimplePointCloud or Open3D point cloud)
        image_match: ImageMatch object with camera reference and mask information
        resolution (float): Ground sampling distance (m per pixel)
        color_attr (str): Attribute for color ('colors' or 'intensities')
        figsize (tuple): Figure size for matplotlib (width, height)
        save_path (str): Optional path to save the visualization
        show_plot (bool): Whether to display the plot
        crop_to_mask (bool): Whether to crop to the mask area

    Returns:
        tuple: (fig, raster_img) - matplotlib figure and rasterized image array
    """
    # Convert to Open3D point cloud if necessary
    if isinstance(pcd, pointclouds.SimplePointCloud):
        o3d_pcd = pcd.get_o3d_pcd()
    else:
        o3d_pcd = pcd.o3d_pcd

    points = np.asarray(o3d_pcd.points)

    # Get camera from image_match
    cam = image_match.cam

    # Get camera transform matrix
    if hasattr(cam, "transform") and cam.transform is not None:
        transform = cam.transform
    else:
        # Create transform from camera vector and position
        if hasattr(cam, "vector") and cam.vector is not None:
            # Use camera vector to create a simple transform
            # This is a simplified approach - you might want to enhance this
            transform = np.eye(4)
            transform[:3, 3] = cam.coords  # Set translation
            # Note: This doesn't set rotation based on vector - you'd need more complex logic
        else:
            raise ValueError("Camera must have either transform or vector attribute")

    # Transform points to camera coordinate system
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (transform @ points_homogeneous.T).T[:, :3]

    # Project to 2D (simple orthographic projection for now)
    # You might want to implement proper perspective projection here
    points_2d = points_cam[:, :2]  # Just take X,Y coordinates

    # Debug: Check if we have any valid points
    print(f"Total points: {len(points_2d)}")
    print(
        f"Points 2D bounds: X[{points_2d[:, 0].min():.3f}, {points_2d[:, 0].max():.3f}], Y[{points_2d[:, 1].min():.3f}, {points_2d[:, 1].max():.3f}]"
    )

    # Get colors
    if hasattr(o3d_pcd, color_attr):
        colors = np.asarray(getattr(o3d_pcd, color_attr))
        if colors.ndim == 1:
            colors = np.vstack((colors, colors, colors)).T
    else:
        colors = np.ones((points.shape[0], 3), dtype=np.float64)

    # Determine bounds
    min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
    min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()

    print(f"Initial bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")

    # If crop_to_mask is True and image_match has a mask
    if crop_to_mask and hasattr(image_match, "mask") and image_match.mask is not None:
        # Get mask bounds in image coordinates
        mask_vals = image_match.mask.vals
        mask_height, mask_width = mask_vals.shape

        # Convert pixel coordinates to 3D world coordinates using camera projection
        if hasattr(image_match, "x") and hasattr(image_match, "y"):
            # Use camera's pixel_to_point method to get 3D world coordinates
            world_coords, _, _ = cam.pixel_to_point(image_match.x, image_match.y, pcd)

            if world_coords is not None:
                # Use the 3D world coordinates as center
                center_x, center_y = world_coords[0], world_coords[1]

                # Estimate scale based on pixel scale if available
                if hasattr(image_match, "pixel_scale") and image_match.pixel_scale:
                    scale = image_match.pixel_scale
                else:
                    scale = resolution  # Use resolution as fallback

                # Calculate bounds based on mask size
                half_width = (mask_width / 2) * scale
                half_height = (mask_height / 2) * scale

                min_x = center_x - half_width
                max_x = center_x + half_width
                min_y = center_y - half_height
                max_y = center_y + half_height

                print(
                    f"Cropped bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]"
                )
                print(f"Center: ({center_x:.3f}, {center_y:.3f}), Scale: {scale:.6f}")
                print(f"Mask size: {mask_width}x{mask_height} pixels")

    # Compute image size
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # Ensure minimum size
    width = max(width, 100)
    height = max(height, 100)

    # Prepare raster and count buffers
    raster = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width), dtype=int)

    # Rasterize points
    points_in_bounds = 0
    for pt, col in zip(points_2d, colors):
        ix = int((pt[0] - min_x) / resolution)
        iy = int((pt[1] - min_y) / resolution)
        if 0 <= ix < width and 0 <= iy < height:
            raster[iy, ix] += col
            counts[iy, ix] += 1
            points_in_bounds += 1

        print(f"Raster size: {width}x{height}")
    print(f"Points in bounds: {points_in_bounds}/{len(points_2d)}")
    print(f"Non-zero pixels: {np.sum(counts > 0)}")

    # If no points in bounds, fall back to full bounds
    if points_in_bounds == 0:
        print("No points in bounds, using full point cloud bounds")
        min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
        min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))
        width = max(width, 100)
        height = max(height, 100)
        raster = np.zeros((height, width, 3), dtype=np.float64)
        counts = np.zeros((height, width), dtype=int)

        # Re-rasterize with full bounds
        for pt, col in zip(points_2d, colors):
            ix = int((pt[0] - min_x) / resolution)
            iy = int((pt[1] - min_y) / resolution)
            if 0 <= ix < width and 0 <= iy < height:
                raster[iy, ix] += col
                counts[iy, ix] += 1

    # Normalize
    mask = counts > 0
    raster[mask] /= counts[mask][:, None]

    # Convert to uint8
    raster_img = (np.clip(raster, 0, 1) * 255).astype(np.uint8)

    # Set figsize
    if figsize is None:
        dpi = plt.rcParams.get("figure.dpi", 100)
        figsize = (width / dpi, height / dpi)

    # Create visualization
    fig = plt.figure(figsize=figsize)
    plt.imshow(raster_img, extent=[min_x, max_x, min_y, max_y], origin="lower")
    plt.title(
        f"Camera View: {cam.cam_id} - Match at pixel ({image_match.x}, {image_match.y})"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Camera view saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, raster_img


def show_vector_in_pcd(pcd, vector, origin, length, Jupyter=False):
    """
    Visualize a 3D vector as a line from the origin to a point Xm away.

    The function normalizes the input vector, computes the endpoint
    by moving 2 meters along the vector from the origin, creates an
    Open3D LineSet, and calls the show method to render it.

    Args:
        vector (array-like): The 3D direction vector.
        origin (array-like): The starting point of the vector.
        Jupyter (bool): If True, uses Jupyter visualization.
    """
    # Ensure inputs are numpy arrays.
    vec = np.asarray(vector, dtype=float)
    origin = np.asarray(origin, dtype=float)

    # Check for zero-length vector.
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Input vector must be non-zero.")

    # Normalize and compute endpoint Xm away.
    vec = vec / norm
    end_point = origin + length * vec

    # Create a LineSet with two points: origin and end_point.
    line_set = geometry.LineSet()
    points = [origin.tolist(), end_point.tolist()]
    line_set.points = utility.Vector3dVector(points)
    line_set.lines = utility.Vector2iVector([[0, 1]])
    # Color the line (red).
    line_set.colors = utility.Vector3dVector([[1, 0, 0]])

    # Visualize using the provided show method.
    show([pcd.o3d_pcd, line_set])


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
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 100)
    except Exception as e:
        print("Error loading TrueType font:", e)
        font = ImageFont.load_default()

    # Define offsets for the ellipse and text.
    circle_radius = 50
    text_offset_x = circle_radius + 50  # Offset text to the right of the ellipse.
    text_offset_y = -50  # Slightly above the center of the ellipse.

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


def show_image_matches_within_camera(
    image_matches, cam, use_label_column=False, resize_width=None
):
    """Show image matches within the camera image"""
    image = draw_image_matches_within_camera(
        image_matches, cam, use_label_column, resize_width
    )
    plt.imshow(np.array(image))
    plt.show()


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


def create_video_from_cams(
    cams,
    annotations,
    output_path,
    sam_predictor=None,
    pcd=None,
    use_label_column=False,
    resize_width=None,
):
    """ """

    # Generate image matches for each camera and save them to disk.
    for cam in tqdm(cams, total=len(cams.items()), desc="Generating frames for video"):
        image_matches = cam.get_image_matches(annotations, pcd=pcd)
        if sam_predictor:
            for match in image_matches:
                match.get_sam2_masks(sam_predictor)

        save_image_matches_within_camera(
            image_matches,
            cam,
            output_path,
            use_label_column,
            resize_width,
        )
    # Create a video from the saved images.
    print("Creating video from frames...")
    # ffmpeg.set_ffmpeg = "/opt/homebrew/bin/ffmpeg"  # TO REMOVE
    (
        ffmpeg.input(f"{output_path}/*.JPG", pattern_type="glob", framerate=2)
        .output(f"{output_path}/output.mp4", r=1)
        .run()
    )


def visualize_elevation_angle(pcd, plane_coeffs, elevation_angle, output_filename=None):
    """
    Visualize the fitted plane and the point cloud.

    Args:
        pcd: The point cloud object.
        a, b, c, d: Plane coefficients.
        output_filename: Optional filename to save the visualization.
    """
    # Convert the Open3D point cloud to NumPy
    points = np.asarray(pcd.points)
    if np.asarray(pcd.colors).size > 0:
        colors = np.asarray(pcd.colors)
    else:
        # If no colors are present, default to gray
        colors = np.tile([0.5, 0.5, 0.5], (points.shape[0], 1))

    a, b, c, d = plane_coeffs

    # ------------------------- 3D Visualization --------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 1) Plot the fitted plane FIRST (with a low alpha)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = (-a * xx - b * yy - d) / c
    ax.plot_surface(xx, yy, zz, color="red", alpha=0.1)

    # 2) Plot the point cloud next (slightly higher alpha)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=4, alpha=0.4)

    # Determine bounding box and origin for arrows
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)
    origin = np.array([mid_x, mid_y, mid_z])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    arrow_length = max_range  # Increase to make arrows more visible

    # 3) Plot the vertical arrow (blue)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        0,
        0,
        arrow_length,
        color="blue",
        linewidth=2,
        arrow_length_ratio=0.05,
    )

    # 4) Plot the plane normal arrow (green)
    plane_normal_unit = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        plane_normal_unit[0] * arrow_length,
        plane_normal_unit[1] * arrow_length,
        plane_normal_unit[2] * arrow_length,
        color="green",
        linewidth=2,
        arrow_length_ratio=0.05,
    )

    # 5) Spherical arc between vertical_normal and plane_normal_unit (orange)
    num_arc_points = 40
    arc_points = []
    for i in range(num_arc_points):
        t = i / (num_arc_points - 1)
        direction = utils.slerp(np.array([0, 0, 1]), plane_normal_unit, t)
        arc_points.append(origin + arrow_length * direction)
    arc_points = np.array(arc_points)

    # Plot the arc
    ax.plot3D(
        arc_points[:, 0],
        arc_points[:, 1],
        arc_points[:, 2],
        color="orange",
        linewidth=3,
    )

    # Annotate near the midpoint of the arc
    mid_idx = num_arc_points // 2
    ax.text(
        arc_points[mid_idx, 0],
        arc_points[mid_idx, 1],
        arc_points[mid_idx, 2],
        f"{np.degrees(np.arccos(np.clip(np.dot(plane_normal_unit, [0, 0, 1]), -1.0, 1.0))):.1f}°",
        color="orange",
        fontsize=10,
    )

    # -------------- Set up the axis labels, title, and limits --------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"Elevation angle: {elevation_angle}°")

    # Make all axes have the same scale
    ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
    ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
    ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)

    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()


def plot_xy_pca(points, mean, eig_vecs, eig_vals) -> None:
    """Scatter the points and show the first two eigen-vectors."""
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.4)
    plt.plot(mean[0], mean[1], "ro")
    scale = 2.0 * np.sqrt(eig_vals)
    colors = ["r", "g"]
    for i in range(2):
        dx, dy = scale[i] * eig_vecs[:, i]
        plt.arrow(
            mean[0],
            mean[1],
            dx,
            dy,
            width=0.01,
            color=colors[i],
            length_includes_head=True,
        )
    plt.gca().set_aspect("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY PCA")
    plt.show()


def plot_views(
    pcd,
    point_size=2,
    width=8,
    height=12,
    max_output_points=50000,
    title=None,
    ortho_resolution=None,
):
    """
    Create a composite figure with:
      - Row 1 (full width): orthoprojection using plot_2d_ortho, titled with pcd.filepath
      - Row 2 (full width): 3D plot using plot(), titled "3D view (N points)"
      - Row 3 (two columns): top-down (XY) and front (XZ) views
      - Row 4 (two columns): side (Y–Z) and side (−Y–Z) views
    """
    # Convert to Open3D point cloud if necessary
    if isinstance(pcd, pointclouds.SimplePointCloud):
        o3d_pcd = pcd.get_o3d_pcd()
        filepath = getattr(pcd, "filepath", None)
        n_points = len(pcd.points)
    else:
        o3d_pcd = pcd.o3d_pcd
        filepath = getattr(pcd, "filepath", None)
        n_points = len(np.asarray(o3d_pcd.points))

    points = np.asarray(o3d_pcd.points)
    colors = np.asarray(o3d_pcd.colors)

    # Decimate once for the 2D scatter views
    if len(points) > max_output_points:
        logger.info(
            f"Decimating point cloud from {len(points)} to {max_output_points} points"
        )
        decimation_factor = len(points) // max_output_points
        indices = np.random.choice(
            len(points), size=len(points) // decimation_factor, replace=False
        )
        points_scatter = points[indices]
        colors_scatter = colors[indices]
    else:
        points_scatter = points
        colors_scatter = colors

    def set_equal_2d(ax, x, y):
        # Keep the same units on X and Y by enforcing equal aspect and
        # using a square view that encloses the data.
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        half = max(xmax - xmin, ymax - ymin) / 2.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal", adjustable="box")

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.2, 0.85, 0.85])

    # Row 1: Orthoprojection (full width)
    ax_ortho = fig.add_subplot(gs[0, :])
    ortho_title = (
        os.path.basename(filepath) if filepath is not None else "Orthoprojection"
    )
    plot_2d_ortho(
        pcd if hasattr(pcd, "points") else o3d_pcd,
        resolution=ortho_resolution,
        ax=ax_ortho,
        title=ortho_title,
        show=False,
    )

    # Row 2: 3D plot (full width)
    ax_3d = fig.add_subplot(gs[1, :], projection="3d")
    plot_title = f"3D view ({n_points:,} points)"
    data_mins = points.min(axis=0)
    data_maxs = points.max(axis=0)
    data_ranges = np.maximum(data_maxs - data_mins, 1e-9)
    plot(pcd, point_size=point_size, ax=ax_3d, title=plot_title)
    ax_3d.set_box_aspect(tuple(data_ranges))
    try:
        ax_3d.margins(0)
    except Exception:
        pass

    # Row 3: Top-down (XY) and Front (XZ)
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_xy.scatter(
        points_scatter[:, 0],
        points_scatter[:, 1],
        c=colors_scatter,
        s=point_size,
        edgecolor="none",
    )
    ax_xy.set_title("Top-down (X–Y)", pad=6)
    set_equal_2d(ax_xy, points_scatter[:, 0], points_scatter[:, 1])

    ax_xz.scatter(
        points_scatter[:, 0],
        points_scatter[:, 2],
        c=colors_scatter,
        s=point_size,
        edgecolor="none",
    )
    ax_xz.set_title("Front (X–Z)", pad=6)
    set_equal_2d(ax_xz, points_scatter[:, 0], points_scatter[:, 2])

    # Row 4: Side (Y–Z) and Side (−Y–Z)
    ax_yz = fig.add_subplot(gs[3, 0])
    ax_nyz = fig.add_subplot(gs[3, 1])
    ax_yz.scatter(
        points_scatter[:, 1],
        points_scatter[:, 2],
        c=colors_scatter,
        s=point_size,
        edgecolor="none",
    )
    ax_yz.set_title("Side (Y–Z)", pad=6)
    set_equal_2d(ax_yz, points_scatter[:, 1], points_scatter[:, 2])

    ax_nyz.scatter(
        -points_scatter[:, 1],
        points_scatter[:, 2],
        c=colors_scatter,
        s=point_size,
        edgecolor="none",
    )
    ax_nyz.set_title("Side (−Y–Z)", pad=6)
    set_equal_2d(ax_nyz, -points_scatter[:, 1], points_scatter[:, 2])

    if title is not None:
        fig.suptitle(title, y=0.995)

    # Fine-tune layout to minimize whitespace while avoiding overlaps (A4 margins ≈ 0.5 in)
    margin_in = 0.5
    fig.subplots_adjust(
        left=margin_in / width,
        right=1.0 - (margin_in / width),
        top=1.0 - (margin_in / height),
        bottom=margin_in / height,
        wspace=0.10,
        hspace=0.20,
    )

    # Rasterize scatter-heavy 2D axes
    for ax in (ax_xy, ax_xz, ax_yz, ax_nyz):
        ax.set_rasterized(True)
        ax.margins(x=0.04, y=0.04)
        ax.tick_params(pad=2)

    return fig
