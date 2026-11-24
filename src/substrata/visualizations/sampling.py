"""Sampling figures.

``show_grid_cells``: the xy grid cells of the point-intercept sampling over the
point cloud, with one cell's subdivisions (the box of
:func:`substrata.measurements.find_optimal_box_position`, split by
:func:`substrata.measurements.subdivide_boxes`; ``substrata intercepts``).
"""

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np


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
    # Title: include overall box as bottom-left and top-right [[x,y],[x,y]]
    try:
        if bboxes and len(bboxes) > 0:
            # Normalize potential single-box input
            if (
                isinstance(bboxes, (list, tuple))
                and len(bboxes) == 2
                and isinstance(bboxes[0], (list, tuple))
                and isinstance(bboxes[1], (list, tuple))
                and len(bboxes[0]) == 2
                and len(bboxes[1]) == 2
                and not (
                    isinstance(bboxes[0][0], (list, tuple))
                    or isinstance(bboxes[1][0], (list, tuple))
                )
            ):
                boxes_iter = [bboxes]
            else:
                boxes_iter = bboxes

            min_x = min(box[0][0] for box in boxes_iter)
            min_y = min(box[0][1] for box in boxes_iter)
            max_x = max(box[1][0] for box in boxes_iter)
            max_y = max(box[1][1] for box in boxes_iter)
            ax_left.set_title(
                f"Box coordinates: [[{min_x:.1f},{min_y:.1f}],[{max_x:.1f},{max_y:.1f}]]"
            )
        else:
            ax_left.set_title("Overall plot with grid cell boundaries")
    except Exception:
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
