# Standard Library
import argparse
import ast
import os
import re
import sys

# Third-Party Libraries
import yaml
import numpy as np

# Local Modules
from substrata.pointclouds import PointCloud, decimate_ply_file, ply_head
from substrata.initializer import ProjectInitializer
from substrata.annotations import Annotations, Scalebars
from substrata import settings

# ---------------------------- helpers: parents & defaults ----------------------------

## Removed cameras parent; firefish/cams2video now rely on initializer


def _cwd_base():
    cwd = os.getcwd()
    base = os.path.basename(cwd.rstrip(os.sep))
    return base, cwd


def _infer_target_depth(base: str, explicit_depth):
    if explicit_depth is not None:
        return explicit_depth
    m = re.search(r"_(\d+)m_", base)
    if m:
        try:
            return -int(m.group(1))
        except Exception:
            return None
    return None


def _get_output_filepath(init: ProjectInitializer, postfix: str):
    """Get the output filepath from the initializer and the postfix."""
    return os.path.join(init.path or os.getcwd(), f"{init.id}_{postfix}")


# -------------------------------------- handlers -------------------------------------


def handle_decimate(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    input_path = args.input or init.ply_full_path
    if not input_path:
        raise SystemExit(
            "No input PLY found. Provide --input or ensure initializer finds a PLY in CWD."
        )

    # Default output: initializer's decimated path or <id>_dec50M.ply beside the source
    default_output = init.ply_dec_path or os.path.join(
        init.path or cwd, f"{init.id}_dec50M.ply"
    )
    output_path = args.output or default_output

    decimate_ply_file(
        input_path=input_path,
        output_path=output_path,
        target_points=args.points,
        show_progress=True,
    )


def handle_head(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    input_path = args.input or init.ply_full_path
    if not input_path:
        raise SystemExit(
            "No input PLY found. Provide --input/--ply or ensure initializer finds a PLY in CWD."
        )

    ply_head(input_path, n=args.num, print_output=True)


def handle_scalebars(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # 1) resolve inputs
    pcd_path = args.input or init.ply_full_path
    if not pcd_path:
        raise SystemExit(
            "No input PLY found. Provide --input/--ply or ensure initializer finds a PLY in CWD."
        )

    markers_path = args.markers or init.markers_filepath
    if not markers_path:
        raise SystemExit(
            "No markers file found. Provide --markers or ensure initializer finds a markers CSV in CWD."
        )

    # 2) load PCD (optionally streaming-decimate on load)
    pcd = PointCloud(pcd_path, max_points=args.points)

    # 3) load markers as annotations
    anns = Annotations()
    anns.get_annotations_from_file(markers_path, header=True)

    # 4) create Scalebars, attach target coords from annotations
    sb = Scalebars(scalebar_data=settings.RGL_SCALEBARS, target_data=anns)

    # 5) save PDF
    output_pdf = args.output_pdf or _get_output_filepath(init, "scalebars.pdf")
    sb.save_pdf(pcd, filepath=output_pdf)

    # 6) optionally persist scale factor to YAML
    if getattr(args, "save_yaml", False):
        scale_factor = sb.calc_scalefactor()
        if scale_factor is not None:
            init.scale_factor = float(scale_factor)
        else:
            print("Warning: failed to compute scale_factor")
        yaml_path = init.yaml_path or os.path.join(
            init.path or os.getcwd(), f"{init.id}.yaml"
        )
        init.save_config_to_yaml(yaml_path)
        print(f"Saved to YAML: {yaml_path}")


def handle_views(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize()

    # Save composite views PDF from initialized point cloud
    output_pdf = args.output_pdf or _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)


def handle_orient(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize(apply_transform=False)

    # Run scale_and_orient workflow (skip if --manual flag is set)
    if not getattr(args, "manual", False):
        # Use markers CSV filepath if provided, otherwise use camera depths
        depth_markers_filepath = getattr(args, "markers", None)
        if depth_markers_filepath is not None:
            depth_markers = Annotations(depth_markers_filepath, orig_coords_only=True)
        else:
            depth_markers = None
        init.scale_and_orient(depth_markers=depth_markers)

    # Handle optional manual transform
    if getattr(args, "transform", False) or getattr(args, "manual", False):
        manual_transform = _get_transform_from_user()
        # Apply manual transform to pointcloud
        # (multiplies with existing world_transform)
        init.pcd.apply_transform(manual_transform)
        # Update initializer's world_transform to match pointcloud
        init.world_transform = init.pcd.world_transform
        # Propagate updated world_transform to cameras/markers/annotations
        init.apply_world_transform(skip_pcd=True)

    # Always save values to YAML
    yaml_path = init.yaml_path or os.path.join(init.path or cwd, f"{init.id}.yaml")
    init.save_config_to_yaml(yaml_path)
    print(f"Saved orientation to YAML: {yaml_path}")

    # Also output composite views as done for the "views" command
    output_pdf = _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)

    # Save depth residuals PDF (from markers if used, otherwise from cameras)
    if not getattr(args, "manual", False):
        output_pdf = _get_output_filepath(init, "depth_residuals.pdf")
        if depth_markers is not None:
            # Use annotation depth residuals if depth_markers were used
            depth_markers.save_depth_residuals_pdf(filepath=output_pdf)
        else:
            # Use camera depth residuals (default behavior)
            init.cams.save_depth_residuals_pdf(filepath=output_pdf)


def handle_firefish(args):
    """Handles the 'firefish' command.

    Args:
        args: Parsed command-line arguments.
    """
    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Arguments
    target_depth = _infer_target_depth(base, args.target_depth)
    pdf_output = os.path.join(cwd, f"{base}_firefish.pdf")
    cam_depths_file = args.cam_depths_file or os.path.join(cwd, f"{base}_camdepths.csv")
    depth_and_outlier_threshold = (
        args.depth_outlier_thresh if args.depth_outlier_thresh is not None else None
    )
    if args.input:
        init.pcd_filepath = args.input

    from substrata.firefish import FireFish

    ff = FireFish(args.firefish_file or os.path.join(cwd, f"{base}_firefish.txt"))

    # Initialize project (loads PCD, cameras/markers if available) without transforms
    init.initialize(apply_transform=False)

    # Optionally filter cameras by group name
    if args.cams_group:
        filtered_cams = init.cams.group(args.cams_group)
        print(
            f"Number of cameras considered (after filter by group): {len(filtered_cams.items())}"
        )
    else:
        filtered_cams = init.cams
        print(f"All cameras are being considered: {len(filtered_cams.items())}")

    if len(filtered_cams.items()) == 0:
        raise ValueError(
            f"No cameras available in this group. Available camera groups: {init.cams.group_names}"
        )

    # Calculate scale factor (but do not apply yet - to get accurate camera distances only)
    init.calc_scale_factor()

    if init.scale_factor is None:
        raise ValueError("Scale factor is not set")
    else:
        print(f"Scale factor: {init.scale_factor}")

    # Run up-vector determination (on unscaled/unoriented pointcloud)
    init.up_vector, init.depth_offset, init.depth_per_unit = ff.determine_up_vector(
        filtered_cams,
        target_depth,
        init.pcd,
        distance_scale_factor=init.scale_factor,
        offset=args.offset,
        camdepths_filepath=cam_depths_file,
        pdf_output_filepath=pdf_output,
        depth_and_outlier_threshold=depth_and_outlier_threshold,
    )

    # Optionally persist orientation results to YAML
    if getattr(args, "save_yaml", False):
        # Apply scale and orientation transforms to pointcloud (using its properties
        # for additional centering and orientation)
        init.scale_and_orient(recalculate=False, plot=False)

        # Save values to YAML
        yaml_path = init.yaml_path or os.path.join(
            init.path or os.getcwd(), f"{init.id}.yaml"
        )
        init.save_config_to_yaml(yaml_path)
        print(f"Saved orientation to YAML: {yaml_path}")

        # Save composite views PDF from initialized point cloud
        output_pdf = _get_output_filepath(init, "views.pdf")
        init.pcd.save_pdf(filepath=output_pdf)

        # Save camera depth residuals PDF
        output_pdf = _get_output_filepath(init, "depth_residuals.pdf")
        init.cams.save_depth_residuals_pdf(filepath=output_pdf)


def handle_cams2video(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer
    from substrata import visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input
    # Allow explicit override of annotations path
    if args.annotations_file:
        init.annotations_filepath = args.annotations_file

    # Initialize (loads PCD and cameras if available)
    init.initialize()

    # Resolve annotations (optional)
    anns = None
    anns_path = init.annotations_filepath
    if anns_path:
        anns = Annotations(anns_path, header=True, orig_coords_only=True)
        anns.apply_transform(init.world_transform)
        print(f"Number of annotations: {len(anns)}")
    # Optional subset by camera group name
    cams_for_video = init.cams
    if getattr(args, "cams_group", None):
        try:
            cams_for_video = init.cams.subset_by_group(args.cams_group)
            print(
                f"Filtered cameras by group '{args.cams_group}': {len(cams_for_video)} cameras"
            )
        except Exception as e:
            print(
                f"Warning: Failed to filter cameras by group '{args.cams_group}': {e}"
            )
            print(f"Available camera groups: {init.cams.group_names}")
            print("Using all cameras instead")

    # Validate that we have cameras to work with
    if not cams_for_video or len(cams_for_video) == 0:
        print("Error: No cameras available for video creation")
        return

    print(f"Using {len(cams_for_video)} cameras for video creation")

    # Compute output path and ensure directory
    output_mp4 = args.output_mp4 or _get_output_filepath(init, "cams.mp4")
    # Create video directly to requested output path
    visualizations.create_annotated_video(
        cams_for_video,
        anns,
        output_filename=output_mp4,
        pcd=init.pcd,
        use_label_column=args.use_label_column,
        resize_width=args.resolution,
    )


def handle_intercepts(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer
    from substrata import measurements, visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if getattr(args, "input", None):
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras if available)
    init.initialize()

    # Require a non-identity world transform
    if init.pcd.world_transform_is_identity:
        sys.exit("World transform is not set")

    # Optionally align along slope before proceeding
    if getattr(args, "slope", False):
        slope_normal, slope_elev = init.pcd.apply_along_slope_transform()

    # Create bounding box, subdivide to grid, and visualize
    if getattr(args, "position", None):
        # Parse manual position [x, y] (top-left coordinate)
        try:
            position = ast.literal_eval(args.position)
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError("Position must be a list or tuple of length 2")
            x_top_left, y_top_left = float(position[0]), float(position[1])
        except (ValueError, SyntaxError) as e:
            sys.exit(
                f"Failed to parse --position argument: {e}. Expected format: [x,y]"
            )

        # Construct bbox from top-left position by adding box_length and box_width
        optimal_bbox = [
            [x_top_left, y_top_left],
            [x_top_left + args.box_length, y_top_left + args.box_width],
        ]
    else:
        # Use automatic optimal box position finding
        optimal_bbox = measurements.find_optimal_box_position(
            init.pcd,
            box_length=args.box_length,
            box_width=args.box_width,
            step_size=0.1,
        )
    try:
        bboxes = measurements.subdivide_boxes(optimal_bbox, args.box_size)
    except ValueError as e:
        sys.exit(f"Failed to subdivide boxes: {e}")

    fig = visualizations.show_grid_cells(init.pcd, bboxes)

    # Sample random XY points inside cells and compute intercepts
    random_points = measurements.generate_random_xy_points_within_cells(bboxes, 1, 0)
    intercepts = init.pcd.get_z_intercepts(
        random_points, args.search_radius, always_return=True, id_prefix=init.id
    )

    # Back-compute original coords and get first image matches if cameras are available
    intercepts.get_original_coords(init.pcd.world_transform)

    # Save intercepts to CSV (and YAML only for --slope)
    if getattr(args, "slope", False):
        fig.savefig(_get_output_filepath(init, "slope_bbox.png"))
        csv_path = _get_output_filepath(init, "slope_intercepts.csv")
        intercepts.save(csv_path)
        yaml_path = os.path.splitext(csv_path)[0] + ".yaml"
        payload = {
            "world_transform": init.pcd.world_transform.tolist(),
            "slope_normal": [float(x) for x in slope_normal],
            "slope_elevation_deg": float(slope_elev),
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(payload, f)
        # additional visualization for sanity check
        a, b, c, d, inliers_idx = measurements.get_best_fit_plane_PCA(init.pcd)
        visualizations.visualize_elevation_angle(
            init.pcd,
            [a, b, c, d],
            point_size=1,
            output_filename=_get_output_filepath(init, "slope_pca.png"),
        )
    else:
        fig.savefig(_get_output_filepath(init, "topdown_bbox.png"))
        csv_path = _get_output_filepath(init, "topdown_intercepts.csv")
        intercepts.save(csv_path)


def handle_align(args):
    from substrata.pointclouds import PointCloud

    if not args.source or not args.target:
        raise SystemExit("align requires --source and --target PLY paths")

    # Load with optional stream decimation
    src = PointCloud(args.source, max_points=args.points)
    tgt = PointCloud(args.target, max_points=args.points)

    # Compute transform that maps source → target space
    T, metrics = tgt.get_auto_align_transform(src)

    # Print and optionally show
    print("Metrics:", metrics)
    print("Alignment transform (source → target):\n", T)


def _parse_transform_from_input(transform_str: str) -> np.ndarray:
    """Parse a 4x4 transform matrix from user input (YAML or array format).

    Supports:
    - YAML format with world_transform key (3x4 or 4x4)
    - Array format (3x4 or 4x4)
    - Promotes 3x4 matrices to 4x4 by adding [0, 0, 0, 1] bottom row
    - Promotes 3x3 matrices to 4x4 by adding bottom row and right column

    Args:
        transform_str: String containing transform in YAML or array format.

    Returns:
        4x4 numpy array representing the transform matrix.
    """
    transform_str = transform_str.strip()

    # Try to parse as YAML first (check for YAML-like structure)
    if ":" in transform_str or "world_transform" in transform_str.lower():
        try:
            yaml_data = yaml.safe_load(transform_str)
            if isinstance(yaml_data, dict) and "world_transform" in yaml_data:
                transform = np.array(yaml_data["world_transform"], dtype=float)
            elif isinstance(yaml_data, list):
                transform = np.array(yaml_data, dtype=float)
            else:
                raise ValueError("YAML format not recognized")
        except Exception as e:
            raise ValueError(f"Failed to parse YAML format: {e}")
    else:
        # Try to parse as array/list format
        try:
            parsed = ast.literal_eval(transform_str)
            transform = np.array(parsed, dtype=float)
        except Exception as e:
            raise ValueError(
                f"Failed to parse array format: {e}. "
                "Expected format: [[...], [...], [...], [...]] or YAML format"
            )

    return transform


def _read_single_transform_input() -> str:
    """Read a single transform matrix input from stdin.

    Returns:
        String containing the transform matrix in YAML or array format.

    Raises:
        SystemExit: If no transform is provided.
    """
    lines = []
    while True:
        try:
            line = input()
            # If we get an empty line and we already have content, we're done
            if line.strip() == "" and lines:
                break
            # If we get an empty line and no content yet, continue waiting
            if line.strip() == "":
                continue
            lines.append(line)
        except EOFError:
            # Ctrl+D signals end of input
            break

    transform_str = "\n".join(lines)
    if not transform_str.strip():
        raise SystemExit("No transform provided")

    return transform_str


def _get_transform_from_user() -> np.ndarray:
    """Prompt user for transform matrix input, read from stdin, and parse it.

    Supports multiple transforms that are multiplied cumulatively.
    After the first transform is parsed successfully, prompts for additional
    transforms. Each new transform is multiplied with the cumulative result
    using np.dot(new_transform, cumulative_transform).

    Returns:
        4x4 numpy array representing the cumulative transform matrix.

    Raises:
        SystemExit: If no transform is provided or parsing fails.
    """
    print("Please paste the transform matrix (YAML or array format):")
    print("  - YAML format: world_transform")
    print("  - Array format: [[...], [...], [...]] or " "[[...], [...], [...], [...]]")

    # Read and parse first transform
    transform_str = _read_single_transform_input()
    try:
        cumulative_transform = _parse_transform_from_input(transform_str)
        print(f"Parsed transform:\n{cumulative_transform}")
    except Exception as e:
        raise SystemExit(f"Failed to parse transform: {e}")

    # Loop for additional transforms
    while True:
        print(
            "If you want to apply any additional transforms paste the next "
            "one below, otherwise press ENTER"
        )
        try:
            line = input().strip()
            if line == "":
                # Empty line means no more transforms
                break
        except EOFError:
            # Ctrl+D means no more transforms
            break

        # Read the additional transform
        additional_lines = [line]
        while True:
            try:
                next_line = input()
                if next_line.strip() == "":
                    break
                additional_lines.append(next_line)
            except EOFError:
                break

        additional_transform_str = "\n".join(additional_lines)
        if not additional_transform_str.strip():
            break

        try:
            additional_transform = _parse_transform_from_input(additional_transform_str)
            print(f"Parsed additional transform:\n{additional_transform}")
            # Multiply: new_transform @ cumulative_transform
            cumulative_transform = np.dot(additional_transform, cumulative_transform)
            print(f"Cumulative transform:\n{cumulative_transform}")
        except Exception as e:
            raise SystemExit(f"Failed to parse additional transform: {e}")

    return cumulative_transform


def handle_images(args):
    """Handle the image matching CLI command.

    Uses project initializer to load data and optionally apply a transform.
    Generates cropped image match visualizations and saves them to PDF.

    Args:
        args: Arguments from argparse.
    """
    from substrata.initializer import ProjectInitializer
    from substrata import visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd, local=getattr(args, "local", False))

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input
    # Allow explicit override of annotations path
    if args.annotations:
        init.annotations_filepath = args.annotations

    # Resolve annotations
    if not init.annotations_filepath:
        raise SystemExit(
            "No annotations file found. Provide --annotations or ensure "
            "initializer finds an annotations CSV in CWD."
        )

    # Initialize (loads PCD and cameras if available)
    init.initialize()

    # Handle optional transform
    if getattr(args, "transform", False):
        transform = _get_transform_from_user()

        # Transform orig_coords and use as new orig_coords
        from substrata import geom

        try:
            for annotation in init.annotations:
                old_coords = annotation.orig_coords
                annotation.orig_coords = geom.transform_coords(
                    annotation.orig_coords, transform
                )
                if annotation.extra_coords:
                    raise ValueError(
                        "Extra coords are not supported for image matching"
                    )
                print(
                    f"{annotation.id} orig_coords changed from {old_coords} "
                    f"to {annotation.orig_coords}"
                )
        except Exception as e:
            raise SystemExit(f"Failed to apply transform: {e}")

    # # Apply world_transform from initializer if available
    # if not init.world_transform_is_identity:
    #     init.annotations.apply_transform(init.world_transform)
    # elif init.scale_factor is not None:
    #     init.annotations.apply_transform(geom.Transform.from_scale(init.scale_factor))

    # Validate that we have cameras and annotations
    if not init.cams or len(init.cams) == 0:
        raise SystemExit("No cameras available for image matching")

    if len(init.annotations) == 0:
        raise SystemExit("No annotations found")

    print(f"Number of annotations: {len(init.annotations)}")
    print(f"Number of cameras: {len(init.cams)}")

    # Get first image matches for each annotation
    image_matches = init.annotations.get_first_image_matches(init.cams, pcd=init.pcd)

    if len(image_matches) == 0:
        raise SystemExit("No image matches found for any annotations")

    print(f"Found {len(image_matches)} image matches")

    # Save to PDF
    pdf_output = args.pdf_output or os.path.join(cwd, f"{base}_imagematches.pdf")
    crop_size = args.size if args.size is not None else 1000
    visualizations.save_cropped_image_matches_to_pdf(
        image_matches, pdf_output, crop_w=crop_size, crop_h=crop_size
    )


def handle_transform(args):
    """Handle the transform CLI command.

    Loads annotations from a file, prompts for one or more transforms,
    applies them to orig_coords of each annotation, and sets coords to match.

    Args:
        args: Arguments from argparse.
    """
    from substrata import geom

    # Load annotations from file
    header = not getattr(args, "no_header", False)
    ignore_header = getattr(args, "ignore_header", False)
    anns = Annotations()
    anns.get_annotations_from_file(
        args.input, header=header, ignore_header=ignore_header
    )

    if len(anns) == 0:
        raise SystemExit("No annotations found in input file.")

    print(f"Loaded {len(anns)} annotations from {args.input}")

    # Prompt for transform(s)
    transform = _get_transform_from_user()

    # Apply inverse if requested
    if getattr(args, "inverse", False):
        transform = np.linalg.inv(transform)
        print(f"Applied inverse transform:\n{transform}")

    # Apply transform to each annotation's orig_coords and set coords to match
    for annotation in anns:
        old_orig_coords = annotation.orig_coords.copy()
        annotation.orig_coords = geom.transform_coords(
            annotation.orig_coords, transform
        )
        annotation.coords = annotation.orig_coords
        print(
            f"{annotation.id} orig_coords transformed from {old_orig_coords} "
            f"to {annotation.orig_coords}"
        )

    # Save transformed annotations
    if args.output:
        output_path = args.output
    else:
        # Default output: add _transformed suffix before extension
        base_path, ext = os.path.splitext(args.input)
        output_path = f"{base_path}_transformed{ext}"

    anns.save(output_path, orig_coords_only=True)
    print(f"Saved transformed annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Substrata CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # decimate (initializer-driven defaults)
    p_dec = subparsers.add_parser(
        "decimate",
        help=(
            "Decimate a PLY. With no args, uses initializer on CWD, output to <id>_dec50M.ply, target=50,000,000."
        ),
    )

    # No file output or visualization in this simplified command
    p_dec.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_dec.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Optional explicit output PLY path (defaults to <id>_dec50M.ply).",
    )
    p_dec.add_argument(
        "-n",
        "--points",
        dest="points",
        type=int,
        default=50_000_000,
        help="Number of points to keep (default: 50,000,000).",
    )
    p_dec.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # head (PLY preview)
    p_head = subparsers.add_parser(
        "head", help="Show first N vertex rows from a PLY file."
    )
    p_head.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_head.add_argument(
        "-n",
        dest="num",
        type=int,
        default=5,
        help="Number of vertex rows to display (default: 5).",
    )
    p_head.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # scalebars
    p_sb = subparsers.add_parser(
        "scalebars",
        help="Generate scalebar PDF from a point cloud and marker annotations.",
    )
    p_sb.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_sb.add_argument(
        "--markers",
        dest="markers",
        type=str,
        default=None,
        help="Optional explicit markers CSV path (overrides initializer).",
    )
    p_sb.add_argument(
        "--output_pdf",
        dest="output_pdf",
        type=str,
        default=None,
        help="Optional output PDF filepath.",
    )
    p_sb.add_argument(
        "-n",
        "--points",
        dest="points",
        type=int,
        default=50000000,
        help="Optional max points to stream-load PLY (decimation on load).",
    )
    p_sb.add_argument(
        "-s",
        "--save_yaml",
        dest="save_yaml",
        action="store_true",
        help="Save computed scale_factor into a YAML config for this project.",
    )
    p_sb.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # views
    p_views = subparsers.add_parser(
        "views", help="Save composite views PDF for a point cloud."
    )
    p_views.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_views.add_argument(
        "--output_pdf",
        dest="output_pdf",
        type=str,
        default=None,
        help="Output PDF filepath.",
    )
    p_views.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # orient
    p_orient = subparsers.add_parser(
        "orient",
        help="Calculate and apply scale and orientation transforms, save to YAML.",
    )
    p_orient.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_orient.add_argument(
        "--transform",
        dest="transform",
        action="store_true",
        help=(
            "Prompt for manual transform matrix to apply after scale_and_orient "
            "(accepts YAML or array format, supports multiple transforms)."
        ),
    )
    p_orient.add_argument(
        "--manual",
        dest="manual",
        action="store_true",
        help=(
            "Skip the automatic scale_and_orient workflow. "
            "Use with --transform to apply only manual transforms."
        ),
    )
    p_orient.add_argument(
        "--markers",
        dest="markers",
        type=str,
        default=None,
        help="Optional markers CSV filepath to use for up vector calculation.",
    )
    p_orient.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # firefish
    p_ff = subparsers.add_parser(
        "firefish",
        help=(
            "Run FireFish/Cameras alignment: initialize FireFish + Cameras and "
            "determine up vector/output PDF."
        ),
    )
    p_ff.add_argument(
        "--firefish-file",
        dest="firefish_file",
        type=str,
        default=None,
        help=(
            "Path to FireFish file. Default: <cwd_basename>_firefish.txt in current folder."
        ),
    )
    p_ff.add_argument(
        "--target-depth",
        dest="target_depth",
        type=int,
        default=None,
        help=(
            "Optional target depth in meters. Default: extracted from <cwd_basename> pattern _<int>m_."
        ),
    )
    p_ff.add_argument(
        "--cam-depths-file",
        dest="cam_depths_file",
        type=str,
        default=None,
        help=(
            "Optional CSV to store camera depths. Default: <cwd_basename>_camdepths.csv in current folder."
        ),
    )
    p_ff.add_argument(
        "--depth-outlier-threshold",
        dest="depth_outlier_thresh",
        type=int,
        default=settings.FIREFISH_DEPTH_ALTITUDE_OUTLIER_THRESHOLD,
        help=(
            "Optional depth/outlier threshold (meters). Defaults to FireFish.determine_up_vector default."
        ),
    )
    p_ff.add_argument(
        "--cams_group",
        dest="cams_group",
        type=str,
        default=None,
        help=("Optional camera group name to subset (uses Cameras.subset_by_group)."),
    )
    p_ff.add_argument(
        "--offset",
        dest="offset",
        type=int,
        default=None,
        help=(
            "Manually set time offset in seconds; skips automatic offset determination."
        ),
    )
    p_ff.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_ff.add_argument(
        "-s",
        "--save_yaml",
        dest="save_yaml",
        action="store_true",
        help="Save computed up_vector, depth_offset, depth_per_unit into YAML.",
    )
    p_ff.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # cams2video
    p_c2v = subparsers.add_parser(
        "cams2video",
        help=(
            "Create a video from cameras by drawing image matches (initializer-driven)."
        ),
    )
    p_c2v.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_c2v.add_argument(
        "--annotations",
        dest="annotations_file",
        type=str,
        default=None,
        help=("Path to annotations CSV. Uses initializer if omitted."),
    )
    p_c2v.add_argument(
        "-l",
        "--label",
        dest="use_label_column",
        action="store_true",
        help=("Use label column from annotations when drawing matches (default: off)."),
    )
    p_c2v.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        type=int,
        default=None,
        help=("Optional width to resize images when creating frames (pixels)."),
    )
    p_c2v.add_argument(
        "--cams_group",
        dest="cams_group",
        type=str,
        default=None,
        help=("Optional camera group name to subset (uses Cameras.subset_by_group)."),
    )
    p_c2v.add_argument(
        "--output_mp4",
        dest="output_mp4",
        type=str,
        default=None,
        help="Optional output MP4 filepath (default: <id>_cams.mp4).",
    )
    p_c2v.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # intercepts
    p_intercepts = subparsers.add_parser(
        "intercepts",
        help=(
            "Find optimal box, subdivide to grid, sample random points, compute Z-intercepts."
        ),
    )
    p_intercepts.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )

    # align
    p_align = subparsers.add_parser(
        "align",
        help=(
            "Register a source PLY to a target PLY and print transform (optionally save aligned source)."
        ),
    )
    p_align.add_argument(
        "--source", dest="source", type=str, required=True, help="Source PLY path"
    )
    p_align.add_argument(
        "--target", dest="target", type=str, required=True, help="Target PLY path"
    )
    p_align.add_argument(
        "--points",
        dest="points",
        type=int,
        default=5_000_000,
        help="Max points to stream-load",
    )
    # No output or visualization flags
    p_intercepts.add_argument(
        "--box-length",
        dest="box_length",
        type=float,
        default=25.0,
        help="Rectangle length in meters for optimal box search (default: 25).",
    )
    p_intercepts.add_argument(
        "--box-width",
        dest="box_width",
        type=float,
        default=4.0,
        help="Rectangle width in meters for optimal box search (default: 4).",
    )
    p_intercepts.add_argument(
        "--position",
        dest="position",
        type=str,
        default=None,
        help=(
            "Manual top-left coordinate [x,y] for the bounding box. "
            "When provided, skips find_optimal_box_position and uses this position "
            "plus box_length and box_width to define the bounding box. "
            "Format: [x,y] (e.g., '[1.4, 2.6]')."
        ),
    )
    p_intercepts.add_argument(
        "--box-size",
        dest="box_size",
        type=float,
        default=0.2,
        help="Grid cell size in meters for subdividing the optimal box (default: 0.2).",
    )
    p_intercepts.add_argument(
        "--search-radius",
        dest="search_radius",
        type=float,
        default=settings.DEFAULT_INTERCEPT_SEARCH_RADIUS,
        help="Search radius in meters for Z-intercept lookup (default: 0.005).",
    )
    p_intercepts.add_argument(
        "--slope",
        dest="slope",
        action="store_true",
        help="Apply along-slope transform to pointcloud before processing.",
    )
    p_intercepts.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # images
    p_images = subparsers.add_parser(
        "images",
        help=("Find image matches of annotations and output cropped images to PDF."),
    )
    p_images.add_argument(
        "--input",
        "--ply",
        dest="input",
        type=str,
        default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_images.add_argument(
        "--annotations",
        dest="annotations",
        type=str,
        default=None,
        help="Optional explicit annotations CSV path (overrides initializer).",
    )
    p_images.add_argument(
        "--transform",
        dest="transform",
        action="store_true",
        help=(
            "Prompt for 4x4 transform matrix to apply to orig_coords "
            "(accepts YAML or array format)."
        ),
    )
    p_images.add_argument(
        "--pdf-output",
        dest="pdf_output",
        type=str,
        default=None,
        help=(
            "Optional output PDF filepath. "
            "Default: <cwd_basename>_imagematches.pdf in current folder."
        ),
    )
    p_images.add_argument(
        "--size",
        dest="size",
        type=int,
        default=None,
        help=(
            "Crop size in pixels for image matches (sets both crop_w and crop_h). "
            "Default: 1000 (uses function default)."
        ),
    )
    p_images.add_argument(
        "--local",
        dest="local",
        action="store_true",
        help="Reset all paths to local (relative to project path).",
    )

    # transform
    p_transform = subparsers.add_parser(
        "transform",
        help=(
            "Transform annotations: load from file, apply transform(s) to orig_coords, "
            "and save result."
        ),
    )
    p_transform.add_argument(
        "input",
        type=str,
        help="Path to input annotations CSV file.",
    )
    p_transform.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help=(
            "Optional output annotations CSV filepath. "
            "Default: <input_basename>_transformed.csv"
        ),
    )
    p_transform.add_argument(
        "--no_header",
        dest="no_header",
        action="store_true",
        help="Set header=False when loading annotations (default: header=True).",
    )
    p_transform.add_argument(
        "--ignore_header",
        dest="ignore_header",
        action="store_true",
        help="Skip the first line of the file and ignore header argument (default: False).",
    )
    p_transform.add_argument(
        "--inverse",
        dest="inverse",
        action="store_true",
        help="Apply the inverse of the cumulative transforms (default: False).",
    )

    args = parser.parse_args()

    handlers = {
        "decimate": handle_decimate,
        "head": handle_head,
        "scalebars": handle_scalebars,
        "views": handle_views,
        "orient": handle_orient,
        "firefish": handle_firefish,
        "cams2video": handle_cams2video,
        "intercepts": handle_intercepts,
        "align": handle_align,
        "images": handle_images,
        "transform": handle_transform,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
