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
            return int(m.group(1))
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
    init = ProjectInitializer(path=cwd)

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
    init = ProjectInitializer(path=cwd)

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
    init = ProjectInitializer(path=cwd)

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
    init = ProjectInitializer(path=cwd)

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize()

    # Apply world_transform if it's not identity (already set from YAML or previous runs)
    if not init.world_transform_is_identity:
        init.apply_world_transform()

    # Save composite views PDF from initialized point cloud
    output_pdf = args.output_pdf or _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)


def handle_orient(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd)

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input

    # Initialize project (loads PCD, cameras/markers if available)
    init.initialize()

    # Run scale_and_orient workflow
    init.scale_and_orient()

    # Always save values to YAML
    yaml_path = init.yaml_path or os.path.join(init.path or cwd, f"{init.id}.yaml")
    init.save_config_to_yaml(yaml_path)
    print(f"Saved orientation to YAML: {yaml_path}")

    # Also output composite views as done for the "views" command
    output_pdf = args.output_pdf or _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)

    # Save camera depth residuals PDF
    output_pdf = args.output_pdf or _get_output_filepath(init, "depth_residuals.pdf")
    init.cams.save_depth_residuals_pdf(filepath=output_pdf)


def handle_firefish(args):
    # Build defaults from current directory name
    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd)

    # Arguments
    target_depth = _infer_target_depth(base, args.target_depth)
    pdf_output = args.pdf_output or os.path.join(cwd, f"{base}_firefish.pdf")
    cam_depths_file = args.cam_depths_file or os.path.join(cwd, f"{base}_camdepths.csv")
    kwargs = {
        "camdepths_filepath": cam_depths_file,
        "pdf_output_filepath": pdf_output,
    }
    if args.depth_outlier_thresh is not None:
        kwargs["depth_and_outlier_threshold"] = args.depth_outlier_thresh
    if args.input:
        init.pcd_filepath = args.input

    # Initialize FireFish file
    from substrata.firefish import FireFish

    ff = FireFish(args.firefish_file or os.path.join(cwd, f"{base}_firefish.txt"))

    # Initialize project (loads PCD, cameras/markers if available) without transforms
    init.initialize(apply_transform=False)

    # Optionally filter cameras by group name
    if args.cams_group:
        cams = init.cams.group(args.cams_group)
    else:
        cams = init.cams

    # Calculate scale factor to get accurate camera distances only
    if init.scale_factor is None:
        init.calc_scale_factor()
    if init.scale_factor is None:
        raise ValueError("Scale factor is not set")
    else:
        print("Scale factor: {init.scale_factor}")

    # Run up-vector determination (on unscaled/unoriented pointcloud)
    up_vector, depth_offset, depth_per_unit = ff.determine_up_vector(
        cams,
        target_depth,
        init.pcd,
        distance_scale_factor=init.scale_factor,  # for camdists only
        offset=args.offset,
        **kwargs,
    )

    # Optionally persist orientation results to YAML
    if getattr(args, "save_yaml", False):
        try:
            # Set scaling/orientation values and run apply_orientation_transforms()
            # via the scale_and_orient() method
            init.up_vector = up_vector
            init.depth_offset = float(depth_offset)
            init.depth_per_unit = float(depth_per_unit)
            init.scale_and_orient()

            # Save values to YAML
            yaml_path = init.yaml_path or os.path.join(
                init.path or os.getcwd(), f"{init.id}.yaml"
            )
            init.save_config_to_yaml(yaml_path)
            print(f"Saved orientation to YAML: {yaml_path}")

            # Save composite views PDF from initialized point cloud
            output_pdf = args.output_pdf or _get_output_filepath(init, "views.pdf")
            init.pcd.save_pdf(filepath=output_pdf)

            # Save camera depth residuals PDF
            output_pdf = args.output_pdf or _get_output_filepath(
                init, "depth_residuals.pdf"
            )
            init.cams.save_depth_residuals_pdf(filepath=output_pdf)

        except Exception as e:
            print(f"Warning: failed to save orientation to YAML: {e}")


def handle_cams2video(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer
    from substrata import visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd)

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
    init = ProjectInitializer(path=cwd)

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
    optimal_bbox = measurements.find_optimal_box_position(
        init.pcd, box_length=args.box_length, box_width=args.box_width, step_size=0.1
    )
    try:
        bboxes = measurements.subdivide_boxes(optimal_bbox, args.box_size)
    except ValueError as e:
        sys.exit(f"Failed to subdivide boxes: {e}")

    fig = visualizations.show_grid_cells(init.pcd, bboxes)

    # Sample random XY points inside cells and compute intercepts
    random_points = measurements.generate_random_xy_points_within_cells(bboxes, 1, 0)
    intercepts = init.pcd.get_z_intercepts(
        random_points, args.search_radius, always_return=True
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


def handle_images(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer
    from substrata import visualizations

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd)

    # Allow explicit override of PLY path
    if args.input:
        init.pcd_filepath = args.input
    # Allow explicit override of annotations path
    if args.annotations:
        init.annotations_filepath = args.annotations

    # Initialize (loads PCD and cameras if available)
    init.initialize()

    # Resolve annotations
    if not init.annotations_filepath:
        raise SystemExit(
            "No annotations file found. Provide --annotations or ensure "
            "initializer finds an annotations CSV in CWD."
        )

    anns = Annotations(init.annotations_filepath, header=True, orig_coords_only=True)

    # Handle optional transform
    if getattr(args, "transform", False):
        print("Please paste the transform matrix (YAML or array format):")
        print("  - YAML format: world_transform")
        print("  - Array format: [[...], [...], [...]] or [[...], [...], [...], [...]]")
        transform_str = ""
        while True:
            try:
                line = input()
                if line.strip() == "":
                    break
                transform_str += line + "\n"
            except EOFError:
                break

        if not transform_str.strip():
            raise SystemExit("No transform provided")

        try:
            transform = _parse_transform_from_input(transform_str)
            print(f"Parsed transform:\n{transform}")

            # Transform orig_coords and use as new orig_coords
            from substrata import geom

            for ann_id in anns.data:
                ann = anns.data[ann_id]
                # Transform current orig_coords
                new_orig_coords = geom.transform_coords(ann.orig_coords, transform)
                # Set as new orig_coords and reset coords
                ann.orig_coords = new_orig_coords
                ann.coords = new_orig_coords.copy()
                print(f"{ann_id} orig_coords: {ann.orig_coords}, coords: {ann.coords}")
                # Also transform extra_coords if present
                for full_id in ann.extra_coords:
                    ann.extra_coords[full_id] = geom.transform_coords(
                        ann.extra_coords[full_id], transform
                    )
                    ann.orig_extra_coords[full_id] = ann.extra_coords[full_id].copy()

            # Reset world_transform to identity since we've updated orig_coords
            anns.world_transform = np.eye(4)
            print(
                "Applied transform to orig_coords and reset "
                "world_transform to identity"
            )

        except Exception as e:
            raise SystemExit(f"Failed to parse or apply transform: {e}")

    # # Apply world_transform from initializer if available
    # if not init.world_transform_is_identity:
    #     anns.apply_transform(init.world_transform)
    # elif init.scale_factor is not None:
    #     anns.apply_transform(geom.Transform.from_scale(init.scale_factor))

    # Validate that we have cameras and annotations
    if not init.cams or len(init.cams) == 0:
        raise SystemExit("No cameras available for image matching")

    if len(anns) == 0:
        raise SystemExit("No annotations found")

    print(f"Number of annotations: {len(anns)}")
    print(f"Number of cameras: {len(init.cams)}")

    # Get first image matches for each annotation
    image_matches = anns.get_first_image_matches(init.cams, pcd=init.pcd)

    if len(image_matches) == 0:
        raise SystemExit("No image matches found for any annotations")

    print(f"Found {len(image_matches)} image matches")

    # Save to PDF
    pdf_output = args.pdf_output or os.path.join(cwd, f"{base}_imagematches.pdf")
    visualizations.save_cropped_image_matches_to_pdf(image_matches, pdf_output)


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
        "--pdf-output",
        dest="pdf_output",
        type=str,
        default=None,
        help=(
            "Optional output PDF filepath. Default: <cwd_basename>_firefish.pdf in current folder."
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
        default=None,
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
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
