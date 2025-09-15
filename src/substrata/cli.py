# Standard Library
import argparse
import os
import re

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
        raise SystemExit("No input PLY found. Provide --input or ensure initializer finds a PLY in CWD.")

    # Default output: initializer's decimated path or <id>_dec50M.ply beside the source
    default_output = init.ply_dec_path or os.path.join(init.path or cwd, f"{init.id}_dec50M.ply")
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
        raise SystemExit("No input PLY found. Provide --input/--ply or ensure initializer finds a PLY in CWD.")

    ply_head(input_path, n=args.num, print_output=True)


def handle_scalebars(args):
    # Use initializer to infer defaults from CWD when not provided
    from substrata.initializer import ProjectInitializer

    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd)

    # 1) resolve inputs
    pcd_path = args.input or init.ply_full_path
    if not pcd_path:
        raise SystemExit("No input PLY found. Provide --input/--ply or ensure initializer finds a PLY in CWD.")

    markers_path = args.markers or init.markers_filepath
    if not markers_path:
        raise SystemExit("No markers file found. Provide --markers or ensure initializer finds a markers CSV in CWD.")

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
        try:
            scale_factor = sb.calc_scalefactor()
            init.scale_factor = float(scale_factor)
            yaml_path = init.yaml_path or os.path.join(init.path or os.getcwd(), f"{init.id}.yaml")
            init.save_config_to_yaml(yaml_path)
            print(f"Saved scale_factor to YAML: {yaml_path}")
        except Exception as e:
            print(f"Warning: failed to save scale_factor to YAML: {e}")


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

    # Optional orientation via initializer workflow - ignore if world_transform is already set
    if getattr(args, "auto_orient", False):
        if init.world_transform_is_identity:
            init.scale_and_orient()
        else:
            print(f"Warning: world_transform is already set, skipping auto-orientation")

    # Save composite views PDF from initialized point cloud
    output_pdf = args.output_pdf or _get_output_filepath(init, "views.pdf")
    init.pcd.save_pdf(filepath=output_pdf)


def handle_firefish(args):
    # Build defaults from current directory name
    base, cwd = _cwd_base()
    init = ProjectInitializer(path=cwd)

    target_depth = _infer_target_depth(base, args.target_depth)
    pdf_output = args.pdf_output or os.path.join(cwd, f"{base}_firefish.pdf")
    cam_depths_file = args.cam_depths_file or os.path.join(cwd, f"{base}_camdepths.csv")

    # Initialize FireFish file
    from substrata.firefish import FireFish
    ff = FireFish(args.firefish_file or os.path.join(cwd, f"{base}_firefish.txt"))

    # depth_outlier_thresh is passed through only if provided; otherwise rely on default
    kwargs = {
        "camdepths_filepath": cam_depths_file,
        "pdf_output_filepath": pdf_output,
    }
    if args.depth_outlier_thresh is not None:
        kwargs["depth_and_outlier_threshold"] = args.depth_outlier_thresh

    if args.input:
        init.pcd_filepath = args.input

    init.initialize(apply_transform=False)

    # Only apply scale_factor (if specified or needed for save_yaml)
    if init.scale_factor is not None or getattr(args, "save_yaml", False):
        init.scale()
        from substrata.geometry import Transform
        init.world_transform = Transform.from_scale(init.scale_factor)

    init.initialize(apply_transform=True)
    pcd = init.pcd
    cams = init.cams

    # Optional subset by camera group name
    if getattr(args, "cams_group", None):
        try:
            cams = cams.subset_by_group(args.cams_group)
            print(f"Subsetting cameras to group '{args.cams_group}' → {len(cams.items())} cameras")
        except Exception:
            pass

    # Run up-vector determination
    up_vector, depth_offset, depth_per_unit, _ = ff.determine_up_vector(
        cams,
        target_depth,
        pcd,
        offset=args.offset,
        **kwargs,
    )

    # Optionally persist orientation results to YAML
    if getattr(args, "save_yaml", False):
        try:
            # Re-initialize the pointcloud with no transform
            init.world_transform = np.eye(4)
            init.initialize()

            # Set scaling/orientation values
            init.up_vector = up_vector
            init.depth_offset = float(depth_offset)
            init.depth_per_unit = float(depth_per_unit)

            init.scale_and_orient()
            # Save values to YAML
            yaml_path = init.yaml_path or os.path.join(init.path or os.getcwd(), f"{init.id}.yaml")
            init.save_config_to_yaml(yaml_path)
            print(f"Saved orientation to YAML: {yaml_path}")
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
        try:
            anns = Annotations()
            anns.get_annotations_from_file(anns_path, header=True, orig_coords_only=False)
        except Exception:
            anns = None

    # Optional subset by camera group name
    cams_for_video = init.cams
    if getattr(args, "cams_group", None):
        try:
            cams_for_video = init.cams.subset_by_group(args.cams_group)
        except Exception:
            pass

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
    p_dec.add_argument(
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_dec.add_argument(
        "--output", dest="output", type=str, default=None,
        help="Optional explicit output PLY path (defaults to <id>_dec50M.ply).",
    )
    p_dec.add_argument(
            "-n", "--points", dest="points", type=int, default=50_000_000,
            help="Number of points to keep (default: 50,000,000).",
    ) 

    # head (PLY preview)
    p_head = subparsers.add_parser(
        "head", help="Show first N vertex rows from a PLY file."
    )
    p_head.add_argument(
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_head.add_argument(
        "-n", dest="num", type=int, default=5,
        help="Number of vertex rows to display (default: 5).",
    )

    # scalebars
    p_sb = subparsers.add_parser(
        "scalebars",
        help="Generate scalebar PDF from a point cloud and marker annotations.",
    )
    p_sb.add_argument(
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_sb.add_argument(
        "--markers", dest="markers", type=str, default=None,
        help="Optional explicit markers CSV path (overrides initializer).",
    )
    p_sb.add_argument(
        "--output_pdf", dest="output_pdf", type=str, default=None,
        help="Optional output PDF filepath.",
    )
    p_sb.add_argument(
        "-n", "--points", dest="points", type=int, default=50000000,
        help="Optional max points to stream-load PLY (decimation on load).",
    )
    p_sb.add_argument(
        "-s", "--save_yaml", dest="save_yaml", action="store_true",
        help="Save computed scale_factor into a YAML config for this project.",
    )

    # views
    p_views = subparsers.add_parser(
        "views", help="Save composite views PDF for a point cloud."
    )
    p_views.add_argument(
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_views.add_argument(
        "--output_pdf", dest="output_pdf", type=str, default=None, 
        help="Output PDF filepath.",
    )
    p_views.add_argument(
        "--auto-orient", dest="auto_orient", action="store_true",
        help="Initialize and orient the project before saving.",
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
        "--firefish-file", dest="firefish_file", type=str, default=None,
        help=("Path to FireFish file. Default: <cwd_basename>_firefish.txt in current folder."),
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
        "--cams-group",
        dest="cams_group",
        type=str,
        default=None,
        help=(
            "Optional camera group name to subset (uses Cameras.subset_by_group)."
        ),
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
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    # removed --camspath-postfix (use --cams-group instead)
    p_ff.add_argument(
        "-s", "--save_yaml", dest="save_yaml", action="store_true",
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
        "--input", "--ply", dest="input", type=str, default=None,
        help="Optional explicit input PLY path (overrides initializer).",
    )
    p_c2v.add_argument(
        "--annotations", dest="annotations_file", type=str, default=None,
        help=("Path to annotations CSV. Uses initializer if omitted."),
    )
    p_c2v.add_argument(
        "-l", "--label", dest="use_label_column", action="store_true",
        help=("Use label column from annotations when drawing matches (default: off)."),
    )
    p_c2v.add_argument(
        "-r", "--resolution", dest="resolution", type=int, default=None,
        help=("Optional width to resize images when creating frames (pixels)."),
    )
    p_c2v.add_argument(
        "--cams-group",
        dest="cams_group",
        type=str,
        default=None,
        help=(
            "Optional camera group name to subset (uses Cameras.subset_by_group)."
        ),
    )
    p_c2v.add_argument(
        "--output_mp4", dest="output_mp4", type=str, default=None,
        help="Optional output MP4 filepath (default: <id>_cams.mp4).",
    )

    args = parser.parse_args()

    handlers = {
        "decimate": handle_decimate,
        "head": handle_head,
        "scalebars": handle_scalebars,
        "views": handle_views,
        "firefish": handle_firefish,
        "cams2video": handle_cams2video,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
