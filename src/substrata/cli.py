# cli.py
import argparse
from substrata.pointclouds import PointCloud, decimate_ply_file, ply_head
from substrata.annotations import Annotations, Scalebars
from substrata import settings
import os
import re


def main():
    parser = argparse.ArgumentParser(description="Substrata CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # decimate
    p_dec = subparsers.add_parser(
        "decimate", help="Decimate a binary PLY to a target number of points."
    )
    p_dec.add_argument("input", type=str, help="Path to input binary PLY file.")
    p_dec.add_argument("output", type=str, help="Path to output PLY file.")
    p_dec.add_argument("target", type=int, help="Number of points to keep.")
    p_dec.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars."
    )

    # head (PLY preview)
    p_head = subparsers.add_parser(
        "head", help="Show first N vertex rows from a PLY file."
    )
    p_head.add_argument("pcd_filename", type=str, help="Input PLY file.")
    p_head.add_argument(
        "-n",
        "--num",
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
    p_sb.add_argument("pcd_filename", type=str, help="Input point cloud (PLY).")
    p_sb.add_argument("markers_filename", type=str, help="Markers CSV for annotations.")
    p_sb.add_argument("output_pdf", type=str, help="Output PDF filepath.")
    p_sb.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional streaming decimation during load.",
    )
    p_sb.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars during load."
    )

    # views
    p_views = subparsers.add_parser(
        "views", help="Save composite views PDF for a point cloud."
    )
    p_views.add_argument("pcd_filename", type=str, help="Input point cloud (PLY).")
    p_views.add_argument("output_pdf", type=str, help="Output PDF filepath.")
    p_views.add_argument(
        "--full",
        action="store_true",
        help="Load full point cloud without decimation (may be large).",
    )
    p_views.add_argument(
        "--auto-orient",
        dest="auto_orient",
        action="store_true",
        help="Auto-orient the point cloud prior to saving (scale/up/offset skipped).",
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
        "--cams-xml",
        dest="cams_xml",
        type=str,
        default=None,
        help=(
            "Path to .cams.xml file. Default: <cwd_basename>.cams.xml in current folder."
        ),
    )
    p_ff.add_argument(
        "--cams-meta",
        dest="cams_meta",
        type=str,
        default=None,
        help=(
            "Path to cameras metadata .meta.json file. Default: <cwd_basename>.meta.json."
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
        "--pcd-filename",
        dest="pcd_filename",
        type=str,
        default=None,
        help=(
            "Optional point cloud (PLY) filepath to load and use during up-vector determination."
        ),
    )
    p_ff.add_argument(
        "--camspath-postfix",
        dest="camspath_postfix",
        type=str,
        default=".photos",
        help=("Optional filter for cameras by filepath postfix (e.g., '.photos')."),
    )

    args = parser.parse_args()

    if args.command == "decimate":
        decimate_ply_file(
            input_path=args.input,
            output_path=args.output,
            target_points=args.target,
            show_progress=not args.no_progress,
        )
    elif args.command == "scalebars":
        # 1) load PCD (optionally streaming-decimate on load)
        pcd = PointCloud(args.pcd_filename, max_points=args.max_points)

        # 2) load markers as annotations
        anns = Annotations()
        anns.get_annotations_from_file(
            args.markers_filename, header=True, orig_coords_only=False
        )

        # 3) create Scalebars, attach target coords from annotations
        # Expect the CSV to provide labels matching scalebar target1/target2 labels
        sb = Scalebars(
            scalebar_data=settings.RGL_SCALEBARS, target_data=anns
        )  # scalebar_data populated via target_data
        # If your scalebar_data must come from a file, replace the above with your loader.

        # 4) save PDF
        sb.save_pdf(pcd, filepath=args.output_pdf)
    elif args.command == "views":
        # Load point cloud with optional streaming decimation to ~50M points
        max_pts = None if args.full else 50_000_000
        pcd = PointCloud(args.pcd_filename, max_points=max_pts)
        # Optionally run auto-orientation with default/None params (skips scale/up/offset)
        if getattr(args, "auto_orient", False):
            pcd.apply_orientation_transforms(None, None, None)
        # Save composite views PDF
        pcd.save_pdf(filepath=args.output_pdf)
    elif args.command == "firefish":
        ### TODO: MAKE GENERIC
        # Build defaults from current directory name
        cwd = os.getcwd()
        base = os.path.basename(cwd.rstrip(os.sep))

        firefish_file = args.firefish_file or os.path.join(cwd, f"{base}_firefish.txt")
        cams_xml = args.cams_xml or os.path.join(cwd, f"{base}.cams.xml")
        cams_meta = args.cams_meta or os.path.join(cwd, f"{base}.meta.json")

        # Infer target depth if not provided: find _<int>m_ pattern
        target_depth = args.target_depth
        if target_depth is None:
            m = re.search(r"_(\d+)m_", base)
            if m:
                try:
                    target_depth = int(m.group(1))
                except Exception:
                    target_depth = None

        pdf_output = args.pdf_output or os.path.join(cwd, f"{base}_firefish.pdf")
        cam_depths_file = args.cam_depths_file or os.path.join(
            cwd, f"{base}_camdepths.csv"
        )

        # Lazy import to avoid overhead when unused
        from substrata.firefish import FireFish
        from substrata.cameras import Cameras

        ff = FireFish(firefish_file)
        cams = Cameras(cams_meta_filepath=cams_meta, cams_xml_filepath=cams_xml)

        # depth_outlier_thresh is passed through only if provided; otherwise rely on default
        kwargs = {
            "camdepths_filepath": cam_depths_file,
            "pdf_output_filepath": pdf_output,
        }
        if args.depth_outlier_thresh is not None:
            kwargs["depth_and_outlier_threshold"] = args.depth_outlier_thresh

        # Load point cloud (default to <cwd_basename>.ply in current folder)
        pcd_filename = args.pcd_filename or os.path.join(cwd, f"{base}.ply")
        pcd = PointCloud(pcd_filename, max_points=50000000)

        # Run up-vector determination
        ff.determine_up_vector(
            cams,
            target_depth,
            pcd,
            cams_filepath_postfix_filter=args.camspath_postfix,
            **kwargs,
        )
    elif args.command == "head":
        ply_head(args.pcd_filename, n=args.num, print_output=True)


if __name__ == "__main__":
    main()
