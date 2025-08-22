# cli.py
import argparse
from substrata.pointclouds import PointCloud, decimate_ply_file
from substrata.annotations import Annotations, Scalebars
from substrata import settings


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


if __name__ == "__main__":
    main()
