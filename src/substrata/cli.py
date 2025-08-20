# cli.py
import argparse
from substrata import pointclouds


def main():
    parser = argparse.ArgumentParser(description="Substrata CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # decimate: stream downsample a binary PLY using reservoir sampling
    p_dec = subparsers.add_parser(
        "decimate", help="Decimate a binary PLY to a target number of points."
    )
    p_dec.add_argument(
        "input",
        type=str,
        help="Path to input binary PLY file.",
    )
    p_dec.add_argument(
        "output",
        type=str,
        help="Path to output PLY file.",
    )
    p_dec.add_argument(
        "target",
        type=int,
        help="Number of points to keep.",
    )
    p_dec.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    args = parser.parse_args()

    if args.command == "decimate":
        pointclouds.decimate_ply_file(
            input_path=args.input,
            output_path=args.output,
            target_points=args.target,
            show_progress=not args.no_progress,
        )


if __name__ == "__main__":
    main()
