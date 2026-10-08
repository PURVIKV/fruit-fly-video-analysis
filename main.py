"""
Real-Time Detailed Video Analysis of Fruit Flies

Main entry point for the project.

    python main.py                      # full test1.mp4 with preview window
    python main.py --max-frames 50      # quick smoke test
    python main.py --no-display         # headless
"""

import argparse

from src.pipeline import VIDEO_PATH, main as run_pipeline


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the fruit-fly analysis pipeline on a video."
    )

    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help=f"input video (default: {VIDEO_PATH})"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many frames (quick smoke test)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open the preview window"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_pipeline(
        video_path=args.video,
        max_frames=args.max_frames,
        display=not args.no_display
    )
