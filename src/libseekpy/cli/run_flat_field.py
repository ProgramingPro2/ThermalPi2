#!/usr/bin/env python3
"""
Create flat-field calibration image for Seek Compact XR.
Cover the lens with uniform temperature, then run. Output can be passed to
viewer/snapshot with -F/--ffc.
"""

import argparse
import sys

from libseekpy.flat_field import create_flat_field


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create flat-field calibration image for Seek Compact XR"
    )
    parser.add_argument(
        "-o", "--output",
        default="flat_field.png",
        help="Output file path (default: flat_field.png)",
    )
    parser.add_argument(
        "-s", "--smoothing",
        type=int,
        default=100,
        help="Number of frames to average (default: 100)",
    )
    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=10,
        help="Number of frames to discard before averaging (default: 10)",
    )
    parser.add_argument(
        "-F", "--ffc",
        default=None,
        help="Optional existing FFC file to load before capture",
    )
    args = parser.parse_args()

    try:
        create_flat_field(
            output_path=args.output,
            warmup=args.warmup,
            smoothing=args.smoothing,
            ffc_path=args.ffc,
        )
        print(f"Saved flat-field calibration to {args.output}")
        return 0
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
