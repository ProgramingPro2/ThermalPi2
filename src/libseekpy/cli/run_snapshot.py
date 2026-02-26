#!/usr/bin/env python3
"""
Capture one or more frames from Seek Compact XR and save to file(s).
Saves 16-bit PNG (raw thermal) and optionally 8-bit normalized or colormap.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from libseekpy import SeekCompactXR, SeekCompactXRError
from libseekpy.constants import THERMAL_HEIGHT, THERMAL_WIDTH
from libseekpy.image_utils import convert_to_greyscale, normalize_to_uint8, apply_colormap


def main() -> int:
    parser = argparse.ArgumentParser(description="Seek Compact XR snapshot")
    parser.add_argument(
        "output",
        nargs="?",
        default="seek_frame.png",
        help="Output path (default: seek_frame.png)",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=1,
        help="Number of frames to capture (default: 1)",
    )
    parser.add_argument(
        "-F", "--ffc",
        default=None,
        help="Path to flat-field calibration PNG (test FFC after creating with run_flat_field)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Contrast: lower 16-bit bound for 8-bit output (default: auto)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Contrast: upper 16-bit bound for 8-bit output (default: auto)",
    )
    parser.add_argument(
        "--16bit",
        action="store_true",
        dest="save_16bit",
        help="Save 16-bit PNG in addition to 8-bit",
    )
    parser.add_argument(
        "--colormap",
        default=None,
        help="OpenCV colormap name for display image (e.g. COLORMAP_INFERNO)",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Use histogram enhancement for 8-bit instead of min-max normalize",
    )
    args = parser.parse_args()

    cam = SeekCompactXR(ffc_filename=args.ffc)
    try:
        if not cam.open():
            print("Failed to open Seek Compact XR camera")
            return 1
    except SeekCompactXRError as e:
        print(f"Error: {e}")
        return 1

    out_path = Path(args.output)
    stem = out_path.stem
    suffix = out_path.suffix or ".png"
    parent = out_path.parent

    try:
        for i in range(args.count):
            frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint16)
            if not cam.read(frame):
                print("Frame acquisition failed")
                return 1
            if args.count == 1:
                path_8 = parent / (stem + suffix)
                path_16 = parent / (stem + "_16bit" + suffix)
            else:
                path_8 = parent / f"{stem}_{i:04d}{suffix}"
                path_16 = parent / f"{stem}_{i:04d}_16bit{suffix}"

            if args.hist:
                grey = convert_to_greyscale(frame)
            else:
                grey = normalize_to_uint8(frame, vmin=args.vmin, vmax=args.vmax)
            if args.colormap:
                try:
                    disp = apply_colormap(grey, args.colormap)
                    import cv2
                    cv2.imwrite(str(path_8), disp)
                except ImportError:
                    import cv2
                    cv2.imwrite(str(path_8), grey)
            else:
                try:
                    import cv2
                    cv2.imwrite(str(path_8), grey)
                except ImportError:
                    from PIL import Image
                    Image.fromarray(grey).save(str(path_8))
            print(f"Saved {path_8}")

            if args.save_16bit:
                try:
                    import cv2
                    cv2.imwrite(str(path_16), frame)
                except ImportError:
                    from PIL import Image
                    Image.fromarray(frame).save(str(path_16))
                print(f"Saved {path_16}")
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
