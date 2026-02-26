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
from libseekpy.image_utils import (
    convert_to_greyscale,
    get_percentile_range,
    normalize_to_uint8,
    apply_colormap,
    otsu_threshold,
    remove_bg_mask,
)


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
        "--range-mode",
        choices=("auto", "fixed", "percentile"),
        default=None,
        help="Contrast range for 8-bit: auto=min/max, fixed=--vmin/--vmax, percentile (default: auto unless both --vmin and --vmax set)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        nargs=2,
        default=[2.0, 98.0],
        metavar=("LOW", "HIGH"),
        help="Percentile range for --range-mode percentile (default: 2 98)",
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
    parser.add_argument(
        "--remove-bg",
        action="store_true",
        help="Remove background: keep only heat above threshold (see --remove-bg-mode), rest black",
    )
    parser.add_argument(
        "--remove-bg-mode",
        choices=("percentile", "otsu", "fixed"),
        default="otsu",
        help="Threshold for --remove-bg: percentile, otsu, or fixed (default: otsu)",
    )
    parser.add_argument(
        "--remove-bg-thresh",
        type=float,
        default=None,
        metavar="N",
        help="Fixed 16-bit threshold when --remove-bg-mode fixed",
    )
    parser.add_argument(
        "--remove-bg-percentile",
        type=float,
        default=50.0,
        metavar="P",
        help="Percentile for --remove-bg when mode=percentile (default 50)",
    )
    parser.add_argument(
        "--remove-bg-blobs",
        type=int,
        default=1,
        metavar="N",
        help="Keep only largest N connected blobs (0=no filter; default 1)",
    )
    args = parser.parse_args()

    range_mode = args.range_mode
    if range_mode is None:
        range_mode = "fixed" if (args.vmin is not None and args.vmax is not None) else "auto"
    if range_mode == "fixed" and (args.vmin is None or args.vmax is None):
        print("Error: --range-mode fixed requires both --vmin and --vmax")
        return 1
    if args.remove_bg and args.remove_bg_mode == "fixed" and args.remove_bg_thresh is None:
        print("Error: --remove-bg-mode fixed requires --remove-bg-thresh N")
        return 1
    if args.remove_bg and args.remove_bg_blobs < 0:
        print("Error: --remove-bg-blobs must be >= 0")
        return 1

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
                if range_mode == "auto":
                    vmin, vmax = None, None
                elif range_mode == "fixed":
                    vmin, vmax = args.vmin, args.vmax
                else:
                    vmin, vmax = get_percentile_range(
                        frame,
                        low_percentile=args.percentile[0],
                        high_percentile=args.percentile[1],
                    )
                grey = normalize_to_uint8(frame, vmin=vmin, vmax=vmax)
            if args.remove_bg:
                if args.remove_bg_mode == "fixed":
                    thresh = float(args.remove_bg_thresh)
                elif args.remove_bg_mode == "percentile":
                    thresh = float(np.percentile(frame, args.remove_bg_percentile))
                else:
                    thresh = otsu_threshold(frame)
                keep_blobs = args.remove_bg_blobs if args.remove_bg_blobs > 0 else None
                mask = remove_bg_mask(frame, thresh, keep_top_n_blobs=keep_blobs)
                grey[~mask] = 0
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
