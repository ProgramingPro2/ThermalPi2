#!/usr/bin/env python3
"""
Debug capture: optional warmup, then save one frame to PNG.
Saves debug_thermal.png (8-bit normalized), debug_thermal_grey.png, debug_thermal_16bit.png in project root.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from libseekpy import SeekCompactXR, SeekCompactXRError
from libseekpy.constants import THERMAL_HEIGHT, THERMAL_WIDTH
from libseekpy.image_utils import convert_to_greyscale, normalize_to_uint8


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one frame to debug PNGs")
    parser.add_argument("-F", "--ffc", default=None, help="Path to flat-field calibration PNG (test FFC)")
    parser.add_argument("--warmup", type=float, default=0, help="Optional seconds to warm up (default: 0)")
    parser.add_argument("--vmin", type=float, default=None, help="Contrast: lower 16-bit bound (default: auto)")
    parser.add_argument("--vmax", type=float, default=None, help="Contrast: upper 16-bit bound (default: auto)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    png_path = project_root / "debug_thermal.png"
    png_16_path = project_root / "debug_thermal_16bit.png"
    grey_path = project_root / "debug_thermal_grey.png"

    cam = SeekCompactXR(ffc_filename=args.ffc)
    try:
        if not cam.open():
            print("Failed to open Seek Compact XR camera")
            return 1
    except SeekCompactXRError as e:
        print(f"Error: {e}")
        return 1

    frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint16)
    if args.warmup > 0:
        print(f"Warming up for {args.warmup} seconds...")
        start = time.monotonic()
        while time.monotonic() - start < args.warmup:
            cam.read(frame)
            time.sleep(0.05)
    if not cam.read(frame):
        print("Frame acquisition failed")
        cam.close()
        return 1

    norm8 = normalize_to_uint8(frame, vmin=args.vmin, vmax=args.vmax)
    grey8 = convert_to_greyscale(frame)

    try:
        import cv2
        cv2.imwrite(str(png_path), norm8)
        cv2.imwrite(str(grey_path), grey8)
        cv2.imwrite(str(png_16_path), frame)
    except ImportError:
        from PIL import Image
        Image.fromarray(norm8).save(str(png_path))
        Image.fromarray(grey8).save(str(grey_path))
        Image.fromarray(frame).save(str(png_16_path))

    cam.close()
    print(f"Saved normalized 8-bit: {png_path}")
    print(f"Saved histogram grey 8-bit: {grey_path}")
    print(f"Saved 16-bit: {png_16_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
