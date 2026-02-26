#!/usr/bin/env python3
"""
Continuous thermal viewer for Seek Compact XR.
Displays live feed using matplotlib (no GTK/OpenCV GUI required).
's' freeze, 'q' quit. Use -F to test flat-field calibration.
"""

import argparse
import sys
import time

import numpy as np

from libseekpy import SeekCompactXR, SeekCompactXRError
from libseekpy.constants import THERMAL_HEIGHT, THERMAL_WIDTH
from libseekpy.image_utils import convert_to_greyscale, normalize_to_uint8


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seek Compact XR live viewer. Use -F flat_field.png to test FFC.",
    )
    parser.add_argument(
        "-F", "--ffc",
        default=None,
        help="Path to flat-field calibration PNG (test FFC by creating with run_flat_field then pass here)",
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="Matplotlib colormap name (default: inferno). Use 'gray' for grayscale.",
    )
    parser.add_argument(
        "--no-colormap",
        action="store_true",
        help="Show grayscale (same as --colormap gray)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0,
        help="Optional seconds to warm up before showing (default: 0)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Contrast: lower bound of 16-bit range mapped to black (default: auto)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Contrast: upper bound of 16-bit range mapped to white (default: auto)",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Use histogram enhancement instead of min-max normalize",
    )
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for the viewer. Install with: pip install matplotlib")
        return 1

    cam = SeekCompactXR(ffc_filename=args.ffc)
    try:
        if not cam.open():
            print("Failed to open Seek Compact XR camera")
            return 1
    except SeekCompactXRError as e:
        print(f"Error: {e}")
        return 1

    cmap_name = "gray" if args.no_colormap else args.colormap
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Unknown colormap '{cmap_name}'. Use e.g. inferno, viridis, plasma, gray.")
        cam.close()
        return 1

    frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint16)
    if args.warmup > 0:
        print(f"Warming up for {args.warmup} seconds...")
        start = time.monotonic()
        while time.monotonic() - start < args.warmup:
            cam.read(frame)
            time.sleep(0.05)
        print("Starting display.")

    freeze = [False]
    quit_flag = [False]

    def on_key(event):
        if event.key == "q":
            quit_flag[0] = True
        elif event.key == "s":
            freeze[0] = not freeze[0]

    fig, ax = plt.subplots(num="Seek Compact XR")
    fig.canvas.mpl_connect("key_press_event", on_key)
    im = ax.imshow(
        np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8),
        cmap=cmap,
        vmin=0,
        vmax=255,
    )
    ax.set_axis_off()
    plt.ion()
    plt.show(block=False)

    try:
        read_retries = 5
        while not quit_flag[0]:
            if not freeze[0]:
                ok = False
                for _ in range(read_retries):
                    if cam.read(frame):
                        ok = True
                        break
                    time.sleep(0.1)
                if not ok:
                    print("Frame acquisition failed")
                    break
                if args.hist:
                    grey = convert_to_greyscale(frame)
                else:
                    grey = normalize_to_uint8(frame, vmin=args.vmin, vmax=args.vmax)
                im.set_data(grey)
                im.set_clim(0, 255)
            fig.canvas.draw_idle()
            plt.pause(0.01)
    finally:
        cam.close()
        plt.ioff()
        plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
