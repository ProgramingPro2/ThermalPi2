#!/usr/bin/env python3
"""
Continuous thermal viewer for Seek Compact XR.
Displays live feed using matplotlib (no GTK/OpenCV GUI required).
's' freeze, 'q' quit. Use -F to test flat-field calibration.
"""

import argparse
import sys
import time
from typing import Optional

import numpy as np

from libseekpy import SeekCompactXR, SeekCompactXRError
from libseekpy.constants import THERMAL_HEIGHT, THERMAL_WIDTH
from libseekpy.image_utils import (
    convert_to_greyscale,
    get_percentile_range,
    normalize_to_uint8,
    otsu_threshold,
    remove_bg_mask,
)


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
        "--range-mode",
        choices=("auto", "fixed", "smoothed", "percentile", "mean_centered"),
        default=None,
        help="Contrast range: auto, fixed, smoothed, percentile, or mean_centered=fixed span centered on smoothed mean (default: auto unless both --vmin and --vmax set)",
    )
    parser.add_argument(
        "--span",
        type=float,
        default=600,
        metavar="N",
        help="Fixed span in 16-bit counts for --range-mode mean_centered (center +/- span/2); default 600",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.1,
        metavar="A",
        help="EMA weight for smoothed range (0 < A <= 1); smaller = smoother (default: 0.1)",
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
        "--hist",
        action="store_true",
        help="Use histogram enhancement instead of min-max normalize",
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
        help="Threshold for --remove-bg: percentile=top (100-P)%%, otsu=bimodal, fixed=--remove-bg-thresh (default: otsu)",
    )
    parser.add_argument(
        "--remove-bg-thresh",
        type=float,
        default=None,
        metavar="N",
        help="Fixed 16-bit threshold for --remove-bg when --remove-bg-mode fixed",
    )
    parser.add_argument(
        "--remove-bg-percentile",
        type=float,
        default=50.0,
        metavar="P",
        help="Percentile threshold for --remove-bg when mode=percentile (keep pixels above P%%; default 50)",
    )
    parser.add_argument(
        "--remove-bg-blobs",
        type=int,
        default=1,
        metavar="N",
        help="Keep only largest N connected heat blobs (0=no filter; default 1)",
    )
    parser.add_argument(
        "--remove-bg-smooth",
        action="store_true",
        help="Smooth remove-bg threshold over time (EMA) in viewer",
    )
    args = parser.parse_args()

    # Resolve range mode: explicit or infer from vmin/vmax
    range_mode = args.range_mode
    if range_mode is None:
        range_mode = "fixed" if (args.vmin is not None and args.vmax is not None) else "auto"
    if range_mode == "fixed" and (args.vmin is None or args.vmax is None):
        print("Error: --range-mode fixed requires both --vmin and --vmax")
        return 1
    if range_mode == "smoothed" and not (0 < args.smooth_alpha <= 1):
        print("Error: --smooth-alpha must be in (0, 1]")
        return 1
    if range_mode == "mean_centered" and (args.span <= 0 or not (0 < args.smooth_alpha <= 1)):
        print("Error: --range-mode mean_centered requires --span > 0 and --smooth-alpha in (0, 1]")
        return 1
    if args.remove_bg and args.remove_bg_mode == "fixed" and args.remove_bg_thresh is None:
        print("Error: --remove-bg-mode fixed requires --remove-bg-thresh N")
        return 1
    if args.remove_bg and args.remove_bg_blobs < 0:
        print("Error: --remove-bg-blobs must be >= 0")
        return 1

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

    smoothed_min: Optional[float] = None
    smoothed_max: Optional[float] = None
    smoothed_mean: Optional[float] = None
    smoothed_remove_bg_thresh: Optional[float] = None
    last_debug_time = time.monotonic()
    DEBUG_INTERVAL = 10.0

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
                    if range_mode == "auto":
                        vmin, vmax = None, None
                    elif range_mode == "fixed":
                        vmin, vmax = args.vmin, args.vmax
                    elif range_mode == "smoothed":
                        fmin, fmax = float(np.min(frame)), float(np.max(frame))
                        if smoothed_min is None:
                            smoothed_min, smoothed_max = fmin, fmax
                        else:
                            alpha = args.smooth_alpha
                            smoothed_min = alpha * fmin + (1 - alpha) * smoothed_min
                            smoothed_max = alpha * fmax + (1 - alpha) * smoothed_max
                        vmin, vmax = smoothed_min, smoothed_max
                    elif range_mode == "mean_centered":
                        fmean = float(np.mean(frame))
                        if smoothed_mean is None:
                            smoothed_mean = fmean
                        else:
                            alpha = args.smooth_alpha
                            smoothed_mean = alpha * fmean + (1 - alpha) * smoothed_mean
                        half = args.span / 2.0
                        vmin = smoothed_mean - half
                        vmax = smoothed_mean + half
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
                    if args.remove_bg_smooth:
                        if smoothed_remove_bg_thresh is None:
                            smoothed_remove_bg_thresh = thresh
                        else:
                            smoothed_remove_bg_thresh = (
                                args.smooth_alpha * thresh
                                + (1 - args.smooth_alpha) * smoothed_remove_bg_thresh
                            )
                        thresh = smoothed_remove_bg_thresh
                    keep_blobs = args.remove_bg_blobs if args.remove_bg_blobs > 0 else None
                    mask = remove_bg_mask(frame, thresh, keep_top_n_blobs=keep_blobs)
                    grey[~mask] = 0

                now = time.monotonic()
                if now - last_debug_time >= DEBUG_INTERVAL:
                    fmin, fmax = int(np.min(frame)), int(np.max(frame))
                    fmean = float(np.mean(frame))
                    if vmin is not None and vmax is not None:
                        print(f"[range] frame min={fmin} max={fmax} mean={fmean:.0f}  display range: {vmin:.0f} - {vmax:.0f}")
                    else:
                        print(f"[range] frame min={fmin} max={fmax} mean={fmean:.0f}  display range: auto (min-max)")
                    last_debug_time = now

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
