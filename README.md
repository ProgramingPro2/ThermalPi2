# libseekpy: Seek Compact XR Python driver

Python port of the [OpenThermal/libseek-thermal](https://github.com/OpenThermal/libseek-thermal) driver for the **Seek Thermal Compact** and **Seek Thermal CompactXR** (USB product id `0x0010`). Same protocol for both cameras.

## Features

- Open/close camera, full init sequence
- Grab and retrieve 207x154 thermal frames (14-bit)
- Dead-pixel detection and correction (from first drift-cal frame)
- Flat-field calibration (FFC) and optional additional flat-field from file
- Create flat-field calibration image (cover lens, run, save PNG)
- Grayscale conversion (histogram-based) and optional colormap for display
- CLI: live viewer, snapshot, flat-field creation

## Requirements

- Linux (udev rule for non-root USB access) or Windows (e.g. Zadig libusb-win32 for the iAP Interface)
- Python 3.8+
- libusb-1.0 (e.g. `libusb-1.0-0-dev` on Debian/Ubuntu)
- PyUSB, NumPy; matplotlib for the live viewer (no GTK/OpenCV GUI); OpenCV and Pillow for snapshot and 16-bit PNG

## Install

From project root, install the package in editable mode so the CLI runs without `PYTHONPATH`:

```bash
pip install -e .
```

Or install dependencies only (then use `PYTHONPATH=src` when running CLI):

```bash
pip install -r requirements.txt
```

Ensure libusb-1.0 is installed:

```bash
# Debian/Ubuntu
sudo apt-get install libusb-1.0-0-dev
```

## USB access (Linux)

Add a udev rule so the camera can be used without root:

```bash
# Example rule in docs/udev.rules.example
SUBSYSTEM=="usb", ATTRS{idVendor}=="289d", ATTRS{idProduct}=="0010", MODE="0666", GROUP="users"
```

Copy to `/etc/udev/rules.d/` (e.g. `99-seek-thermal.rules`), then:

```bash
sudo udevadm control --reload-rules
# Unplug and replug the camera
```

## Usage from Python

Set `PYTHONPATH` to include `src` so the package is found (e.g. from project root: `PYTHONPATH=src`), or install the project in editable mode.

```python
from libseekpy import SeekCompactXR
from libseekpy.constants import THERMAL_HEIGHT, THERMAL_WIDTH
import numpy as np

cam = SeekCompactXR(ffc_filename="flat_field.png")  # optional FFC
cam.open()
frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint16)
cam.read(frame)
# use frame (207x154 uint16)
cam.close()
```

## CLI scripts

From project root after `pip install -e .` (or with `PYTHONPATH=src`):

- **Viewer:** `python -m libseekpy.cli.run_viewer`  
  Uses matplotlib (no GTK/OpenCV GUI required). No warmup by default; optional `--warmup N`.  
  Contrast: `--range-mode auto|fixed|smoothed|percentile|mean_centered` (default: auto, or fixed if both `--vmin` and `--vmax` set).  
  `mean_centered` = fixed span (default 600) centered on smoothed frame mean; use `--span N`, `--smooth-alpha`.  
  Options: `-F flat_field.png` (test FFC), `--colormap inferno`, `--no-colormap`, `--hist`, `--remove-bg` (keep only top 25%% heat). Keys: `s` freeze, `q` quit.

- **Snapshot:** `python -m libseekpy.cli.run_snapshot [output.png]`  
  Options: `-n count`, `-F flat_field.png` (test FFC), `--range-mode auto|fixed|percentile`, `--vmin`/`--vmax` (for fixed), `--percentile LOW HIGH` (for percentile), `--hist`, `--remove-bg`, `--16bit`, `--colormap COLORMAP_INFERNO`.

- **Debug capture:** `python -m libseekpy.cli.run_debug_capture`  
  Saves debug_thermal.png, debug_thermal_grey.png, debug_thermal_16bit.png. Optional `--warmup N`, `-F flat_field.png`, `--vmin` / `--vmax`.

- **Flat-field creation:** `python -m libseekpy.cli.run_flat_field`  
  Cover the lens with uniform temperature first.  
  Options: `-o flat_field.png`, `-s 100`, `-w 10`, `-F existing_ffc.png`.

## Contrast range modes (viewer and snapshot)

Display contrast is set by which 16-bit range is mapped to black/white. By default the viewer uses the **brightest and darkest pixels in each frame** (auto), which can flicker when the scene changes. You can switch to other modes:

| Mode | When to use | How to use |
|------|-------------|------------|
| **auto** | Default; follow scene min/max every frame (may flicker). | Omit `--range-mode` and do not set both `--vmin` and `--vmax`. |
| **fixed** | Stable scale when you know the 16-bit range (e.g. room temp). | `--vmin N --vmax M` or `--range-mode fixed --vmin N --vmax M`. |
| **smoothed** | Stable scale that slowly follows the scene (viewer only). | `--range-mode smoothed`. Optional: `--smooth-alpha 0.05` (smaller = smoother, default 0.1). |
| **percentile** | Ignore hot/cold outliers (e.g. single hot pixel). | `--range-mode percentile`. Optional: `--percentile 2 98` (default). |
| **mean_centered** | Fixed contrast span; center follows smoothed frame mean (compensates camera drift). | `--range-mode mean_centered`. Optional: `--span 600` (default), `--smooth-alpha 0.1`. |

**Viewer examples:**

```bash
# Auto (default): range = min/max of each frame
python -m libseekpy.cli.run_viewer

# Smoothed: range follows scene slowly, less flicker
python -m libseekpy.cli.run_viewer --range-mode smoothed

# Smoothed with stronger smoothing (slower to react)
python -m libseekpy.cli.run_viewer --range-mode smoothed --smooth-alpha 0.05

# Percentile: 2nd–98th percentile of each frame (dampens outliers)
python -m libseekpy.cli.run_viewer --range-mode percentile

# Fixed: always map 16-bit 10000–20000 to black–white
python -m libseekpy.cli.run_viewer --vmin 10000 --vmax 20000

# Mean-centered: span 600 centered on smoothed mean (stable contrast, tracks camera heating)
python -m libseekpy.cli.run_viewer --range-mode mean_centered
```

**Snapshot examples:**

```bash
# Auto (default)
python -m libseekpy.cli.run_snapshot out.png

# Percentile for one frame
python -m libseekpy.cli.run_snapshot --range-mode percentile out.png

# Fixed range
python -m libseekpy.cli.run_snapshot --vmin 10000 --vmax 20000 out.png
```

Snapshot does not support `smoothed` (single-frame capture). Use `--hist` for histogram-based enhancement instead of min/max or percentile.

## Flat-field calibration (test FFC)

1. Cover the lens with an object of uniform temperature (e.g. lens cap or uniform surface).
2. Run: `python -m libseekpy.cli.run_flat_field -o flat_field.png`
3. Test FFC with viewer or snapshot:  
   `python -m libseekpy.cli.run_viewer -F flat_field.png`  
   or  
   `python -m libseekpy.cli.run_snapshot -F flat_field.png out.png`

Calibration is temperature-sensitive; redo periodically or after long idle.

## Package layout

- `libseekpy.constants`: USB IDs, device commands, frame dimensions
- `libseekpy.device`: SeekDevice (open/close, request_set/request_get, fetch_frame)
- `libseekpy.seek_compact_xr`: SeekCompactXR (open, grab, retrieve, read, dead pixels, FFC)
- `libseekpy.flat_field`: create_flat_field()
- `libseekpy.image_utils`: convert_to_greyscale, get_percentile_range, normalize_to_uint8, apply_colormap

## Limitations

- No absolute temperature readings (same as libseek-thermal).
- Seek CompactPRO (0x0011) and Seek Nano are not supported by this package.
