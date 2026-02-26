"""
Image utilities: 14-bit to 8-bit grayscale conversion (histogram enhancement),
optional colormap for display.
"""

from typing import Optional

import numpy as np

NUM_BINS = 10


def convert_to_greyscale(arr_uint16: np.ndarray) -> np.ndarray:
    """
    Convert 14-bit thermal frame to 8-bit grayscale using histogram-based
    enhancement (10 bins, cumulative scaling). Matches SeekCam::convertToGreyScale.
    arr_uint16: 2D uint16 array (e.g. 207x154).
    Returns: 2D uint8 array same shape.
    """
    src = np.asarray(arr_uint16, dtype=np.float64)
    tmin = float(np.min(src))
    tmax = float(np.max(src))
    num_pixels = src.size
    rsize = (tmax - tmin) / NUM_BINS
    dst = np.zeros_like(src, dtype=np.float64)
    rnstart = 0.0
    for n in range(NUM_BINS):
        low = tmin + n * rsize
        high = low + rsize
        mask = (src >= low) & (src < high)
        count = np.count_nonzero(mask)
        if count == 0:
            continue
        rnint = (count << 8) / num_pixels
        dst[mask] = (src[mask] - low) * rnint / rsize + rnstart
        rnstart += rnint
    return np.clip(dst, 0, 255).astype(np.uint8)


def apply_colormap(grey_8bit: np.ndarray, colormap: str = "COLORMAP_INFERNO") -> np.ndarray:
    """
    Apply OpenCV colormap to 8-bit grayscale image for display.
    Requires opencv-python. Returns BGR image (e.g. for imshow/imwrite).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for apply_colormap") from None
    cmap = getattr(cv2, colormap, cv2.COLORMAP_INFERNO)
    return cv2.applyColorMap(grey_8bit, cmap)


def normalize_to_uint8(
    arr_uint16: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize 16-bit to 8-bit. If vmin/vmax are given, use them as the range;
    otherwise use min/max of the array (auto contrast).
    """
    a = np.asarray(arr_uint16, dtype=np.float64)
    lo = float(np.min(a)) if vmin is None else vmin
    hi = float(np.max(a)) if vmax is None else vmax
    if hi <= lo:
        return np.zeros_like(arr_uint16, dtype=np.uint8)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
