"""
Image utilities: 14-bit to 8-bit grayscale conversion (histogram enhancement),
optional colormap for display, background-removal (Otsu, percentile, blob filter).
"""

from typing import Optional, Tuple

import numpy as np

NUM_BINS = 10
OTSU_BINS = 256


def otsu_threshold(arr_uint16: np.ndarray) -> float:
    """
    Otsu's method: threshold that best separates two classes (e.g. background vs
    foreground). Uses 256-bin histogram. Returns threshold in 16-bit value range.
    """
    a = np.asarray(arr_uint16, dtype=np.float64).ravel()
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return lo
    hist, bin_edges = np.histogram(a, bins=OTSU_BINS, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return lo
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cumsum = np.cumsum(hist)
    sum_weighted = np.cumsum(hist * bin_centers)
    best_var = -1.0
    best_t = 0
    for t in range(1, OTSU_BINS):
        w_b = cumsum[t - 1] / total
        w_f = 1.0 - w_b
        if w_b <= 0 or w_f <= 0:
            continue
        m_b = sum_weighted[t - 1] / cumsum[t - 1]
        m_f = (sum_weighted[-1] - sum_weighted[t - 1]) / (total - cumsum[t - 1])
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return float(bin_edges[best_t])


def remove_bg_mask(
    frame_uint16: np.ndarray,
    threshold: float,
    keep_top_n_blobs: Optional[int] = None,
) -> np.ndarray:
    """
    Binary mask: True where frame >= threshold. If keep_top_n_blobs is set,
    keep only the largest N connected components (by area); rest become False.
    """
    mask = (np.asarray(frame_uint16, dtype=np.float64) >= threshold).astype(np.uint8)
    if keep_top_n_blobs is None or keep_top_n_blobs <= 0:
        return mask.astype(bool)
    try:
        import cv2
    except ImportError:
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask * 255, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    top_indices = np.argsort(areas)[::-1][:keep_top_n_blobs]
    keep_labels = set((i + 1 for i in top_indices))
    out = np.zeros_like(mask, dtype=bool)
    for lab in keep_labels:
        out[labels == lab] = True
    return out


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


def get_percentile_range(
    arr_uint16: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> Tuple[float, float]:
    """
    Return (vmin, vmax) from percentile of pixel values. Reduces impact of
    outliers/hot pixels. For use with normalize_to_uint8.
    """
    a = np.asarray(arr_uint16, dtype=np.float64).ravel()
    vmin = float(np.percentile(a, low_percentile))
    vmax = float(np.percentile(a, high_percentile))
    return (vmin, vmax)


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
