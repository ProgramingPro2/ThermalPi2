"""
libseekpy: Python driver for Seek Thermal Compact / CompactXR (product id 0x0010).
Based on the protocol from OpenThermal/libseek-thermal.
"""

from .constants import (
    SEEK_VENDOR_ID,
    SEEK_COMPACT_XR_PRODUCT_ID,
    DeviceCommand,
    THERMAL_WIDTH,
    THERMAL_HEIGHT,
    THERMAL_RAW_WIDTH,
    THERMAL_RAW_HEIGHT,
    THERMAL_OFFSET,
)
from .device import SeekDevice, SeekDeviceError
from .seek_compact_xr import SeekCompactXR, SeekCompactXRError
from .flat_field import create_flat_field
from .image_utils import (
    convert_to_greyscale,
    get_percentile_range,
    normalize_to_uint8,
    apply_colormap,
    otsu_threshold,
    remove_bg_mask,
)

__all__ = [
    "SeekDevice",
    "SeekDeviceError",
    "SeekCompactXR",
    "SeekCompactXRError",
    "create_flat_field",
    "convert_to_greyscale",
    "get_percentile_range",
    "normalize_to_uint8",
    "apply_colormap",
    "otsu_threshold",
    "remove_bg_mask",
    "DeviceCommand",
    "SEEK_VENDOR_ID",
    "SEEK_COMPACT_XR_PRODUCT_ID",
    "THERMAL_WIDTH",
    "THERMAL_HEIGHT",
    "THERMAL_RAW_WIDTH",
    "THERMAL_RAW_HEIGHT",
    "THERMAL_OFFSET",
]
