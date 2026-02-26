"""
Seek Compact XR camera driver (same protocol as Seek Thermal Compact).
Initialization, frame grab/retrieve, dead-pixel correction, optional flat-field calibration.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .constants import (
    DeviceCommand,
    SEEK_VENDOR_ID,
    SEEK_COMPACT_XR_PRODUCT_ID,
    THERMAL_RAW_WIDTH,
    THERMAL_RAW_HEIGHT,
    THERMAL_WIDTH,
    THERMAL_HEIGHT,
    THERMAL_RAW_SIZE,
    THERMAL_REQUEST_SIZE,
    THERMAL_OFFSET,
    ROI_Y,
    ROI_X,
)
from .device import SeekDevice, SeekDeviceError

DEAD_PIXEL_MARKER = 0xFFFF
HIST_SIZE = 0x4000


class SeekCompactXRError(Exception):
    """Raised when camera operations fail."""

    pass


class SeekCompactXR:
    """
    Seek Thermal Compact / CompactXR camera (product id 0x0010).
    Open, init, grab, retrieve, read; dead-pixel and optional flat-field correction.
    """

    def __init__(self, ffc_filename: Optional[str] = None) -> None:
        self._ffc_filename = ffc_filename
        self._dev = SeekDevice(SEEK_VENDOR_ID, SEEK_COMPACT_XR_PRODUCT_ID)
        self._raw_buffer = np.zeros(THERMAL_RAW_SIZE, dtype=np.uint16)
        self._is_opened = False
        self._flat_field_calibration: Optional[np.ndarray] = None
        self._additional_ffc: Optional[np.ndarray] = None
        self._dead_pixel_mask: Optional[np.ndarray] = None
        self._dead_pixel_list: List[Tuple[int, int]] = []
        self._roi_slice = (slice(ROI_Y, ROI_Y + THERMAL_HEIGHT), slice(ROI_X, ROI_X + THERMAL_WIDTH))

    def open(self) -> bool:
        """Open camera: load optional flat-field file, then open_cam."""
        if self._ffc_filename:
            path = Path(self._ffc_filename)
            if not path.exists():
                raise SeekCompactXRError(f"FFC file not found: {self._ffc_filename}")
            try:
                from PIL import Image
                img = np.array(Image.open(path))
                if img.shape != (THERMAL_HEIGHT, THERMAL_WIDTH):
                    raise SeekCompactXRError(
                        f"FFC image size must be {THERMAL_HEIGHT}x{THERMAL_WIDTH}, got {img.shape}"
                    )
                if img.dtype == np.uint16:
                    self._additional_ffc = img.copy()
                else:
                    self._additional_ffc = img.astype(np.uint16)
            except Exception as e:
                raise SeekCompactXRError(f"Failed to load FFC image: {e}") from e
        return self._open_cam()

    def close(self) -> None:
        """Set operation mode to 0 and close device."""
        if self._dev.is_opened():
            try:
                self._dev.request_set(DeviceCommand.SET_OPERATION_MODE, [0x00, 0x00])
                self._dev.request_set(DeviceCommand.SET_OPERATION_MODE, [0x00, 0x00])
                self._dev.request_set(DeviceCommand.SET_OPERATION_MODE, [0x00, 0x00])
            except SeekDeviceError:
                pass
            self._dev.close()
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened

    def _open_cam(self) -> bool:
        """Open device and run init with up to 3 retries."""
        for attempt in range(3):
            try:
                self._dev.open()
            except SeekDeviceError as e:
                raise SeekCompactXRError(f"Open failed: {e}") from e

            try:
                if not self._init_cam():
                    self._dev.close()
                    continue
                if not self._get_frame():
                    self._dev.close()
                    continue
                if self.frame_id() != 4:
                    self._dev.close()
                    continue
                roi_frame = self._raw_buffer.reshape(THERMAL_RAW_HEIGHT, THERMAL_RAW_WIDTH)[
                    self._roi_slice
                ].copy()
                self._create_dead_pixel_list(roi_frame)
                if not self.grab():
                    self._dev.close()
                    continue
                self._is_opened = True
                return True
            except (SeekDeviceError, SeekCompactXRError):
                self._dev.close()
                continue
        return False

    def _init_cam(self) -> bool:
        """Run exact init sequence from SeekThermal.cpp."""
        try:
            self._dev.request_set(DeviceCommand.TARGET_PLATFORM, [0x01])
        except SeekDeviceError:
            self.close()
            self._dev.open()
            self._dev.request_set(DeviceCommand.TARGET_PLATFORM, [0x01])

        steps = [
            (DeviceCommand.SET_OPERATION_MODE, [0x00, 0x00], None),
            (DeviceCommand.GET_FIRMWARE_INFO, None, 4),
            (DeviceCommand.READ_CHIP_ID, None, 12),
            (DeviceCommand.SET_FACTORY_SETTINGS_FEATURES, [0x20, 0x00, 0x30, 0x00, 0x00, 0x00], None),
            (DeviceCommand.GET_FACTORY_SETTINGS, None, 64),
            (DeviceCommand.SET_FACTORY_SETTINGS_FEATURES, [0x20, 0x00, 0x50, 0x00, 0x00, 0x00], None),
            (DeviceCommand.GET_FACTORY_SETTINGS, None, 64),
            (DeviceCommand.SET_FACTORY_SETTINGS_FEATURES, [0x0C, 0x00, 0x70, 0x00, 0x00, 0x00], None),
            (DeviceCommand.GET_FACTORY_SETTINGS, None, 24),
            (DeviceCommand.SET_FACTORY_SETTINGS_FEATURES, [0x06, 0x00, 0x08, 0x00, 0x00, 0x00], None),
            (DeviceCommand.GET_FACTORY_SETTINGS, None, 12),
            (DeviceCommand.SET_IMAGE_PROCESSING_MODE, [0x08, 0x00], None),
            (DeviceCommand.GET_OPERATION_MODE, None, 2),
            (DeviceCommand.SET_IMAGE_PROCESSING_MODE, [0x08, 0x00], None),
            (DeviceCommand.SET_OPERATION_MODE, [0x01, 0x00], None),
            (DeviceCommand.GET_OPERATION_MODE, None, 2),
        ]
        for item in steps:
            cmd, payload, get_len = item[0], item[1], item[2]
            if get_len is not None:
                self._dev.request_get(cmd, get_len)
            else:
                self._dev.request_set(cmd, payload)
        return True

    def _get_frame(self) -> bool:
        """Request one frame and read into internal buffer."""
        try:
            self._dev.request_set(
                DeviceCommand.START_GET_IMAGE_TRANSFER,
                list((THERMAL_RAW_SIZE).to_bytes(4, "little")),
            )
            self._dev.fetch_frame(
                self._raw_buffer,
                THERMAL_RAW_SIZE,
                THERMAL_REQUEST_SIZE,
            )
            return True
        except SeekDeviceError:
            return False

    def frame_id(self) -> int:
        """Frame type: 1=FFC, 3=normal, 4=drift cal, etc."""
        return int(self._raw_buffer[10])

    def frame_counter(self) -> int:
        """Frame counter from metadata."""
        return int(self._raw_buffer[40])

    def grab(self) -> bool:
        """Grab until we get a normal frame (frame_id==3); save FFC when frame_id==1."""
        for _ in range(120):
            if not self._get_frame():
                return False
            fid = self.frame_id()
            if fid == 3:
                return True
            if fid == 1:
                self._flat_field_calibration = self._raw_buffer.reshape(
                    THERMAL_RAW_HEIGHT, THERMAL_RAW_WIDTH
                )[self._roi_slice].copy()
            time.sleep(0.02)
        return False

    def retrieve(self, dst: np.ndarray) -> None:
        """
        Fill dst (207x154 uint16) with calibrated frame: offset - FFC, dead-pixel filter,
        then optional offset - additional_ffc.
        """
        raw_roi = self._raw_buffer.reshape(THERMAL_RAW_HEIGHT, THERMAL_RAW_WIDTH)[
            self._roi_slice
        ]
        if self._flat_field_calibration is None:
            cal = np.zeros_like(raw_roi, dtype=np.uint16)
        else:
            cal = self._flat_field_calibration
        # raw_roi + (offset - ffc), then dead-pixel filter
        out = np.clip(
            raw_roi.astype(np.int32) + THERMAL_OFFSET - cal.astype(np.int32),
            0,
            65535,
        ).astype(np.uint16)
        self._apply_dead_pixel_filter(out, dst)
        if self._additional_ffc is not None:
            dst[:] = np.clip(
                dst.astype(np.int32) + THERMAL_OFFSET - self._additional_ffc.astype(np.int32),
                0,
                65535,
            ).astype(np.uint16)

    def read(self, dst: np.ndarray) -> bool:
        """Grab one frame and retrieve into dst (207x154 uint16). Returns True on success."""
        if not self.grab():
            return False
        self.retrieve(dst)
        return True

    def _create_dead_pixel_list(self, frame: np.ndarray) -> None:
        """Build dead pixel mask and ordered list from drift-cal frame (207x154)."""
        frame_float = frame.astype(np.float32)
        max_val = float(np.max(frame_float))
        hist, _ = np.histogram(
            frame_float.ravel(),
            bins=HIST_SIZE,
            range=(0, float(HIST_SIZE)),
        )
        hist[0] = 0
        peak_bin = int(np.argmax(hist))
        threshold_val = 2 * peak_bin - max_val
        dead_mask = np.where(frame_float > threshold_val, 255, 0).astype(np.uint8)
        dead_count = int(np.sum(dead_mask == 255))
        dead_frac = dead_count / dead_mask.size
        if dead_frac > 0.5:
            self._dead_pixel_mask = np.zeros_like(dead_mask, dtype=np.uint8)
            self._dead_pixel_list = []
        else:
            self._dead_pixel_mask = dead_mask
            tmp = dead_mask.astype(np.uint16)
            self._dead_pixel_list = []
            rows, cols = frame.shape
            has_unlisted = True
            while has_unlisted:
                has_unlisted = False
                for y in range(rows):
                    for x in range(cols):
                        if tmp[y, x] != 0:
                            continue
                        if dead_mask[y, x] == 0:
                            continue
                        mean_val = self._calc_mean_value(tmp, x, y, 0)
                        if mean_val != 0:
                            self._dead_pixel_list.append((x, y))
                            tmp[y, x] = 255
                        else:
                            has_unlisted = True

    def _calc_mean_value(
        self,
        img: np.ndarray,
        px: int,
        py: int,
        dead_marker: int,
    ) -> int:
        """Mean of non-dead neighbors (4-connected)."""
        rows, cols = img.shape
        total = 0
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                v = int(img[ny, nx])
                if v != dead_marker:
                    total += v
                    count += 1
        return total // count if count else 0

    def _apply_dead_pixel_filter(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Copy src to dst, mark dead pixels with DEAD_PIXEL_MARKER, then fill from neighbors."""
        dst.fill(DEAD_PIXEL_MARKER)
        good = self._dead_pixel_mask == 0
        np.copyto(dst, src, where=good)
        for (x, y) in self._dead_pixel_list:
            dst[y, x] = self._calc_mean_value(dst, x, y, DEAD_PIXEL_MARKER)
