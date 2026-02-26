"""
Seek USB device interface using PyUSB (libusb-1.0).
Provides control transfers and bulk frame read for Seek Compact/CompactXR.
"""

import struct
from typing import List, Optional

import usb.core
import usb.util
import numpy as np

from .constants import (
    DeviceCommand,
    BULK_IN_ENDPOINT,
    USB_TIMEOUT_MS,
    THERMAL_RAW_SIZE,
    THERMAL_REQUEST_SIZE,
)

# bmRequestType: vendor request to interface
# OUT = 0x00 | 0x40 | 0x01, IN = 0x80 | 0x40 | 0x01
LIBUSB_RECIPIENT_INTERFACE = 0x01
LIBUSB_REQUEST_TYPE_VENDOR = 0x40
LIBUSB_ENDPOINT_OUT = 0x00
LIBUSB_ENDPOINT_IN = 0x80
BM_REQUEST_OUT = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_INTERFACE
BM_REQUEST_IN = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_INTERFACE


class SeekDeviceError(Exception):
    """Raised when device operations fail."""

    pass


class SeekDevice:
    """
    Low-level USB device for Seek thermal cameras.
    Open/close, control transfers, and bulk frame read.
    """

    def __init__(
        self,
        vendor_id: int,
        product_id: int,
        timeout: int = USB_TIMEOUT_MS,
    ) -> None:
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._timeout = timeout
        self._device: Optional[usb.core.Device] = None

    def open(self) -> bool:
        """Open USB device and claim interface 0. Set configuration 1 if needed."""
        if self._device is not None:
            raise SeekDeviceError("Device already opened")

        self._device = usb.core.find(idVendor=self._vendor_id, idProduct=self._product_id)
        if self._device is None:
            raise SeekDeviceError(
                f"Device not found {self._vendor_id:04x}:{self._product_id:04x}"
            )

        try:
            if self._device.is_kernel_driver_active(0):
                self._device.detach_kernel_driver(0)
        except usb.core.USBError:
            pass

        try:
            cfg = self._device.get_active_configuration()
            if cfg.bConfigurationValue != 1:
                self._device.set_configuration(1)
        except usb.core.USBError as e:
            self._device = None
            raise SeekDeviceError(f"Set configuration failed: {e}") from e

        try:
            usb.util.claim_interface(self._device, 0)
        except usb.core.USBError as e:
            self._device = None
            raise SeekDeviceError(f"Claim interface failed: {e}") from e

        return True

    def close(self) -> None:
        """Release interface and close device."""
        if self._device is None:
            return
        try:
            usb.util.release_interface(self._device, 0)
        except usb.core.USBError:
            pass
        try:
            usb.util.dispose_resources(self._device)
        except usb.core.USBError:
            pass
        self._device = None

    def is_opened(self) -> bool:
        """Return True if device is open."""
        return self._device is not None

    def request_set(self, command: int, data: List[int]) -> bool:
        """Vendor control transfer to device (OUT)."""
        if self._device is None:
            raise SeekDeviceError("Device not opened")
        if not data:
            data = [0] * 16
        buf = bytes(data)
        try:
            n = self._device.ctrl_transfer(
                BM_REQUEST_OUT,
                command,
                0,
                0,
                buf,
                timeout=self._timeout,
            )
            return n == len(buf)
        except usb.core.USBError as e:
            raise SeekDeviceError(f"Control transfer (set) failed: {e}") from e

    def request_get(self, command: int, length: int) -> bytes:
        """Vendor control transfer from device (IN). Returns received bytes."""
        if self._device is None:
            raise SeekDeviceError("Device not opened")
        if length <= 0:
            length = 16
        try:
            buf = self._device.ctrl_transfer(
                BM_REQUEST_IN,
                command,
                0,
                0,
                length,
                timeout=self._timeout,
            )
            return bytes(buf)
        except usb.core.USBError as e:
            raise SeekDeviceError(f"Control transfer (get) failed: {e}") from e

    def fetch_frame(
        self,
        buffer: np.ndarray,
        size: int,
        request_size: int = THERMAL_REQUEST_SIZE,
    ) -> bool:
        """
        Request and read one raw frame via START_GET_IMAGE_TRANSFER then bulk read.
        buffer: numpy array of dtype uint16, size elements (flat).
        size: number of uint16 words to read (e.g. THERMAL_RAW_SIZE).
        request_size: bytes per bulk read chunk.
        Corrects endianness in place (swap bytes per uint16).
        """
        if self._device is None:
            raise SeekDeviceError("Device not opened")
        payload = struct.pack("<I", size)
        self.request_set(DeviceCommand.START_GET_IMAGE_TRANSFER, list(payload))

        total_bytes = size * 2
        done = 0
        raw_list: List[bytes] = []
        while done < total_bytes:
            chunk_size = min(request_size, total_bytes - done)
            try:
                data = self._device.read(
                    BULK_IN_ENDPOINT,
                    chunk_size,
                    timeout=self._timeout,
                )
            except usb.core.USBError as e:
                raise SeekDeviceError(f"Bulk read failed: {e}") from e
            raw_list.append(bytes(data))
            done += len(data)

        raw = b"".join(raw_list)
        if len(raw) != total_bytes:
            raise SeekDeviceError(
                f"Frame size mismatch: got {len(raw)}, expected {total_bytes}"
            )
        # Device sends little-endian uint16; interpret and copy into buffer
        arr = np.frombuffer(raw, dtype=np.dtype(np.uint16).newbyteorder("<"))
        np.copyto(buffer.ravel()[:size], arr[:size])
        return True
